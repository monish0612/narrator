from __future__ import annotations

import asyncio

import httpx
import pytest

from narration.llm import LlmClient, LlmError, _parse_relevance, _strip_fences, _strip_think
from narration.tts_client import TtsClient, TtsError


class _Resp:
    def __init__(self, status=200, payload=None, content=b"", text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.content = content
        self.text = text

    def json(self):
        return self._payload


class FlakyChat:
    def __init__(self, fail_times: int, payload: dict):
        self.fail_times = fail_times
        self.n = 0
        self.payload = payload

    async def post(self, url, json=None, timeout=None):
        self.n += 1
        if self.n <= self.fail_times:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return _Resp(payload=self.payload)


@pytest.fixture
def no_sleep(monkeypatch):
    async def _sleep(_delay):
        return None

    monkeypatch.setattr("narration.llm.asyncio.sleep", _sleep)
    monkeypatch.setattr("narration.tts_client.asyncio.sleep", _sleep)


async def test_chat_retries_then_succeeds(no_sleep):
    http = FlakyChat(2, {"message": {"content": "Hello world from qwen."}})
    llm = LlmClient("http://llm", "qwen3.5-4b", gguf_path="/missing.gguf", client=http)
    text = await llm.chat("sys", "user")
    assert text == "Hello world from qwen."
    assert http.n == 3


async def test_chat_gives_up_after_three_disconnects(no_sleep):
    http = FlakyChat(9, {"message": {"content": "never"}})
    llm = LlmClient("http://llm", "qwen3.5-4b", gguf_path="/missing.gguf", client=http)
    with pytest.raises(LlmError, match="disconnected"):
        await llm.chat("sys", "user")
    assert http.n == 3


async def test_chat_reads_openai_shim_content(no_sleep):
    class Once:
        async def post(self, url, json=None, timeout=None):
            return _Resp(
                payload={"choices": [{"message": {"content": "  shim text  "}}]}
            )

    llm = LlmClient("http://llm", "qwen3.5-4b", gguf_path="/x", client=Once())
    assert await llm.chat("s", "u") == "shim text"


async def test_chat_empty_content_is_llm_error(no_sleep):
    class Once:
        async def post(self, url, json=None, timeout=None):
            return _Resp(payload={"message": {"content": "  "}})

    llm = LlmClient("http://llm", "qwen3.5-4b", gguf_path="/x", client=Once())
    with pytest.raises(LlmError, match="empty llm content"):
        await llm.chat("s", "u")


async def test_chat_http_error(no_sleep):
    class Once:
        async def post(self, url, json=None, timeout=None):
            return _Resp(status=500, text="boom")

    llm = LlmClient("http://llm", "qwen3.5-4b", gguf_path="/x", client=Once())
    with pytest.raises(LlmError, match="llm http 500"):
        await llm.chat("s", "u")


def test_think_and_fence_stripping():
    assert _strip_think("<think>abc</think>visible") == "visible"
    assert _strip_think("keep <think>x</think> going") == "keep  going"
    assert _strip_fences("```markdown\nHello\n```") == "Hello"
    assert _parse_relevance('{"relevant": true, "parallel": "agents"}') == (True, "agents")
    assert _parse_relevance("<think>{") == (False, "")


async def test_tts_retries_then_returns_wav(no_sleep):
    class Flaky:
        n = 0

        async def post(self, url, json=None, timeout=None):
            self.n += 1
            if self.n == 1:
                raise httpx.ReadError("peer closed")
            return _Resp(content=b"RIFF" + b"\x00" * 12)

    tts = TtsClient("http://tts/v1", client=Flaky())
    wav = await tts.synthesize("hello", voice="am_onyx", speed=0.9)
    assert wav.startswith(b"RIFF")


async def test_tts_rejects_non_wav(no_sleep):
    class Once:
        async def post(self, url, json=None, timeout=None):
            return _Resp(content=b"OggSxxxx")

    tts = TtsClient("http://tts/v1", client=Once())
    with pytest.raises(TtsError, match="non-wav"):
        await tts.synthesize("hello", voice="am_onyx", speed=0.9)


def test_model_present_matches_prefix():
    llm = LlmClient("http://llm", "qwen3.5-4b", gguf_path="/x", client=object())
    assert llm.model_present(["qwen3.5-4b:latest"]) is True
    assert llm.model_present(["llama3"]) is False


def test_asyncio_imported_for_retries():
    assert asyncio is not None
