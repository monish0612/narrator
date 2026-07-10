from __future__ import annotations

import fakeredis
import httpx
import pytest
import pytest_asyncio
import respx

from narrator.core import retry as retry_mod
from narrator.core.breaker import OPEN, CircuitBreaker
from narrator.core.errors import BreakerOpen, UnspeakableError
from narrator.engines.openai_compat import OpenAICompatEngine

BASE = "http://tts:8880/v1"
SPEECH = f"{BASE}/audio/speech"
VOICES = f"{BASE}/audio/voices"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    # Make tenacity waits instant so retry tests are fast.
    monkeypatch.setattr(retry_mod._WaitRespectRetryAfter, "__call__", lambda self, rs: 0.0)


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient() as c:
        yield c


def _breaker(threshold: int = 5) -> CircuitBreaker:
    return CircuitBreaker(fakeredis.FakeAsyncRedis(), "tts", threshold=threshold)


def _wav() -> bytes:
    return b"RIFF" + b"\x00" * 40


@respx.mock
async def test_happy_path(client):
    route = respx.post(SPEECH).mock(
        return_value=httpx.Response(
            200,
            content=_wav(),
            headers={"X-Audio-Duration-Ms": "1234", "X-Sample-Rate": "24000"},
        )
    )
    eng = OpenAICompatEngine(client, base_url=BASE, breaker=_breaker())
    result = await eng.synthesize("hello", voice="af_heart", speed=1.0)
    assert route.called
    assert result.audio == _wav()
    assert result.duration_ms == 1234
    assert result.sample_rate == 24000


@respx.mock
async def test_500_then_success(client):
    route = respx.post(SPEECH).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, content=_wav(), headers={"X-Audio-Duration-Ms": "10"}),
        ]
    )
    eng = OpenAICompatEngine(client, base_url=BASE, breaker=_breaker())
    result = await eng.synthesize("hi", voice="af_heart", speed=1.0)
    assert route.call_count == 2
    assert result.duration_ms == 10


@respx.mock
async def test_400_raises_unspeakable_no_retry(client):
    route = respx.post(SPEECH).mock(
        return_value=httpx.Response(400, json={"reason": "symbols_only"})
    )
    eng = OpenAICompatEngine(client, base_url=BASE, breaker=_breaker())
    with pytest.raises(UnspeakableError) as ei:
        await eng.synthesize("###", voice="af_heart", speed=1.0)
    assert ei.value.reason == "symbols_only"
    assert route.call_count == 1  # no retry on terminal error


@respx.mock
async def test_breaker_opens_after_exhaustion(client):
    route = respx.post(SPEECH).mock(return_value=httpx.Response(503))
    br = _breaker(threshold=1)
    eng = OpenAICompatEngine(client, base_url=BASE, breaker=br)

    # First call exhausts 6 retry attempts -> one breaker failure -> opens.
    from narrator.core.errors import TTSError

    with pytest.raises(TTSError):
        await eng.synthesize("hi", voice="af_heart", speed=1.0)
    assert route.call_count == 6
    assert await br.get_state() == OPEN

    # Second call is short-circuited by the open breaker (no HTTP).
    with pytest.raises(BreakerOpen):
        await eng.synthesize("hi", voice="af_heart", speed=1.0)
    assert route.call_count == 6


@respx.mock
async def test_list_voices(client):
    respx.get(VOICES).mock(return_value=httpx.Response(200, json={"voices": ["af_heart", "af_bella"]}))
    eng = OpenAICompatEngine(client, base_url=BASE, breaker=_breaker())
    assert await eng.list_voices() == ["af_heart", "af_bella"]


@respx.mock
async def test_health(client):
    respx.get("http://tts:8880/health").mock(return_value=httpx.Response(200, json={"ready": True}))
    eng = OpenAICompatEngine(client, base_url=BASE, breaker=_breaker())
    assert await eng.health() is True
