from __future__ import annotations

import time
from pathlib import Path

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from narration.breaker import PipelineBreaker
from narration.delete import delete_artifacts
from narration.encode import qa_should_fail
from narration.http_range import _RANGE
from narration.llm import _parse_relevance, _strip_think
from narration.normalize import build_cache_key, normalize_article_text
from narration.prompts import is_ai_news, word_count
from narration.ram_gate import mem_available_bytes, next_backoff_s, should_defer
from narration.store import Store


@pytest_asyncio.fixture
async def redis():
    r = FakeAsyncRedis()
    yield r
    await r.aclose()


@pytest.fixture
def store(redis, tmp_path: Path) -> Store:
    return Store(redis, prefix="narration:", data_dir=str(tmp_path), ttl_s=3600)


def test_normalize_collapses_whitespace():
    assert normalize_article_text("  Hello\n\n  world  ") == "Hello world"


def test_cache_key_stable_and_sensitive():
    a = build_cache_key(
        article_text="Hello world",
        voice="am_onyx",
        speed=0.9,
        model_version="v1",
        audio_format="opus",
        bitrate=32000,
    )
    b = build_cache_key(
        article_text="Hello   world",
        voice="am_onyx",
        speed=0.9,
        model_version="v1",
        audio_format="opus",
        bitrate=32000,
    )
    assert a == b
    c = build_cache_key(
        article_text="Hello world",
        voice="am_onyx",
        speed=1.0,
        model_version="v1",
        audio_format="opus",
        bitrate=32000,
    )
    assert a != c
    assert len(a) == 64


def test_ai_news_category_match():
    assert is_ai_news("AI News")
    assert is_ai_news("ai news")
    assert not is_ai_news("Finance")


def test_word_count():
    assert word_count("one two three") == 3


def test_relevance_parser():
    raw = '```json\n{"relevant": true, "reason": "x", "parallel": "bots"}\n```'
    relevant, parallel = _parse_relevance(raw)
    assert relevant is True
    assert parallel == "bots"
    assert _parse_relevance("not json") == (False, "")
    think = '<think>nope</think>{"relevant": false, "reason": "x", "parallel": ""}'
    assert _parse_relevance(think) == (False, "")
    assert _strip_think("hello <think>secret</think> world") == "hello  world"


def test_ram_gate_from_meminfo():
    blob = "MemTotal:        8000000 kB\nMemAvailable:      500000 kB\n"
    avail = mem_available_bytes(blob)
    assert avail == 500000 * 1024
    assert should_defer(1_073_741_824, available=avail) is True
    assert should_defer(1_073_741_824, available=2_147_483_648) is False
    assert next_backoff_s(0) == 30
    assert next_backoff_s(1) == 60
    assert next_backoff_s(2) == 120


def test_qa_fail_rules():
    assert qa_should_fail({"clipped": True, "silence_s": 0, "rms_db": -16}, 100) == "clipped"
    assert qa_should_fail({"clipped": False, "silence_s": 40, "rms_db": -16}, 100) == "too_much_silence"
    assert qa_should_fail({"clipped": False, "silence_s": 1, "rms_db": -50}, 100) == "inaudible"
    assert qa_should_fail({"clipped": False, "silence_s": 1, "rms_db": -16}, 100) is None


def test_range_header_parse():
    m = _RANGE.match("bytes=0-1023")
    assert m is not None
    assert m.group(1) == "0"
    assert m.group(2) == "1023"


async def test_breaker_opens_on_three_different_articles(redis):
    br = PipelineBreaker(redis, "narration:", "pipeline", threshold=3, cooldown_s=900)
    assert await br.is_open() is False
    assert await br.record_failure("a1") is False
    assert await br.record_failure("a1") is False  # same article does not extra-count via set
    assert await br.record_failure("a2") is False
    opened = await br.record_failure("a3")
    assert opened is True
    assert await br.is_open() is True
    reset = await br.record_success()
    assert reset is True
    assert await br.is_open() is False


async def test_delete_idempotent_missing_files(store, redis):
    sleeps: list[float] = []

    async def no_sleep(d: float) -> None:
        sleeps.append(d)

    ok = await delete_artifacts(store, "abc123", attempts=3, sleep=no_sleep)
    assert ok is True
    # already gone is success; no exhausted path
    assert not sleeps


async def test_delete_retries_then_succeeds(store, monkeypatch):
    from narration import delete as delete_mod

    cache = "deadbeef"
    path = store.opus_path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    calls = {"n": 0}
    real = delete_mod._unlink

    async def flaky(p: Path) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("busy")
        await real(p)

    monkeypatch.setattr(delete_mod, "_unlink", flaky)

    sleeps: list[float] = []

    async def no_sleep(d: float) -> None:
        sleeps.append(d)

    ok = await delete_artifacts(store, cache, attempts=5, sleep=no_sleep)
    assert ok is True
    assert not path.exists()
    assert calls["n"] >= 3
    assert sleeps


async def test_store_roundtrip(store):
    rec = {"cache_key": "abc", "status": "ready", "file_path": "", "created_at": time.time()}
    opus = store.opus_path("abc")
    opus.parent.mkdir(parents=True, exist_ok=True)
    opus.write_bytes(b"OpusFake")
    rec["file_path"] = str(opus)
    await store.put_cache("abc", rec)
    got = await store.get_cache("abc")
    assert got is not None
    assert got["status"] == "ready"
    await store.bind_article("news-1", {"article_id": "news-1", "cache_key": "abc", "status": "ready"})
    art = await store.get_article("news-1")
    assert art["cache_key"] == "abc"
