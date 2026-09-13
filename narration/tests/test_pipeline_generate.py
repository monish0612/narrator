from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from narration.breaker import PipelineBreaker
from narration.pipeline import Pipeline
from narration.store import STATUS_FALLBACK, STATUS_READY, Store
from narration.telegram import Telegram


@pytest_asyncio.fixture
async def redis():
    r = FakeAsyncRedis()
    yield r
    await r.aclose()


@pytest.fixture
def store(redis, tmp_path: Path) -> Store:
    return Store(redis, prefix="narration:", data_dir=str(tmp_path), ttl_s=3600)


class FakeLlm:
    def __init__(self, script: str | None = None, error: Exception | None = None) -> None:
        self.calls = 0
        self.script = script or ("This is a spoken explainer sentence about the story. " * 40)
        self.error = error

    async def ensure_model(self) -> str:
        return "qwen3.5-4b"

    async def generate_script(self, **kwargs) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.script


def _pipe(store, redis, llm=None) -> Pipeline:
    return Pipeline(
        store,
        llm or FakeLlm(),
        tts=None,  # synth is stubbed
        breaker=PipelineBreaker(redis, "narration:", "pipeline", threshold=3, cooldown_s=900),
        telegram=Telegram("", ""),
        voice="am_onyx",
        speed=0.9,
        model_version="test-v1",
        audio_format="opus",
        bitrate=32000,
        hd_bitrate=48000,
        max_duration_s=600,
        word_min=40,
        word_max=1450,
        ram_floor=1,
        ram_alert_floor=1,
    )


@pytest.fixture
def ram_ok(monkeypatch):
    monkeypatch.setattr("narration.pipeline.should_defer", lambda *a, **k: False)


def _stub_synth(monkeypatch, store, *, metrics=None, duration=12.5):
    async def fake_synth(self, cache, script, speed, *, hd):
        opus = store.opus_path(cache)
        opus.parent.mkdir(parents=True, exist_ok=True)
        opus.write_bytes(b"OggS" + b"\x00" * 80)
        return (
            opus,
            duration,
            metrics
            or {
                "clipped": False,
                "silence_s": 0.2,
                "rms_db": -16.0,
                "rms_parsed": True,
            },
        )

    monkeypatch.setattr(Pipeline, "_synth_and_encode", fake_synth)


async def test_generate_ready_when_record_includes_article_id(store, redis, ram_ok, monkeypatch):
    llm = FakeLlm()
    pipe = _pipe(store, redis, llm)
    _stub_synth(monkeypatch, store)
    text = "A bank support inbox story. " * 40
    out = await pipe.generate(
        {
            "article_id": "news-bind-1",
            "title": "One capital letter",
            "source": "TDS",
            "category": "AI News",
            "text": text,
        }
    )
    assert out["status"] == STATUS_READY
    assert out.get("cache_hit") is not True
    art = await store.get_article("news-bind-1")
    assert art["status"] == STATUS_READY
    assert art["article_id"] == "news-bind-1"
    assert art["cache_key"] == out["cache_key"]
    assert llm.calls == 1


async def test_cache_hit_survives_status_and_article_id_in_existing(
    store, redis, ram_ok, monkeypatch
):
    llm = FakeLlm()
    pipe = _pipe(store, redis, llm)
    text = "Cached article body that already has audio. " * 30
    cache = pipe.cache_for(text)
    opus = store.opus_path(cache)
    opus.parent.mkdir(parents=True, exist_ok=True)
    opus.write_bytes(b"OggS" + b"\x00" * 80)
    existing = {
        "article_id": "news-old",
        "cache_key": cache,
        "status": STATUS_READY,
        "file_path": str(opus),
        "duration_s": 99.0,
    }
    await store.put_cache(cache, existing)

    out = await pipe.generate({"article_id": "news-new", "text": text})
    assert out["status"] == STATUS_READY
    assert out["cache_hit"] is True
    assert llm.calls == 0
    art = await store.get_article("news-new")
    assert art["status"] == STATUS_READY
    assert art["article_id"] == "news-new"
    assert art["cache_key"] == cache


async def test_breaker_open_skips_llm(store, redis, ram_ok):
    llm = FakeLlm()
    pipe = _pipe(store, redis, llm)
    await pipe.breaker.record_failure("a")
    await pipe.breaker.record_failure("b")
    await pipe.breaker.record_failure("c")
    assert await pipe.breaker.is_open() is True
    out = await pipe.generate({"article_id": "news-br", "text": "hello world " * 20})
    assert out["status"] == STATUS_FALLBACK
    assert out["reason"] == "breaker_open"
    assert llm.calls == 0


async def test_llm_failure_marks_fallback_not_crash(store, redis, ram_ok, monkeypatch):
    from narration.llm import LlmError

    llm = FakeLlm(error=LlmError("empty llm content"))
    pipe = _pipe(store, redis, llm)
    _stub_synth(monkeypatch, store)
    out = await pipe.generate({"article_id": "news-llm-fail", "text": "hello world " * 20})
    assert out["status"] == STATUS_FALLBACK
    assert "LlmError" in out["reason"]
    art = await store.get_article("news-llm-fail")
    assert art["status"] == STATUS_FALLBACK


async def test_qa_clipped_marks_fallback(store, redis, ram_ok, monkeypatch):
    pipe = _pipe(store, redis)
    _stub_synth(
        monkeypatch,
        store,
        metrics={"clipped": True, "silence_s": 0.1, "rms_db": -8.0, "rms_parsed": True},
    )
    out = await pipe.generate({"article_id": "news-clip", "text": "hello world " * 20})
    assert out["status"] == STATUS_FALLBACK
    assert "qa_fail:clipped" in out["reason"]


async def test_lock_held_returns_queued(store, redis, ram_ok):
    pipe = _pipe(store, redis)
    text = "locked body " * 20
    cache = pipe.cache_for(text)
    from narration.keys import lock_key

    await store._r.set(lock_key(store.prefix, cache), "other", nx=True, ex=60)
    out = await pipe.generate({"article_id": "news-lock", "text": text})
    assert out["status"] == "queued"
    assert out["reason"] == "lock_held"


async def test_peak_near_zero_metrics_do_not_fail_qa(store, redis, ram_ok, monkeypatch):
    pipe = _pipe(store, redis)
    _stub_synth(
        monkeypatch,
        store,
        metrics={
            "clipped": False,
            "silence_s": 1.0,
            "rms_db": -12.0,
            "rms_parsed": True,
            "peak_db": 0.0,
        },
    )
    out = await pipe.generate({"article_id": "news-loud", "text": "hello world " * 20})
    assert out["status"] == STATUS_READY
    assert Path(store.opus_path(out["cache_key"])).exists()
