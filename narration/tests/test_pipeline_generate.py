from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from narration.breaker import PipelineBreaker
from narration.pipeline import Pipeline, drop_article_audio, mark_listened
from narration.store import STATUS_DELETED, STATUS_FALLBACK, STATUS_READY, Store
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
    async def fake_synth(self, cache, script, speed, *, hd, **_kwargs):
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
    script = store.script_path(out["cache_key"]).read_text(encoding="utf-8")
    from narration.spoken_host import script_has_closer

    assert script_has_closer(script)
    assert art["host_touch"] == "v1"


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


async def test_drop_article_audio_removes_opus_script_and_cache(
    store, redis, ram_ok, monkeypatch
):
    pipe = _pipe(store, redis)
    _stub_synth(monkeypatch, store)
    text = "Drop this spoken explainer after listen. " * 30
    out = await pipe.generate({"article_id": "news-drop-1", "text": text})
    cache = out["cache_key"]
    opus = store.opus_path(cache)
    script = store.script_path(cache)
    assert opus.exists()
    assert script.exists()
    assert await store.get_cache(cache)

    dropped = await drop_article_audio(store, "news-drop-1")
    assert dropped["deleted"] is True
    assert dropped["cache_key"] == cache
    assert not opus.exists()
    assert not script.exists()
    assert await store.get_cache(cache) is None
    art = await store.get_article("news-drop-1")
    assert art["status"] == STATUS_DELETED


async def test_generate_noops_when_article_already_deleted(store, redis, ram_ok, monkeypatch):
    llm = FakeLlm()
    pipe = _pipe(store, redis, llm)
    _stub_synth(monkeypatch, store)
    await drop_article_audio(store, "news-gone")
    out = await pipe.generate({"article_id": "news-gone", "text": "hello world " * 20})
    assert out["status"] == STATUS_DELETED
    assert out["reason"] == "article_dropped"
    assert llm.calls == 0
    art = await store.get_article("news-gone")
    assert art["status"] == STATUS_DELETED


async def test_cache_hit_does_not_resurrect_deleted_article(store, redis, ram_ok, monkeypatch):
    llm = FakeLlm()
    pipe = _pipe(store, redis, llm)
    text = "Shared cache body used by a dropped article. " * 30
    cache = pipe.cache_for(text)
    opus = store.opus_path(cache)
    opus.parent.mkdir(parents=True, exist_ok=True)
    opus.write_bytes(b"OggS" + b"\x00" * 80)
    await store.put_cache(
        cache,
        {
            "article_id": "news-shared",
            "cache_key": cache,
            "status": STATUS_READY,
            "file_path": str(opus),
            "duration_s": 12.0,
        },
    )
    await store.bind_article(
        "news-dropped",
        {"article_id": "news-dropped", "cache_key": cache, "status": STATUS_DELETED},
    )
    out = await pipe.generate({"article_id": "news-dropped", "text": text})
    assert out["status"] == STATUS_DELETED
    assert llm.calls == 0
    art = await store.get_article("news-dropped")
    assert art["status"] == STATUS_DELETED


async def test_set_article_does_not_overwrite_deleted(store, redis, ram_ok):
    pipe = _pipe(store, redis)
    await drop_article_audio(store, "news-lock-del")
    await pipe._set_article("news-lock-del", status=STATUS_READY, cache_key="abc")
    art = await store.get_article("news-lock-del")
    assert art["status"] == STATUS_DELETED


async def test_generate_deletes_files_if_dropped_during_synth(
    store, redis, ram_ok, monkeypatch
):
    pipe = _pipe(store, redis)
    text = "In flight drop should not leave opus behind. " * 30

    async def fake_synth(self, cache, script, speed, *, hd, **_kwargs):
        opus = store.opus_path(cache)
        opus.parent.mkdir(parents=True, exist_ok=True)
        opus.write_bytes(b"OggS" + b"\x00" * 80)
        await drop_article_audio(store, "news-mid-drop")
        return (
            opus,
            12.5,
            {"clipped": False, "silence_s": 0.2, "rms_db": -16.0, "rms_parsed": True},
        )

    monkeypatch.setattr(Pipeline, "_synth_and_encode", fake_synth)
    out = await pipe.generate({"article_id": "news-mid-drop", "text": text})
    assert out["status"] == STATUS_DELETED
    cache = pipe.cache_for(text)
    assert not store.opus_path(cache).exists()
    assert await store.get_cache(cache) is None
    art = await store.get_article("news-mid-drop")
    assert art["status"] == STATUS_DELETED


async def test_mark_listened_keeps_opus_and_ready_status(
    store, redis, ram_ok, monkeypatch
):
    pipe = _pipe(store, redis)
    _stub_synth(monkeypatch, store)
    text = "Replay this explainer after the first listen. " * 30
    out = await pipe.generate({"article_id": "news-replay-1", "text": text})
    cache = out["cache_key"]
    opus = store.opus_path(cache)
    script = store.script_path(cache)
    assert opus.exists()
    marked = await mark_listened(store, article_id="news-replay-1", cache=cache)
    assert marked["deleted"] is False
    assert marked["ok"] is True
    assert opus.exists()
    assert script.exists()
    assert await store.get_cache(cache) is not None
    art = await store.get_article("news-replay-1")
    assert art["status"] == STATUS_READY
    assert art["listened"] is True
    again = await pipe.generate({"article_id": "news-replay-1", "text": text})
    assert again["status"] == STATUS_READY
    assert again.get("cache_hit") is True
    dropped = await drop_article_audio(store, "news-replay-1")
    assert dropped["deleted"] is True
    assert not opus.exists()
    assert not script.exists()
    art = await store.get_article("news-replay-1")
    assert art["status"] == STATUS_DELETED


async def test_mark_listened_does_not_resurrect_dropped_article(store):
    await drop_article_audio(store, "news-replay-gone")
    out = await mark_listened(store, article_id="news-replay-gone", cache="abc")
    assert out["ok"] is False
    art = await store.get_article("news-replay-gone")
    assert art["status"] == STATUS_DELETED


def test_jobs_post_skips_deleted_before_cache_hit_and_exposes_drop():
    src = (Path(__file__).resolve().parents[1] / "src" / "narration" / "api" / "main.py").read_text(
        encoding="utf-8"
    )
    deleted_at = src.find('if existing_art and existing_art.get("status") == STATUS_DELETED')
    hit_at = src.find("hit = await store.get_cache(cache)")
    drop_at = src.find('@app.post("/v1/drop")')
    complete_at = src.find('@app.post("/v1/complete")')
    assert 0 < deleted_at < hit_at
    assert drop_at > 0
    assert "drop_article_audio" in src
    assert "mark_listened" in src
    assert "complete_listen" not in src[complete_at:]
    assert "force" in src
    assert "article_removed" in src
    assert "ensure_host_touch" in src


async def test_generate_crash_after_drop_does_not_leave_tmp(
    store, redis, ram_ok, monkeypatch
):
    from narration.tts_client import TtsError

    pipe = _pipe(store, redis)
    text = "Crash after clear-all must not keep wavs. " * 30
    cache = pipe.cache_for(text)

    async def fake_synth(self, c, script, speed, *, hd, **_kwargs):
        wav = store.tmp_wav(c, 0)
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        opus = store.opus_path(c)
        opus.parent.mkdir(parents=True, exist_ok=True)
        opus.write_bytes(b"OggS" + b"\x00" * 40)
        await drop_article_audio(store, "news-crash-drop")
        raise TtsError("boom")

    monkeypatch.setattr(Pipeline, "_synth_and_encode", fake_synth)
    out = await pipe.generate({"article_id": "news-crash-drop", "text": text})
    assert out["status"] == STATUS_DELETED
    assert not store.opus_path(cache).exists()
    assert not store.tmp_wav(cache, 0).exists()
    assert await store.get_cache(cache) is None


def test_worker_skips_dropped_before_generate_and_sweeps_tmp():
    src = (Path(__file__).resolve().parents[1] / "src" / "narration" / "worker" / "main.py").read_text(
        encoding="utf-8"
    )
    dropped_at = src.find('rec.get("status") == STATUS_DELETED')
    generate_at = src.find("return await pipe.generate(payload)")
    assert 0 < dropped_at < generate_at
    assert "iter_stale_tmp" in src
    assert "article_dropped" in src
    assert "max_jobs = 1" in src
    assert "REAPER_AGE_S = 168 * 3600" in src
    assert "48 * 3600" not in src


def test_iter_expired_meta_honours_168h_cap(store):
    cache = "age168h01"
    meta = store.meta_path(cache)
    meta.parent.mkdir(parents=True, exist_ok=True)
    now = 1_800_000_000.0
    meta.write_text(
        '{"cache_key":"%s","created_at":%s}' % (cache, now - 167 * 3600),
        encoding="utf-8",
    )
    keep = store.iter_expired_meta(max_age_s=168 * 3600, now=now)
    assert cache not in keep
    meta.write_text(
        '{"cache_key":"%s","created_at":%s}' % (cache, now - 168 * 3600 - 1),
        encoding="utf-8",
    )
    gone = store.iter_expired_meta(max_age_s=168 * 3600, now=now)
    assert cache in gone


def test_stale_tmp_is_visible_to_reaper(store):
    p = store.tmp_wav("deadtmp", 0)
    p.write_bytes(b"x")
    now = p.stat().st_mtime + 4 * 3600
    stale = store.iter_stale_tmp(max_age_s=3 * 3600, now=now)
    assert p in stale
    fresh = store.iter_stale_tmp(max_age_s=3 * 3600, now=p.stat().st_mtime + 10)
    assert p not in fresh


class _FakeTts:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def synthesize(self, text, *, voice, speed):
        self.calls += 1
        if self.fail:
            raise RuntimeError("tts down")
        return b"RIFF" + b"\x00" * 40


async def test_ensure_host_touch_stamps_when_script_already_has_closer(store):
    from narration.spoken_host import HOST_TOUCH_VERSION, ensure_host_touch

    cache = "touchready01"
    opus = store.opus_path(cache)
    opus.parent.mkdir(parents=True, exist_ok=True)
    opus.write_bytes(b"OggS" + b"\x00" * 80)
    store.write_script(cache, "Body text. That's it from this article. I'll leave it there.")
    rec = {
        "cache_key": cache,
        "article_id": "news-touch-1",
        "status": STATUS_READY,
        "file_path": str(opus),
    }
    await store.put_cache(cache, rec)
    tts = _FakeTts()
    out = await ensure_host_touch(
        store, tts, rec, voice="am_onyx", speed=0.9, bitrate=32000
    )
    assert out["host_touch"] == HOST_TOUCH_VERSION
    assert tts.calls == 0


async def test_ensure_host_touch_tts_failure_leaves_opus(store):
    from narration.spoken_host import ensure_host_touch

    cache = "touchfail01"
    opus = store.opus_path(cache)
    opus.parent.mkdir(parents=True, exist_ok=True)
    original = b"OggS" + b"\x00" * 80
    opus.write_bytes(original)
    store.write_script(cache, "The bank cut rates this morning.")
    rec = {
        "cache_key": cache,
        "article_id": "news-touch-fail",
        "status": STATUS_READY,
        "file_path": str(opus),
    }
    await store.put_cache(cache, rec)
    out = await ensure_host_touch(
        store, _FakeTts(fail=True), rec, voice="am_onyx", speed=0.9, bitrate=32000
    )
    assert opus.read_bytes() == original
    assert out.get("host_touch") != "v1"


async def test_ensure_host_touch_appends_closer_when_missing(store, monkeypatch):
    from narration.spoken_host import HOST_TOUCH_VERSION, ensure_host_touch, script_has_closer

    async def fake_decode(src, dest):
        dest.write_bytes(b"RIFF" + b"\x00" * 24)

    async def fake_concat(paths, dest):
        dest.write_bytes(b"RIFF" + b"\x00" * 40)

    async def fake_encode(wav, dest, *, bitrate, sample_rate=24000):
        dest.write_bytes(b"OggS" + b"TOUCH")

    async def fake_dur(_path):
        return 12.0

    monkeypatch.setattr("narration.spoken_host.decode_to_wav", fake_decode)
    monkeypatch.setattr("narration.spoken_host.concat_wavs", fake_concat)
    monkeypatch.setattr("narration.spoken_host.encode_opus", fake_encode)
    monkeypatch.setattr("narration.spoken_host.ffprobe_duration_s", fake_dur)

    cache = "touchappend01"
    opus = store.opus_path(cache)
    opus.parent.mkdir(parents=True, exist_ok=True)
    opus.write_bytes(b"OggS" + b"\x00" * 80)
    store.write_script(cache, "The bank cut rates this morning.")
    rec = {
        "cache_key": cache,
        "article_id": "news-touch-append",
        "status": STATUS_READY,
        "file_path": str(opus),
    }
    await store.put_cache(cache, rec)
    out = await ensure_host_touch(
        store, _FakeTts(), rec, voice="am_onyx", speed=0.9, bitrate=32000
    )
    assert out["host_touch"] == HOST_TOUCH_VERSION
    assert opus.read_bytes().endswith(b"TOUCH")
    script = store.script_path(cache).read_text(encoding="utf-8")
    assert script_has_closer(script)
