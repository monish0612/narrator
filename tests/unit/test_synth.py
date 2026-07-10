from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import fakeredis
import pytest
import pytest_asyncio

from narrator.core.cache import TTSCache
from narrator.core.errors import JobCancelled, TTSError, UnspeakableError
from narrator.core.gate import SynthGate
from narrator.core.models import Chunk, Job, JobParams, Manifest, SynthesisResult
from narrator.core.state import StateStore
from narrator.pipeline.chunker import _sanitized_hash
from narrator.pipeline.synth import Synthesizer


def make_wav_bytes(ms: int = 200, rate: int = 24000) -> bytes:
    n = int(rate * ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


class FakeEngine:
    _base = "http://tts:8880/v1"

    def __init__(self, *, fail_times: int = 0, unspeakable_on: set[str] | None = None) -> None:
        self.calls = 0
        self._fail_times = fail_times
        self._unspeakable_on = unspeakable_on or set()
        self._wav = make_wav_bytes(200)

    async def synthesize(self, text, *, voice, speed, response_format="wav") -> SynthesisResult:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TTSError("flaky")
        if text in self._unspeakable_on:
            raise UnspeakableError("bad")
        return SynthesisResult(audio=self._wav, duration_ms=200, sample_rate=24000)


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(idx=idx, text=text, sha256=_sanitized_hash(text), pause_ms_after=0)


CHUNKS = [_chunk(1, "First chunk here."), _chunk(2, "Second chunk here."), _chunk(3, "Third one.")]


@pytest_asyncio.fixture
async def env(tmp_path: Path):
    redis = fakeredis.FakeAsyncRedis()
    store = StateStore(tmp_path / "ledger.db", redis)
    await store.connect()
    job = Job(id="jb_1", lane="fast", params=JobParams(mode="verbatim"))
    await store.create_job(job)
    cache = TTSCache(store, tmp_path / "cache", max_gb=5)
    gate = SynthGate(redis, poll_interval=0.01)
    yield {
        "store": store,
        "redis": redis,
        "job": job,
        "cache": cache,
        "gate": gate,
        "tmp": tmp_path,
    }
    await store.close()
    await redis.aclose()


def _synth(env, engine, *, lane="fast", concurrency=1, cache=None) -> Synthesizer:
    return Synthesizer(
        engine=engine,
        cache=cache or env["cache"],
        gate=env["gate"],
        state=env["store"],
        model="kokoro-int8",
        concurrency=concurrency,
        lane=lane,
    )


async def test_fresh_run_completes_manifest(env):
    engine = FakeEngine()
    job_dir = env["tmp"] / "jobs" / "jb_1"
    manifest = await _synth(env, engine).run(env["job"], CHUNKS, job_dir)
    assert engine.calls == 3
    assert manifest.done_count == 3
    for c in CHUNKS:
        assert (job_dir / "chunks" / f"{c.idx:06d}.wav").exists()


async def test_resume_only_remainder(env):
    # Distinct caches per run so re-production is attributable to the engine.
    cache_a = TTSCache(env["store"], env["tmp"] / "cacheA", max_gb=5)
    cache_b = TTSCache(env["store"], env["tmp"] / "cacheB", max_gb=5)
    job_dir = env["tmp"] / "jobs" / "jb_1"
    # Kill after chunk 1: seed the manifest with only chunk 1 done.
    engine_a = FakeEngine()
    await _synth(env, engine_a, cache=cache_a).run(env["job"], [CHUNKS[0]], job_dir)
    assert engine_a.calls == 1
    engine_b = FakeEngine()
    manifest = await _synth(env, engine_b, cache=cache_b).run(env["job"], CHUNKS, job_dir)
    assert engine_b.calls == 2  # only chunks 2 and 3
    assert manifest.done_count == 3


async def test_truncated_wav_resynthesized(env):
    cache_a = TTSCache(env["store"], env["tmp"] / "cacheA", max_gb=5)
    cache_b = TTSCache(env["store"], env["tmp"] / "cacheB", max_gb=5)
    job_dir = env["tmp"] / "jobs" / "jb_1"
    await _synth(env, FakeEngine(), cache=cache_a).run(env["job"], CHUNKS, job_dir)
    # Truncate chunk 2 (crash mid-write) -> invalid RIFF/size.
    (job_dir / "chunks" / "000002.wav").write_bytes(b"RIFF")
    engine_b = FakeEngine()
    await _synth(env, engine_b, cache=cache_b).run(env["job"], CHUNKS, job_dir)
    assert engine_b.calls == 1  # only the truncated chunk


async def test_flaky_engine_records_attempts(env):
    engine = FakeEngine(fail_times=2)
    job_dir = env["tmp"] / "jobs" / "jb_1"
    manifest = await _synth(env, engine).run(env["job"], [CHUNKS[0]], job_dir)
    assert engine.calls == 3  # 2 fails + 1 success
    assert manifest.chunks[0].attempts == 3
    assert manifest.chunks[0].status == "done"


async def test_cache_hit_skips_engine(env):
    engine_a = FakeEngine()
    await _synth(env, engine_a).run(env["job"], CHUNKS, env["tmp"] / "jobs" / "jb_1")
    assert engine_a.calls == 3
    engine_b = FakeEngine()
    manifest = await _synth(env, engine_b).run(env["job"], CHUNKS, env["tmp"] / "jobs" / "jb_2")
    assert engine_b.calls == 0  # all served from cache
    assert all(c.cache_hit for c in manifest.chunks)


async def test_cancel_raises_and_writes_manifest(env):
    await env["store"].request_cancel("jb_1")
    engine = FakeEngine()
    job_dir = env["tmp"] / "jobs" / "jb_1"
    with pytest.raises(JobCancelled):
        await _synth(env, engine).run(env["job"], CHUNKS, job_dir)
    assert engine.calls == 0
    mpath = job_dir / "manifest.json"
    assert mpath.exists()
    Manifest.model_validate_json(mpath.read_text())  # consistent/parseable


async def test_bulk_parks_while_fast_pending(env):
    await env["gate"].fast_enter()
    engine = FakeEngine()
    job_dir = env["tmp"] / "jobs" / "jb_1"
    task = asyncio.create_task(_synth(env, engine, lane="bulk").run(env["job"], CHUNKS, job_dir))
    await asyncio.sleep(0.1)
    assert not task.done()  # parked, yielding to fast lane
    assert engine.calls == 0
    await env["gate"].fast_exit()
    manifest = await asyncio.wait_for(task, timeout=3)
    assert manifest.done_count == 3
