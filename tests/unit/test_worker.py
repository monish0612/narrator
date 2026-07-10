from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest
import pytest_asyncio

from narrator.core.cache import TTSCache
from narrator.core.errors import DeliveryTerminal
from narrator.core.gate import SynthGate
from narrator.core.models import Job, JobParams, JobStatus, Manifest, SynthesisResult
from narrator.core.state import StateStore
from narrator.pipeline.assemble import AssemblyResult
from narrator.storage.local import LocalStorage
from narrator.worker.tasks import (
    JobRunner,
    choose_lane,
    estimate_minutes,
    sign_payload,
)


def test_estimate_and_lane_routing():
    # explainer is fixed to the target length regardless of input size
    assert estimate_minutes(10_000_000, "explainer", explainer_target=30) == 30
    # verbatim scales with input
    assert estimate_minutes(8500, "verbatim", explainer_target=30) == pytest.approx(10.0, rel=0.01)
    assert choose_lane(10, 20) == "fast"
    assert choose_lane(45, 20) == "bulk"


def test_sign_payload_is_deterministic():
    sig1 = sign_payload("secret", b'{"a":1}')
    sig2 = sign_payload("secret", b'{"a":1}')
    assert sig1 == sig2
    assert sign_payload("other", b'{"a":1}') != sig1
    assert len(sig1) == 64


class FakeEngine:
    _base = "http://tts:8880/v1"

    def __init__(self):
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * 4800)
        self._wav = buf.getvalue()

    async def synthesize(self, text, *, voice, speed, response_format="wav"):
        return SynthesisResult(audio=self._wav, duration_ms=200, sample_rate=24000)


class FakeAssembler:
    async def assemble(self, chunks, manifest, job_dir, *, title, output_format="mp3", repair=None, cancel_check=None):
        out = Path(job_dir) / f"final.{output_format}"
        out.write_bytes(b"ID3" + b"\x00" * 4096)
        return AssemblyResult(out, duration_seconds=12.3, size_bytes=out.stat().st_size)


class FailingStorage:
    async def upload(self, local_path, *, filename, mime_type, meta):
        raise DeliveryTerminal("invalid_grant")


@pytest_asyncio.fixture
async def env(tmp_path: Path):
    redis = fakeredis.FakeAsyncRedis()
    state = StateStore(tmp_path / "ledger.db", redis)
    await state.connect()
    yield {
        "state": state,
        "redis": redis,
        "cache": TTSCache(state, tmp_path / "cache", max_gb=5),
        "gate": SynthGate(redis, poll_interval=0.01),
        "jobs_dir": tmp_path / "jobs",
        "outputs": tmp_path / "outputs",
        "tmp": tmp_path,
    }
    await state.close()
    await redis.aclose()


def _runner(env, *, storage=None, gemini_client=None) -> JobRunner:
    return JobRunner(
        state=env["state"],
        engine=FakeEngine(),
        cache=env["cache"],
        gate=env["gate"],
        storage=storage or LocalStorage(env["outputs"]),
        assembler=FakeAssembler(),
        jobs_dir=env["jobs_dir"],
        gemini_client=gemini_client,
        gemini_breaker=None,
    )


async def _make_job(env, mode="verbatim", *, cancelled=False) -> Job:
    src = env["tmp"] / "input.txt"
    src.write_text("Hello world. This is a test.\n\nSecond paragraph here now.", encoding="utf-8")
    job = Job(id="jb_1", lane="fast", params=JobParams(mode=mode), input_path=str(src), source_kind="text")
    await env["state"].create_job(job)
    if cancelled:
        await env["state"].request_cancel(job.id)
    return job


async def test_full_pipeline_completes(env):
    await _make_job(env)
    await _runner(env).run("jb_1")
    job = await env["state"].get_job("jb_1")
    assert job.status is JobStatus.COMPLETED
    assert job.result is not None
    assert job.result.backend == "local"
    assert Path(job.result.local_path).exists()
    assert job.result.duration_seconds == 12.3


async def test_delivery_terminal_goes_upload_pending(env):
    await _make_job(env)
    await _runner(env, storage=FailingStorage()).run("jb_1")
    job = await env["state"].get_job("jb_1")
    assert job.status is JobStatus.UPLOAD_PENDING
    assert job.result.local_path is not None
    assert Path(job.result.local_path).exists()
    assert "invalid_grant" in (job.stall_reason or "")


async def test_cancel_before_run(env):
    await _make_job(env, cancelled=True)
    await _runner(env).run("jb_1")
    job = await env["state"].get_job("jb_1")
    assert job.status is JobStatus.CANCELLED


async def test_explainer_without_gemini_fails_clearly(env):
    await _make_job(env, mode="explainer")
    await _runner(env, gemini_client=None).run("jb_1")
    job = await env["state"].get_job("jb_1")
    assert job.status is JobStatus.FAILED
    assert "GEMINI_API_KEY" in (job.error or "")


async def test_manifest_written(env):
    await _make_job(env)
    await _runner(env).run("jb_1")
    mpath = env["jobs_dir"] / "jb_1" / "manifest.json"
    assert mpath.exists()
    Manifest.model_validate_json(mpath.read_text())
