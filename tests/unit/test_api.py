from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from narrator.api.main import create_app
from narrator.core.config import settings
from narrator.core.models import Job, JobParams, JobResult, JobStatus
from narrator.core.state import StateStore

KEY = next(iter(settings.api_key_set))
HDR = {"X-API-Key": KEY}


class FakeEngine:
    async def list_voices(self):
        return ["af_heart", "af_bella"]

    async def health(self):
        return True


@pytest_asyncio.fixture
async def app_client(tmp_path: Path):
    redis = fakeredis.FakeAsyncRedis()
    state = StateStore(tmp_path / "ledger.db", redis)
    await state.connect()
    enqueued: list[tuple[str, str]] = []

    async def fake_enqueue(job_id: str, lane: str) -> None:
        enqueued.append((job_id, lane))

    app = create_app()
    app.state.state = state
    app.state.enqueue = fake_enqueue
    app.state.engine = FakeEngine()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        yield client, state, enqueued
    await state.close()
    await redis.aclose()


async def test_auth_required(app_client):
    client, _, _ = app_client
    r = await client.post("/v1/jobs", json={"text": "hi"})
    assert r.status_code == 401


async def test_create_job_routes_lane_and_enqueues(app_client):
    client, state, enqueued = app_client
    r = await client.post("/v1/jobs", headers=HDR, json={"text": "Hello. World.", "mode": "verbatim"})
    assert r.status_code == 202
    body = r.json()
    assert body["lane"] == "fast"
    assert enqueued and enqueued[0][0] == body["job_id"]
    job = await state.get_job(body["job_id"])
    assert job is not None and job.status is JobStatus.QUEUED


async def test_create_long_verbatim_routes_bulk(app_client):
    client, _, _ = app_client
    long_text = "word " * 60000  # ~35 min verbatim -> bulk
    r = await client.post("/v1/jobs", headers=HDR, json={"text": long_text, "mode": "verbatim"})
    assert r.status_code == 202
    assert r.json()["lane"] == "bulk"


async def test_invalid_params_422(app_client):
    client, _, _ = app_client
    r = await client.post("/v1/jobs", headers=HDR, json={"text": "hi", "speed": 9.0})
    assert r.status_code == 422


async def test_idempotency(app_client):
    client, _, _ = app_client
    hdr = {**HDR, "Idempotency-Key": "abc-123"}
    r1 = await client.post("/v1/jobs", headers=hdr, json={"text": "hello there"})
    r2 = await client.post("/v1/jobs", headers=hdr, json={"text": "hello there"})
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r2.json().get("idempotent") is True


async def test_max_active_jobs_429(app_client, monkeypatch):
    client, state, _ = app_client
    monkeypatch.setattr(settings, "max_active_jobs", 1)
    await state.create_job(Job(id="jb_active", lane="fast", params=JobParams(), status=JobStatus.SYNTHESIZING))
    r = await client.post("/v1/jobs", headers=HDR, json={"text": "another one"})
    assert r.status_code == 429


async def test_rate_limit_429(app_client, monkeypatch):
    client, _, _ = app_client
    monkeypatch.setattr(settings, "rate_limit_per_min", 2)
    ok1 = await client.post("/v1/jobs", headers=HDR, json={"text": "a a"})
    ok2 = await client.post("/v1/jobs", headers=HDR, json={"text": "b b"})
    blocked = await client.post("/v1/jobs", headers=HDR, json={"text": "c c"})
    assert ok1.status_code == 202 and ok2.status_code == 202
    assert blocked.status_code == 429


async def test_get_and_list(app_client):
    client, _, _ = app_client
    created = (await client.post("/v1/jobs", headers=HDR, json={"text": "hi there"})).json()
    got = await client.get(f"/v1/jobs/{created['job_id']}", headers=HDR)
    assert got.status_code == 200 and got.json()["job_id"] == created["job_id"]
    listed = await client.get("/v1/jobs", headers=HDR)
    assert any(j["job_id"] == created["job_id"] for j in listed.json()["jobs"])


async def test_cancel_and_retry(app_client):
    client, state, _ = app_client
    jid = (await client.post("/v1/jobs", headers=HDR, json={"text": "cancel me"})).json()["job_id"]
    c = await client.post(f"/v1/jobs/{jid}/cancel", headers=HDR)
    assert c.status_code == 202
    assert await state.is_cancelled(jid)

    # retry only works from a terminal FAILED/UPLOAD_PENDING state
    job = await state.get_job(jid)
    await state.transition(job, JobStatus.FAILED, error="boom")
    r = await client.post(f"/v1/jobs/{jid}/retry", headers=HDR)
    assert r.status_code == 200 and r.json()["status"] == "QUEUED"


async def test_download_local(app_client, tmp_path: Path):
    client, state, _ = app_client
    artifact = tmp_path / "final.mp3"
    artifact.write_bytes(b"ID3AUDIO")
    job = Job(id="jb_dl", lane="fast", params=JobParams())
    job.result = JobResult(backend="local", local_path=str(artifact), filename="final.mp3", size_bytes=8)
    job.status = JobStatus.COMPLETED
    await state.create_job(job)
    r = await client.get("/v1/jobs/jb_dl/download", headers=HDR)
    assert r.status_code == 200
    assert r.content == b"ID3AUDIO"


async def test_voices_and_health_and_metrics(app_client):
    client, _, _ = app_client
    v = await client.get("/v1/voices", headers=HDR)
    assert "af_heart" in v.json()["voices"]
    h = await client.get("/healthz")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    m = await client.get("/metrics")
    assert m.status_code == 200
    assert "narrator_jobs_active" in m.text


async def test_job_not_found_404(app_client):
    client, _, _ = app_client
    r = await client.get("/v1/jobs/nope", headers=HDR)
    assert r.status_code == 404
