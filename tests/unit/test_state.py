from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest_asyncio

from narrator.core.models import Job, JobParams, JobStatus
from narrator.core.state import StateStore


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    redis = fakeredis.FakeAsyncRedis()
    st = StateStore(tmp_path / "ledger.db", redis)
    await st.connect()
    try:
        yield st
    finally:
        await st.close()
        await redis.aclose()


def _job(job_id: str = "jb_1", lane: str = "fast", **kw) -> Job:
    return Job(id=job_id, lane=lane, params=JobParams(mode="verbatim"), **kw)


async def test_create_and_get(store: StateStore):
    await store.create_job(_job())
    fetched = await store.get_job("jb_1")
    assert fetched is not None
    assert fetched.status is JobStatus.QUEUED
    # mirrored to redis
    assert await store.redis.get("job:jb_1") is not None


async def test_transition_and_progress(store: StateStore):
    job = await store.create_job(_job(total_chunks=4))
    await store.transition(job, JobStatus.SYNTHESIZING, stage="synthesizing")
    await store.update_progress(job, done_chunks=2)
    fetched = await store.get_job("jb_1")
    assert fetched.status is JobStatus.SYNTHESIZING
    assert fetched.stage == "synthesizing"
    assert fetched.done_chunks == 2
    assert fetched.progress == 0.5


async def test_upload_pending_path(store: StateStore):
    job = await store.create_job(_job())
    await store.transition(job, JobStatus.UPLOADING, stage="uploading")
    await store.transition(job, JobStatus.UPLOAD_PENDING, stage="upload_pending")
    fetched = await store.get_job("jb_1")
    assert fetched.status is JobStatus.UPLOAD_PENDING
    assert fetched.status.is_terminal
    # UPLOAD_PENDING must not count as active work
    assert await store.count_active() == 0


async def test_count_active(store: StateStore):
    a = await store.create_job(_job("jb_a"))
    await store.create_job(_job("jb_b"))
    assert await store.count_active() == 2
    await store.transition(a, JobStatus.COMPLETED)
    assert await store.count_active() == 1


async def test_cancel_flags(store: StateStore):
    await store.create_job(_job())
    assert not await store.is_cancelled("jb_1")
    await store.request_cancel("jb_1")
    assert await store.is_cancelled("jb_1")
    await store.clear_cancel("jb_1")
    assert not await store.is_cancelled("jb_1")


async def test_idempotency(store: StateStore):
    assert await store.idempotency_setnx("key-1", "jb_1") is None
    assert await store.idempotency_setnx("key-1", "jb_2") == "jb_1"


async def test_kv(store: StateStore):
    assert await store.kv_get("folder:2026-07") is None
    await store.kv_set("folder:2026-07", "drive-id-123")
    assert await store.kv_get("folder:2026-07") == "drive-id-123"


async def test_reconcile_reenqueues_orphans(store: StateStore):
    job = await store.create_job(_job("jb_x", lane="bulk"))
    await store.transition(job, JobStatus.SYNTHESIZING, stage="synthesizing")

    seen: list[tuple[str, str]] = []

    async def fake_enqueue(job_id: str, lane: str) -> None:
        seen.append((job_id, lane))

    orphans = await store.reconcile(fake_enqueue)
    assert orphans == [("jb_x", "bulk")]
    assert seen == [("jb_x", "bulk")]
    fetched = await store.get_job("jb_x")
    assert fetched.status is JobStatus.QUEUED
    assert fetched.stall_reason is None
