"""State store (ARCHITECTURE.md section 6 / 15-F).

SQLite (WAL) is the durable truth; Redis is the live mirror + pub/sub event bus.
Every job state transition flows through here so that SSE, ``/metrics`` and
crash recovery stay consistent. No other module writes the ``jobs`` table.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import aiosqlite

from narrator.core.logging import get_logger
from narrator.core.models import Job, JobStatus

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    lane        TEXT NOT NULL,
    data        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS kv (
    k           TEXT PRIMARY KEY,
    v           TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_index (
    key         TEXT PRIMARY KEY,
    size        INTEGER NOT NULL,
    last_used   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_last_used ON cache_index(last_used);
"""

_ACTIVE_STATUS_VALUES = tuple(
    s.value
    for s in (
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
        JobStatus.SYNTHESIZING,
        JobStatus.ASSEMBLING,
        JobStatus.UPLOADING,
    )
)

EnqueueFn = Callable[[str, str], Awaitable[Any]]


def _events_channel(job_id: str) -> str:
    return f"job:{job_id}:events"


def _mirror_key(job_id: str) -> str:
    return f"job:{job_id}"


def _cancel_key(job_id: str) -> str:
    return f"job:{job_id}:cancel"


class StateStore:
    """Async SQLite + Redis state manager. One instance per process."""

    def __init__(self, db_path: str | Path, redis_client: Any) -> None:
        self._db_path = Path(db_path)
        self._redis = redis_client
        self._db: aiosqlite.Connection | None = None

    # --- lifecycle -----------------------------------------------------------
    async def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("StateStore.connect() was not awaited")
        return self._db

    @property
    def redis(self) -> Any:
        return self._redis

    async def __aenter__(self) -> StateStore:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # --- job CRUD ------------------------------------------------------------
    async def create_job(self, job: Job) -> Job:
        now = time.time()
        job.created_at = job.created_at or now
        job.updated_at = now
        await self.db.execute(
            "INSERT INTO jobs (id, status, lane, data, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                job.id,
                job.status.value,
                job.lane,
                job.model_dump_json(),
                job.created_at,
                job.updated_at,
            ),
        )
        await self.db.commit()
        await self._mirror(job)
        await self.publish_event(job.id, self._event_payload(job, "created"))
        return job

    async def get_job(self, job_id: str) -> Job | None:
        cur = await self.db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return None
        return Job.model_validate_json(row["data"])

    async def list_jobs(self, *, limit: int = 50, offset: int = 0) -> list[Job]:
        cur = await self.db.execute(
            "SELECT data FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [Job.model_validate_json(r["data"]) for r in rows]

    async def count_active(self) -> int:
        placeholders = ",".join("?" for _ in _ACTIVE_STATUS_VALUES)
        cur = await self.db.execute(
            f"SELECT COUNT(*) AS n FROM jobs WHERE status IN ({placeholders})",
            _ACTIVE_STATUS_VALUES,
        )
        row = await cur.fetchone()
        await cur.close()
        return int(row["n"]) if row else 0

    async def counts_by_status(self) -> dict[str, int]:
        cur = await self.db.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        rows = await cur.fetchall()
        await cur.close()
        return {r["status"]: int(r["n"]) for r in rows}

    async def save_job(self, job: Job, *, event: str = "update") -> Job:
        """Persist the full job record, mirror to Redis, publish an event."""
        job.updated_at = time.time()
        await self.db.execute(
            "UPDATE jobs SET status = ?, lane = ?, data = ?, updated_at = ? WHERE id = ?",
            (job.status.value, job.lane, job.model_dump_json(), job.updated_at, job.id),
        )
        await self.db.commit()
        await self._mirror(job)
        await self.publish_event(job.id, self._event_payload(job, event))
        return job

    async def transition(
        self,
        job: Job,
        status: JobStatus,
        *,
        stage: str | None = None,
        stall_reason: str | None = None,
        error: str | None = None,
    ) -> Job:
        """Move a job to a new status and persist/publish it atomically."""
        job.status = status
        if stage is not None:
            job.stage = stage
        # A live status always clears any prior stall_reason unless one is given.
        job.stall_reason = stall_reason
        if error is not None:
            job.error = error
        await self.save_job(job, event="transition")
        log.info("job.transition", status=status.value, stage=job.stage)
        return job

    async def update_progress(
        self,
        job: Job,
        *,
        done_chunks: int | None = None,
        total_chunks: int | None = None,
        eta_seconds: float | None = None,
        stall_reason: str | None = None,
    ) -> Job:
        if done_chunks is not None:
            job.done_chunks = done_chunks
        if total_chunks is not None:
            job.total_chunks = total_chunks
        if eta_seconds is not None:
            job.eta_seconds = eta_seconds
        job.stall_reason = stall_reason
        await self.save_job(job, event="progress")
        return job

    # --- Redis mirror + events ----------------------------------------------
    async def _mirror(self, job: Job) -> None:
        try:
            await self._redis.set(_mirror_key(job.id), job.model_dump_json())
        except Exception as exc:  # mirror is best-effort; SQLite is truth
            log.warning("state.mirror_failed", job_id=job.id, error=str(exc))

    def _event_payload(self, job: Job, event: str) -> dict[str, Any]:
        return {
            "event": event,
            "job_id": job.id,
            "status": job.status.value,
            "stage": job.stage,
            "progress": job.progress,
            "done_chunks": job.done_chunks,
            "total_chunks": job.total_chunks,
            "stall_reason": job.stall_reason,
            "eta_seconds": job.eta_seconds,
            "ts": time.time(),
        }

    async def publish_event(self, job_id: str, payload: dict[str, Any]) -> None:
        try:
            await self._redis.publish(_events_channel(job_id), json.dumps(payload))
        except Exception as exc:
            log.warning("state.publish_failed", job_id=job_id, error=str(exc))

    # --- cancellation --------------------------------------------------------
    async def request_cancel(self, job_id: str) -> None:
        await self._redis.set(_cancel_key(job_id), "1")

    async def is_cancelled(self, job_id: str) -> bool:
        return bool(await self._redis.exists(_cancel_key(job_id)))

    async def clear_cancel(self, job_id: str) -> None:
        await self._redis.delete(_cancel_key(job_id))

    # --- idempotency ---------------------------------------------------------
    async def idempotency_setnx(self, key: str, job_id: str, ttl_seconds: int = 86400) -> str | None:
        """Return the existing job id for this key, or None if we just claimed it."""
        stored = await self._redis.set(f"idem:{key}", job_id, nx=True, ex=ttl_seconds)
        if stored:
            return None
        existing = await self._redis.get(f"idem:{key}")
        if existing is None:
            return None
        return existing.decode() if isinstance(existing, bytes) else str(existing)

    # --- key/value (folder ids etc.) ----------------------------------------
    async def kv_get(self, key: str) -> str | None:
        cur = await self.db.execute("SELECT v FROM kv WHERE k = ?", (key,))
        row = await cur.fetchone()
        await cur.close()
        return row["v"] if row else None

    async def kv_set(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT INTO kv (k, v, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v, updated_at = excluded.updated_at",
            (key, value, time.time()),
        )
        await self.db.commit()

    # --- cache index (used by core/cache.py) ---------------------------------
    async def cache_touch(self, key: str, size: int) -> None:
        await self.db.execute(
            "INSERT INTO cache_index (key, size, last_used) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET last_used = excluded.last_used",
            (key, size, time.time()),
        )
        await self.db.commit()

    async def cache_get(self, key: str) -> aiosqlite.Row | None:
        cur = await self.db.execute("SELECT key, size, last_used FROM cache_index WHERE key = ?", (key,))
        row = await cur.fetchone()
        await cur.close()
        return row

    async def cache_delete(self, key: str) -> None:
        await self.db.execute("DELETE FROM cache_index WHERE key = ?", (key,))
        await self.db.commit()

    async def cache_total_size(self) -> int:
        cur = await self.db.execute("SELECT COALESCE(SUM(size), 0) AS total FROM cache_index")
        row = await cur.fetchone()
        await cur.close()
        return int(row["total"]) if row else 0

    async def cache_lru(self, limit: int) -> list[aiosqlite.Row]:
        cur = await self.db.execute(
            "SELECT key, size, last_used FROM cache_index ORDER BY last_used ASC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        await cur.close()
        return rows

    # --- crash recovery ------------------------------------------------------
    async def reconcile(self, enqueue: EnqueueFn | None = None) -> list[tuple[str, str]]:
        """Re-enqueue orphaned running jobs after a crash/restart.

        Any job left in an active (non-terminal) state is reset to QUEUED and,
        if an ``enqueue`` callback is supplied, re-submitted to its lane. Resume
        is safe because the manifest checkpoints every finished chunk.
        """
        placeholders = ",".join("?" for _ in _ACTIVE_STATUS_VALUES)
        cur = await self.db.execute(
            f"SELECT data FROM jobs WHERE status IN ({placeholders})",
            _ACTIVE_STATUS_VALUES,
        )
        rows = await cur.fetchall()
        await cur.close()
        orphans: list[tuple[str, str]] = []
        for row in rows:
            job = Job.model_validate_json(row["data"])
            job.status = JobStatus.QUEUED
            job.stall_reason = None
            await self.save_job(job, event="reconcile")
            orphans.append((job.id, job.lane))
            log.info("state.reconcile.orphan", job_id=job.id, lane=job.lane)
        if enqueue is not None:
            for job_id, lane in orphans:
                await enqueue(job_id, lane)
        return orphans
