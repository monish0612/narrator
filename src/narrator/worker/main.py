"""arq worker entrypoint (ARCHITECTURE.md sections 2-3, 8, 15).

One process per lane (WORKER_LANE=fast|bulk), each consuming its own queue. The
fast worker also owns reconciliation + cron (cleanup / cache eviction / ledger
backup). Dependencies are built once in ``startup`` and shared via the arq ctx.
"""

from __future__ import annotations

import os

import httpx
from arq import cron, func
from arq.connections import RedisSettings

from narrator.core.breaker import CircuitBreaker
from narrator.core.cache import TTSCache
from narrator.core.config import settings
from narrator.core.factory import build_engine, build_storage
from narrator.core.gate import SynthGate
from narrator.core.logging import configure_logging, get_logger
from narrator.core.state import StateStore
from narrator.pipeline.assemble import Assembler
from narrator.worker.shutdown import GracefulShutdown
from narrator.worker.tasks import JobRunner, backup_ledger, cleanup_expired

log = get_logger(__name__)

WORKER_LANE = os.getenv("WORKER_LANE", "fast")
QUEUE_NAME = f"narrator:{WORKER_LANE}"


def lane_queue(lane: str) -> str:
    return f"narrator:{lane}"


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def synth_job(ctx: dict, job_id: str) -> None:
    runner: JobRunner = ctx["runner"]
    await runner.run(job_id)


async def _reconcile(ctx: dict) -> None:
    state: StateStore = ctx["state"]
    pool = ctx["redis"]

    async def enqueue(jid: str, lane: str) -> None:
        await pool.enqueue_job("synth_job", jid, _queue_name=lane_queue(lane))

    orphans = await state.reconcile(enqueue)
    if orphans:
        log.info("worker.reconciled", count=len(orphans))


async def cron_cleanup(ctx: dict) -> None:
    await cleanup_expired(
        ctx["state"], settings.retention_days, settings.jobs_dir, settings.outputs_dir
    )


async def cron_cache_evict(ctx: dict) -> None:
    await ctx["cache"].evict_to_limit()


async def cron_backup(ctx: dict) -> None:
    await backup_ledger(settings.ledger_path, settings.data_dir / "backups")


async def startup(ctx: dict) -> None:
    configure_logging()
    redis = ctx["redis"]  # ArqRedis is a redis.asyncio client
    state = StateStore(settings.ledger_path, redis)
    await state.connect()

    client = httpx.AsyncClient(timeout=300)
    tts_breaker = CircuitBreaker(redis, "tts")
    gemini_breaker = CircuitBreaker(redis, "gemini")
    drive_breaker = CircuitBreaker(redis, "drive")

    cache = TTSCache(state, settings.cache_dir, settings.cache_max_gb)
    gate = SynthGate(redis)
    engine = build_engine(client, tts_breaker)
    storage = build_storage(state, drive_breaker=drive_breaker)

    gemini_client = None
    if settings.gemini_api_key:
        from narrator.processors.gemini_client import GoogleGenaiClient

        gemini_client = GoogleGenaiClient(settings.gemini_api_key, settings.gemini_model)

    runner = JobRunner(
        state=state,
        engine=engine,
        cache=cache,
        gate=gate,
        storage=storage,
        assembler=Assembler(settings.silence_dir),
        jobs_dir=settings.jobs_dir,
        gemini_client=gemini_client,
        gemini_breaker=gemini_breaker,
        synth_concurrency=settings.synth_concurrency,
        non_english_policy=settings.non_english_policy,
        max_failed_pct=settings.max_failed_chunk_pct,
        webhook_secret=settings.webhook_secret,
    )

    ctx.update(
        state=state,
        client=client,
        cache=cache,
        gate=gate,
        runner=runner,
    )
    shutdown = GracefulShutdown()
    shutdown.install()
    ctx["shutdown"] = shutdown

    if WORKER_LANE == "fast":
        await _reconcile(ctx)
    log.info("worker.started", lane=WORKER_LANE, queue=QUEUE_NAME)


async def shutdown_(ctx: dict) -> None:
    client: httpx.AsyncClient = ctx.get("client")
    state: StateStore = ctx.get("state")
    if client is not None:
        await client.aclose()
    if state is not None:
        await state.close()
    log.info("worker.stopped", lane=WORKER_LANE)


_CRON = (
    [
        cron(cron_cleanup, hour=3, minute=0, run_at_startup=False),
        cron(cron_cache_evict, minute={0, 30}, run_at_startup=False),
        cron(cron_backup, hour=4, minute=0, run_at_startup=False),
    ]
    if WORKER_LANE == "fast"
    else []
)


class WorkerSettings:
    functions = [func(synth_job, name="synth_job", max_tries=5)]
    queue_name = QUEUE_NAME
    redis_settings = redis_settings()
    on_startup = startup
    on_shutdown = shutdown_
    cron_jobs = _CRON
    max_jobs = settings.max_active_jobs if WORKER_LANE == "fast" else 1
    job_timeout = 24 * 60 * 60
    keep_result = settings.retention_days * 86400
    allow_abort_jobs = True
