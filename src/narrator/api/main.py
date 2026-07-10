"""Narrator API app (ARCHITECTURE.md section 7).

Auth'd REST + SSE over the job lifecycle. Dependencies (state, arq enqueue pool,
TTS engine for /v1/voices) are created in the lifespan unless already injected on
``app.state`` (tests inject fakes so no Redis/arq is required).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from narrator.api.routes import jobs, system
from narrator.core.config import settings
from narrator.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    owns = getattr(app.state, "state", None) is None
    if owns:
        configure_logging()
        import redis.asyncio as aioredis
        from arq import create_pool
        from arq.connections import RedisSettings

        from narrator.core.breaker import CircuitBreaker
        from narrator.core.factory import build_engine
        from narrator.core.state import StateStore

        redis = aioredis.from_url(settings.redis_url)
        state = StateStore(settings.ledger_path, redis)
        await state.connect()
        app.state.state = state
        app.state._redis = redis

        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        app.state._pool = pool

        async def _enqueue(job_id: str, lane: str) -> None:
            await pool.enqueue_job("synth_job", job_id, _queue_name=f"narrator:{lane}")

        app.state.enqueue = _enqueue

        client = httpx.AsyncClient(timeout=30)
        app.state._client = client
        app.state.engine = build_engine(client, CircuitBreaker(redis, "tts"))
        log.info("api.started")
    try:
        yield
    finally:
        if owns:
            for attr in ("_client",):
                obj = getattr(app.state, attr, None)
                if obj is not None:
                    await obj.aclose()
            state = getattr(app.state, "state", None)
            if state is not None:
                await state.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Narrator", version="2.0.0", lifespan=lifespan)
    app.state.state = None
    app.state.enqueue = None
    app.state.engine = None
    app.include_router(system.router)
    app.include_router(jobs.router)
    return app


app = create_app()
