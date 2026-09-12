"""FastAPI surface. Internal network only — the Nexus API proxies this."""

from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from narration.breaker import PipelineBreaker
from narration.config import load_settings
from narration.http_range import range_file_response
from narration.keys import lock_key
from narration.llm import LlmClient
from narration.logging import configure_logging, get_logger
from narration.normalize import build_cache_key
from narration.pipeline import complete_listen
from narration.store import STATUS_FALLBACK, STATUS_QUEUED, STATUS_READY, Store
from narration.telegram import Telegram
from narration.tts_client import TtsClient

log = get_logger("narration.api")


class JobIn(BaseModel):
    article_id: str = Field(..., min_length=1)
    title: str = ""
    source: str = ""
    category: str = ""
    text: str = Field(..., min_length=1)
    hd: bool = False
    voice: str | None = None


class CompleteIn(BaseModel):
    cache_key: str | None = None
    article_id: str | None = None


def _auth(settings, x_api_key: str | None) -> None:
    if not settings.narration_api_key:
        return
    if not x_api_key or x_api_key != settings.narration_api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = load_settings()
    app.state.settings = settings
    r = redis.from_url(settings.redis_url, decode_responses=False)
    app.state.redis = r
    app.state.store = Store(
        r,
        prefix=settings.redis_key_prefix,
        data_dir=settings.data_dir,
        ttl_s=settings.redis_ttl_seconds,
    )
    app.state.tg = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    app.state.breaker = PipelineBreaker(
        r,
        settings.redis_key_prefix,
        "pipeline",
        threshold=settings.breaker_threshold,
        cooldown_s=settings.breaker_cooldown_s,
    )
    app.state.pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.tts = TtsClient(settings.tts_base_url)
    app.state.llm = LlmClient(settings.llm_base_url, settings.llm_model, gguf_path=settings.gguf_path)
    yield
    await app.state.tts.aclose()
    await app.state.llm.aclose()
    await r.aclose()
    await app.state.pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="narration-orchestrator", version="1.0.0", lifespan=lifespan)

    def settings():
        return app.state.settings

    def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        _auth(app.state.settings, x_api_key)

    @app.get("/healthz")
    async def healthz():
        ok_redis = False
        try:
            await app.state.redis.ping()
            ok_redis = True
        except Exception:
            ok_redis = False
        tts_ok = await app.state.tts.health()
        status = "ok" if ok_redis else "degraded"
        code = 200 if ok_redis else 503
        return JSONResponse(
            status_code=code,
            content={"status": status, "redis": ok_redis, "tts": tts_ok},
        )

    @app.post("/v1/jobs")
    async def enqueue(body: JobIn, _: None = Depends(require_key)):
        s = app.state.settings
        store: Store = app.state.store
        cache = build_cache_key(
            article_text=body.text,
            voice=body.voice or s.tts_voice,
            speed=s.tts_speed,
            model_version=s.model_version,
            audio_format=s.audio_format,
            bitrate=s.audio_bitrate,
        )
        hit = await store.get_cache(cache)
        if hit and hit.get("status") == STATUS_READY:
            await store.bind_article(
                body.article_id,
                {**hit, "article_id": body.article_id, "cache_key": cache, "status": STATUS_READY},
            )
            return {"status": STATUS_READY, "cache_key": cache, "cache_hit": True}

        if await app.state.breaker.is_open():
            rec = {"article_id": body.article_id, "cache_key": cache, "status": STATUS_FALLBACK, "reason": "breaker_open"}
            await store.bind_article(body.article_id, rec)
            return rec

        lock = lock_key(s.redis_key_prefix, cache)
        got = await app.state.redis.set(lock, body.article_id, nx=True, ex=30)
        if got:
            await store.bind_article(
                body.article_id,
                {"article_id": body.article_id, "cache_key": cache, "status": STATUS_QUEUED},
            )
            await app.state.pool.enqueue_job(
                "generate_narration",
                body.model_dump(),
                _job_id=f"nar:{cache}",
            )
            # Short enqueue lock — the worker takes a longer one.
            await app.state.redis.delete(lock)
        else:
            await store.bind_article(
                body.article_id,
                {"article_id": body.article_id, "cache_key": cache, "status": STATUS_QUEUED},
            )
        return {"status": STATUS_QUEUED, "cache_key": cache}

    @app.get("/v1/jobs/{article_id}")
    async def job_status(article_id: str, _: None = Depends(require_key)):
        rec = await app.state.store.get_article(article_id)
        if not rec:
            raise HTTPException(status_code=404, detail="not_found")
        cache = rec.get("cache_key")
        if cache and rec.get("status") == STATUS_READY:
            live = await app.state.store.get_cache(cache)
            if not live:
                rec["status"] = STATUS_FALLBACK
                rec["reason"] = "audio_missing"
                await app.state.store.bind_article(article_id, rec)
        return rec

    @app.get("/v1/audio/{cache_key}.opus")
    async def audio(
        cache_key: str,
        request: Request,
        hd: bool = False,
        _: None = Depends(require_key),
    ):
        store: Store = app.state.store
        rec = await store.get_cache(cache_key)
        if not rec:
            raise HTTPException(status_code=404, detail="not_found")
        path = store.opus_path(cache_key, hd=hd and bool(rec.get("hd_file_path")))
        if not path.exists():
            path = store.opus_path(cache_key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="not_found")
        return range_file_response(path, request)

    @app.post("/v1/complete")
    async def complete(body: CompleteIn, _: None = Depends(require_key)):
        store: Store = app.state.store
        cache = body.cache_key
        if not cache and body.article_id:
            rec = await store.get_article(body.article_id)
            cache = (rec or {}).get("cache_key")
        if not cache:
            raise HTTPException(status_code=400, detail="cache_key required")
        ok = await complete_listen(store, cache, app.state.tg)
        if body.article_id:
            await store.bind_article(
                body.article_id,
                {"article_id": body.article_id, "cache_key": cache, "status": "deleted"},
            )
        return {"deleted": ok, "cache_key": cache}

    return app


app = create_app()
