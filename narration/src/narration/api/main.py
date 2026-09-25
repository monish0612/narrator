"""FastAPI surface. Internal network only — the Nexus API proxies this."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
import time

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from narration.breaker import PipelineBreaker
from narration.config import load_settings
from narration.enqueue_policy import classify_enqueue
from narration.http_range import range_file_response
from narration.keys import lock_key
from narration.llm import LlmClient
from narration.logging import configure_logging, get_logger
from narration.normalize import build_cache_key
from narration.tts_cloud import SCRIPT_VERSION, normalize_engine, normalize_voice
from narration.pipeline import drop_article_audio, mark_listened
from narration.spoken_host import HOST_TOUCH_VERSION, ensure_host_touch
from narration.store import STATUS_DELETED, STATUS_FALLBACK, STATUS_QUEUED, STATUS_READY, Store
from narration.telegram import Telegram
from narration.tts_client import TtsClient

log = get_logger("narration.api")

_touching: set[str] = set()


def schedule_host_touch(app: FastAPI, rec: dict) -> None:
    """Append the spoken closer without blocking status or playback."""
    cache = str(rec.get("cache_key") or "").strip()
    if not cache or cache in _touching or rec.get("host_touch") == HOST_TOUCH_VERSION:
        return
    _touching.add(cache)

    async def _run() -> None:
        try:
            await ensure_host_touch(
                app.state.store,
                app.state.tts,
                rec,
                voice=app.state.settings.tts_voice,
                speed=app.state.settings.tts_speed,
                bitrate=app.state.settings.audio_bitrate,
            )
        finally:
            _touching.discard(cache)

    asyncio.get_running_loop().create_task(_run())


class JobIn(BaseModel):
    article_id: str = Field(..., min_length=1)
    title: str = ""
    source: str = ""
    category: str = ""
    text: str = Field(..., min_length=1)
    hd: bool = False
    voice: str | None = None
    model: str | None = None
    tts_model: str | None = None
    force: bool = False


class CompleteIn(BaseModel):
    cache_key: str | None = None
    article_id: str | None = None


class DropIn(BaseModel):
    article_id: str | None = None
    article_ids: list[str] = Field(default_factory=list)


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
    app.state.llm = LlmClient(
        settings.llm_base_url,
        settings.llm_model,
        gguf_path=settings.gguf_path,
        gemini_api_key=settings.gemini_api_key,
        fallback_models=[m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()],
        timeout=45,
    )
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
        engine = normalize_engine(body.tts_model)
        voice = normalize_voice(engine, body.voice)
        cache = build_cache_key(
            article_text=body.text,
            voice=voice,
            speed=s.tts_speed,
            model_version=f"{SCRIPT_VERSION}+{engine}",
            audio_format=s.audio_format,
            bitrate=s.audio_bitrate,
        )
        existing_art = await store.get_article(body.article_id)
        hit = await store.get_cache(cache)
        requested_ready = bool(hit and hit.get("status") == STATUS_READY)
        decision = classify_enqueue(
            existing=existing_art,
            requested_ready=requested_ready,
            force=body.force,
        )
        if decision == "deleted":
            return {"status": STATUS_DELETED, "reason": "article_dropped"}
        if decision == "keep_ready":
            kept = dict(existing_art or {})
            kept["status"] = STATUS_READY
            kept["cache_hit"] = True
            return kept
        if decision == "in_flight":
            return dict(existing_art or {"status": STATUS_QUEUED, "cache_key": cache})
        if decision == "bind_ready":
            await store.bind_article(
                body.article_id,
                {**hit, "article_id": body.article_id, "cache_key": cache, "status": STATUS_READY},
            )
            return {"status": STATUS_READY, "cache_key": cache, "cache_hit": True}

        if existing_art and existing_art.get("status") == STATUS_DELETED and body.force:
            await store.bind_article(
                body.article_id,
                {
                    "article_id": body.article_id,
                    "cache_key": cache,
                    "status": STATUS_QUEUED,
                    "reason": "resurrect_replay",
                },
            )

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
            if body.force:
                # The worker runs this article before whatever backlog job
                # it dequeues next, so an open Listen does not wait behind
                # days of ingest.
                rush_key = f"{s.redis_key_prefix}rush"
                await app.state.redis.set(
                    rush_key,
                    json.dumps(body.model_dump()),
                    ex=900,
                )
            await app.state.pool.enqueue_job(
                "generate_narration",
                body.model_dump(),
                _job_id=f"nar:{cache}:{body.article_id}:{int(time.time())}",
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
            opus = app.state.store.opus_path(str(cache))
            chunk0 = app.state.store.chunk_opus_path(str(cache), 0)
            if (not live and not chunk0.exists()) or (not opus.exists() and not chunk0.exists()):
                rec["status"] = STATUS_FALLBACK
                rec["reason"] = "audio_missing"
                await app.state.store.bind_article(article_id, rec)
            else:
                rec = {**live, **rec, "article_id": article_id, "cache_key": cache, "status": STATUS_READY}
                schedule_host_touch(app, rec)
        return rec

    @app.get("/v1/audio/{cache_key}/chunks/{index}.opus")
    async def chunk_audio(cache_key: str, index: int, request: Request, _: None = Depends(require_key)):
        path = app.state.store.chunk_opus_path(cache_key, index)
        if not path.exists():
            raise HTTPException(status_code=404, detail="not_ready")
        return range_file_response(path, request)

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
        schedule_host_touch(app, rec)
        path = store.opus_path(cache_key, hd=hd and bool(rec.get("hd_file_path")))
        if not path.exists():
            path = store.opus_path(cache_key)
        if not path.exists():
            raise HTTPException(status_code=404, detail="not_found")
        return range_file_response(path, request)

    @app.post("/v1/complete")
    async def complete(body: CompleteIn, _: None = Depends(require_key)):
        return await mark_listened(
            app.state.store,
            article_id=body.article_id,
            cache=body.cache_key,
        )

    @app.post("/v1/drop")
    async def drop(body: DropIn, _: None = Depends(require_key)):
        store: Store = app.state.store
        ids: list[str] = []
        if body.article_id:
            ids.append(body.article_id)
        ids.extend(body.article_ids)
        seen: set[str] = set()
        results = []
        for aid in ids:
            aid = str(aid or "").strip()
            if not aid or aid in seen:
                continue
            seen.add(aid)
            results.append(await drop_article_audio(store, aid, app.state.tg))
        return {"dropped": len(results), "results": results}

    return app


app = create_app()
