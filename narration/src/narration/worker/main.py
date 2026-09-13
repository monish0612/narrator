"""arq worker: concurrency = 1, hourly reaper, RAM-defer reschedule."""

from __future__ import annotations

import time

from arq.connections import RedisSettings
from arq.cron import cron

from narration.breaker import PipelineBreaker
from narration.config import load_settings
from narration.delete import delete_artifacts
from narration.llm import LlmClient
from narration.logging import configure_logging, get_logger
from narration.pipeline import Pipeline, RamDeferred
from narration.store import STATUS_DELETED, Store
from narration.telegram import Telegram, format_alert
from narration.tts_client import TtsClient

log = get_logger("narration.worker")

REAPER_AGE_S = 168 * 3600


async def generate_narration(ctx, payload: dict) -> dict:
    pipe: Pipeline = ctx["pipeline"]
    store: Store = ctx["store"]
    article_id = str(payload.get("article_id") or "")
    rec = await store.get_article(article_id) if article_id else None
    if rec and rec.get("status") == STATUS_DELETED:
        keys = {rec.get("cache_key")}
        if payload.get("text"):
            keys.add(pipe.cache_for(str(payload.get("text"))))
        for cache in keys:
            if cache:
                await delete_artifacts(store, str(cache))
        return {"status": STATUS_DELETED, "reason": "article_dropped"}
    try:
        return await pipe.generate(payload)
    except RamDeferred as exc:
        rec = await store.get_article(article_id) if article_id else None
        if rec and rec.get("status") == STATUS_DELETED:
            keys = {rec.get("cache_key")}
            if payload.get("text"):
                keys.add(pipe.cache_for(str(payload.get("text"))))
            for cache in keys:
                if cache:
                    await delete_artifacts(store, str(cache))
            return {"status": STATUS_DELETED, "reason": "article_dropped"}
        await ctx["redis"].enqueue_job(
            "generate_narration",
            payload,
            _defer_by=exc.delay_s,
            _job_id=f"nar:{pipe.cache_for(str(payload.get('text') or ''))}:d{int(time.time())}",
        )
        log.info("job.deferred", delay_s=exc.delay_s, article_id=payload.get("article_id"))
        return {"status": "deferred", "delay_s": exc.delay_s}


async def reaper(ctx) -> dict:
    store: Store = ctx["store"]
    tg: Telegram = ctx["tg"]
    now = time.time()
    settings = ctx.get("settings")
    age_s = REAPER_AGE_S
    hours = getattr(settings, "reaper_age_hours", None)
    if hours is not None:
        try:
            age_s = max(3600, int(hours) * 3600)
        except (TypeError, ValueError):
            age_s = REAPER_AGE_S
    keys = store.iter_expired_meta(max_age_s=age_s, now=now)
    deleted = 0
    failed = 0
    scanned = len(keys)
    for cache in keys:
        ok = await delete_artifacts(store, cache)
        if ok:
            deleted += 1
        else:
            failed += 1
            await tg.send(format_alert("reaper delete failed", cache_key=cache))
    stale_tmp = store.iter_stale_tmp(max_age_s=3 * 3600, now=now)
    for p in stale_tmp:
        try:
            p.unlink()
        except OSError:
            failed += 1
    if scanned or failed:
        await tg.send(format_alert("reaper", scanned=scanned, deleted=deleted, failed=failed))
    log.info("reaper.done", scanned=scanned, deleted=deleted, failed=failed)
    return {"scanned": scanned, "deleted": deleted, "failed": failed}


async def startup(ctx) -> None:
    configure_logging()
    settings = load_settings()
    ctx["settings"] = settings
    store = Store(
        ctx["redis"],
        prefix=settings.redis_key_prefix,
        data_dir=settings.data_dir,
        ttl_s=settings.redis_ttl_seconds,
    )
    tg = Telegram(settings.telegram_bot_token, settings.telegram_chat_id)
    llm = LlmClient(settings.llm_base_url, settings.llm_model, gguf_path=settings.gguf_path)
    tts = TtsClient(settings.tts_base_url)
    breaker = PipelineBreaker(
        ctx["redis"],
        settings.redis_key_prefix,
        "pipeline",
        threshold=settings.breaker_threshold,
        cooldown_s=settings.breaker_cooldown_s,
    )
    ctx["store"] = store
    ctx["tg"] = tg
    ctx["llm"] = llm
    ctx["tts"] = tts
    ctx["pipeline"] = Pipeline(
        store,
        llm,
        tts,
        breaker,
        tg,
        voice=settings.tts_voice,
        speed=settings.tts_speed,
        model_version=settings.model_version,
        audio_format=settings.audio_format,
        bitrate=settings.audio_bitrate,
        hd_bitrate=settings.hd_bitrate,
        max_duration_s=settings.max_duration_s,
        word_min=settings.word_target_min,
        word_max=settings.word_target_max,
        ram_floor=settings.ram_floor_bytes,
        ram_alert_floor=settings.ram_alert_floor_bytes,
    )


async def shutdown(ctx) -> None:
    await ctx["llm"].aclose()
    await ctx["tts"].aclose()


def _redis_settings() -> RedisSettings:
    try:
        return RedisSettings.from_dsn(load_settings().redis_url)
    except Exception:
        return RedisSettings(host="localhost", database=1)


class WorkerSettings:
    functions = [generate_narration, reaper]
    cron_jobs = [
        cron(
            reaper,
            hour={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            minute={7},
        )
    ]
    max_jobs = 1
    job_timeout = 2700
    max_tries = 3
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _redis_settings()
