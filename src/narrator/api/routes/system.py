"""Health, readiness, metrics and voices routes (ARCHITECTURE.md sections 7 / 16)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from narrator.api.deps import get_state, require_api_key
from narrator.core.breaker import CircuitBreaker
from narrator.core.config import settings
from narrator.core.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    state = get_state(request)
    redis_ok = False
    try:
        redis_ok = bool(await state.redis.ping())
    except Exception:
        redis_ok = False
    status = "ok" if redis_ok else "degraded"
    return {"status": status, "redis": redis_ok, "version": "2.0.0"}


@router.get("/readyz")
async def readyz(request: Request) -> dict:
    state = get_state(request)
    engine = getattr(request.app.state, "engine", None)
    tts_ok = True
    if engine is not None:
        try:
            tts_ok = await engine.health()
        except Exception:
            tts_ok = False
    try:
        redis_ok = bool(await state.redis.ping())
    except Exception:
        redis_ok = False
    ready = redis_ok and tts_ok
    return {"ready": ready, "redis": redis_ok, "tts": tts_ok}


@router.get("/v1/voices")
async def voices(request: Request, api_key: str = Depends(require_api_key)) -> dict:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {"voices": [settings.tts_voice]}
    try:
        return {"voices": await engine.list_voices()}
    except Exception as exc:
        log.warning("api.voices_failed", error=str(exc))
        return {"voices": [settings.tts_voice]}


@router.get("/metrics")
async def metrics(request: Request) -> PlainTextResponse:
    state = get_state(request)
    lines: list[str] = []
    counts = await state.counts_by_status()
    lines.append("# HELP narrator_jobs Jobs by status")
    lines.append("# TYPE narrator_jobs gauge")
    for status, n in counts.items():
        lines.append(f'narrator_jobs{{status="{status}"}} {n}')
    lines.append(f"narrator_jobs_active {await state.count_active()}")

    lines.append("# HELP narrator_breaker_state 0=closed 1=half_open 2=open")
    lines.append("# TYPE narrator_breaker_state gauge")
    for name in ("tts", "gemini", "drive"):
        br = CircuitBreaker(state.redis, name)
        lines.append(f'narrator_breaker_state{{dependency="{name}"}} {await br.state_int()}')

    fast = await state.redis.llen("arq:queue:narrator:fast")
    bulk = await state.redis.llen("arq:queue:narrator:bulk")
    lines.append(f"narrator_queue_depth{{lane=\"fast\"}} {int(fast or 0)}")
    lines.append(f"narrator_queue_depth{{lane=\"bulk\"}} {int(bulk or 0)}")
    return PlainTextResponse("\n".join(lines) + "\n")
