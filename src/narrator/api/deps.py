"""API dependencies: auth, rate limiting, shared resource accessors (section 7)."""

from __future__ import annotations

import time

from fastapi import Header, HTTPException, Request

from narrator.core.config import settings
from narrator.core.state import StateStore


def get_state(request: Request) -> StateStore:
    state = getattr(request.app.state, "state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="state not initialized")
    return state


async def enqueue(request: Request, job_id: str, lane: str) -> None:
    fn = getattr(request.app.state, "enqueue", None)
    if fn is None:
        raise HTTPException(status_code=503, detail="queue not initialized")
    await fn(job_id, lane)


async def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if not x_api_key or x_api_key not in settings.api_key_set:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return x_api_key


async def rate_limit(
    request: Request,
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> str:
    api_key = await require_api_key(x_api_key)
    state = get_state(request)
    window = 60
    bucket = int(time.time() // window)
    key = f"ratelimit:{api_key}:{bucket}"
    count = await state.redis.incr(key)
    if int(count) == 1:
        await state.redis.expire(key, window)
    if int(count) > settings.rate_limit_per_min:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return api_key
