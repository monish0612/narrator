"""Deletion with retry-and-verify. Idempotent. Never silent on exhaustion."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from narration.keys import cache_key as redis_cache_key, delete_fail_key, lock_key
from narration.logging import get_logger
from narration.store import Store

log = get_logger("narration.delete")

Sleeper = Callable[[float], Awaitable[None]]


def _jittered(base: float) -> float:
    return base + random.uniform(0, base * 0.25)


async def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


async def delete_artifacts(
    store: Store,
    cache: str,
    *,
    attempts: int = 5,
    sleep: Sleeper | None = None,
    on_exhausted: Callable[[str], Awaitable[None]] | None = None,
) -> bool:
    """Delete opus + HD + script + tmp + redis. 'already gone' is success."""
    sleeper = sleep or asyncio.sleep
    lock = lock_key(store.prefix, f"del:{cache}")
    got = await store._r.set(lock, "1", nx=True, ex=30)
    if not got:
        await asyncio.sleep(0.05)
    last_err = ""
    paths = [
        store.opus_path(cache),
        store.opus_path(cache, hd=True),
        store.meta_path(cache),
        store.script_path(cache),
    ]
    tmp_dir = store.tmp / store.shard(cache)
    if tmp_dir.exists():
        paths.extend(tmp_dir.glob(f"{cache}.*"))

    for i in range(attempts):
        try:
            for p in paths:
                await _unlink(p)
            await store._r.delete(redis_cache_key(store.prefix, cache))
            leftover = [p for p in paths if p.exists()]
            if leftover:
                raise OSError(f"still present: {leftover[0]}")
            await store._r.delete(delete_fail_key(store.prefix, cache))
            log.info("delete.ok", cache_key=cache, attempt=i + 1)
            return True
        except Exception as exc:
            last_err = str(exc)
            log.warning("delete.retry", cache_key=cache, attempt=i + 1, error=last_err[:200])
            await sleeper(_jittered(0.2 * (2**i)))

    await store._r.set(delete_fail_key(store.prefix, cache), last_err[:500], ex=store.ttl)
    log.error("delete.exhausted", cache_key=cache, error=last_err[:200])
    if on_exhausted:
        await on_exhausted(last_err)
    return False
