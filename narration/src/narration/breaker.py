"""Circuit breaker: 3 consecutive failures across *different* articles → cooldown.

Open routes new work to on-device fallback instead of hammering a sick service.
One Telegram alert on trip, one on reset.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from narration.keys import breaker_key
from narration.logging import get_logger

log = get_logger("narration.breaker")

CLOSED = "closed"
OPEN = "open"


class PipelineBreaker:
    def __init__(
        self,
        redis: Any,
        prefix: str,
        name: str,
        *,
        threshold: int = 3,
        cooldown_s: int = 900,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._r = redis
        self._prefix = prefix
        self.name = name
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._now = clock

    def _k(self, suffix: str) -> str:
        return breaker_key(self._prefix, self.name, suffix)

    async def is_open(self) -> bool:
        state = await self._r.get(self._k("state"))
        if not state:
            return False
        val = state.decode() if isinstance(state, bytes) else str(state)
        if val != OPEN:
            return False
        opened = await self._r.get(self._k("opened_at"))
        if not opened:
            return True
        opened_s = float(opened.decode() if isinstance(opened, bytes) else opened)
        if self._now() - opened_s >= self._cooldown:
            await self.reset()
            return False
        return True

    async def record_success(self) -> bool:
        """Returns True if this call closed a previously-open breaker."""
        was_open = await self.is_open()
        await self._r.delete(self._k("fail_articles"), self._k("state"), self._k("opened_at"))
        if was_open:
            log.info("breaker.reset", breaker=self.name)
            return True
        return False

    async def record_failure(self, article_id: str) -> bool:
        """Returns True if this call newly opened the breaker."""
        if await self.is_open():
            return False
        await self._r.sadd(self._k("fail_articles"), article_id)
        await self._r.expire(self._k("fail_articles"), self._cooldown * 2)
        n = int(await self._r.scard(self._k("fail_articles")))
        if n >= self._threshold:
            await self._r.set(self._k("state"), OPEN)
            await self._r.set(self._k("opened_at"), str(self._now()))
            log.warning("breaker.open", breaker=self.name, articles=n)
            return True
        return False

    async def reset(self) -> None:
        await self._r.delete(self._k("fail_articles"), self._k("state"), self._k("opened_at"))
