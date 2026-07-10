"""Redis-backed circuit breaker (ARCHITECTURE.md section 14).

Shared across workers so a down dependency pauses work everywhere instead of
burning retries. States: closed -> open (after N consecutive retry-exhausted
failures) -> half-open (single probe after cooldown) -> closed. Cooldown starts
at 60 s and doubles on repeated probe failure up to a 10 min cap.

Only ``RetryableError`` (and transport) failures count - terminal errors pass
through untouched. An open breaker raises ``BreakerOpen`` which the pipeline
surfaces as ``stall_reason`` (never as a job failure).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from narrator.core.errors import BreakerOpen, RetryableError
from narrator.core.logging import get_logger
from narrator.core.retry import _TRANSPORT_RETRYABLE

log = get_logger(__name__)

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

STATE_TO_INT = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}


class CircuitBreaker:
    def __init__(
        self,
        redis: Any,
        name: str,
        *,
        threshold: int = 5,
        base_cooldown: float = 60.0,
        max_cooldown: float = 600.0,
        count_exceptions: tuple[type[BaseException], ...] = (RetryableError, *_TRANSPORT_RETRYABLE),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._r = redis
        self.name = name
        self._threshold = threshold
        self._base = base_cooldown
        self._max = max_cooldown
        self._count = count_exceptions
        self._now = clock

    # --- key helpers ---------------------------------------------------------
    def _k(self, suffix: str) -> str:
        return f"breaker:{self.name}:{suffix}"

    async def _get(self, suffix: str, default: str | None = None) -> str | None:
        v = await self._r.get(self._k(suffix))
        if v is None:
            return default
        return v.decode() if isinstance(v, bytes) else str(v)

    async def get_state(self) -> str:
        return await self._get("state", CLOSED) or CLOSED

    async def state_int(self) -> int:
        return STATE_TO_INT.get(await self.get_state(), 0)

    async def _cooldown(self) -> float:
        raw = await self._get("cooldown")
        return float(raw) if raw else self._base

    # --- transitions ---------------------------------------------------------
    async def _open(self, cooldown: float) -> None:
        await self._r.set(self._k("state"), OPEN)
        await self._r.set(self._k("opened_at"), str(self._now()))
        await self._r.set(self._k("cooldown"), str(cooldown))
        log.warning("breaker.open", breaker=self.name, cooldown=cooldown)

    async def _close(self) -> None:
        await self._r.set(self._k("state"), CLOSED)
        await self._r.set(self._k("failures"), "0")
        await self._r.set(self._k("cooldown"), str(self._base))
        await self._r.delete(self._k("probe"))
        log.info("breaker.close", breaker=self.name)

    async def _enter(self) -> None:
        state = await self.get_state()
        if state == CLOSED:
            return
        if state == OPEN:
            opened_at = float(await self._get("opened_at", "0") or 0)
            cooldown = await self._cooldown()
            if self._now() - opened_at >= cooldown:
                # Cooldown elapsed: exactly one caller becomes the half-open probe.
                got = await self._r.set(self._k("probe"), "1", nx=True, ex=int(cooldown) + 30)
                if got:
                    await self._r.set(self._k("state"), HALF_OPEN)
                    log.info("breaker.half_open", breaker=self.name)
                    return
            raise BreakerOpen(self.name)
        # HALF_OPEN: a probe is already in flight - block everyone else.
        raise BreakerOpen(self.name)

    async def on_success(self) -> None:
        await self._close()

    async def on_failure(self) -> None:
        state = await self.get_state()
        if state == HALF_OPEN:
            cooldown = min(await self._cooldown() * 2, self._max)
            await self._r.set(self._k("failures"), str(self._threshold))
            await self._open(cooldown)
            await self._r.delete(self._k("probe"))
            return
        failures = await self._r.incr(self._k("failures"))
        if int(failures) >= self._threshold:
            await self._open(self._base)

    @asynccontextmanager
    async def guard(self):
        """Wrap a (already-retried) external call. Trips only on counted errors."""
        await self._enter()
        try:
            yield
        except self._count:
            await self.on_failure()
            raise
        except BaseException:
            # Terminal / unexpected errors do not reflect dependency health.
            raise
        else:
            await self.on_success()
