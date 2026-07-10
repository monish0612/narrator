"""Synthesis fairness gate (ARCHITECTURE.md section 8).

Exactly one synthesis stream should own the CPU at a time. The fast lane
registers intent via ``synth:fast_pending``; the bulk lane calls
``bulk_wait_turn()`` before every chunk and parks (emitting
``stall_reason:"yielding_to_fast_lane"``) while any fast job is pending, so a
fast job preempts a marathon within one chunk (~60-90 s).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from narrator.core.logging import get_logger

log = get_logger(__name__)

_FAST_PENDING = "synth:fast_pending"
YIELD_REASON = "yielding_to_fast_lane"


class SynthGate:
    def __init__(self, redis: Any, *, poll_interval: float = 1.0) -> None:
        self._r = redis
        self._poll = poll_interval

    async def fast_enter(self) -> None:
        await self._r.incr(_FAST_PENDING)

    async def fast_exit(self) -> None:
        # Never let the counter go negative.
        val = await self._r.decr(_FAST_PENDING)
        if int(val) < 0:
            await self._r.set(_FAST_PENDING, "0")

    async def fast_pending(self) -> int:
        raw = await self._r.get(_FAST_PENDING)
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @asynccontextmanager
    async def fast_lane(self):
        """Mark a fast job as pending/running for its whole lifetime."""
        await self.fast_enter()
        try:
            yield
        finally:
            await self.fast_exit()

    async def bulk_wait_turn(
        self,
        *,
        on_stall: Callable[[], Awaitable[None]] | None = None,
        on_resume: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Block while a fast job is pending. Returns True if it had to park."""
        stalled = False
        while await self.fast_pending() > 0:
            if not stalled:
                stalled = True
                log.info("gate.bulk_yield", reason=YIELD_REASON)
                if on_stall is not None:
                    await on_stall()
            await asyncio.sleep(self._poll)
        if stalled and on_resume is not None:
            await on_resume()
        return stalled
