"""Graceful shutdown (ARCHITECTURE.md sections 3 / 15-A).

On SIGTERM the worker stops accepting new chunks and lets the in-flight chunk
finish, checkpoints the manifest, and requeues the job (QUEUED). Coolify's ~30 s
stop grace is enough for one 60-90 s chunk boundary; resume is manifest-driven so
no synthesis is lost.
"""

from __future__ import annotations

import asyncio
import signal

from narrator.core.logging import get_logger

log = get_logger(__name__)


class GracefulShutdown:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def install(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._trigger, sig)
            except (NotImplementedError, RuntimeError):
                # Windows / no running loop: fall back to signal.signal.
                signal.signal(sig, lambda *_, s=sig: self._trigger(s))

    def _trigger(self, sig) -> None:
        log.info("shutdown.signal", signal=int(sig))
        self._event.set()

    @property
    def is_shutting_down(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def trigger(self) -> None:
        self._event.set()
