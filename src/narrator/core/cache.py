"""Content-addressed TTS synthesis cache (ARCHITECTURE.md section 9).

key = sha256(engine_model | voice | speed | sanitized_text)
path = /data/cache/tts/{k[:2]}/{k}.wav

Cache reads hard-link into the job's chunk dir (never copy). The SQLite
``cache_index`` tracks size + last_used for hourly LRU eviction to CACHE_MAX_GB.
Making repeated/boilerplate/overlapping text free is a core speed lever.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from narrator.core.logging import get_logger
from narrator.core.state import StateStore

log = get_logger(__name__)


def cache_key(*, model: str, voice: str, speed: float, sanitized_text: str) -> str:
    payload = f"{model}|{voice}|{speed}|{sanitized_text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _link_or_copy(src: Path, dest: Path) -> None:
    """Hard-link src -> dest; fall back to copy across devices."""
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        import shutil

        shutil.copyfile(src, dest)


class TTSCache:
    def __init__(self, state: StateStore, cache_dir: Path, max_gb: float) -> None:
        self._state = state
        self._dir = Path(cache_dir)
        self._max_bytes = int(max_gb * (1024**3))

    def path_for(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.wav"

    @staticmethod
    def _looks_like_wav(path: Path, *, min_size: int = 1024) -> bool:
        try:
            if not path.exists() or path.stat().st_size < min_size:
                return False
            with open(path, "rb") as fh:
                header = fh.read(12)
            return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
        except OSError:
            return False

    async def get(self, key: str, dest: Path) -> bool:
        """If cached (and valid), hard-link into ``dest`` and touch LRU."""
        src = self.path_for(key)
        if not self._looks_like_wav(src):
            # Missing or corrupt (e.g. a stale/poisoned entry) -> treat as miss.
            if src.exists():
                await self._state.cache_delete(key)
            return False
        await asyncio.to_thread(_link_or_copy, src, dest)
        await self._state.cache_touch(key, src.stat().st_size)
        log.debug("cache.hit", key=key)
        return True

    async def put(self, key: str, src: Path) -> None:
        """Store a freshly synthesized wav in the cache (idempotent)."""
        dst = self.path_for(key)
        if dst.exists() and dst.stat().st_size > 0:
            await self._state.cache_touch(key, dst.stat().st_size)
            return
        await asyncio.to_thread(_link_or_copy, src, dst)
        size = dst.stat().st_size
        await self._state.cache_touch(key, size)
        log.debug("cache.put", key=key, size=size)

    async def evict_to_limit(self) -> int:
        """Evict least-recently-used entries until total <= CACHE_MAX_GB.

        Returns the number of entries evicted.
        """
        total = await self._state.cache_total_size()
        evicted = 0
        while total > self._max_bytes:
            rows = await self._state.cache_lru(limit=64)
            if not rows:
                break
            for row in rows:
                key = row["key"]
                size = int(row["size"])
                path = self.path_for(key)
                try:
                    if path.exists():
                        await asyncio.to_thread(path.unlink)
                except OSError as exc:
                    log.warning("cache.evict_failed", key=key, error=str(exc))
                await self._state.cache_delete(key)
                total -= size
                evicted += 1
                if total <= self._max_bytes:
                    break
        if evicted:
            log.info("cache.evicted", count=evicted, total_bytes=total)
        return evicted
