"""Content-addressed NVMe store + Redis metadata (DB 1, prefixed, TTL'd)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from narration.keys import article_key, cache_key as redis_cache_key
from narration.logging import get_logger

log = get_logger("narration.store")

STATUS_QUEUED = "queued"
STATUS_GENERATING = "generating"
STATUS_READY = "ready"
STATUS_FALLBACK = "fallback"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"


class Store:
    def __init__(self, redis: Any, *, prefix: str, data_dir: str, ttl_s: int) -> None:
        self._r = redis
        self.prefix = prefix
        self.ttl = ttl_s
        self.root = Path(data_dir)
        self.audio = self.root / "audio"
        self.scripts = self.root / "scripts"
        self.tmp = self.root / "tmp"
        for p in (self.audio, self.scripts, self.tmp):
            p.mkdir(parents=True, exist_ok=True)

    def shard(self, cache: str) -> str:
        return cache[:2]

    def chunk_opus_path(self, cache: str, index: int) -> Path:
        d = self.audio / self.shard(cache)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{cache}.{index}.opus"

    def opus_path(self, cache: str, *, hd: bool = False) -> Path:
        name = f"{cache}.hd.opus" if hd else f"{cache}.opus"
        return self.audio / self.shard(cache) / name

    def meta_path(self, cache: str) -> Path:
        return self.audio / self.shard(cache) / f"{cache}.json"

    def script_path(self, cache: str) -> Path:
        return self.scripts / self.shard(cache) / f"{cache}.txt"

    def tmp_wav(self, cache: str, idx: int) -> Path:
        d = self.tmp / self.shard(cache)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{cache}.{idx:04d}.wav"

    async def get_cache(self, cache: str) -> dict[str, Any] | None:
        raw = await self._r.get(redis_cache_key(self.prefix, cache))
        if not raw:
            path = self.meta_path(cache)
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return None
        data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        opus = Path(data.get("file_path") or self.opus_path(cache))
        if not opus.exists():
            return None
        return data

    async def put_cache(self, cache: str, record: dict[str, Any]) -> None:
        blob = json.dumps(record, ensure_ascii=False)
        path = self.meta_path(cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(blob, encoding="utf-8")
        os.replace(tmp, path)
        await self._r.set(redis_cache_key(self.prefix, cache), blob, ex=self.ttl)

    async def bind_article(self, article_id: str, record: dict[str, Any]) -> None:
        blob = json.dumps(record, ensure_ascii=False)
        await self._r.set(article_key(self.prefix, article_id), blob, ex=self.ttl)

    async def get_article(self, article_id: str) -> dict[str, Any] | None:
        raw = await self._r.get(article_key(self.prefix, article_id))
        if not raw:
            return None
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    def write_script(self, cache: str, script: str) -> Path:
        path = self.script_path(cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(script, encoding="utf-8")
        os.replace(tmp, path)
        return path

    def iter_expired_meta(self, *, max_age_s: int, now: float) -> list[str]:
        out: list[str] = []
        if not self.audio.exists():
            return out
        for meta in self.audio.glob("*/*.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            created = float(data.get("created_at") or 0)
            if created and (now - created) >= max_age_s:
                out.append(str(data.get("cache_key") or meta.stem))
        return out

    def iter_stale_tmp(self, *, max_age_s: int, now: float) -> list[Path]:
        """Tmp wavs left behind if a job was killed after Clear All."""
        out: list[Path] = []
        if not self.tmp.exists():
            return out
        for p in self.tmp.glob("*/*"):
            if not p.is_file():
                continue
            try:
                age = now - p.stat().st_mtime
            except OSError:
                continue
            if age >= max_age_s:
                out.append(p)
        return out
