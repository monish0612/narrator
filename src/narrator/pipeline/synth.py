"""Synthesis stage (ARCHITECTURE.md section 9).

Per chunk: sanitize -> cache -> (bulk lane) gate turn -> engine -> persist wav ->
cache put -> atomic manifest -> progress event. Fully resumable: a manifest
tracks every chunk; on rerun only pending/invalid chunks are synthesized. One
poisoned chunk never kills a job (silence placeholder + warning), governed by
MAX_FAILED_CHUNK_PCT.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from narrator.core.cache import TTSCache, cache_key
from narrator.core.errors import (
    BreakerOpen,
    JobCancelled,
    JobFailedResumable,
    RetryableError,
    SkipChunk,
    UnspeakableError,
)
from narrator.core.gate import YIELD_REASON, SynthGate
from narrator.core.logging import get_logger
from narrator.core.models import Chunk, Job, Manifest, ManifestChunk, ManifestEngine
from narrator.core.retry import _TRANSPORT_RETRYABLE
from narrator.core.state import StateStore
from narrator.pipeline.sanitize import sanitize_chunk
from narrator.pipeline.wavtools import is_valid_wav, read_wav_duration_ms, write_silence_wav

log = get_logger(__name__)

MAX_CHUNK_ATTEMPTS = 6
SILENCE_SKIP_MS = 200
SILENCE_FAIL_MS = 300
_RETRY_CONTINUE = (RetryableError, *_TRANSPORT_RETRYABLE)
_NON_SPEAKABLE = re.compile(r"[^A-Za-z0-9\s.,!?;:'\"-]")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _aggressive_clean(text: str) -> str:
    return re.sub(r"\s+", " ", _NON_SPEAKABLE.sub(" ", text)).strip()


def _persist_wav(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _write_manifest(path: Path, manifest: Manifest) -> None:
    manifest.updated_at = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(manifest.model_dump_json(), encoding="utf-8")
    os.replace(tmp, path)


class Synthesizer:
    def __init__(
        self,
        *,
        engine,
        cache: TTSCache,
        gate: SynthGate,
        state: StateStore,
        model: str = "kokoro-int8",
        non_english_policy: str = "skip_warn",
        concurrency: int = 2,
        max_failed_pct: float = 1.0,
        lane: str = "fast",
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._gate = gate
        self._state = state
        self._model = model
        self._policy = non_english_policy
        self._concurrency = max(1, concurrency)
        self._max_failed_pct = max_failed_pct
        self._lane = lane
        self._voice = "af_heart"
        self._speed = 1.0

    # --- manifest ------------------------------------------------------------
    def load_or_create_manifest(self, job: Job, chunks: list[Chunk], job_dir: Path) -> Manifest:
        engine = ManifestEngine(
            base_url=getattr(self._engine, "_base", ""),
            model=self._model,
            voice=job.params.voice,
            speed=job.params.speed,
        )
        chunks_dir = job_dir / "chunks"
        old_by_idx: dict[int, ManifestChunk] = {}
        mpath = job_dir / "manifest.json"
        if mpath.exists():
            try:
                old = Manifest.model_validate_json(mpath.read_text(encoding="utf-8"))
                old_by_idx = {c.idx: c for c in old.chunks}
            except Exception as exc:
                log.warning("synth.manifest_reload_failed", error=str(exc))

        entries: list[ManifestChunk] = []
        for chunk in chunks:
            prev = old_by_idx.get(chunk.idx)
            reusable = (
                prev is not None
                and prev.sha256 == chunk.sha256
                and prev.status == "done"
                and is_valid_wav(chunks_dir / f"{chunk.idx:06d}.wav")
            )
            if reusable:
                entries.append(prev)
            else:
                entries.append(
                    ManifestChunk(idx=chunk.idx, sha256=chunk.sha256, chars=len(chunk.text))
                )
        manifest = Manifest(
            job_id=job.id,
            engine=engine,
            chunks=entries,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        return manifest

    # --- run -----------------------------------------------------------------
    async def run(self, job: Job, chunks: list[Chunk], job_dir: Path) -> Manifest:
        self._voice = job.params.voice
        self._speed = job.params.speed
        job_dir = Path(job_dir)
        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        mpath = job_dir / "manifest.json"

        manifest = self.load_or_create_manifest(job, chunks, job_dir)
        await asyncio.to_thread(_write_manifest, mpath, manifest)

        by_idx = {c.idx: c for c in chunks}
        pending = [e for e in manifest.chunks if e.status != "done"]

        chunk_times: deque[float] = deque(maxlen=50)
        manifest_lock = asyncio.Lock()
        dispatch_lock = asyncio.Lock()
        pending_iter = iter(pending)
        cancelled = False
        stalled_flag = {"on": False}

        async def _on_stall() -> None:
            stalled_flag["on"] = True
            await self._state.update_progress(job, stall_reason=YIELD_REASON)

        async def _on_resume() -> None:
            if stalled_flag["on"]:
                stalled_flag["on"] = False
                await self._state.update_progress(job, stall_reason=None)

        async def worker_loop() -> None:
            nonlocal cancelled
            while True:
                async with dispatch_lock:
                    if cancelled:
                        return
                    try:
                        entry = next(pending_iter)
                    except StopIteration:
                        return
                    if await self._state.is_cancelled(job.id):
                        cancelled = True
                        return
                    if self._lane == "bulk":
                        await self._gate.bulk_wait_turn(on_stall=_on_stall, on_resume=_on_resume)
                await self._synth_one(by_idx[entry.idx], entry, chunks_dir, chunk_times)
                async with manifest_lock:
                    await asyncio.to_thread(_write_manifest, mpath, manifest)
                    await self._publish_progress(job, manifest, chunk_times)

        workers = [asyncio.create_task(worker_loop()) for _ in range(self._concurrency)]
        await asyncio.gather(*workers)

        await asyncio.to_thread(_write_manifest, mpath, manifest)
        if cancelled:
            raise JobCancelled(job.id)

        total = len(manifest.chunks)
        failed = manifest.failed_count
        if total and (100.0 * failed / total) > self._max_failed_pct:
            raise JobFailedResumable(
                f"{failed}/{total} chunks failed (> {self._max_failed_pct}%)"
            )
        return manifest

    async def _publish_progress(self, job: Job, manifest: Manifest, chunk_times: deque[float]) -> None:
        done = manifest.done_count
        total = len(manifest.chunks)
        remaining = max(0, total - done)
        eta = None
        if chunk_times and remaining:
            avg = sum(chunk_times) / len(chunk_times)
            eta = round(avg * remaining, 1)
        await self._state.update_progress(
            job, done_chunks=done, total_chunks=total, eta_seconds=eta, stall_reason=None
        )

    # --- per chunk -----------------------------------------------------------
    async def _synth_one(
        self,
        chunk: Chunk,
        entry: ManifestChunk,
        chunks_dir: Path,
        chunk_times: deque[float],
    ) -> None:
        chunk_path = chunks_dir / f"{chunk.idx:06d}.wav"
        rel = f"chunks/{chunk.idx:06d}.wav"
        voice = self._voice
        speed = self._speed

        try:
            cleaned = sanitize_chunk(chunk.text, non_english_policy=self._policy)
        except SkipChunk as exc:
            await asyncio.to_thread(write_silence_wav, chunk_path, SILENCE_SKIP_MS)
            entry.status = "done"
            entry.file = rel
            entry.ms = SILENCE_SKIP_MS
            log.info("synth.chunk_skipped", chunk_idx=chunk.idx, reason=exc.args[0])
            return

        key = cache_key(model=self._model, voice=voice, speed=speed, sanitized_text=cleaned)
        if await self._cache.get(key, chunk_path):
            entry.status = "done"
            entry.cache_hit = True
            entry.file = rel
            entry.ms = read_wav_duration_ms(chunk_path)
            return

        last: Exception | None = None
        for _ in range(MAX_CHUNK_ATTEMPTS):
            entry.attempts += 1
            try:
                t0 = time.perf_counter()
                result = await self._engine.synthesize(cleaned, voice=voice, speed=speed)
            except UnspeakableError:
                if await self._try_aggressive(chunk, entry, chunk_path, rel, key, voice, speed):
                    return
                await asyncio.to_thread(write_silence_wav, chunk_path, SILENCE_FAIL_MS)
                entry.status = "failed"
                entry.file = rel
                entry.ms = SILENCE_FAIL_MS
                log.warning("synth.chunk_unspeakable", chunk_idx=chunk.idx)
                return
            except BreakerOpen:
                raise
            except _RETRY_CONTINUE as exc:
                last = exc
                continue
            else:
                await asyncio.to_thread(_persist_wav, chunk_path, result.audio)
                await self._cache.put(key, chunk_path)
                entry.status = "done"
                entry.file = rel
                entry.ms = result.duration_ms or read_wav_duration_ms(chunk_path)
                chunk_times.append(time.perf_counter() - t0)
                return

        await asyncio.to_thread(write_silence_wav, chunk_path, SILENCE_FAIL_MS)
        entry.status = "failed"
        entry.file = rel
        entry.ms = SILENCE_FAIL_MS
        log.warning("synth.chunk_failed", chunk_idx=chunk.idx, attempts=entry.attempts, error=str(last))

    async def _try_aggressive(self, chunk, entry, chunk_path, rel, key, voice, speed) -> bool:
        aggressive = _aggressive_clean(chunk.text)
        if not aggressive:
            return False
        try:
            result = await self._engine.synthesize(aggressive, voice=voice, speed=speed)
        except Exception:
            return False
        await asyncio.to_thread(_persist_wav, chunk_path, result.audio)
        entry.status = "done"
        entry.file = rel
        entry.ms = result.duration_ms or read_wav_duration_ms(chunk_path)
        log.info("synth.chunk_recovered_aggressive", chunk_idx=chunk.idx)
        return True
