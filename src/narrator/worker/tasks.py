"""Job pipeline + lane routing + webhooks + cron (ARCHITECTURE.md sections 2-3, 15).

``JobRunner.run`` drives one job through the full state machine, writing local
first and surfacing dependency outages as stalls (not failures). Terminal
delivery errors route to UPLOAD_PENDING with a downloadable local artifact.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import shutil
import time
from pathlib import Path

import httpx

from narrator.core.errors import (
    BreakerOpen,
    DeliveryTerminal,
    IngestError,
    JobCancelled,
    JobFailedResumable,
)
from narrator.core.logging import bind_job, clear_context, get_logger
from narrator.core.models import Document, Job, JobResult, JobStatus
from narrator.core.protocols import JobContext
from narrator.pipeline import ingest as ingest_mod
from narrator.pipeline.assemble import Assembler
from narrator.pipeline.chunker import SentenceChunker
from narrator.pipeline.synth import Synthesizer

log = get_logger(__name__)

_VERBATIM_CPM = 850  # chars per minute of speech (~155 wpm)
_MIME = {"mp3": "audio/mpeg", "opus": "audio/ogg"}


# --- lane routing ------------------------------------------------------------
def estimate_minutes(char_count: int, mode: str, *, explainer_target: int) -> float:
    if mode == "explainer":
        return float(explainer_target)
    return max(0.1, char_count / _VERBATIM_CPM)


def choose_lane(minutes: float, fast_lane_max_minutes: float) -> str:
    return "fast" if minutes <= fast_lane_max_minutes else "bulk"


# --- webhooks ----------------------------------------------------------------
def sign_payload(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


async def send_webhook(url: str, secret: str | None, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Narrator-Signature"] = f"sha256={sign_payload(secret, body)}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, content=body, headers=headers)
    except httpx.HTTPError as exc:
        log.warning("webhook.failed", url=url, error=str(exc))


def _safe_filename(title: str, ext: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "narration"
    return f"{base[:80]}.{ext}"


class JobRunner:
    def __init__(
        self,
        *,
        state,
        engine,
        cache,
        gate,
        storage,
        assembler: Assembler,
        jobs_dir: Path,
        gemini_client=None,
        gemini_breaker=None,
        synth_concurrency: int = 2,
        non_english_policy: str = "skip_warn",
        max_failed_pct: float = 1.0,
        webhook_secret: str | None = None,
        model: str = "kokoro-int8",
    ) -> None:
        self._state = state
        self._engine = engine
        self._cache = cache
        self._gate = gate
        self._storage = storage
        self._assembler = assembler
        self._jobs_dir = Path(jobs_dir)
        self._gemini_client = gemini_client
        self._gemini_breaker = gemini_breaker
        self._synth_concurrency = synth_concurrency
        self._policy = non_english_policy
        self._max_failed_pct = max_failed_pct
        self._webhook_secret = webhook_secret
        self._model = model
        self._chunker = SentenceChunker()

    # --- helpers -------------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return self._jobs_dir / job_id

    def _ingest(self, job: Job) -> Document:
        path = job.input_path
        if job.source_kind == "pdf":
            return ingest_mod.ingest_pdf_bytes(Path(path).read_bytes(), title=job.params.title)
        if job.source_kind in ("txt", "md"):
            return ingest_mod.ingest_file(path, title=job.params.title)
        text = Path(path).read_text(encoding="utf-8") if path else ""
        return ingest_mod.ingest_text(text, title=job.params.title)

    def _build_processor(self, mode: str):
        if mode == "verbatim":
            from narrator.processors.verbatim import VerbatimProcessor

            return VerbatimProcessor()
        from narrator.processors.explainer import GeminiExplainerProcessor

        if self._gemini_client is None or self._gemini_breaker is None:
            raise IngestError("explainer mode requires GEMINI_API_KEY", status_code=422)
        return GeminiExplainerProcessor(self._gemini_client, self._gemini_breaker)

    async def _notify(self, job: Job) -> None:
        if not job.params.webhook_url:
            return
        payload = {
            "job_id": job.id,
            "status": job.status.value,
            "result": job.result.model_dump() if job.result else None,
            "warnings": job.warnings,
            "ts": time.time(),
        }
        await send_webhook(job.params.webhook_url, self._webhook_secret, payload)

    # --- main pipeline -------------------------------------------------------
    async def run(self, job_id: str) -> None:
        job = await self._state.get_job(job_id)
        if job is None:
            log.warning("job.missing", job_id=job_id)
            return
        bind_job(job_id, lane=job.lane)
        try:
            await self._run_inner(job)
        finally:
            clear_context()

    async def _run_inner(self, job: Job) -> None:
        job_dir = self._job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        if await self._state.is_cancelled(job.id):
            await self._finish(job, JobStatus.CANCELLED)
            return
        try:
            await self._state.transition(job, JobStatus.PREPROCESSING, stage="ingest")
            doc = self._ingest(job)
            job.char_count = doc.char_count

            ctx = JobContext(job_id=job.id, params=job.params)
            processor = self._build_processor(job.params.mode)
            script = await processor.process(doc, ctx)
            job.warnings = list(ctx.warnings)
            chunks = self._chunker.split(script)
            if not chunks:
                raise IngestError("no narratable content produced", status_code=422)

            await self._state.transition(job, JobStatus.SYNTHESIZING, stage="synthesizing")
            synth = Synthesizer(
                engine=self._engine,
                cache=self._cache,
                gate=self._gate,
                state=self._state,
                model=self._model,
                non_english_policy=self._policy,
                concurrency=self._synth_concurrency,
                max_failed_pct=self._max_failed_pct,
                lane=job.lane,
            )
            manifest = await synth.run(job, chunks, job_dir)

            await self._state.transition(job, JobStatus.ASSEMBLING, stage="assembling")
            title = job.params.title or doc.title or job.id
            fmt = job.params.output_format
            result = await self._assembler.assemble(
                chunks,
                manifest,
                job_dir,
                title=title,
                output_format=fmt,
                cancel_check=lambda: self._state.is_cancelled(job.id),
            )

            await self._state.transition(job, JobStatus.UPLOADING, stage="delivering")
            filename = _safe_filename(title, fmt)
            try:
                stored = await self._storage.upload(
                    str(result.path),
                    filename=filename,
                    mime_type=_MIME.get(fmt, "application/octet-stream"),
                    meta={"drive_folder_id": job.params.drive_folder_id},
                )
            except DeliveryTerminal as exc:
                job.result = JobResult(
                    local_path=str(result.path),
                    filename=filename,
                    duration_seconds=result.duration_seconds,
                    size_bytes=result.size_bytes,
                    backend="local",
                )
                await self._finish(job, JobStatus.UPLOAD_PENDING, stall_reason=str(exc))
                return

            job.result = JobResult(
                drive_file_id=stored.drive_file_id,
                web_view_link=stored.web_view_link,
                local_path=stored.local_path or str(result.path),
                filename=filename,
                duration_seconds=result.duration_seconds,
                size_bytes=stored.size_bytes or result.size_bytes,
                backend=stored.backend,
            )
            await self._finish(job, JobStatus.COMPLETED, stage="done")

        except JobCancelled:
            await self._finish(job, JobStatus.CANCELLED)
        except BreakerOpen as exc:
            # Dependency down: keep the job live + stalled and let arq retry.
            await self._state.update_progress(job, stall_reason=str(exc))
            log.warning("job.stalled", job_id=job.id, reason=str(exc))
            raise
        except (IngestError, JobFailedResumable) as exc:
            await self._finish(job, JobStatus.FAILED, error=str(exc))
        except Exception as exc:  # unexpected: fail loudly but recorded
            log.exception("job.failed_unexpected", job_id=job.id)
            await self._finish(job, JobStatus.FAILED, error=f"unexpected: {exc}")

    async def _finish(
        self,
        job: Job,
        status: JobStatus,
        *,
        stage: str | None = None,
        stall_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        await self._state.transition(job, status, stage=stage, stall_reason=stall_reason, error=error)
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.UPLOAD_PENDING, JobStatus.CANCELLED):
            await self._state.clear_cancel(job.id)
            await self._notify(job)


# --- cron helpers ------------------------------------------------------------
def _rmtree_all(dirs: list[Path]) -> None:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


async def cleanup_expired(state, retention_days: int, jobs_dir: Path, outputs_dir: Path) -> int:
    cutoff = time.time() - retention_days * 86400
    stale = [
        jobs_dir / job.id
        for job in await state.list_jobs(limit=100000)
        if job.updated_at < cutoff and job.status.is_terminal
    ]
    if stale:
        await asyncio.to_thread(_rmtree_all, stale)
    return len(stale)


def _copy_ledger(ledger_path: Path, backup_dir: Path, dest: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        shutil.copyfile(ledger_path, dest)


async def backup_ledger(ledger_path: Path, backup_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d")
    dest = backup_dir / f"ledger-{stamp}.db"
    await asyncio.to_thread(_copy_ledger, ledger_path, backup_dir, dest)
    return dest
