"""Job routes (ARCHITECTURE.md section 7)."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from narrator.api.deps import enqueue, get_state, rate_limit, require_api_key
from narrator.core.config import settings
from narrator.core.logging import get_logger
from narrator.core.models import Job, JobParams, JobStatus
from narrator.core.state import StateStore
from narrator.worker.tasks import choose_lane, estimate_minutes

log = get_logger(__name__)
router = APIRouter()

_EXT_KINDS = {"pdf": "pdf", "txt": "txt", "md": "md"}


def _new_job_id() -> str:
    return "jb_" + uuid.uuid4().hex[:16]


def _write_input(job_id: str, data: bytes, source_kind: str) -> Path:
    job_dir = settings.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = "pdf" if source_kind == "pdf" else ("md" if source_kind == "md" else "txt")
    path = job_dir / f"input.{ext}"
    path.write_bytes(data)
    return path


async def _parse_request(request: Request) -> tuple[bytes, str, JobParams, int]:
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" in ctype:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=422, detail="multipart request requires a 'file' field")
        data = await upload.read()
        ext = Path(upload.filename or "").suffix.lower().lstrip(".")
        source_kind = _EXT_KINDS.get(ext, "txt")
        params_dict = {k: v for k, v in form.items() if k != "file"}
        size = len(data)
    else:
        body = await request.json()
        text = body.get("text") or ""
        if not text.strip():
            raise HTTPException(status_code=422, detail="'text' is required for JSON requests")
        data = text.encode("utf-8")
        source_kind = "text"
        params_dict = {k: v for k, v in body.items() if k != "text"}
        size = len(text)
    try:
        params = JobParams(**params_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
    return data, source_kind, params, size


@router.post("/v1/jobs")
async def create_job(request: Request, api_key: str = Depends(rate_limit)) -> JSONResponse:
    state = get_state(request)
    data, source_kind, params, size = await _parse_request(request)

    if size > settings.max_input_chars:
        raise HTTPException(status_code=413, detail=f"input exceeds MAX_INPUT_CHARS ({settings.max_input_chars})")

    idem = request.headers.get("Idempotency-Key")
    job_id = _new_job_id()
    if idem:
        existing = await state.idempotency_setnx(idem, job_id)
        if existing:
            job = await state.get_job(existing)
            if job is not None:
                return JSONResponse(
                    {"job_id": job.id, "status": job.status.value, "lane": job.lane, "idempotent": True}
                )

    if await state.count_active() >= settings.max_active_jobs:
        raise HTTPException(status_code=429, detail="MAX_ACTIVE_JOBS reached; try again later")

    minutes = estimate_minutes(size, params.mode, explainer_target=settings.explainer_target_minutes)
    lane = choose_lane(minutes, settings.fast_lane_max_minutes)

    input_path = await asyncio.to_thread(_write_input, job_id, data, source_kind)
    job = Job(
        id=job_id,
        lane=lane,
        params=params,
        input_path=str(input_path),
        source_kind=source_kind,
        char_count=size if source_kind == "text" else 0,
    )
    await state.create_job(job)
    await enqueue(request, job_id, lane)
    log.info("api.job_created", job_id=job_id, lane=lane, mode=params.mode)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": job.status.value, "lane": lane},
    )


@router.get("/v1/jobs")
async def list_jobs(request: Request, limit: int = 50, offset: int = 0, api_key: str = Depends(require_api_key)) -> dict:
    state = get_state(request)
    jobs = await state.list_jobs(limit=min(limit, 200), offset=offset)
    return {"jobs": [_job_public(j) for j in jobs]}


@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request, api_key: str = Depends(require_api_key)) -> dict:
    job = await _require_job(request, job_id)
    return _job_public(job)


@router.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, api_key: str = Depends(require_api_key)) -> JSONResponse:
    state = get_state(request)
    job = await _require_job(request, job_id)
    if job.status.is_terminal:
        raise HTTPException(status_code=409, detail=f"job already {job.status.value}")
    await state.request_cancel(job_id)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "cancelling"})


@router.post("/v1/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request, api_key: str = Depends(require_api_key)) -> dict:
    state = get_state(request)
    job = await _require_job(request, job_id)
    if job.status not in (JobStatus.FAILED, JobStatus.UPLOAD_PENDING):
        raise HTTPException(status_code=409, detail=f"cannot retry a {job.status.value} job")
    job.error = None
    job.stall_reason = None
    await state.clear_cancel(job_id)
    await state.transition(job, JobStatus.QUEUED, stage="queued")
    await enqueue(request, job_id, job.lane)
    return {"job_id": job_id, "status": job.status.value}


@router.get("/v1/jobs/{job_id}/download")
async def download_job(job_id: str, request: Request, api_key: str = Depends(require_api_key)):
    job = await _require_job(request, job_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail=f"no artifact yet (status {job.status.value})")
    if job.result.backend == "gdrive" and job.result.web_view_link:
        return JSONResponse({"url": job.result.web_view_link, "drive_file_id": job.result.drive_file_id})
    local = job.result.local_path
    if local and await asyncio.to_thread(os.path.exists, local):
        return FileResponse(local, filename=job.result.filename or Path(local).name)
    raise HTTPException(status_code=404, detail="artifact not found on disk")


@router.get("/v1/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, api_key: str = Depends(require_api_key)):
    state = get_state(request)
    job = await _require_job(request, job_id)

    async def generator():
        yield {"event": "snapshot", "data": json.dumps(_job_public(job))}
        pubsub = state.redis.pubsub()
        await pubsub.subscribe(f"job:{job_id}:events")
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    yield {"event": "heartbeat", "data": "{}"}
                    continue
                data = msg["data"]
                text = data.decode() if isinstance(data, bytes) else str(data)
                yield {"event": "update", "data": text}
                try:
                    if JobStatus(json.loads(text).get("status")).is_terminal:
                        break
                except (ValueError, KeyError, json.JSONDecodeError):
                    pass
        finally:
            await pubsub.unsubscribe(f"job:{job_id}:events")
            await pubsub.aclose()

    return EventSourceResponse(generator())


# --- helpers -----------------------------------------------------------------
async def _require_job(request: Request, job_id: str) -> Job:
    state: StateStore = get_state(request)
    job = await state.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _job_public(job: Job) -> dict:
    return {
        "job_id": job.id,
        "status": job.status.value,
        "lane": job.lane,
        "mode": job.params.mode,
        "progress": job.progress,
        "done_chunks": job.done_chunks,
        "total_chunks": job.total_chunks,
        "stage": job.stage,
        "stall_reason": job.stall_reason,
        "eta_seconds": job.eta_seconds,
        "error": job.error,
        "warnings": job.warnings,
        "result": job.result.model_dump() if job.result else None,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
