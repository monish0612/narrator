"""Domain models (ARCHITECTURE.md sections 5-7).

Pydantic models used across the pipeline, API and worker. ``JobParams`` is the
validated public request surface (section 7).
"""

from __future__ import annotations

import enum
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Mode = Literal["verbatim", "explainer"]
ExplainerStyle = Literal["news", "teacher", "podcast"]
OutputFormat = Literal["mp3", "opus"]
Lane = Literal["fast", "bulk"]


class JobStatus(enum.StrEnum):
    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    SYNTHESIZING = "SYNTHESIZING"
    ASSEMBLING = "ASSEMBLING"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    UPLOAD_PENDING = "UPLOAD_PENDING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self in _ACTIVE_STATES


_TERMINAL_STATES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.UPLOAD_PENDING}
)
# States that count against MAX_ACTIVE_JOBS / can be reconciled after a crash.
_ACTIVE_STATES = frozenset(
    {
        JobStatus.QUEUED,
        JobStatus.PREPROCESSING,
        JobStatus.SYNTHESIZING,
        JobStatus.ASSEMBLING,
        JobStatus.UPLOADING,
    }
)


class JobParams(BaseModel):
    """Public, validated job parameters (section 7)."""

    model_config = {"extra": "forbid"}

    mode: Mode = "explainer"
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    title: str | None = Field(None, max_length=300)
    output_format: OutputFormat = "mp3"
    explainer_style: ExplainerStyle = "news"
    target_language: str = "en"
    webhook_url: str | None = None
    drive_folder_id: str | None = None

    @field_validator("voice")
    @classmethod
    def _voice_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("voice must be non-empty")
        return v.strip()

    @field_validator("webhook_url")
    @classmethod
    def _webhook_shape(cls, v: str | None) -> str | None:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("webhook_url must be an http(s) URL")
        return v


class Document(BaseModel):
    """A normalized, ingested source document."""

    text: str
    char_count: int
    source_type: Literal["text", "txt", "md", "pdf"] = "text"
    title: str | None = None
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class Segment(BaseModel):
    """A unit of narration script - text plus a trailing pause hint."""

    text: str
    pause_ms_after: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class NarrationScript(BaseModel):
    """The processor output: ordered segments ready for chunking."""

    segments: list[Segment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


class Chunk(BaseModel):
    """A synthesis unit. ``sha256`` is computed over the *sanitized* text."""

    idx: int
    text: str
    sha256: str
    pause_ms_after: int = 0


class SynthesisResult(BaseModel):
    """Result of a single TTS synthesis call."""

    model_config = {"arbitrary_types_allowed": True}

    audio: bytes = b""
    duration_ms: int = 0
    sample_rate: int = 24000
    response_format: str = "wav"
    cache_hit: bool = False


class StoredFile(BaseModel):
    """A delivered artifact (local or Drive)."""

    backend: Literal["local", "gdrive"]
    filename: str
    size_bytes: int = 0
    duration_seconds: float = 0.0
    drive_file_id: str | None = None
    web_view_link: str | None = None
    local_path: str | None = None


class JobResult(BaseModel):
    drive_file_id: str | None = None
    web_view_link: str | None = None
    local_path: str | None = None
    filename: str | None = None
    duration_seconds: float = 0.0
    size_bytes: int = 0
    backend: str | None = None


ChunkStatus = Literal["pending", "done", "failed"]


class ManifestEngine(BaseModel):
    base_url: str
    model: str
    voice: str
    speed: float


class ManifestChunk(BaseModel):
    idx: int
    sha256: str
    chars: int
    status: ChunkStatus = "pending"
    file: str | None = None
    ms: int = 0
    attempts: int = 0
    cache_hit: bool = False


class Manifest(BaseModel):
    """Frozen resumability contract (ARCHITECTURE.md section 6)."""

    job_id: str
    engine: ManifestEngine
    chunks: list[ManifestChunk] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @property
    def done_count(self) -> int:
        return sum(1 for c in self.chunks if c.status == "done")

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.chunks if c.status == "failed")


class Job(BaseModel):
    """Durable job record (SQLite truth, mirrored to Redis)."""

    id: str
    status: JobStatus = JobStatus.QUEUED
    lane: Lane = "fast"
    params: JobParams = Field(default_factory=JobParams)

    input_path: str | None = None
    source_kind: str = "text"
    char_count: int = 0
    total_chunks: int = 0
    done_chunks: int = 0

    stage: str | None = None
    stall_reason: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    result: JobResult | None = None

    eta_seconds: float | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @property
    def progress(self) -> float:
        if self.total_chunks <= 0:
            return 0.0
        return round(min(1.0, self.done_chunks / self.total_chunks), 4)
