"""The modularity contract (ARCHITECTURE.md section 5).

Pipeline and API code import ONLY these Protocols. Concrete implementations are
selected exclusively in ``core/factory.py`` from settings. Do not import
concrete engine/processor/storage classes anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from narrator.core.models import (
    Chunk,
    Document,
    JobParams,
    NarrationScript,
    StoredFile,
    SynthesisResult,
)


@dataclass
class JobContext:
    """Ambient context handed to processors (job id, params, warning sink)."""

    job_id: str
    params: JobParams
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@runtime_checkable
class ContentProcessor(Protocol):
    async def process(self, doc: Document, ctx: JobContext) -> NarrationScript: ...


@runtime_checkable
class TTSEngine(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        response_format: str = "wav",
    ) -> SynthesisResult: ...

    async def list_voices(self) -> list[str]: ...

    async def health(self) -> bool: ...


@runtime_checkable
class StorageBackend(Protocol):
    async def upload(
        self,
        local_path: str,
        *,
        filename: str,
        mime_type: str,
        meta: dict,
    ) -> StoredFile: ...


@runtime_checkable
class Chunker(Protocol):
    def split(self, script: NarrationScript) -> list[Chunk]: ...
