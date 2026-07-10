"""Concrete-implementation factory (ARCHITECTURE.md section 5).

The ONLY place that chooses concrete engines / processors / storage / chunker
from settings. Pipeline and API code import Protocols and call these builders -
swapping a model, destination or processor is a settings change or one adapter,
never edits scattered across the pipeline.
"""

from __future__ import annotations

import httpx

from narrator.core.breaker import CircuitBreaker
from narrator.core.config import settings
from narrator.core.protocols import ContentProcessor, StorageBackend
from narrator.core.state import StateStore
from narrator.engines.openai_compat import OpenAICompatEngine
from narrator.pipeline.chunker import SentenceChunker


def build_chunker() -> SentenceChunker:
    return SentenceChunker()


def build_storage(
    state: StateStore,
    *,
    drive_breaker: CircuitBreaker | None = None,
) -> StorageBackend:
    if settings.storage_backend == "local":
        from narrator.storage.local import LocalStorage

        return LocalStorage(settings.outputs_dir)
    from narrator.storage.gdrive import GoogleDriveStorage

    if drive_breaker is None:
        raise ValueError("gdrive storage requires a drive breaker")
    return GoogleDriveStorage(
        breaker=drive_breaker,
        state=state,
        root_folder_name=settings.gdrive_root_folder_name,
        credentials={
            "client_id": settings.gdrive_client_id or "",
            "client_secret": settings.gdrive_client_secret or "",
            "refresh_token": settings.gdrive_refresh_token or "",
        },
    )


def build_engine(client: httpx.AsyncClient, breaker: CircuitBreaker) -> OpenAICompatEngine:
    return OpenAICompatEngine(client, base_url=settings.tts_base_url, breaker=breaker)


def build_gemini_client():
    from narrator.processors.gemini_client import GoogleGenaiClient

    return GoogleGenaiClient(api_key=settings.require_gemini(), model=settings.gemini_model)


def build_processor(
    mode: str | None = None,
    *,
    gemini_breaker: CircuitBreaker | None = None,
    gemini_client=None,
) -> ContentProcessor:
    chosen = mode or settings.processor_default
    if chosen == "verbatim":
        from narrator.processors.verbatim import VerbatimProcessor

        return VerbatimProcessor()
    if chosen == "explainer":
        from narrator.processors.explainer import GeminiExplainerProcessor

        client = gemini_client or build_gemini_client()
        if gemini_breaker is None:
            raise ValueError("explainer processor requires a gemini breaker")
        return GeminiExplainerProcessor(
            client,
            gemini_breaker,
            target_minutes=settings.explainer_target_minutes,
        )
    raise ValueError(f"unknown processor mode: {chosen}")
