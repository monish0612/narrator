"""Typed error hierarchy (ARCHITECTURE.md sections 14-15).

The retry/breaker layer keys entirely off these types:
* ``RetryableError`` subclasses are retried by tenacity and counted by the
  circuit breaker.
* ``TerminalError`` subclasses are never retried and never trip a breaker - they
  route to their designed automated response (silence placeholder, verbatim
  fallback, UPLOAD_PENDING, typed 4xx).
"""

from __future__ import annotations


class NarratorError(Exception):
    """Base class for all Narrator domain errors."""


class RetryableError(NarratorError):
    """A transient dependency failure - safe to retry behind a breaker."""


class TerminalError(NarratorError):
    """A permanent failure - never retry, never trip the breaker."""


class BreakerOpen(NarratorError):
    """Raised when a circuit breaker is open; surfaced as ``stall_reason``."""

    def __init__(self, name: str) -> None:
        super().__init__(f"circuit breaker '{name}' is open")
        self.name = name


# --- TTS ---------------------------------------------------------------------
class TTSError(RetryableError):
    """Transient TTS failure (timeout, conn reset, 5xx, 429)."""


class UnspeakableError(TerminalError):
    """TTS rejected the text as unspeakable (400 with machine-readable reason)."""

    def __init__(self, reason: str = "unspeakable") -> None:
        super().__init__(f"unspeakable input: {reason}")
        self.reason = reason


# --- Gemini ------------------------------------------------------------------
class GeminiError(RetryableError):
    """Transient Gemini failure (429/5xx/timeout). ``retry_after`` honored."""

    def __init__(self, message: str = "gemini error", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InvalidJSONError(RetryableError):
    """Gemini returned malformed/truncated JSON - repair then one re-ask."""


class SafetyBlockError(TerminalError):
    """Gemini safety block - section falls back to verbatim + warning."""


# --- Storage / delivery ------------------------------------------------------
class DeliveryRetryable(RetryableError):
    """Transient delivery failure (rate limit, 5xx, resumable chunk error)."""


class DeliveryTerminal(TerminalError):
    """Terminal delivery failure (invalid_grant / storageQuotaExceeded).

    Never a job failure: the job goes UPLOAD_PENDING with a downloadable local
    artifact and a human rotates the token / clears quota.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"delivery terminal: {reason}")
        self.reason = reason


# --- Pipeline ----------------------------------------------------------------
class SkipChunk(NarratorError):
    """Sanitizer produced empty output - emit a short silence, not an error."""


class AssemblyError(NarratorError):
    """ffmpeg assembly failed (non-corrupt-chunk cause) - resumable FAILED."""


class JobCancelled(NarratorError):
    """The cancel flag was observed between chunks/stages."""


class JobFailedResumable(NarratorError):
    """Too many chunks failed (> MAX_FAILED_CHUNK_PCT). Resumable via /retry."""


class IngestError(NarratorError):
    """Input could not be ingested. Carries an HTTP status for the API (422)."""

    def __init__(self, reason: str, *, status_code: int = 422) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
