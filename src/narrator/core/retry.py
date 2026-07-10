"""Retry policies + status classifiers (ARCHITECTURE.md section 14).

All external calls retry only via these tenacity policies (never ad-hoc
try/sleep). Callables must raise the typed exceptions from ``core.errors`` -
``RetryableError`` subclasses are retried, ``TerminalError`` subclasses are
raised immediately. Policies are then wrapped by ``core.breaker``.
"""

from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from narrator.core.errors import (
    DeliveryRetryable,
    DeliveryTerminal,
    GeminiError,
    InvalidJSONError,
    RetryableError,
    TTSError,
    UnspeakableError,
)
from narrator.core.logging import get_logger

log = get_logger(__name__)

# Transport-level exceptions that are always retryable regardless of adapter.
_TRANSPORT_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)

_RETRY_TYPES = (RetryableError, *_TRANSPORT_RETRYABLE)


class _WaitRespectRetryAfter(wait_base):
    """Honor ``retry_after`` on the raised exception (Gemini/HTTP 429), else
    fall back to exponential jitter."""

    def __init__(self, initial: float, maximum: float) -> None:
        self._fallback = wait_exponential_jitter(initial=initial, max=maximum)
        self._cap = maximum

    def __call__(self, retry_state) -> float:  # type: ignore[override]
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            try:
                return min(float(retry_after), self._cap)
            except (TypeError, ValueError):
                pass
        return self._fallback(retry_state)


def _log_retry(retry_state) -> None:  # pragma: no cover - logging only
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    log.warning(
        "retry",
        attempt=retry_state.attempt_number,
        error=type(exc).__name__ if exc else None,
    )


def _retrying(max_attempts: int, *, initial: float = 1.0, maximum: float = 60.0) -> AsyncRetrying:
    return AsyncRetrying(
        retry=retry_if_exception_type(_RETRY_TYPES),
        wait=_WaitRespectRetryAfter(initial, maximum),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
        before_sleep=_log_retry,
    )


# --- per-dependency policies (section 14 matrix) -----------------------------
def tts_retrying() -> AsyncRetrying:
    """TTS: timeout / conn-reset / 5xx / 429 - 6 attempts, 1 -> 60 s."""
    return _retrying(6, initial=1.0, maximum=60.0)


def gemini_retrying() -> AsyncRetrying:
    """Gemini: 429 (honor retry-after) / 5xx / timeout / bad JSON - 5 attempts."""
    return _retrying(5, initial=1.0, maximum=60.0)


def drive_retrying() -> AsyncRetrying:
    """Drive: rate-limit / 429 / 5xx / resumable-chunk errors - 8 attempts."""
    return _retrying(8, initial=1.0, maximum=60.0)


# --- status classifiers ------------------------------------------------------
def classify_tts_response(response: httpx.Response) -> None:
    """Raise the correct typed error for a non-2xx TTS response, or return."""
    code = response.status_code
    if code < 400:
        return
    if code == 429 or code >= 500:
        raise TTSError(f"tts http {code}")
    # 400 and other 4xx are treated as unspeakable input (never retried).
    reason = f"http {code}"
    try:
        body = response.json()
        reason = str(body.get("reason") or body.get("detail") or reason)
    except Exception:
        pass
    raise UnspeakableError(reason)


def classify_gemini_status(status_code: int, *, retry_after: float | None = None) -> None:
    if status_code == 429:
        raise GeminiError("gemini rate limited", retry_after=retry_after)
    if status_code in (500, 503) or status_code >= 500:
        raise GeminiError(f"gemini http {status_code}")


def classify_drive_error(status_code: int, reason: str) -> None:
    """Map a Google Drive error to retryable vs terminal (section 14)."""
    reason_l = (reason or "").lower()
    if status_code == 401 or "invalid_grant" in reason_l:
        raise DeliveryTerminal("invalid_grant")
    if status_code == 403 and "storagequotaexceeded" in reason_l.replace(" ", ""):
        raise DeliveryTerminal("storageQuotaExceeded")
    if status_code == 403 and ("userratelimitexceeded" in reason_l.replace(" ", "") or "ratelimit" in reason_l):
        raise DeliveryRetryable(f"drive 403 {reason}")
    if status_code == 429 or status_code >= 500:
        raise DeliveryRetryable(f"drive {status_code} {reason}")


__all__ = [
    "InvalidJSONError",
    "classify_drive_error",
    "classify_gemini_status",
    "classify_tts_response",
    "drive_retrying",
    "gemini_retrying",
    "tts_retrying",
]
