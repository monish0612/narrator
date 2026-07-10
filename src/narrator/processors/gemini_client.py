"""google-genai client adapter (ARCHITECTURE.md sections 6 / 15-D).

Wraps the async google-genai SDK behind the ``GeminiClient`` protocol and maps
SDK responses/exceptions to Narrator's typed errors: safety blocks ->
``SafetyBlockError`` (terminal), 429 -> ``GeminiError(retry_after=...)``, 5xx ->
``GeminiError``. The SDK is imported lazily so importing this module never
requires the package to be installed.
"""

from __future__ import annotations

from narrator.core.errors import GeminiError, SafetyBlockError
from narrator.core.logging import get_logger

log = get_logger(__name__)


class GoogleGenaiClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai  # lazy

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        client = self._ensure_client()
        from google.genai import types as genai_types

        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        try:
            resp = await client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # map SDK errors to typed errors
            self._raise_typed(exc)
            raise

        # Safety / empty-candidate handling.
        feedback = getattr(resp, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise SafetyBlockError(f"prompt blocked: {block_reason}")
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "")
            if "SAFETY" in finish.upper():
                raise SafetyBlockError("response blocked by safety filter")
        text = getattr(resp, "text", None)
        if not text:
            raise GeminiError("empty gemini response")
        return text

    @staticmethod
    def _raise_typed(exc: Exception) -> None:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        retry_after = None
        details = getattr(exc, "response_json", None) or {}
        if isinstance(details, dict):
            retry_after = details.get("retry_after")
        if code == 429:
            raise GeminiError("gemini rate limited", retry_after=retry_after) from exc
        if isinstance(code, int) and code >= 500:
            raise GeminiError(f"gemini server error {code}") from exc
        # Unknown -> treat as transient so it retries then trips the breaker.
        raise GeminiError(str(exc)) from exc
