"""OpenAI-compatible TTS engine adapter (ARCHITECTURE.md sections 4-5, 12).

The only thing the pipeline knows about synthesis is this adapter and the
``TTS_BASE_URL``. Every call goes through the tenacity TTS policy wrapped by the
TTS circuit breaker. A 400 becomes a terminal ``UnspeakableError`` (no retry);
5xx/429/timeouts retry then, once exhausted, count against the breaker.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from tenacity import AsyncRetrying

from narrator.core.breaker import CircuitBreaker
from narrator.core.logging import get_logger
from narrator.core.models import SynthesisResult
from narrator.core.retry import classify_tts_response, tts_retrying

log = get_logger(__name__)


def _health_root(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


class OpenAICompatEngine:
    """Implements the ``TTSEngine`` protocol against an OpenAI-speech server."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        breaker: CircuitBreaker,
        model: str = "kokoro-int8",
        retrying_factory: Callable[[], AsyncRetrying] = tts_retrying,
    ) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._breaker = breaker
        self._model = model
        self._retrying_factory = retrying_factory

    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        response_format: str = "wav",
    ) -> SynthesisResult:
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
        }

        async def _call() -> httpx.Response:
            resp = await self._client.post(f"{self._base}/audio/speech", json=payload)
            classify_tts_response(resp)  # raises UnspeakableError / TTSError
            return resp

        async with self._breaker.guard():
            resp = await self._retrying_factory()(_call)

        duration_ms = int(resp.headers.get("X-Audio-Duration-Ms", "0") or 0)
        sample_rate = int(resp.headers.get("X-Sample-Rate", "24000") or 24000)
        return SynthesisResult(
            audio=resp.content,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            response_format=response_format,
        )

    async def list_voices(self) -> list[str]:
        async def _call() -> httpx.Response:
            resp = await self._client.get(f"{self._base}/audio/voices")
            resp.raise_for_status()
            return resp

        resp = await self._retrying_factory()(_call)
        data = resp.json()
        return list(data.get("voices", []))

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{_health_root(self._base)}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
