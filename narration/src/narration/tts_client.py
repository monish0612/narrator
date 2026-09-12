"""OpenAI-compatible Kokoro client. Sequential; never concurrent with the LLM step."""

from __future__ import annotations

import httpx

from narration.logging import get_logger

log = get_logger("narration.tts")


class TtsError(Exception):
    pass


class TtsClient:
    def __init__(self, base_url: str, *, timeout: float = 600, client: httpx.AsyncClient | None = None) -> None:
        self.base = base_url.rstrip("/")
        self._own = client is None
        self._http = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._own:
            await self._http.aclose()

    async def health(self) -> bool:
        try:
            r = await self._http.get(self.base.rsplit("/v1", 1)[0] + "/health")
            return r.status_code == 200 and bool((r.json() or {}).get("ready", True))
        except Exception:
            return False

    async def synthesize(self, text: str, *, voice: str, speed: float) -> bytes:
        url = f"{self.base}/audio/speech"
        resp = await self._http.post(
            url,
            json={
                "model": "kokoro-int8",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": "wav",
            },
        )
        if resp.status_code >= 400:
            raise TtsError(f"tts http {resp.status_code}: {resp.text[:300]}")
        if not resp.content or resp.content[:4] != b"RIFF":
            raise TtsError("tts returned non-wav payload")
        return resp.content
