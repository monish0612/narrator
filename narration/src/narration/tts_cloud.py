"""Cloud TTS: Chirp 3 HD primary, Gemini Flash TTS fallback. One default lives here."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
from typing import Any

import httpx

from narration.logging import get_logger

log = get_logger("narration.tts_cloud")

SCRIPT_VERSION = "script-v3"
DEFAULT_TTS_MODEL = "chirp3-hd"
DAILY_CHAR_CAP = 60_000

CHIRP_VOICES = (
    "en-US-Chirp3-HD-Charon",
    "en-US-Chirp3-HD-Kore",
    "en-US-Chirp3-HD-Aoede",
)
GEMINI_VOICES = ("Charon", "Kore", "Aoede")
DEFAULT_VOICE = {
    "chirp3-hd": "en-US-Chirp3-HD-Charon",
    "gemini-2.5-flash-preview-tts": "Charon",
}
FALLBACK_ENGINE = {
    "chirp3-hd": "gemini-2.5-flash-preview-tts",
    "gemini-2.5-flash-preview-tts": "chirp3-hd",
}
TIMEOUT_S = {"chirp3-hd": 10.0, "gemini-2.5-flash-preview-tts": 30.0}
# Approximate paid cost for one ~3,700-character narration after the free tier.
COST_HINT = {
    "chirp3-hd": "$0.02 after 1M free chars",
    "gemini-2.5-flash-preview-tts": "$0.05",
}


class CloudTtsError(Exception):
    def __init__(self, message: str, *, status: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def normalize_engine(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in ("chirp3-hd", "chirp", "chirp-3-hd"):
        return "chirp3-hd"
    if text in ("gemini-2.5-flash-preview-tts", "gemini-flash-tts", "gemini"):
        return "gemini-2.5-flash-preview-tts"
    return DEFAULT_TTS_MODEL


def normalize_voice(engine: str, voice: str | None) -> str:
    engine = normalize_engine(engine)
    raw = (voice or "").strip()
    allowed = CHIRP_VOICES if engine == "chirp3-hd" else GEMINI_VOICES
    if raw in allowed:
        return raw
    short = raw.split("-")[-1]
    for name in allowed:
        if name == short or name.endswith(short):
            return name
    return DEFAULT_VOICE[engine]


def closest_voice(src_engine: str, src_voice: str, dst_engine: str) -> str:
    short = src_voice.split("-")[-1]
    return normalize_voice(dst_engine, short)


class CloudTts:
    def __init__(self, redis: Any, prefix: str, *, gemini_api_key: str, sa_json: str, daily_cap: int = DAILY_CHAR_CAP) -> None:
        self.redis = redis
        self.prefix = prefix
        self.gemini_api_key = gemini_api_key
        self.sa = json.loads(sa_json)
        self.daily_cap = daily_cap
        self._token = ""
        self._token_exp = 0.0
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _access_token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        now = int(time.time())
        header = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        claim = _b64(json.dumps({
            "iss": self.sa["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
        }).encode())
        pem = "/tmp/tts-sa.pem"
        with open(pem, "w", encoding="utf-8") as fh:
            fh.write(self.sa["private_key"])
        os.chmod(pem, 0o600)
        proc = await asyncio.create_subprocess_exec(
            "openssl", "dgst", "-sha256", "-sign", pem,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        sig, _ = await proc.communicate(f"{header}.{claim}".encode())
        if proc.returncode:
            raise CloudTtsError("token sign failed", retryable=True)
        jwt = f"{header}.{claim}.{_b64(sig)}"
        resp = await self._http.post(
            "https://oauth2.googleapis.com/token",
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": jwt},
        )
        if resp.status_code >= 400:
            raise CloudTtsError(f"token http {resp.status_code}", status=resp.status_code, retryable=resp.status_code >= 500)
        self._token = resp.json()["access_token"]
        self._token_exp = time.time() + 3500
        return self._token

    async def breaker_open(self, engine: str) -> bool:
        until = await self.redis.get(f"{self.prefix}tts_skip:{engine}")
        if not until:
            return False
        return float(until) > time.time()

    async def note_failure(self, engine: str) -> None:
        key = f"{self.prefix}tts_fail:{engine}"
        now = time.time()
        await self.redis.zadd(key, {str(now): now})
        await self.redis.zremrangebyscore(key, 0, now - 300)
        count = await self.redis.zcard(key)
        if int(count) >= 5:
            await self.redis.set(f"{self.prefix}tts_skip:{engine}", str(now + 600), ex=600)

    async def note_success(self, engine: str) -> None:
        await self.redis.delete(f"{self.prefix}tts_fail:{engine}")

    async def charge(self, chars: int) -> bool:
        day = time.strftime("%Y%m%d", time.gmtime())
        key = f"{self.prefix}tts_chars:{day}"
        used = int(await self.redis.incrby(key, chars))
        await self.redis.expire(key, 172800)
        return used <= self.daily_cap

    async def synthesize(self, text: str, *, engine: str, voice: str) -> tuple[bytes, dict[str, Any]]:
        engine = normalize_engine(engine)
        voice = normalize_voice(engine, voice)
        if await self.breaker_open(engine):
            raise CloudTtsError(f"breaker {engine}", retryable=False)
        if not await self.charge(len(text)):
            raise CloudTtsError("daily_tts_cap", retryable=False)
        last: Exception | None = None
        for attempt in range(3):
            try:
                if engine == "chirp3-hd":
                    wav, meta = await self._chirp(text, voice)
                else:
                    wav, meta = await self._gemini(text, voice)
                await self.note_success(engine)
                meta.update({"engine": engine, "voice": voice, "chars": len(text), "attempt": attempt})
                return wav, meta
            except CloudTtsError as exc:
                last = exc
                if not exc.retryable or exc.status in (400, 401, 403):
                    await self.note_failure(engine)
                    raise
                if attempt == 2:
                    await self.note_failure(engine)
                    raise
                delay = (0.4 * (2 ** attempt)) + random.random() * 0.3
                log.warning("tts_cloud.retry", engine=engine, attempt=attempt + 1, error=str(exc)[:160])
                await asyncio.sleep(delay)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last = exc
                if attempt == 2:
                    await self.note_failure(engine)
                    raise CloudTtsError(f"timeout {engine}", retryable=True) from exc
                await asyncio.sleep((0.4 * (2 ** attempt)) + random.random() * 0.3)
        raise CloudTtsError(str(last))

    async def _chirp(self, text: str, voice: str) -> tuple[bytes, dict[str, Any]]:
        token = await self._access_token()
        resp = await self._http.post(
            "https://texttospeech.googleapis.com/v1/text:synthesize",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "input": {"text": text},
                "voice": {"languageCode": "en-US", "name": voice},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
            },
            timeout=TIMEOUT_S["chirp3-hd"],
        )
        return _wav_from_response(resp, "chirp3-hd")

    async def _gemini(self, text: str, voice: str) -> tuple[bytes, dict[str, Any]]:
        resp = await self._http.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent",
            headers={"x-goog-api-key": self.gemini_api_key},
            json={
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
                },
            },
            timeout=TIMEOUT_S["gemini-2.5-flash-preview-tts"],
        )
        if resp.status_code in (429,) or resp.status_code >= 500:
            raise CloudTtsError(f"gemini tts http {resp.status_code}", status=resp.status_code, retryable=True)
        if resp.status_code >= 400:
            raise CloudTtsError(f"gemini tts http {resp.status_code}", status=resp.status_code, retryable=False)
        data = resp.json()
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        b64 = ""
        for part in parts:
            inline = part.get("inlineData") or {}
            if inline.get("data"):
                b64 = inline["data"]
                break
        if not b64:
            raise CloudTtsError("gemini tts empty", retryable=True)
        pcm = base64.b64decode(b64)
        usage = data.get("usageMetadata") or {}
        return _pcm_wav(pcm, 24000), {"audio_tokens": usage.get("candidatesTokenCount")}


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _pcm_wav(pcm: bytes, rate: int) -> bytes:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _wav_from_response(resp: httpx.Response, engine: str) -> tuple[bytes, dict[str, Any]]:
    if resp.status_code in (429,) or resp.status_code >= 500:
        raise CloudTtsError(f"{engine} http {resp.status_code}", status=resp.status_code, retryable=True)
    if resp.status_code in (400, 401, 403) or resp.status_code >= 400:
        raise CloudTtsError(f"{engine} http {resp.status_code}", status=resp.status_code, retryable=False)
    audio = base64.b64decode(resp.json().get("audioContent") or "")
    if audio[:4] != b"RIFF":
        audio = _pcm_wav(audio, 24000)
    return audio, {}


def load_sa_json() -> str:
    raw = os.environ.get("GOOGLE_TTS_SA_JSON", "").strip()
    if not raw or raw.startswith("{{"):
        raise CloudTtsError("GOOGLE_TTS_SA_JSON missing")
    if not raw.startswith("{"):
        raw = base64.b64decode(raw).decode()
    return raw
