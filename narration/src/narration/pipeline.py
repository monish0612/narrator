"""Sequential explainer → TTS → Opus pipeline. Concurrency is enforced by arq, not here."""

from __future__ import annotations

import time
from typing import Any

from narration.breaker import PipelineBreaker
from narration.chunker import pack_chunks
from narration.delete import delete_artifacts
from narration.encode import (
    concat_wavs,
    encode_opus,
    ffprobe_duration_s,
    qa_should_fail,
    qa_wav,
)
from narration.keys import lock_key, ram_defer_key
from narration.llm import LlmClient, LlmError
from narration.logging import get_logger
from narration.normalize import build_cache_key
from narration.prompts import word_count
from narration.ram_gate import mem_available_bytes, next_backoff_s, should_defer
from narration.store import (
    STATUS_FALLBACK,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
    Store,
)
from narration.telegram import Telegram, format_alert
from narration.tts_client import TtsClient, TtsError

log = get_logger("narration.pipeline")


class RamDeferred(Exception):
    def __init__(self, delay_s: int) -> None:
        super().__init__(f"ram defer {delay_s}s")
        self.delay_s = delay_s


class Pipeline:
    def __init__(
        self,
        store: Store,
        llm: LlmClient,
        tts: TtsClient,
        breaker: PipelineBreaker,
        telegram: Telegram,
        *,
        voice: str,
        speed: float,
        model_version: str,
        audio_format: str,
        bitrate: int,
        hd_bitrate: int,
        max_duration_s: int,
        word_min: int,
        word_max: int,
        ram_floor: int,
        ram_alert_floor: int,
        hd: bool = False,
    ) -> None:
        self.store = store
        self.llm = llm
        self.tts = tts
        self.breaker = breaker
        self.tg = telegram
        self.voice = voice
        self.speed = speed
        self.model_version = model_version
        self.audio_format = audio_format
        self.bitrate = bitrate
        self.hd_bitrate = hd_bitrate
        self.max_duration_s = max_duration_s
        self.word_min = word_min
        self.word_max = word_max
        self.ram_floor = ram_floor
        self.ram_alert_floor = ram_alert_floor
        self.hd = hd

    def cache_for(self, text: str) -> str:
        return build_cache_key(
            article_text=text,
            voice=self.voice,
            speed=self.speed,
            model_version=self.model_version,
            audio_format=self.audio_format,
            bitrate=self.bitrate,
        )

    async def _set_article(self, article_id: str, **fields: Any) -> None:
        prev = await self.store.get_article(article_id) or {}
        prev.update(fields)
        prev.setdefault("article_id", article_id)
        await self.store.bind_article(article_id, prev)

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        article_id = str(payload["article_id"])
        title = str(payload.get("title") or "")
        source = str(payload.get("source") or "")
        category = str(payload.get("category") or "")
        text = str(payload.get("text") or "")
        hd = bool(payload.get("hd") or self.hd)
        cache = self.cache_for(text)

        existing = await self.store.get_cache(cache)
        if existing and existing.get("status") == STATUS_READY:
            await self._set_article(article_id, cache_key=cache, status=STATUS_READY, **existing)
            return {"status": STATUS_READY, "cache_key": cache, "cache_hit": True}

        if await self.breaker.is_open():
            await self._set_article(article_id, cache_key=cache, status=STATUS_FALLBACK, reason="breaker_open")
            return {"status": STATUS_FALLBACK, "cache_key": cache, "reason": "breaker_open"}

        await self._ram_gate(cache)

        lock = lock_key(self.store.prefix, cache)
        got = await self.store._r.set(lock, article_id, nx=True, ex=1800)
        if not got:
            # Another job owns this cache key — waiters just poll article status.
            await self._set_article(article_id, cache_key=cache, status=STATUS_QUEUED)
            return {"status": STATUS_QUEUED, "cache_key": cache, "reason": "lock_held"}

        await self._set_article(article_id, cache_key=cache, status=STATUS_GENERATING)
        try:
            await self.llm.ensure_model()
            script = await self.llm.generate_script(
                title=title,
                source=source,
                category=category,
                article_text=text,
                word_min=self.word_min,
                word_max=self.word_max,
            )
            self.store.write_script(cache, script)
            wpm_words = word_count(script)

            speed = self.speed
            opus, duration, metrics = await self._synth_and_encode(cache, script, speed, hd=hd)
            if duration > self.max_duration_s:
                bump = min(1.1, round(speed * 1.08, 2))
                log.warning("duration.over_cap", duration_s=duration, retry_speed=bump)
                opus, duration, metrics = await self._synth_and_encode(cache, script, bump, hd=hd)
                if duration > self.max_duration_s:
                    await self.tg.send(
                        format_alert(
                            "duration still over 600s after speed bump",
                            article=article_id,
                            duration_s=round(duration, 1),
                        )
                    )

            reason = qa_should_fail(metrics, duration)
            if reason:
                raise TtsError(f"qa_fail:{reason}")

            wpm = (wpm_words / (duration / 60.0)) if duration else 0.0
            record = {
                "cache_key": cache,
                "file_path": str(opus),
                "hd_file_path": str(self.store.opus_path(cache, hd=True)) if hd else None,
                "duration_s": duration,
                "wpm": round(wpm, 1),
                "article_text_path": str(self.store.script_path(cache)),
                "status": STATUS_READY,
                "created_at": time.time(),
                "voice": self.voice,
                "speed": speed,
                "article_id": article_id,
            }
            await self.store.put_cache(cache, record)
            await self._set_article(article_id, **record)
            reset = await self.breaker.record_success()
            if reset:
                await self.tg.send(format_alert("breaker reset", breaker=self.breaker.name))
            await self.tg.send(
                format_alert(
                    "synthesis ok",
                    article=article_id,
                    duration_s=round(duration, 1),
                    wpm=round(wpm, 1),
                )
            )
            avail = mem_available_bytes()
            if avail is not None and avail < self.ram_alert_floor:
                await self.tg.send(format_alert("RAM pressure", free_mb=round(avail / 1024 / 1024)))
            log.info("pipeline.ok", article_id=article_id, cache_key=cache, duration_s=duration)
            return {"status": STATUS_READY, "cache_key": cache, "duration_s": duration}
        except RamDeferred:
            raise
        except Exception as exc:
            log.error("pipeline.fail", article_id=article_id, error=str(exc)[:300])
            tripped = await self.breaker.record_failure(article_id)
            if tripped:
                await self.tg.send(format_alert("breaker open — on-device fallback", breaker=self.breaker.name))
            await self.tg.send(format_alert("synthesis failed", article=article_id, error=str(exc)[:180]))
            await self._set_article(
                article_id,
                cache_key=cache,
                status=STATUS_FALLBACK,
                reason=str(exc)[:200],
            )
            return {"status": STATUS_FALLBACK, "cache_key": cache, "reason": str(exc)[:200]}
        finally:
            await self.store._r.delete(lock)

    async def _ram_gate(self, cache: str) -> None:
        if not should_defer(self.ram_floor):
            await self.store._r.delete(ram_defer_key(self.store.prefix, cache))
            return
        raw = await self.store._r.incr(ram_defer_key(self.store.prefix, cache))
        await self.store._r.expire(ram_defer_key(self.store.prefix, cache), 3600)
        delay = next_backoff_s(int(raw) - 1)
        if int(raw) > 2:
            avail = mem_available_bytes()
            await self.tg.send(
                format_alert(
                    "RAM gate deferred >2",
                    delay_s=delay,
                    free_mb=None if avail is None else round(avail / 1024 / 1024),
                )
            )
        raise RamDeferred(delay)

    async def _synth_and_encode(
        self, cache: str, script: str, speed: float, *, hd: bool
    ) -> tuple[Any, float, dict]:
        chunks = pack_chunks(script)
        wavs = []
        for i, chunk in enumerate(chunks):
            wav_bytes = await self.tts.synthesize(chunk, voice=self.voice, speed=speed)
            dest = self.store.tmp_wav(cache, i)
            dest.write_bytes(wav_bytes)
            wavs.append(dest)
        concat = self.store.tmp / self.store.shard(cache) / f"{cache}.concat.wav"
        await concat_wavs(wavs, concat)
        metrics = await qa_wav(concat)
        duration = await ffprobe_duration_s(concat)
        opus = self.store.opus_path(cache)
        await encode_opus(concat, opus, bitrate=self.bitrate)
        if hd:
            await encode_opus(concat, self.store.opus_path(cache, hd=True), bitrate=self.hd_bitrate)
        for p in wavs:
            p.unlink(missing_ok=True)
        concat.unlink(missing_ok=True)
        return opus, duration, metrics


async def complete_listen(store: Store, cache: str, telegram: Telegram) -> bool:
    async def exhausted(err: str) -> None:
        await telegram.send(format_alert("delete exhausted", cache_key=cache, error=err[:180]))

    return await delete_artifacts(store, cache, on_exhausted=exhausted)
