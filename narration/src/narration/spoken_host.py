"""Spoken host beats: a personal open on some articles, a closer on every one.

The closer is the listener's cue that the piece finished rather than stalled.
The greeting is stable per article id so replay never flips, and only about
one in three articles get it so it stays a surprise.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from narration.encode import concat_wavs, decode_to_wav, encode_opus, ffprobe_duration_s
from narration.logging import get_logger

log = get_logger("narration.host")

LISTENER_NAME = "Monish"
HOST_TOUCH_VERSION = "v1"

_CLOSERS = (
    "That's it from this article. I'll leave it there.",
    "And that's this piece. I'll stop there.",
    "That's the article. Catch you on the next one.",
    "That's all from this one. I'm done here.",
    "And that's the end of this article.",
)

_CLOSER_NEEDLES = (
    "that's it from this",
    "that's this piece",
    "that's the article",
    "that's all from this",
    "end of this article",
    "i'll leave it there",
    "i'll stop there",
    "i'm done here",
    "catch you on the next",
)

_FALLBACK_OPEN = "Hey Monish. Let me walk you through this one."


def should_personal_open(article_id: str, *, every_n: int = 3) -> bool:
    """Stable ~1/every_n articles. Replay of the same id never changes."""
    if every_n <= 1:
        return True
    aid = (article_id or "").strip()
    if not aid:
        return False
    return hashlib.sha256(aid.encode("utf-8")).digest()[0] % every_n == 0


def pick_closer(article_id: str) -> str:
    aid = (article_id or "article").encode("utf-8")
    idx = hashlib.sha256(aid).digest()[1] % len(_CLOSERS)
    return _CLOSERS[idx]


def script_has_closer(script: str) -> bool:
    tail = " ".join((script or "").lower().split()[-60:])
    return any(n in tail for n in _CLOSER_NEEDLES)


def script_has_personal_open(script: str) -> bool:
    head = " ".join((script or "").split()[:70]).lower()
    return LISTENER_NAME.lower() in head


def finish_spoken_script(
    script: str,
    *,
    personal_open: bool,
    article_id: str,
) -> str:
    """Idempotent backstop so a forgetful model cannot leave a cold stop."""
    text = (script or "").strip()
    if not text:
        return text
    if personal_open and not script_has_personal_open(text):
        text = f"{_FALLBACK_OPEN} {text}"
    if not script_has_closer(text):
        text = f"{text} {pick_closer(article_id)}"
    return text


def personal_open_instruction(enabled: bool) -> str:
    if enabled:
        return (
            "Open with a brief personal beat to Monish — one or two sentences, "
            "warm and specific to THIS article's topic, like a friend sitting down "
            "with him. Address him by name once, then explain normally. "
            "Do not sound like a radio host, a podcast intro, or an assistant. "
            "Do not greet him again later in the script."
        )
    return (
        "Do not address the listener by name. Start with the one-sentence hook."
    )


async def ensure_host_touch(
    store: Any,
    tts: Any,
    rec: dict[str, Any],
    *,
    voice: str,
    speed: float,
    bitrate: int,
) -> dict[str, Any]:
    """Stamp a closer onto already-ready opus that was generated before this host beat.

    Never raises into the HTTP path. If TTS/ffmpeg fails, the original file stays.
    """
    if not rec or rec.get("host_touch") == HOST_TOUCH_VERSION:
        return rec
    cache = str(rec.get("cache_key") or "").strip()
    if not cache:
        return rec
    opus = store.opus_path(cache)
    if not opus.exists():
        return rec

    article_id = str(rec.get("article_id") or "")
    script_path = store.script_path(cache)
    script = ""
    try:
        if script_path.exists():
            script = script_path.read_text(encoding="utf-8")
    except OSError:
        script = ""

    if script_has_closer(script):
        return await _stamp_host_touch(store, rec, cache)

    lock_key = f"{store.prefix}touch:{cache}"
    try:
        got = await store._r.set(lock_key, "1", nx=True, ex=90)
    except Exception:
        got = True
    if not got:
        return rec

    closer = pick_closer(article_id)
    tmp_dir = store.tmp / store.shard(cache)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    closer_wav = tmp_dir / f"{cache}.closer.wav"
    body_wav = tmp_dir / f"{cache}.body.wav"
    concat = tmp_dir / f"{cache}.touch.wav"
    out_opus = tmp_dir / f"{cache}.touch.opus"
    try:
        wav_bytes = await tts.synthesize(closer, voice=voice, speed=speed)
        if not wav_bytes or wav_bytes[:4] != b"RIFF":
            raise RuntimeError("closer tts was not wav")
        closer_wav.write_bytes(wav_bytes)
        await decode_to_wav(opus, body_wav)
        await concat_wavs([body_wav, closer_wav], concat)
        await encode_opus(concat, out_opus, bitrate=bitrate)
        os.replace(out_opus, opus)
        duration = await ffprobe_duration_s(opus)
        rec = dict(rec)
        rec["duration_s"] = duration
        rec["host_touch"] = HOST_TOUCH_VERSION
        rec["file_path"] = str(opus)
        hd = store.opus_path(cache, hd=True)
        if hd.exists():
            hd_body = tmp_dir / f"{cache}.hd.body.wav"
            hd_concat = tmp_dir / f"{cache}.hd.touch.wav"
            hd_out = tmp_dir / f"{cache}.hd.touch.opus"
            await decode_to_wav(hd, hd_body)
            await concat_wavs([hd_body, closer_wav], hd_concat)
            await encode_opus(hd_concat, hd_out, bitrate=bitrate)
            os.replace(hd_out, hd)
            rec["hd_file_path"] = str(hd)
            hd_body.unlink(missing_ok=True)
            hd_concat.unlink(missing_ok=True)
        finished = finish_spoken_script(
            script,
            personal_open=False,
            article_id=article_id,
        )
        store.write_script(cache, finished)
        await store.put_cache(cache, rec)
        if article_id:
            await store.bind_article(article_id, rec)
        log.info("host.touch_appended", article_id=article_id, cache_key=cache)
        return rec
    except Exception as exc:
        log.warning("host.touch_failed", cache_key=cache, error=str(exc)[:200])
        return rec
    finally:
        for p in (closer_wav, body_wav, concat, out_opus):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            await store._r.delete(lock_key)
        except Exception:
            pass


async def _stamp_host_touch(store: Any, rec: dict[str, Any], cache: str) -> dict[str, Any]:
    rec = dict(rec)
    rec["host_touch"] = HOST_TOUCH_VERSION
    try:
        live = await store.get_cache(cache) or rec
        live.update(rec)
        await store.put_cache(cache, live)
        aid = str(rec.get("article_id") or "")
        if aid:
            await store.bind_article(aid, live)
    except Exception as exc:
        log.warning("host.touch_stamp_failed", cache_key=cache, error=str(exc)[:180])
    return rec
