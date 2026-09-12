"""ffmpeg encode + duration + silencedetect/astats QA. Files only, never RAM."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from narration.logging import get_logger

log = get_logger("narration.encode")

_SILENCE_DUR = re.compile(r"silence_duration:\s*([0-9.]+)")
_RMS = re.compile(r"RMS level dB:\s*([-\d.]+)")
_PEAK = re.compile(r"Peak level dB:\s*([-\d.]+)")


class EncodeError(Exception):
    pass


async def _run(args: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _out, err = await proc.communicate()
    text = (err or b"").decode("utf-8", errors="replace")
    return proc.returncode or 0, text


async def concat_wavs(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise EncodeError("concat failed: no wav chunks")
    for i, p in enumerate(paths):
        raw = p.read_bytes()[:12] if p.exists() else b""
        if len(raw) < 12 or raw[:4] != b"RIFF":
            raise EncodeError(f"concat failed: chunk {i} is not a wav ({p.name})")
    if len(paths) == 1:
        dest.write_bytes(paths[0].read_bytes())
        return

    # Kokoro chunks can differ in rate/layout; -c copy then fails with
    # "Invalid data found when processing input". Resample into one stream.
    args: list[str] = ["ffmpeg", "-y"]
    for p in paths:
        args.extend(["-i", str(p)])
    n = len(paths)
    streams = "".join(f"[{i}:a]" for i in range(n))
    args.extend(
        [
            "-filter_complex",
            f"{streams}concat=n={n}:v=0:a=1[a]",
            "-map",
            "[a]",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    code, err = await _run(args)
    if code != 0:
        raise EncodeError(f"concat failed: {err[-400:]}")


async def encode_opus(wav: Path, dest: Path, *, bitrate: int, sample_rate: int = 24000) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    code, err = await _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "libopus",
            "-b:a",
            str(int(bitrate)),
            "-vbr",
            "on",
            "-application",
            "voip",
            str(dest),
        ]
    )
    if code != 0:
        raise EncodeError(f"opus encode failed: {err[-400:]}")


async def ffprobe_duration_s(path: Path) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode:
        raise EncodeError(f"ffprobe failed: {err.decode()[-300:]}")
    try:
        return float((out or b"0").decode().strip())
    except ValueError as exc:
        raise EncodeError("ffprobe duration unreadable") from exc


async def qa_wav(wav: Path) -> dict[str, float | bool]:
    """silencedetect + astats. Returns metrics; caller decides fail vs warn."""
    code, err = await _run(
        [
            "ffmpeg",
            "-i",
            str(wav),
            "-af",
            "silencedetect=noise=-40dB:d=0.5,astats=metadata=1:reset=1",
            "-f",
            "null",
            "-",
        ]
    )
    silences = [float(x) for x in _SILENCE_DUR.findall(err)]
    rms = [float(x) for x in _RMS.findall(err)]
    peaks = [float(x) for x in _PEAK.findall(err)]
    total_silence = sum(silences)
    peak = max(peaks) if peaks else -99.0
    rms_db = rms[-1] if rms else -99.0
    clipped = peak >= -0.1
    return {
        "ffmpeg_ok": code == 0,
        "silence_s": total_silence,
        "peak_db": peak,
        "rms_db": rms_db,
        "clipped": clipped,
    }


def qa_should_fail(metrics: dict[str, float | bool], duration_s: float) -> str | None:
    if metrics.get("clipped"):
        return "clipped"
    silence = float(metrics.get("silence_s") or 0)
    if duration_s > 0 and silence / duration_s > 0.25:
        return "too_much_silence"
    rms = float(metrics.get("rms_db") or -99)
    if rms < -40:
        return "inaudible"
    return None
