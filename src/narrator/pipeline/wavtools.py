"""WAV file helpers for synthesis + assembly.

Silence generation, RIFF/size validation (a truncated wav from a crash is
detected here), and duration probing - all via the stdlib ``wave`` module so no
extra native dependency is needed in the worker image beyond ffmpeg.
"""

from __future__ import annotations

import wave
from pathlib import Path

_MIN_VALID_BYTES = 1024


def write_silence_wav(path: str | Path, ms: int, *, sample_rate: int = 24000) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = max(0, int(sample_rate * ms / 1000))
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * frames)
    return p


def is_valid_wav(path: str | Path, *, min_size: int = _MIN_VALID_BYTES) -> bool:
    """True if the file exists, is large enough, and has a RIFF/WAVE header.

    A truncated wav from a crash (too small or missing the WAVE tag) reads as
    invalid, so the synth stage re-synthesizes exactly that chunk on resume.
    """
    p = Path(path)
    try:
        if not p.exists() or p.stat().st_size < min_size:
            return False
        with open(p, "rb") as fh:
            header = fh.read(12)
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except OSError:
        return False


def read_wav_duration_ms(path: str | Path) -> int:
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
        if rate <= 0:
            return 0
        return int(round(1000.0 * frames / rate))
    except (wave.Error, OSError, EOFError):
        return 0
