"""WAV encoding helpers.

Produces standard PCM16 mono WAV bytes from float samples using the stdlib
``wave`` module, so the request path never hard-depends on libsndfile/soundfile.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def float_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32/-1..1 samples to 16-bit PCM mono WAV bytes."""
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_duration_ms(samples: np.ndarray, sample_rate: int) -> int:
    n = int(np.asarray(samples).reshape(-1).shape[0])
    if sample_rate <= 0:
        return 0
    return int(round(1000.0 * n / sample_rate))
