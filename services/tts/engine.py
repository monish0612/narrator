"""Kokoro int8 ONNX engine wrapper (ARCHITECTURE.md section 12).

One ``InferenceSession`` for the whole process: ``intra_op=ONNX_INTRA_OP``,
``inter_op=1``, sequential execution, full graph optimization. An internal
``asyncio.Semaphore(1)`` serializes inference (concurrency lives at the
worker/pipeline level). ``ready`` flips true only after a real warmup synthesis
so the healthcheck never lets a cold-start request time out.

``kokoro_onnx`` is imported lazily inside :meth:`load` so the module (and blend
parser / request validation tests) import without onnxruntime installed.
"""

from __future__ import annotations

import asyncio
import collections
import os
import time
from pathlib import Path

import numpy as np

from tts.blend import VoiceSpecError, blend_style_vector, is_blend, parse_voice_spec


class UnspeakableInput(Exception):
    """Text produced no speakable phonemes (maps to HTTP 400)."""

    def __init__(self, reason: str = "unspeakable") -> None:
        super().__init__(reason)
        self.reason = reason


class BlendUnsupported(Exception):
    """The pinned kokoro_onnx build cannot synthesize custom blend vectors."""


def _rss_bytes() -> int:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return int(ru * 1024) if ru < (1 << 40) else int(ru)
    except Exception:
        try:
            with open("/proc/self/statm") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return 0


class KokoroEngine:
    def __init__(
        self,
        model_path: str | Path,
        voices_path: str | Path,
        *,
        intra_op: int = 2,
        default_lang: str = "en-us",
    ) -> None:
        self._model_path = str(model_path)
        self._voices_path = str(voices_path)
        self._intra_op = intra_op
        self._lang = default_lang
        self._kokoro = None
        self._sem = asyncio.Semaphore(1)
        self._ready = False
        self._supports_blend = False
        self._voices: list[str] = []
        # rolling stats (last 50 inferences)
        self._rtf_window: collections.deque[float] = collections.deque(maxlen=50)
        self._count = 0

    # --- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        """Construct the ONNX session (blocking; run via to_thread at startup)."""
        # Thread pinning must be set before onnxruntime spins up its pools.
        os.environ.setdefault("OMP_NUM_THREADS", str(self._intra_op))
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(self._intra_op))

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        so = ort.SessionOptions()
        so.intra_op_num_threads = self._intra_op
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            session = ort.InferenceSession(
                self._model_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._kokoro = Kokoro.from_session(session=session, voices_path=self._voices_path)
        except (AttributeError, TypeError):
            # Older kokoro_onnx without from_session(): construct from paths.
            self._kokoro = Kokoro(self._model_path, self._voices_path)

        self._voices = self._read_voices()
        self._supports_blend = self._detect_blend_support()

    def _read_voices(self) -> list[str]:
        k = self._kokoro
        for attr in ("get_voices", "voices"):
            obj = getattr(k, attr, None)
            if callable(obj):
                try:
                    return sorted(obj())
                except Exception:
                    continue
            if isinstance(obj, dict):
                return sorted(obj.keys())
        return []

    def _voice_style(self, name: str) -> np.ndarray:
        """Fetch a single voice's style vector from the kokoro voices table."""
        k = self._kokoro
        getter = getattr(k, "get_voice_style", None)
        if callable(getter):
            return np.asarray(getter(name), dtype=np.float32)
        voices = getattr(k, "voices", None)
        if isinstance(voices, dict) and name in voices:
            return np.asarray(voices[name], dtype=np.float32)
        raise VoiceSpecError(f"no style vector for voice {name!r}")

    def _detect_blend_support(self) -> bool:
        """Probe whether create() accepts a raw numpy style vector as ``voice``."""
        if not self._voices:
            return False
        try:
            style = self._voice_style(self._voices[0])
            # A dry structural check: we can obtain a vector and create() exists.
            return isinstance(style, np.ndarray) and hasattr(self._kokoro, "create")
        except Exception:
            return False

    async def warmup(self) -> None:
        await self.synthesize("Narrator is ready.", voice="af_heart", speed=1.0)
        self._ready = True

    # --- properties ----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def voices(self) -> list[str]:
        return list(self._voices)

    def stats(self) -> dict[str, float | int]:
        rtf = sum(self._rtf_window) / len(self._rtf_window) if self._rtf_window else 0.0
        return {
            "count": self._count,
            "rolling_rtf": round(rtf, 3),
            "rss_bytes": _rss_bytes(),
            "ready": self._ready,
        }

    # --- synthesis -----------------------------------------------------------
    def _resolve_voice(self, voice: str):
        """Return a value acceptable as kokoro.create(voice=...)."""
        if is_blend(voice):
            if not self._supports_blend:
                raise BlendUnsupported("voice blending unsupported by this build")
            styles = {name: self._voice_style(name) for name, _ in parse_voice_spec(voice)}
            return blend_style_vector(styles, voice)
        return voice

    def _create(self, text: str, voice, speed: float) -> tuple[np.ndarray, int]:
        k = self._kokoro
        samples, sample_rate = k.create(text, voice=voice, speed=speed, lang=self._lang)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        response_format: str = "wav",
    ) -> tuple[np.ndarray, int, int]:
        """Return (float samples, sample_rate, wall_ms). Serialized by semaphore."""
        if not text or not text.strip():
            raise UnspeakableInput("empty input")
        resolved = self._resolve_voice(voice)
        async with self._sem:
            start = time.perf_counter()
            samples, sample_rate = await asyncio.to_thread(self._create, text, resolved, speed)
            wall = time.perf_counter() - start
        if samples.size == 0:
            raise UnspeakableInput("no phonemes produced")
        audio_seconds = samples.size / sample_rate if sample_rate else 0.0
        if wall > 0 and audio_seconds > 0:
            self._rtf_window.append(audio_seconds / wall)
        self._count += 1
        wall_ms = int(round(wall * 1000))
        return samples, sample_rate, wall_ms
