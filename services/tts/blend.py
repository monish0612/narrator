"""Voice-spec parsing and style-vector blending (ARCHITECTURE.md section 12).

Grammar: ``name`` or ``name(w)+name(w)+...`` where ``w`` is a positive weight.
Examples:
    "af_heart"                 -> [("af_heart", 1.0)]
    "af_heart(2)+af_bella(1)"  -> [("af_heart", 2.0), ("af_bella", 1.0)]

Blending weighted-averages the per-voice style vectors. The exact create/voice
signature of the installed ``kokoro_onnx`` package is the source of truth at
build time; if custom style vectors are unsupported by the pinned version, the
engine restricts to single voices and returns a 400 for blend specs.
"""

from __future__ import annotations

import re

import numpy as np

_VOICE_TOKEN = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(?:\(\s*([0-9]*\.?[0-9]+)\s*\))?\s*$")


class VoiceSpecError(ValueError):
    """Malformed voice spec."""


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    """Parse a voice spec into ``[(name, weight), ...]``.

    Raises ``VoiceSpecError`` on malformed input.
    """
    if not spec or not spec.strip():
        raise VoiceSpecError("empty voice spec")
    parts = [p for p in spec.split("+")]
    out: list[tuple[str, float]] = []
    for part in parts:
        m = _VOICE_TOKEN.match(part)
        if not m:
            raise VoiceSpecError(f"invalid voice token: {part!r}")
        name = m.group(1)
        weight = float(m.group(2)) if m.group(2) is not None else 1.0
        if weight <= 0:
            raise VoiceSpecError(f"weight must be positive: {part!r}")
        out.append((name, weight))
    if not out:
        raise VoiceSpecError("no voices parsed")
    return out


def is_blend(spec: str) -> bool:
    try:
        return len(parse_voice_spec(spec)) > 1
    except VoiceSpecError:
        return False


def blend_style_vector(
    styles: dict[str, np.ndarray],
    spec: str,
) -> np.ndarray:
    """Weighted-average the named style vectors into one (weights normalized).

    ``styles`` maps voice name -> style vector (numpy array). Raises
    ``VoiceSpecError`` if a referenced voice is unknown.
    """
    parsed = parse_voice_spec(spec)
    total_w = sum(w for _, w in parsed)
    acc: np.ndarray | None = None
    for name, weight in parsed:
        if name not in styles:
            raise VoiceSpecError(f"unknown voice: {name}")
        vec = np.asarray(styles[name], dtype=np.float32)
        contribution = vec * (weight / total_w)
        acc = contribution if acc is None else acc + contribution
    assert acc is not None
    return acc
