"""Chunk sanitizer (ARCHITECTURE.md section 9-1 / 15-I).

``clean()`` is the deterministic text transform used both for the chunk sha256
(chunker) and the cache key / synthesis input (synth), so the two never drift.
``sanitize_chunk()`` layers the *skip decisions* on top (empty-after-clean and
non-English policy) and raises ``SkipChunk`` - a skipped chunk becomes a short
silence + warning downstream, never an error.
"""

from __future__ import annotations

import re
import unicodedata

from narrator.core.errors import SkipChunk

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_REPEAT_RE = re.compile(r"(.)\1{3,}", re.DOTALL)
_WS_RE = re.compile(r"\s+")
_LONG_TOKEN = 40

# Emoji / pictographic / symbol ranges + variation selectors.
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff\U00002b00-\U00002bff\ufe0f\u200d]",
    flags=re.UNICODE,
)


def _strip_control(text: str) -> str:
    out = []
    for ch in text:
        if ch in ("\n", "\t", "\r"):
            out.append(" ")
            continue
        if unicodedata.category(ch)[0] == "C":
            continue
        out.append(ch)
    return "".join(out)


def _break_long_tokens(text: str) -> str:
    parts = []
    for token in text.split(" "):
        if len(token) > _LONG_TOKEN:
            parts.extend(token[i : i + _LONG_TOKEN] for i in range(0, len(token), _LONG_TOKEN))
        else:
            parts.append(token)
    return " ".join(parts)


def clean(text: str) -> str:
    """Deterministic sanitization (never raises, never applies skip policy)."""
    text = unicodedata.normalize("NFC", text)
    text = _strip_control(text)
    text = _URL_RE.sub(" link ", text)
    text = _EMAIL_RE.sub(" email address ", text)
    text = _EMOJI_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1\1\1", text)
    text = _break_long_tokens(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def non_latin_ratio(text: str) -> float:
    letters = 0
    non_latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        try:
            name = unicodedata.name(ch)
        except ValueError:
            non_latin += 1
            continue
        if not name.startswith("LATIN"):
            non_latin += 1
    if letters == 0:
        return 0.0
    return non_latin / letters


def sanitize_chunk(text: str, *, non_english_policy: str = "skip_warn") -> str:
    """Clean + apply skip policy. Raises ``SkipChunk`` for empty / non-English.

    ``SkipChunk.args[0]`` is a machine reason (``"empty"`` / ``"non_english"``).
    """
    cleaned = clean(text)
    if not cleaned:
        raise SkipChunk("empty")
    if non_english_policy != "read_anyway" and non_latin_ratio(cleaned) > 0.30:
        raise SkipChunk("non_english")
    return cleaned
