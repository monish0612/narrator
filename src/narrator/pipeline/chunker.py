"""Sentence-safe chunker (ARCHITECTURE.md sections 5 / 9).

Packs whole sentences to ``CHUNK_TARGET_CHARS`` without ever exceeding
``CHUNK_HARD_MAX``. Sentences longer than the hard max are clause-split at
``; : , -`` (em/en dash), then hard-wrapped as a last resort. Pause hints ride
on the last chunk of each segment. sha256 is computed over the *sanitized* text
so resume/cache stay stable.
"""

from __future__ import annotations

import hashlib
import re

from narrator.core.config import settings
from narrator.core.models import Chunk, NarrationScript
from narrator.pipeline.sanitize import clean

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[;:,\u2014\u2013\-])\s+")


def _sanitized_hash(text: str) -> str:
    return hashlib.sha256(clean(text).encode("utf-8")).hexdigest()


class SentenceChunker:
    """Implements the ``Chunker`` protocol."""

    def __init__(self, target_chars: int | None = None, hard_max: int | None = None) -> None:
        self.target = target_chars or settings.chunk_target_chars
        self.hard_max = hard_max or settings.chunk_hard_max

    # --- public --------------------------------------------------------------
    def split(self, script: NarrationScript) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 1
        for seg in script.segments:
            texts = self._chunk_segment(seg.text)
            for i, text in enumerate(texts):
                pause = seg.pause_ms_after if i == len(texts) - 1 else 0
                chunks.append(
                    Chunk(idx=idx, text=text, sha256=_sanitized_hash(text), pause_ms_after=pause)
                )
                idx += 1
        return chunks

    # --- internals -----------------------------------------------------------
    def _chunk_segment(self, text: str) -> list[str]:
        sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
        units: list[str] = []
        for sentence in sentences:
            if len(sentence) <= self.hard_max:
                units.append(sentence.strip())
            else:
                units.extend(self._split_monster(sentence))
        return self._pack(units)

    def _split_monster(self, sentence: str) -> list[str]:
        out: list[str] = []
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= self.hard_max:
                out.append(clause)
            else:
                out.extend(self._hard_wrap(clause))
        return out

    def _hard_wrap(self, text: str) -> list[str]:
        out: list[str] = []
        buf = ""
        for word in text.split(" "):
            while len(word) > self.hard_max:
                out.append(word[: self.hard_max])
                word = word[self.hard_max :]
            candidate = f"{buf} {word}".strip()
            if len(candidate) > self.hard_max:
                if buf:
                    out.append(buf)
                buf = word
            else:
                buf = candidate
        if buf:
            out.append(buf)
        return out

    def _pack(self, units: list[str]) -> list[str]:
        out: list[str] = []
        buf = ""
        for unit in units:
            if not buf:
                buf = unit
                continue
            if len(buf) + 1 + len(unit) > self.target:
                out.append(buf)
                buf = unit
            else:
                buf = f"{buf} {unit}"
        if buf:
            out.append(buf)
        return out
