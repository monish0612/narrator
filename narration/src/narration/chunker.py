"""Sentence-safe chunk packing for Kokoro (CPU, sequential)."""

from __future__ import annotations

import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text or "") if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def pack_playback_chunks(text: str) -> list[str]:
    """Chunk 0 is the first 1–2 sentences. Later chunks are 2–4 sentences."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks = [" ".join(sentences[:2])]
    rest = sentences[2:]
    i = 0
    while i < len(rest):
        take = 3 if len(rest) - i >= 3 else len(rest) - i
        if take > 4:
            take = 4
        chunks.append(" ".join(rest[i : i + take]))
        i += take
    return [c for c in chunks if c.strip()]


def pack_chunks(text: str, *, target: int = 800, hard_max: int = 1500) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    buf: list[str] = []
    n = 0
    for sent in sentences:
        extra = len(sent) + (1 if buf else 0)
        if buf and n + extra > target:
            chunks.append(" ".join(buf))
            buf = [sent]
            n = len(sent)
            continue
        if len(sent) > hard_max:
            if buf:
                chunks.append(" ".join(buf))
                buf = []
                n = 0
            for i in range(0, len(sent), hard_max):
                chunks.append(sent[i : i + hard_max])
            continue
        buf.append(sent)
        n += extra
    if buf:
        chunks.append(" ".join(buf))
    return chunks
