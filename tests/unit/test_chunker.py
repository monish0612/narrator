from __future__ import annotations

import hashlib

from narrator.core.models import NarrationScript, Segment
from narrator.pipeline.chunker import SentenceChunker
from narrator.pipeline.sanitize import clean


def _script(*texts_and_pauses) -> NarrationScript:
    segs = [Segment(text=t, pause_ms_after=p) for t, p in texts_and_pauses]
    return NarrationScript(segments=segs)


def test_no_chunk_exceeds_hard_max():
    chunker = SentenceChunker(target_chars=1200, hard_max=2000)
    big = " ".join(f"This is sentence number {i}." for i in range(400))
    chunks = chunker.split(_script((big, 0)))
    assert chunks
    assert all(len(c.text) <= 2000 for c in chunks)


def test_no_mid_sentence_splits():
    chunker = SentenceChunker(target_chars=200, hard_max=400)
    text = " ".join(f"Sentence number {i}." for i in range(80))
    chunks = chunker.split(_script((text, 0)))
    assert len(chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_monster_sentence_clause_splits():
    chunker = SentenceChunker(target_chars=1200, hard_max=2000)
    monster = ("this is a distinct clause, " * 400) + "the end."
    assert len(monster) > 6000
    chunks = chunker.split(_script((monster, 0)))
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 for c in chunks)


def test_pause_hints_on_last_chunk_of_segment():
    chunker = SentenceChunker(target_chars=1200, hard_max=2000)
    chunks = chunker.split(_script(("First paragraph.", 600), ("Second paragraph.", 900)))
    assert chunks[0].pause_ms_after == 600
    assert chunks[1].pause_ms_after == 900


def test_deterministic_and_sanitized_hash():
    chunker = SentenceChunker(target_chars=300, hard_max=600)
    script = _script(("Alpha beta gamma. Delta epsilon.", 500))
    first = chunker.split(script)
    second = chunker.split(script)
    assert [c.sha256 for c in first] == [c.sha256 for c in second]
    for c in first:
        assert c.sha256 == hashlib.sha256(clean(c.text).encode()).hexdigest()


def test_indices_are_sequential():
    chunker = SentenceChunker(target_chars=100, hard_max=200)
    text = " ".join(f"S{i} is here." for i in range(30))
    chunks = chunker.split(_script((text, 0)))
    assert [c.idx for c in chunks] == list(range(1, len(chunks) + 1))
