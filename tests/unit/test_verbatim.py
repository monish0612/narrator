from __future__ import annotations

from narrator.core.models import Document, JobParams
from narrator.core.protocols import JobContext
from narrator.processors.verbatim import VerbatimProcessor


def _ctx() -> JobContext:
    return JobContext(job_id="jb", params=JobParams(mode="verbatim"))


async def test_paragraphs_become_segments():
    doc = Document(
        text="First heading:\n\nBody paragraph one is here and it is fairly long indeed.\n\nFinal paragraph.",
        char_count=1,
    )
    script = await VerbatimProcessor().process(doc, _ctx())
    assert len(script.segments) == 3
    assert script.segments[-1].pause_ms_after == 0  # last pause zeroed


async def test_heading_gets_longer_pause():
    doc = Document(text="Chapter One:\n\nThe body text goes here as a sentence.", char_count=1)
    script = await VerbatimProcessor().process(doc, _ctx())
    assert script.segments[0].pause_ms_after == 900  # heading
    # only two segments; last is zeroed


async def test_abbreviation_and_symbol_expansion():
    doc = Document(text="See e.g. the results, up 25% and Dr. Smith agrees.", char_count=1)
    script = await VerbatimProcessor().process(doc, _ctx())
    text = script.segments[0].text
    assert "for example" in text
    assert "25 percent" in text
    assert "Doctor" in text
