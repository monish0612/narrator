"""Verbatim processor (ARCHITECTURE.md sections 5 / 1).

Reads the document as-is: paragraph -> segment, with pause hints (600 ms between
paragraphs, 900 ms after headings/section ends) and light abbreviation / symbol
expansion so numbers and abbreviations read naturally.
"""

from __future__ import annotations

import re

from narrator.core.models import Document, NarrationScript, Segment
from narrator.core.protocols import JobContext

_PARA_PAUSE_MS = 600
_SECTION_PAUSE_MS = 900

# Order matters: multi-char first.
_ABBREV = {
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\betc\.": "et cetera",
    r"\bvs\.": "versus",
    r"\bDr\.": "Doctor",
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miss",
    r"\bProf\.": "Professor",
    r"\bSt\.": "Saint",
    r"\bNo\.\s*(?=\d)": "number ",
    r"\bvol\.": "volume",
    r"\bp\.\s*(?=\d)": "page ",
}


def _expand(text: str) -> str:
    for pattern, repl in _ABBREV.items():
        text = re.sub(pattern, repl, text)
    text = text.replace("&", " and ")
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _looks_like_heading(para: str) -> bool:
    if len(para) <= 60 and not para.endswith((".", "!", "?")):
        return True
    return para.endswith(":")


class VerbatimProcessor:
    """Implements the ``ContentProcessor`` protocol."""

    async def process(self, doc: Document, ctx: JobContext) -> NarrationScript:
        paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
        segments: list[Segment] = []
        for para in paragraphs:
            expanded = _expand(para)
            if not expanded:
                continue
            pause = _SECTION_PAUSE_MS if _looks_like_heading(para) else _PARA_PAUSE_MS
            segments.append(Segment(text=expanded, pause_ms_after=pause))
        if segments:
            segments[-1].pause_ms_after = 0
        return NarrationScript(segments=segments, warnings=list(ctx.warnings))
