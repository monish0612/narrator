"""Gemini map-reduce explainer (ARCHITECTURE.md sections 6 / 15-D).

Rewrites a document into a spoken explainer script. Sections <= 30k chars on
paragraph boundaries; MAP -> strict JSON per section; REDUCE -> plain spoken text
with [pause:ms]. Every external call goes through the Gemini retry policy +
breaker. Safety block => verbatim fallback for that section. Grounding validator:
every number / proper-noun in the script must exist in the source pool - one
regeneration, then a warning.

The genai SDK is reached only via the injected ``GeminiClient`` so this module
(and its tests) never make live calls.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from narrator.core.breaker import CircuitBreaker
from narrator.core.errors import InvalidJSONError, SafetyBlockError
from narrator.core.logging import get_logger
from narrator.core.models import Document, NarrationScript, Segment
from narrator.core.protocols import JobContext
from narrator.core.retry import gemini_retrying
from narrator.processors import prompts
from narrator.processors.verbatim import VerbatimProcessor

log = get_logger(__name__)

_PAUSE_MARKER = re.compile(r"\[pause:(\d+)\]")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")

# Capitalized words that commonly begin sentences and are not proper nouns.
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Their", "They", "Then",
    "And", "But", "For", "Was", "Were", "Are", "You", "Your", "His", "Her",
    "Its", "Our", "When", "What", "Which", "While", "With", "From", "Here",
    "How", "Why", "Who", "Now", "Some", "Many", "Most", "Also", "After",
    "Before", "Because", "However", "Meanwhile", "Finally", "First", "Second",
    "Third", "In", "On", "At", "To", "Of", "As", "It", "He", "She", "We", "So",
}


class GeminiClient(Protocol):
    async def generate(self, prompt: str, *, json_mode: bool = False) -> str: ...


def _repair_json(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    start = s.find("{")
    if start > 0:
        s = s[start:]
    opens = s.count("{")
    closes = s.count("}")
    if closes < opens:
        s = s + "}" * (opens - closes)
    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


class GeminiExplainerProcessor:
    """Implements the ``ContentProcessor`` protocol via Gemini map-reduce."""

    def __init__(
        self,
        client: GeminiClient,
        breaker: CircuitBreaker,
        *,
        target_minutes: int = 30,
        section_max_chars: int = 30000,
    ) -> None:
        self._client = client
        self._breaker = breaker
        self._target_minutes = target_minutes
        self._section_max = section_max_chars
        self._verbatim = VerbatimProcessor()

    # --- public --------------------------------------------------------------
    async def process(self, doc: Document, ctx: JobContext) -> NarrationScript:
        sections = self.sectionize(doc.text)
        pool = self._source_pool(doc.text)

        maps: list[dict[str, Any]] = []
        fallback_segments: list[Segment] = []
        for section in sections:
            try:
                maps.append(await self._map_section(section))
            except SafetyBlockError:
                ctx.warn("explainer: a section was safety-blocked; used verbatim fallback")
                fallback_segments.extend(await self._verbatim_segments(section, ctx))
            except InvalidJSONError:
                ctx.warn("explainer: a section returned unparseable JSON; skipped")

        segments: list[Segment] = []
        if maps:
            style = ctx.params.explainer_style
            script_text = await self._reduce(style, maps)
            script_text = await self._ground_check(script_text, style, maps, pool, ctx)
            segments.extend(self._parse_script(script_text))
        segments.extend(fallback_segments)
        if segments:
            segments[-1].pause_ms_after = 0
        return NarrationScript(segments=segments, warnings=list(ctx.warnings))

    # --- sectioning ----------------------------------------------------------
    def sectionize(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")
        sections: list[str] = []
        buf = ""
        for para in paragraphs:
            while len(para) > self._section_max:
                # A single monster paragraph: hard-slice on the boundary.
                if buf:
                    sections.append(buf)
                    buf = ""
                sections.append(para[: self._section_max])
                para = para[self._section_max :]
            if not buf:
                buf = para
            elif len(buf) + 2 + len(para) > self._section_max:
                sections.append(buf)
                buf = para
            else:
                buf = f"{buf}\n\n{para}"
        if buf:
            sections.append(buf)
        return [s for s in sections if s.strip()]

    # --- gemini calls --------------------------------------------------------
    async def _guarded_generate(self, prompt: str, *, json_mode: bool = False) -> str:
        async def _call() -> str:
            return await self._client.generate(prompt, json_mode=json_mode)

        async with self._breaker.guard():
            return await gemini_retrying()(_call)

    async def _map_section(self, section: str) -> dict[str, Any]:
        prompt = prompts.build_map_prompt(section)
        raw = await self._guarded_generate(prompt, json_mode=True)
        parsed = self._parse_json(raw)
        if parsed is not None:
            return parsed
        # Repair failed: one re-ask, then give up (InvalidJSONError).
        raw2 = await self._guarded_generate(prompt + "\nReturn ONLY valid JSON.", json_mode=True)
        parsed = self._parse_json(raw2)
        if parsed is None:
            raise InvalidJSONError("could not parse map JSON after repair + re-ask")
        return parsed

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_repair_json(raw))
        except json.JSONDecodeError:
            return None

    async def _reduce(self, style: str, maps: list[dict[str, Any]]) -> str:
        prompt = prompts.build_reduce_prompt(style, maps, self._target_minutes)
        return await self._guarded_generate(prompt)

    async def _verbatim_segments(self, section: str, ctx: JobContext) -> list[Segment]:
        doc = Document(text=section, char_count=len(section), source_type="text")
        script = await self._verbatim.process(doc, ctx)
        return script.segments

    # --- grounding -----------------------------------------------------------
    def _source_pool(self, text: str) -> dict[str, set[str]]:
        numbers = {n.replace(",", "") for n in _NUMBER_RE.findall(text)}
        names = set(_PROPER_RE.findall(text))
        return {"numbers": numbers, "names": names}

    def _ungrounded_terms(self, script_text: str, pool: dict[str, set[str]]) -> list[str]:
        # Pause markers carry digits that are not content numbers.
        scan = _PAUSE_MARKER.sub(" ", script_text)
        bad: set[str] = set()
        for num in _NUMBER_RE.findall(scan):
            if num.replace(",", "") not in pool["numbers"]:
                bad.add(num)
        for name in _PROPER_RE.findall(scan):
            if name in _STOPWORDS:
                continue
            if name not in pool["names"]:
                bad.add(name)
        return sorted(bad)

    async def _ground_check(
        self,
        script_text: str,
        style: str,
        maps: list[dict[str, Any]],
        pool: dict[str, set[str]],
        ctx: JobContext,
    ) -> str:
        ungrounded = self._ungrounded_terms(script_text, pool)
        if not ungrounded:
            return script_text
        log.info("explainer.grounding_regen", terms=ungrounded[:10])
        prompt = prompts.build_regen_prompt(style, maps, self._target_minutes, ungrounded)
        regenerated = await self._guarded_generate(prompt)
        still = self._ungrounded_terms(regenerated, pool)
        if still:
            ctx.warn(f"explainer: unverified terms after regeneration: {still[:5]}")
        return regenerated

    # --- script parsing ------------------------------------------------------
    def _parse_script(self, text: str) -> list[Segment]:
        parts = _PAUSE_MARKER.split(text)
        segments: list[Segment] = []
        i = 0
        while i < len(parts):
            chunk = parts[i].strip()
            pause = int(parts[i + 1]) if i + 1 < len(parts) else 0
            if chunk:
                segments.append(Segment(text=chunk, pause_ms_after=pause))
            i += 2
        return segments
