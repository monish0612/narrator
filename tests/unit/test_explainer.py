from __future__ import annotations

import fakeredis
import pytest

from narrator.core import retry as retry_mod
from narrator.core.breaker import CircuitBreaker
from narrator.core.errors import GeminiError, SafetyBlockError
from narrator.core.models import Document, JobParams
from narrator.core.protocols import JobContext
from narrator.processors.explainer import GeminiExplainerProcessor


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(retry_mod._WaitRespectRetryAfter, "__call__", lambda self, rs: 0.0)


class FakeGemini:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, bool]] = []

    async def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self.calls.append((prompt, json_mode))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _breaker() -> CircuitBreaker:
    return CircuitBreaker(fakeredis.FakeAsyncRedis(), "gemini")


def _ctx() -> JobContext:
    return JobContext(job_id="jb_1", params=JobParams(mode="explainer", explainer_style="news"))


def _doc(text: str) -> Document:
    return Document(text=text, char_count=len(text), source_type="text")


async def test_sectionize_respects_bound():
    proc = GeminiExplainerProcessor(FakeGemini([]), _breaker(), section_max_chars=100)
    text = "\n\n".join(f"para{i} " + "x" * 50 for i in range(10))
    sections = proc.sectionize(text)
    assert len(sections) > 1
    assert all(len(s) <= 100 for s in sections)


async def test_monster_paragraph_hard_sliced():
    proc = GeminiExplainerProcessor(FakeGemini([]), _breaker(), section_max_chars=100)
    sections = proc.sectionize("y" * 350)
    assert all(len(s) <= 100 for s in sections)
    assert len(sections) == 4


async def test_429_is_retried_and_recovers():
    responses = [
        GeminiError("rate limited", retry_after=1),
        '{"key_points": ["kp"], "entities": [], "numbers": []}',
        "the summary goes here. [pause:600] and it wraps up.",
    ]
    client = FakeGemini(responses)
    proc = GeminiExplainerProcessor(client, _breaker())
    script = await proc.process(_doc("Some content about topics."), _ctx())
    assert len(client.calls) == 3  # 429 retried once, then reduce
    assert script.segments


async def test_truncated_json_repaired():
    responses = [
        '{"key_points": ["a", "b"], "entities": ["X"], "numbers": ["5"]',  # missing }
        "the summary goes here. [pause:400] more content follows.",
    ]
    client = FakeGemini(responses)
    proc = GeminiExplainerProcessor(client, _breaker())
    script = await proc.process(_doc("Content with a value of 5 units."), _ctx())
    assert len(client.calls) == 2  # repaired, no re-ask
    assert script.segments


async def test_safety_block_falls_back_to_verbatim():
    responses = [SafetyBlockError("blocked")]
    client = FakeGemini(responses)
    proc = GeminiExplainerProcessor(client, _breaker())
    ctx = _ctx()
    script = await proc.process(_doc("A perfectly innocuous paragraph of text."), ctx)
    assert len(client.calls) == 1  # no reduce, fell back
    assert script.segments  # verbatim fallback produced segments
    assert any("safety" in w.lower() for w in ctx.warnings)


async def test_grounding_regeneration_triggers():
    responses = [
        '{"key_points": ["kp"], "entities": [], "numbers": []}',
        "the result was 42 percent. [pause:300] that is the story.",  # 42 not in source
        "the result was notable. [pause:300] that is the story.",  # regenerated, grounded
    ]
    client = FakeGemini(responses)
    proc = GeminiExplainerProcessor(client, _breaker())
    ctx = _ctx()
    script = await proc.process(_doc("A document about outcomes and results."), ctx)
    assert len(client.calls) == 3  # map, reduce, regen
    assert not any("unverified" in w.lower() for w in ctx.warnings)
    assert script.segments


async def test_output_has_segments_and_pauses():
    responses = [
        '{"key_points": ["kp"], "entities": [], "numbers": []}',
        "intro beat here. [pause:600] body beat here. [pause:900] recap here.",
    ]
    proc = GeminiExplainerProcessor(FakeGemini(responses), _breaker())
    script = await proc.process(_doc("Some plain content."), _ctx())
    assert len(script.segments) == 3
    assert script.segments[0].pause_ms_after == 600
    assert script.segments[1].pause_ms_after == 900
    assert script.segments[-1].pause_ms_after == 0  # last pause zeroed
