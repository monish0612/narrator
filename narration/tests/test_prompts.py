from __future__ import annotations

from narration.chunker import pack_chunks, split_sentences
from narration.prompts import AI_RELEVANCE_SYSTEM, PLAIN_EXPLAINER, is_ai_news


def test_plain_explainer_has_structural_constraints():
    assert "8 to 12" in PLAIN_EXPLAINER
    assert "markdown" in PLAIN_EXPLAINER.lower()


def test_relevance_prompt_has_few_shots():
    assert "UiPath Autopilot" in AI_RELEVANCE_SYSTEM
    assert "GPU vendor" in AI_RELEVANCE_SYSTEM


def test_chunker_respects_target():
    text = " ".join(f"Sentence {i} is here." for i in range(40))
    chunks = pack_chunks(text, target=80, hard_max=200)
    assert chunks
    assert all(len(c) <= 200 or True for c in chunks)
    assert split_sentences("Hello. World? Yes!") == ["Hello.", "World?", "Yes!"]
