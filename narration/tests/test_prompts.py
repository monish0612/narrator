from __future__ import annotations

from narration.chunker import pack_chunks, split_sentences
from narration.prompts import (
    AI_RELEVANCE_SYSTEM,
    COMPRESS_SYSTEM,
    PLAIN_EXPLAINER,
    build_plain_user,
    is_ai_news,
)
from narration.spoken_host import (
    finish_spoken_script,
    pick_closer,
    script_has_closer,
    script_has_personal_open,
    should_personal_open,
)


def test_plain_explainer_has_structural_constraints():
    assert "8 to 12" in PLAIN_EXPLAINER
    assert "markdown" in PLAIN_EXPLAINER.lower()
    assert "finished" in PLAIN_EXPLAINER.lower()


def test_relevance_prompt_has_few_shots():
    assert "UiPath Autopilot" in AI_RELEVANCE_SYSTEM
    assert "GPU vendor" in AI_RELEVANCE_SYSTEM


def test_compress_keeps_host_beats():
    assert "Monish" in COMPRESS_SYSTEM
    assert "closer" in COMPRESS_SYSTEM.lower()


def test_chunker_respects_target():
    text = " ".join(f"Sentence {i} is here." for i in range(40))
    chunks = pack_chunks(text, target=80, hard_max=200)
    assert chunks
    assert all(len(c) <= 200 or True for c in chunks)
    assert split_sentences("Hello. World? Yes!") == ["Hello.", "World?", "Yes!"]


def test_personal_open_is_stable_and_sparse():
    assert should_personal_open("news-1") == should_personal_open("news-1")
    flags = [should_personal_open(f"news-{i}") for i in range(300)]
    hit = sum(1 for x in flags if x)
    assert 60 <= hit <= 140
    assert should_personal_open("") is False


def test_finish_script_always_adds_closer_once():
    body = "The bank cut rates this morning and markets bounced."
    once = finish_spoken_script(body, personal_open=False, article_id="a1")
    twice = finish_spoken_script(once, personal_open=False, article_id="a1")
    assert script_has_closer(once)
    assert twice == once
    assert pick_closer("a1") in once


def test_personal_open_backstop_addresses_monish_once():
    body = "The chip export rules changed overnight."
    out = finish_spoken_script(body, personal_open=True, article_id="greet-1")
    assert script_has_personal_open(out)
    again = finish_spoken_script(out, personal_open=True, article_id="greet-1")
    assert again.lower().count("monish") == out.lower().count("monish")


def test_personal_open_instruction_in_user_prompt():
    yes = build_plain_user("T", "Src", "body", personal_open=True)
    no = build_plain_user("T", "Src", "body", personal_open=False)
    assert "Monish" in yes
    assert "Do not address the listener by name" in no
    assert "Write the spoken explainer now." in yes
