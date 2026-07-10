from __future__ import annotations

import pytest

from narrator.core.errors import SkipChunk
from narrator.pipeline.sanitize import clean, non_latin_ratio, sanitize_chunk


def test_url_becomes_speakable():
    out = clean("Visit https://example.com/page?x=1 now")
    assert "https" not in out
    assert "link" in out


def test_email_becomes_speakable():
    out = clean("Mail me at john.doe@example.co.uk please")
    assert "@" not in out
    assert "email address" in out


def test_long_token_broken():
    blob = "A" * 500  # e.g. a base64 blob
    out = clean(f"data {blob} end")
    assert all(len(tok) <= 40 for tok in out.split(" "))


def test_emoji_dropped():
    out = clean("great job \U0001f600\U0001f44d done")
    assert "\U0001f600" not in out and "\U0001f44d" not in out
    assert "great job" in out


def test_repeat_collapse():
    assert clean("soooooo good") == "sooo good"


def test_clean_is_deterministic():
    text = "Hello   WORLD \n\n visit www.x.com \U0001f600"
    assert clean(text) == clean(text)


def test_empty_after_clean_raises_skip():
    with pytest.raises(SkipChunk) as ei:
        sanitize_chunk("\U0001f600\U0001f600  \n\t")
    assert ei.value.args[0] == "empty"


def test_hebrew_default_policy_skips():
    hebrew = "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd \u05de\u05d4 \u05e9\u05dc\u05d5\u05de\u05da"
    assert non_latin_ratio(hebrew) > 0.3
    with pytest.raises(SkipChunk) as ei:
        sanitize_chunk(hebrew)
    assert ei.value.args[0] == "non_english"


def test_hebrew_read_anyway():
    hebrew = "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"
    out = sanitize_chunk(hebrew, non_english_policy="read_anyway")
    assert out  # returned, not skipped
