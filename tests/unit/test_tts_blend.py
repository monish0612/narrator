from __future__ import annotations

import numpy as np
import pytest

from tts.blend import (
    VoiceSpecError,
    blend_style_vector,
    is_blend,
    parse_voice_spec,
)


def test_single_voice():
    assert parse_voice_spec("af_heart") == [("af_heart", 1.0)]
    assert is_blend("af_heart") is False


def test_blend_spec():
    parsed = parse_voice_spec("af_heart(2)+af_bella(1)")
    assert parsed == [("af_heart", 2.0), ("af_bella", 1.0)]
    assert is_blend("af_heart(2)+af_bella(1)") is True


def test_implicit_weight_in_blend():
    assert parse_voice_spec("a+b") == [("a", 1.0), ("b", 1.0)]


@pytest.mark.parametrize("bad", ["", "   ", "af heart", "af_heart()", "a(-1)", "a(0)", "+", "a+"])
def test_malformed_specs(bad):
    with pytest.raises(VoiceSpecError):
        parse_voice_spec(bad)


def test_blend_vector_equal_weights():
    styles = {"a": np.array([0.0, 0.0]), "b": np.array([2.0, 2.0])}
    out = blend_style_vector(styles, "a(1)+b(1)")
    assert np.allclose(out, [1.0, 1.0])


def test_blend_vector_weighted():
    styles = {"a": np.array([0.0, 0.0]), "b": np.array([2.0, 2.0])}
    out = blend_style_vector(styles, "a(3)+b(1)")
    assert np.allclose(out, [0.5, 0.5])


def test_blend_vector_unknown_voice():
    with pytest.raises(VoiceSpecError):
        blend_style_vector({"a": np.array([1.0])}, "a+missing")
