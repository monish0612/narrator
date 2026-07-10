from __future__ import annotations

import pytest
from pydantic import ValidationError

from narrator.core.models import JobParams, JobStatus


def test_defaults():
    p = JobParams()
    assert p.mode == "explainer"
    assert p.voice == "af_heart"
    assert p.speed == 1.0
    assert p.output_format == "mp3"
    assert p.explainer_style == "news"


@pytest.mark.parametrize("speed", [0.4, 2.1, -1.0, 10.0])
def test_bad_speed_rejected(speed):
    with pytest.raises(ValidationError):
        JobParams(speed=speed)


def test_bad_mode_rejected():
    with pytest.raises(ValidationError):
        JobParams(mode="sing")


def test_bad_output_format_rejected():
    with pytest.raises(ValidationError):
        JobParams(output_format="flac")


def test_bad_explainer_style_rejected():
    with pytest.raises(ValidationError):
        JobParams(explainer_style="rap")


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        JobParams(unexpected="x")


def test_webhook_shape():
    with pytest.raises(ValidationError):
        JobParams(webhook_url="ftp://example.com/hook")
    assert JobParams(webhook_url="https://ok/hook").webhook_url == "https://ok/hook"


def test_status_terminality():
    assert JobStatus.COMPLETED.is_terminal
    assert JobStatus.UPLOAD_PENDING.is_terminal
    assert JobStatus.FAILED.is_terminal
    assert JobStatus.CANCELLED.is_terminal
    assert not JobStatus.SYNTHESIZING.is_terminal
    assert JobStatus.QUEUED.is_active
    assert not JobStatus.COMPLETED.is_active
