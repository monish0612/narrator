from __future__ import annotations

import pytest
from pydantic import ValidationError

from narration.config import Settings


def test_rejects_redis_db0(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://default:x@redis:6379/0")
    with pytest.raises(ValidationError):
        Settings()


def test_accepts_redis_db1(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://default:x@redis:6379/1")
    s = Settings()
    assert s.redis_url.endswith("/1")


def test_accepts_narration_redis_url_alias(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("NARRATION_REDIS_URL", "redis://default:x@redis:6379/1")
    s = Settings()
    assert s.redis_url.endswith("/1")


def test_default_reaper_and_ttl_cover_168_hours(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://default:x@redis:6379/1")
    monkeypatch.delenv("REDIS_TTL_SECONDS", raising=False)
    monkeypatch.delenv("NARRATION_REAPER_AGE_HOURS", raising=False)
    monkeypatch.delenv("REAPER_AGE_HOURS", raising=False)
    s = Settings()
    assert s.reaper_age_hours == 168
    assert s.redis_ttl_seconds == 691200
    assert s.redis_ttl_seconds >= 168 * 3600


def test_reaper_hours_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://default:x@redis:6379/1")
    monkeypatch.setenv("NARRATION_REAPER_AGE_HOURS", "0")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("NARRATION_REAPER_AGE_HOURS", "721")
    with pytest.raises(ValidationError):
        Settings()
