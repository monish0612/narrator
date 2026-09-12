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
