"""Chaos suite gating.

These tests drive a *running* Narrator compose stack and inject faults via the
Docker CLI (stop/kill/restart services), asserting the system converges to a
correct terminal state. They are deselected by default (marker ``chaos``) and
additionally skipped unless explicitly enabled, because they need Docker + a
live deployment - typically run on the box, not on a dev laptop.

Enable:
    NARRATOR_CHAOS=1 \
    NARRATOR_BASE_URL=http://localhost:8000 \
    NARRATOR_API_KEY=... \
    uv run pytest -m chaos tests/chaos -v
"""

from __future__ import annotations

import os
import shutil

import pytest


def pytest_collection_modifyitems(config, items):
    if os.getenv("NARRATOR_CHAOS") == "1":
        return
    skip = pytest.mark.skip(reason="chaos suite disabled (set NARRATOR_CHAOS=1 + a live stack)")
    for item in items:
        if "chaos" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("NARRATOR_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.getenv("NARRATOR_API_KEY")
    if not key:
        pytest.skip("NARRATOR_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def compose_cmd() -> list[str]:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    raw = os.getenv(
        "NARRATOR_COMPOSE",
        "docker compose -f docker-compose.yml -f docker-compose.dev.yml",
    )
    return raw.split()
