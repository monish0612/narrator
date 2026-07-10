"""Five fault-injection scenarios (ARCHITECTURE.md sections 14-15).

Each scenario submits real work, breaks a dependency at a controlled moment,
and asserts the platform *converges* to the correct terminal state with the
audio artifact intact - never silent data loss, never a wedged job.
"""

from __future__ import annotations

import subprocess
import time

import httpx
import pytest

pytestmark = pytest.mark.chaos

# A document big enough that synthesis spans several seconds, giving us a window
# to inject a fault mid-flight.
LONG_TEXT = (
    "Chaos engineering validates that a system withstands turbulent conditions. "
    "This narration is intentionally long so synthesis runs long enough to break "
    "a dependency mid-flight and observe recovery. "
) * 40


# --- helpers -----------------------------------------------------------------
def _submit(base_url: str, api_key: str, *, mode: str = "verbatim") -> str:
    r = httpx.post(
        f"{base_url}/v1/jobs",
        headers={"X-API-Key": api_key},
        json={"text": LONG_TEXT, "mode": mode},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["job_id"]


def _status(base_url: str, api_key: str, job_id: str) -> dict:
    r = httpx.get(
        f"{base_url}/v1/jobs/{job_id}",
        headers={"X-API-Key": api_key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _wait_status(base_url, api_key, job_id, targets: set[str], timeout: float) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        last = _status(base_url, api_key, job_id)["status"]
        if last in targets:
            return last
        time.sleep(2)
    raise AssertionError(f"job {job_id} stuck at {last!r}, wanted {targets}")


def _compose(compose_cmd: list[str], *args: str) -> None:
    subprocess.run([*compose_cmd, *args], check=True, timeout=180)


def _download_ok(base_url, api_key, job_id) -> bool:
    r = httpx.get(
        f"{base_url}/v1/jobs/{job_id}/download",
        headers={"X-API-Key": api_key},
        timeout=60,
    )
    return r.status_code == 200 and len(r.content) > 1024


# --- scenarios ---------------------------------------------------------------
def test_tts_outage_then_recovery(base_url, api_key, compose_cmd):
    """TTS dies mid-synthesis -> job stalls (not fails) -> TTS back -> COMPLETED."""
    job_id = _submit(base_url, api_key)
    _wait_status(base_url, api_key, job_id, {"SYNTHESIZING"}, timeout=60)
    _compose(compose_cmd, "stop", "tts")
    try:
        # Breaker opens; job stalls rather than failing.
        _wait_status(base_url, api_key, job_id, {"STALLED", "SYNTHESIZING"}, timeout=90)
    finally:
        _compose(compose_cmd, "start", "tts")
    assert _wait_status(base_url, api_key, job_id, {"COMPLETED"}, timeout=600) == "COMPLETED"
    assert _download_ok(base_url, api_key, job_id)


def test_redis_restart_reconciles(base_url, api_key, compose_cmd):
    """Redis bounces mid-job -> AOF + startup reconcile re-enqueue -> COMPLETED."""
    job_id = _submit(base_url, api_key)
    _wait_status(base_url, api_key, job_id, {"SYNTHESIZING"}, timeout=60)
    _compose(compose_cmd, "restart", "redis")
    assert _wait_status(base_url, api_key, job_id, {"COMPLETED"}, timeout=600) == "COMPLETED"
    assert _download_ok(base_url, api_key, job_id)


def test_worker_deploy_sigterm_resume(base_url, api_key, compose_cmd):
    """Graceful deploy: SIGTERM at a chunk boundary -> manifest resume -> COMPLETED."""
    job_id = _submit(base_url, api_key)
    _wait_status(base_url, api_key, job_id, {"SYNTHESIZING"}, timeout=60)
    # `restart` sends SIGTERM (honoring stop_grace_period) then starts fresh code.
    _compose(compose_cmd, "restart", "worker-bulk", "worker-fast")
    assert _wait_status(base_url, api_key, job_id, {"COMPLETED"}, timeout=600) == "COMPLETED"
    assert _download_ok(base_url, api_key, job_id)


def test_worker_hard_kill_resume(base_url, api_key, compose_cmd):
    """Container OOM/hard kill -> restart policy -> manifest resume -> COMPLETED."""
    job_id = _submit(base_url, api_key)
    _wait_status(base_url, api_key, job_id, {"SYNTHESIZING"}, timeout=60)
    _compose(compose_cmd, "kill", "worker-bulk", "worker-fast")
    _compose(compose_cmd, "start", "worker-bulk", "worker-fast")
    assert _wait_status(base_url, api_key, job_id, {"COMPLETED"}, timeout=600) == "COMPLETED"
    assert _download_ok(base_url, api_key, job_id)


def test_delivery_failure_upload_pending(base_url, api_key, compose_cmd):
    """Terminal delivery error -> UPLOAD_PENDING with a downloadable local artifact.

    Only meaningful when the stack runs with STORAGE_BACKEND=gdrive and an
    invalid/expired token (401 invalid_grant). With local storage there is no
    delivery step, so the scenario is skipped.
    """
    import os

    if os.getenv("NARRATOR_STORAGE_BACKEND", "local") != "gdrive":
        pytest.skip("set NARRATOR_STORAGE_BACKEND=gdrive (with a bad token) to run this scenario")
    job_id = _submit(base_url, api_key)
    status = _wait_status(base_url, api_key, job_id, {"UPLOAD_PENDING", "COMPLETED"}, timeout=600)
    assert status == "UPLOAD_PENDING"
    # Audio is safe and still downloadable despite the failed upload.
    assert _download_ok(base_url, api_key, job_id)
