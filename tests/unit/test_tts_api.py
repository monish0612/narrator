from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from tts.engine import BlendUnsupported, UnspeakableInput
from tts.main import create_app


class FakeEngine:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.voices = ["af_bella", "af_heart"]

    async def synthesize(self, text, *, voice, speed, response_format="wav"):
        if text.strip() in {"###", "\u0000"}:
            raise UnspeakableInput("no phonemes produced")
        if "+" in voice:
            raise BlendUnsupported("blends off in this build")
        # 0.5 s of quiet tone at 24 kHz
        n = 12000
        samples = 0.01 * np.sin(np.linspace(0, 3.14, n)).astype(np.float32)
        return samples, 24000, 40

    def stats(self):
        return {"count": 3, "rolling_rtf": 4.2, "rss_bytes": 123456, "ready": self.ready}


def _client(ready: bool = True) -> TestClient:
    app = create_app()
    app.state.engine = FakeEngine(ready=ready)
    return TestClient(app)


def test_speech_happy_path():
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": "Hello world.", "voice": "af_heart"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert int(r.headers["X-Audio-Duration-Ms"]) > 0


def test_speech_empty_input_422():
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": ""})
    assert r.status_code == 422


def test_speech_bad_speed_422():
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": "hi", "speed": 9.0})
    assert r.status_code == 422


def test_speech_unspeakable_400():
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": "###"})
    assert r.status_code == 400
    assert r.json()["reason"] == "no phonemes produced"


def test_speech_blend_unsupported_400():
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "af_heart(2)+af_bella(1)"})
    assert r.status_code == 400
    assert r.json()["reason"] == "blend_unsupported"


def test_unsupported_format_400():
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": "hi", "response_format": "opus"})
    assert r.status_code == 400


def test_voices():
    with _client() as c:
        r = c.get("/v1/audio/voices")
    assert r.status_code == 200
    assert "af_heart" in r.json()["voices"]


def test_health_ready_and_warming():
    with _client(ready=True) as c:
        assert c.get("/health").status_code == 200
    with _client(ready=False) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert r.json()["ready"] is False


def test_stats():
    with _client() as c:
        r = c.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and body["rolling_rtf"] == 4.2
