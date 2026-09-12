"""TTS FastAPI service (ARCHITECTURE.md section 12).

OpenAI-compatible speech endpoint plus voices/health/stats. The engine is held
on ``app.state.engine``; tests inject a fake before startup so the real ONNX
session is never constructed.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from tts.audio import float_to_wav_bytes, wav_duration_ms
from tts.engine import BlendUnsupported, KokoroEngine, UnspeakableInput


class SpeechRequest(BaseModel):
    model_config = {"extra": "ignore"}

    model: str = "kokoro-int8"
    input: str = Field(..., min_length=1)
    voice: str = "am_onyx"
    speed: float = Field(0.9, ge=0.5, le=2.0)
    response_format: str = "wav"


def build_engine_from_env() -> KokoroEngine:
    models_dir = Path(os.getenv("MODELS_DIR", "/models"))
    model_path = os.getenv("MODEL_PATH", str(models_dir / "kokoro-v1.0.int8.onnx"))
    voices_path = os.getenv("VOICES_PATH", str(models_dir / "voices-v1.0.bin"))
    intra_op = int(os.getenv("ONNX_INTRA_OP", "2"))
    return KokoroEngine(model_path, voices_path, intra_op=intra_op)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = getattr(app.state, "engine", None)
    if engine is None:
        import asyncio

        engine = build_engine_from_env()
        await asyncio.to_thread(engine.load)
        await engine.warmup()
        app.state.engine = engine
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Narrator TTS", version="2.0.0", lifespan=lifespan)
    app.state.engine = None

    def get_engine(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            raise RuntimeError("engine not initialized")
        return engine

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, engine=Depends(get_engine)) -> Response:
        if req.response_format not in ("wav",):
            return JSONResponse(
                status_code=400,
                content={"reason": "unsupported_format", "detail": req.response_format},
            )
        try:
            samples, sample_rate, _wall_ms = await engine.synthesize(
                req.input, voice=req.voice, speed=req.speed
            )
        except UnspeakableInput as exc:
            return JSONResponse(status_code=400, content={"reason": exc.reason})
        except BlendUnsupported as exc:
            return JSONResponse(status_code=400, content={"reason": "blend_unsupported", "detail": str(exc)})
        wav = float_to_wav_bytes(samples, sample_rate)
        duration_ms = wav_duration_ms(samples, sample_rate)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration-Ms": str(duration_ms),
                "X-Sample-Rate": str(sample_rate),
            },
        )

    @app.get("/v1/audio/voices")
    async def voices(engine=Depends(get_engine)) -> dict:
        return {"voices": list(engine.voices)}

    @app.get("/health")
    async def health(engine=Depends(get_engine)) -> Response:
        ready = bool(getattr(engine, "ready", False))
        payload = {"status": "ready" if ready else "warming_up", "ready": ready}
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/stats")
    async def stats(engine=Depends(get_engine)) -> dict:
        return dict(engine.stats())

    return app


app = create_app()
