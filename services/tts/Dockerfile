# Narrator TTS service - Kokoro int8 ONNX (ARCHITECTURE.md section 12)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/models

# espeak-ng for phonemization; curl for the healthcheck; libsndfile for soundfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime deps for the TTS service only (no torch).
RUN pip install \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2" \
    "numpy>=1.26" \
    "onnxruntime>=1.18" \
    "kokoro-onnx>=0.4.9" \
    "soundfile>=0.12"

COPY services/tts /app/tts

# Non-root user; /models is a mounted volume owned by root by default, so make
# it writable for the download step.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8880

# Download+verify models (idempotent), then serve.
CMD ["sh", "-c", "python -m tts.download_models && exec uvicorn tts.main:app --host 0.0.0.0 --port 8880"]
