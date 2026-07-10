# Narrator worker - one process per lane (ARCHITECTURE.md sections 8, 13)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    DATA_DIR=/data \
    WORKER_LANE=fast

# ffmpeg + ffprobe for assembly; curl for optional debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

# arq consumes the lane queue selected by WORKER_LANE (fast|bulk).
CMD ["arq", "narrator.worker.main.WorkerSettings"]
