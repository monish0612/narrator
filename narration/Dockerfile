FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2" \
    "pydantic-settings>=2.4" \
    "arq>=0.26" \
    "redis>=5" \
    "httpx>=0.27" \
    "structlog>=24.1" \
    "tenacity>=9.0"

COPY narration/src /app/src
COPY narration/start.sh /app/start.sh

RUN mkdir -p /data \
    && sed -i 's/\r$//' /app/start.sh \
    && chmod +x /app/start.sh

EXPOSE 8870

# /data is a Coolify volume (root-owned). Stay root so writes succeed.
HEALTHCHECK --interval=30s --timeout=8s --retries=8 --start-period=40s \
    CMD curl -fsS http://127.0.0.1:8870/healthz || exit 1

CMD ["sh", "-c", "mkdir -p /data && exec /app/start.sh"]
