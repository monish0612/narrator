# Narrator API (ARCHITECTURE.md sections 7, 13)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    DATA_DIR=/data

# curl for the Coolify healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv

# Install runtime deps first (cache-friendly), then the package itself.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

CMD ["uvicorn", "narrator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
