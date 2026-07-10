# Narrator v2

Long-form text-to-speech narration platform. Single-box production architecture
(Hostinger KVM 2 - Coolify - Kokoro-82M int8 ONNX - Gemini explainer - Google
Drive delivery). `ARCHITECTURE.md` is the source of truth; `CURSOR_PROMPT.md`
tracks the phased build.

## Status

Under construction. See `ARCHITECTURE.md` for the full design and `.env.example`
for the configuration surface.

## Quickstart (local dev)

```bash
uv sync
cp .env.example .env          # set API_KEYS (openssl rand -hex 24)
uv run ruff check .
uv run pytest -q
```

Local stack (API + workers + tts + redis):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The full operator runbook, curl examples, env table, and deployment steps are
filled in during Phase 15.
