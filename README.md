# Narrator v2

Long-form text-to-speech narration platform for a single box (Hostinger KVM 2
→ Coolify → Traefik). Turns `.txt` / `.md` / `.pdf` (or raw text) into a single
narrated audio file, either **verbatim** or as a Gemini-generated **explainer**.
Kokoro-82M int8 ONNX does inference in-repo; delivery is local disk now and
Google Drive when you enable it.

`ARCHITECTURE.md` is the source of truth. `CURSOR_PROMPT.md` tracks the phased
build. `.cursor/rules/narrator.mdc` holds the invariants.

## Capabilities

- **Two processing modes** — `verbatim` (read the document) and `explainer`
  (Gemini map-reduce summary tuned to a target minutes length, with grounding +
  safety-block fallback to verbatim).
- **Resilient marathon synthesis** — content-addressed cache, manifest-based
  resume, per-chunk retry, Redis circuit breakers, and a fairness gate so a
  short "fast-lane" job preempts a long "bulk-lane" one.
- **Graceful deploys** — workers trap SIGTERM, flush the manifest at a chunk
  boundary, and resume mid-job on the new version. Zero work lost across
  releases.
- **Delivery that never loses audio** — a terminal upload failure parks the job
  in `UPLOAD_PENDING` with the finished file still downloadable.
- **Operable** — keyed REST + SSE, `/healthz`, Prometheus `/metrics`, structured
  JSON logs, hard `mem_limit`s and low `cpu_shares` so your website always wins.

## Architecture at a glance

```
client ──X-API-Key──▶ api ──enqueue──▶ redis ──▶ worker-fast / worker-bulk
                       │                              │
                       │                        ingest→process→synth→assemble→deliver
                       ▼                              │
                  SQLite (truth)                      ▼
                  + Redis mirror                 tts (Kokoro int8 ONNX)
```

Five services (`api`, `worker-fast`, `worker-bulk`, `tts`, `redis`). Only `api`
is published. See `ARCHITECTURE.md` §2, §13.

## Quickstart (local dev)

```bash
uv sync
cp .env.example .env          # set API_KEYS (openssl rand -hex 24)
uv run ruff check .
uv run pytest -q              # 124 passed, chaos + ffmpeg e2e skipped by default
```

Full stack with published ports (`api` :8000, `tts` :8880, `redis` :6379):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

First `tts` boot downloads ~120 MB of Kokoro weights (sha256-verified) to the
`tts-models` volume; its healthcheck (`start_period 90s`) only goes green after
a real warmup synthesis, so no request ever hits a cold model.

## API

All routes except `/healthz`, `/readyz`, `/metrics` require `X-API-Key`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/jobs` | Create a job (JSON `{"text": ...}` or multipart `file=`). Returns `202 {job_id, lane, status}`. |
| GET | `/v1/jobs` | List jobs (`?limit=&offset=`). |
| GET | `/v1/jobs/{id}` | Job status + result. |
| GET | `/v1/jobs/{id}/events` | SSE stream of status transitions + progress. |
| POST | `/v1/jobs/{id}/cancel` | Cooperative cancel. |
| POST | `/v1/jobs/{id}/retry` | Re-queue a `FAILED` / `UPLOAD_PENDING` job (resumes via manifest). |
| GET | `/v1/jobs/{id}/download` | Stream the local artifact, or `{url}` for Drive. |
| GET | `/v1/voices` | Available TTS voices. |
| GET | `/healthz` / `/readyz` | Liveness / readiness. |
| GET | `/metrics` | Prometheus text. |

**Job parameters** (JSON body keys or multipart form fields):

| Field | Default | Notes |
|---|---|---|
| `text` | — | required for JSON; use `file` for multipart uploads |
| `mode` | `explainer` | `explainer` \| `verbatim` |
| `voice` | `af_heart` | single voice or a blend like `af_heart:0.6,af_bella:0.4` |
| `speed` | `1.0` | 0.5–2.0 |
| `title` | `null` | ≤300 chars, used in the output filename/metadata |
| `output_format` | `mp3` | `mp3` \| `wav` \| `opus` |
| `webhook_url` | `null` | http(s); terminal states POST an HMAC-SHA256 signed payload |

Send `Idempotency-Key` to make retried submissions return the same job.

### Example

```bash
KEY=your-api-key
HOST=https://narrator.example.com

# submit
curl -sX POST "$HOST/v1/jobs" -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Long document text...","mode":"explainer","title":"My Doc"}'

# upload a PDF instead
curl -sX POST "$HOST/v1/jobs" -H "X-API-Key: $KEY" \
  -F file=@paper.pdf -F mode=verbatim

# poll + download
curl -s "$HOST/v1/jobs/$JOB_ID" -H "X-API-Key: $KEY"
curl -s "$HOST/v1/jobs/$JOB_ID/download" -H "X-API-Key: $KEY" -o out.mp3
```

## Deploy on Coolify (guided)

1. **Create the GitHub repo** `monish0612/narrator` (empty), then push — the
   remote is already configured:

   ```bash
   git push -u origin main
   ```

2. **Coolify → + New → Docker Compose** → pick your server + the `narrator`
   repo + branch `main` → compose file `docker-compose.yml`. Name it `narrator`.

3. **Env tab** (mark secrets 🔒):

   | Var | Value |
   |---|---|
   | `API_KEYS` 🔒 | `openssl rand -hex 24` (comma-separate for multiple) |
   | `STORAGE_BACKEND` | `local` |
   | `GEMINI_API_KEY` 🔒 | your key (needed for explainer mode) |
   | `GEMINI_MODEL` | `gemini-2.5-flash` (optional) |
   | `ONNX_INTRA_OP` / `SYNTH_CONCURRENCY` | leave default; tune with `bench.py` |

   Leave `GDRIVE_*` unset — Drive stays dormant.

4. **Domain**: expose **only** `api` (port 8000). Start with the auto
   `sslip.io` URL; optionally add `narrator.<yourdomain>` later (A record →
   your VPS IP; Traefik issues SSL). Keep `tts`/`redis`/workers unpublished.
   **Persistent volumes**: `narrator-data`, `tts-models`, `redis-data`.

5. **Deploy.** Watch the `tts` healthcheck reach healthy (first boot downloads
   the model, ≤ ~90s + download time).

6. **Verify:**

   ```bash
   curl -fsS https://<host>/healthz
   curl -H "X-API-Key: $KEY" https://<host>/v1/voices     # lists af_heart
   HOST=https://<host> API_KEY=$KEY ./scripts/smoke_test.sh
   ```

7. **Tune** once, over SSH: `python scripts/bench.py --base-url http://tts:8880/v1`,
   apply the recommended `ONNX_INTRA_OP` / `SYNTH_CONCURRENCY` in the env tab,
   redeploy. Enable the deploy webhook for push-to-deploy.

## Enable Google Drive delivery (later)

```bash
python scripts/gdrive_auth.py    # one-time OAuth; prints GDRIVE_* env values
```

Set `STORAGE_BACKEND=gdrive` + `GDRIVE_CLIENT_ID/SECRET/REFRESH_TOKEN` in
Coolify and redeploy. A `401 invalid_grant` (expired token) parks jobs in
`UPLOAD_PENDING` with the audio still downloadable — rotate the token and
`POST /v1/jobs/{id}/retry`.

## Operations runbook

- **Logs**: structured JSON on stdout (per-stage at INFO, per-chunk at DEBUG).
  Rely on Docker/Coolify rotation.
- **Health**: `api` `/healthz` (used by Coolify), `tts` `/health` (warmup gate).
  A bad env (missing key, wrong `TTS_BASE_URL`) fails config validation at boot
  → healthcheck stays red → Coolify keeps the previous version serving.
- **Stuck job?** Check `GET /v1/jobs/{id}` → `stall_reason`. `STALLED` means a
  dependency breaker is open (TTS/Gemini down); it auto-resumes. `FAILED` and
  `UPLOAD_PENDING` are operator-retryable via `/retry`.
- **State & recovery**: SQLite WAL is the source of truth (`/data`, backed up
  daily to `/data/backups`). Redis is a mirror; on worker start `reconcile()`
  re-enqueues orphans. Redis runs `appendonly` + `noeviction`.
- **Resource pressure**: caps ≈ 2.7 G total (tts 1024M, worker-bulk 640M,
  worker-fast 384M, api 512M, redis 192M) with low `cpu_shares`. If the box
  gets tight, lower `SYNTH_CONCURRENCY`, or scale per `ARCHITECTURE.md` §16.
- **Retention**: terminal jobs older than `RETENTION_DAYS` (default 7) are
  swept nightly; the synthesis cache evicts LRU at `CACHE_MAX_GB`.

## Testing

```bash
uv run ruff check .
uv run pytest -q                                  # unit + integration
uv run pytest -q -m "not chaos"                   # explicit skip of chaos

# chaos suite (needs a live compose stack + Docker):
NARRATOR_CHAOS=1 NARRATOR_BASE_URL=http://localhost:8000 NARRATOR_API_KEY=... \
  uv run pytest -m chaos tests/chaos -v
```

The `tests/chaos` suite injects five faults (TTS outage, Redis restart, worker
SIGTERM deploy, worker hard-kill, delivery failure) and asserts the platform
converges to a correct terminal state.

## Configuration

All tunables live in `core/config.py` with fail-fast boot validation. See
`.env.example` for the full surface. Never commit real secrets — production
values live in the Coolify env tab.
