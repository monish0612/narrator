# CURSOR MASTER PROMPT — Build "Narrator" v2 (single-box KVM 2 edition)

**How to use:** Empty Git repo containing `ARCHITECTURE.md` + `.cursor/rules/narrator.mdc`. Open in Cursor → Agent mode (Claude Sonnet 4.x, Max context) → paste everything below the line. Work phase by phase. **Never skip an acceptance gate.**

---

## ROLE

You are a principal engineer who has shipped 100+ queue-based media pipelines. Build **Narrator** exactly per `ARCHITECTURE.md` (read fully first — it is the source of truth and wins over this prompt on conflict). Your bias: every external call can fail, every process can die mid-write, and the design must make both boring.

## GROUND RULES (non-negotiable)

1. Python **3.12** · **uv** (`pyproject.toml` + `uv.lock`) · **ruff** · **pytest + pytest-asyncio + respx + fakeredis**.
2. Runtime deps: `fastapi uvicorn[standard] pydantic>=2 pydantic-settings arq redis>=5 httpx structlog aiosqlite python-multipart tenacity sse-starlette pymupdf google-api-python-client google-auth google-auth-oauthlib google-genai`. TTS service adds: `kokoro-onnx onnxruntime numpy soundfile` (+ apt `espeak-ng`). Worker image adds apt `ffmpeg`.
3. **Async everywhere**; blocking work via subprocess / `asyncio.to_thread` / aiosqlite. One shared `httpx.AsyncClient` per process.
4. **Memory discipline:** stream uploads in ≤1 MB blocks; audio only via ffmpeg on files; Drive uploads resumable from path. Nothing large in RAM, ever.
5. Pipeline/API import **only** `core/protocols.py` Protocols; concretes chosen in `core/factory.py` from settings. All tunables in `core/config.py` (ARCHITECTURE §18) — **fail-fast validation at boot** with explicit messages.
6. `manifest.json` (§6) is frozen; atomic writes (`.tmp`+`os.replace`); resume rule incl. RIFF/size validation of existing chunks.
7. All state transitions through `core/state.py` (SQLite truth + Redis mirror + `job:{id}:events`). structlog JSON with `job_id`/`stage`/`chunk_idx`; no `print`.
8. Retries/breakers only via `core/retry.py` + `core/breaker.py` — no ad-hoc `try/sleep` loops anywhere.
9. After **every phase**: `uv run ruff check . && uv run pytest -q` green + that phase's acceptance commands, or do not proceed.
10. No secrets in repo; keep `.env.example` complete with comments.

---

## PHASE 0 — Scaffold
**Build:** repo layout (§19); pyproject with deps above (+dev: ruff, pytest, pytest-asyncio, respx, fakeredis); `core/config.py` (all §18 vars, boot validation: required keys present when `STORAGE_BACKEND=gdrive`/mode explainer, URL shapes, numeric ranges); `core/logging.py`; `.gitignore`; `.env.example`; README stub.
**Accept:** `uv sync` ok · `uv run python -c "from narrator.core.config import settings; print(settings.chunk_target_chars)"` → `1200` · missing `API_KEYS` raises a clear one-line error · ruff clean.

## PHASE 1 — Models, Protocols, State
**Build:** `core/models.py` (`Document`, `NarrationScript`/`Segment(text,pause_ms_after,meta)`, `Chunk(idx,text,sha256,pause_ms_after)`, `JobParams` per §7 with validation, `Job`, `JobStatus` incl. **UPLOAD_PENDING**, `SynthesisResult`, `StoredFile`); `core/protocols.py` (§5 verbatim); `core/state.py` — SQLite WAL (`jobs`, `kv`, `cache_index` tables, auto-migrate, `busy_timeout=5000`) + Redis mirror + `publish_event` + cancel helpers + `reconcile()` (orphaned running jobs → re-enqueue by lane).
**Accept:** tests for transitions, UPLOAD_PENDING path, reconcile, params validation (bad speed/mode/format rejected).

## PHASE 2 — Resilience core
**Build:**
- `core/retry.py`: tenacity policy factories per dependency implementing the §14 matrix (retryable exception/status classifiers; Gemini honors `retry-after`; Drive classifies `invalid_grant`/`storageQuotaExceeded` as terminal-typed exceptions).
- `core/breaker.py`: Redis-backed circuit breaker (closed/open/half-open; open after 5 consecutive retry-exhausted failures; cooldown 60 s doubling to 10 min cap; half-open single probe; exposes state for `/metrics` and `stall_reason`).
- `core/gate.py`: synth fairness gate — `fast_enter/exit` maintain `synth:fast_pending`; `bulk_wait_turn()` parks while fast pending, emitting `stall_reason:"yielding_to_fast_lane"` events.
- `core/cache.py`: content-addressed TTS cache (§9): key builder, get (hard-link out), put, SQLite `cache_index` touch, `evict_to_limit(CACHE_MAX_GB)`.
**Accept:** unit tests (fakeredis): breaker opens/half-opens/closes on scripted failures; gate lets fast preempt a simulated bulk loop within one iteration; cache hit avoids the synth callable and eviction respects LRU order.

## PHASE 3 — TTS service (`services/tts/`)
**Build:** standalone FastAPI app per §12: `download_models.py` (resume + sha256 verify into `/models`, envs `MODEL_URL`, `VOICES_URL` defaulting to the official kokoro-onnx v1.0 release assets — int8 model + voices bin); `engine.py` (one `InferenceSession`: `intra_op=ONNX_INTRA_OP`, `inter_op=1`, sequential, `ORT_ENABLE_ALL`; internal `asyncio.Semaphore(1)`; warmup flag); `blend.py` (`af_heart(2)+af_bella(1)` parser → weighted-average style vector; **inspect the installed `kokoro_onnx` package for exact create/voice-vector signatures — the installed package is source of truth; if custom vectors are unsupported in the pinned version, single voices only + explicit 400 for blends**); `main.py` endpoints `POST /v1/audio/speech` (wav bytes via soundfile; 400 `{reason:"unspeakable"}` on empty phonemization), `GET /v1/audio/voices`, `GET /health` (`ready` only post-warmup), `GET /stats` (rolling RTF, count, RSS); `Dockerfile` (python:3.12-slim + espeak-ng + curl, non-root, entry runs download→uvicorn :8880).
**Accept:** `docker build services/tts && docker run` → `/health` ready after warmup · `curl /v1/audio/speech` with a sentence returns a playable wav · blend request either synthesizes or 400s per capability · unit tests for blend parser + request validation (session mocked).

## PHASE 4 — Engine adapter
**Build:** `engines/openai_compat.py::OpenAICompatEngine` — shared client (`base_url=TTS_BASE_URL`, timeout 300); `synthesize()` POST `/audio/speech`, parse wav duration from header; `list_voices()`; `health()` (root `/health`); wrapped with `core/retry.tts_policy` + TTS breaker.
**Accept:** respx tests — happy path · 500→retry→success · 400 raises typed `UnspeakableError` (no retry) · breaker opens after scripted exhaustion.

## PHASE 5 — Ingest, sanitizer, chunker
**Build:** `pipeline/ingest.py` (§8-ish: txt/md/pdf streamed; UTF-8 + detection fallback with replacement-ratio warning; normalization: NFC, de-hyphenation, header/footer/page-number strip; **422-typed errors**: zero-text PDF → "needs OCR", encrypted PDF); `pipeline/sanitize.py` (§9-1: control chars, repeat-collapse, URL/email → spoken tokens, >40-char token breaking, emoji strip, non-Latin ratio → `NON_ENGLISH_POLICY`, empty→`SkipChunk`); `pipeline/chunker.py::SentenceChunker` (sentence-safe packing 1200/2000, clause-split monsters at `; : , —`, pause hints carried, sha256 over **sanitized** text).
**Accept:** tests — no chunk > hard max · no mid-sentence splits · 6k-char sentence clause-splits · URL/base64 blob becomes speakable · Hebrew paragraph under default policy → SkipChunk+warning · scanned-pdf fixture → typed 422 · deterministic hashes.

## PHASE 6 — Processors
**Build:** `processors/verbatim.py` (paragraph segments, pause hints 600/900 ms, abbreviation/number-adjacent expansions); `processors/prompts.py` (MAP → strict JSON `{key_points,entities,numbers}`; REDUCE presets `news`(default)/`teacher`/`podcast` → plain spoken text + `[pause:ms]` only, verbalized numbers, hook→body→recap; VALIDATOR); `processors/explainer.py::GeminiExplainerProcessor` (`google-genai` async; sections ≤30k chars on boundaries; per-section retry via `core/retry.gemini_policy` + breaker; JSON repair then one re-ask; **safety-block ⇒ verbatim fallback for that section + warning**; grounding validator — every number/proper-noun in script ∈ source pool, one regeneration then `warnings[]`; length scales to `EXPLAINER_TARGET_MINUTES`); `core/factory.py`.
**Accept:** mocked-genai tests: sectioning bound · 429 honored · truncated-JSON repair path · safety fallback · grounding regen triggers · output has segments+pauses. **No live API calls in tests.**

## PHASE 7 — Synthesis stage
**Build:** `pipeline/synth.py` per §9: manifest load-or-create; resume validation (RIFF+size, sha256 match else re-synth); per-chunk flow sanitize→cache→gate check (bulk lane: `bulk_wait_turn()` between chunks)→engine→persist wav→cache put→atomic manifest→progress INCR→event; `Semaphore(SYNTH_CONCURRENCY)`; cancel flag between chunks; failed-chunk policy (6 attempts → silence placeholder + warning; > `MAX_FAILED_CHUNK_PCT` → job FAILED resumable); rolling ETA(50).
**Accept:** mocked-engine tests — fresh run completes manifest · **kill-after-N then rerun synthesizes only the remainder** · truncated wav on disk gets re-synthesized · flaky engine (2 fails) records attempts · cache hit skips engine call · cancel → CANCELLED with consistent manifest · bulk loop parks while fast_pending set.

## PHASE 8 — Assembly
**Build:** `pipeline/assemble.py` per §10/§13: global silence generation (`/data/silence`, once); chunk validation with one-shot repair; concat list; single-pass ffmpeg (exact flags in doc) via `create_subprocess_exec` in **its own process group** (killpg on cancel/shutdown), stderr → logs; ffprobe duration/size; typed `AssemblyError`.
**Accept:** integration (ffmpeg in test image): 3 generated wavs + pauses → final.mp3, ffprobe duration ≈ sum ± 0.5 s · corrupt middle chunk triggers repair path · cancel kills the process group (no zombie).

## PHASE 9 — Storage + Drive bootstrap
**Build:** `storage/local.py` (→ `/data/outputs/{id}.{ext}`, download served by API); `storage/gdrive.py` per §11 (refresh-token creds; ensure `Narrator/` + `YYYY-MM` with ids cached in `kv`; `MediaFileUpload(resumable=True, chunksize=8MB)` `next_chunk()` loop with progress events, `drive_policy` retries; **pre-check by filename in folder for idempotent re-upload**; terminal classes raise typed `DeliveryTerminal`); `scripts/gdrive_auth.py` (InstalledAppFlow, scope `drive.file`, `run_local_server(port=0, access_type="offline", prompt="consent")`, prints refresh token + next steps).
README one-time GCP setup (verbatim steps): create project → enable **Google Drive API** → OAuth consent screen External → add yourself → **Publish to Production** (else refresh tokens expire in 7 days) → Credentials → OAuth client ID → **Desktop app** → copy id/secret → `uv run python scripts/gdrive_auth.py` → browser consent → paste `GDRIVE_CLIENT_ID/SECRET/REFRESH_TOKEN` into Coolify.
**Accept:** local backend integration test · mocked-googleapiclient tests (folder ensure, resumable loop, retry, `invalid_grant` → `DeliveryTerminal`, idempotent re-upload finds existing) · auth script `--help` runs.

## PHASE 10 — Worker (two lanes, graceful shutdown)
**Build:** `worker/tasks.py::run_job(ctx, job_id)` — stages QUEUED→…→COMPLETED with events; `DeliveryTerminal` ⇒ **UPLOAD_PENDING** (artifact copied to `/data/outputs`, downloadable) never FAILED; webhook (HMAC, 3×) on terminal states; chunk cleanup after successful delivery. Cron tasks: `retry_pending_uploads` (30 min), `cleanup_expired` (hourly: finished-job chunks, RETENTION_DAYS purge, `cache.evict_to_limit`, tmp sweep), nightly `backup_ledger` (tar ledger+manifests → Drive, best-effort). `worker/shutdown.py` + `worker/main.py`: arq `WorkerSettings` parameterized by `QUEUE` env (`fast`/`bulk`), `max_jobs=1`, `job_timeout=172800`, `max_tries=3`, `on_startup=reconcile`; SIGTERM → finish current chunk, flush manifest, re-enqueue own job to its lane, exit < 150 s.
**Accept:** e2e with mocked engine + local storage: submit→COMPLETED · forced `DeliveryTerminal` → UPLOAD_PENDING then `retry_pending_uploads` completes it · SIGTERM mid-synth → clean exit, job re-enqueued, rerun resumes · exception mid-synth → FAILED, `retry` path resumes.

## PHASE 11 — API
**Build:** per §7 — key auth dependency (constant-time, multi-key) · `POST /v1/jobs` (streamed multipart; preflights: ext/size/`MAX_INPUT_CHARS`/disk estimate 507/`MAX_ACTIVE_JOBS` 429+Retry-After; Idempotency-Key SETNX 24 h; **lane routing** by estimated minutes; enqueue to `q:fast|q:bulk`) · `GET /v1/jobs/{id}` (incl. `stall_reason`, warnings, result) · `GET /v1/jobs` · SSE `/events` (pub/sub bridge, 15 s heartbeat, closes on terminal) · `/download` (streams local artifact for LocalStorage/UPLOAD_PENDING) · `POST /cancel` · `POST /retry` (FAILED → re-enqueue, resumes) · `POST /retry-upload` · `GET /v1/voices` (10-min cache) · `GET /healthz` (redis, tts `/health`, disk ≥ 2 GB) · `GET /metrics` (all §16 gauges incl. `breaker_state`, `cache_hit_ratio`) · 10/min/key sliding-window on submit · global exception handler → problem-JSON.
**Accept:** ASGITransport tests: 401/403 · 202+readable job · idempotent resubmit same id · 429 + 507 paths · lane routing by size · cancel/retry/retry-upload state changes · SSE emits and closes · healthz degrades with tts mock down · download streams bytes.

## PHASE 12 — Docker + compose (local)
**Build:** `docker/api.Dockerfile` (3.12-slim, uv, curl, non-root, uvicorn :8000); `docker/worker.Dockerfile` (+ ffmpeg, cmd `arq narrator.worker.main.WorkerSettings`); `docker-compose.yml` **exactly** ARCHITECTURE §13 (5 services, mem limits, cpu_shares, stop_grace_period, healthchecks); `docker-compose.dev.yml` (bind mounts, `STORAGE_BACKEND=local`, publish api 8000 + tts 8880).
**Accept:** `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build` → all healthy (tts ready post-warmup) → Phase 13 smoke passes locally.

## PHASE 13 — Smoke + bench + chaos
**Build:**
- `scripts/sample.txt` (~3k chars) + `scripts/smoke_test.sh` (submit verbatim job → poll → assert COMPLETED → for local backend, `GET /download` non-empty; exit codes for CI).
- `scripts/bench.py`: against live tts, matrix `{ONNX_INTRA_OP:1,2} × {SYNTH_CONCURRENCY:1,2,3}` on ~60 s of text → prints RTF table + recommended envs.
- `tests/chaos/` (compose-driven, marked `chaos`): ① `docker kill worker-bulk` mid-job → restart → COMPLETED with zero re-synthesized done-chunks (assert via manifest attempts) ② stop `tts` 90 s mid-job → breaker opens (`stall_reason`) → start → job completes ③ `docker restart redis` mid-job → completes ④ tiny-quota disk (tmpfs mount) → job pauses resumable, not corrupt ⑤ truncate a done chunk wav → resume re-synthesizes exactly that chunk.
**Accept:** smoke exits 0 · bench prints table · **all five chaos scenarios converge to COMPLETED/resumable with no corrupt artifacts**.

## PHASE 14 — Coolify production deploy
Checklist (also into README):
1. Push repo to GitHub/GitLab.
2. Coolify → **+ New → Docker Compose** → server + repo/branch → `docker-compose.yml`.
3. Env tab (mark secret): `API_KEYS` (`openssl rand -hex 24`), `STORAGE_BACKEND=gdrive`, `GDRIVE_CLIENT_ID/SECRET/REFRESH_TOKEN`, `GEMINI_API_KEY`; optional `GEMINI_MODEL`, `EXPLAINER_TARGET_MINUTES`, `ONNX_INTRA_OP`, `SYNTH_CONCURRENCY`.
4. Domain on `api` only (port 8000); confirm tts/redis/workers unpublished. Volumes `narrator-data`, `tts-models`, `redis-data` persistent.
5. Deploy; first tts boot downloads models (watch start_period 90 s → healthy).
6. `curl -fsS https://<domain>/healthz` ok · `curl -H "X-API-Key: $KEY" https://<domain>/v1/voices` lists `af_heart`.
7. `scripts/smoke_test.sh https://<domain> $KEY` → COMPLETED → file visible in **Drive → Narrator/YYYY-MM/**.
8. SSH once: `uv run python scripts/bench.py --base https://<domain>` (or against internal tts) → apply recommended envs in Coolify → redeploy.
9. Enable repo deploy webhook. 10. Mid-job redeploy drill: start a job, trigger deploy, verify auto-resume.
**Accept:** steps 6–8 and 10 pass on the live domain.

## PHASE 15 — Hardening + README
**Verify/finish:** disk preflight + mid-run guard exercised · `MAX_FAILED_CHUNK_PCT` path tested · webhook HMAC verified in test · metrics under a 20-tiny-job burst · README final: quickstart, curl examples both modes, env table, ops runbook (**token rotation, disk-full, breaker-open, UPLOAD_PENDING recovery**), scaling note (Hostinger plan resize = env only; optional second KVM: workers+tts pointed at this Redis over private networking).
**Accept:** full suite + chaos green · fresh-clone `uv sync && uv run pytest -q` passes · live explainer smoke: 50-page PDF `{"mode":"explainer"}` → COMPLETED → Drive link plays coherent plain-language narration in `af_heart`.

---

## DEFINITION OF DONE
- [ ] All 16 phase gates passed in order; ruff + pytest + chaos suite green
- [ ] Deployed on Coolify (KVM 2 only); `/healthz` green; smoke → Google Drive end-to-end
- [ ] Kill any container mid-job → job converges with zero lost chunks
- [ ] Drive failure produces UPLOAD_PENDING with working `/download`, then recovers via retry
- [ ] Fast job submitted mid-marathon starts within ~90 s (gate preemption observed)
- [ ] Changing voice/model endpoint/Gemini model = env change only; no secrets in git
