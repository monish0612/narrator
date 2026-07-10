#!/usr/bin/env bash
# End-to-end smoke test against a running Narrator API (ARCHITECTURE.md section 7).
#
#   HOST=https://narrator.example.com API_KEY=xxxx ./scripts/smoke_test.sh
#
# Env:
#   HOST      base URL of the api service        (default http://localhost:8000)
#   API_KEY   a valid key from API_KEYS          (required)
#   MODE      verbatim|explainer                 (default verbatim)
#   TIMEOUT   seconds to wait for COMPLETED       (default 300)
set -euo pipefail

HOST="${HOST:-http://localhost:8000}"
MODE="${MODE:-verbatim}"
TIMEOUT="${TIMEOUT:-300}"
: "${API_KEY:?set API_KEY to a value from the API_KEYS secret}"

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }

echo "== Narrator smoke test -> ${HOST} (mode=${MODE}) =="

# 1. Liveness (no auth).
curl -fsS "${HOST}/healthz" >/dev/null && pass "GET /healthz" || fail "GET /healthz"

# 2. Voices (auth) - must list the default voice.
voices="$(curl -fsS -H "X-API-Key: ${API_KEY}" "${HOST}/v1/voices")"
echo "${voices}" | grep -q "af_heart" && pass "GET /v1/voices lists af_heart" || fail "voices missing af_heart: ${voices}"

# 3. Submit a job.
read -r -d '' TEXT <<'EOF' || true
Narrator is a long form text to speech narration platform. It turns documents
into natural sounding audio. This smoke test confirms the full pipeline works:
ingest, chunk, synthesize, assemble, and deliver a downloadable artifact.
EOF
payload="$(printf '{"text":%s,"mode":"%s"}' "$(printf '%s' "${TEXT}" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" "${MODE}")"
created="$(curl -fsS -X POST -H "X-API-Key: ${API_KEY}" -H 'Content-Type: application/json' -d "${payload}" "${HOST}/v1/jobs")"
job_id="$(printf '%s' "${created}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')"
lane="$(printf '%s' "${created}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("lane",""))')"
[ -n "${job_id}" ] && pass "POST /v1/jobs -> ${job_id} (lane=${lane})" || fail "no job_id in ${created}"

# 4. Poll to terminal state.
deadline=$(( $(date +%s) + TIMEOUT ))
status=""
while [ "$(date +%s)" -lt "${deadline}" ]; do
  body="$(curl -fsS -H "X-API-Key: ${API_KEY}" "${HOST}/v1/jobs/${job_id}")"
  status="$(printf '%s' "${body}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')"
  case "${status}" in
    COMPLETED) pass "job reached COMPLETED"; break ;;
    FAILED|CANCELLED) fail "job ended ${status}: ${body}" ;;
  esac
  sleep 5
done
[ "${status}" = "COMPLETED" ] || fail "timed out after ${TIMEOUT}s (last status: ${status})"

# 5. Download the artifact.
out="$(mktemp -t narrator-smoke-XXXXXX.mp3)"
code="$(curl -fsS -o "${out}" -w '%{http_code}' -H "X-API-Key: ${API_KEY}" "${HOST}/v1/jobs/${job_id}/download")"
size="$(wc -c < "${out}" | tr -d ' ')"
{ [ "${code}" = "200" ] && [ "${size}" -gt 1024 ]; } \
  && pass "downloaded artifact (${size} bytes -> ${out})" \
  || fail "download failed (http ${code}, ${size} bytes)"

echo "== ALL CHECKS PASSED =="
