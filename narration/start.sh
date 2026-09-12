#!/bin/sh
set -eu
# API + arq worker in one container so they share /data. Concurrency=1 is
# enforced by arq max_jobs, not by running two workers.
arq narration.worker.main.WorkerSettings &
arq_pid=$!
uvicorn narration.api.main:app --host 0.0.0.0 --port 8870 &
uv_pid=$!
trap 'kill "$arq_pid" "$uv_pid" 2>/dev/null || true' INT TERM
wait -n "$arq_pid" "$uv_pid"
status=$?
kill "$arq_pid" "$uv_pid" 2>/dev/null || true
exit "$status"
