#!/bin/sh
set -eu
# API + arq worker in one container so they share /data. Concurrency=1 is
# enforced by arq max_jobs, not by running two workers.
# Portable: debian slim /bin/sh is dash and does not support `wait -n`.
mkdir -p /data
arq narration.worker.main.WorkerSettings &
arq_pid=$!
uvicorn narration.api.main:app --host 0.0.0.0 --port 8870 &
uv_pid=$!
trap 'kill "$arq_pid" "$uv_pid" 2>/dev/null || true' INT TERM

status=0
while kill -0 "$arq_pid" 2>/dev/null && kill -0 "$uv_pid" 2>/dev/null; do
  sleep 2
done
if kill -0 "$arq_pid" 2>/dev/null; then
  echo "uvicorn exited first" >&2
else
  echo "arq exited first" >&2
  status=1
fi
if ! kill -0 "$uv_pid" 2>/dev/null; then
  status=1
fi
kill "$arq_pid" "$uv_pid" 2>/dev/null || true
exit "$status"
