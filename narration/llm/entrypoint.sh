#!/bin/sh
set -eu

GGUF_PATH="${GGUF_PATH:-/models/Qwen3.5-4B-Q4_K_M.gguf}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-4b}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0}"
export OLLAMA_KEEP_ALIVE="$KEEP_ALIVE"

telegram() {
  msg="$1"
  if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    wget -qO- --header="Content-Type: application/json" \
      --post-data="{\"chat_id\":\"${TELEGRAM_CHAT_ID}\",\"text\":\"[narration-llm] ${msg}\"}" \
      "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" >/dev/null 2>&1 || true
  fi
  echo "[narration-llm] $msg"
}

ollama serve &
pid=$!

i=0
while [ "$i" -lt 60 ]; do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  i=$((i + 1))
  sleep 1
done

if ! ollama list >/dev/null 2>&1; then
  telegram "ollama serve failed to become ready"
  wait "$pid"
  exit 1
fi

if ollama list | grep -qi "qwen3.5"; then
  telegram "model already present — serving"
  wait "$pid"
  exit 0
fi

if [ -f "$GGUF_PATH" ]; then
  telegram "creating ${LLM_MODEL} from mounted GGUF"
  printf 'FROM %s\nPARAMETER num_ctx 8192\nPARAMETER num_thread 2\nPARAMETER temperature 0.4\n' "$GGUF_PATH" > /tmp/Modelfile
  if ollama create "$LLM_MODEL" -f /tmp/Modelfile; then
    telegram "GGUF import ok"
    wait "$pid"
    exit 0
  fi
  telegram "GGUF import failed for ${GGUF_PATH}"
fi

if ollama pull "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"; then
  telegram "pulled hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
  wait "$pid"
  exit 0
fi

if ollama pull "oamazonasgabriel/qwen3.5-4b"; then
  telegram "pulled oamazonasgabriel/qwen3.5-4b"
  wait "$pid"
  exit 0
fi

telegram "HALT: no GGUF at ${GGUF_PATH} and pull failed. scp Qwen3.5-4B-Q4_K_M.gguf onto the llm volume, then restart. Not crash-looping."
# Stay up so Coolify does not restart-storm; /api/tags remains reachable; orchestrator falls back.
wait "$pid"
