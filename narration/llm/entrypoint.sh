#!/bin/sh
set -eu

GGUF_PATH="${GGUF_PATH:-/models/Qwen3.5-4B-Q4_K_M.gguf}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-4b}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"
GGUF_URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
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

file_ok() {
  [ -f "$1" ] && [ "$(wc -c < "$1" | tr -d ' ')" -gt 1000000000 ]
}

download_gguf() {
  mkdir -p "$(dirname "$GGUF_PATH")"
  tmp="${GGUF_PATH}.part"
  telegram "downloading GGUF (~2.74GB) from Hugging Face"
  set +e
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 8 -C - -A "narration-llm" -o "$tmp" "$GGUF_URL"
    rc=$?
  else
    wget -c -U "narration-llm" -O "$tmp" "$GGUF_URL"
    rc=$?
  fi
  set -e
  if [ "$rc" -eq 0 ] && file_ok "$tmp"; then
    mv "$tmp" "$GGUF_PATH"
    telegram "GGUF download ok"
    return 0
  fi
  telegram "GGUF download failed (rc=${rc:-?})"
  return 1
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

if ! file_ok "$GGUF_PATH"; then
  download_gguf || true
fi

if file_ok "$GGUF_PATH"; then
  telegram "creating ${LLM_MODEL} from GGUF"
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

telegram "HALT: GGUF missing and pull failed. Not crash-looping."
wait "$pid"
