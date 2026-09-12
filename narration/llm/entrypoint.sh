#!/bin/sh
set -eu

GGUF_PATH="${GGUF_PATH:-/models/Qwen3.5-4B-Q4_K_M.gguf}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-4b}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf}"
GGUF_SHA256="${GGUF_SHA256:-00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4}"
BLOB="/root/.ollama/models/blobs/sha256-${GGUF_SHA256}"
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

bytes_of() {
  wc -c < "$1" | tr -d ' '
}

# Unsloth Q4_K_M is ~2.74GB. Reject stubs from a failed older pull.
file_ok() {
  [ -f "$1" ] || return 1
  n=$(bytes_of "$1")
  [ "$n" -gt 2500000000 ] && [ "$n" -lt 3600000000 ]
}

write_modelfile() {
  printf 'FROM %s\nPARAMETER num_ctx 8192\nPARAMETER num_thread 2\nPARAMETER temperature 0.4\n' "$1" > /tmp/Modelfile
}

create_from() {
  write_modelfile "$1"
  ollama create "$LLM_MODEL" -f /tmp/Modelfile
}

model_ready() {
  ollama show "$LLM_MODEL" >/dev/null 2>&1
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

recover_blob() {
  if file_ok "$GGUF_PATH"; then
    return 0
  fi
  if file_ok "$BLOB"; then
    mkdir -p "$(dirname "$GGUF_PATH")"
    ln -sf "$BLOB" "$GGUF_PATH"
    telegram "recovered GGUF blob already on the ollama volume"
  fi
}

ollama serve &
pid=$!

i=0
while [ "$i" -lt 90 ]; do
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

if model_ready; then
  telegram "model already present — serving"
  wait "$pid"
  exit 0
fi

recover_blob
if ! file_ok "$GGUF_PATH"; then
  download_gguf || true
fi

if file_ok "$GGUF_PATH"; then
  telegram "creating ${LLM_MODEL} from GGUF"
  set +e
  create_from "$GGUF_PATH"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && model_ready; then
    telegram "GGUF import ok"
    wait "$pid"
    exit 0
  fi
  telegram "GGUF import failed for ${GGUF_PATH}"
fi

# Library tag talks to ollama.com (not HF) and is Q4_K_M qwen35.
if ollama pull "qwen3.5:4b"; then
  set +e
  create_from "qwen3.5:4b"
  if ! model_ready; then
    ollama cp "qwen3.5:4b" "$LLM_MODEL"
  fi
  set -e
  if model_ready; then
    telegram "pulled qwen3.5:4b"
    wait "$pid"
    exit 0
  fi
fi

if ollama pull "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"; then
  set +e
  create_from "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
  if ! model_ready; then
    ollama cp "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M" "$LLM_MODEL"
  fi
  set -e
  recover_blob
  if model_ready; then
    telegram "pulled hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
    wait "$pid"
    exit 0
  fi
fi

if ollama pull "oamazonasgabriel/qwen3.5-4b"; then
  set +e
  create_from "oamazonasgabriel/qwen3.5-4b"
  if ! model_ready; then
    ollama cp "oamazonasgabriel/qwen3.5-4b" "$LLM_MODEL"
  fi
  set -e
  if model_ready; then
    telegram "pulled oamazonasgabriel/qwen3.5-4b"
    wait "$pid"
    exit 0
  fi
fi

telegram "HALT: GGUF missing and pull failed. Not crash-looping."
wait "$pid"
