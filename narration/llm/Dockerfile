# Thin wrapper around the official Ollama image so we can:
#   1. Prefer a mounted GGUF (no network).
#   2. Try Hugging Face / Ollama pull if the GGUF is absent.
#   3. Stay up (health red, no crash-loop) and Telegram-alert if both fail.
#
# Native Qwen3.5 context is huge; num_ctx=8192 is required to fit the 4.5 GB cap.
FROM ollama/ollama:0.12.6

ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_KEEP_ALIVE=5m \
    GGUF_PATH=/models/Qwen3.5-4B-Q4_K_M.gguf \
    LLM_MODEL=qwen3.5-4b

COPY narration/llm/entrypoint.sh /usr/local/bin/narration-llm-entrypoint.sh
COPY narration/llm/Modelfile /opt/narration/Modelfile
RUN chmod +x /usr/local/bin/narration-llm-entrypoint.sh

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=8s --retries=12 --start-period=60s \
    CMD ollama list >/dev/null 2>&1 || exit 1

ENTRYPOINT ["/usr/local/bin/narration-llm-entrypoint.sh"]
