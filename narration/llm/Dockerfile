# Ollama 0.12.6 cannot load Qwen3.5 (arch qwen35). 0.32.x can import the
# Unsloth Q4_K_M GGUF and also pull the library tag qwen3.5:4b.
# Official image has no curl/wget — install them so a missing volume can
# self-heal from Hugging Face instead of skipping straight to a broken pull.
FROM ollama/ollama:0.32.9

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_KEEP_ALIVE=5m \
    GGUF_PATH=/models/Qwen3.5-4B-Q4_K_M.gguf \
    LLM_MODEL=qwen3.5-4b \
    GGUF_SHA256=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4

COPY narration/llm/entrypoint.sh /usr/local/bin/narration-llm-entrypoint.sh
COPY narration/llm/Modelfile /opt/narration/Modelfile
RUN sed -i 's/\r$//' /usr/local/bin/narration-llm-entrypoint.sh \
    && chmod +x /usr/local/bin/narration-llm-entrypoint.sh

EXPOSE 11434

# Serve-ready only — do not require the model or a GGUF download would
# look unhealthy and Coolify would restart mid-import (spec: no crash-loop).
HEALTHCHECK --interval=30s --timeout=8s --retries=12 --start-period=90s \
    CMD ollama list >/dev/null 2>&1 || exit 1

ENTRYPOINT ["/usr/local/bin/narration-llm-entrypoint.sh"]
