"""Ollama OpenAI-compatible client + self-healing model ensure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from narration.logging import get_logger
from narration.prompts import (
    AI_EXPLAINER_WITH_PARALLEL,
    AI_RELEVANCE_SYSTEM,
    COMPRESS_SYSTEM,
    PLAIN_EXPLAINER,
    build_ai_user,
    build_compress_user,
    build_plain_user,
    build_relevance_user,
    is_ai_news,
    word_count,
)

log = get_logger("narration.llm")


class LlmError(Exception):
    pass


class LlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        gguf_path: str,
        timeout: float = 420,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.model = model
        self.gguf_path = gguf_path
        self.num_ctx = 8192
        self._own = client is None
        self._http = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._own:
            await self._http.aclose()

    async def chat(self, system: str, user: str, *, temperature: float = 0.4, max_tokens: int = 4096) -> str:
        url = f"{self.base}/v1/chat/completions"
        resp = await self._http.post(
            url,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "keep_alive": "5m",
                # Qwen3.5 native context cannot fit the 4.5 GB cap.
                "options": {"num_ctx": self.num_ctx, "num_thread": 2},
            },
        )
        if resp.status_code >= 400:
            raise LlmError(f"llm http {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"malformed llm response: {exc}") from exc

    async def tags(self) -> list[str]:
        try:
            resp = await self._http.get(f"{self.base}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models") or []
            return [str(m.get("name") or m.get("model") or "") for m in models]
        except Exception as exc:
            raise LlmError(f"tags failed: {exc}") from exc

    def model_present(self, names: list[str]) -> bool:
        needle = self.model.lower()
        for n in names:
            if needle in n.lower() or n.lower().startswith(needle):
                return True
        return False

    async def model_ready(self) -> bool:
        try:
            resp = await self._http.post(f"{self.base}/api/show", json={"name": self.model})
            return resp.status_code < 400
        except Exception:
            return False

    async def ensure_model(self) -> str:
        if await self.model_ready():
            return self.model
        names: list[str] = []
        try:
            names = await self.tags()
        except LlmError as exc:
            log.warning("llm.tags_unready", error=str(exc)[:200])
        if self.model_present(names) and await self.model_ready():
            return self.model
        gguf = Path(self.gguf_path)
        if gguf.is_file():
            modelfile = (
                f"FROM {gguf}\n"
                "PARAMETER num_ctx 8192\n"
                "PARAMETER num_thread 2\n"
                "PARAMETER temperature 0.4\n"
            )
            log.info("llm.create_from_gguf", path=str(gguf))
            resp = await self._http.post(
                f"{self.base}/api/create",
                json={"name": self.model, "modelfile": modelfile, "stream": False},
                timeout=600,
            )
            if resp.status_code >= 400:
                raise LlmError(f"create from gguf failed: {resp.status_code} {resp.text[:300]}")
            if not await self.model_ready():
                raise LlmError("gguf imported but ollama /api/show failed")
            return self.model

        if await self.model_ready():
            return self.model

        for candidate in (
            f"hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
            "oamazonasgabriel/qwen3.5-4b",
        ):
            log.info("llm.pull", name=candidate)
            try:
                resp = await self._http.post(
                    f"{self.base}/api/pull",
                    json={"name": candidate, "stream": False},
                    timeout=1800,
                )
                if resp.status_code < 400:
                    self.model = candidate
                    if await self.model_ready():
                        return self.model
                    log.warning("llm.pull_not_ready", name=candidate)
                log.warning("llm.pull_rejected", name=candidate, status=resp.status_code)
            except Exception as exc:
                log.warning("llm.pull_failed", name=candidate, error=str(exc)[:200])
        raise LlmError(
            f"model missing: place {gguf} on the llm volume or allow huggingface.co pull"
        )

    async def generate_script(
        self,
        *,
        title: str,
        source: str,
        category: str,
        article_text: str,
        word_min: int,
        word_max: int,
    ) -> str:
        article_text = (article_text or "")[:24000]
        if is_ai_news(category):
            raw = await self.chat(AI_RELEVANCE_SYSTEM, build_relevance_user(title, article_text), temperature=0.1)
            relevant, parallel = _parse_relevance(raw)
            if relevant and parallel:
                script = await self.chat(
                    AI_EXPLAINER_WITH_PARALLEL,
                    build_ai_user(title, source, article_text, parallel),
                )
            else:
                script = await self.chat(PLAIN_EXPLAINER, build_plain_user(title, source, article_text))
        else:
            script = await self.chat(PLAIN_EXPLAINER, build_plain_user(title, source, article_text))

        script = _strip_think(_strip_fences(script))
        if word_count(script) > word_max:
            compressed = await self.chat(
                COMPRESS_SYSTEM.format(n=word_max),
                build_compress_user(script, word_max),
                temperature=0.2,
            )
            compressed = _strip_think(_strip_fences(compressed))
            if word_count(compressed) < word_count(script):
                script = compressed
        return script


def _strip_think(text: str) -> str:
    t = text or ""
    while True:
        start = t.find("<think>")
        end = t.find("</think>")
        if start < 0 or end < 0 or end < start:
            break
        t = (t[:start] + t[end + len("</think>") :]).strip()
    return t.strip()


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def _parse_relevance(raw: str) -> tuple[bool, str]:
    blob = _strip_fences(_strip_think(raw))
    try:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            data: dict[str, Any] = json.loads(blob[start : end + 1])
            relevant = bool(data.get("relevant"))
            parallel = str(data.get("parallel") or "").strip()
            return relevant, parallel
    except json.JSONDecodeError:
        log.warning("llm.relevance_unparsed")
    return False, ""
