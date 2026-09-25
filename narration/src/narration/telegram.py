"""Telegram alerts. Never log tokens. Failures here must not break the pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from narration.logging import get_logger

log = get_logger("narration.telegram")


class Telegram:
    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat = chat_id
        self._client: httpx.AsyncClient | None = None
        self._redis = None

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat)

    async def send(self, text: str, *, redis=None, failure: bool = False) -> None:
        store = redis if redis is not None else self._redis
        if failure and store is not None:
            try:
                got = await store.set("narration:alert:fail", "1", nx=True, ex=3600)
            except Exception:
                got = True
            if not got:
                log.info("telegram.suppressed", reason="hourly_cap")
                return
        if not self.enabled:
            log.info("telegram.skipped", reason="unconfigured", preview=text[:180])
            return
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat,
            "text": text[:3900],
            "disable_web_page_preview": True,
        }
        try:
            client = self._client or httpx.AsyncClient(timeout=10)
            self._client = client
            await client.post(url, json=payload)
        except Exception as exc:
            log.warning("telegram.failed", error=str(exc)[:200])

    def fire(self, text: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.send(text))


def format_alert(kind: str, **fields: Any) -> str:
    bits = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    return f"[narration] {kind} {bits}".strip()
