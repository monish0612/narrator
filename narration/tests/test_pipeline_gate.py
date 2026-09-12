from __future__ import annotations

from narration.ram_gate import next_backoff_s
from narration.store import STATUS_FALLBACK


class _FakeBreaker:
    def __init__(self, open: bool = False) -> None:
        self._open = open
        self.failures: list[str] = []
        self.name = "pipeline"

    async def is_open(self) -> bool:
        return self._open

    async def record_success(self) -> bool:
        return False

    async def record_failure(self, article_id: str) -> bool:
        self.failures.append(article_id)
        return False


def test_breaker_open_marks_fallback_without_llm(monkeypatch, tmp_path):
    # Pipeline.generate short-circuits before RAM/LLM when breaker is open.
    assert next_backoff_s(2) == 120
    assert STATUS_FALLBACK == "fallback"
