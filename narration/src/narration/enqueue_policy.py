"""Decide whether a narration request may start a new synthesis.

Opening an article on the phone sends the summary. Ingest sends the full
article. Those strings hash to different cache keys. A second request must
not throw away audio that is already ready or already in flight.
"""

from __future__ import annotations

IN_FLIGHT = frozenset({"queued", "generating"})


def classify_enqueue(
    *,
    existing: dict | None,
    requested_ready: bool,
    force: bool,
) -> str:
    """Return keep_ready, in_flight, bind_ready, deleted, or start."""
    if existing and existing.get("status") == "deleted":
        removed = existing.get("reason") == "article_removed"
        if removed or not force:
            return "deleted"
    if existing and existing.get("status") == "ready" and existing.get("cache_key"):
        return "keep_ready"
    if existing and existing.get("status") in IN_FLIGHT:
        return "in_flight"
    if requested_ready:
        return "bind_ready"
    return "start"


def plan_jobs(current: dict, rush: dict | None) -> list[dict]:
    """Run the article the user is waiting on before the backlog job."""
    if not rush:
        return [current]
    rush_id = str(rush.get("article_id") or "").strip()
    current_id = str(current.get("article_id") or "").strip()
    if not rush_id or rush_id == current_id:
        return [current]
    return [rush, current]
