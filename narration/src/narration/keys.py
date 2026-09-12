"""Redis key helpers. Every key this feature writes is prefixed and TTL'd."""

from __future__ import annotations


def with_prefix(prefix: str, *parts: str) -> str:
    p = prefix if prefix.endswith(":") else f"{prefix}:"
    return p + ":".join(parts)


def cache_key(prefix: str, cache: str) -> str:
    return with_prefix(prefix, "cache", cache)


def lock_key(prefix: str, cache: str) -> str:
    return with_prefix(prefix, "lock", cache)


def article_key(prefix: str, article_id: str) -> str:
    return with_prefix(prefix, "article", article_id)


def breaker_key(prefix: str, name: str, suffix: str) -> str:
    return with_prefix(prefix, "breaker", name, suffix)


def delete_fail_key(prefix: str, cache: str) -> str:
    return with_prefix(prefix, "delete_fail", cache)


def ram_defer_key(prefix: str, cache: str) -> str:
    return with_prefix(prefix, "ram_defer", cache)
