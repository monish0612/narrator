"""structlog JSON logging (ARCHITECTURE.md section 16).

JSON only, always carrying ``job_id`` (plus ``stage`` / ``chunk_idx`` where
relevant). Per-chunk logs at DEBUG; INFO is per-stage. No ``print`` anywhere in
the pipeline - use ``get_logger`` / ``bind_job``.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotently configure structlog to emit JSON to stdout."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind_job(job_id: str, **kwargs: object) -> None:
    """Bind ``job_id`` (and optional stage/chunk_idx) to the context vars so
    every subsequent log line in this task carries them automatically."""
    structlog.contextvars.bind_contextvars(job_id=job_id, **kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
