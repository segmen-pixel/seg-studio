# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Structured logging configuration for the Trainer API.

Usage:
    from .core.logging_config import configure_logging
    configure_logging(log_dir=Path("logs"))

Environment variables:
    LOG_LEVEL  - DEBUG, INFO (default), WARNING, ERROR
    LOG_FORMAT - text (default) or json
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class _CorrelationFilter(logging.Filter):
    """Inject correlation_id from ContextVar into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Late import to avoid circular dependency (error_handlers imports logging)
        try:
            from .error_handlers import correlation_id_var
            record.correlation_id = correlation_id_var.get("")  # type: ignore[attr-defined]
        except Exception:
            record.correlation_id = ""  # type: ignore[attr-defined]
        return True


_TEXT_FMT = "%(asctime)s  %(levelname)-8s  [%(correlation_id)s]  %(name)s  %(message)s"
_TEXT_FMT_NO_CID = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log entries."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", ""),
        }
        if record.exc_info and record.exc_info[1] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        # Remove empty values
        payload = {k: v for k, v in payload.items() if v}
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_dir: Path | None = None) -> None:
    """Initialise logging. Call once at startup before any logger use."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get("LOG_FORMAT", "text").lower()

    cid_filter = _CorrelationFilter()

    # --- stdout handler ---
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(cid_filter)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt=_TEXT_FMT, datefmt=_DATE_FMT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # --- file handlers with rotation ---
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

        # All levels: app.log (10 x 20MB)
        app_fh = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=20 * 1024 * 1024, backupCount=10,
            encoding="utf-8", errors="backslashreplace",
        )
        app_fh.addFilter(cid_filter)
        app_fh.setFormatter(logging.Formatter(fmt=_TEXT_FMT, datefmt=_DATE_FMT))
        root.addHandler(app_fh)

        # Errors only: trainer_errors.log (5 x 10MB)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "trainer_errors.log", maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8", errors="backslashreplace",
        )
        fh.setLevel(logging.WARNING)
        fh.addFilter(cid_filter)
        fh.setFormatter(logging.Formatter(fmt=_TEXT_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)

    # --- quiet noisy third-party loggers ---
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
