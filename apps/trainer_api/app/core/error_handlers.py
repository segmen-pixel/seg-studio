# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unified error response builder and FastAPI exception handlers.

Security rules:
  - NEVER include file system paths in responses.
  - NEVER include Python tracebacks in responses.
  - ALWAYS log the full context server-side with correlation_id.

Usage:
    from .core.error_handlers import register_error_handlers
    register_error_handlers(app)
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import error_codes as E
from .exceptions import AppError

logger = logging.getLogger("trainer_api.errors")

# ---------------------------------------------------------------------------
# Correlation ID — per-request trace key
# ---------------------------------------------------------------------------
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the current request's correlation ID (empty if outside request)."""
    return correlation_id_var.get("")


def _make_cid() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------
def _build_body(
    code: str,
    message: str,
    correlation_id: str,
    hint: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "correlation_id": correlation_id,
        },
    }
    if hint:
        body["error"]["hint"] = hint
    # Also include "detail" at top level for backward compat with
    # existing frontend code that reads response.detail.
    body["detail"] = message
    return body


def _log_error(
    code: str,
    cid: str,
    message: str,
    *,
    detail: str = "",
    context: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    meta = E.get_meta(code)
    level = getattr(logging, meta.log_level, logging.WARNING)
    parts = [f"[{code}] [{cid}] {message}"]
    if detail:
        parts.append(f"  detail: {detail}")
    if context:
        parts.append(f"  context: {context}")
    msg = "\n".join(parts)
    logger.log(level, msg, exc_info=exc if level >= logging.ERROR else None)


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------
def register_error_handlers(app: FastAPI) -> None:
    """Register exception handlers and correlation-ID middleware."""

    # ── Correlation ID middleware ──────────────────────────────────────
    @app.middleware("http")
    async def _cid_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        cid = request.headers.get("X-Correlation-ID") or _make_cid()
        correlation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    # ── AppError (structured domain exceptions) ───────────────────────
    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        cid = correlation_id_var.get("") or _make_cid()
        meta = E.get_meta(exc.code)
        _log_error(
            exc.code, cid, exc.user_message,
            detail=exc.detail, context=exc.context, exc=exc.__cause__,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=_build_body(
                exc.code,
                exc.user_message,
                cid,
                hint=meta.hint_en,
            ),
        )

    # ── FastAPI request validation ────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        cid = correlation_id_var.get("") or _make_cid()
        fields = [str(e.get("loc", ["?"])[-1]) for e in exc.errors()]
        message = f"Invalid input: {', '.join(fields)}"
        logger.warning("[%s] [%s] Validation: %s", E.VALIDATION_REQUIRED_PARAM, cid, exc.errors())
        return JSONResponse(
            status_code=422,
            content=_build_body(E.VALIDATION_REQUIRED_PARAM, message, cid),
        )

    # ── Starlette HTTPException (backward compat) ─────────────────────
    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exc(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        cid = correlation_id_var.get("") or _make_cid()
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        code = "NSS-7004" if exc.status_code >= 500 else "NSS-1006"
        logger.warning("[%s] [%s] HTTPException %d on %s %s: %s",
                       code, cid, exc.status_code, request.method, request.url.path, detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_body(code, detail, cid),
        )

    # ── Global catch-all ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
        cid = correlation_id_var.get("") or _make_cid()
        _log_error(
            E.SYSTEM_INTERNAL, cid, "Unhandled exception",
            detail=str(exc), exc=exc,
        )
        return JSONResponse(
            status_code=500,
            content=_build_body(
                E.SYSTEM_INTERNAL,
                E.get_meta(E.SYSTEM_INTERNAL).message_en,
                cid,
            ),
        )
