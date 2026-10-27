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
#: What the message is to its code, for a screen that says errors in its own
#: language (trainer_ui/src/i18n/errorText.ts):
#: "usual" -- the code's registered message, which a screen has its own words for;
#: "specific" -- the code names the error and the message says more (a file's name);
#: "uncoded" -- an HTTPException: the message is all there is, and the code is
#: only the bucket its status falls in. A screen that put the bucket's words in
#: its place told a person "a required parameter is missing" for every refusal.
KINDS = ("usual", "specific", "uncoded")


def _build_body(
    code: str,
    message: str,
    correlation_id: str,
    hint: str | None = None,
    kind: str = "uncoded",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "correlation_id": correlation_id,
            "kind": kind,
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
                kind="usual" if exc.user_message == meta.message_en else "specific",
            ),
        )

    # ── FastAPI request validation ────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        cid = correlation_id_var.get("") or _make_cid()
        errors = exc.errors()
        fields = [str(e.get("loc", ["?"])[-1]) for e in errors]
        message = f"Invalid input: {', '.join(fields)}"
        logger.warning("[%s] [%s] Validation: %s", E.VALIDATION_REQUIRED_PARAM, cid, errors)
        # NSS-1006's words are "a required parameter is missing", and a screen
        # puts them beside a "specific" message. True when every field named is
        # missing; for a value of the wrong type or out of range the Japanese
        # screen said a parameter was missing while the English one said the
        # input was invalid. Those go uncoded: the message is all there is.
        missing = bool(errors) and all(e.get("type") == "missing" for e in errors)
        return JSONResponse(
            status_code=422,
            content=_build_body(E.VALIDATION_REQUIRED_PARAM, message, cid,
                                kind="specific" if missing else "uncoded"),
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
                kind="usual",
            ),
        )
