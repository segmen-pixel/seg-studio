// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { errorText, storedLang, type ErrorKind } from "../i18n/errorText";

export const API_BASE = window.location.origin + "/api/v1";
// v2 streaming surface (/v2/*, /ws/v2/*) is mounted at the root, not under /api/v1.
export const API_ORIGIN = window.location.origin;

export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024; // 200 MB — matches server limit
export const MAX_ZIP_IMPORT_BYTES = 4 * 1024 * 1024 * 1024; // 4 GB — ZIP project import
export const MAX_RECIPE_BYTES = 10 * 1024 * 1024;  // 10 MB — recipes are JSON

/** ZIP import size limit in bytes, configurable via Settings (localStorage "seg_max_zip_import_gb"). Defaults to 4 GB. */
export function getZipImportLimitBytes(): number {
  const gb = parseFloat(localStorage.getItem("seg_max_zip_import_gb") || "4");
  const safe = Number.isFinite(gb) && gb > 0 ? gb : 4;
  return Math.round(safe * 1024 * 1024 * 1024);
}

export function assertFileSize(file: File, max: number = MAX_UPLOAD_BYTES): void {
  if (file.size > max) {
    const maxMB = Math.round(max / (1024 * 1024));
    throw new Error(`File "${file.name}" is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed: ${maxMB} MB.`);
  }
}

// ---------------------------------------------------------------------------
// Structured API Error
// ---------------------------------------------------------------------------

/**
 * Structured error from the backend's unified error response.
 *
 * `message` is what a screen shows: the server's sentence in the screen's
 * language where its code has one (errorText). Most places that show an error
 * show `message`, so it is put right here and not at each of them.
 */
export class ApiError extends Error {
  /** NSS error code (e.g. "NSS-3004") or null for legacy responses */
  readonly code: string | null;
  /** What the message is to the code: the code's usual one, more specific, or uncoded */
  readonly kind: ErrorKind;
  /** The server's own sentence, in English */
  readonly serverMessage: string;
  /** HTTP status code */
  readonly status: number;
  /** Correlation ID for log tracing */
  readonly correlationId: string | null;
  /** Hint from backend (user-actionable suggestion) */
  readonly hint: string | null;
  /** Raw response body */
  readonly raw: string;

  constructor(
    status: number,
    message: string,
    opts: { code?: string | null; kind?: ErrorKind; correlationId?: string | null; hint?: string | null;
            raw?: string } = {},
  ) {
    const code = opts.code ?? null;
    const kind = opts.kind ?? "uncoded";
    super(errorText(code, kind, message, storedLang()));
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.kind = kind;
    this.serverMessage = message;
    this.correlationId = opts.correlationId ?? null;
    this.hint = opts.hint ?? null;
    this.raw = opts.raw ?? "";
  }

  /** True for transient errors that may succeed on retry */
  get isTransient(): boolean {
    return this.status === 0 || this.status === 408 || this.status === 429
      || this.status === 502 || this.status === 503 || this.status === 504;
  }

  /** User-facing display: code + message */
  get displayMessage(): string {
    const parts: string[] = [];
    // An uncoded error carries the bucket code of its status, which names nothing.
    if (this.code && this.kind !== "uncoded") parts.push(`[${this.code}]`);
    parts.push(this.message);
    // The dictionary's words carry the advice; the hint is the server's, in English.
    if (this.hint && this.message === this.serverMessage) parts.push(`— ${this.hint}`);
    return parts.join(" ");
  }
}

/**
 * Parse a non-ok Response into an ApiError.
 * Handles both new structured format `{error: {code, message, ...}}`
 * and legacy format `{detail: "..."}`.
 */
export async function parseApiError(res: Response): Promise<ApiError> {
  const raw = await res.text();
  let message = raw || res.statusText;
  let code: string | null = null;
  let kind: ErrorKind = "uncoded";
  let correlationId: string | null = null;
  let hint: string | null = null;

  try {
    const parsed = JSON.parse(raw);
    if (parsed.error) {
      // New structured format
      code = parsed.error.code ?? null;
      if (parsed.error.kind === "usual" || parsed.error.kind === "specific") kind = parsed.error.kind;
      message = parsed.error.message ?? message;
      correlationId = parsed.error.correlation_id ?? null;
      hint = parsed.error.hint ?? null;
    } else if (parsed.detail) {
      // Legacy FastAPI format
      message = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
    }
  } catch {
    // Not JSON — use raw text as message
  }

  return new ApiError(res.status, message, { code, kind, correlationId, hint, raw });
}
