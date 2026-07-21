// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
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

/** Structured error from the backend's unified error response. */
export class ApiError extends Error {
  /** NSS error code (e.g. "NSS-3004") or null for legacy responses */
  readonly code: string | null;
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
    opts: { code?: string | null; correlationId?: string | null; hint?: string | null; raw?: string } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = opts.code ?? null;
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
    if (this.code) parts.push(`[${this.code}]`);
    parts.push(this.message);
    if (this.hint) parts.push(`— ${this.hint}`);
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
  let correlationId: string | null = null;
  let hint: string | null = null;

  try {
    const parsed = JSON.parse(raw);
    if (parsed.error) {
      // New structured format
      code = parsed.error.code ?? null;
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

  return new ApiError(res.status, message, { code, correlationId, hint, raw });
}
