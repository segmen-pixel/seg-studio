// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
const TZ_SUFFIX_RE = /(Z|[+-]\d{2}:\d{2})$/i;

export function parseApiDate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const normalized = TZ_SUFFIX_RE.test(value) ? value : `${value}Z`;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed;
}

export function formatApiDate(value: string | null | undefined): string {
  const parsed = parseApiDate(value);
  if (!parsed) return "-";
  return parsed.toLocaleString();
}
