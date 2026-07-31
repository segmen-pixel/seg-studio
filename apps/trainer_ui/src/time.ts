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

/** ``YYYY-MM-DD HH:mm`` in the viewer's zone.
 *
 * The reports list used to slice this out of the raw string, which reads a UTC
 * clock as if it were local -- nine hours early in JST. Building it from a
 * parsed Date is the only way the digits and the zone agree.
 */
export function formatApiDateShort(value: string | null | undefined): string {
  const parsed = parseApiDate(value);
  if (!parsed) return "-";
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}` +
    ` ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}`
  );
}

export function formatApiDate(value: string | null | undefined): string {
  const parsed = parseApiDate(value);
  if (!parsed) return "-";
  return parsed.toLocaleString();
}
