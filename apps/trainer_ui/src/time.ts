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

/** When a run began, and whether that is actually known.
 *
 * `started_at` is the moment training began. `created_at` is when the row was
 * written, which for a run that waited for a free GPU is when it was queued --
 * measured at 23 of 96 rows more than a minute apart, worst case 397.8 minutes.
 *
 * Rows written before `started_at` existed carry null, and a run still sitting
 * in the queue has not started at all, so a caller that has to show something
 * still falls back to `created_at`. What it must not do is label that a start
 * time, which is why `isStart` comes back with the date rather than being left
 * for each caller to work out again.
 */
export function resolveRunStart(
  startedAt: string | null | undefined,
  createdAt: string | null | undefined,
): { date: Date | null; isStart: boolean } {
  const started = parseApiDate(startedAt);
  if (started) return { date: started, isStart: true };
  return { date: parseApiDate(createdAt), isStart: false };
}

export function formatApiDate(value: string | null | undefined): string {
  const parsed = parseApiDate(value);
  if (!parsed) return "-";
  return parsed.toLocaleString();
}
