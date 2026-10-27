// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef, useState } from "react";

import { fetchRunVerdicts } from "../../api";
import type { ItemVerdict } from "../../api";

/** Long enough that dragging the slider does not queue a request per pixel,
 *  short enough that letting go feels immediate. */
const DEBOUNCE_MS = 180;

/** How often to look again while the server fills in older predictions. */
const BACKFILL_POLL_MS = 1500;

/** Stop chasing eventually: a backfill that never reports zero would
 *  otherwise poll for the life of the tab. */
const BACKFILL_MAX_POLLS = 60;

/**
 * Detected / missed / over-detected for every image in the run, at whatever
 * the confidence slider and the area filters currently say.
 *
 * The judging lives on the server so the report and this tab cannot drift
 * apart on what "detected" means. That costs one request per settled slider
 * position, not one per image.
 */
export function useVerdicts(
  projectId: string | null,
  runId: string | null,
  backend: string,
  tta: boolean,
  threshold: number,
  minArea: number,
  maxArea: number,
  enabled: boolean,
) {
  const [verdicts, setVerdicts] = useState<Record<string, ItemVerdict>>({});
  const [counts, setCounts] = useState<Record<string, number> | null>(null);
  const [pending, setPending] = useState(0);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  const polls = useRef(0);

  // A run predicted before summaries existed has none, and the server builds
  // them in the background the first time the tab asks. Keep asking until it
  // says there is nothing left, so the list fills in on its own rather than
  // waiting for someone to re-run inference.
  useEffect(() => {
    if (!pending) return;
    if (polls.current >= BACKFILL_MAX_POLLS) return;
    const id = window.setTimeout(() => {
      polls.current += 1;
      setTick((n) => n + 1);
    }, BACKFILL_POLL_MS);
    return () => window.clearTimeout(id);
  }, [pending, tick]);

  useEffect(() => {
    if (!enabled || !projectId || !runId) {
      setVerdicts({});
      setCounts(null);
      setPending(0);
      return;
    }
    const mine = ++seq.current;
    const timer = window.setTimeout(() => {
      fetchRunVerdicts(projectId, runId, { backend, tta, threshold, minArea, maxArea })
        .then((r) => {
          // The slider fires these faster than they come back, and they can
          // land out of order. Only the newest one may paint.
          if (mine !== seq.current) return;
          setVerdicts(r.verdicts ?? {});
          setCounts(r.counts ?? null);
          setPending(r.pending ?? 0);
        })
        .catch(() => {
          if (mine !== seq.current) return;
          setVerdicts({});
          setCounts(null);
          setPending(0);
        });
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
    // `tick` is in here on purpose: it is how the backfill poll re-runs this.
  }, [enabled, projectId, runId, backend, tta, threshold, minArea, maxArea, tick]);

  // A change of run or of slider position starts the chase over.
  useEffect(() => {
    polls.current = 0;
  }, [projectId, runId, backend, tta, threshold, minArea, maxArea]);

  return { verdicts, counts, pending };
}
