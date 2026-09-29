// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useMemo, useState } from "react";
import { fetchRunMetrics } from "../../api";
import type { TrainRunItem } from "../../utils";

// Extracted verbatim from Training.tsx (pre-OSS refactor): derived insight
// state for the best run — best-by-F1 selection, detailed metrics fetch,
// F1 delta vs the runner-up, convergence verdict, per-class F1 rows, and
// dataset stats.
export function useBestRunInsights(projectId: string | null, effectiveRuns: TrainRunItem[]) {
  // Best run by F1
  const bestRun = useMemo(() => {
    let best: typeof effectiveRuns[number] | null = null;
    for (const r of effectiveRuns) {
      if (typeof r.best_f1 === "number" && r.best_f1 > (best?.best_f1 ?? -1)) best = r;
    }
    return best;
  }, [effectiveRuns]);

  // Fetch detailed metrics for the best run
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [bestMetrics, setBestMetrics] = useState<any>(null);
  useEffect(() => {
    if (!projectId || !bestRun) { setBestMetrics(null); return; }
    let cancelled = false;
    fetchRunMetrics(projectId, bestRun.run_id).then((m) => { if (!cancelled) setBestMetrics(m?.metrics ?? m); }).catch(() => {});
    return () => { cancelled = true; };
  }, [projectId, bestRun?.run_id, bestRun?.best_f1, bestRun?.updated_at]); // eslint-disable-line react-hooks/exhaustive-deps

  // Delta vs previous best
  const f1Delta = useMemo(() => {
    if (!bestRun || typeof bestRun.best_f1 !== "number") return null;
    let secondBest = -1;
    for (const r of effectiveRuns) {
      if (r.run_id === bestRun.run_id) continue;
      if (typeof r.best_f1 === "number" && r.best_f1 > secondBest) secondBest = r.best_f1;
    }
    return secondBest >= 0 ? bestRun.best_f1 - secondBest : null;
  }, [bestRun, effectiveRuns]);

  // Convergence status
  const convergenceStatus = useMemo(() => {
    if (!bestMetrics) return null;
    const bestEpoch = bestMetrics.best_epoch as number | undefined;
    // epochs_effective, not dataset_stats.epochs: the latter is the REQUESTED
    // cap, and the convergence extension can run well past it. A run extended
    // from 30 to 90 with its best at epoch 40 was being called "undertrained"
    // because 40 >= 30 - 2.
    const totalEpochs = (bestMetrics.epochs_effective as number | undefined)
      ?? (bestMetrics.dataset_stats?.epochs as number | undefined);
    // The overfit branch used to live here, comparing F1_train against
    // best_F1_val. It could never fire: train.py initialises train_f1 = 0.0 and
    // never reassigns it, so metrics.json always carries F1_train: 0.0 and
    // `0.0 - f1Val > 0.10` is false for every f1Val >= 0. Rather than keep a
    // promise the data cannot support, the verdict reports only what it can
    // actually determine. Restoring it needs train_f1 populated from a
    // train-split evaluation measured under the same regime as best_F1_val.
    if (typeof bestEpoch === "number" && typeof totalEpochs === "number" && bestEpoch >= totalEpochs - 2) {
      return "undertrained" as const;
    }
    return "converged" as const;
  }, [bestMetrics]);

  // Per-class F1 from metrics
  const perClassF1 = useMemo(() => {
    if (!bestMetrics?.per_class_f1_val) return [];
    const entries = Object.entries(bestMetrics.per_class_f1_val as Record<string, number>)
      .map(([id, f1]) => ({ classId: parseInt(id), f1 }))
      .sort((a, b) => a.f1 - b.f1);
    return entries;
  }, [bestMetrics]);

  // Dataset stats
  const dsStats = bestMetrics?.dataset_stats as { num_train?: number; fg_ratio?: number; num_active_classes?: number } | null;

  return { bestRun, bestMetrics, f1Delta, convergenceStatus, perClassF1, dsStats };
}
