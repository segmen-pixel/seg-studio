// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef, useState } from "react";
import { fetchLibraryStats, modelSearch } from "../../api";
import type { TranslationKey } from "../../i18n";

// Extracted verbatim from Training.tsx (pre-OSS refactor): model-search
// state (search progress, result text, abort control), the combo-library
// stats fetch + library-changed listener, and the warmup-anchor ETA
// recalibration flow.
export function useModelSearch(
  projectId: string | null,
  lang: string,
  t: (key: TranslationKey) => string,
) {
  const [isSearchingModel, setIsSearchingModel] = useState(false);
  const [modelSearchResult, setModelSearchResult] = useState("");
  const modelSearchAbortRef = useRef<AbortController | null>(null);
  const [libraryStats, setLibraryStats] = useState<{ total_profiles: number; total_projects: number; bestMatch?: number } | null>(null);

  // Fetch library stats on mount + listen for library-changed events
  useEffect(() => {
    refreshLibraryStats();
    const handler = () => refreshLibraryStats();
    window.addEventListener("library-changed", handler);
    return () => window.removeEventListener("library-changed", handler);
  }, []);

  function refreshLibraryStats(bestMatch?: number) {
    fetchLibraryStats().then(stats => setLibraryStats({ ...stats, bestMatch })).catch(() => {});
  }

  function handleCancelModelSearch() {
    modelSearchAbortRef.current?.abort();
    modelSearchAbortRef.current = null;
    setIsSearchingModel(false);
    setModelSearchResult(t("training.modelSearch.cancelled"));
  }

  // Last anchor_elapsed_sec used for warmup-calibration of training-time
  // predictions.  Persisted in component state so the user can replay
  // model-search after running the anchor combo for the first time.
  const [anchorElapsedSec, setAnchorElapsedSec] = useState<number | null>(null);

  async function handleModelSearch(opts: { anchorElapsedSec?: number | null } = {}) {
    if (!projectId || isSearchingModel) return;
    const abort = new AbortController();
    modelSearchAbortRef.current = abort;
    setIsSearchingModel(true);
    setModelSearchResult("Searching...");
    const effectiveAnchor = opts.anchorElapsedSec ?? anchorElapsedSec ?? undefined;
    try {
      const result = await modelSearch(projectId, abort.signal, {
        anchorElapsedSec: effectiveAnchor ?? undefined,
      });
      if (abort.signal.aborted) return;
      if (result.found === 0) {
        setModelSearchResult(t("training.modelSearch.noResults"));
      } else {
        const lines = result.matches.map((m: { project_name: string; similarity: number; arch: string; best_f1: number; checkpoint_exists: boolean }) =>
          `  ${m.project_name} (sim=${(m.similarity * 100).toFixed(0)}%, arch=${m.arch}, F1=${m.best_f1.toFixed(3)}${m.checkpoint_exists ? " ✓" : ""})`
        );
        let summary =
          `Found ${result.found} similar projects:\n` +
          lines.join("\n") +
          `\n\n→ Recommended: arch=${result.target_arch}, epochs=${result.recommended_epochs}, confidence=${result.confidence}`;
        const cr = result.config_recommendation;
        if (cr) {
          summary += `\n\n→ Auto-config: ${cr.arch} bc=${cr.base_channels} p=${cr.patch_size} (confidence=${cr.confidence})`;
          // v6 Phase 6 — surface the warmup-calibrated training-time estimate.
          if (cr.pred_elapsed_min != null) {
            const tag = cr.time_calibrated ? "calibrated" : "physical-only";
            summary += `\n  ⏱ estimated training time: ~${cr.pred_elapsed_min.toFixed(1)} min (${tag})`;
            if (!cr.time_calibrated && cr.time_anchor_combo) {
              summary += `\n  → for an accurate ETA, run the warmup anchor combo first:`;
              summary += `\n      ${cr.time_anchor_combo}`;
              summary += `\n     then click "Recalibrate ETAs" with its actual elapsed seconds.`;
            }
          }
          // v6 VRAM predictor — WDDM-aware OOM verdict for this GPU.
          if (cr.vram) {
            const drv = cr.vram.driver.toUpperCase();
            const peak = cr.vram.pred_vram_mb.toFixed(0);
            const budget = cr.vram.budget_mb.toFixed(0);
            const gpu = cr.vram.gpu_total_mb.toFixed(0);
            if (cr.vram.oom_risk) {
              summary += `\n  ⚠ VRAM: this combo may OOM on the current GPU `
                + `(${gpu} MB, ${drv}) — predicted peak ~${peak} MB vs `
                + `safe budget ${budget} MB. Use a smaller base_channels `
                + `or ce/focal loss.`;
            } else {
              summary += `\n  ✓ VRAM: fits the current GPU `
                + `(${gpu} MB, ${drv}) — predicted peak ~${peak} MB, `
                + `safe budget ${budget} MB.`;
            }
          }
        }
        setModelSearchResult(summary);
      }
      const bestMatch = result.found > 0 ? Math.max(...result.matches.map((m: { similarity: number }) => m.similarity)) : undefined;
      refreshLibraryStats(bestMatch);
    } catch (err) {
      if (!abort.signal.aborted) {
        setModelSearchResult(`Model search failed: ${(err as Error).message}`);
      }
    } finally {
      if (!abort.signal.aborted) setIsSearchingModel(false);
      modelSearchAbortRef.current = null;
    }
  }

  function handleRecalibrateETAs() {
    if (!projectId || isSearchingModel) return;
    const raw = window.prompt(
      t("training.modelSearch.anchorPrompt"),
      anchorElapsedSec ? String(anchorElapsedSec) : "",
    );
    if (raw == null) return;
    const v = Number(raw);
    if (!Number.isFinite(v) || v <= 0) {
      setModelSearchResult(t("training.modelSearch.invalidSeconds"));
      return;
    }
    setAnchorElapsedSec(v);
    void handleModelSearch({ anchorElapsedSec: v });
  }

  // Reset on project change (called from the project-switch effect in Training.tsx)
  function resetModelSearch() {
    setModelSearchResult("");
    setLibraryStats(null);
  }

  return {
    isSearchingModel,
    modelSearchResult,
    libraryStats,
    anchorElapsedSec,
    handleModelSearch,
    handleCancelModelSearch,
    handleRecalibrateETAs,
    resetModelSearch,
  };
}
