// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useState, type RefObject } from "react";
import { useMaskStore } from "../../store";
import {
  fetchAnnotateItems,
  fetchPrelabelCandidates,
  fetchRuns,
  prelabelRun,
  runPredictMaskUrl,
  type PrelabelProgress,
} from "../../api";
import type { TrainRunItem } from "../../utils";
import { decodeMaskBlob } from "../imageProcessing";
import { useI18n } from "../../i18n";
import type { RecipeOps } from "./useRecipe";
import { waitForAllPendingSaves } from "./useMaskIO";

/**
 * Draft annotations from a trained run.
 *
 * A prediction and an annotation are the same kind of thing -- a class id per
 * pixel -- so a finished run can hand the annotator a starting point instead
 * of a blank canvas.  Two scopes: the open image, previewed and confirmed like
 * any other assist, or every unannotated image at once, written server-side.
 */
export function usePrelabel(
  projectId: string | null,
  activeImageId: string | null,
  activeImageIdRef: RefObject<string | null>,
  width: number,
  height: number,
  getOps: () => RecipeOps,
  // The shared assist preview buffer: the canvas already draws it and drops it
  // on an image switch, so a draft behaves like every other assist.
  assistPreview: Uint8Array | null,
  setAssistPreview: (mask: Uint8Array | null) => void,
  setStatus: (msg: string) => void,
  setBusyMessage: (msg: string | null) => void,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setImages: (updater: (prev: any[]) => any[]) => void
) {
  const { t } = useI18n();
  const [prelabelRuns, setPrelabelRuns] = useState<TrainRunItem[]>([]);
  const [prelabelRunId, setPrelabelRunId] = useState<string | null>(null);
  // Distinguishes "the preview on screen is a draft" from a SAM or spot preview.
  const [prelabelPreviewActive, setPrelabelPreviewActive] = useState(false);
  const [isPrelabelRunning, setIsPrelabelRunning] = useState(false);
  const [prelabelUnannotated, setPrelabelUnannotated] = useState<number | null>(null);

  const applyDelta = useMaskStore((s) => s.applyDelta);

  // Only a run that finished has a model to predict with.
  useEffect(() => {
    if (!projectId) {
      setPrelabelRuns([]);
      setPrelabelRunId(null);
      return;
    }
    let cancelled = false;
    fetchRuns(projectId)
      .then((runs: TrainRunItem[]) => {
        if (cancelled) return;
        const usable = (runs || []).filter(
          (run) => run.status === "completed" && run.has_model !== false
        );
        setPrelabelRuns(usable);
        setPrelabelRunId((current) =>
          current && usable.some((run) => run.run_id === current)
            ? current
            : usable[0]?.run_id ?? null
        );
      })
      .catch((err: unknown) => console.warn("usePrelabel: run list failed:", err));
    return () => { cancelled = true; };
  }, [projectId]);

  // How many images a project-wide draft would touch.
  useEffect(() => {
    if (!projectId || !prelabelRunId) {
      setPrelabelUnannotated(null);
      return;
    }
    let cancelled = false;
    fetchPrelabelCandidates(projectId, prelabelRunId)
      .then((counts) => { if (!cancelled) setPrelabelUnannotated(counts.unannotated); })
      .catch(() => { if (!cancelled) setPrelabelUnannotated(null); });
    return () => { cancelled = true; };
  }, [projectId, prelabelRunId, isPrelabelRunning]);

  async function handlePrelabelPreview() {
    if (!projectId || !prelabelRunId || !activeImageId || !width || !height) return;
    const targetId = activeImageId;
    setIsPrelabelRunning(true);
    setAssistPreview(null);
    setPrelabelPreviewActive(false);
    setStatus(t("prelabel.predicting"));
    try {
      const res = await fetch(runPredictMaskUrl(projectId, prelabelRunId, targetId));
      if (!res.ok) {
        // A run with no checkpoint says so in the body; "HTTP 404" would not.
        let detail = `HTTP ${res.status}`;
        try { detail = (await res.json()).detail ?? detail; } catch { /* keep the status */ }
        throw new Error(detail);
      }
      const mask = await decodeMaskBlob(await res.blob(), width, height);
      // The user may have moved on while the model was running.
      if (activeImageIdRef.current !== targetId) return;
      setAssistPreview(mask);
      setPrelabelPreviewActive(true);
      const fgCount = mask.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
      setStatus(t("prelabel.previewed").replace("{px}", String(fgCount)));
    } catch (err) {
      setStatus(`${t("prelabel.title")}: ${(err as Error).message}`);
    } finally {
      setIsPrelabelRunning(false);
    }
  }

  function handlePrelabelConfirm() {
    if (!assistPreview) return;
    const ops = getOps();
    const curMask = useMaskStore.getState().maskIndex;
    const indices: number[] = [], prev: number[] = [], next: number[] = [];
    for (let i = 0; i < assistPreview.length; i++) {
      if (assistPreview[i] > 0 && assistPreview[i] !== curMask[i]) {
        indices.push(i);
        prev.push(curMask[i]);
        next.push(assistPreview[i]);
      }
    }
    if (indices.length > 0) {
      const idxArr = new Uint32Array(indices);
      ops.markTouched(idxArr);
      applyDelta(idxArr, new Uint8Array(prev), new Uint8Array(next));
      ops.markDirty();
    }
    setAssistPreview(null);
    setPrelabelPreviewActive(false);
    setStatus(t("prelabel.confirmed").replace("{px}", String(indices.length)));
  }

  function handlePrelabelCancel() {
    setAssistPreview(null);
    setPrelabelPreviewActive(false);
    setStatus("");
  }

  async function handlePrelabelAll() {
    if (!projectId || !prelabelRunId) return;
    const ops = getOps();
    setIsPrelabelRunning(true);
    setBusyMessage(t("prelabel.applyingAll"));
    try {
      // Flush every dirty mask and let in-flight PUTs settle, so no local edit
      // lands after the server writes and silently reverts it.
      await ops.flushDirtyMasks();
      await waitForAllPendingSaves();
      let summary: PrelabelProgress["summary"] | undefined;
      let runError: string | undefined;
      await prelabelRun(projectId, prelabelRunId, (progress) => {
        if (progress.error) runError = progress.error;
        else if (progress.summary) summary = progress.summary;
        else setBusyMessage(
          t("prelabel.progress")
            .replace("{done}", String(progress.done))
            .replace("{total}", String(progress.total))
        );
      });
      if (runError) {
        // Nothing was written, so no cache is stale and nothing needs reloading.
        setStatus(`${t("prelabel.title")}: ${runError}`);
        return;
      }
      setStatus(
        t("prelabel.appliedAll")
          .replace("{written}", String(summary?.written ?? 0))
          .replace("{skipped}", String(summary?.skipped ?? 0))
      );
      // The server rewrote an unknown set of masks: every cached client copy
      // is now suspect, and the canvas must show the server's answer.
      ops.invalidateMaskCaches();
      await ops.reloadActiveMask();
      const data = await fetchAnnotateItems(projectId);
      setImages(() => data.items || []);
    } catch (err) {
      setStatus(`${t("prelabel.title")}: ${(err as Error).message}`);
    } finally {
      setIsPrelabelRunning(false);
      setBusyMessage(null);
    }
  }

  return {
    prelabelRuns,
    prelabelRunId,
    setPrelabelRunId,
    // Only a live buffer counts: switching image clears the preview but not the flag.
    prelabelPreviewActive: prelabelPreviewActive && assistPreview !== null,
    isPrelabelRunning,
    prelabelUnannotated,
    handlePrelabelPreview,
    handlePrelabelConfirm,
    handlePrelabelCancel,
    handlePrelabelAll,
  };
}
