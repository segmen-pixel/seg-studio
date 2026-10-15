// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useMaskStore } from "../../store";
import { useI18n } from "../../i18n";
import type { SamRefValue, SpotDetectRefValue, SuperpixelRefValue, CrackTraceRefValue } from "./useDrawingEvents";
import { markPointsFrom, runSpotDetect } from "./toolActions";

/**
 * Coalesces a slider drag into one request. Module level because one annotator
 * is on screen at a time, and the timer has to outlive a render.
 */
let spotSliderTimer: number | null = null;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AiAssistOps = {
  markDirty: () => void;
  markTouched: (indices: Uint32Array) => void;
  autoSave: (imageId: string) => Promise<void>;
};

export type PrepareReport = {
  train_count: number;
  val_count: number;
  with_mask: number;
  auto_val_from_train_count?: number;
} | null;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAiAssist(
  projectId: string | null,
  activeImageId: string | null,
  activeImageIdRef: React.MutableRefObject<string | null>,
  width: number,
  height: number,
  assistPreview: Uint8Array | null,
  setAssistPreview: (v: Uint8Array | null) => void,
  getOps: () => AiAssistOps,
  samRef: React.MutableRefObject<SamRefValue>,
  samBoxDraftRef: React.MutableRefObject<{ start: [number, number]; end: [number, number] } | null>,
  spotDetectRef: React.MutableRefObject<SpotDetectRefValue>,
  superpixelRef: React.MutableRefObject<SuperpixelRefValue>,
  crackTraceRef: React.MutableRefObject<CrackTraceRefValue>,
  spotCount: number,
  setSpotCount: (v: number) => void,
  setStatus: (msg: string) => void,
  spotSensitivity: number,
  setSpotSensitivity?: (v: number) => void,
  setSpotPhase?: (phase: "idle" | "sample" | "detect") => void,
) {
  // ---- store (individual selectors to avoid re-renders on unrelated state) ----
  const applyDelta = useMaskStore((s) => s.applyDelta);
  const { t } = useI18n();

  // ---------------------------------------------------------------------------
  // handleSamConfirm
  // ---------------------------------------------------------------------------
  function handleSamConfirm() {
    if (!assistPreview || !samRef.current) return;
    const ops = getOps();
    const curMask = useMaskStore.getState().maskIndex;
    const indices: number[] = [],
      prev: number[] = [],
      next: number[] = [];
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
    (samRef as React.MutableRefObject<SamRefValue>).current = null;
    setStatus(t("aiAssist.status.samConfirmed").replace("{px}", String(indices.length)));
  }

  // ---------------------------------------------------------------------------
  // handleSamInvert
  // ---------------------------------------------------------------------------
  /** Keep what SAM did not choose, before anything is written.
   *
   * SAM answers a click with a region, and sometimes the region is the
   * surround rather than the thing -- a pale mark on a dark surface and a dark
   * mark on a pale one are the same question to it. Until now that meant
   * cancelling and clicking again somewhere else. The complement is already in
   * hand, so it is offered instead.
   *
   * Bounded by the box when the box tool drew one: the person said where they
   * were looking, and inverting past that would hand back the whole picture.
   * Without a box the complement is the frame, which is what "not this" means
   * when nothing narrowed it.
   */
  function handleSamInvert() {
    if (!assistPreview || !samRef.current) return;
    let cls = 0;
    for (let i = 0; i < assistPreview.length; i++) {
      if (assistPreview[i] > 0) { cls = assistPreview[i]; break; }
    }
    if (!cls) return;
    const box = samRef.current.box;
    const x0 = box ? Math.max(0, Math.min(box[0], box[2])) : 0;
    const x1 = box ? Math.min(width, Math.max(box[0], box[2])) : width;
    const y0 = box ? Math.max(0, Math.min(box[1], box[3])) : 0;
    const y1 = box ? Math.min(height, Math.max(box[1], box[3])) : height;
    const next = new Uint8Array(assistPreview.length);
    let n = 0;
    for (let y = y0; y < y1; y++) {
      const row = y * width;
      for (let x = x0; x < x1; x++) {
        const i = row + x;
        if (assistPreview[i] === 0) { next[i] = cls; n++; }
      }
    }
    setAssistPreview(next);
    setStatus(
      t("aiAssist.status.samInverted")
        .replace("{px}", String(n))
        .replace("{scope}", t(box ? "aiAssist.scope.box" : "aiAssist.scope.frame")),
    );
  }

  // ---------------------------------------------------------------------------
  // handleSamCancel
  // ---------------------------------------------------------------------------
  function handleSamCancel() {
    setAssistPreview(null);
    (samRef as React.MutableRefObject<SamRefValue>).current = null;
    samBoxDraftRef.current = null;
    setStatus("");
  }

  // ---------------------------------------------------------------------------
  // handleSpotConfirm
  // ---------------------------------------------------------------------------
  function handleSpotConfirm() {
    if (!assistPreview || !spotDetectRef.current) return;
    const ops = getOps();
    const curMask = useMaskStore.getState().maskIndex;
    const indices: number[] = [],
      prev: number[] = [],
      next: number[] = [];
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
    (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = null;
    setStatus(t("aiAssist.status.spotConfirmed").replace("{n}", String(spotCount)).replace("{px}", String(indices.length)));
  }

  // ---------------------------------------------------------------------------
  // handleSpotCancel
  // ---------------------------------------------------------------------------
  function handleSpotCancel() {
    setAssistPreview(null);
    (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = null;
    setSpotPhase?.("idle");
    setStatus("");
  }

  // ---------------------------------------------------------------------------
  // handleSuperpixelConfirm
  // ---------------------------------------------------------------------------
  function handleSuperpixelConfirm() {
    if (!assistPreview || !superpixelRef.current) return;
    const ops = getOps();
    const curMask = useMaskStore.getState().maskIndex;
    const indices: number[] = [],
      prev: number[] = [],
      next: number[] = [];
    for (let i = 0; i < assistPreview.length; i++) {
      // Preview pixels outside the selected segments are 0 — without the
      // > 0 guard the confirm would erase every existing annotation there.
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
    (superpixelRef as React.MutableRefObject<SuperpixelRefValue>).current = null;
    setStatus(t("aiAssist.status.superpixelConfirmed").replace("{px}", String(indices.length)));
  }

  // ---------------------------------------------------------------------------
  // handleSuperpixelCancel
  // ---------------------------------------------------------------------------
  function handleSuperpixelCancel() {
    setAssistPreview(null);
    (superpixelRef as React.MutableRefObject<SuperpixelRefValue>).current = null;
    setStatus("");
  }

  // ---------------------------------------------------------------------------
  // handleCrackConfirm
  // ---------------------------------------------------------------------------
  function handleCrackConfirm() {
    if (!assistPreview || !crackTraceRef.current) return;
    const selCount = crackTraceRef.current.selections.size;
    const ops = getOps();
    const curMask = useMaskStore.getState().maskIndex;
    const indices: number[] = [],
      prev: number[] = [],
      next: number[] = [];
    for (let i = 0; i < assistPreview.length; i++) {
      if (assistPreview[i] > 0 && assistPreview[i] !== 254 && assistPreview[i] !== curMask[i]) {
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
    (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = null;
    setStatus(t("aiAssist.status.crackConfirmed").replace("{n}", String(selCount)).replace("{px}", String(indices.length)));
  }

  // ---------------------------------------------------------------------------
  // handleCrackCancel
  // ---------------------------------------------------------------------------
  function handleCrackCancel() {
    setAssistPreview(null);
    (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = null;
    setStatus("");
  }

  // ---------------------------------------------------------------------------
  // handleSpotRunDetect — the detector runs on the server
  // ---------------------------------------------------------------------------
  // It used to run here, in two copies that disagreed about their own rule: the
  // sensitivity search flood-filled at one threshold while the worker that made
  // the mask filled at half of it, so the search scored an operator that never
  // ran. The count it tuned for and the specks the mask then held could differ
  // many times over, and the status line reported the score of a mask nobody
  // saw. What is left here is the painted example going out and a mask coming
  // back; no score map is held in the tab.
  async function runSpot(sensitivity?: number) {
    const ref = spotDetectRef.current;
    if (!ref || !ref.sampleMask) {
      setStatus(t("aiAssist.status.spotNoSamples"));
      return;
    }
    const startImageId = activeImageIdRef.current;
    if (!projectId || !startImageId || width === 0 || height === 0) {
      setStatus(t("aiAssist.status.spotNoImage"));
      return;
    }
    const activeClassId = useMaskStore.getState().activeClassId;
    const points = markPointsFrom(ref.sampleMask, width);
    if (points.length === 0) {
      setStatus(t("aiAssist.status.spotNoSamples"));
      return;
    }
    const sampleMask = ref.sampleMask;
    setStatus(t("aiAssist.status.spotAnalyzing"));
    try {
      const out = await runSpotDetect(
        projectId, startImageId, points, activeClassId, width, height, sensitivity);
      // The image can change while the server is working; that answer is for a
      // picture the user is no longer looking at.
      if (activeImageIdRef.current !== startImageId) return;
      if (out.result.why) {
        setStatus(`Spot Detect: ${out.result.why}`);
        return;
      }
      (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = {
        phase: "detect",
        sampleMask,
        mode: out.result.mode,
        sizeRange: out.result.size_range,
        colorTolerance: out.result.color_tolerance,
        sensitivityRange: out.result.sensitivity_range,
      };
      setAssistPreview(out.preview);
      setSpotCount(out.count);
      if (setSpotSensitivity) setSpotSensitivity(out.result.sensitivity);
      setSpotPhase?.("detect");
      const cached = out.result.scores_cached ? "" : " first run";
      setStatus(
        `Spot Detect: ${out.count} spots (${out.result.mode} sens=${out.result.sensitivity} ` +
        `sz:${out.result.size_range[0]}-${out.result.size_range[1]} ` +
        `ΔE≤${Math.round(out.result.color_tolerance)} ${out.result.time_ms}ms${cached})`
      );
    } catch (err) {
      setStatus(t("aiAssist.status.spotFailed").replace("{error}", (err as Error).message));
    }
  }

  async function handleSpotRunDetect() {
    await runSpot();
  }

  // The slider is a round trip now. The server keeps the score map per image so
  // the call is a fraction of the first one, but a drag would still send a
  // request per pixel of travel, so let the drag settle first.
  function handleSpotSensitivity(sensitivity: number) {
    if (spotSliderTimer !== null) window.clearTimeout(spotSliderTimer);
    spotSliderTimer = window.setTimeout(() => {
      spotSliderTimer = null;
      void runSpot(sensitivity);
    }, 250);
  }

  // ---------------------------------------------------------------------------
  // Return
  // ---------------------------------------------------------------------------
  return {
    // functions
    handleSamConfirm,
    handleSamInvert,
    handleSamCancel,
    handleSpotConfirm,
    handleSpotCancel,
    handleSpotRunDetect,
    handleSpotSensitivity,
    handleSuperpixelConfirm,
    handleSuperpixelCancel,
    handleCrackConfirm,
    handleCrackCancel,
  };
}
