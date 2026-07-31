// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useMaskStore } from "../../store";
import type { SamRefValue, SpotDetectRefValue, SuperpixelRefValue, CrackTraceRefValue } from "./useDrawingEvents";
import { analyzeMaskForSpotDetect, autoOptimizeSensitivity, thresholdSpots } from "../imageProcessing";

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
  colorTolerance: number,
  setSpotSensitivity?: (v: number) => void,
  setSpotPhase?: (phase: "idle" | "sample" | "detect") => void,
) {
  // ---- store (individual selectors to avoid re-renders on unrelated state) ----
  const applyDelta = useMaskStore((s) => s.applyDelta);

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
    setStatus(`SAM confirmed: ${indices.length}px applied`);
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
    setStatus(`Spot Detect: ${spotCount} spots, ${indices.length}px applied`);
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
      if (assistPreview[i] !== curMask[i]) {
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
    setStatus(`Superpixel confirmed: ${indices.length}px applied`);
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
    setStatus(`Crack: ${selCount} cracks, ${indices.length}px applied`);
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
  // handleSpotRunDetect — frontend-only detection using Web Worker
  // ---------------------------------------------------------------------------
  async function handleSpotRunDetect() {
    const ref = spotDetectRef.current;
    if (!ref || !ref.sampleMask || !ref.imageData || ref.imageData.length === 0) {
      setStatus("Spot Detect: no samples painted");
      return;
    }
    if (width === 0 || height === 0) {
      setStatus("Spot Detect: no image loaded");
      return;
    }
    const activeClassId = useMaskStore.getState().activeClassId;
    const sampleMask = ref.sampleMask;
    const pixels = ref.imageData;

    setStatus("Spot Detect: analyzing...");
    const t0 = performance.now();

    try {
      // 1. Analyze painted samples to determine best channel, color, size range
      const analysis = analyzeMaskForSpotDetect(pixels, width, height, sampleMask);
      const channel = analysis?.channel ?? "gray";
      const targetLab = analysis?.targetLab ?? null;
      const autoColorTolerance = analysis?.autoColorTolerance ?? 20;
      const sizeRange = analysis?.sizeRange;
      const bestSnr = analysis?.bestSnr ?? 0;

      // Decide mode: high color SNR (≥2.5) → color distance, else DoG
      const useColorMode = bestSnr >= 2.5 && !!targetLab;

      // 2. Downscale if image is large
      // Color mode: no downscale (O(n), no blur) — preserves 1px spots
      // DoG mode: downscale to 1280px (blur is expensive at full res)
      const MAX_EDGE = useColorMode ? 999999 : 1280;
      const maxDim = Math.max(width, height);
      const imgScale = maxDim > MAX_EDGE ? MAX_EDGE / maxDim : 1;
      const dw = Math.round(width * imgScale);
      const dh = Math.round(height * imgScale);

      let workerPixels: Uint8ClampedArray;
      if (imgScale < 1) {
        const srcBitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(pixels), width, height));
        const osc = new OffscreenCanvas(dw, dh);
        const ctx = osc.getContext("2d")!;
        ctx.drawImage(srcBitmap, 0, 0, dw, dh);
        srcBitmap.close();
        workerPixels = ctx.getImageData(0, 0, dw, dh).data;
      } else {
        workerPixels = pixels;
      }

      // Size range: for color mode, allow 1px minimum (tiny spots)
      // The painted sample size is brush size, not actual spot size
      const colorSizeRange: [number, number] | undefined = sizeRange
        ? [1, Math.max(sizeRange[1], 500)]
        : [1, 2000];

      const scaledSizeRange: [number, number] | undefined = useColorMode
        ? colorSizeRange
        : sizeRange
          ? [Math.max(1, Math.round(sizeRange[0] * imgScale * imgScale)), Math.round(sizeRange[1] * imgScale * imgScale)]
          : undefined;

      const worker = new Worker(new URL("../spotDetectWorker.ts", import.meta.url), { type: "module" });

      if (useColorMode) {
        // ====== COLOR DISTANCE MODE ======
        // 3a. Compute color distance map in Worker
        const colorResult = await new Promise<{ distMap: Float32Array; mean: number }>((resolve, reject) => {
          worker.onmessage = (ev) => {
            if (ev.data.kind === "colorDist") resolve({ distMap: ev.data.distMap, mean: ev.data.mean });
          };
          worker.onerror = (err) => reject(new Error(err.message));
          worker.postMessage({ kind: "colorDist", pixels: workerPixels, width: dw, height: dh, targetLab, reqId: 1 });
        });

        // 4a. Auto-optimize tolerance by IoU against sample mask
        let downSampleMask: Uint8Array;
        if (imgScale < 1) {
          downSampleMask = new Uint8Array(dw * dh);
          for (let y = 0; y < dh; y++) {
            const srcY = Math.min(height - 1, Math.round(y / imgScale));
            for (let x = 0; x < dw; x++) {
              downSampleMask[y * dw + x] = sampleMask[Math.min(width - 1, Math.round(x / imgScale)) + srcY * width] > 0 ? 1 : 0;
            }
          }
        } else {
          downSampleMask = new Uint8Array(sampleMask.length);
          for (let i = 0; i < sampleMask.length; i++) downSampleMask[i] = sampleMask[i] > 0 ? 1 : 0;
        }

        // Grid search: try tolerances from autoColorTolerance*0.5 to autoColorTolerance*3
        let bestTol = autoColorTolerance;
        let bestIoU = -1;
        const n = dw * dh;
        for (let m = 0.3; m <= 3.0; m += 0.3) {
          const tol = autoColorTolerance * m;
          let inter = 0, union = 0;
          for (let i = 0; i < n; i++) {
            const gt = downSampleMask[i] > 0;
            const pred = colorResult.distMap[i] <= tol;
            if (gt && pred) inter++;
            if (gt || pred) union++;
          }
          const iou = union > 0 ? inter / union : 0;
          if (iou > bestIoU) { bestIoU = iou; bestTol = tol; }
        }

        // 5a. Final threshold via Worker (with connected components + size filter)
        const threshResult = await new Promise<{ mask: Uint8Array; count: number }>((resolve, reject) => {
          worker.onmessage = (ev) => {
            if (ev.data.kind === "colorThreshold") {
              if (ev.data.error) { reject(new Error(ev.data.error)); return; }
              resolve({ mask: ev.data.mask, count: ev.data.count });
            }
          };
          worker.onerror = (err) => reject(new Error(err.message));
          worker.postMessage({ kind: "colorThreshold", tolerance: bestTol, classId: activeClassId, sizeRange: scaledSizeRange, reqId: 2 });
        });

        worker.terminate();

        // 6a. Upscale
        let finalMask: Uint8Array;
        if (imgScale < 1) {
          finalMask = new Uint8Array(width * height);
          for (let y = 0; y < height; y++) {
            const srcY = Math.min(dh - 1, Math.round(y * imgScale));
            for (let x = 0; x < width; x++) {
              finalMask[y * width + x] = threshResult.mask[srcY * dw + Math.min(dw - 1, Math.round(x * imgScale))];
            }
          }
        } else {
          finalMask = threshResult.mask;
        }

        const elapsed = performance.now() - t0;

        // 7a. For slider: compute full-res color dist map if feasible
        const fullResPixelCount = width * height;
        let sliderDistMap: Float32Array;
        let sliderW = dw, sliderH = dh;
        if (fullResPixelCount <= 2_000_000 && imgScale < 1) {
          // Recompute at full res
          const worker2 = new Worker(new URL("../spotDetectWorker.ts", import.meta.url), { type: "module" });
          const fullColor = await new Promise<{ distMap: Float32Array }>((resolve, reject) => {
            worker2.onmessage = (ev) => { if (ev.data.kind === "colorDist") resolve({ distMap: ev.data.distMap }); };
            worker2.onerror = (err) => reject(new Error(err.message));
            worker2.postMessage({ kind: "colorDist", pixels, width, height, targetLab, reqId: 3 });
          });
          worker2.terminate();
          sliderDistMap = fullColor.distMap;
          sliderW = width; sliderH = height;
        } else {
          sliderDistMap = colorResult.distMap;
        }

        // 8a. Update ref
        const isDownscaled = sliderW !== width;
        const newRef: NonNullable<SpotDetectRefValue> = {
          imageData: isDownscaled ? workerPixels : pixels,
          scores: new Float32Array(0), std: 0, extrema: new Uint32Array(0),
          targetLab,
          phase: "detect",
          sampleMask,
          sizeRange: isDownscaled ? scaledSizeRange : colorSizeRange,
          downscaleWidth: isDownscaled ? sliderW : undefined,
          downscaleHeight: isDownscaled ? sliderH : undefined,
          autoColorTolerance: bestTol,
          mode: "color",
          colorDistMap: sliderDistMap,
        };
        (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = newRef;
        setSpotCount(threshResult.count);
        setAssistPreview(finalMask);
        if (setSpotSensitivity) setSpotSensitivity(bestTol);
        setSpotPhase?.("detect");

        const sizeLabel = sizeRange ? ` sz:${sizeRange[0]}-${sizeRange[1]}` : "";
        setStatus(
          `Spot Detect: ${threshResult.count} spots (color mode [${channel}] ΔE≤${Math.round(bestTol)} ` +
          `IoU=${(bestIoU * 100).toFixed(0)}%${sizeLabel} ${elapsed.toFixed(0)}ms)`
        );

      } else {
        // ====== DoG MODE (original) ======
        const computeResult = await new Promise<{
          scores: Float32Array; std: number; extrema: Uint32Array;
        }>((resolve, reject) => {
          worker.onmessage = (ev) => {
            if (ev.data.kind === "compute") resolve({ scores: ev.data.scores, std: ev.data.std, extrema: ev.data.extrema });
          };
          worker.onerror = (err) => reject(new Error(err.message));
          worker.postMessage({ kind: "compute", pixels: workerPixels, width: dw, height: dh, channel, reqId: 1 });
        });

        let downSampleMask: Uint8Array;
        if (imgScale < 1) {
          downSampleMask = new Uint8Array(dw * dh);
          for (let y = 0; y < dh; y++) {
            const srcY = Math.min(height - 1, Math.round(y / imgScale));
            for (let x = 0; x < dw; x++) {
              downSampleMask[y * dw + x] = sampleMask[Math.min(width - 1, Math.round(x / imgScale)) + srcY * width] > 0 ? activeClassId : 0;
            }
          }
        } else {
          downSampleMask = sampleMask;
        }

        const { bestSensitivity, bestIoU } = autoOptimizeSensitivity(
          computeResult.scores, computeResult.std, computeResult.extrema,
          dw, dh, activeClassId, downSampleMask,
          workerPixels, targetLab, autoColorTolerance, scaledSizeRange,
        );

        const threshResult = await new Promise<{ mask: Uint8Array; count: number }>((resolve, reject) => {
          worker.onmessage = (ev) => {
            if (ev.data.kind === "threshold") {
              if (ev.data.error) { reject(new Error(ev.data.error)); return; }
              resolve({ mask: ev.data.mask, count: ev.data.count });
            }
          };
          worker.onerror = (err) => reject(new Error(err.message));
          worker.postMessage({ kind: "threshold", sensitivity: bestSensitivity, classId: activeClassId, colorTolerance: autoColorTolerance, targetLab, sizeRange: scaledSizeRange, reqId: 2 });
        });

        worker.terminate();

        let finalMask: Uint8Array;
        if (imgScale < 1) {
          finalMask = new Uint8Array(width * height);
          for (let y = 0; y < height; y++) {
            const srcY = Math.min(dh - 1, Math.round(y * imgScale));
            for (let x = 0; x < width; x++) {
              finalMask[y * width + x] = threshResult.mask[srcY * dw + Math.min(dw - 1, Math.round(x * imgScale))];
            }
          }
        } else {
          finalMask = threshResult.mask;
        }

        const elapsed = performance.now() - t0;

        const fullResPixelCount = width * height;
        let fullScores: Float32Array;
        let fullStd: number;
        let fullExtrema: Uint32Array;

        if (fullResPixelCount <= 2_000_000 || imgScale >= 1) {
          if (imgScale >= 1) {
            fullScores = computeResult.scores; fullStd = computeResult.std; fullExtrema = computeResult.extrema;
          } else {
            const worker2 = new Worker(new URL("../spotDetectWorker.ts", import.meta.url), { type: "module" });
            const fullRes = await new Promise<{ scores: Float32Array; std: number; extrema: Uint32Array }>((resolve, reject) => {
              worker2.onmessage = (ev) => { if (ev.data.kind === "compute") resolve({ scores: ev.data.scores, std: ev.data.std, extrema: ev.data.extrema }); };
              worker2.onerror = (err) => reject(new Error(err.message));
              worker2.postMessage({ kind: "compute", pixels, width, height, channel, reqId: 3 });
            });
            worker2.terminate();
            fullScores = fullRes.scores; fullStd = fullRes.std; fullExtrema = fullRes.extrema;
          }
        } else {
          fullScores = computeResult.scores; fullStd = computeResult.std; fullExtrema = computeResult.extrema;
        }

        const isDownscaled = imgScale < 1 && fullResPixelCount > 2_000_000;
        const newRef: NonNullable<SpotDetectRefValue> = {
          imageData: isDownscaled ? workerPixels : pixels,
          scores: fullScores, std: fullStd, extrema: fullExtrema,
          targetLab,
          phase: "detect",
          sampleMask,
          sizeRange: isDownscaled ? scaledSizeRange : sizeRange,
          downscaleWidth: isDownscaled ? dw : undefined,
          downscaleHeight: isDownscaled ? dh : undefined,
          autoColorTolerance,
          mode: "dog",
        };
        (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = newRef;
        setSpotCount(threshResult.count);
        setAssistPreview(finalMask);
        if (setSpotSensitivity) setSpotSensitivity(bestSensitivity);
        setSpotPhase?.("detect");

        const sizeLabel = sizeRange ? ` sz:${sizeRange[0]}-${sizeRange[1]}` : "";
        const channelLabel = channel === "gray" ? "" : ` [${channel}]`;
        setStatus(
          `Spot Detect: ${threshResult.count} spots (DoG${channelLabel} sens=${bestSensitivity} ` +
          `IoU=${(bestIoU * 100).toFixed(0)}%${sizeLabel} ΔE≤${Math.round(autoColorTolerance)} ${elapsed.toFixed(0)}ms)`
        );
      }
    } catch (err) {
      setStatus(`Spot Detect failed: ${(err as Error).message}`);
    }
  }

  // ---------------------------------------------------------------------------
  // Return
  // ---------------------------------------------------------------------------
  return {
    // functions
    handleSamConfirm,
    handleSamCancel,
    handleSpotConfirm,
    handleSpotCancel,
    handleSpotRunDetect,
    handleSuperpixelConfirm,
    handleSuperpixelCancel,
    handleCrackConfirm,
    handleCrackCancel,
  };
}
