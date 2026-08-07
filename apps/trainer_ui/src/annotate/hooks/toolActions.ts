// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { rgbToLab, wandFlood, detectSpots, thresholdSpots, sampleClickLab, analyzeMaskForSpotDetect, autoOptimizeSensitivity, computeSpotScores } from "../imageProcessing";
import { samSegment, superpixelMap, crackTrace, crackTraceAdaptive } from "../../api";
import type { WandRefValue, SamRefValue, SpotDetectRefValue, SuperpixelRefValue, CrackTraceRefValue } from "./useDrawingEvents";
import type { SamModelId } from "../annotatorContext";

type BrushResult = {
  indices: number[];
  prev: number[];
  next: number[];
  dirty: { x: number; y: number; w: number; h: number } | null;
};

export function collectBrushStroke(
  from: [number, number], to: [number, number], value: number,
  brushSize: number, width: number, height: number,
  maskIndex: Uint8Array,
): BrushResult {
  const indices: number[] = [];
  const prev: number[] = [];
  const next: number[] = [];
  const radius = Math.max(1, Math.floor(brushSize / 2));
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const dist = Math.hypot(dx, dy);
  const step = Math.max(1, Math.floor(radius / 2));
  const steps = Math.max(1, Math.ceil(dist / step));
  let dirtyMinX = width, dirtyMinY = height, dirtyMaxX = -1, dirtyMaxY = -1;
  const radiusSq = radius * radius;
  const changed = new Map<number, number>();
  const stamp = (cx: number, cy: number) => {
    const sx = Math.max(0, cx - radius), ex = Math.min(width - 1, cx + radius);
    const sy = Math.max(0, cy - radius), ey = Math.min(height - 1, cy + radius);
    for (let y = sy; y <= ey; y += 1) {
      for (let x = sx; x <= ex; x += 1) {
        const ddx = x - cx, ddy = y - cy;
        if (ddx * ddx + ddy * ddy > radiusSq) continue;
        const idx = y * width + x;
        if (maskIndex[idx] === value || changed.has(idx)) continue;
        changed.set(idx, maskIndex[idx]);
        if (x < dirtyMinX) dirtyMinX = x;
        if (y < dirtyMinY) dirtyMinY = y;
        if (x > dirtyMaxX) dirtyMaxX = x;
        if (y > dirtyMaxY) dirtyMaxY = y;
      }
    }
  };
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 0 : i / steps;
    stamp(Math.round(from[0] + dx * t), Math.round(from[1] + dy * t));
  }
  if (changed.size > 0) {
    for (const [idx, prevValue] of changed.entries()) {
      indices.push(idx); prev.push(prevValue); next.push(value);
    }
  }
  if (indices.length === 0) return { indices, prev, next, dirty: null };
  return { indices, prev, next, dirty: { x: dirtyMinX, y: dirtyMinY, w: dirtyMaxX - dirtyMinX + 1, h: dirtyMaxY - dirtyMinY + 1 } };
}

export function floodFill(
  start: [number, number], value: number,
  width: number, height: number,
  maskIndex: Uint8Array,
): BrushResult | null {
  const target = maskIndex[start[1] * width + start[0]];
  if (target === value) return null;
  const indices: number[] = [], prev: number[] = [], next: number[] = [];
  const queue: [number, number][] = [start];
  const visited = new Uint8Array(maskIndex.length);
  let minX = width, minY = height, maxX = 0, maxY = 0;
  while (queue.length > 0) {
    const [x, y] = queue.pop()!;
    const idx = y * width + x;
    if (visited[idx]) continue;
    visited[idx] = 1;
    if (maskIndex[idx] !== target) continue;
    indices.push(idx); prev.push(maskIndex[idx]); next.push(value);
    if (x < minX) minX = x; if (y < minY) minY = y;
    if (x > maxX) maxX = x; if (y > maxY) maxY = y;
    if (x > 0) queue.push([x - 1, y]);
    if (x < width - 1) queue.push([x + 1, y]);
    if (y > 0) queue.push([x, y - 1]);
    if (y < height - 1) queue.push([x, y + 1]);
  }
  if (indices.length === 0) return null;
  return { indices, prev, next, dirty: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 } };
}

/**
 * Pick the connected component of same-classId pixels containing `start`
 * using 4-connectivity. Returns null when the clicked pixel is background.
 */
export function pickConnectedRegion(
  start: [number, number],
  width: number, height: number,
  maskIndex: Uint8Array,
): { indices: Uint32Array; classId: number; bbox: { x: number; y: number; w: number; h: number } } | null {
  const startIdx = start[1] * width + start[0];
  const target = maskIndex[startIdx];
  if (target === 0) return null;
  const visited = new Uint8Array(maskIndex.length);
  const stack: number[] = [startIdx];
  const out: number[] = [];
  visited[startIdx] = 1;
  let minX = start[0], minY = start[1], maxX = start[0], maxY = start[1];
  while (stack.length > 0) {
    const idx = stack.pop()!;
    if (maskIndex[idx] !== target) continue;
    out.push(idx);
    const x = idx % width;
    const y = (idx - x) / width;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (x > 0) { const ni = idx - 1; if (!visited[ni]) { visited[ni] = 1; stack.push(ni); } }
    if (x < width - 1) { const ni = idx + 1; if (!visited[ni]) { visited[ni] = 1; stack.push(ni); } }
    if (y > 0) { const ni = idx - width; if (!visited[ni]) { visited[ni] = 1; stack.push(ni); } }
    if (y < height - 1) { const ni = idx + width; if (!visited[ni]) { visited[ni] = 1; stack.push(ni); } }
  }
  return {
    indices: new Uint32Array(out),
    classId: target,
    bbox: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 },
  };
}

export function initWand(
  pos: [number, number], clientY: number,
  pixels: Uint8ClampedArray, width: number, height: number,
  activeClassId: number,
): { refValue: NonNullable<WandRefValue>; preview: Uint8Array; status: string } | null {
  const si = (pos[1] * width + pos[0]) * 4;
  const clickLab = rgbToLab(pixels[si], pixels[si + 1], pixels[si + 2]);
  const n = width * height;
  const distMap = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const bi = i * 4;
    const lab = rgbToLab(pixels[bi], pixels[bi + 1], pixels[bi + 2]);
    const dL = lab[0] - clickLab[0], da = lab[1] - clickLab[1], db2 = lab[2] - clickLab[2];
    distMap[i] = dL * dL + da * da + db2 * db2;
  }
  const patchR = 5;
  let sumSq = 0, count = 0;
  for (let dy = -patchR; dy <= patchR; dy++) {
    for (let dx = -patchR; dx <= patchR; dx++) {
      const px = pos[0] + dx, py = pos[1] + dy;
      if (px >= 0 && px < width && py >= 0 && py < height) {
        sumSq += distMap[py * width + px]; count++;
      }
    }
  }
  const sigma = Math.sqrt(sumSq / Math.max(1, count));
  const autoTol = Math.max(3, sigma * 2.5);
  const refValue: NonNullable<WandRefValue> = { startY: clientY, startPos: pos, distMap, autoTolerance: autoTol, tolerance: autoTol };
  const preview = wandFlood(pos, distMap, autoTol, activeClassId, width, height);
  return { refValue, preview, status: `Wand: tolerance=${autoTol.toFixed(0)} (auto)` };
}

export function initSpotDetect(
  pixels: Uint8ClampedArray, width: number, height: number,
  spotSensitivity: number, activeClassId: number,
  clickPos?: [number, number] | null, colorTolerance?: number,
  maskIndex?: Uint8Array,
): { refValue: NonNullable<SpotDetectRefValue>; mask: Uint8Array; count: number; status: string; autoSensitivity?: number } | null {
  let targetLab = clickPos ? sampleClickLab(pixels, width, height, clickPos[0], clickPos[1]) : null;
  const clickIdx = clickPos ? clickPos[1] * width + clickPos[0] : undefined;
  let channel: "gray" | "L" | "a" | "b" = "gray";
  let effectiveColorTolerance = colorTolerance;
  let sizeRange: [number, number] | undefined;
  let autoMode = false;
  let effectiveSensitivity = spotSensitivity;

  // Auto-analyze painted mask: painted pixels = spot color samples
  if (maskIndex) {
    const analysis = analyzeMaskForSpotDetect(pixels, width, height, maskIndex);
    if (analysis) {
      channel = analysis.channel;
      targetLab = analysis.targetLab;
      effectiveColorTolerance = analysis.autoColorTolerance;
      sizeRange = analysis.sizeRange;
      autoMode = true;

      // Auto-optimize sensitivity by maximizing mIoU against painted samples
      const { scores, std, extrema } = computeSpotScores(pixels, width, height, channel);
      const { bestSensitivity, bestIoU } = autoOptimizeSensitivity(
        scores, std, extrema, width, height, activeClassId,
        maskIndex, pixels, targetLab, effectiveColorTolerance, sizeRange,
      );
      effectiveSensitivity = bestSensitivity;

      const result = detectSpots(pixels, width, height, effectiveSensitivity, activeClassId, targetLab, effectiveColorTolerance, clickIdx, channel, sizeRange);
      const channelLabel = channel === "gray" ? "" : ` [${channel}]`;
      const sizeLabel = sizeRange ? ` sz:${sizeRange[0]}-${sizeRange[1]}` : "";
      return {
        refValue: { imageData: pixels, scores: result.scores, std: result.std, extrema: result.extrema, targetLab, clickIdx, phase: "detect" as const, sizeRange },
        mask: result.mask,
        count: result.count,
        status: `Spot Detect: ${result.count} spots (auto${channelLabel} sens=${effectiveSensitivity} IoU=${(bestIoU * 100).toFixed(0)}%${sizeLabel} ΔE≤${Math.round(effectiveColorTolerance!)})`,
        autoSensitivity: effectiveSensitivity,
      };
    }
  }

  const result = detectSpots(pixels, width, height, effectiveSensitivity, activeClassId, targetLab, effectiveColorTolerance, clickIdx, channel, sizeRange);
  return {
    refValue: { imageData: pixels, scores: result.scores, std: result.std, extrema: result.extrema, targetLab, clickIdx, phase: "detect" as const },
    mask: result.mask,
    count: result.count,
    status: targetLab
      ? `Spot Detect: ${result.count} spots (color filtered)`
      : `Spot Detect: ${result.count} spots`,
  };
}

export function refilterSpot(
  scores: Float32Array, std: number, extrema: Uint32Array,
  width: number, height: number,
  sensitivity: number, classId: number,
  pixels?: Uint8ClampedArray, targetLab?: [number, number, number] | null, colorTolerance?: number,
  clickIdx?: number,
  sizeRange?: [number, number],
): { mask: Uint8Array; count: number } {
  return thresholdSpots(scores, std, extrema, width, height, sensitivity, classId, pixels, targetLab, colorTolerance, clickIdx, sizeRange);
}

/** Re-threshold using precomputed color distance map. */
export function refilterColorSpot(
  distMap: Float32Array, width: number, height: number,
  tolerance: number, classId: number,
  sizeRange?: [number, number],
): { mask: Uint8Array; count: number } {
  const n = width * height;
  const mask = new Uint8Array(n);
  const visited = new Uint8Array(n);
  const minSize = sizeRange ? sizeRange[0] : 1;
  const maxSize = sizeRange ? sizeRange[1] : 2000;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (visited[i] || distMap[i] > tolerance) continue;
    const queue = [i];
    visited[i] = 1;
    let head = 0;
    while (head < queue.length) {
      const ci = queue[head++];
      const cx = ci % width;
      const cy = (ci - cx) / width;
      if (cx > 0 && !visited[ci - 1] && distMap[ci - 1] <= tolerance) { visited[ci - 1] = 1; queue.push(ci - 1); }
      if (cx < width - 1 && !visited[ci + 1] && distMap[ci + 1] <= tolerance) { visited[ci + 1] = 1; queue.push(ci + 1); }
      if (cy > 0 && !visited[ci - width] && distMap[ci - width] <= tolerance) { visited[ci - width] = 1; queue.push(ci - width); }
      if (cy < height - 1 && !visited[ci + width] && distMap[ci + width] <= tolerance) { visited[ci + width] = 1; queue.push(ci + width); }
    }
    if (queue.length < minSize || queue.length > maxSize) continue;
    for (let qi = 0; qi < queue.length; qi++) mask[queue[qi]] = classId;
    count++;
  }
  return { mask, count };
}

export function initSam(
  pos: [number, number], isBackground: boolean,
  samRefCurrent: SamRefValue,
  projectId: string, activeImageId: string,
  width: number, height: number, activeClassId: number,
  samModel: SamModelId,
  setAssistPreview: (p: Uint8Array | null) => void,
  setStatus: (s: string) => void,
  renderUi: (pos: [number, number] | null) => void,
): SamRefValue {
  const label = isBackground ? 0 : 1;
  const ref = samRefCurrent ?? { points: [], labels: [], box: null, pending: false };
  ref.points.push(pos);
  ref.labels.push(label);
  if (ref.pending) return ref;
  ref.pending = true;
  const statusParts = [`${ref.points.length} pts`];
  if (ref.box) statusParts.push("box");
  setStatus(`SAM: ${statusParts.join(" + ")}, predicting...`);
  samSegment(projectId, activeImageId, ref.points, ref.labels, samModel, ref.box)
    .then((result) => {
      if (!ref) return;
      ref.pending = false;
      const img = new window.Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = width; c.height = height;
        const ctx2 = c.getContext("2d");
        if (!ctx2) return;
        ctx2.drawImage(img, 0, 0, width, height);
        const data = ctx2.getImageData(0, 0, width, height).data;
        const preview = new Uint8Array(width * height);
        for (let i = 0; i < preview.length; i++) {
          if (data[i * 4] > 127) preview[i] = activeClassId;
        }
        setAssistPreview(preview);
        setStatus(`SAM: score=${result.score.toFixed(3)}, ${result.predict_time_ms}ms`);
        renderUi(pos);
      };
      img.src = `data:image/png;base64,${result.mask}`;
    })
    .catch((err) => {
      ref.pending = false;
      setStatus(`SAM error: ${err.message}`);
    });
  return ref;
}

export function initSamBox(
  box: [number, number, number, number],
  samRefCurrent: SamRefValue,
  projectId: string, activeImageId: string,
  width: number, height: number, activeClassId: number,
  samModel: SamModelId,
  setAssistPreview: (p: Uint8Array | null) => void,
  setStatus: (s: string) => void,
  renderUi: (pos: [number, number] | null) => void,
): SamRefValue {
  const ref = samRefCurrent ?? { points: [], labels: [], box: null, pending: false };
  ref.box = box;
  if (ref.pending) return ref;
  ref.pending = true;
  setStatus("SAM: box prompt, predicting...");
  samSegment(projectId, activeImageId, ref.points.length > 0 ? ref.points : null, ref.labels.length > 0 ? ref.labels : null, samModel, box)
    .then((result) => {
      if (!ref) return;
      ref.pending = false;
      const img = new window.Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = width; c.height = height;
        const ctx2 = c.getContext("2d");
        if (!ctx2) return;
        ctx2.drawImage(img, 0, 0, width, height);
        const data = ctx2.getImageData(0, 0, width, height).data;
        const preview = new Uint8Array(width * height);
        for (let i = 0; i < preview.length; i++) {
          if (data[i * 4] > 127) preview[i] = activeClassId;
        }
        setAssistPreview(preview);
        setStatus(`SAM: score=${result.score.toFixed(3)}, ${result.predict_time_ms}ms`);
        renderUi(null);
      };
      img.src = `data:image/png;base64,${result.mask}`;
    })
    .catch((err) => {
      ref.pending = false;
      setStatus(`SAM error: ${err.message}`);
    });
  return ref;
}

export function moveWand(
  ref: NonNullable<WandRefValue>, clientY: number,
  activeClassId: number, width: number, height: number,
): { preview: Uint8Array; tolerance: number; status: string } {
  const deltaY = ref.startY - clientY;
  const tolerance = Math.max(1, ref.autoTolerance + deltaY * 0.3);
  ref.tolerance = tolerance;
  const preview = wandFlood(ref.startPos, ref.distMap, tolerance, activeClassId, width, height);
  return { preview, tolerance, status: `Wand: tolerance=${tolerance.toFixed(0)}` };
}

export function commitWand(
  ref: NonNullable<WandRefValue>,
  activeClassId: number, width: number, height: number,
  maskIndex: Uint8Array,
): { indices: number[]; prev: number[]; next: number[]; count: number } {
  const flood = wandFlood(ref.startPos, ref.distMap, ref.tolerance, activeClassId, width, height);
  const indices: number[] = [], prev: number[] = [], next: number[] = [];
  for (let i = 0; i < flood.length; i++) {
    if (flood[i] > 0 && maskIndex[i] !== activeClassId) {
      indices.push(i); prev.push(maskIndex[i]); next.push(activeClassId);
    }
  }
  return { indices, prev, next, count: indices.length };
}

// ---------------------------------------------------------------------------
// Superpixel helpers
// ---------------------------------------------------------------------------

export function decodeSegmentMap(rgbaData: Uint8ClampedArray, w: number, h: number): Uint16Array {
  const map = new Uint16Array(w * h);
  for (let i = 0; i < w * h; i++) {
    map[i] = rgbaData[i * 4] | (rgbaData[i * 4 + 1] << 8);
  }
  return map;
}

export function decodeBoundaryMask(grayData: Uint8ClampedArray, w: number, h: number): Uint8Array {
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    mask[i] = grayData[i * 4] > 127 ? 1 : 0;
  }
  return mask;
}

export function buildSuperpixelPreview(
  segmentMap: Uint16Array, selections: Map<number, number>,
  w: number, h: number,
): Uint8Array {
  const preview = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const cls = selections.get(segmentMap[i]);
    if (cls !== undefined) preview[i] = cls;
  }
  return preview;
}

export async function initSuperpixel(
  projectId: string, activeImageId: string,
  width: number, height: number,
  nSegments: number,
  setStatus: (s: string) => void,
): Promise<NonNullable<SuperpixelRefValue> | null> {
  setStatus("Superpixel: computing...");
  try {
    const result = await superpixelMap(projectId, activeImageId, nSegments);
    // Decode segment map from RGBA PNG base64
    const segImg = new window.Image();
    const segLoaded = new Promise<void>((resolve) => { segImg.onload = () => resolve(); });
    segImg.src = `data:image/png;base64,${result.segments_b64}`;
    await segLoaded;
    const c1 = document.createElement("canvas");
    c1.width = width; c1.height = height;
    const ctx1 = c1.getContext("2d")!;
    ctx1.drawImage(segImg, 0, 0, width, height);
    const segData = ctx1.getImageData(0, 0, width, height).data;
    const segmentMap = decodeSegmentMap(segData, width, height);

    // Decode boundaries
    const bndImg = new window.Image();
    const bndLoaded = new Promise<void>((resolve) => { bndImg.onload = () => resolve(); });
    bndImg.src = `data:image/png;base64,${result.boundaries_b64}`;
    await bndLoaded;
    const c2 = document.createElement("canvas");
    c2.width = width; c2.height = height;
    const ctx2 = c2.getContext("2d")!;
    ctx2.drawImage(bndImg, 0, 0, width, height);
    const bndData = ctx2.getImageData(0, 0, width, height).data;
    const boundaryMask = decodeBoundaryMask(bndData, width, height);

    setStatus(`Superpixel: ${result.n_segments} segments, ${result.time_ms}ms`);
    return {
      segmentMap,
      boundaryMask,
      selections: new Map(),
      loading: false,
    };
  } catch (err) {
    setStatus(`Superpixel: ${(err as Error).message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Crack Trace (backend Meijering + hysteresis)
// ---------------------------------------------------------------------------

export function buildCrackPreview(
  labelMap: Uint16Array, selections: Map<number, number>,
  w: number, h: number,
): Uint8Array {
  const preview = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const crackId = labelMap[i];
    if (crackId === 0) continue;
    const cls = selections.get(crackId);
    preview[i] = cls !== undefined ? cls : 254; // selected → classId, unselected → candidate
  }
  return preview;
}

export async function initCrackTrace(
  projectId: string, activeImageId: string,
  width: number, height: number,
  sensitivity: number, widthPx: number,
  activeClassId: number,
  setStatus: (s: string) => void,
): Promise<{ refValue: NonNullable<CrackTraceRefValue>; preview: Uint8Array; count: number } | null> {
  setStatus("Crack Trace: computing...");
  try {
    const result = await crackTrace(projectId, activeImageId, sensitivity, widthPx);
    // Decode label map from RGBA PNG base64
    const img = new window.Image();
    const loaded = new Promise<void>((resolve) => { img.onload = () => resolve(); });
    img.src = `data:image/png;base64,${result.label_map_b64}`;
    await loaded;
    const c = document.createElement("canvas");
    c.width = width; c.height = height;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0, width, height);
    const data = ctx.getImageData(0, 0, width, height).data;
    const labelMap = decodeSegmentMap(data, width, height);

    // Start with no selections — user clicks to select
    const selections = new Map<number, number>();

    const preview = buildCrackPreview(labelMap, selections, width, height);
    const cached = result.crack_map_cached ? " (cached)" : "";
    setStatus(`Crack: ${result.n_cracks} candidates, ${result.time_ms}ms${cached} — Left: select, Right/Shift: deselect`);

    return {
      refValue: { labelMap, selections, nCracks: result.n_cracks, loading: false, sensitivity, widthPx },
      preview,
      count: result.n_cracks,
    };
  } catch (err) {
    setStatus(`Crack Trace: ${(err as Error).message}`);
    return null;
  }
}


/**
 * Adaptive crack detection: when user clicks on a spot with no existing
 * candidate, call the backend with click coordinates.  The backend uses
 * the local Meijering response to derive a threshold and returns the
 * connected crack region.  We merge it into the existing label map.
 */
export async function adaptiveCrackTrace(
  projectId: string, activeImageId: string,
  clickX: number, clickY: number,
  width: number, height: number,
  sensitivity: number, widthPx: number,
  activeClassId: number,
  crackTraceRef: React.MutableRefObject<CrackTraceRefValue>,
  setAssistPreview: (p: Uint8Array | null) => void,
  setStatus: (s: string) => void,
  ops: { drawOverlay: () => void },
): Promise<void> {
  const ctRef = crackTraceRef.current;
  if (!ctRef) return;

  try {
    const result = await crackTraceAdaptive(projectId, activeImageId, clickX, clickY, sensitivity, widthPx);
    if (!result.label_map_b64) {
      setStatus("Crack: no crack found at click point");
      return;
    }

    // Decode the adaptive label map
    const img = new window.Image();
    const loaded = new Promise<void>((resolve) => { img.onload = () => resolve(); });
    img.src = `data:image/png;base64,${result.label_map_b64}`;
    await loaded;
    const c = document.createElement("canvas");
    c.width = width; c.height = height;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0, width, height);
    const data = ctx.getImageData(0, 0, width, height).data;
    const newLabelMap = decodeSegmentMap(data, width, height);

    // Merge: assign a new label ID to the adaptive region
    const newId = ctRef.nCracks + 1;
    for (let i = 0; i < width * height; i++) {
      if (newLabelMap[i] > 0 && ctRef.labelMap[i] === 0) {
        ctRef.labelMap[i] = newId;
      }
    }
    ctRef.nCracks = newId;

    // Auto-select the new crack
    ctRef.selections.set(newId, activeClassId);

    const preview = buildCrackPreview(ctRef.labelMap, ctRef.selections, width, height);
    setAssistPreview(preview);
    setStatus(`Crack: adaptive +1, ${ctRef.selections.size}/${ctRef.nCracks} selected (${result.time_ms}ms)`);
    ops.drawOverlay();
  } catch (err) {
    setStatus(`Crack adaptive: ${(err as Error).message}`);
  }
}
