// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { samSegment, superpixelMap, crackTrace, crackTraceAdaptive, spotDetect } from "../../api";
import type { SpotDetectResult } from "../../api";
import type { SamLevel, SamRefValue, SuperpixelRefValue, CrackTraceRefValue } from "./useDrawingEvents";
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

export function collectRectFill(
  from: [number, number], to: [number, number], value: number,
  width: number, height: number,
  maskIndex: Uint8Array,
): BrushResult | null {
  const x1 = Math.max(0, Math.min(width - 1, Math.round(Math.min(from[0], to[0]))));
  const y1 = Math.max(0, Math.min(height - 1, Math.round(Math.min(from[1], to[1]))));
  const x2 = Math.max(0, Math.min(width - 1, Math.round(Math.max(from[0], to[0]))));
  const y2 = Math.max(0, Math.min(height - 1, Math.round(Math.max(from[1], to[1]))));
  const indices: number[] = [], prev: number[] = [], next: number[] = [];
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = y1; y <= y2; y += 1) {
    const row = y * width;
    for (let x = x1; x <= x2; x += 1) {
      const idx = row + x;
      // Pixels already holding the value are skipped, the same rule the brush
      // and the bucket follow. Emitting them would push undo entries whose
      // prev === next, so one undo would appear to do nothing.
      if (maskIndex[idx] === value) continue;
      indices.push(idx); prev.push(maskIndex[idx]); next.push(value);
      if (x < minX) minX = x; if (y < minY) minY = y;
      if (x > maxX) maxX = x; if (y > maxY) maxY = y;
    }
  }
  if (indices.length === 0) return null;
  return { indices, prev, next, dirty: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 } };
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

/** The pixels the user painted, as the server wants them: [x, y] pairs. */
export function markPointsFrom(sampleMask: Uint8Array, width: number): number[][] {
  const points: number[][] = [];
  for (let i = 0; i < sampleMask.length; i++) {
    if (sampleMask[i] > 0) {
      const x = i % width;
      points.push([x, (i - x) / width]);
    }
  }
  return points;
}

/**
 * Ask the server for the spots that look like the painted example.
 *
 * This used to run in the browser, in two copies that disagreed: the
 * sensitivity search flood-filled at one threshold while the worker that made
 * the mask filled at half of it, so the search scored an operator that never
 * ran: the count it tuned for and the specks the mask then held could differ
 * many times over. There is one implementation now, and it is not here -- the
 * browser sends where it painted and paints what comes back.
 */
export async function runSpotDetect(
  projectId: string, itemId: string,
  markPoints: number[][], classId: number,
  width: number, height: number,
  sensitivity?: number,
): Promise<{ preview: Uint8Array; count: number; result: SpotDetectResult }> {
  const result = await spotDetect(projectId, itemId, markPoints, classId, sensitivity);
  const preview = new Uint8Array(width * height);
  if (!result.mask) return { preview, count: 0, result };
  const img = new window.Image();
  const loaded = new Promise<void>((resolve) => { img.onload = () => resolve(); });
  img.src = `data:image/png;base64,${result.mask}`;
  await loaded;
  const c = document.createElement("canvas");
  c.width = width; c.height = height;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0, width, height);
  const data = ctx.getImageData(0, 0, width, height).data;
  for (let i = 0; i < preview.length; i++) preview[i] = data[i * 4];
  return { preview, count: result.count, result };
}

function decodeBinaryMask(b64: string, width: number, height: number): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = width; c.height = height;
      const ctx = c.getContext("2d");
      if (!ctx) { reject(new Error("no 2d context")); return; }
      ctx.drawImage(img, 0, 0, width, height);
      const data = ctx.getImageData(0, 0, width, height).data;
      const out = new Uint8Array(width * height);
      for (let i = 0; i < out.length; i++) if (data[i * 4] > 127) out[i] = 1;
      resolve(out);
    };
    img.onerror = () => reject(new Error("mask decode failed"));
    img.src = `data:image/png;base64,${b64}`;
  });
}

/** Paint a stored binary mask with whichever class is selected right now.
 *
 * The candidates are kept class-free on purpose: the user can change class
 * between the click and the choice of granularity, and a preview baked at
 * click time would then apply the wrong one.
 */
export function samLevelPreview(mask: Uint8Array, classId: number): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[i] = classId;
  return out;
}

export function initSam(
  pos: [number, number],
  samRefCurrent: SamRefValue,
  projectId: string, activeImageId: string,
  width: number, height: number, activeClassId: number,
  samModel: SamModelId,
  setAssistPreview: (p: Uint8Array | null) => void,
  setStatus: (s: string) => void,
  renderUi: (pos: [number, number] | null) => void,
): SamRefValue {
  const ref = samRefCurrent ?? { points: [], box: null, pending: false };
  ref.points.push(pos);
  if (ref.pending) return ref;
  ref.pending = true;
  const statusParts = [`${ref.points.length} pts`];
  if (ref.box) statusParts.push("box");
  setStatus(`SAM: ${statusParts.join(" + ")}, predicting...`);
  samSegment(projectId, activeImageId, ref.points, samModel, ref.box)
    .then(async (result) => {
      if (!ref) return;
      ref.pending = false;
      // A server that does not send candidates still works: its single mask
      // is the whole-object one, which is what argmax was already returning.
      const raw = result.levels?.length
        ? result.levels
        : [{ level: "whole", mask: result.mask, score: result.score, area: 0 }];
      const levels: SamLevel[] = await Promise.all(
        raw.map(async (lv) => ({
          level: lv.level,
          score: lv.score,
          area: lv.area,
          mask: await decodeBinaryMask(lv.mask, width, height),
        }))
      );
      ref.levels = levels;
      const pick = levels.find((l) => l.level === result.default_level) ?? levels[levels.length - 1];
      ref.activeLevel = pick.level;
      setAssistPreview(samLevelPreview(pick.mask, activeClassId));
      setStatus(`SAM: ${pick.level}, score=${pick.score.toFixed(3)}, ${result.predict_time_ms}ms`);
      renderUi(pos);
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
  const ref = samRefCurrent ?? { points: [], box: null, pending: false };
  ref.box = box;
  if (ref.pending) return ref;
  ref.pending = true;
  setStatus("SAM: box prompt, predicting...");
  samSegment(projectId, activeImageId, ref.points.length > 0 ? ref.points : null, samModel, box)
    .then(async (result) => {
      if (!ref) return;
      ref.pending = false;
      // The box asked for the same three candidates the click gets and then
      // threw two of them away: only result.mask was decoded, ref.levels was
      // never set, and the granularity buttons -- which appear whenever there
      // is more than one level -- had nothing to show. Where small parts sit
      // close together the level that fits is not always the default one, so
      // the choice matters most exactly where a box is the natural way to ask.
      const raw = result.levels?.length
        ? result.levels
        : [{ level: "whole", mask: result.mask, score: result.score, area: 0 }];
      const levels: SamLevel[] = await Promise.all(
        raw.map(async (lv) => ({
          level: lv.level,
          score: lv.score,
          area: lv.area,
          mask: await decodeBinaryMask(lv.mask, width, height),
        }))
      );
      ref.levels = levels;
      const pick = levels.find((l) => l.level === result.default_level) ?? levels[levels.length - 1];
      ref.activeLevel = pick.level;
      setAssistPreview(samLevelPreview(pick.mask, activeClassId));
      setStatus(`SAM: ${pick.level}, score=${pick.score.toFixed(3)}, ${result.predict_time_ms}ms`);
      renderUi(null);
    })
    .catch((err) => {
      ref.pending = false;
      setStatus(`SAM error: ${err.message}`);
    });
  return ref;
}

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

    // An image switch (or cancel) cleared the ref while the request was in
    // flight — merging now would paint the old image's crack onto the new one.
    if (crackTraceRef.current !== ctRef) return;

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
