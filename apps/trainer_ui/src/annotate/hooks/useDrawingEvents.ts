// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useRef, useState, type RefObject } from "react";
import { useMaskStore } from "../../store";
import type { SamModelId } from "../annotatorContext";
import {
  collectBrushStroke, floodFill,
  initWand, initSam, initSamBox,
  moveWand,
  commitWand,
  initSuperpixel, buildSuperpixelPreview,
  initCrackTrace, buildCrackPreview, adaptiveCrackTrace,
  pickConnectedRegion,
} from "./toolActions";

export type DrawingOps = {
  drawOverlay: (dirty?: { x: number; y: number; w: number; h: number }) => void;
  scheduleDrawOverlay: () => void;
  markDirty: () => void;
  markTouched: (indices: Uint32Array) => void;
  renderUi: (pos: [number, number] | null) => void;
  scheduleRenderUi: (pos: [number, number] | null) => void;
  handleSamConfirm: () => void;
  handleSamCancel: () => void;
  handleSpotConfirm: () => void;
  handleSpotCancel: () => void;
  handleSuperpixelConfirm: () => void;
  handleSuperpixelCancel: () => void;
  handleCrackConfirm: () => void;
  handleCrackCancel: () => void;
};

export type StrokeAccRef = RefObject<Map<number, number> | null>;
export type WandRefValue = {
  startY: number; startPos: [number, number]; distMap: Float32Array; autoTolerance: number; tolerance: number;
} | null;
export type SamMode = "point" | "box";
export type SamRefValue = { points: [number, number][]; labels: number[]; box: [number, number, number, number] | null; pending: boolean } | null;
export type SpotDetectRefValue = {
  imageData: Uint8ClampedArray;
  scores: Float32Array;
  std: number;
  extrema: Uint32Array;
  targetLab?: [number, number, number] | null;
  clickIdx?: number;
  /** "sample" = user is painting spot samples; "detect" = detection results shown */
  phase: "sample" | "detect";
  /** Temporary sample mask for painted spot samples (only in sample phase) */
  sampleMask?: Uint8Array;
  /** Auto-detected size filter range [min, max] in pixels */
  sizeRange?: [number, number];
  /** Downscaled dimensions (set when scores are at reduced resolution) */
  downscaleWidth?: number;
  downscaleHeight?: number;
  /** Auto-computed color tolerance from sample analysis */
  autoColorTolerance?: number;
  /** Detection mode: "dog" = blob detection, "color" = color distance */
  mode?: "dog" | "color";
  /** Color distance map (for color mode slider re-filtering) */
  colorDistMap?: Float32Array;
} | null;
export type SuperpixelRefValue = {
  segmentMap: Uint16Array;
  boundaryMask: Uint8Array;
  selections: Map<number, number>;
  loading: boolean;
} | null;
export type MoveRefValue = {
  classId: number;
  startPos: [number, number];
  /** Original pixel indices of the picked connected component (in image-space). */
  indices: Uint32Array;
  /** Saved original classId values at each index (all equal to classId). */
  prevValues: Uint8Array;
  /** Current drag offset in image-space pixels. */
  offset: [number, number];
} | null;
export type CrackTraceRefValue = {
  labelMap: Uint16Array;
  selections: Map<number, number>;
  nCracks: number;
  loading: boolean;
  sensitivity: number;
  widthPx: number;
} | null;

export function useDrawingEvents(
  getOps: () => DrawingOps,
  containerRef: RefObject<HTMLDivElement | null>,
  imageCanvasRef: RefObject<HTMLCanvasElement | null>,
  brushSize: number,
  strokeAccRef: RefObject<Map<number, number> | null>,
  wandRef: RefObject<WandRefValue>,
  samRef: RefObject<SamRefValue>,
  spotDetectRef: RefObject<SpotDetectRefValue>,
  spotSensitivity: number,
  setSpotCount: (count: number) => void,
  setAssistPreview: (preview: Uint8Array | null) => void,
  setRecipePreview: (preview: Uint8Array | null) => void,
  setStatus: (msg: string) => void,
  samModel: SamModelId,
  projectId: string | null,
  activeImageId: string | null,
  spacePressed: boolean,
  isPanning: boolean,
  setIsPanning: (v: boolean) => void,
  panStartRef: RefObject<{ x: number; y: number; offsetX: number; offsetY: number } | null>,
  setMeasureStart: (v: [number, number] | null) => void,
  setMeasureEnd: (v: [number, number] | null) => void,
  measureStart: [number, number] | null,
  measureEnd: [number, number] | null,
  samBoxDraftRef: RefObject<{ start: [number, number]; end: [number, number] } | null>,
  superpixelRef: RefObject<SuperpixelRefValue>,
  nSegments: number,
  assistPreview: Uint8Array | null,
  crackTraceRef: RefObject<CrackTraceRefValue>,
  crackSensitivity: number,
  crackWidth: number,
  colorTolerance: number,
  setSpotPhase?: (phase: "idle" | "sample" | "detect") => void,
  moveRef?: RefObject<MoveRefValue>,
) {
  const [isDrawing, setIsDrawing] = useState(false);
  const lastPosRef = useRef<[number, number] | null>(null);
  const strokeStartRef = useRef<[number, number] | null>(null);
  // Cached image pixels (avoids getImageData per tool init)
  const cachedPixelsRef = useRef<{ url: string | null; pixels: Uint8ClampedArray | null }>({ url: null, pixels: null });
  function getPixels(): Uint8ClampedArray | null {
    if (cachedPixelsRef.current.url === imageUrl && cachedPixelsRef.current.pixels) return cachedPixelsRef.current.pixels;
    const imgCanvas = imageCanvasRef.current;
    if (!imgCanvas || width === 0 || height === 0) return null;
    const ctx = imgCanvas.getContext("2d");
    if (!ctx) return null;
    const pixels = ctx.getImageData(0, 0, width, height).data;
    cachedPixelsRef.current = { url: imageUrl, pixels };
    return pixels;
  }
  // Reusable TypedArray pool to avoid GC pressure in hot path
  const poolIdx = useRef<Uint32Array>(new Uint32Array(4096));
  const poolPrev = useRef<Uint8Array>(new Uint8Array(4096));
  const poolNext = useRef<Uint8Array>(new Uint8Array(4096));
  function ensurePool(size: number) {
    if (poolIdx.current.length < size) {
      const n = Math.max(size, poolIdx.current.length * 2);
      poolIdx.current = new Uint32Array(n);
      poolPrev.current = new Uint8Array(n);
      poolNext.current = new Uint8Array(n);
    }
  }
  const maskIndex = useMaskStore((s) => s.maskIndex);
  const width = useMaskStore((s) => s.width);
  const height = useMaskStore((s) => s.height);
  const activeClassId = useMaskStore((s) => s.activeClassId);
  const classes = useMaskStore((s) => s.classes);
  const tool = useMaskStore((s) => s.tool);
  const scale = useMaskStore((s) => s.scale);
  const offsetX = useMaskStore((s) => s.offsetX);
  const offsetY = useMaskStore((s) => s.offsetY);
  const imageUrl = useMaskStore((s) => s.imageUrl);
  const applyDelta = useMaskStore((s) => s.applyDelta);
  const applyDeltaSilent = useMaskStore((s) => s.applyDeltaSilent);
  const pushUndo = useMaskStore((s) => s.pushUndo);
  const setView = useMaskStore((s) => s.setView);

  function screenToImage(clientX: number, clientY: number) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const x = (clientX - rect.left - offsetX) / scale;
    const y = (clientY - rect.top - offsetY) / scale;
    if (x < 0 || y < 0 || x >= width || y >= height) return null;
    return [Math.floor(x), Math.floor(y)] as [number, number];
  }

  function applyBrushStroke(from: [number, number], to: [number, number], value: number, acc: Map<number, number>) {
    const ops = getOps();
    const result = collectBrushStroke(from, to, value, brushSize, width, height, maskIndex);
    const len = result.indices.length;
    if (len > 0) {
      for (let i = 0; i < len; i++) {
        if (!acc.has(result.indices[i])) acc.set(result.indices[i], result.prev[i]);
      }
      ensurePool(len);
      const idxArr = poolIdx.current.subarray(0, len);
      const prevArr = poolPrev.current.subarray(0, len);
      const nextArr = poolNext.current.subarray(0, len);
      for (let i = 0; i < len; i++) { idxArr[i] = result.indices[i]; prevArr[i] = result.prev[i]; nextArr[i] = result.next[i]; }
      ops.markTouched(idxArr);
      applyDeltaSilent(idxArr, prevArr, nextArr);
    }
    return result;
  }

  function handlePointerDown(event: React.PointerEvent) {
    if (!imageUrl) return;
    const ops = getOps();
    if (event.button === 1 || spacePressed || event.ctrlKey) {
      setIsPanning(true);
      (panStartRef as React.MutableRefObject<typeof panStartRef.current>).current = {
        x: event.clientX, y: event.clientY, offsetX, offsetY,
      };
      (event.target as Element).setPointerCapture(event.pointerId);
      return;
    }
    if (event.button === 2 && (tool === "sam" || tool === "sambox" || tool === "superpixel" || tool === "cracktrace")) { /* allow right-click */ }
    else if (event.button !== 0) return;
    const pos = screenToImage(event.clientX, event.clientY);
    if (!pos) return;
    (event.target as Element).setPointerCapture(event.pointerId);
    setIsDrawing(true); lastPosRef.current = pos; strokeStartRef.current = pos;

    // Guard: block painting tools when no valid foreground class is selected
    const needsForeground = tool !== "eraser" && tool !== "measure" && tool !== "colorpick" && tool !== "roi";
    if (needsForeground && (activeClassId === 0 || !classes.find((c) => c.id === activeClassId))) {
      setStatus("Add or select a foreground class before painting.");
      setIsDrawing(false);
      return;
    }

    if (tool === "brush" || tool === "eraser") {
      const value = tool === "eraser" ? 0 : activeClassId;
      const acc = new Map<number, number>();
      (strokeAccRef as React.MutableRefObject<Map<number, number> | null>).current = acc;
      const result = applyBrushStroke(pos, pos, value, acc);
      if (result.indices.length > 0) { ops.drawOverlay(result.dirty ?? undefined); ops.markDirty(); }
    }
    if (tool === "measure") {
      if (!measureStart || measureEnd) { setMeasureStart(pos); setMeasureEnd(pos); }
      else { setMeasureEnd(pos); }
      ops.renderUi(pos); return;
    }
    if (tool === "bucket") {
      const result = floodFill(pos, activeClassId, width, height, maskIndex);
      if (result) {
        const idxArr = new Uint32Array(result.indices);
        ops.markTouched(idxArr);
        applyDelta(idxArr, new Uint8Array(result.prev), new Uint8Array(result.next));
        ops.drawOverlay(result.dirty ?? undefined);
        ops.markDirty();
      }
      setIsDrawing(false);
    }
    if (tool === "wand") {
      const pixels = getPixels();
      if (!pixels) { setIsDrawing(false); return; }
      const r = initWand(pos, event.clientY, pixels, width, height, activeClassId);
      if (!r) { setIsDrawing(false); return; }
      (wandRef as React.MutableRefObject<WandRefValue>).current = r.refValue;
      setAssistPreview(r.preview); setStatus(r.status); return;
    }
    if (tool === "spotdetect") {
      const ref = spotDetectRef.current;
      // In sample phase (or no ref yet): paint into sampleMask like a brush
      if (!ref || ref.phase === "sample") {
        const n = width * height;
        const sampleMask = ref?.sampleMask ?? new Uint8Array(n);
        // Paint a circular brush mark onto sampleMask
        const bh = Math.max(1, Math.floor(brushSize / 2));
        for (let dy = -bh; dy <= bh; dy++) {
          for (let dx = -bh; dx <= bh; dx++) {
            if (dx * dx + dy * dy > bh * bh) continue;
            const px = pos[0] + dx, py = pos[1] + dy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
              sampleMask[py * width + px] = activeClassId || 1;
            }
          }
        }
        // Show painted samples as preview
        setAssistPreview(new Uint8Array(sampleMask));
        setSpotPhase?.("sample");
        const sampleCount = sampleMask.reduce((acc, v) => acc + (v > 0 ? 1 : 0), 0);
        setStatus(`Spot Detect: ${sampleCount}px sampled — click "Detect" to find similar spots`);
        const pixels = getPixels();
        (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = {
          imageData: pixels ?? new Uint8ClampedArray(0),
          scores: new Float32Array(0), std: 0, extrema: new Uint32Array(0),
          phase: "sample", sampleMask,
        };
        return;
      }
      // In detect phase: clicking adds more samples (switch back to sample phase)
      if (ref.phase === "detect" && ref.sampleMask) {
        ref.phase = "sample";
        const bh = Math.max(1, Math.floor(brushSize / 2));
        for (let dy = -bh; dy <= bh; dy++) {
          for (let dx = -bh; dx <= bh; dx++) {
            if (dx * dx + dy * dy > bh * bh) continue;
            const px = pos[0] + dx, py = pos[1] + dy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
              ref.sampleMask[py * width + px] = activeClassId || 1;
            }
          }
        }
        setAssistPreview(new Uint8Array(ref.sampleMask));
        return;
      }
    }
    if (tool === "superpixel" && projectId && activeImageId) {
      const spRef = superpixelRef.current;
      if (!spRef) {
        // First click — load superpixel map from API
        (superpixelRef as React.MutableRefObject<SuperpixelRefValue>).current = {
          segmentMap: new Uint16Array(0), boundaryMask: new Uint8Array(0),
          selections: new Map(), loading: true,
        };
        initSuperpixel(projectId, activeImageId, width, height, nSegments, setStatus)
          .then((result) => {
            if (result) {
              (superpixelRef as React.MutableRefObject<SuperpixelRefValue>).current = result;
              ops.drawOverlay();
            } else {
              (superpixelRef as React.MutableRefObject<SuperpixelRefValue>).current = null;
            }
          });
        setIsDrawing(false);
        return;
      }
      if (spRef.loading) { setIsDrawing(false); return; }
      // Subsequent click — toggle superpixel selection
      const segId = spRef.segmentMap[pos[1] * width + pos[0]];
      if (event.button === 2 || event.shiftKey) {
        // Right-click or shift: deselect
        spRef.selections.delete(segId);
      } else {
        spRef.selections.set(segId, activeClassId);
      }
      const preview = buildSuperpixelPreview(spRef.segmentMap, spRef.selections, width, height);
      setAssistPreview(preview);
      const selCount = spRef.selections.size;
      setStatus(`Superpixel: ${selCount} segments selected`);
      ops.drawOverlay();
      setIsDrawing(false);
      return;
    }
    if (tool === "cracktrace" && projectId && activeImageId) {
      const ctRef = crackTraceRef.current;
      if (!ctRef) {
        // First click — load crack trace from API
        (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = {
          labelMap: new Uint16Array(0), selections: new Map(), nCracks: 0, loading: true,
          sensitivity: crackSensitivity, widthPx: crackWidth,
        };
        initCrackTrace(projectId, activeImageId, width, height, crackSensitivity, crackWidth, activeClassId, setStatus)
          .then((result) => {
            if (result) {
              (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = result.refValue;
              setAssistPreview(result.preview);
              ops.drawOverlay();
            } else {
              (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = null;
            }
          });
        setIsDrawing(false);
        return;
      }
      if (ctRef.loading) { setIsDrawing(false); return; }
      // Subsequent click — toggle crack selection
      const crackId = ctRef.labelMap[pos[1] * width + pos[0]];
      if (crackId > 0) {
        if (event.button === 2 || event.shiftKey) {
          ctRef.selections.delete(crackId);
        } else {
          ctRef.selections.set(crackId, activeClassId);
        }
        const preview = buildCrackPreview(ctRef.labelMap, ctRef.selections, width, height);
        setAssistPreview(preview);
        setStatus(`Crack: ${ctRef.selections.size}/${ctRef.nCracks} selected`);
        ops.drawOverlay();
      } else if (event.button === 0 && !event.shiftKey) {
        // No candidate at click — try adaptive detection
        setStatus("Crack: adaptive detecting...");
        adaptiveCrackTrace(projectId, activeImageId, pos[0], pos[1], width, height,
          crackSensitivity, crackWidth, activeClassId, crackTraceRef, setAssistPreview, setStatus, ops);
      }
      setIsDrawing(false);
      return;
    }
    if (tool === "move") {
      const picked = pickConnectedRegion(pos, width, height, maskIndex);
      if (!picked) {
        setStatus("Move: click on a mask region to pick it up.");
        setIsDrawing(false);
        return;
      }
      const zeros = new Uint8Array(picked.indices.length);
      const prevValues = new Uint8Array(picked.indices.length);
      prevValues.fill(picked.classId);
      // Silent lift: clear origin so the drag preview is the only visible mask.
      applyDeltaSilent(picked.indices, prevValues, zeros);
      const preview = new Uint8Array(width * height);
      for (let i = 0; i < picked.indices.length; i++) preview[picked.indices[i]] = picked.classId;
      setAssistPreview(preview);
      if (moveRef) {
        (moveRef as React.MutableRefObject<MoveRefValue>).current = {
          classId: picked.classId,
          startPos: pos,
          indices: picked.indices,
          prevValues,
          offset: [0, 0],
        };
      }
      ops.drawOverlay();
      setStatus(`Move: ${picked.indices.length.toLocaleString()}px picked — drag to reposition`);
      return;
    }
    if ((tool === "sam" || tool === "sambox") && projectId && activeImageId) {
      if (tool === "sambox" && !samRef.current?.box) {
        // No box yet — start drag
        (samBoxDraftRef as React.MutableRefObject<typeof samBoxDraftRef.current>).current = { start: pos, end: pos };
        ops.renderUi(pos);
        return;
      }
      // Click inside preview mask → confirm
      if (assistPreview && samRef.current && !samRef.current.pending && event.button === 0) {
        const px = Math.round(pos[0]);
        const py = Math.round(pos[1]);
        if (px >= 0 && px < width && py >= 0 && py < height) {
          const idx = py * width + px;
          if (assistPreview[idx] > 0) {
            ops.handleSamConfirm();
            return;
          }
        }
      }
      const isBackground = event.button === 2;
      const newRef = initSam(
        pos, isBackground, samRef.current, projectId, activeImageId,
        width, height, activeClassId, samModel, setAssistPreview, setStatus, ops.renderUi,
      );
      (samRef as React.MutableRefObject<SamRefValue>).current = newRef;
      return;
    }
  }

  function handlePointerMove(event: React.PointerEvent) {
    const pos = screenToImage(event.clientX, event.clientY);
    const ops = getOps();
    ops.scheduleRenderUi(pos);
    if (isPanning && panStartRef.current) {
      const dx = event.clientX - panStartRef.current.x;
      const dy = event.clientY - panStartRef.current.y;
      setView(scale, panStartRef.current.offsetX + dx, panStartRef.current.offsetY + dy);
      return;
    }
    if (tool === "wand" && wandRef.current && isDrawing) {
      const r = moveWand(wandRef.current, event.clientY, activeClassId, width, height);
      setAssistPreview(r.preview); setStatus(r.status); return;
    }
    if (tool === "move" && moveRef?.current && isDrawing && pos) {
      const mv = moveRef.current;
      const dx = pos[0] - mv.startPos[0];
      const dy = pos[1] - mv.startPos[1];
      mv.offset = [dx, dy];
      const preview = new Uint8Array(width * height);
      for (let i = 0; i < mv.indices.length; i++) {
        const src = mv.indices[i];
        const x = (src % width) + dx;
        const y = ((src - (src % width)) / width) + dy;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        preview[y * width + x] = mv.classId;
      }
      setAssistPreview(preview);
      ops.drawOverlay();
      setStatus(`Move: Δ=(${dx}, ${dy})`);
      return;
    }
    if (tool === "sambox" && samBoxDraftRef.current && isDrawing && pos) {
      (samBoxDraftRef as React.MutableRefObject<typeof samBoxDraftRef.current>).current = { start: samBoxDraftRef.current.start, end: pos };
      ops.renderUi(pos); return;
    }
    const lastPos = lastPosRef.current;
    if (!isDrawing || !pos || !lastPos) return;
    if (tool === "measure") { setMeasureEnd(pos); return; }
    // SpotDetect sample brush drag
    if (tool === "spotdetect") {
      const ref = spotDetectRef.current;
      if (ref && (ref.phase === "sample" || ref.phase === "detect") && ref.sampleMask) {
        // Paint line from lastPos to pos into sampleMask
        const bh = Math.max(1, Math.floor(brushSize / 2));
        const dx = pos[0] - lastPos[0], dy = pos[1] - lastPos[1];
        const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
        for (let s = 0; s <= steps; s++) {
          const cx = Math.round(lastPos[0] + dx * s / steps);
          const cy = Math.round(lastPos[1] + dy * s / steps);
          for (let bdy = -bh; bdy <= bh; bdy++) {
            for (let bdx = -bh; bdx <= bh; bdx++) {
              if (bdx * bdx + bdy * bdy > bh * bh) continue;
              const px = cx + bdx, py = cy + bdy;
              if (px >= 0 && px < width && py >= 0 && py < height) {
                ref.sampleMask[py * width + px] = activeClassId || 1;
              }
            }
          }
        }
        lastPosRef.current = pos;
        if (ref.phase === "sample") {
          setAssistPreview(new Uint8Array(ref.sampleMask));
        }
        return;
      }
    }
    if (tool === "brush" || tool === "eraser") {
      const value = tool === "eraser" ? 0 : activeClassId;
      const acc = strokeAccRef.current;
      if (event.shiftKey && strokeStartRef.current) {
        if (acc && acc.size > 0) {
          ensurePool(acc.size);
          const restoreIdx = poolIdx.current.subarray(0, acc.size);
          const restorePrev = poolPrev.current.subarray(0, acc.size);
          const restoreNext = poolNext.current.subarray(0, acc.size);
          let i = 0;
          for (const [idx, orig] of acc) { restoreIdx[i] = idx; restorePrev[i] = 0; restoreNext[i] = orig; i++; }
          applyDeltaSilent(restoreIdx, restorePrev, restoreNext);
          acc.clear();
        }
        if (acc) applyBrushStroke(strokeStartRef.current, pos, value, acc);
        ops.drawOverlay(); ops.markDirty();
      } else {
        // Use coalesced events for smoother strokes — process all
        // intermediate pointer positions the browser batched together
        const nativeEvent = event.nativeEvent as PointerEvent;
        const coalesced = nativeEvent.getCoalescedEvents?.() ?? [];
        const points: [number, number][] = [];
        for (let ci = 0; ci < coalesced.length; ci++) {
          const cp = screenToImage(coalesced[ci].clientX, coalesced[ci].clientY);
          if (cp) points.push(cp);
        }
        if (points.length === 0) points.push(pos);

        // Accumulate all segments into one dirty rect + one state update
        const allIndices: number[] = [];
        const allPrev: number[] = [];
        const allNext: number[] = [];
        let dirtyMinX = width, dirtyMinY = height, dirtyMaxX = -1, dirtyMaxY = -1;
        let prev = lastPos;
        for (let pi = 0; pi < points.length; pi++) {
          const cur = points[pi];
          const result = collectBrushStroke(prev, cur, value, brushSize, width, height, maskIndex);
          const len = result.indices.length;
          if (len > 0) {
            if (acc) { for (let i = 0; i < len; i++) { if (!acc.has(result.indices[i])) acc.set(result.indices[i], result.prev[i]); } }
            for (let i = 0; i < len; i++) { allIndices.push(result.indices[i]); allPrev.push(result.prev[i]); allNext.push(result.next[i]); }
            if (result.dirty) {
              if (result.dirty.x < dirtyMinX) dirtyMinX = result.dirty.x;
              if (result.dirty.y < dirtyMinY) dirtyMinY = result.dirty.y;
              const rx = result.dirty.x + result.dirty.w - 1;
              const ry = result.dirty.y + result.dirty.h - 1;
              if (rx > dirtyMaxX) dirtyMaxX = rx;
              if (ry > dirtyMaxY) dirtyMaxY = ry;
            }
          }
          prev = cur;
        }
        if (allIndices.length > 0) {
          ensurePool(allIndices.length);
          const idxArr = poolIdx.current.subarray(0, allIndices.length);
          const prevArr = poolPrev.current.subarray(0, allIndices.length);
          const nextArr = poolNext.current.subarray(0, allIndices.length);
          for (let i = 0; i < allIndices.length; i++) { idxArr[i] = allIndices[i]; prevArr[i] = allPrev[i]; nextArr[i] = allNext[i]; }
          ops.markTouched(idxArr);
          applyDeltaSilent(idxArr, prevArr, nextArr);
          const dirty = dirtyMaxX >= 0 ? { x: dirtyMinX, y: dirtyMinY, w: dirtyMaxX - dirtyMinX + 1, h: dirtyMaxY - dirtyMinY + 1 } : undefined;
          ops.drawOverlay(dirty); ops.markDirty();
        }
        lastPosRef.current = points[points.length - 1];
      }
    }
  }

  function handlePointerUp() {
    const ops = getOps();
    if (isPanning) {
      setIsPanning(false);
      (panStartRef as React.MutableRefObject<typeof panStartRef.current>).current = null;
    }
    if (tool === "sambox" && samBoxDraftRef.current && projectId && activeImageId) {
      const draft = samBoxDraftRef.current;
      const x1 = Math.min(draft.start[0], draft.end[0]);
      const y1 = Math.min(draft.start[1], draft.end[1]);
      const x2 = Math.max(draft.start[0], draft.end[0]);
      const y2 = Math.max(draft.start[1], draft.end[1]);
      (samBoxDraftRef as React.MutableRefObject<{ start: [number, number]; end: [number, number] } | null>).current = null;
      if (x2 - x1 >= 3 && y2 - y1 >= 3) {
        const box: [number, number, number, number] = [x1, y1, x2, y2];
        const newRef = initSamBox(
          box, samRef.current, projectId, activeImageId,
          width, height, activeClassId, samModel, setAssistPreview, setStatus, ops.renderUi,
        );
        (samRef as React.MutableRefObject<SamRefValue>).current = newRef;
      }
      setIsDrawing(false); lastPosRef.current = null;
      return;
    }
    // SpotDetect: after painting more samples, switch back to sample phase for re-detect.
    // Only transition if the user was actually drawing (isDrawing) — pointerLeave fires
    // handlePointerUp without prior drawing and must NOT reset detect→sample.
    if (tool === "spotdetect") {
      const ref = spotDetectRef.current;
      if (isDrawing && ref?.phase === "detect" && ref.sampleMask) {
        // Switch back to sample phase — user should click Detect again
        ref.phase = "sample";
        setSpotPhase?.("sample");
        setAssistPreview(new Uint8Array(ref.sampleMask));
        const sampleCount = ref.sampleMask.reduce((acc: number, v: number) => acc + (v > 0 ? 1 : 0), 0);
        setStatus(`Spot Detect: ${sampleCount}px sampled — click "Detect" to re-analyze`);
      }
      setIsDrawing(false); lastPosRef.current = null;
      return;
    }
    if (tool === "move" && moveRef?.current) {
      const mv = moveRef.current;
      const [dx, dy] = mv.offset;
      // Restore origin (silent) so maskIndex reflects pre-move state.
      const zeros = new Uint8Array(mv.indices.length);
      applyDeltaSilent(mv.indices, zeros, mv.prevValues);
      setAssistPreview(null);
      (moveRef as React.MutableRefObject<MoveRefValue>).current = null;

      if (dx === 0 && dy === 0) {
        ops.drawOverlay();
        setIsDrawing(false);
        lastPosRef.current = null;
        setStatus("Move: cancelled");
        return;
      }

      // Build union of source and target indices, skipping out-of-bounds
      // targets and collisions (target pixel already contains a different class).
      const fresh = useMaskStore.getState().maskIndex;
      const unionPrev = new Map<number, number>();
      const unionNext = new Map<number, number>();
      for (let i = 0; i < mv.indices.length; i++) {
        const src = mv.indices[i];
        unionPrev.set(src, mv.classId);
        unionNext.set(src, 0);
      }
      let painted = 0;
      let skippedCollision = 0;
      let skippedOob = 0;
      for (let i = 0; i < mv.indices.length; i++) {
        const src = mv.indices[i];
        const sx = src % width;
        const sy = (src - sx) / width;
        const tx = sx + dx;
        const ty = sy + dy;
        if (tx < 0 || tx >= width || ty < 0 || ty >= height) { skippedOob++; continue; }
        const ti = ty * width + tx;
        const existing = fresh[ti];
        if (unionPrev.has(ti)) {
          // Target overlaps source region — safe to paint classId since
          // original value there is our own classId.
          unionNext.set(ti, mv.classId);
          painted++;
        } else if (existing !== 0) {
          // Collision with another class — skip (non-destructive move).
          skippedCollision++;
        } else {
          unionPrev.set(ti, 0);
          unionNext.set(ti, mv.classId);
          painted++;
        }
      }
      const finalIdx: number[] = [];
      const finalPrev: number[] = [];
      const finalNext: number[] = [];
      for (const [idx, prev] of unionPrev) {
        const next = unionNext.get(idx) ?? prev;
        if (prev !== next) { finalIdx.push(idx); finalPrev.push(prev); finalNext.push(next); }
      }
      if (finalIdx.length > 0) {
        const idxArr = new Uint32Array(finalIdx);
        const prevArr = new Uint8Array(finalPrev);
        const nextArr = new Uint8Array(finalNext);
        ops.markTouched(idxArr);
        applyDelta(idxArr, prevArr, nextArr);
        ops.markDirty();
      }
      ops.drawOverlay();
      const parts = [`${painted.toLocaleString()}px moved`];
      if (skippedCollision > 0) parts.push(`${skippedCollision.toLocaleString()} skipped (overlap)`);
      if (skippedOob > 0) parts.push(`${skippedOob.toLocaleString()} clipped`);
      setStatus(`Move: ${parts.join(", ")}`);
      setIsDrawing(false);
      lastPosRef.current = null;
      return;
    }
    if (tool === "wand" && wandRef.current) {
      const r = commitWand(wandRef.current, activeClassId, width, height, maskIndex);
      if (r.indices.length > 0) {
        const idxArr = new Uint32Array(r.indices);
        ops.markTouched(idxArr);
        applyDelta(idxArr, new Uint8Array(r.prev), new Uint8Array(r.next));
        ops.markDirty();
      }
      setAssistPreview(null); setRecipePreview(null);
      (wandRef as React.MutableRefObject<WandRefValue>).current = null;
      setStatus(`Wand: ${r.count}px applied`);
    }
    const acc = strokeAccRef.current;
    if (acc && acc.size > 0) {
      const indices = new Uint32Array(acc.size);
      const prevValues = new Uint8Array(acc.size);
      let i = 0;
      acc.forEach((prev, idx) => { indices[i] = idx; prevValues[i] = prev; i++; });
      pushUndo(indices, prevValues);
      (strokeAccRef as React.MutableRefObject<Map<number, number> | null>).current = null;
    }
    setIsDrawing(false); lastPosRef.current = null;
  }

  return { handlePointerDown, handlePointerMove, handlePointerUp, isDrawing };
}
