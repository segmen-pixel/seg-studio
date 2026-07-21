// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useRef } from "react";
import { useMaskStore } from "../../store";

const BITMAP_CACHE_MAX = 30;

type OverlayDeps = {
  overlayCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  uiCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** Must be called at draw time to get latest mutable state */
  getState: () => {
    width: number; height: number; maskIndex: Uint8Array;
    lut: Uint8ClampedArray;
    scale: number; offsetX: number; offsetY: number;
    assistPreview: Uint8Array | null;
    recipePreview: Uint8Array | null;
    measureStart: [number, number] | null;
    measureEnd: [number, number] | null;
    tool: string;
    brushSize: number;
    wandRef: { tolerance: number } | null;
    samRef: { points: [number, number][]; labels: number[]; box: [number, number, number, number] | null } | null;
    samBoxDraft: { start: [number, number]; end: [number, number] } | null;
    samMode: "point" | "box";
    spotDetectRef: object | null;
    spotCount: number;
    superpixelBoundary: Uint8Array | null;
    previewStyle: number;
    blinkPhase: number;
  };
};

export function useCanvasRendering(deps: OverlayDeps) {
  const rafIdRef = useRef<number | null>(null);
  const uiRafIdRef = useRef<number | null>(null);
  const pendingUiPosRef = useRef<[number, number] | null>(null);
  const bitmapCacheRef = useRef<Map<string, ImageBitmap>>(new Map());
  const workerRef = useRef<Worker | null>(null);
  const overlayReqIdRef = useRef(0);

  function getWorker(): Worker {
    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL('../overlayWorker.ts', import.meta.url),
        { type: 'module' }
      );
      workerRef.current.onmessage = (e: MessageEvent<{ buffer: ArrayBuffer; width: number; height: number; roi: { x: number; y: number; w: number; h: number } | null; reqId: number }>) => {
        const { buffer, width, height, roi, reqId } = e.data;
        // Discard stale responses from previous requests
        if (reqId !== overlayReqIdRef.current) return;
        const canvas = deps.overlayCanvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;
        const imageData = new ImageData(new Uint8ClampedArray(buffer), width, height);
        ctx.putImageData(imageData, roi?.x ?? 0, roi?.y ?? 0);
      };
    }
    return workerRef.current;
  }

  function terminateWorker() {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
  }

  function getBitmapFromCache(url: string): ImageBitmap | null {
    const cache = bitmapCacheRef.current;
    const bmp = cache.get(url);
    if (!bmp) return null;
    cache.delete(url);
    cache.set(url, bmp);
    return bmp;
  }

  async function putBitmapToCache(url: string): Promise<ImageBitmap> {
    const existing = getBitmapFromCache(url);
    if (existing) return existing;
    const res = await fetch(url);
    const blob = await res.blob();
    const bmp = await createImageBitmap(blob);
    const cache = bitmapCacheRef.current;
    cache.set(url, bmp);
    if (cache.size > BITMAP_CACHE_MAX) {
      const first = cache.keys().next().value;
      if (first) { cache.get(first)?.close(); cache.delete(first); }
    }
    return bmp;
  }

  function drawOverlay(dirty?: { x: number; y: number; w: number; h: number }) {
    const canvas = deps.overlayCanvasRef.current;
    const s = deps.getState();
    const { width, height, maskIndex, lut } = s;
    if (!canvas || width === 0 || height === 0) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height; dirty = undefined;
    }

    if (!dirty) {
      // Bump request ID — any in-flight Worker response with old ID will be discarded
      const reqId = ++overlayReqIdRef.current;

      // Don't clearRect here — the Worker response will overwrite via
      // putImageData. Clearing causes a visible flash between clear and
      // the async Worker response.  For image switches, invalidateOverlay()
      // ensures stale responses are discarded.

      // Calculate visible region (ROI) to avoid processing off-screen pixels
      const container = deps.containerRef.current;
      // Read view state directly from zustand store — React state closures
      // may be stale after handleFit (setView updates store synchronously
      // but React hasn't re-rendered yet).
      const freshView = useMaskStore.getState();
      const { scale, offsetX, offsetY } = freshView;
      const preview = s.assistPreview || s.recipePreview;
      let roi: { x: number; y: number; w: number; h: number } | undefined;
      // Skip ROI when preview is active — ROI-rendered preview disappears
      // outside the visible region when user zooms out.
      if (!preview && container && scale > 0) {
        const vw = container.clientWidth;
        const vh = container.clientHeight;
        const x0 = Math.max(0, Math.floor(-offsetX / scale));
        const y0 = Math.max(0, Math.floor(-offsetY / scale));
        const x1 = Math.min(width, Math.ceil((vw - offsetX) / scale));
        const y1 = Math.min(height, Math.ceil((vh - offsetY) / scale));
        const rw = x1 - x0;
        const rh = y1 - y0;
        // Only use ROI if it's significantly smaller than full image (>30% savings)
        if (rw > 0 && rh > 0 && rw * rh < width * height * 0.7) {
          roi = { x: x0, y: y0, w: rw, h: rh };
        }
      }
      getWorker().postMessage({
        maskIndex,
        lut,
        width,
        height,
        preview,
        superpixelBoundary: s.superpixelBoundary,
        previewStyle: s.previewStyle,
        blinkPhase: s.blinkPhase,
        roi,
        reqId,
      });
      return;
    }

    // Dirty rect: small & fast, draw on main thread immediately
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const lut32 = new Uint32Array(256);
    for (let c = 0; c < 256; c++) {
      const b = c * 4;
      lut32[c] = lut[b] | (lut[b + 1] << 8) | (lut[b + 2] << 16) | (lut[b + 3] << 24);
    }
    const { x, y, w, h } = dirty;
    const imgData = ctx.getImageData(x, y, w, h);
    const buf32 = new Uint32Array(imgData.data.buffer);
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        buf32[row * w + col] = lut32[maskIndex[(y + row) * width + (x + col)]];
      }
    }
    ctx.putImageData(imgData, x, y);
  }

  function scheduleDrawOverlay() {
    // Cancel any pending rAF to ensure we always use the LATEST state.
    // Without this, a stale rAF from setImage (zeroed mask) could run
    // instead of the correct rAF from setMask.
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current!);
    }
    rafIdRef.current = requestAnimationFrame(() => { rafIdRef.current = null; drawOverlay(); });
  }

  /** Call before setImage/setMask during image switch to immediately
   *  invalidate in-flight Worker responses so stale overlay data
   *  from the old image doesn't get painted on the new image.
   *  Does NOT clear the canvas — the old overlay stays briefly visible
   *  until the new drawOverlay replaces it (avoids partial-ROI artifacts). */
  function invalidateOverlay() {
    ++overlayReqIdRef.current;
  }

  function scheduleRenderUi(pos: [number, number] | null) {
    pendingUiPosRef.current = pos;
    if (uiRafIdRef.current !== null) return;
    uiRafIdRef.current = requestAnimationFrame(() => { uiRafIdRef.current = null; renderUi(pendingUiPosRef.current); });
  }

  function renderUi(pos: [number, number] | null) {
    const canvas = deps.uiCanvasRef.current;
    const s = deps.getState();
    const { width, height } = s;
    if (!canvas || width === 0 || height === 0) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (s.tool === "measure" && s.measureStart) {
      const end = s.measureEnd ?? s.measureStart;
      ctx.strokeStyle = "rgba(0,200,255,0.9)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(s.measureStart[0], s.measureStart[1]); ctx.lineTo(end[0], end[1]); ctx.stroke();
      const dist = Math.hypot(end[0] - s.measureStart[0], end[1] - s.measureStart[1]).toFixed(1);
      ctx.fillStyle = "rgba(0,200,255,0.9)"; ctx.font = "12px sans-serif";
      ctx.fillText(`${dist}px`, (s.measureStart[0] + end[0]) / 2 + 6, (s.measureStart[1] + end[1]) / 2 - 6);
    }
    if (!pos) return;
    if (s.tool === "brush" || s.tool === "eraser" || s.tool === "spotdetect") {
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(pos[0], pos[1], s.brushSize / 2, 0, Math.PI * 2); ctx.stroke();
    }
    if (s.tool === "wand" && s.wandRef) {
      const text = `tol: ${s.wandRef.tolerance.toFixed(0)}`;
      ctx.font = "12px sans-serif"; ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.strokeText(text, pos[0] + 12, pos[1] - 8);
      ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillText(text, pos[0] + 12, pos[1] - 8);
    }
    if (s.tool === "sam" || s.tool === "sambox") {
      // Draw box draft (during drag)
      if (s.samBoxDraft) {
        const bx = Math.min(s.samBoxDraft.start[0], s.samBoxDraft.end[0]);
        const by = Math.min(s.samBoxDraft.start[1], s.samBoxDraft.end[1]);
        const bw = Math.abs(s.samBoxDraft.end[0] - s.samBoxDraft.start[0]);
        const bh = Math.abs(s.samBoxDraft.end[1] - s.samBoxDraft.start[1]);
        ctx.strokeStyle = "rgba(0,120,255,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
        ctx.strokeRect(bx, by, bw, bh); ctx.setLineDash([]);
      }
      // Draw confirmed box
      if (s.samRef?.box) {
        const [bx1, by1, bx2, by2] = s.samRef.box;
        ctx.strokeStyle = "rgba(0,120,255,0.9)"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
        ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1); ctx.setLineDash([]);
      }
      // Draw point markers
      if (s.samRef) {
        for (let i = 0; i < s.samRef.points.length; i++) {
          const [px, py] = s.samRef.points[i];
          ctx.fillStyle = s.samRef.labels[i] === 1 ? "rgba(0,200,0,0.9)" : "rgba(200,0,0,0.9)";
          ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = "white"; ctx.lineWidth = 1.5; ctx.stroke();
        }
      }
      // Status text
      const samText = s.samMode === "box"
        ? (s.samRef?.box ? "SAM: box" + (s.samRef.points.length > 0 ? ` + ${s.samRef.points.length} pts` : "") : "SAM: draw box")
        : (s.samRef ? `SAM: ${s.samRef.points.length} pts` : "SAM: click");
      ctx.font = "12px sans-serif"; ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.strokeText(samText, pos[0] + 12, pos[1] - 8);
      ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillText(samText, pos[0] + 12, pos[1] - 8);
    }
    if (s.tool === "spotdetect" && s.spotDetectRef) {
      const text = `spots: ${s.spotCount}`;
      ctx.font = "12px sans-serif"; ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.strokeText(text, pos[0] + 12, pos[1] - 8);
      ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillText(text, pos[0] + 12, pos[1] - 8);
    }
    if (s.tool === "superpixel" && s.superpixelBoundary) {
      ctx.font = "12px sans-serif"; ctx.lineWidth = 3;
      const text = "SPix: click to select";
      ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.strokeText(text, pos[0] + 12, pos[1] - 8);
      ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillText(text, pos[0] + 12, pos[1] - 8);
    }
  }

  return { drawOverlay, scheduleDrawOverlay, invalidateOverlay, renderUi, scheduleRenderUi, getBitmapFromCache, putBitmapToCache, bitmapCacheRef, terminateWorker };
}
