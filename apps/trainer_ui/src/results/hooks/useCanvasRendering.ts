// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect } from "react";
import type { useResultsState } from "./useResultsState";

type State = ReturnType<typeof useResultsState>;

/**
 * Canvas rendering logic: image draw, overlay draw, zoom/pan, fit, region labels.
 */
export function useCanvasRendering(s: State) {
  // ── Zoom / Pan handlers ──

  function handlePreviewWheel(e: React.WheelEvent) {
    e.preventDefault();
    const container = s.previewRef.current;
    if (!container || s.width === 0) return;
    const rect = container.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.max(0.1, Math.min(10, s.previewScale * factor));
    const ox = mx - (mx - s.previewOffset.x) * (newScale / s.previewScale);
    const oy = my - (my - s.previewOffset.y) * (newScale / s.previewScale);
    s.setPreviewScale(newScale);
    s.setPreviewOffset({ x: ox, y: oy });
  }

  function handlePreviewPointerDown(e: React.PointerEvent) {
    if (e.button !== 0 && e.button !== 1) return;
    if (s.calibrating) return;
    s.panRef.current = { startX: e.clientX, startY: e.clientY, ox: s.previewOffset.x, oy: s.previewOffset.y };
    (e.target as Element).setPointerCapture(e.pointerId);
  }

  function handlePreviewPointerMove(e: React.PointerEvent) {
    if (!s.panRef.current) return;
    s.setPreviewOffset({
      x: s.panRef.current.ox + (e.clientX - s.panRef.current.startX),
      y: s.panRef.current.oy + (e.clientY - s.panRef.current.startY),
    });
  }

  function handlePreviewPointerUp() { s.panRef.current = null; }

  function fitPreview() {
    const container = s.previewRef.current;
    if (!container || s.width === 0 || s.height === 0) return;
    const rect = container.getBoundingClientRect();
    const scaleFit = Math.min(rect.width / s.width, rect.height / s.height);
    const offsetX = (rect.width - s.width * scaleFit) / 2;
    const offsetY = (rect.height - s.height * scaleFit) / 2;
    s.setPreviewScale(scaleFit);
    s.setPreviewOffset({ x: offsetX, y: offsetY });
  }

  // ── Effects ──

  // Draw base image on canvas
  useEffect(() => {
    if (!s.imageUrl || s.width === 0 || s.height === 0) return;
    const img = new Image();
    img.src = s.imageUrl;
    img.onload = () => {
      const canvas = s.imageCanvasRef.current;
      if (!canvas) return;
      // Assigning to canvas.width/.height always clears the canvas, even when
      // the value is unchanged. Skip the resize when dimensions already match
      // and drop the now-redundant clearRect so the old image stays painted
      // until drawImage overwrites it — removes the black flash visible when
      // browsing through same-size images in the Results panel.
      if (canvas.width !== s.width || canvas.height !== s.height) {
        canvas.width = s.width;
        canvas.height = s.height;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, s.width, s.height);
      fitPreview();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.imageUrl, s.width, s.height]);

  // Fit on resize
  useEffect(() => {
    if (s.width === 0 || s.height === 0) return;
    fitPreview();
    const onResize = () => fitPreview();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.width, s.height]);

  // Draw overlay — GT (annotation) mask with class colors
  useEffect(() => {
    const canvas = s.overlayCanvasRef.current;
    if (!canvas || s.width === 0 || s.height === 0) return;
    canvas.width = s.width;
    canvas.height = s.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const srcMask = s.gtMaskIndex;
    if (srcMask.length !== s.width * s.height) { ctx.clearRect(0, 0, s.width, s.height); return; }
    const useOffscreen = typeof OffscreenCanvas !== "undefined";
    const drawCtx = useOffscreen
      ? new OffscreenCanvas(s.width, s.height).getContext("2d", { willReadFrequently: true })!
      : ctx;
    const imageData = drawCtx.createImageData(s.width, s.height);
    for (let i = 0; i < srcMask.length; i += 1) {
      const classId = srcMask[i] ?? 0;
      if (classId === 0 || classId === 255) continue;
      const base = i * 4;
      const colorBase = classId * 4;
      imageData.data[base] = s.lut[colorBase] ?? 0;
      imageData.data[base + 1] = s.lut[colorBase + 1] ?? 0;
      imageData.data[base + 2] = s.lut[colorBase + 2] ?? 0;
      imageData.data[base + 3] = s.lut[colorBase + 3] ?? 0;
    }
    if (useOffscreen) {
      drawCtx.putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, s.width, s.height);
      ctx.drawImage(drawCtx.canvas as OffscreenCanvas, 0, 0);
    } else {
      ctx.putImageData(imageData, 0, 0);
    }
  }, [s.gtMaskIndex, s.lut, s.width, s.height]);

  // Region labels via worker
  useEffect(() => {
    if (s.width === 0 || s.height === 0 || s.maskIndex.length === 0) { s.setRegionLabels([]); return; }
    if (!s.regionWorkerRef.current) {
      s.regionWorkerRef.current = new Worker(
        new URL("../../annotate/regionLabelWorker.ts", import.meta.url),
        { type: "module" },
      );
      s.regionWorkerRef.current.onmessage = (e: MessageEvent<{ labels: Array<{ classId: number; cx: number; topY: number; count: number; name: string; color: [number, number, number] }>; reqId: number }>) => {
        if (e.data.reqId === s.regionReqIdRef.current) s.setRegionLabels(e.data.labels);
      };
    }
    const reqId = ++s.regionReqIdRef.current;
    const classNames: Record<number, string> = {};
    const classColors: Record<number, [number, number, number]> = {};
    const activeClassIds: number[] = [];
    s.effectiveClasses.forEach((c) => {
      if (c.id > 0) { classNames[c.id] = c.name; classColors[c.id] = c.color; activeClassIds.push(c.id); }
    });
    s.regionWorkerRef.current.postMessage({
      maskIndex: s.effectiveMaskIndex,
      width: s.width,
      height: s.height,
      activeClassIds,
      classNames,
      classColors,
      reqId,
      minArea: s.ppMinArea,
      maxArea: s.ppMaxArea,
    });
    return () => { s.regionReqIdRef.current++; };
  }, [s.effectiveMaskIndex, s.width, s.height, s.effectiveClasses, s.ppMinArea, s.ppMaxArea]);

  // Prediction overlay: class-color fill with pulsing pattern (shown when Predict is ON)
  useEffect(() => {
    const canvas = s.gtOutlineCanvasRef.current;
    if (!canvas || !s.showGtOutline || s.width === 0 || s.height === 0) {
      if (canvas) { canvas.width = 1; canvas.height = 1; }
      return;
    }
    const hasPred = s.effectiveMaskIndex.length === s.width * s.height;
    if (!hasPred) {
      canvas.width = 1; canvas.height = 1;
      return;
    }
    canvas.width = s.width;
    canvas.height = s.height;

    const w = s.width;
    const h = s.height;
    const pat = s.predOverlayPattern;

    // Build prediction class-color fill
    const predColorCanvas = document.createElement("canvas");
    predColorCanvas.width = w; predColorCanvas.height = h;
    const pcCtx = predColorCanvas.getContext("2d")!;
    const pcData = pcCtx.createImageData(w, h);
    for (let i = 0; i < s.effectiveMaskIndex.length; i++) {
      const cls = s.effectiveMaskIndex[i] ?? 0;
      if (cls === 0 || cls === 255) continue;
      const b = i * 4;
      const cb = cls * 4;
      pcData.data[b] = s.lut[cb] ?? 0;
      pcData.data[b + 1] = s.lut[cb + 1] ?? 0;
      pcData.data[b + 2] = s.lut[cb + 2] ?? 0;
      pcData.data[b + 3] = s.lut[cb + 3] ?? 0;
    }
    pcCtx.putImageData(pcData, 0, 0);

    // Build prediction foreground mask (white where predicted)
    const predFgCanvas = document.createElement("canvas");
    predFgCanvas.width = w; predFgCanvas.height = h;
    const pfCtx = predFgCanvas.getContext("2d")!;
    const pfData = pfCtx.createImageData(w, h);
    for (let i = 0; i < s.effectiveMaskIndex.length; i++) {
      const v = s.effectiveMaskIndex[i] ?? 0;
      if (v > 0 && v !== 255) {
        const b = i * 4;
        pfData.data[b] = 255; pfData.data[b + 1] = 255;
        pfData.data[b + 2] = 255; pfData.data[b + 3] = 255;
      }
    }
    pfCtx.putImageData(pfData, 0, 0);

    // Create pattern tile for pulsing effect
    const CYAN: [number, number, number] = [0, 220, 255];
    function makePatternCanvas(): HTMLCanvasElement | null {
      if (pat === "none" || pat === "tint") return null;
      const sz = pat === "fine-dots" ? 6 : pat === "dots" ? 10 : 8;
      const pc = document.createElement("canvas");
      pc.width = sz; pc.height = sz;
      const p = pc.getContext("2d")!;
      p.strokeStyle = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},0.6)`;
      p.fillStyle = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},0.6)`;
      if (pat === "hatch") {
        p.lineWidth = 1; p.beginPath();
        p.moveTo(0, sz); p.lineTo(sz, 0);
        p.moveTo(-sz * 0.5, sz * 0.5); p.lineTo(sz * 0.5, -sz * 0.5);
        p.moveTo(sz * 0.5, sz * 1.5); p.lineTo(sz * 1.5, sz * 0.5);
        p.stroke();
      } else if (pat === "dots") {
        p.beginPath(); p.arc(sz / 2, sz / 2, 2.5, 0, Math.PI * 2); p.fill();
      } else if (pat === "fine-dots") {
        p.beginPath(); p.arc(sz / 2, sz / 2, 1.2, 0, Math.PI * 2); p.fill();
      } else if (pat === "crosshatch") {
        p.lineWidth = 0.8; p.beginPath();
        p.moveTo(0, sz); p.lineTo(sz, 0);
        p.moveTo(-sz * 0.5, sz * 0.5); p.lineTo(sz * 0.5, -sz * 0.5);
        p.moveTo(sz * 0.5, sz * 1.5); p.lineTo(sz * 1.5, sz * 0.5);
        p.stroke(); p.beginPath();
        p.moveTo(0, 0); p.lineTo(sz, sz);
        p.moveTo(-sz * 0.5, sz * 0.5); p.lineTo(sz * 0.5, sz * 1.5);
        p.moveTo(sz * 0.5, -sz * 0.5); p.lineTo(sz * 1.5, sz * 0.5);
        p.stroke();
      }
      return pc;
    }
    const patternCanvas = makePatternCanvas();

    let animId = 0;
    const ctx = canvas.getContext("2d")!;
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = w; tmpCanvas.height = h;
    const tmpCtx = tmpCanvas.getContext("2d")!;

    function draw() {
      const t = performance.now() * 0.003;
      const pulse = 0.3 + 0.7 * Math.pow(Math.max(0, Math.sin(t * 2.5)), 1.5);

      ctx.clearRect(0, 0, w, h);

      // 1. Draw prediction class-color fill (static)
      ctx.drawImage(predColorCanvas, 0, 0);

      // 2. Draw pulsing pattern overlay (clipped to pred foreground)
      if (patternCanvas) {
        tmpCtx.clearRect(0, 0, w, h);
        tmpCtx.drawImage(predFgCanvas, 0, 0);
        tmpCtx.globalCompositeOperation = "source-in";
        tmpCtx.globalAlpha = pulse;
        const canvasPat = tmpCtx.createPattern(patternCanvas, "repeat");
        if (canvasPat) { tmpCtx.fillStyle = canvasPat; tmpCtx.fillRect(0, 0, w, h); }
        tmpCtx.globalCompositeOperation = "source-over";
        tmpCtx.globalAlpha = 1;
        ctx.drawImage(tmpCanvas, 0, 0);
      } else {
        // Default: pulsing tint
        tmpCtx.clearRect(0, 0, w, h);
        tmpCtx.drawImage(predFgCanvas, 0, 0);
        tmpCtx.globalCompositeOperation = "source-in";
        tmpCtx.globalAlpha = 1;
        tmpCtx.fillStyle = `rgba(${CYAN[0]},${CYAN[1]},${CYAN[2]},${0.2 * pulse})`;
        tmpCtx.fillRect(0, 0, w, h);
        tmpCtx.globalCompositeOperation = "source-over";
        tmpCtx.globalAlpha = 1;
        ctx.drawImage(tmpCanvas, 0, 0);
      }

      animId = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(animId);
  }, [s.showGtOutline, s.effectiveMaskIndex, s.width, s.height, s.predOverlayPattern, s.lut]);

  // Cleanup worker on unmount
  useEffect(() => () => { s.regionWorkerRef.current?.terminate(); }, []);

  return {
    handlePreviewWheel,
    handlePreviewPointerDown,
    handlePreviewPointerMove,
    handlePreviewPointerUp,
    fitPreview,
  };
}
