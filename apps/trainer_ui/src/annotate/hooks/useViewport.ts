// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useRef, useState, type RefObject } from "react";
import { useMaskStore } from "../../store";

/**
 * Viewport management: pan, zoom, fit-to-container.
 *
 * Reads `scale`, `offsetX`, `offsetY`, `imageUrl`, `setView` from useMaskStore
 * directly so callers don't need to thread them through.
 */
export function useViewport(
  containerRef: RefObject<HTMLDivElement | null>,
  setBrushSize?: React.Dispatch<React.SetStateAction<number>>,
  onBrushSizeChange?: (clientX: number, clientY: number) => void,
) {
  const [isPanning, setIsPanning] = useState(false);
  const [spacePressed, setSpacePressed] = useState(false);
  const panStartRef = useRef<{
    x: number;
    y: number;
    offsetX: number;
    offsetY: number;
  } | null>(null);

  // ---- handleFit ----
  function handleFit() {
    const rect = containerRef.current?.getBoundingClientRect();
    const st = useMaskStore.getState();
    const w = st.width;
    const h = st.height;
    if (!rect || w === 0 || h === 0) {
      st.setView(1, 0, 0);
      return;
    }
    const scaleFit = Math.min(rect.width / w, rect.height / h);
    const offsetXFit = (rect.width - w * scaleFit) / 2;
    const offsetYFit = (rect.height - h * scaleFit) / 2;
    st.setView(scaleFit, offsetXFit, offsetYFit);
  }

  // ---- handleZoom ----
  function handleZoom(delta: number) {
    const { scale, offsetX, offsetY, setView } = useMaskStore.getState();
    const next = Math.max(0.25, Math.min(4, scale + delta));
    setView(next, offsetX, offsetY);
  }

  // ---- wheel handler (native, non-passive) ----
  function handleWheelNative(event: WheelEvent) {
    const { imageUrl, scale, offsetX, offsetY, setView } =
      useMaskStore.getState();
    if (!imageUrl) return;
    event.preventDefault();
    // Ctrl + scroll → brush size (linear feel: step proportional to current size)
    if (event.ctrlKey && setBrushSize) {
      const d = Math.sign(event.deltaY);
      setBrushSize((prev) => {
        const step = Math.max(1, Math.round(prev * 0.15));
        const next = d > 0 ? Math.max(2, prev - step) : Math.min(200, prev + step);
        return next;
      });
      onBrushSizeChange?.(event.clientX, event.clientY);
      return;
    }
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const d = Math.sign(event.deltaY);
    const factor = d > 0 ? 0.9 : 1.1;
    const nextScale = Math.max(0.25, Math.min(6, scale * factor));
    const cx = event.clientX - rect.left;
    const cy = event.clientY - rect.top;
    const ix = (cx - offsetX) / scale;
    const iy = (cy - offsetY) / scale;
    const nextOffsetX = cx - ix * nextScale;
    const nextOffsetY = cy - iy * nextScale;
    setView(nextScale, nextOffsetX, nextOffsetY);
    // Re-render brush cursor at new scale
    onBrushSizeChange?.(event.clientX, event.clientY);
  }

  // ---- useEffect: attach native wheel listener ----
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (event: WheelEvent) => handleWheelNative(event);
    el.addEventListener("wheel", handler, { passive: false });
    return () => {
      el.removeEventListener("wheel", handler);
    };
  });

  // ---- space-key tracking (for pan-while-held) ----
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.code === "Space") {
        // Only capture space if not typing in an input
        const target = event.target as HTMLElement | null;
        if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
        event.preventDefault();
        setSpacePressed(true);
      }
    }
    function onKeyUp(event: KeyboardEvent) {
      if (event.code === "Space") setSpacePressed(false);
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  return {
    handleFit,
    handleZoom,
    isPanning,
    setIsPanning,
    spacePressed,
    panStartRef,
  };
}
