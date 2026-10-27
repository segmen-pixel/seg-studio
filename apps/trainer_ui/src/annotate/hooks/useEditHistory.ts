// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useMaskStore } from "../../store";
import type { CacheEntry } from "../annotatorTypes";

type DrawOps = {
  drawOverlay: (dirty?: { x: number; y: number; w: number; h: number }) => void;
  scheduleDrawOverlay: () => void;
  markDirty: () => void;
  markTouched: (indices: Uint32Array) => void;
};

/**
 * Undo / redo / clear / copy / cut / paste and per-class erase.
 *
 * `getOps` is a getter that returns the latest draw helpers so the hook
 * never captures a stale closure.
 */
export function useEditHistory(
  getOps: () => DrawOps,
  activeImageId: string | null,
  cacheRef: React.MutableRefObject<Map<string, CacheEntry>>
) {
  const undo = useMaskStore((s) => s.undo);
  const redo = useMaskStore((s) => s.redo);
  const clearAll = useMaskStore((s) => s.clearAll);
  const _copyAll = useMaskStore((s) => s.copyAll);
  const cutAll = useMaskStore((s) => s.cutAll);
  const pasteAll = useMaskStore((s) => s.pasteAll);
  const activeClassId = useMaskStore((s) => s.activeClassId);
  const applyDelta = useMaskStore((s) => s.applyDelta);

  function handleUndo() {
    undo();
    getOps().scheduleDrawOverlay();
    getOps().markDirty();
  }

  function handleRedo() {
    redo();
    getOps().scheduleDrawOverlay();
    getOps().markDirty();
  }

  function handleClear() {
    clearAll();
    // Mark all pixels as touched (intentional clear)
    const cached = activeImageId ? cacheRef.current.get(activeImageId) : null;
    if (cached) cached.touched.fill(1);
    getOps().scheduleDrawOverlay();
    getOps().markDirty();
  }

  function handleCut() {
    cutAll();
    getOps().scheduleDrawOverlay();
    getOps().markDirty();
  }

  function handlePaste() {
    pasteAll();
    const cached = activeImageId ? cacheRef.current.get(activeImageId) : null;
    if (cached) cached.touched.fill(1);
    getOps().scheduleDrawOverlay();
    getOps().markDirty();
  }

  function clearClassById(classId: number) {
    const { maskIndex, width, height } = useMaskStore.getState();
    if (classId === 0 || width === 0 || height === 0) return;
    const indices: number[] = [];
    const prev: number[] = [];
    const next: number[] = [];
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        if (maskIndex[idx] !== classId) continue;
        indices.push(idx);
        prev.push(maskIndex[idx]);
        next.push(0);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
    if (indices.length === 0) return;
    const idxArr = new Uint32Array(indices);
    getOps().markTouched(idxArr);
    applyDelta(idxArr, new Uint8Array(prev), new Uint8Array(next));
    if (maxX >= minX && maxY >= minY) {
      getOps().drawOverlay({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 });
    } else {
      getOps().scheduleDrawOverlay();
    }
    getOps().markDirty();
  }

  function clearActiveClass() {
    clearClassById(activeClassId);
  }

  return {
    handleUndo,
    handleRedo,
    handleClear,
    handleCut,
    handlePaste,
    clearClassById,
    clearActiveClass,
  };
}
