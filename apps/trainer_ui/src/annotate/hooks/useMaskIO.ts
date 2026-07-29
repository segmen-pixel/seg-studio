// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useMaskStore } from "../../store";
import { annotateMaskUrl } from "../../api";
import type { ImageItem, CacheEntry } from "../annotatorTypes";

// PNG encode worker (singleton, shared across hook instances)
let pngWorker: Worker | null = null;
let pngReqId = 0;
const pngCallbacks = new Map<number, (blob: Blob | null) => void>();

function getPngWorker(): Worker {
  if (!pngWorker) {
    pngWorker = new Worker(
      new URL('../pngEncodeWorker.ts', import.meta.url),
      { type: 'module' }
    );
    pngWorker.onmessage = (e: MessageEvent<{ id: number; blob: Blob | null }>) => {
      const cb = pngCallbacks.get(e.data.id);
      if (cb) { pngCallbacks.delete(e.data.id); cb(e.data.blob); }
    };
  }
  return pngWorker;
}

function encodeMaskPng(mask: Uint8Array, touched: Uint8Array | null, w: number, h: number): Promise<Blob | null> {
  return new Promise((resolve) => {
    const id = ++pngReqId;
    pngCallbacks.set(id, resolve);
    getPngWorker().postMessage({ mask, touched, width: w, height: h, id });
  });
}

/**
 * Mask persistence: load, save, autosave, flush, prefetch.
 */
export function useMaskIO(
  projectId: string | null,
  projectRef: React.MutableRefObject<string | null>,
  cacheRef: React.MutableRefObject<Map<string, CacheEntry>>,
  maskLoadPromisesRef: React.MutableRefObject<Map<string, Promise<CacheEntry>>>,
  prefetchEpochRef: React.MutableRefObject<number>,
  autoSaveTimerRef: React.MutableRefObject<number | null>,
  activeImageId: string | null,
  _getFilteredImages: () => ImageItem[]
) {
  // ---------------------------------------------------------------------------
  // loadMaskFor – fetch a mask PNG from the server and decode to Uint8Array
  // ---------------------------------------------------------------------------
  async function loadMaskFor(
    imageId: string,
    w: number,
    h: number,
    hasMask?: boolean,
    projectIdValue?: string
  ): Promise<CacheEntry> {
    const targetProjectId = projectIdValue ?? projectId;
    if (!targetProjectId) {
      return emptyEntry(w, h);
    }
    if (hasMask === false) {
      return emptyEntry(w, h);
    }
    const url = annotateMaskUrl(targetProjectId, imageId);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("mask not found");
      const blob = await res.blob();
      const img = new Image();
      const data = await new Promise<ImageData>((resolve, reject) => {
        const blobUrl = URL.createObjectURL(blob);
        img.onload = () => {
          URL.revokeObjectURL(blobUrl);
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.drawImage(img, 0, 0, w, h);
          resolve(ctx.getImageData(0, 0, w, h));
        };
        img.onerror = () => {
          URL.revokeObjectURL(blobUrl);
          reject(new Error("Failed to load mask image from blob"));
        };
        img.src = blobUrl;
      });
      const mask = new Uint8Array(w * h);
      const touched = new Uint8Array(w * h);
      const src32 = new Uint32Array(data.data.buffer);
      for (let i = 0; i < mask.length; i++) {
        const v = src32[i] & 0xFF; // extract R channel from little-endian ABGR
        mask[i] = v === 255 ? 0 : v;
        touched[i] = v !== 255 ? 1 : 0;
      }
      return {
        width: w,
        height: h,
        maskIndex: mask,
        touched,
        dirty: false,
        revision: cacheRef.current.get(imageId)?.revision ?? 0,
        undoStack: [],
        redoStack: [],
      };
    } catch {
      return emptyEntry(w, h);
    }
  }

  // ---------------------------------------------------------------------------
  // saveMask – encode mask to PNG and PUT to server
  // ---------------------------------------------------------------------------
  async function saveMask(
    imageId: string,
    mask: Uint8Array,
    w: number,
    h: number,
    projectIdOverride?: string | null,
    touchedOverride?: Uint8Array | null
  ) {
    const pid = projectIdOverride ?? projectId;
    if (!pid) return;
    const cached = cacheRef.current.get(imageId);
    const touched = touchedOverride ?? cached?.touched ?? null;
    // Encode PNG in Worker (off main thread)
    const blob = await encodeMaskPng(mask, touched, w, h);
    if (!blob) return;
    const form = new FormData();
    form.append("file", blob, `${imageId}.png`);
    const res = await fetch(annotateMaskUrl(pid, imageId), {
      method: "PUT",
      body: form,
    });
    if (!res.ok) throw new Error(await res.text());
  }

  // ---------------------------------------------------------------------------
  // autoSave – save a single image if dirty
  // ---------------------------------------------------------------------------
  async function autoSave(imageId: string) {
    if (!projectId) return;
    const cached = cacheRef.current.get(imageId);
    if (!cached || !cached.dirty) return;
    await saveMask(imageId, cached.maskIndex, cached.width, cached.height);
    cached.dirty = false;
    cached.revision += 1;
    cached.lastSavedAt = new Date().toISOString();
    cacheRef.current.set(imageId, cached);
  }

  // ---------------------------------------------------------------------------
  // flushDirtyMasks – save ALL dirty entries
  // ---------------------------------------------------------------------------
  async function flushDirtyMasks() {
    if (!projectId) return;
    const entries = Array.from(cacheRef.current.entries());
    for (const [id, entry] of entries) {
      if (!entry.dirty) continue;
      try {
        await saveMask(id, entry.maskIndex, entry.width, entry.height);
        entry.dirty = false;
        entry.revision += 1;
        entry.lastSavedAt = new Date().toISOString();
        cacheRef.current.set(id, entry);
      } catch {
        // ignore save errors on background flush
      }
    }
  }

  // ---------------------------------------------------------------------------
  // getOrLoadMaskEntry – return from cache or load
  // ---------------------------------------------------------------------------
  function getOrLoadMaskEntry(
    item: ImageItem,
    projectIdValue: string
  ): Promise<CacheEntry> {
    const cached = cacheRef.current.get(item.id);
    if (cached) return Promise.resolve(cached);
    const pending = maskLoadPromisesRef.current.get(item.id);
    if (pending) return pending;
    const task = loadMaskFor(
      item.id,
      item.width,
      item.height,
      item.annotation?.hasMask,
      projectIdValue
    )
      .then((loaded) => {
        const current = cacheRef.current.get(item.id);
        if (!current || !current.dirty) {
          cacheRef.current.set(item.id, loaded);
        }
        return loaded;
      })
      .finally(() => {
        maskLoadPromisesRef.current.delete(item.id);
      });
    maskLoadPromisesRef.current.set(item.id, task);
    return task;
  }

  // ---------------------------------------------------------------------------
  // prefetchProjectMasks – prefetch masks around the active image.
  //
  // Previously this fetched every image with hasMask=true eagerly, which on
  // large projects (e.g. 1800+ masked images) pulled tens of megabytes on
  // startup even though the user only interacts with a small neighbourhood
  // at a time. Now we only prefetch a window around `activeId` (or the
  // beginning of the list if no active image). Images outside the window
  // are loaded on demand via getOrLoadMaskEntry.
  // ---------------------------------------------------------------------------
  const PREFETCH_WINDOW_RADIUS = 50; // masks on each side of the active image

  async function prefetchProjectMasks(
    projectIdValue: string,
    items: ImageItem[],
    onProgress?: (loaded: number, total: number) => void,
    opts?: { activeId?: string | null; radius?: number }
  ) {
    const token = prefetchEpochRef.current;
    const radius = opts?.radius ?? PREFETCH_WINDOW_RADIUS;
    const activeId = opts?.activeId ?? null;

    const activeIdx = activeId
      ? items.findIndex((item) => item.id === activeId)
      : -1;
    const windowStart = activeIdx >= 0 ? Math.max(0, activeIdx - radius) : 0;
    const windowEnd = activeIdx >= 0
      ? Math.min(items.length, activeIdx + radius + 1)
      : Math.min(items.length, radius * 2 + 1);
    const windowSlice = items.slice(windowStart, windowEnd);

    const queue = windowSlice.filter(
      (item) => item.annotation?.hasMask && !cacheRef.current.has(item.id)
    );
    if (queue.length === 0) return;
    const total = queue.length;
    let loaded = 0;
    onProgress?.(0, total);
    const concurrency = 3;
    let cursor = 0;
    const worker = async () => {
      while (
        cursor < queue.length &&
        prefetchEpochRef.current === token &&
        projectRef.current === projectIdValue
      ) {
        const idx = cursor;
        cursor += 1;
        const item = queue[idx];
        if (!item) return;
        try {
          await getOrLoadMaskEntry(item, projectIdValue);
        } catch {
          // ignore prefetch errors
        }
        loaded += 1;
        onProgress?.(loaded, total);
      }
    };
    await Promise.all(Array.from({ length: concurrency }, () => worker()));
    // Bump maskVersion so perImageClasses re-scans with newly cached masks
    const st = useMaskStore.getState();
    useMaskStore.setState({ maskVersion: st.maskVersion + 1 });
  }

  // ---------------------------------------------------------------------------
  // scheduleAutosave – debounced 600ms
  // ---------------------------------------------------------------------------
  function scheduleAutosave(setStatus?: (msg: string) => void) {
    if (!activeImageId) return;
    if (autoSaveTimerRef.current !== null) {
      window.clearTimeout(autoSaveTimerRef.current);
    }
    (autoSaveTimerRef as React.MutableRefObject<number | null>).current =
      window.setTimeout(() => {
        (autoSaveTimerRef as React.MutableRefObject<number | null>).current =
          null;
        autoSave(activeImageId).catch((err) => {
          setStatus?.(`Auto-save failed: ${(err as Error).message}`);
        });
      }, 600);
  }

  // ---------------------------------------------------------------------------
  // markDirty / markTouched – write-back helpers
  // ---------------------------------------------------------------------------
  function markDirty() {
    if (!activeImageId) return;
    const cached = cacheRef.current.get(activeImageId);
    if (cached) {
      cached.dirty = true;
      // Update reference (not a copy) so autoSave sees the latest mask data.
      // This is safe: immer creates a new draft on next applyDelta, so the
      // reference won't be silently mutated after this point.
      const state = useMaskStore.getState();
      cached.maskIndex = state.maskIndex;
    }
    scheduleAutosave();
  }

  function markTouched(indices: Uint32Array) {
    if (!activeImageId) return;
    const cached = cacheRef.current.get(activeImageId);
    if (!cached) return;
    for (let i = 0; i < indices.length; i++) {
      cached.touched[indices[i]] = 1;
    }
  }

  return {
    loadMaskFor,
    saveMask,
    autoSave,
    flushDirtyMasks,
    getOrLoadMaskEntry,
    prefetchProjectMasks,
    scheduleAutosave,
    markDirty,
    markTouched,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function emptyEntry(w: number, h: number): CacheEntry {
  return {
    width: w,
    height: h,
    maskIndex: new Uint8Array(w * h),
    touched: new Uint8Array(w * h),
    dirty: false,
    revision: 0,
    undoStack: [],
    redoStack: [],
  };
}
