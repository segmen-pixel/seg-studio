// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { useMaskStore, type ClassItem } from "../../store";
import type { ImageItem, CacheEntry } from "../annotatorTypes";

type ClassPixelStat = {
  classId: number;
  name: string;
  color: [number, number, number];
  pixels: number;
  ratio: number;
};

export type RegionLabel = {
  classId: number;
  cx: number;
  topY: number;
  count: number;
  name: string;
  color: [number, number, number];
};

/**
 * Debounced pixel-level statistics and connected-component region labels.
 * Region labels are computed in a Web Worker to avoid blocking the main thread.
 */
export function usePixelStats(
  images: ImageItem[],
  classesDraft: ClassItem[],
  activeImageId: string | null,
  _cacheRef: React.MutableRefObject<Map<string, CacheEntry>>,
  _maskVersion: number
) {
  // Subscribe only to maskVersion (lightweight number comparison).
  // Heavy computation is guarded to run only on image switch.
  const storeMaskVersion = useMaskStore((s) => s.maskVersion);

  // ---- per-class pixel counts (on image switch only) ----
  const [classPixelStats, setClassPixelStats] = useState<ClassPixelStat[]>([]);
  const pixelStatsTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevPixelImageRef = useRef<string | null>(null);

  useEffect(() => {
    // Only recompute on image switch, not on every stroke
    const imageChanged = prevPixelImageRef.current !== activeImageId;
    prevPixelImageRef.current = activeImageId;
    if (!imageChanged && classPixelStats.length > 0) return;

    if (pixelStatsTimerRef.current) clearTimeout(pixelStatsTimerRef.current);
    const computeStats = () => {
      const result: ClassPixelStat[] = [];
      if (classesDraft.length === 0) {
        setClassPixelStats(result);
        return;
      }
      const state = useMaskStore.getState();
      const counts = new Map<number, number>();
      let totalPixels = 0;
      const idx = (activeImageId && state.maskIndex && state.maskIndex.length > 0) ? state.maskIndex : null;
      if (idx) {
        for (let i = 0; i < idx.length; i++) {
          const v = idx[i]!;
          if (v === 0) continue;
          counts.set(v, (counts.get(v) ?? 0) + 1);
          totalPixels++;
        }
      }
      for (const cls of classesDraft) {
        if (cls.id === 0) continue;
        const px = counts.get(cls.id) ?? 0;
        result.push({
          classId: cls.id,
          name: cls.name,
          color: cls.color,
          pixels: px,
          ratio: totalPixels > 0 ? px / totalPixels : 0,
        });
      }
      setClassPixelStats(result);
    };
    // On image switch: compute immediately. Otherwise debounce.
    if (imageChanged) {
      computeStats();
    } else {
      pixelStatsTimerRef.current = setTimeout(computeStats, 500);
    }
    return () => {
      if (pixelStatsTimerRef.current) clearTimeout(pixelStatsTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeImageId, classesDraft, storeMaskVersion]);

  // ---- connected-component region labels (Web Worker) ----
  const [regionLabels, setRegionLabels] = useState<RegionLabel[]>([]);
  const prevImageRef = useRef<string | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const reqIdRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Lazy worker init
  function getWorker(): Worker {
    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL('../regionLabelWorker.ts', import.meta.url),
        { type: 'module' }
      );
      workerRef.current.onmessage = (e: MessageEvent<{ labels: RegionLabel[]; reqId: number }>) => {
        // Discard stale responses
        if (e.data.reqId !== reqIdRef.current) return;
        setRegionLabels(e.data.labels);
      };
    }
    return workerRef.current;
  }

  useEffect(() => {
    const imageChanged = prevImageRef.current !== activeImageId;
    prevImageRef.current = activeImageId;
    // Clear labels immediately on image switch
    if (imageChanged) {
      setRegionLabels([]);
    }

    if (timerRef.current) clearTimeout(timerRef.current);

    const dispatch = () => {
      const fresh = useMaskStore.getState();
      const { maskIndex, width, height, classes } = fresh;
      if (!maskIndex || width === 0 || height === 0) { setRegionLabels([]); return; }

      const activeClassIds = classes.filter((c) => c.id !== 0).map((c) => c.id);
      const classNames: Record<number, string> = {};
      const classColors: Record<number, [number, number, number]> = {};
      for (const c of classes) {
        classNames[c.id] = c.name;
        classColors[c.id] = c.color;
      }

      const reqId = ++reqIdRef.current;
      getWorker().postMessage({
        maskIndex,
        width,
        height,
        activeClassIds,
        classNames,
        classColors,
        reqId,
      });
    };

    // Short debounce: worker handles the heavy computation off-thread,
    // so we only need a brief debounce to batch rapid strokes.
    const delay = imageChanged ? 50 : 300;
    timerRef.current = setTimeout(dispatch, delay);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeImageId, storeMaskVersion]);

  // Cleanup worker on unmount
  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  return { classPixelStats, regionLabels, setRegionLabels };
}
