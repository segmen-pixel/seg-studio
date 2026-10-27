// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef, useState } from "react";
import type React from "react";
import { useMaskStore, type ClassItem } from "../../store";
import type { ImageItem, CacheEntry } from "../annotatorTypes";

/** Map from imageId to sorted array of non-zero classIds found in that image's mask. */
export type PerImageClassMap = Map<string, number[]>;

/**
 * Reads per-image class presence from image list items (annotation.classIds),
 * then incrementally updates the active image from the live Zustand maskIndex.
 */
export function usePerImageClassPresence(
  images: ImageItem[],
  classesDraft: ClassItem[],
  activeImageId: string | null,
  _cacheRef: React.MutableRefObject<Map<string, CacheEntry>>,
  _maskVersion: number,
  _projectId: string | null
): PerImageClassMap {
  const storeMaskVersion = useMaskStore((s) => s.maskVersion);
  const [result, setResult] = useState<PerImageClassMap>(new Map());
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // The classes that still exist, as a value that only changes when they do:
  // classesDraft is a new array on every keystroke in the class panel.
  const knownKey = classesDraft
    .filter((c) => c.id !== 0)
    .map((c) => c.id)
    .sort((a, b) => a - b)
    .join(",");

  // Build the map from the image list's stored classIds, keeping only ids the
  // class list still has. Deleting a class left the row's green bar behind:
  // the bar asks "any class present?" of the stored ids, while a dot needs a
  // colour and a deleted class has none, so the dots went and the bar stayed.
  // The stored ids are not rewritten here -- classes_reconcile_check is what
  // reports that drift -- they are simply not counted as present.
  useEffect(() => {
    const known = new Set(knownKey ? knownKey.split(",").map(Number) : []);
    const map: PerImageClassMap = new Map();
    for (const img of images) {
      const classIds = (img.annotation?.classIds ?? []).filter((id) => known.has(id));
      if (classIds.length > 0) {
        map.set(img.id, classIds);
      }
    }
    setResult(map);
  }, [images, knownKey]);

  // Update active image's class presence live, but debounce so strokes do not spam rerenders.
  useEffect(() => {
    if (!activeImageId) return;

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      const idx = useMaskStore.getState().maskIndex;
      if (idx.length === 0) {
        setResult((prev) => {
          const next = new Map(prev);
          next.delete(activeImageId);
          return next;
        });
        return;
      }

      const knownIds = new Set(
        classesDraft.filter((c) => c.id !== 0).map((c) => c.id)
      );
      if (knownIds.size === 0) {
        setResult((prev) => {
          const next = new Map(prev);
          next.delete(activeImageId);
          return next;
        });
        return;
      }

      const found = new Set<number>();
      for (let i = 0; i < idx.length; i++) {
        const v = idx[i]!;
        if (v !== 0 && knownIds.has(v)) {
          found.add(v);
          if (found.size === knownIds.size) break;
        }
      }
      const classes = found.size > 0 ? Array.from(found).sort((a, b) => a - b) : [];

      setResult((prev) => {
        const next = new Map(prev);
        if (classes.length > 0) {
          next.set(activeImageId, classes);
        } else {
          next.delete(activeImageId);
        }
        return next;
      });
    }, 300);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [activeImageId, classesDraft, storeMaskVersion]);

  return result;
}
