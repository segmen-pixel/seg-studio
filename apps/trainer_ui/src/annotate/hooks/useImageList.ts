// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useMaskStore } from "../../store";
import {
  annotateImageUrl,
  deleteAnnotateItem,
  exportAnnotateAnnotations,
  fetchAnnotateItems,
  prepareAnnotateDataset,
  uploadAnnotateImages,
  importAnnotateZip,
  uploadVideoFrames,
} from "../../api";
import { downloadBlob } from "../imageProcessing";
import { useI18n } from "../../i18n";
import type { ImageItem, CacheEntry } from "../annotatorTypes";

export type ImageListOps = {
  saveMask: (
    imageId: string,
    mask: Uint8Array,
    w: number,
    h: number,
    projectIdOverride?: string | null
  ) => Promise<void>;
  autoSave: (imageId: string) => Promise<void>;
  handleFit: () => void;
  /** selectImage used only for preload (ImageBitmap warm-up). */
  selectImage?: (item: ImageItem) => void;
  /** putBitmapToCache for adjacent image preloading. */
  putBitmapToCache?: (url: string) => Promise<ImageBitmap>;
  /** getBitmapFromCache check. */
  getBitmapFromCache?: (url: string) => ImageBitmap | null;
  /** Immediately invalidate in-flight Worker responses + clear overlay canvas.
   *  Call before setImage/setMask during image switch. */
  invalidateOverlay?: () => void;
};

export type DatasetStats = {
  total: number;
  withMask: number;
};

export type PrepareReport = {
  train_count: number;
  val_count: number;
  with_mask: number;
  auto_val_from_train_count?: number;
} | null;

async function convertToPng(file: File): Promise<File> {
  if (file.type === "image/png") return file;
  // TIFF is not supported by browser <img> — send as-is, backend converts
  if (/\.tiff?$/i.test(file.name)) return file;
  const img = new Image();
  const url = URL.createObjectURL(file);
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = reject;
    img.src = url;
  });
  URL.revokeObjectURL(url);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0);
  const blob = await new Promise<Blob>((resolve) =>
    canvas.toBlob((b) => resolve(b!), "image/png")
  );
  const stem = file.name.replace(/\.[^.]+$/, "");
  return new File([blob], `${stem}.png`, { type: "image/png" });
}

export function useImageList(
  projectId: string | null,
  projectRef: React.MutableRefObject<string | null>,
  cacheRef: React.MutableRefObject<Map<string, CacheEntry>>,
  maskLoadPromisesRef: React.MutableRefObject<Map<string, Promise<CacheEntry>>>,
  prefetchEpochRef: React.MutableRefObject<number>,
  loadingTargetRef: React.MutableRefObject<string | null>,
  allowBlockingUi: boolean,
  activeImageId: string | null,
  setActiveImageId: (id: string | null) => void,
  activeImageIdRef: React.MutableRefObject<string | null>,
  autoSaveTimerRef: React.MutableRefObject<number | null>,
  pendingSwitchRef: React.MutableRefObject<string | null>,
  getOps: () => ImageListOps,
  /** getOrLoadMaskEntry from useMaskIO */
  getOrLoadMaskEntry: (item: ImageItem, projectIdValue: string) => Promise<CacheEntry>,
  /** prefetchProjectMasks from useMaskIO */
  prefetchProjectMasks: (
    projectIdValue: string,
    items: ImageItem[],
    onProgress?: (loaded: number, total: number) => void,
    opts?: { activeId?: string | null; radius?: number },
  ) => Promise<void>,
  setStatus: (msg: string) => void,
  setBusyMessage: (msg: string | null) => void,
  setPrefetchMessage: (msg: string | null) => void
) {
  const { t } = useI18n();
  // ---- local state ----
  /** Tracks which image ID's data is actually in the zustand store right now.
   *  React's `activeImageId` can advance ahead of the store during concurrent
   *  selectImage calls — this ref stays in sync with setImage/setMask. */
  const storeImageIdRef = useRef<string | null>(null);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isListDragActive, setIsListDragActive] = useState(false);
  const listDragDepthRef = useRef(0);
  const lastSelectedIndexRef = useRef<number | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStats>({
    total: 0,
    withMask: 0,
  });
  const [prepareReport, setPrepareReport] = useState<PrepareReport>(null);

  // ---- derived ----
  const filteredImages = images;

  useEffect(() => {
    setDatasetStats({
      total: images.length,
      withMask: images.filter((item) => item.annotation?.hasForeground || item.annotation?.markedClean).length,
    });
  }, [images]);

  const applyImageAnnotationSummary = useCallback((imageId: string, classIds: number[]) => {
    let nextImages: ImageItem[] | null = null;
    setImages((prev) => {
      let changed = false;
      nextImages = prev.map((item) => {
        if (item.id !== imageId) return item;
        const prevClassIds = item.annotation?.classIds ?? [];
        const sameLength = prevClassIds.length === classIds.length;
        const sameIds =
          sameLength && prevClassIds.every((value, index) => value === classIds[index]);
        const nextHasForeground = classIds.length > 0;
        // hasMask = true when mask has been saved (even if all-background / "Mark Clean")
        const nextHasMask = item.annotation?.hasMask || nextHasForeground;
        if ((item.annotation?.hasMask ?? false) === nextHasMask
            && (item.annotation?.hasForeground ?? false) === nextHasForeground
            && sameIds) {
          return item;
        }
        changed = true;
        return {
          ...item,
          annotation: {
            revision: item.annotation?.revision ?? 0,
            lastSavedAt: item.annotation?.lastSavedAt ?? null,
            hasForeground: nextHasForeground,
            hasMask: nextHasMask,
            classIds,
          },
        };
      });
      return changed && nextImages ? nextImages : prev;
    });
  }, []);

  function scheduleInteractionUnlock(targetImageId: string) {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (
          loadingTargetRef.current === targetImageId &&
          pendingSwitchRef.current === targetImageId
        ) {
          loadingTargetRef.current = null;
          setBusyMessage(null);
        }
      });
    });
  }

  // ---- store accessors (individual selectors to avoid full-store re-renders) ----
  const setImage = useMaskStore((s) => s.setImage);
  const setMask = useMaskStore((s) => s.setMask);
  const setHistory = useMaskStore((s) => s.setHistory);
  const setView = useMaskStore((s) => s.setView);

  async function loadAnnotateItems(id: string, forceReselect: boolean = false, sync: boolean = false) {
    const shouldLockLoad =
      allowBlockingUi && images.length === 0 && !activeImageIdRef.current;
    try {
      if (shouldLockLoad) {
        loadingTargetRef.current = "__dataset__";
        setBusyMessage("Loading images...");
      }
      const data = await fetchAnnotateItems(id, sync);
      if (projectRef.current !== id) return;
      const items: ImageItem[] = data.items || [];
      setImages(items);
      const stats: DatasetStats = {
        total: items.length,
        withMask: items.filter((item) => item.annotation?.hasForeground || item.annotation?.markedClean).length,
      };
      setDatasetStats(stats);
      setSelectedIds((prev) => {
        const existing = new Set(items.map((item) => item.id));
        const next = new Set<string>();
        prev.forEach((pid) => {
          if (existing.has(pid)) next.add(pid);
        });
        return next;
      });
      // Cancel any previous prefetch before starting a new one. Only fetch a
      // window around the (future) active image so very large projects don't
      // spend tens of MB before the user can start working.
      prefetchEpochRef.current += 1;
      {
        let lastUpdate = 0;
        const activeIdForWindow = forceReselect
          ? (items[0]?.id ?? null)
          : (activeImageIdRef.current ?? items[0]?.id ?? null);
        void prefetchProjectMasks(id, items, (loaded, total) => {
          if (loaded >= total) {
            setPrefetchMessage(null);
          } else {
            const now = Date.now();
            if (now - lastUpdate > 200 || loaded === 0) {
              lastUpdate = now;
              setPrefetchMessage(`Caching saved masks... ${loaded}/${total}`);
            }
          }
        }, { activeId: activeIdForWindow });
      }
      if (items.length > 0) {
        const firstVisibleItem = items[0];
        if (forceReselect) {
          if (firstVisibleItem) await selectImage(firstVisibleItem);
        } else {
          const currentActiveId = activeImageIdRef.current;
          const exists =
            currentActiveId && items.find((item) => item.id === currentActiveId);
          if (!exists && firstVisibleItem) {
            await selectImage(firstVisibleItem);
          } else if (shouldLockLoad && loadingTargetRef.current === "__dataset__") {
            loadingTargetRef.current = null;
            setBusyMessage(null);
          }
        }
      } else {
        loadingTargetRef.current = null;
        setBusyMessage(null);
        setActiveImageId(null);
        storeImageIdRef.current = null;
        setImage("", 0, 0);
        setMask(new Uint8Array(), 0, 0);
        setHistory([], []);
        setView(1, 0, 0);
      }
    } catch (err) {
      loadingTargetRef.current = null;
      setBusyMessage(null);
      setPrefetchMessage(null);
      setStatus(`Dataset failed: ${(err as Error).message}`);
    }
  }

  async function selectImage(item: ImageItem) {
    if (!projectId) return;
    (pendingSwitchRef as React.MutableRefObject<string | null>).current = item.id;
    const ops = getOps();
    const curState = useMaskStore.getState();
    // Use storeImageIdRef (not activeImageId) for snapshot — React state can
    // advance ahead of the store during concurrent selectImage calls.
    const snapshotId = storeImageIdRef.current;
    if (snapshotId && curState.width > 0 && curState.height > 0) {
      const prev = cacheRef.current.get(snapshotId);
      const wasDirty = prev?.dirty ?? false;
      // Snapshot mask data into cache (~10ms for 10.5MB slice).
      // Using slice() so the cache owns its own copy — safe even after
      // the store's maskIndex gets replaced by setMask below.
      const snapshot = curState.maskIndex.slice();
      cacheRef.current.set(snapshotId, {
        width: curState.width,
        height: curState.height,
        maskIndex: snapshot,
        touched: prev?.touched ?? new Uint8Array(curState.width * curState.height),
        dirty: false,  // snapshot is authoritative, mark clean
        revision: prev?.revision ?? 0,
        undoStack: curState.undoStack,
        redoStack: curState.redoStack,
        lastSavedAt: prev?.lastSavedAt,
      });
      // Fire-and-forget save — don't await, don't block image switch.
      // The snapshot in cache is safe regardless of save completion.
      if (wasDirty) {
        // Cancel any pending scheduleAutosave timer
        if (autoSaveTimerRef.current !== null) {
          window.clearTimeout(autoSaveTimerRef.current);
          (autoSaveTimerRef as React.MutableRefObject<number | null>).current = null;
        }
        // Save the snapshot directly (not via autoSave, which checks dirty flag)
        ops.saveMask(snapshotId, snapshot, curState.width, curState.height).catch(() => {});
      }
    }
    // --- Cache eviction (adaptive max: 20 for small images, 8 for large) ---
    const megapixels = (item.width * item.height) / 1e6;
    const CACHE_MAX = megapixels > 4 ? 8 : 20;
    if (cacheRef.current.size > CACHE_MAX) {
      // Protect: active image, target image, dirty entries, adjacent images
      const idx = filteredImages.findIndex((img) => img.id === item.id);
      const adjacentIds = new Set<string>();
      for (const delta of [-2, -1, 0, 1, 2]) {
        const neighbor = filteredImages[idx + delta];
        if (neighbor) adjacentIds.add(neighbor.id);
      }
      if (snapshotId) adjacentIds.add(snapshotId);
      for (const [id, entry] of cacheRef.current) {
        if (cacheRef.current.size <= CACHE_MAX) break;
        if (adjacentIds.has(id) || entry.dirty) continue;
        cacheRef.current.delete(id);
      }
    }
    if (pendingSwitchRef.current !== item.id) return;
    const url = annotateImageUrl(projectId, item.filename);
    const hasCached = cacheRef.current.has(item.id);
    const maskPromise = hasCached
      ? Promise.resolve(cacheRef.current.get(item.id)!)
      : getOrLoadMaskEntry(item, projectId);

    const shouldLockSwitch =
      allowBlockingUi &&
      (!hasCached || !activeImageIdRef.current);
    if (shouldLockSwitch) {
      loadingTargetRef.current = item.id;
      setBusyMessage(`Loading ${item.name}...`);
    }
    const imageWarmPromise = ops.putBitmapToCache
      ? ops.putBitmapToCache(url).catch(() => null)
      : Promise.resolve(null);
    const cached = cacheRef.current.get(item.id);
    if (cached) {
      // Don't await bitmap warmup — switch immediately so the UI feels responsive.
      // The paintImage effect will load the bitmap asynchronously.
      // For large images (>4MP), bitmap warmup was skipped during preload,
      // so awaiting here would block the switch for seconds.
      if (pendingSwitchRef.current !== item.id) return;
      // Immediately invalidate in-flight Worker responses before switching
      ops.invalidateOverlay?.();
      // Synchronous atomic update — setImage clears mask, setMask restores correct one
      setActiveImageId(item.id);
      setImage(url, item.width, item.height);
      setMask(cached.maskIndex, cached.width, cached.height);
      storeImageIdRef.current = item.id;  // store now holds this image's data
      setHistory(cached.undoStack, cached.redoStack);
      requestAnimationFrame(() => ops.handleFit());
      if (shouldLockSwitch) scheduleInteractionUnlock(item.id);
      preloadAdjacentImages(item);
      return;
    }
    // Load mask data — don't await bitmap warmup, the paintImage effect
    // will handle bitmap loading asynchronously.  Awaiting bitmap here
    // blocks image switching for seconds on large images (3840x2748).
    const loaded = await maskPromise;
    // Bitmap warmup continues in background (fire-and-forget)
    if (pendingSwitchRef.current !== item.id) return;
    // Immediately invalidate in-flight Worker responses before switching
    ops.invalidateOverlay?.();
    setActiveImageId(item.id);
    setImage(url, item.width, item.height);
    setMask(loaded.maskIndex, loaded.width, loaded.height);
    storeImageIdRef.current = item.id;  // store now holds this image's data
    setHistory(loaded.undoStack, loaded.redoStack);
    requestAnimationFrame(() => ops.handleFit());
    if (shouldLockSwitch) scheduleInteractionUnlock(item.id);
    preloadAdjacentImages(item);
  }

  function preloadAdjacentImages(currentItem: ImageItem) {
    if (!projectId) return;
    const ops = getOps();
    const idx = filteredImages.findIndex((img) => img.id === currentItem.id);
    if (idx < 0) return;
    const megapixels = (currentItem.width * currentItem.height) / 1e6;
    // For large images: preload mask data only (skip heavy ImageBitmap decode)
    const deltas = megapixels > 4 ? [1, -1] : [-1, 1, 2, -2];
    for (const delta of deltas) {
      const neighbor = filteredImages[idx + delta];
      if (neighbor) {
        // Preload mask data (lightweight — just fetch + decode PNG to Uint8Array)
        if (!cacheRef.current.has(neighbor.id)) {
          getOrLoadMaskEntry(neighbor, projectId).catch(() => {});
        }
        // Preload image bitmap (skip for large images — too heavy)
        if (megapixels <= 4) {
          const url = annotateImageUrl(projectId, neighbor.filename);
          if (ops.putBitmapToCache) {
            ops.putBitmapToCache(url).catch(() => {});
          }
        }
      }
    }
  }

  function handleSelectClick(
    item: ImageItem,
    event: React.MouseEvent,
    openImage: boolean
  ) {
    event.preventDefault();
    const idx = filteredImages.findIndex((img) => img.id === item.id);
    if (idx < 0) return;
    const isToggle = event.metaKey || event.ctrlKey;
    const existingSelected = Array.from(selectedIds);
    const firstSelectedIndex = filteredImages.findIndex((img) =>
      existingSelected.includes(img.id)
    );
    const anchorIndex =
      lastSelectedIndexRef.current ??
      (firstSelectedIndex >= 0 ? firstSelectedIndex : null);
    const isRange = event.shiftKey && anchorIndex !== null;
    setSelectedIds((prev) => {
      let next = new Set(prev);
      if (isRange && anchorIndex !== null) {
        const start = Math.min(anchorIndex, idx);
        const end = Math.max(anchorIndex, idx);
        const rangeIds = filteredImages.slice(start, end + 1).map((img) => img.id);
        rangeIds.forEach((id) => next.add(id));
        next.add(item.id);
      } else if (isToggle) {
        if (next.has(item.id)) next.delete(item.id);
        else next.add(item.id);
      } else {
        next = new Set([item.id]);
      }
      return next;
    });
    lastSelectedIndexRef.current = idx;
    if (openImage && !isRange && !isToggle) {
      selectImage(item);
    }
  }

  const arrowNavTimerRef = useRef<number | null>(null);
  // While a rapid burst of arrow keys is in flight, keep track of where the
  // cursor should be *after* the pending activeImageId state flush. Without
  // this, a second keystroke that arrives before React re-renders uses the
  // stale closure's activeImageId and the list feels like it's skipping keys.
  const pendingArrowIdxRef = useRef<number | null>(null);

  function handleArrowNav(direction: "up" | "down", shiftKey: boolean) {
    const baselineIdx = pendingArrowIdxRef.current ?? filteredImages.findIndex(
      (img) => img.id === (activeImageIdRef.current ?? activeImageId),
    );
    if (baselineIdx < 0 && filteredImages.length > 0) {
      const first = filteredImages[0]!;
      setActiveImageId(first.id);
      selectImage(first);
      setSelectedIds(new Set([first.id]));
      lastSelectedIndexRef.current = 0;
      pendingArrowIdxRef.current = null;
      return;
    }
    const nextIdx = direction === "up"
      ? Math.max(0, baselineIdx - 1)
      : Math.min(filteredImages.length - 1, baselineIdx + 1);
    if (nextIdx === baselineIdx) return;

    const nextItem = filteredImages[nextIdx]!;
    pendingArrowIdxRef.current = nextIdx;

    // Update list highlight immediately (cheap)
    setActiveImageId(nextItem.id);

    // Debounce the heavy selectImage call — if the user keeps pressing arrow,
    // only load the final image after 150ms of inactivity.
    if (arrowNavTimerRef.current) clearTimeout(arrowNavTimerRef.current);
    arrowNavTimerRef.current = window.setTimeout(() => {
      arrowNavTimerRef.current = null;
      pendingArrowIdxRef.current = null;
      selectImage(nextItem);
    }, 150);

    if (shiftKey) {
      // Shift+Arrow: select range from anchor to current position
      const anchor = lastSelectedIndexRef.current ?? baselineIdx;
      const start = Math.min(anchor, nextIdx);
      const end = Math.max(anchor, nextIdx);
      const rangeIds = filteredImages.slice(start, end + 1).map((img) => img.id);
      setSelectedIds(new Set(rangeIds));
    } else {
      // Arrow only: single select
      setSelectedIds(new Set([nextItem.id]));
      lastSelectedIndexRef.current = nextIdx;
    }

    // Scroll into view
    requestAnimationFrame(() => {
      const el = document.querySelector(`[data-image-id="${nextItem.id}"]`);
      el?.scrollIntoView({ block: "nearest" });
    });
  }

  function selectAllFiltered() {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      images.forEach((item) => next.add(item.id));
      return next;
    });
  }

  async function handleDeleteSelected() {
    if (!projectId) return;
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    const ok = window.confirm(
      `Delete ${ids.length} image(s) and masks? This cannot be undone.`
    );
    if (!ok) return;
    const total = ids.length;
    setBusyMessage(
      t("imageList.deletingProgress")
        .replace("{done}", "0")
        .replace("{total}", String(total)),
    );
    const removed = new Set<string>();
    let failed = 0;
    // One bulk request per 2000 IDs. The server takes the project lock
    // and rewrites index.json exactly once per chunk, so 30k deletes go
    // from "minutes" to "a few seconds."
    try {
      const { deleteAnnotateItemsBulk } = await import("../../api");
      const { deleted, not_found } = await deleteAnnotateItemsBulk(
        projectId,
        ids,
        (done, tot) => {
          setBusyMessage(
            t("imageList.deletingProgress")
              .replace("{done}", String(done))
              .replace("{total}", String(tot)),
          );
        },
      );
      // We can't know per-id which of the N requested IDs landed in the
      // bulk "not found" bucket, so if the server says anything was
      // deleted we optimistically treat every ID as removed. Stragglers
      // come back on the next /datasets/annotate refresh anyway.
      if (deleted > 0 || not_found === ids.length) {
        ids.forEach((id) => removed.add(id));
      }
      failed = ids.length - removed.size;
    } catch (err) {
      failed = ids.length;
      setStatus(
        t("imageList.uploadFailed").replace(
          "{msg}",
          (err as Error).message,
        ),
      );
    }
    setBusyMessage(null);
    if (removed.size > 0) {
      setImages((prev) => prev.filter((item) => !removed.has(item.id)));
      removed.forEach((id) => {
        cacheRef.current.delete(id);
        maskLoadPromisesRef.current.delete(id);
      });
      setSelectedIds((prev) => {
        const next = new Set(prev);
        removed.forEach((id) => next.delete(id));
        return next;
      });
      if (activeImageId && removed.has(activeImageId)) {
        setActiveImageId(null);
        setImage("", 0, 0);
        setMask(new Uint8Array(), 0, 0);
        setHistory([], []);
        setView(1, 0, 0);
      }
    }
    if (failed > 0) {
      setStatus(t("imageList.deleteResult").replace("{removed}", String(removed.size)).replace("{failed}", String(failed)));
    } else {
      setStatus(t("imageList.deleted").replace("{count}", String(removed.size)));
    }
  }

  async function handleExportAnnotations() {
    if (!projectId) return;
    try {
      const res = await exportAnnotateAnnotations(projectId);
      const count = res.annotation_count ?? 0;
      setStatus(`Annotations exported (${count}).`);
    } catch (err) {
      setStatus(`Export failed: ${(err as Error).message}`);
    }
  }

  function exportCsv() {
    const rows = ["id,name,set,width,height"];
    images.forEach((item) => {
      rows.push(
        `${item.id},${item.name},${item.set},${item.width},${item.height}`
      );
    });
    const blob = new Blob([rows.join("\n")], {
      type: "text/csv;charset=utf-8",
    });
    downloadBlob(blob, "dataset.csv");
  }

  async function handleImageBatch(files: FileList | File[]) {
    if (!projectId) return;
    const batchProjectId = projectId; // snapshot at call time
    const isStale = () => projectRef.current !== batchProjectId;

    const allFiles = Array.from(files);
    const zipFiles = allFiles.filter((f) => /\.zip$/i.test(f.name));
    const videoFiles = allFiles.filter((f) =>
      /\.(mp4|avi|mov|mkv|wmv|flv|webm|m4v|mpg|mpeg)$/i.test(f.name),
    );
    const imageFiles = allFiles.filter((file) => {
      if (/\.zip$/i.test(file.name)) return false;
      if (/\.(mp4|avi|mov|mkv|wmv|flv|webm|m4v|mpg|mpeg)$/i.test(file.name)) return false;
      if (file.type.startsWith("image/")) return true;
      return /\.(png|jpe?g|bmp|tiff?|webp|gif)$/i.test(file.name);
    });
    if (zipFiles.length === 0 && imageFiles.length === 0 && videoFiles.length === 0) {
      setStatus(t("imageList.dropHint"));
      return;
    }
    try {
      // Import ZIP files
      for (const zf of zipFiles) {
        if (isStale()) return;
        setBusyMessage(`ZIP import... ${zf.name}`);
        const result = await importAnnotateZip(batchProjectId, zf);
        if (isStale()) return;
        const maskPart = result.mask_count ? t("imageList.importMaskPart").replace("{maskCount}", String(result.mask_count)) : "";
        setStatus(t("imageList.importResult").replace("{imgCount}", String(result.image_count)).replace("{maskPart}", maskPart));
      }
      // Extract frames from video files
      for (const vf of videoFiles) {
        if (isStale()) return;
        const intervalStr = window.prompt(
          t("annotate.video.extractPrompt").replace("{name}", vf.name),
          "30",
        );
        if (intervalStr === null) continue; // cancelled
        const interval = Math.max(1, Math.min(3600, parseInt(intervalStr, 10) || 30));
        setBusyMessage(t("annotate.video.extracting").replace("{name}", vf.name).replace("{interval}", String(interval)));
        const result = await uploadVideoFrames(batchProjectId, vf, interval, (msg) => {
          if (!isStale()) setBusyMessage(msg);
        }, t("annotate.video.uploading"));
        if (isStale()) return;
        setStatus(t("annotate.video.extracted").replace("{name}", vf.name).replace("{count}", String(result.frame_count)));
      }

      // Upload regular images (convert non-PNG to PNG first, with progress)
      if (imageFiles.length > 0) {
        // Check for duplicate names
        const existingNames = new Set(images.map((img) => img.name?.toLowerCase() ?? ""));
        const duplicates = imageFiles.filter((f) => existingNames.has(f.name.toLowerCase()));
        let filesToUpload = imageFiles;

        if (duplicates.length > 0) {
          const dupNames = duplicates.slice(0, 5).map((f) => f.name).join(", ");
          const more = duplicates.length > 5 ? `... +${duplicates.length - 5}` : "";
          const choice = await new Promise<"skip" | "overwrite" | "both" | "cancel">((resolve) => {
            const msg = t("imageList.dupFound").replace("{count}", String(duplicates.length)).replace("{names}", dupNames + more);
            const skip = window.confirm(`${msg}\n\n${t("imageList.dupSkipPrompt")}`);
            if (skip) { resolve("skip"); return; }
            const overwrite = window.confirm(t("imageList.dupOverwritePrompt"));
            if (overwrite) { resolve("overwrite"); return; }
            resolve("both");
          });

          if (choice === "cancel") return;
          if (choice === "skip") {
            filesToUpload = imageFiles.filter((f) => !existingNames.has(f.name.toLowerCase()));
            if (filesToUpload.length === 0) {
              setStatus(t("imageList.dupSkipped").replace("{count}", String(duplicates.length)));
              return;
            }
          } else if (choice === "overwrite") {
            // Delete existing duplicates first
            const dupIds = images
              .filter((img) => duplicates.some((f) => f.name.toLowerCase() === (img.name?.toLowerCase() ?? "")))
              .map((img) => img.id);
            if (dupIds.length > 0) {
              const { deleteAnnotateItem } = await import("../../api");
              for (const id of dupIds) {
                await deleteAnnotateItem(batchProjectId, id);
              }
            }
          }
          // "both" = keep all, upload as-is (default behavior)
        }

        const total = filesToUpload.length;
        // Separate files that need conversion from those sent as-is
        const needsConvert = filesToUpload.filter((f) => f.type !== "image/png" && !/\.tiff?$/i.test(f.name));
        const passthrough = filesToUpload.filter((f) => f.type === "image/png" || /\.tiff?$/i.test(f.name));
        // Convert non-PNG/non-TIFF files with progress
        const pngFiles: File[] = [...passthrough];
        if (needsConvert.length > 0) {
          for (let i = 0; i < needsConvert.length; i++) {
            pngFiles.push(await convertToPng(needsConvert[i]!));
            if (!isStale()) setBusyMessage(t("imageList.convertingProgress").replace("{done}", String(i + 1)).replace("{total}", String(needsConvert.length)));
          }
        }
        if (isStale()) return;
        // Upload all files
        setBusyMessage(t("imageList.uploadingProgress").replace("{done}", "0").replace("{total}", String(total)));
        await uploadAnnotateImages(batchProjectId, pngFiles, (done, tot) => {
          if (!isStale()) setBusyMessage(t("imageList.uploadingProgress").replace("{done}", String(done)).replace("{total}", String(tot)));
        });
      }
      if (isStale()) return;
      setBusyMessage(t("imageList.loading"));
      await loadAnnotateItems(batchProjectId);
    } catch (err) {
      if (isStale()) return;
      setStatus(t("imageList.uploadFailed").replace("{msg}", (err as Error).message));
    } finally {
      if (!isStale()) setBusyMessage(null);
    }
  }

  async function handlePrepareDataset() {
    if (!projectId) return;
    try {
      const res = await prepareAnnotateDataset(projectId);
      if (res && res.report) {
        setPrepareReport({
          train_count: res.report.train_count ?? 0,
          val_count: res.report.val_count ?? 0,
          with_mask: res.report.with_mask ?? 0,
          auto_val_from_train_count: res.report.auto_val_from_train_count ?? 0,
        });
      }
      setStatus("Prepared dataset.");
    } catch (err) {
      setStatus(`Prepare failed: ${(err as Error).message}`);
    }
  }

  function handleListDragEnter(event: React.DragEvent<HTMLDivElement>) {
    if (!Array.from(event.dataTransfer.types).includes("Files")) return;
    event.preventDefault();
    event.stopPropagation();
    if (!projectId) return;
    listDragDepthRef.current += 1;
    setIsListDragActive(true);
  }

  function handleListDragOver(event: React.DragEvent<HTMLDivElement>) {
    if (!Array.from(event.dataTransfer.types).includes("Files")) return;
    event.preventDefault();
    event.stopPropagation();
    if (!projectId) return;
    event.dataTransfer.dropEffect = "copy";
  }

  function handleListDragLeave(event: React.DragEvent<HTMLDivElement>) {
    if (!Array.from(event.dataTransfer.types).includes("Files")) return;
    event.preventDefault();
    event.stopPropagation();
    listDragDepthRef.current = Math.max(0, listDragDepthRef.current - 1);
    if (listDragDepthRef.current === 0) {
      setIsListDragActive(false);
    }
  }

  function handleListDrop(event: React.DragEvent<HTMLDivElement>) {
    if (!Array.from(event.dataTransfer.types).includes("Files")) return;
    event.preventDefault();
    event.stopPropagation();
    listDragDepthRef.current = 0;
    setIsListDragActive(false);
    if (!projectId) {
      setStatus(t("imageList.selectProject"));
      return;
    }
    if (event.dataTransfer.files.length > 0) {
      void handleImageBatch(event.dataTransfer.files);
    }
  }

  return {
    // state
    images,
    setImages,
    filteredImages,
    selectedIds,
    setSelectedIds,
    isListDragActive,
    setIsListDragActive,
    listDragDepthRef,
    lastSelectedIndexRef,
    datasetStats,
    setDatasetStats,
    prepareReport,
    setPrepareReport,
    applyImageAnnotationSummary,
    // functions
    loadAnnotateItems,
    selectImage,
    handleSelectClick,
    handleArrowNav,
    selectAllFiltered,
    handleDeleteSelected,
    handleExportAnnotations,
    exportCsv,
    handleImageBatch,
    handlePrepareDataset,
    preloadAdjacentImages,
    storeImageIdRef,
    // drag handlers
    handleListDragEnter,
    handleListDragOver,
    handleListDragLeave,
    handleListDrop,
  };
}
