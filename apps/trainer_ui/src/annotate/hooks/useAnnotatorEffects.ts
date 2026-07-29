// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect } from "react";
import { useMaskStore, useTrainingStore } from "../../store";
import type { ClassItem } from "../../store";
import type { ImageItem, CacheEntry } from "../annotatorTypes";
import { saveClasses } from "../../api";

type EffectDeps = {
  projectId: string | null;
  projectRef: React.MutableRefObject<string | null>;
  cacheRef: React.MutableRefObject<Map<string, CacheEntry>>;
  maskLoadPromisesRef: React.MutableRefObject<Map<string, Promise<CacheEntry>>>;
  prefetchEpochRef: React.MutableRefObject<number>;
  pendingSwitchRef: React.MutableRefObject<string | null>;
  loadingTargetRef: React.MutableRefObject<string | null>;
  autoSaveTimerRef: React.MutableRefObject<number | null>;
  classSaveTimerRef: React.MutableRefObject<number | null>;
  classIdCounterRef: React.MutableRefObject<number>;
  classesDraftRef: React.MutableRefObject<ClassItem[]>;
  activeImageId: string | null;
  active?: boolean;
  saveRef?: React.MutableRefObject<(() => Promise<void>) | null>;
  // Canvas / rendering
  imageCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  overlayCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  imageUrl: string | null;
  width: number;
  height: number;
  maskVersion: number;
  lut: Uint8ClampedArray;
  scale: number;
  offsetX: number;
  offsetY: number;
  // State
  measureStart: [number, number] | null;
  measureEnd: [number, number] | null;
  tool: string;
  brushSize: number;
  classesDraft: ClassItem[];
  assistPreview: Uint8Array | null;
  recipePreview: Uint8Array | null;
  filteredImages: ImageItem[];
  // Setters
  setImages: React.Dispatch<React.SetStateAction<ImageItem[]>>;
  setActiveImageId: React.Dispatch<React.SetStateAction<string | null>>;
  setSelectedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  setAssistPreview: React.Dispatch<React.SetStateAction<Uint8Array | null>>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setActiveRecipe: React.Dispatch<React.SetStateAction<any>>;
  setRecipePreview: React.Dispatch<React.SetStateAction<Uint8Array | null>>;
  setClassesDraft: React.Dispatch<React.SetStateAction<ClassItem[]>>;
  setMeasureStart: React.Dispatch<React.SetStateAction<[number, number] | null>>;
  setMeasureEnd: React.Dispatch<React.SetStateAction<[number, number] | null>>;
  setBusyMessage: React.Dispatch<React.SetStateAction<string | null>>;
  // Operations
  drawOverlay: () => void;
  scheduleDrawOverlay: () => void;
  handleFit: () => void;
  renderUi: (pos: [number, number] | null) => void;
  saveMask: (imageId: string, mask: Uint8Array, w: number, h: number, pid?: string | null, touched?: Uint8Array | null) => Promise<void>;
  flushDirtyMasks: () => Promise<void>;
  loadClasses: (id: string) => void;
  loadAnnotateItems: (id: string, force?: boolean, sync?: boolean) => Promise<void>;
  loadActiveRecipe: () => void;
  selectImage: (item: ImageItem) => void;
  getBitmapFromCache: (url: string) => ImageBitmap | null;
  putBitmapToCache: (url: string) => Promise<ImageBitmap>;
};

export function useAnnotatorEffects(deps: EffectDeps) {
  const setImage = useMaskStore((s) => s.setImage);
  const setMask = useMaskStore((s) => s.setMask);
  const setHistory = useMaskStore((s) => s.setHistory);
  const setView = useMaskStore((s) => s.setView);
  const setClasses = useMaskStore((s) => s.setClasses);
  const setActiveClass = useMaskStore((s) => s.setActiveClass);
  const _runs = useTrainingStore((s) => s.runs);

  // Paint image to canvas
  useEffect(() => {
    if (!deps.imageUrl || deps.width === 0 || deps.height === 0) return;
    const imageUrl = deps.imageUrl;
    let cancelled = false;

    // Do NOT clear image canvas here — keep old image visible until new bitmap
    // is ready.  Clearing immediately causes black screen when switching rapidly.

    function paintToCanvas(source: ImageBitmap | HTMLImageElement) {
      if (cancelled) return;
      const canvas = deps.imageCanvasRef.current;
      if (!canvas) return;
      // Assigning to canvas.width/.height ALWAYS wipes the canvas, even when
      // the value is unchanged. Skip the assignment when dimensions already
      // match so the old image stays painted until drawImage overwrites it
      // in the same frame — eliminates the black flash on rapid image-list
      // navigation through a folder of same-size images.
      if (canvas.width !== deps.width || canvas.height !== deps.height) {
        canvas.width = deps.width;
        canvas.height = deps.height;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(source, 0, 0, deps.width, deps.height);
      deps.scheduleDrawOverlay();
      requestAnimationFrame(() => {
        if (cancelled) return;
        deps.handleFit();
        // Re-draw overlay AFTER handleFit so the ROI covers the correct
        // viewport area.  Without this, the first drawOverlay uses stale
        // scale=1/offset=0 (from setImage reset) and only renders a tiny
        // top-left ROI, leaving markings in other areas invisible.
        deps.scheduleDrawOverlay();
        if (
          deps.activeImageId &&
          deps.loadingTargetRef.current === deps.activeImageId
        ) {
          deps.loadingTargetRef.current = null;
          deps.setBusyMessage(null);
        }
      });
    }
    const cached = deps.getBitmapFromCache(imageUrl);
    if (cached) {
      paintToCanvas(cached);
    } else if (typeof createImageBitmap !== "undefined") {
      deps.putBitmapToCache(imageUrl).then((bmp) => paintToCanvas(bmp)).catch((err) => {
        console.warn("[Annotator] ImageBitmap failed, falling back:", err);
        const img = new Image(); img.src = imageUrl;
        img.onload = () => paintToCanvas(img);
      });
    } else {
      const img = new Image(); img.src = imageUrl;
      img.onload = () => paintToCanvas(img);
    }
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.imageUrl, deps.width, deps.height]);

  // Container resize -> fit
  useEffect(() => {
    requestAnimationFrame(() => deps.handleFit());
    const el = deps.containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => deps.handleFit());
    observer.observe(el);
    return () => observer.disconnect();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.width, deps.height]);

  // Active tab -> fit
  useEffect(() => {
    if (deps.active) requestAnimationFrame(() => deps.handleFit());
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.active]);

  // Overlay update — triggered by visual dependencies only.
  // Mask content changes (brush, undo, bucket, etc.) already call
  // drawOverlay(dirty) or scheduleDrawOverlay() explicitly, so
  // maskVersion is intentionally excluded to avoid redundant full
  // redraws that cause flicker during brush strokes.
  // scale/offsetX/offsetY are included because the overlay Worker uses
  // ROI (visible region) — zooming changes the ROI, so the overlay
  // must be redrawn to cover the newly visible area. Without this,
  // assistPreview (SpotDetect etc.) disappears outside the old ROI.
  useEffect(() => {
    deps.scheduleDrawOverlay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.lut, deps.width, deps.height, deps.assistPreview, deps.recipePreview, deps.scale, deps.offsetX, deps.offsetY]);

  // Reset previews on image change
  useEffect(() => {
    deps.setAssistPreview(null);
    deps.setRecipePreview(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.activeImageId]);

  // renderUi on tool state change
  useEffect(() => {
    deps.renderUi(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.measureStart, deps.measureEnd, deps.tool, deps.brushSize, deps.width, deps.height, deps.scale, deps.offsetX, deps.offsetY]);

  // Cleanup autosave timer
  useEffect(() => {
    const timerRef = deps.autoSaveTimerRef;
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // projectId change -> reset
  useEffect(() => {
    const oldProjectId = deps.projectRef.current;
    // Collect dirty entries WITHOUT deep-copying large arrays.
    // We transfer ownership: these refs are valid because we clear the cache below
    // and no other code will mutate them after this point.
    const oldDirty: [string, { maskIndex: Uint8Array; touched: Uint8Array; width: number; height: number }][] = [];
    for (const [id, entry] of deps.cacheRef.current.entries()) {
      if (entry.dirty) oldDirty.push([id, { maskIndex: entry.maskIndex, touched: entry.touched, width: entry.width, height: entry.height }]);
    }
    deps.projectRef.current = deps.projectId;
    deps.prefetchEpochRef.current += 1;
    deps.cacheRef.current.clear();
    deps.maskLoadPromisesRef.current.clear();
    deps.pendingSwitchRef.current = null;
    deps.loadingTargetRef.current = null;
    if (deps.autoSaveTimerRef.current !== null) {
      window.clearTimeout(deps.autoSaveTimerRef.current);
      deps.autoSaveTimerRef.current = null;
    }
    if (deps.classSaveTimerRef.current !== null) {
      window.clearTimeout(deps.classSaveTimerRef.current);
      deps.classSaveTimerRef.current = null;
      // Flush pending class save for old project before switching
      if (oldProjectId) {
        const draft = deps.classesDraftRef.current;
        if (draft.length > 0) {
          saveClasses(oldProjectId, {
            version: 1, ignore_index: 255,
            classes: draft,
            next_class_id: deps.classIdCounterRef.current,
          }).catch((e: unknown) => console.warn("useAnnotatorEffects: auto-save classes failed:", e));
        }
      }
    }
    if (oldProjectId && oldDirty.length > 0) {
      (async () => {
        for (const [id, entry] of oldDirty) {
          try { await deps.saveMask(id, entry.maskIndex, entry.width, entry.height, oldProjectId, entry.touched); } catch { /* ignore */ }
        }
      })();
    }
    deps.setBusyMessage(null);
    deps.setImages(() => []);
    deps.setActiveImageId(null);
    deps.setSelectedIds(new Set());
    deps.setAssistPreview(null);
    deps.setActiveRecipe(null);
    deps.setRecipePreview(null);
    deps.setClassesDraft([]);
    deps.classesDraftRef.current = [];  // sync ref immediately (React state is async)
    setClasses([]);
    setActiveClass(0);
    deps.setMeasureStart(null);
    deps.setMeasureEnd(null);
    setImage("", 0, 0);
    setMask(new Uint8Array(), 0, 0);
    setHistory([], []);
    setView(1, 0, 0);
    if (!deps.projectId) return;
    if (deps.active) {
      deps.loadingTargetRef.current = "__dataset__";
      deps.setBusyMessage("Loading annotate workspace...");
    }
    deps.loadClasses(deps.projectId);
    // sync=true so a fresh project switch re-walks the disk —
    // catches masks added since this project was last opened (e.g. just-imported items).
    deps.loadAnnotateItems(deps.projectId, false, true);
    deps.loadActiveRecipe();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.projectId]);

  // NOTE: Intentionally no re-activation reload here. The projectId-change effect
  // (above) handles loading on project switch. Re-entering the Annotate tab
  // preserves the existing image list/selection instantly (stale-while-revalidate).
  // Explicit refresh is available via the UI toolbar if needed.

  // Sync classesDraft to store
  useEffect(() => {
    if (deps.classesDraft.length > 0) setClasses(deps.classesDraft);
  }, [deps.classesDraft, setClasses]);

  // Flush dirty masks AND pending class save on deactivate
  useEffect(() => {
    if (deps.active === false) {
      // Flush pending debounced class save immediately
      if (deps.classSaveTimerRef.current !== null) {
        window.clearTimeout(deps.classSaveTimerRef.current);
        deps.classSaveTimerRef.current = null;
        const pid = deps.projectRef.current;
        const draft = deps.classesDraftRef.current;
        if (pid && draft.length > 0) {
          saveClasses(pid, {
            version: 1, ignore_index: 255,
            classes: draft,
            next_class_id: deps.classIdCounterRef.current,
          }).catch((e: unknown) => console.warn("useAnnotatorEffects: auto-save classes failed:", e));
        }
      }
      void deps.flushDirtyMasks();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.active]);

  // Fit + redraw overlay on reactivate
  useEffect(() => {
    if (deps.active) requestAnimationFrame(() => { deps.handleFit(); deps.scheduleDrawOverlay(); });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.active, deps.width, deps.height]);

  // Register save function for parent
  useEffect(() => {
    if (!deps.saveRef) return;
    deps.saveRef.current = async () => {
      if (deps.activeImageId && deps.width > 0 && deps.height > 0) {
        const cached = deps.cacheRef.current.get(deps.activeImageId);
        if (cached) {
          const state = useMaskStore.getState();
          cached.maskIndex = new Uint8Array(state.maskIndex);
          cached.width = state.width;
          cached.height = state.height;
          // Only mark dirty if the user actually painted something (revision > 0
          // means it was previously saved, so always flush; otherwise check if
          // any non-zero pixel exists to avoid saving a blank mask on tab-switch).
          if (!cached.dirty) {
            const hasContent = cached.revision > 0 || cached.maskIndex.some(v => v !== 0);
            if (hasContent) cached.dirty = true;
          }
          deps.cacheRef.current.set(deps.activeImageId, cached);
        }
      }
      if (deps.classSaveTimerRef.current !== null) {
        window.clearTimeout(deps.classSaveTimerRef.current);
        deps.classSaveTimerRef.current = null;
      }
      const classSave = (async () => {
        const pid = deps.projectRef.current;
        const draft = deps.classesDraftRef.current;
        if (!pid || draft.length === 0) return;
        await saveClasses(pid, { version: 1, ignore_index: 255, classes: draft, next_class_id: deps.classIdCounterRef.current });
      })();
      await Promise.all([deps.flushDirtyMasks(), classSave]);
    };
    return () => { if (deps.saveRef) deps.saveRef.current = null; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deps.activeImageId, deps.width, deps.height, deps.projectId]);

  // Arrow key navigation is handled by useKeyboard (which calls handleArrowNav).
  // The ImageListPanel listbox onKeyDown calls event.preventDefault() so that
  // useKeyboard's guard (`if (event.defaultPrevented) return`) prevents double-firing
  // when the list has focus.  A separate handler here would be a third path and
  // cause the double-step bug.
}
