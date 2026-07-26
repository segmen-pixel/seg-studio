// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n";
import {
  saveClasses,
  fetchTileInfo,
  tilesDziUrl,
  maskTileBaseUrl as getMaskTileBaseUrl,
  type Project,
} from "../api";
import { useMaskStore, useTrainingStore, type ClassItem } from "../store";
import type { ImageItem, CacheEntry } from "./annotatorTypes";
import type { SamModelId } from "./annotatorContext";
import { buildLut, downloadBlob } from "./imageProcessing";

import { ImageListPanel } from "./components/ImageListPanel";
import { CanvasArea } from "./components/CanvasArea";
import { ClassPanel } from "./components/ClassPanel";
import { AiAssistPanel } from "./components/AiAssistPanel";

import { useCanvasRendering } from "./hooks/useCanvasRendering";
import { useViewport } from "./hooks/useViewport";
import { useMaskIO } from "./hooks/useMaskIO";
import { useClassManager } from "./hooks/useClassManager";
import { useEditHistory } from "./hooks/useEditHistory";
import { useImageList } from "./hooks/useImageList";
import { usePixelStats, type RegionLabel } from "./hooks/usePixelStats";
import { usePerImageClassPresence } from "./hooks/usePerImageClassPresence";
import { useAiAssist } from "./hooks/useAiAssist";
import { useRecipe } from "./hooks/useRecipe";
import { useDrawingEvents } from "./hooks/useDrawingEvents";
import { useKeyboard } from "./hooks/useKeyboard";
import { useAnnotatorEffects } from "./hooks/useAnnotatorEffects";
import type { SamRefValue, SpotDetectRefValue, WandRefValue, SuperpixelRefValue, CrackTraceRefValue, MoveRefValue } from "./hooks/useDrawingEvents";
import { initCrackTrace } from "./hooks/toolActions";

export default React.memo(function Annotator({
  projectId, projects: _projects, onProjectChange: _onProjectChange, active, saveRef, previewStyle, setPreviewStyle: _setPreviewStyle, showToast, descMode, pendingImageId, onPendingImageHandled,
}: {
  projectId: string | null;
  projects: Project[];
  onProjectChange: (id: string) => void;
  active?: boolean;
  saveRef?: React.MutableRefObject<(() => Promise<void>) | null>;
  previewStyle: number;
  setPreviewStyle: (v: number) => void;
  showToast?: (msg: string) => void;
  descMode?: boolean;
  pendingImageId?: string | null;
  onPendingImageHandled?: () => void;
}) {
  const { t } = useI18n();
  // Canvas refs
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const uiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Tool refs
  const strokeAccRef = useRef<Map<number, number> | null>(null);
  const wandRef = useRef<WandRefValue>(null);
  const samRef = useRef<SamRefValue>(null);
  const spotDetectRef = useRef<SpotDetectRefValue>(null);
  const superpixelRef = useRef<SuperpixelRefValue>(null);
  const crackTraceRef = useRef<CrackTraceRefValue>(null);
  const moveRef = useRef<MoveRefValue>(null);

  // Tiled viewer (large images)
  const [dziUrl, setDziUrl] = useState<string | null>(null);
  const [maskTileUrl, setMaskTileUrl] = useState<string | null>(null);

  // Persistence refs
  const cacheRef = useRef<Map<string, CacheEntry>>(new Map());
  const projectRef = useRef<string | null>(projectId);
  const maskLoadPromisesRef = useRef<Map<string, Promise<CacheEntry>>>(new Map());
  const prefetchEpochRef = useRef(0);
  const pendingSwitchRef = useRef<string | null>(null);
  const loadingTargetRef = useRef<string | null>(null);
  const autoSaveTimerRef = useRef<number | null>(null);
  const classSaveTimerRef = useRef<number | null>(null);
  const skipClassAutoSaveRef = useRef(false);
  const classIdCounterRef = useRef(0);

  // UI state
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const setStatus = useMemo(() => showToast ?? (() => {}), [showToast]);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  // Warn user before closing/refreshing while an import is in progress
  useEffect(() => {
    if (!busyMessage) return;
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [busyMessage]);
  const [prefetchMessage, setPrefetchMessage] = useState<string | null>(null);
  const [overlayAlpha, setOverlayAlpha] = useState(140);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const prevOverlayAlphaRef = useRef(140);
  const [imageBrightness, setImageBrightness] = useState(100);
  const [imageContrast, setImageContrast] = useState(100);
  const [brushSize, setBrushSize] = useState(18);
  const [measureStart, setMeasureStart] = useState<[number, number] | null>(null);
  const [measureEnd, setMeasureEnd] = useState<[number, number] | null>(null);
  const [spotSensitivity, setSpotSensitivity] = useState(15);
  const [spotCount, setSpotCount] = useState(0);
  const [spotPhase, setSpotPhase] = useState<"idle" | "sample" | "detect">("idle");
  const [colorTolerance, setColorTolerance] = useState(15);
  const [samModel, setSamModel] = useState<SamModelId>("mobile_sam");
  const blinkPhaseRef = useRef(0);
  const [_blinkPhase, _setBlinkPhase] = useState(0);
  const [nSegments, setNSegments] = useState(500);
  const [crackSensitivity, setCrackSensitivity] = useState(25);
  const [crackWidth, setCrackWidth] = useState(0);
  const samBoxDraftRef = useRef<{ start: [number, number]; end: [number, number] } | null>(null);
  const [activeImageId, setActiveImageId] = useState<string | null>(null);
  const activeImageIdRef = useRef(activeImageId);
  activeImageIdRef.current = activeImageId;

  // Clear AI assist refs on image change
  useEffect(() => {
    (samRef as React.MutableRefObject<SamRefValue>).current = null;
    (spotDetectRef as React.MutableRefObject<SpotDetectRefValue>).current = null;
    (superpixelRef as React.MutableRefObject<SuperpixelRefValue>).current = null;
    (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = null;
  }, [activeImageId]);

  // Assist state
  const [assistPreview, setAssistPreview] = useState<Uint8Array | null>(null);
  const [gpuBusy, setGpuBusy] = useState(false);
  const [gpuBusyReason, setGpuBusyReason] = useState<string | null>(null);
  // Poll GPU device busy state — block GPU tools only when ALL CUDA devices are busy
  useEffect(() => {
    const check = async () => {
      try {
        const { fetchTorchDevices } = await import("../api/hardware");
        const data = await fetchTorchDevices();
        const devices = data?.devices ?? [];
        // Block only when every CUDA device is occupied (FIFO: SAM uses whichever GPU is free)
        const cudaDevices = devices.filter((d: any) => d.kind === "cuda");
        const allCudaBusy = cudaDevices.length > 0 && cudaDevices.every((d: any) => d.busy);
        setGpuBusy(allCudaBusy);
        if (allCudaBusy) {
          const firstBusy = cudaDevices.find((d: any) => d.busy);
          setGpuBusyReason(firstBusy?.busy_owner_kind === "training" ? "training" : "inference");
        } else {
          setGpuBusyReason(null);
        }
      } catch { /* ignore */ }
    };
    void check();
    const iv = window.setInterval(check, 5000);
    return () => window.clearInterval(iv);
  }, []);

  // Classes state
  const [classesDraft, setClassesDraft] = useState<ClassItem[]>([]);
  const classesDraftRef = useRef<ClassItem[]>(classesDraft);
  classesDraftRef.current = classesDraft;
  const setRegionLabelsRef = useRef<(labels: RegionLabel[]) => void>(() => {});

  // 1. Zustand stores (individual selectors to avoid full-store re-renders)
  const imageUrl = useMaskStore((s) => s.imageUrl);
  const width = useMaskStore((s) => s.width);
  const height = useMaskStore((s) => s.height);
  const maskIndex = useMaskStore((s) => s.maskIndex);
  const maskVersion = useMaskStore((s) => s.maskVersion);
  const classes = useMaskStore((s) => s.classes);
  const activeClassId = useMaskStore((s) => s.activeClassId);
  const tool = useMaskStore((s) => s.tool);
  const scale = useMaskStore((s) => s.scale);
  const offsetX = useMaskStore((s) => s.offsetX);
  const offsetY = useMaskStore((s) => s.offsetY);
  const _setImage = useMaskStore((s) => s.setImage);
  const setMask = useMaskStore((s) => s.setMask);
  const setTool = useMaskStore((s) => s.setTool);
  const setActiveClass = useMaskStore((s) => s.setActiveClass);
  const _setHistory = useMaskStore((s) => s.setHistory);
  const _setClasses = useMaskStore((s) => s.setClasses);
  const setView = useMaskStore((s) => s.setView);
  const _runs = useTrainingStore((s) => s.runs);

  useEffect(() => {
    if (!busyMessage || !active || !activeImageId || width === 0 || height === 0) return;
    const timer = window.setTimeout(() => {
      if (loadingTargetRef.current === activeImageId) {
        loadingTargetRef.current = null;
        setBusyMessage(null);
      }
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [busyMessage, active, activeImageId, width, height]);

  // 2. LUT memo
  const classesForLut = classesDraft.length ? classesDraft : classes;
  const lut = useMemo(() => buildLut(classesForLut, overlayAlpha), [classesForLut, overlayAlpha]);

  // 4. useCanvasRendering
  const { drawOverlay, scheduleDrawOverlay, invalidateOverlay, renderUi, scheduleRenderUi, getBitmapFromCache, putBitmapToCache, terminateWorker } =
    useCanvasRendering({
      containerRef, overlayCanvasRef, uiCanvasRef,
      getState: () => ({
        width, height, maskIndex, lut, scale, offsetX, offsetY,
        assistPreview, recipePreview,
        measureStart, measureEnd, tool, brushSize,
        wandRef: wandRef.current ? { tolerance: wandRef.current.tolerance } : null,
        samRef: samRef.current ? { points: samRef.current.points, labels: samRef.current.labels, box: samRef.current.box } : null,
        samBoxDraft: samBoxDraftRef.current,
        samMode: tool === "sambox" ? "box" : "point",
        spotDetectRef: spotDetectRef.current, spotCount,
        superpixelBoundary: superpixelRef.current?.boundaryMask ?? null,
        previewStyle, blinkPhase: blinkPhaseRef.current,
      }),
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => terminateWorker, []);

  // Blink interval for preview style
  useEffect(() => {
    if (previewStyle !== 1) return;
    const id = setInterval(() => {
      blinkPhaseRef.current = blinkPhaseRef.current ? 0 : 1;
      scheduleDrawOverlay();
    }, 500);
    return () => clearInterval(id);
  }, [previewStyle, scheduleDrawOverlay]);

  // 5. useViewport
  const { handleFit, handleZoom: _handleZoom, isPanning, setIsPanning, spacePressed, panStartRef } =
    useViewport(containerRef, setBrushSize, (cx, cy) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const { scale: s, offsetX: ox, offsetY: oy } = useMaskStore.getState();
      const ix = Math.floor((cx - rect.left - ox) / s);
      const iy = Math.floor((cy - rect.top - oy) / s);
      if (ix >= 0 && iy >= 0 && ix < width && iy < height) {
        scheduleRenderUi([ix, iy]);
      }
    });

  // 6. useMaskIO
  const { loadMaskFor: _loadMaskFor, saveMask, autoSave, flushDirtyMasks, getOrLoadMaskEntry, prefetchProjectMasks, scheduleAutosave: _scheduleAutosave, markDirty, markTouched } =
    useMaskIO(projectId, projectRef, cacheRef, maskLoadPromisesRef, prefetchEpochRef, autoSaveTimerRef, activeImageId, () => imageListHook.filteredImages);

  // 7. useClassManager
  const { loadClasses, saveClassList: _saveClassList, addClass, updateClass, deactivateClass: _deactivateClass, handleDeleteClass } =
    useClassManager(
      projectId, projectRef, classesDraft, setClassesDraft, classesDraftRef, classIdCounterRef,
      classSaveTimerRef, skipClassAutoSaveRef, cacheRef, activeImageId, autoSaveTimerRef,
      saveMask, scheduleDrawOverlay,
      (labels: RegionLabel[]) => setRegionLabelsRef.current(labels), setStatus
    );

  // 8. useEditHistory
  const editOpsRef = useRef({ drawOverlay, scheduleDrawOverlay, markDirty, markTouched });
  editOpsRef.current = { drawOverlay, scheduleDrawOverlay, markDirty, markTouched };
  const { handleUndo, handleRedo, handleClear, handleCut: _handleCut, handlePaste: _handlePaste, clearClassById, clearActiveClass: _clearActiveClass } =
    useEditHistory(() => editOpsRef.current, activeImageId, cacheRef);

  // 9. useImageList
  const imageListOpsRef = useRef({
    saveMask, autoSave, handleFit,
    selectImage: undefined as ((item: ImageItem) => void) | undefined,
    putBitmapToCache, getBitmapFromCache, invalidateOverlay,
  });
  imageListOpsRef.current = { saveMask, autoSave, handleFit, selectImage: undefined, putBitmapToCache, getBitmapFromCache, invalidateOverlay };
  const imageListHook = useImageList(
    projectId, projectRef, cacheRef, maskLoadPromisesRef, prefetchEpochRef,
    loadingTargetRef, !!active,
    activeImageId, setActiveImageId, activeImageIdRef, autoSaveTimerRef, pendingSwitchRef,
    () => imageListOpsRef.current, getOrLoadMaskEntry, prefetchProjectMasks, setStatus, setBusyMessage, setPrefetchMessage
  );
  const {
    images, setImages, filteredImages,
    selectedIds, setSelectedIds, isListDragActive,
    datasetStats, prepareReport, setPrepareReport: _setPrepareReport,
    applyImageAnnotationSummary,
    loadAnnotateItems, selectImage,
    handleSelectClick, handleArrowNav, selectAllFiltered,
    handleDeleteSelected,
    exportCsv: _exportCsv,
    handleImageBatch, handlePrepareDataset: _handlePrepareDataset,
    handleListDragEnter, handleListDragOver, handleListDragLeave, handleListDrop,
  } = imageListHook;

  // Jump to image from Results tab
  useEffect(() => {
    if (!pendingImageId || !images.length) return;
    const target = images.find((img) => img.id === pendingImageId);
    if (target) {
      selectImage(target);
      onPendingImageHandled?.();
    }
  }, [pendingImageId, images]);

  // When the active image moves, top up the windowed mask prefetch so the
  // next neighbourhood (±window radius) is cached before the user scrolls
  // into it. Items already cached are skipped inside prefetchProjectMasks.
  useEffect(() => {
    if (!projectId || !activeImageId || images.length === 0) return;
    void prefetchProjectMasks(projectId, images, undefined, { activeId: activeImageId });
  }, [projectId, activeImageId, images.length, prefetchProjectMasks, images]);

  // Check if active image has DZI tiles (large image support)
  useEffect(() => {
    if (!projectId || !activeImageId) { setDziUrl(null); setMaskTileUrl(null); return; }
    let cancelled = false;
    fetchTileInfo(projectId, activeImageId).then((info) => {
      if (cancelled) return;
      if (info.tiled) {
        setDziUrl(tilesDziUrl(projectId, activeImageId));
        setMaskTileUrl(getMaskTileBaseUrl(projectId, activeImageId));
      } else {
        setDziUrl(null);
        setMaskTileUrl(null);
      }
    }).catch(() => { if (!cancelled) { setDziUrl(null); setMaskTileUrl(null); } });
    return () => { cancelled = true; };
  }, [projectId, activeImageId]);

  // 10. usePixelStats
  const { classPixelStats, regionLabels, setRegionLabels } =
    usePixelStats(images, classesDraft, activeImageId, cacheRef, maskVersion);
  setRegionLabelsRef.current = setRegionLabels;

  // 10b. usePerImageClassPresence
  const perImageClasses = usePerImageClassPresence(images, classesDraft, activeImageId, cacheRef, maskVersion, projectId);

  // Track previous activeImageId so we can flush its summary on switch
  const prevActiveImageIdForSummaryRef = useRef<string | null>(null);

  useEffect(() => {
    // When leaving an image, flush its annotation summary from the cache immediately
    // so dots and mask counts stay in sync without a full reload.
    const prevId = prevActiveImageIdForSummaryRef.current;
    if (prevId && prevId !== activeImageId) {
      const cached = cacheRef.current.get(prevId);
      if (cached && cached.maskIndex.length > 0) {
        const knownIds = new Set(
          classesDraft.filter((c) => c.id !== 0).map((c) => c.id)
        );
        const found = new Set<number>();
        if (knownIds.size > 0) {
          for (let i = 0; i < cached.maskIndex.length; i++) {
            const v = cached.maskIndex[i]!;
            if (v !== 0 && knownIds.has(v)) found.add(v);
          }
        }
        applyImageAnnotationSummary(
          prevId,
          found.size > 0 ? Array.from(found).sort((a, b) => a - b) : []
        );
      }
    }
    prevActiveImageIdForSummaryRef.current = activeImageId;

    if (!activeImageId) return;
    const timer = window.setTimeout(() => {
      const idx = useMaskStore.getState().maskIndex;
      if (!idx || idx.length === 0) {
        applyImageAnnotationSummary(activeImageId, []);
        return;
      }
      const knownIds = new Set(
        classesDraft.filter((c) => c.id !== 0).map((c) => c.id)
      );
      if (knownIds.size === 0) {
        applyImageAnnotationSummary(activeImageId, []);
        return;
      }
      const found = new Set<number>();
      for (let i = 0; i < idx.length; i++) {
        const value = idx[i]!;
        if (value !== 0 && knownIds.has(value)) found.add(value);
      }
      applyImageAnnotationSummary(
        activeImageId,
        Array.from(found).sort((a, b) => a - b)
      );
    }, 180);
    return () => window.clearTimeout(timer);
  }, [activeImageId, applyImageAnnotationSummary, classesDraft, maskVersion]);

  // 11. useAiAssist
  const aiOpsRef = useRef({ markDirty, markTouched, autoSave });
  aiOpsRef.current = { markDirty, markTouched, autoSave };
  const {
    handleSamConfirm, handleSamCancel, handleSpotConfirm, handleSpotCancel, handleSpotRunDetect,
    handleSuperpixelConfirm, handleSuperpixelCancel,
    handleCrackConfirm, handleCrackCancel,
  } = useAiAssist(
    projectId, activeImageId, activeImageIdRef, width, height,
    assistPreview, setAssistPreview,
    () => aiOpsRef.current, samRef, samBoxDraftRef, spotDetectRef, superpixelRef, crackTraceRef, spotCount, setSpotCount, setStatus,
    spotSensitivity, colorTolerance, setSpotSensitivity, setSpotPhase,
  );

  // 12. useRecipe
  const recipeOpsRef = useRef({ markDirty, markTouched, autoSave });
  recipeOpsRef.current = { markDirty, markTouched, autoSave };
  const {
    activeRecipe, setActiveRecipe, recipePreview, setRecipePreview,
    isRecipeRunning, recipeInputRef,
    handleRecipeImport, loadActiveRecipe, handleRecipePreview, handleAutoLabelPreview, handleRecipeConfirm, handleRecipeCancel, handleRecipeApplyAll,
  } = useRecipe(projectId, activeImageId, activeImageIdRef, width, height, () => recipeOpsRef.current, setStatus, setBusyMessage, setImages);

  // 13. useDrawingEvents
  const drawingOpsRef = useRef({
    drawOverlay, scheduleDrawOverlay, markDirty, markTouched, renderUi, scheduleRenderUi,
    handleSamConfirm, handleSamCancel, handleSpotConfirm, handleSpotCancel,
    handleSuperpixelConfirm, handleSuperpixelCancel,
    handleCrackConfirm, handleCrackCancel,
  });
  drawingOpsRef.current = {
    drawOverlay, scheduleDrawOverlay, markDirty, markTouched, renderUi, scheduleRenderUi,
    handleSamConfirm, handleSamCancel, handleSpotConfirm, handleSpotCancel,
    handleSuperpixelConfirm, handleSuperpixelCancel,
    handleCrackConfirm, handleCrackCancel,
  };
  const handleSuperpixelRecompute = () => {
    superpixelRef.current = null;
    setAssistPreview(null);
    setStatus("Superpixel: click to recompute");
  };
  const handleCrackRecompute = () => {
    if (!projectId || !activeImageId || width <= 0 || height <= 0) return;
    const ctRef = crackTraceRef.current;
    if (ctRef) ctRef.loading = true;
    setStatus("Crack Trace: recomputing...");
    initCrackTrace(projectId, activeImageId, width, height, crackSensitivity, crackWidth, activeClassId, setStatus)
      .then((result) => {
        if (result) {
          (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = result.refValue;
          setAssistPreview(result.preview);
          drawOverlay();
        } else {
          (crackTraceRef as React.MutableRefObject<CrackTraceRefValue>).current = null;
          setAssistPreview(null);
        }
      });
  };
  const { handlePointerDown, handlePointerMove, handlePointerUp } =
    useDrawingEvents(
      () => drawingOpsRef.current, containerRef, imageCanvasRef,
      brushSize, strokeAccRef, wandRef, samRef, spotDetectRef,
      spotSensitivity, setSpotCount, setAssistPreview, setRecipePreview, setStatus, samModel,
      projectId, activeImageId, spacePressed, isPanning, setIsPanning, panStartRef,
      setMeasureStart, setMeasureEnd, measureStart, measureEnd,
      samBoxDraftRef,
      superpixelRef, nSegments,
      assistPreview,
      crackTraceRef, crackSensitivity, crackWidth,
      colorTolerance, setSpotPhase,
      moveRef,
    );
  // handleSaveAll
  async function handleSaveAll() {
    if (!projectId) return;
    if (activeImageId && width > 0 && height > 0) {
      const cached = cacheRef.current.get(activeImageId);
      if (cached) {
        const state = useMaskStore.getState();
        cached.maskIndex = new Uint8Array(state.maskIndex); cached.width = state.width;
        cached.height = state.height; cached.dirty = true;
        cacheRef.current.set(activeImageId, cached);
      }
    }
    if (classSaveTimerRef.current !== null) { window.clearTimeout(classSaveTimerRef.current); classSaveTimerRef.current = null; }
    const draft = classesDraftRef.current;
    const classSave = draft.length > 0
      ? saveClasses(projectId, { version: 1, ignore_index: 255, classes: draft, next_class_id: classIdCounterRef.current })
      : Promise.resolve();
    await Promise.all([flushDirtyMasks(), classSave]);
    setStatus("Saved.");
  }

  // 14. useKeyboard
  const handleMarkClean = useCallback(async () => {
    if (!projectId) return;
    const targetIds = selectedIds.size > 0 ? Array.from(selectedIds) : (activeImageId ? [activeImageId] : []);
    if (targetIds.length === 0) return;
    // Mirror handleUnmarkClean: only surface the busy overlay for multi-image
    // operations. Single-image mark-clean usually completes in well under
    // 100ms, so the overlay used to flash briefly and look like a glitch.
    const showOverlay = targetIds.length > 1;
    if (showOverlay) setBusyMessage(t("annotate.okBulk.marking").replace("{count}", String(targetIds.length)));
    const startedAt = performance.now();
    try {
      const { markImagesClean } = await import("../api/datasets");
      await markImagesClean(projectId, targetIds);
      if (activeImageId && targetIds.includes(activeImageId)) {
        const state = useMaskStore.getState();
        if (state.width > 0) {
          state.maskIndex.fill(0);
          useMaskStore.setState({ maskVersion: state.maskVersion + 1, undoStack: [], redoStack: [] });
        }
        const cached = cacheRef.current.get(activeImageId);
        if (cached) {
          cached.maskIndex.fill(0);
          cached.dirty = false;
          cached.touched.fill(1);
        }
      }
      await loadAnnotateItems(projectId);
      setStatus(t("annotate.okBulk.marked").replace("{count}", String(targetIds.length)));
    } catch (e) {
      setStatus(t("annotate.okBulk.markFailed").replace("{msg}", (e as Error).message));
    } finally {
      if (showOverlay) {
        // Guarantee a minimum visible time so the overlay does not flash
        // briefly even when the API + loadAnnotateItems finish fast.
        const elapsed = performance.now() - startedAt;
        const minMs = 700;
        if (elapsed < minMs) await new Promise((r) => setTimeout(r, minMs - elapsed));
        setBusyMessage(null);
      }
    }
  }, [projectId, selectedIds, activeImageId, loadAnnotateItems, setStatus, setBusyMessage, t]);

  // Batch variant of the per-class 消去 button: when a multi-selection is
  // active, clear the class from every selected image server-side (after a
  // confirm — this path has no undo, unlike the single-image clear).
  const handleClearClassSelected = useCallback(async (classId: number) => {
    if (!projectId || selectedIds.size === 0) return;
    const ids = Array.from(selectedIds);
    const cls = classesDraftRef.current.find((c) => c.id === classId);
    const clsName = cls?.name ?? `class${classId}`;
    const ok = window.confirm(
      t("classPanel.clearSelectedConfirm")
        .replace("{n}", String(ids.length))
        .replace("{cls}", clsName),
    );
    if (!ok) return;
    const showOverlay = ids.length > 1;
    if (showOverlay) {
      setBusyMessage(t("classPanel.clearSelectedBusy").replace("{n}", String(ids.length)));
    }
    try {
      const { clearClassFromImages } = await import("../api/datasets");
      const res = await clearClassFromImages(projectId, ids, classId);
      if (activeImageId && ids.includes(activeImageId)) {
        const state = useMaskStore.getState();
        if (state.width > 0) {
          const mi = state.maskIndex;
          for (let i = 0; i < mi.length; i += 1) if (mi[i] === classId) mi[i] = 0;
          useMaskStore.setState({ maskVersion: state.maskVersion + 1, undoStack: [], redoStack: [] });
        }
        const cached = cacheRef.current.get(activeImageId);
        if (cached) {
          const cm = cached.maskIndex;
          for (let i = 0; i < cm.length; i += 1) if (cm[i] === classId) cm[i] = 0;
          cached.dirty = false;
          cached.touched.fill(1);
        }
      }
      // Visited-but-inactive selected images may sit in the client mask
      // cache with the class still present — drop them so they reload
      // from the server instead of writing stale pixels back.
      for (const iid of ids) {
        if (iid !== activeImageId) cacheRef.current.delete(iid);
      }
      await loadAnnotateItems(projectId);
      setStatus(t("classPanel.clearSelectedDone")
        .replace("{cls}", clsName)
        .replace("{n}", String(res.updated)));
    } catch (e) {
      setStatus(t("classPanel.clearSelectedFailed").replace("{msg}", (e as Error).message));
    } finally {
      if (showOverlay) setBusyMessage(null);
    }
  }, [projectId, selectedIds, activeImageId, loadAnnotateItems, setStatus, setBusyMessage, t]);

  // Symmetric with handleMarkClean: if a specific imageId is passed (e.g.
  // from the per-row ✕ badge), unmark just that one. Otherwise fall back to
  // the current multi-selection, then to the active image — mirroring the
  // mark side so multi-selecting rows and clicking "OK解除" clears them all
  // at once.
  const handleUnmarkClean = useCallback(async (imageId?: string) => {
    if (!projectId) return;
    const targetIds = imageId
      ? [imageId]
      : selectedIds.size > 0
        ? Array.from(selectedIds)
        : (activeImageId ? [activeImageId] : []);
    if (targetIds.length === 0) return;
    const showOverlay = targetIds.length > 1;
    if (showOverlay) setBusyMessage(t("annotate.okBulk.clearing").replace("{count}", String(targetIds.length)));
    const startedAt = performance.now();
    try {
      const { unmarkImagesClean } = await import("../api/datasets");
      await unmarkImagesClean(projectId, targetIds);
      if (activeImageId && targetIds.includes(activeImageId)) {
        const cached = cacheRef.current.get(activeImageId);
        if (cached) { cached.dirty = false; }
      }
      await loadAnnotateItems(projectId);
      setStatus(
        targetIds.length > 1
          ? t("annotate.okBulk.cleared").replace("{count}", String(targetIds.length))
          : t("annotate.okBulk.clearedOne"),
      );
    } catch (e) {
      setStatus(t("annotate.okBulk.clearFailed").replace("{msg}", (e as Error).message));
    } finally {
      if (showOverlay) {
        const elapsed = performance.now() - startedAt;
        const minMs = 700;
        if (elapsed < minMs) await new Promise((r) => setTimeout(r, minMs - elapsed));
        setBusyMessage(null);
      }
    }
  }, [projectId, selectedIds, activeImageId, loadAnnotateItems, setStatus, setBusyMessage, t]);

  const kbOpsRef = useRef({
    handleUndo, handleRedo, handleClear, handleSamConfirm, handleSamCancel, handleSpotConfirm, handleSpotCancel, handleSuperpixelConfirm, handleSuperpixelCancel, handleCrackConfirm, handleCrackCancel, handleMarkClean, handleSaveAll, handleArrowNav,
  });
  kbOpsRef.current = {
    handleUndo, handleRedo, handleClear, handleSamConfirm, handleSamCancel, handleSpotConfirm, handleSpotCancel, handleSuperpixelConfirm, handleSuperpixelCancel, handleCrackConfirm, handleCrackCancel, handleMarkClean, handleSaveAll, handleArrowNav,
  };
  const gpuBusyRef = useRef(gpuBusy);
  gpuBusyRef.current = gpuBusy;
  useKeyboard(() => kbOpsRef.current, samRef, spotDetectRef, superpixelRef, crackTraceRef, classesDraftRef, setBrushSize, gpuBusyRef);

  // 15. useAnnotatorEffects
  useAnnotatorEffects({
    projectId, projectRef, cacheRef, maskLoadPromisesRef, prefetchEpochRef, pendingSwitchRef, loadingTargetRef,
    autoSaveTimerRef, classSaveTimerRef, classIdCounterRef, classesDraftRef,
    activeImageId, active, saveRef, imageCanvasRef, overlayCanvasRef, containerRef,
    imageUrl, width, height, maskVersion, lut, scale, offsetX, offsetY,
    measureStart, measureEnd, tool, brushSize, classesDraft,
    assistPreview, recipePreview, filteredImages,
    setImages, setActiveImageId, setSelectedIds,
    setAssistPreview, setActiveRecipe, setRecipePreview,
    setClassesDraft, setMeasureStart, setMeasureEnd, setBusyMessage,
    drawOverlay, scheduleDrawOverlay, handleFit, renderUi,
    saveMask, flushDirtyMasks, loadClasses, loadAnnotateItems, loadActiveRecipe, selectImage,
    getBitmapFromCache, putBitmapToCache,
  });

  // Computed
  const _activeImageItem = useMemo(
    () => images.find((item) => item.id === activeImageId) ?? null, [images, activeImageId]
  );
  const measureDistance = measureStart && measureEnd
    ? Math.hypot(measureEnd[0] - measureStart[0], measureEnd[1] - measureStart[1]).toFixed(1) : null;

  // Inline helpers
  function _handleMaskImport(file: File) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement("canvas");
      canvas.width = img.width; canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height).data;
      const mask = new Uint8Array(img.width * img.height);
      for (let i = 0; i < mask.length; i += 1) mask[i] = data[i * 4];
      setMask(mask, img.width, img.height);
      setStatus(`Mask imported ${file.name}`);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      console.warn(`Failed to load mask import image: ${file.name}`);
    };
    img.src = url;
  }

  function _exportMask() {
    if (width === 0 || height === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < maskIndex.length; i += 1) {
      const base = i * 4; const value = maskIndex[i];
      imageData.data[base] = value; imageData.data[base + 1] = value;
      imageData.data[base + 2] = value; imageData.data[base + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    canvas.toBlob((blob) => { if (blob) downloadBlob(blob, "mask.png"); });
  }

  function _exportComposite() {
    if (!imageUrl || width === 0 || height === 0) return;
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, width, height);
      ctx.drawImage(overlayCanvasRef.current!, 0, 0);
      canvas.toBlob((blob) => { if (blob) downloadBlob(blob, "overlay.png"); });
    };
  }

  // JSX
  return (
    <div
      className={`annotate-layout${descMode ? " desc-mode" : ""}${isListDragActive ? " drag-active" : ""}`}
      onDragEnter={handleListDragEnter}
      onDragOver={handleListDragOver}
      onDragLeave={handleListDragLeave}
      onDrop={handleListDrop}
    >
      <ImageListPanel
        filteredImages={filteredImages} activeImageId={activeImageId}
        selectedIds={selectedIds}
        isListDragActive={isListDragActive} datasetStats={datasetStats}
        classPixelStats={classPixelStats}
        perImageClasses={perImageClasses} classesDraft={classesDraft}
        prefetchMessage={prefetchMessage}
        projectId={projectId}
        handleImageBatch={handleImageBatch} selectAllFiltered={selectAllFiltered}
        handleDeleteSelected={handleDeleteSelected}
        handleSelectClick={handleSelectClick}
        onMoveSelection={handleArrowNav}
        onClearSelection={() => setSelectedIds(new Set())}
        onUnmarkClean={handleUnmarkClean}
        onAugmentComplete={() => {
          if (projectId) void imageListHook.loadAnnotateItems(projectId);
        }}
        handleListDragEnter={handleListDragEnter} handleListDragOver={handleListDragOver}
        handleListDragLeave={handleListDragLeave} handleListDrop={handleListDrop}
      />
      <CanvasArea
        containerRef={containerRef} imageCanvasRef={imageCanvasRef}
        overlayCanvasRef={overlayCanvasRef} uiCanvasRef={uiCanvasRef}
        handlePointerDown={handlePointerDown} handlePointerMove={handlePointerMove}
        handlePointerUp={handlePointerUp}
        tool={tool} setTool={setTool} brushSize={brushSize} setBrushSize={setBrushSize}
        width={width} height={height} scale={scale} offsetX={offsetX} offsetY={offsetY}
        isPanning={isPanning} spacePressed={spacePressed}
        imageBrightness={imageBrightness} setImageBrightness={setImageBrightness}
        imageContrast={imageContrast} setImageContrast={setImageContrast}
        overlayAlpha={overlayAlpha} setOverlayAlpha={setOverlayAlpha}
        setView={setView} regionLabels={regionLabels}
        measureDistance={measureDistance}
        setMeasureStart={setMeasureStart} setMeasureEnd={setMeasureEnd}
        handleSamCancel={handleSamCancel} handleSpotCancel={handleSpotCancel}
        handleCrackCancel={handleCrackCancel}
        samRefCurrent={samRef.current} spotDetectRefCurrent={spotDetectRef.current}
        crackTraceRefCurrent={crackTraceRef.current}
        overlayVisible={overlayVisible} setOverlayVisible={setOverlayVisible}
        prevOverlayAlphaRef={prevOverlayAlphaRef}
        handleUndo={handleUndo} handleRedo={handleRedo}
        handleFit={handleFit}
        recipeInputRef={recipeInputRef} handleRecipeImport={handleRecipeImport}
        projectId={projectId} isRecipeRunning={isRecipeRunning}
        onMarkClean={handleMarkClean}
        activeImageId={activeImageId}
        activeImageName={images.find((img) => img.id === activeImageId)?.name ?? null}
        descMode={!!descMode}
        gpuBusy={gpuBusy}
        dziUrl={dziUrl}
        maskTileBaseUrl={maskTileUrl}
        lut={lut}
        activeClassId={activeClassId}
      />
      <aside className="toolbox">
        <ClassPanel
          classesDraft={classesDraft} activeClassId={activeClassId}
          setActiveClass={setActiveClass} addClass={addClass}
          updateClass={updateClass} handleDeleteClass={handleDeleteClass}
          clearClassById={clearClassById}
          selectedCount={selectedIds.size}
          onClearClassSelected={handleClearClassSelected}
          onMarkClean={handleMarkClean}
          onUnmarkClean={(selectedIds.size > 0 || activeImageId) ? () => handleUnmarkClean() : undefined}
          isClean={(() => {
            const imgs = imageListHook.images;
            // When rows are multi-selected, surface the "OK 解除" button if
            // any of them is currently marked clean — so a bulk unmark is
            // reachable even when the *active* image isn't the clean one.
            if (selectedIds.size > 0) {
              for (const id of selectedIds) {
                if (imgs.find(i => i.id === id)?.annotation?.markedClean) return true;
              }
              return false;
            }
            return !!(activeImageId && imgs.find(i => i.id === activeImageId)?.annotation?.markedClean);
          })()}
          prepareReport={prepareReport}
        />
        <AiAssistPanel
          projectId={projectId} activeImageId={activeImageId}
          tool={tool} width={width} height={height}
          activeClassId={activeClassId} activeRecipe={activeRecipe}
          recipePreview={recipePreview} isRecipeRunning={isRecipeRunning}
          handleRecipePreview={handleRecipePreview}
          handleAutoLabelPreview={handleAutoLabelPreview}
          handleRecipeConfirm={handleRecipeConfirm} handleRecipeCancel={handleRecipeCancel}
          handleRecipeApplyAll={handleRecipeApplyAll}
          samModel={samModel} setSamModel={setSamModel}
          samRefCurrent={samRef.current}
          handleSamConfirm={handleSamConfirm} handleSamCancel={handleSamCancel}
          spotSensitivity={spotSensitivity} setSpotSensitivity={setSpotSensitivity}
          spotCount={spotCount} setSpotCount={setSpotCount}
          spotDetectRefCurrent={spotDetectRef.current} spotPhase={spotPhase}
          handleSpotConfirm={handleSpotConfirm} handleSpotCancel={handleSpotCancel} handleSpotRunDetect={handleSpotRunDetect}
          colorTolerance={colorTolerance} setColorTolerance={setColorTolerance}
          assistPreview={assistPreview} setAssistPreview={setAssistPreview} setStatus={setStatus}
          superpixelRefCurrent={superpixelRef.current}
          nSegments={nSegments} setNSegments={setNSegments}
          handleSuperpixelConfirm={handleSuperpixelConfirm} handleSuperpixelCancel={handleSuperpixelCancel}
          handleSuperpixelRecompute={handleSuperpixelRecompute}
          crackTraceRefCurrent={crackTraceRef.current}
          crackSensitivity={crackSensitivity} setCrackSensitivity={setCrackSensitivity}
          crackWidth={crackWidth} setCrackWidth={setCrackWidth}
          handleCrackConfirm={handleCrackConfirm} handleCrackCancel={handleCrackCancel}
          handleCrackRecompute={handleCrackRecompute}
        />
      </aside>
      {busyMessage && (
        <div className="annotator-busy-overlay" aria-hidden="false">
          <div
            className="annotator-busy-card"
            role="status"
            aria-live="polite"
            aria-busy="true"
          >
            {busyMessage}
          </div>
        </div>
      )}
    </div>
  );
})
