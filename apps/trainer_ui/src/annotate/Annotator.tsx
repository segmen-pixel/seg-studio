// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n";
import { PaneSplitter, usePaneSize } from "../components/PaneSplitter";
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
import ClassMergeDialog from "./components/ClassMergeDialog";
import { AiAssistPanel } from "./components/AiAssistPanel";

import { useCanvasRendering } from "./hooks/useCanvasRendering";
import { useViewport } from "./hooks/useViewport";
import { useMaskIO, waitForPendingSaves } from "./hooks/useMaskIO";
import { useClassManager } from "./hooks/useClassManager";
import { useEditHistory } from "./hooks/useEditHistory";
import { useImageList } from "./hooks/useImageList";
import { usePixelStats, type RegionLabel } from "./hooks/usePixelStats";
import { usePerImageClassPresence } from "./hooks/usePerImageClassPresence";
import { useAiAssist } from "./hooks/useAiAssist";
import { useRecipe } from "./hooks/useRecipe";
import { usePrelabel } from "./hooks/usePrelabel";
import { useDrawingEvents } from "./hooks/useDrawingEvents";
import { useKeyboard } from "./hooks/useKeyboard";
import { useAnnotatorEffects } from "./hooks/useAnnotatorEffects";
import type { SamRefValue, SpotDetectRefValue, SuperpixelRefValue, CrackTraceRefValue, MoveRefValue } from "./hooks/useDrawingEvents";
import { initCrackTrace } from "./hooks/toolActions";
import type { AgentEvent } from "../api/agent";

export default React.memo(function Annotator({
  projectId, projects: _projects, onProjectChange: _onProjectChange, active, saveRef, previewStyle, setPreviewStyle: _setPreviewStyle, showToast, descMode, pendingImageId, onPendingImageHandled,
  agentSeq, agentEvents, agentFollow,
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
  /** feed position of the agent activity poll; bumps when the MCP bridge wrote */
  agentSeq?: number;
  /** what it wrote since the last poll */
  agentEvents?: AgentEvent[];
  /** jump to the image the bridge just wrote, so a person can watch it work */
  agentFollow?: boolean;
}) {
  const { t } = useI18n();
  // Canvas refs
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const uiCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Tool refs
  const strokeAccRef = useRef<Map<number, number> | null>(null);
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
  const [samModel, setSamModel] = useState<SamModelId>("mobile_sam");
  const blinkPhaseRef = useRef(0);
  const [_blinkPhase, _setBlinkPhase] = useState(0);
  const [nSegments, setNSegments] = useState(500);
  const [crackSensitivity, setCrackSensitivity] = useState(25);
  const [crackWidth, setCrackWidth] = useState(0);
  const samBoxDraftRef = useRef<{ start: [number, number]; end: [number, number] } | null>(null);
  const rectDraftRef = useRef<{ start: [number, number]; end: [number, number] } | null>(null);
  const [activeImageId, setActiveImageId] = useState<string | null>(null);
  const [mergeOpen, setMergeOpen] = useState(false);
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
  // Poll GPU device busy state — block GPU tools only when ALL CUDA devices are busy
  useEffect(() => {
    const check = async () => {
      try {
        const { fetchTorchDevices } = await import("../api/hardware");
        const data = await fetchTorchDevices();
        const devices = data?.devices ?? [];
        // Block only when every CUDA device is occupied (FIFO: SAM uses whichever GPU is free)
        const cudaDevices = devices.filter((d) => d.kind === "cuda");
        const allCudaBusy = cudaDevices.length > 0 && cudaDevices.every((d) => d.busy);
        setGpuBusy(allCudaBusy);
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
  // Which project the draft above was loaded from. The draft alone does not
  // say, and four of the five writers only know which project is open now.
  const classesOwnerRef = useRef<string | null>(null);
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
        samRef: samRef.current ? { points: samRef.current.points, box: samRef.current.box } : null,
        samBoxDraft: samBoxDraftRef.current,
        rectDraft: rectDraftRef.current,
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
    // Same reason as the results overlay: this panel stays mounted while
    // another tab is shown, and this timer repaints the overlay twice a
    // second whether or not anyone can see it.
    if (previewStyle !== 1 || !active) return;
    const id = setInterval(() => {
      blinkPhaseRef.current = blinkPhaseRef.current ? 0 : 1;
      scheduleDrawOverlay();
    }, 500);
    return () => clearInterval(id);
  }, [previewStyle, active, scheduleDrawOverlay]);

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
  const { loadClasses, saveClassList: _saveClassList, addClass, updateClass, deactivateClass: _deactivateClass, handleDeleteClass, handleMergeClass } =
    useClassManager(
      projectId, projectRef, classesDraft, setClassesDraft, classesDraftRef, classesOwnerRef, classIdCounterRef,
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
  // Blank all three annotate canvases. The image-paint effect bails on an
  // empty imageUrl, so after the active image is deleted the old pixels
  // would otherwise stay on screen.
  const clearCanvases = () => {
    invalidateOverlay();
    for (const ref of [imageCanvasRef, overlayCanvasRef, uiCanvasRef]) {
      const canvas = ref.current;
      if (canvas) { canvas.width = 0; canvas.height = 0; }
    }
  };
  const imageListOpsRef = useRef({
    saveMask, autoSave, handleFit,
    selectImage: undefined as ((item: ImageItem) => void) | undefined,
    putBitmapToCache, getBitmapFromCache, invalidateOverlay, clearCanvases,
  });
  imageListOpsRef.current = { saveMask, autoSave, handleFit, selectImage: undefined, putBitmapToCache, getBitmapFromCache, invalidateOverlay, clearCanvases };
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
  // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot jump; the parent's ack callback must not retrigger it
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
    handleSamConfirm, handleSamInvert, handleSamCancel, handleSpotConfirm, handleSpotCancel, handleSpotRunDetect,
    handleSpotSensitivity,
    handleSuperpixelConfirm, handleSuperpixelCancel,
    handleCrackConfirm, handleCrackCancel,
  } = useAiAssist(
    projectId, activeImageId, activeImageIdRef, width, height,
    assistPreview, setAssistPreview,
    () => aiOpsRef.current, samRef, samBoxDraftRef, spotDetectRef, superpixelRef, crackTraceRef, spotCount, setSpotCount, setStatus,
    spotSensitivity, setSpotSensitivity, setSpotPhase,
  );

  // 12. useRecipe
  // The recipe apply rewrites an unknown set of masks server-side; these two
  // let it drop every suspect client copy and pull the active image back
  // from the server (see useRecipe.handleRecipeApplyAll).
  const invalidateMaskCaches = () => {
    cacheRef.current.clear();
    maskLoadPromisesRef.current.clear();
  };
  // Images the MCP bridge wrote that this tab has not shown fresh since. The
  // write makes every copy of them here suspect -- the cache, a snapshot taken
  // on leaving, an item in a list read before the write -- until a fresh load
  // has been put on screen while it was the image there.
  const agentWroteRef = useRef<Set<string>>(new Set());
  const agentBulkRef = useRef(false);
  /** Which image the store holds now, from refs, not from the render that made the closure. */
  const shownImageId = () => imageListHook.storeImageIdRef.current ?? activeImageIdRef.current;
  const reloadActiveMask = async (fresh?: boolean) => {
    // Read when it runs, not when the closure was made: a timer or a follow
    // can call this after the screen has moved on.
    const id = shownImageId();
    if (!projectId || !id) return;
    if (pendingSwitchRef.current && pendingSwitchRef.current !== id) return;   // arriving elsewhere
    const item = images.find((it) => it.id === id);
    if (!item) return;
    const entry = await getOrLoadMaskEntry(item, projectId, fresh);
    // Only onto the image it was loaded for. A follow moved the screen on while
    // one of these was in flight, and image N's mask was painted onto N+1.
    if (shownImageId() !== id || (pendingSwitchRef.current && pendingSwitchRef.current !== id)) return;
    useMaskStore.getState().setMask(entry.maskIndex, entry.width, entry.height);
    editOpsRef.current.scheduleDrawOverlay();
    if (fresh) agentWroteRef.current.delete(id);
  };
  // 12b. What the MCP bridge wrote, reflected here as it lands.
  // Only the touched images are dropped from the client cache -- and never
  // one with unsaved paint in it -- then the list is re-read past its 10 s
  // TTL so the dots and counters move, and the active image is pulled back
  // from the server if it was among them.
  const agentReloadTimerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!projectId || !agentEvents || agentEvents.length === 0) return;
    const mine = agentEvents.filter((e) => e.project_id === projectId);
    if (mine.length === 0) return;
    let bulk = false;
    let wrote = false;
    for (const e of mine) {
      // Neither writes a mask: sam_segment says which image a run is working
      // on, and step is what it said at one of its steps before labelling.
      if (e.action === "sam_segment" || e.action === "step") continue;
      wrote = true;
      if (e.item_id) {
        const entry = cacheRef.current.get(e.item_id);
        if (entry && (entry as { dirty?: boolean }).dirty) continue;
        cacheRef.current.delete(e.item_id);
        maskLoadPromisesRef.current.delete(e.item_id);
        agentWroteRef.current.add(e.item_id);
      } else {
        bulk = true; // mark-clean / clear-class / upload: ids unknown here
      }
    }
    if (bulk) {
      agentBulkRef.current = true;
      for (const [iid, entry] of cacheRef.current) {
        if (iid !== activeImageId && !(entry as { dirty?: boolean }).dirty) {
          cacheRef.current.delete(iid);
          maskLoadPromisesRef.current.delete(iid);
        }
      }
    }
    // Only a write is worth re-reading the project for. The other events are
    // the frequent ones -- one per point the model tries -- and re-listing the
    // whole project with a sync on each of those made the screen crawl during
    // a run for nothing: they change no mask.
    if (!wrote) {
      if (agentFollow && active) followNewest();
      return;
    }
    if (agentReloadTimerRef.current) window.clearTimeout(agentReloadTimerRef.current);
    agentReloadTimerRef.current = window.setTimeout(() => {
      agentReloadTimerRef.current = null;
      // The list first, then the mask: reloadActiveMask reads the item out of
      // `images`, and started in the same tick it read the state from before
      // the write. On an image the client still believed had no mask that
      // decoded as empty and cached it, so the bridge's first mask on a fresh
      // image never appeared -- the one case this whole path is for. `fresh`
      // also gets past the browser's cache of the mask URL.
      void (async () => {
        try {
          await loadAnnotateItems(projectId, false, true);
        } finally {
          // Decided when it fires, from what the store holds then. Decided when
          // the poll was read, a follow in between or a second write poll
          // re-arming this timer lost it.
          const shown = shownImageId();
          const all = agentBulkRef.current;
          agentBulkRef.current = false;
          if (all || (shown && agentWroteRef.current.has(shown))) void reloadActiveMask(true);
        }
      })();
    }, 250);
    // Follow: move to the newest image the bridge wrote, unless the person
    // has unsaved paint on the one they are looking at.
    function followNewest() {
      const newest = [...mine].reverse().find((e) => e.item_id && e.item_id !== activeImageId);
      const here = activeImageId ? cacheRef.current.get(activeImageId) : undefined;
      if (newest?.item_id && !(here && (here as { dirty?: boolean }).dirty)) {
        const target = images.find((img) => img.id === newest.item_id);
        if (target) selectImage(target);
      }
    }
    if (agentFollow && active) followNewest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentSeq]);

  // Arriving at an image the bridge wrote -- followed there, or clicked --
  // before a fresh copy of it was on screen: what the switch showed came from
  // a cache or a list read before the write, and could be blank or old.
  useEffect(() => {
    if (activeImageId && agentWroteRef.current.has(activeImageId)) void reloadActiveMask(true);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeImageId]);
  useEffect(() => {
    agentWroteRef.current.clear();
    agentBulkRef.current = false;
  }, [projectId]);

  const recipeOpsRef = useRef({ markDirty, markTouched, autoSave, flushDirtyMasks, invalidateMaskCaches, reloadActiveMask });
  recipeOpsRef.current = { markDirty, markTouched, autoSave, flushDirtyMasks, invalidateMaskCaches, reloadActiveMask };
  const {
    activeRecipe, setActiveRecipe, recipePreview, setRecipePreview,
    isRecipeRunning, recipeInputRef,
    handleRecipeImport, loadActiveRecipe, handleRecipePreview, handleRecipeConfirm, handleRecipeCancel, handleRecipeApplyAll,
  } = useRecipe(projectId, activeImageId, activeImageIdRef, width, height, () => recipeOpsRef.current, setStatus, setBusyMessage, setImages);

  // 12b. usePrelabel - draft annotations from a finished run. Same ops as the
  // recipe apply: it too rewrites masks server-side for the whole project.
  const {
    prelabelRuns, prelabelRunId, setPrelabelRunId, prelabelPreviewActive,
    isPrelabelRunning, prelabelUnannotated,
    handlePrelabelPreview, handlePrelabelConfirm, handlePrelabelCancel, handlePrelabelAll,
  } = usePrelabel(projectId, activeImageId, activeImageIdRef, width, height, () => recipeOpsRef.current, assistPreview, setAssistPreview, setStatus, setBusyMessage, setImages);

  // 13. useDrawingEvents
  const drawingOpsRef = useRef({
    drawOverlay, scheduleDrawOverlay, markDirty, markTouched, renderUi, scheduleRenderUi,
    handleSamConfirm, handleSamInvert, handleSamCancel, handleSpotConfirm, handleSpotCancel,
    handleSuperpixelConfirm, handleSuperpixelCancel,
    handleCrackConfirm, handleCrackCancel,
  });
  drawingOpsRef.current = {
    drawOverlay, scheduleDrawOverlay, markDirty, markTouched, renderUi, scheduleRenderUi,
    handleSamConfirm, handleSamInvert, handleSamCancel, handleSpotConfirm, handleSpotCancel,
    handleSuperpixelConfirm, handleSuperpixelCancel,
    handleCrackConfirm, handleCrackCancel,
  };
  const handleSuperpixelRecompute = () => {
    superpixelRef.current = null;
    setAssistPreview(null);
    setStatus(t("annotate.superpixel.recompute"));
  };
  const handleCrackRecompute = () => {
    if (!projectId || !activeImageId || width <= 0 || height <= 0) return;
    const imgId = activeImageId;
    const ctRef = crackTraceRef.current;
    if (ctRef) ctRef.loading = true;
    setStatus(t("annotate.crackTrace.recomputing"));
    initCrackTrace(projectId, activeImageId, width, height, crackSensitivity, crackWidth, activeClassId, setStatus)
      .then((result) => {
        // A recompute that outlives an image switch would push the old
        // image's cracks onto the new one.
        if (activeImageIdRef.current !== imgId) return;
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
      brushSize, strokeAccRef, samRef, spotDetectRef,
      spotSensitivity, setSpotCount, setAssistPreview, setRecipePreview, setStatus, samModel,
      projectId, activeImageId, spacePressed, isPanning, setIsPanning, panStartRef,
      setMeasureStart, setMeasureEnd, measureStart, measureEnd,
      samBoxDraftRef,
      rectDraftRef,
      superpixelRef, nSegments,
      assistPreview,
      crackTraceRef, crackSensitivity, crackWidth,
      setSpotPhase,
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
    setStatus(t("annotate.saved"));
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
      // A debounced autosave — or the fire-and-forget save fired on image
      // switch — still in flight would PUT the old mask AFTER the server-side
      // write below and resurrect the markings. Cancel the timer, drop the
      // dirty flags so nothing re-fires, then let the wire go quiet.
      if (autoSaveTimerRef.current !== null) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      for (const iid of targetIds) {
        const entry = cacheRef.current.get(iid);
        if (entry) entry.dirty = false;
      }
      await waitForPendingSaves(targetIds);
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
        // The overlay canvas deliberately does not react to maskVersion —
        // content changes must redraw explicitly (useAnnotatorEffects).
        editOpsRef.current.scheduleDrawOverlay();
      }
      // Marked-clean images sitting in the cache with paint still in them
      // would show that paint again on the next visit — drop them (and any
      // in-flight load, which would re-insert pre-clean pixels) so they
      // reload from the now-zeroed server mask.
      for (const iid of targetIds) {
        if (iid !== activeImageId) cacheRef.current.delete(iid);
        maskLoadPromisesRef.current.delete(iid);
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
      // A debounced autosave — or the fire-and-forget save fired on image
      // switch — still in flight would PUT the old mask AFTER the server-side
      // write below and resurrect the markings. Cancel the timer, drop the
      // dirty flags so nothing re-fires, then let the wire go quiet.
      if (autoSaveTimerRef.current !== null) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      for (const iid of ids) {
        const entry = cacheRef.current.get(iid);
        if (entry) entry.dirty = false;
      }
      await waitForPendingSaves(ids);
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
        // The overlay canvas deliberately does not react to maskVersion —
        // content changes must redraw explicitly (useAnnotatorEffects).
        editOpsRef.current.scheduleDrawOverlay();
      }
      // Visited-but-inactive selected images may sit in the client mask
      // cache with the class still present — drop them (and any in-flight
      // load, which would re-insert pre-clear pixels) so they reload from
      // the server instead of writing stale pixels back.
      for (const iid of ids) {
        if (iid !== activeImageId) cacheRef.current.delete(iid);
        maskLoadPromisesRef.current.delete(iid);
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
      // Quiesce like mark-clean, minus the dirty-flag drop: unmarking does
      // not change mask content, so a truthful dirty flag is safe — and
      // clearing it here used to silently discard unsaved strokes.
      if (autoSaveTimerRef.current !== null) {
        window.clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      await waitForPendingSaves(targetIds);
      const { unmarkImagesClean } = await import("../api/datasets");
      await unmarkImagesClean(projectId, targetIds);
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
    handleUndo, handleRedo, handleClear, handleSamConfirm, handleSamInvert, handleSamCancel, handleSpotConfirm, handleSpotCancel, handleSuperpixelConfirm, handleSuperpixelCancel, handleCrackConfirm, handleCrackCancel, handleMarkClean, handleSaveAll, handleArrowNav,
  });
  kbOpsRef.current = {
    handleUndo, handleRedo, handleClear, handleSamConfirm, handleSamInvert, handleSamCancel, handleSpotConfirm, handleSpotCancel, handleSuperpixelConfirm, handleSuperpixelCancel, handleCrackConfirm, handleCrackCancel, handleMarkClean, handleSaveAll, handleArrowNav,
  };
  const gpuBusyRef = useRef(gpuBusy);
  gpuBusyRef.current = gpuBusy;
  useKeyboard(() => kbOpsRef.current, samRef, spotDetectRef, superpixelRef, crackTraceRef, classesDraftRef, setBrushSize, gpuBusyRef, moveRef);

  // 15. useAnnotatorEffects
  useAnnotatorEffects({
    projectId, projectRef, cacheRef, maskLoadPromisesRef, prefetchEpochRef, pendingSwitchRef, loadingTargetRef,
    autoSaveTimerRef, classSaveTimerRef, classIdCounterRef, classesDraftRef, classesOwnerRef,
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
      setStatus(t("annotate.maskImported").replace("{name}", file.name));
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

  // 16. Pane widths, one splitter either side of the canvas
  const layoutRef = useRef<HTMLDivElement>(null);
  const leftPane = usePaneSize({
    storageKey: "seg-annotate-pane-left",
    fallback: 240, min: 180, max: 560,
    containerRef: layoutRef,
    measure: (ev, rect) => ev.clientX - rect.left,
  });
  const rightPane = usePaneSize({
    storageKey: "seg-annotate-pane-right",
    fallback: 260, min: 200, max: 560,
    containerRef: layoutRef,
    measure: (ev, rect) => rect.right - ev.clientX,
    arrowSign: -1,
  });

  // JSX

  // The list has to be re-read after a class is deleted, the way it is after
  // every other change to what is on the images. The delete itself does clear
  // the review flag on the server; the badge stayed on screen because this
  // never asked the server again, and the rows in hand were the ones fetched
  // before the class went away.
  const deleteClassAndReload = useCallback(() => {
    void (async () => {
      await handleDeleteClass();
      if (projectId) await loadAnnotateItems(projectId);
    })();
  }, [handleDeleteClass, loadAnnotateItems, projectId]);

  return (
    <div
      ref={layoutRef}
      className={`annotate-layout${descMode ? " desc-mode" : ""}${isListDragActive ? " drag-active" : ""}`}
      style={{
        "--pane-left": `${leftPane.size}px`,
        "--pane-right": `${rightPane.size}px`,
      } as React.CSSProperties}
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
      <PaneSplitter handle={leftPane.handle} title={t("pane.resize")} />
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
      <PaneSplitter handle={rightPane.handle} title={t("pane.resize")} />
      <aside className="toolbox">
        <ClassPanel
          classesDraft={classesDraft} activeClassId={activeClassId}
          setActiveClass={setActiveClass} addClass={addClass}
          updateClass={updateClass} handleDeleteClass={deleteClassAndReload}
          onOpenMerge={() => setMergeOpen(true)}
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
        <ClassMergeDialog
          open={mergeOpen}
          classes={classesDraft}
          initialFromId={activeClassId}
          onClose={() => setMergeOpen(false)}
          onMerge={(fromId, toId, rename) => { setMergeOpen(false); void handleMergeClass(fromId, toId, rename); }}
        />
        <AiAssistPanel
          projectId={projectId} activeImageId={activeImageId}
          tool={tool} width={width} height={height}
          activeClassId={activeClassId} activeRecipe={activeRecipe}
          recipePreview={recipePreview} isRecipeRunning={isRecipeRunning}
          handleRecipePreview={handleRecipePreview}
          handleRecipeConfirm={handleRecipeConfirm} handleRecipeCancel={handleRecipeCancel}
          handleRecipeApplyAll={handleRecipeApplyAll}
          prelabelRuns={prelabelRuns} prelabelRunId={prelabelRunId}
          setPrelabelRunId={setPrelabelRunId} prelabelPreviewActive={prelabelPreviewActive}
          isPrelabelRunning={isPrelabelRunning} prelabelUnannotated={prelabelUnannotated}
          handlePrelabelPreview={handlePrelabelPreview} handlePrelabelConfirm={handlePrelabelConfirm}
          handlePrelabelCancel={handlePrelabelCancel} handlePrelabelAll={handlePrelabelAll}
          samModel={samModel} setSamModel={setSamModel}
          samRefCurrent={samRef.current}
          handleSamConfirm={handleSamConfirm} handleSamInvert={handleSamInvert} handleSamCancel={handleSamCancel}
          spotSensitivity={spotSensitivity} setSpotSensitivity={setSpotSensitivity}
          spotCount={spotCount} setSpotCount={setSpotCount}
          spotDetectRefCurrent={spotDetectRef.current} spotPhase={spotPhase}
          handleSpotConfirm={handleSpotConfirm} handleSpotCancel={handleSpotCancel} handleSpotRunDetect={handleSpotRunDetect}
          handleSpotSensitivity={handleSpotSensitivity}
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
