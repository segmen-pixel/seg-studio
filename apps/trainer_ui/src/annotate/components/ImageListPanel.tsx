// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useMemo, useRef, useState, useCallback, useEffect } from "react";
import { useI18n } from "../../i18n";
import type { ImageItem } from "../annotatorTypes";
import type { DatasetStats } from "../hooks/useImageList";
import type { PerImageClassMap } from "../hooks/usePerImageClassPresence";
import type { ClassItem } from "../../store";
import { RenameImportDialog } from "./RenameImportDialog";
import { AugmentDialog } from "./AugmentDialog";
import { AugmentIcon } from "./AugmentIcon";
import { augmentAnnotate } from "../../api/datasets";

// ---------------------------------------------------------------------------
// Virtual scroll constants
// ---------------------------------------------------------------------------
const ITEM_HEIGHT = 42; // matches contain-intrinsic-size in CSS
const OVERSCAN = 10;    // extra items rendered above/below viewport
const VIRTUAL_THRESHOLD = 200; // only virtualize when list exceeds this

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ClassPixelStat = {
  classId: number;
  name: string;
  color: [number, number, number];
  pixels: number;
  ratio: number;
};

export type ImageListPanelProps = {
  // image list data
  filteredImages: ImageItem[];
  activeImageId: string | null;
  selectedIds: Set<string>;
  isListDragActive: boolean;
  datasetStats: DatasetStats;
  classPixelStats: ClassPixelStat[];
  perImageClasses: PerImageClassMap;
  classesDraft: ClassItem[];
  prefetchMessage?: string | null;
  projectId: string | null;

  // handlers
  handleImageBatch: (files: FileList) => void;
  /** Called after augment generates new items so the list can reload. */
  onAugmentComplete?: () => void;
  selectAllFiltered: () => void;
  handleDeleteSelected: () => void;
  handleSelectClick: (item: ImageItem, event: React.MouseEvent, activateItem: boolean) => void;
  onMoveSelection: (direction: "up" | "down", shiftKey: boolean) => void;
  onClearSelection?: () => void;
  onUnmarkClean?: (imageId: string) => void;

  // drag handlers
  handleListDragEnter: (e: React.DragEvent<HTMLDivElement>) => void;
  handleListDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  handleListDragLeave: (e: React.DragEvent<HTMLDivElement>) => void;
  handleListDrop: (e: React.DragEvent<HTMLDivElement>) => void;

};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const ImageListPanel = React.memo(function ImageListPanel({
  filteredImages, activeImageId, selectedIds,
  isListDragActive,
  datasetStats, classPixelStats,
  perImageClasses, classesDraft, prefetchMessage,
  projectId,
  handleImageBatch, selectAllFiltered,
  handleDeleteSelected,
  handleSelectClick,
  onMoveSelection,
  onClearSelection,
  onUnmarkClean,
  onAugmentComplete,
  handleListDragEnter, handleListDragOver, handleListDragLeave, handleListDrop,
}: ImageListPanelProps) {
  const { t } = useI18n();
  const listRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(400);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [augmentDialogOpen, setAugmentDialogOpen] = useState(false);
  const [augmentRunning, setAugmentRunning] = useState(false);
  const [augmentError, setAugmentError] = useState<string | null>(null);

  const handleAugmentConfirm = useCallback(
    async (params: {
      count: number;
      classId: number;
      perlinStrength: number;
      colorJitter: number;
      defectsPerImage: [number, number];
      modePerlin: boolean;
      modeLighting: boolean;
      lightingVariants: Array<"daytime" | "evening" | "night">;
      useCleanHosts: boolean;
    }) => {
      if (!projectId) return;
      setAugmentRunning(true);
      setAugmentError(null);
      try {
        await augmentAnnotate(projectId, {
          count: params.count,
          perlin_strength: params.perlinStrength,
          color_jitter: params.colorJitter,
          defects_per_image: params.defectsPerImage,
          // 0 = "All classes"; positive id = single-class synthesis.
          class_id: params.classId,
          modes: { perlin: params.modePerlin, lighting: params.modeLighting },
          lighting_variants: params.lightingVariants,
          use_clean_hosts: params.useCleanHosts,
        });
        setAugmentDialogOpen(false);
        onAugmentComplete?.();
      } catch (err) {
        setAugmentError((err as Error).message || "augment failed");
      } finally {
        setAugmentRunning(false);
      }
    },
    [projectId, onAugmentComplete],
  );

  const handleRenameImport = useCallback((files: File[]) => {
    if (files.length === 0) return;
    const dt = new DataTransfer();
    for (const f of files) dt.items.add(f);
    handleImageBatch(dt.files);
  }, [handleImageBatch]);

  // Build a classId -> color lookup for rendering dots
  const classColorMap = useMemo(() => {
    const m = new Map<number, [number, number, number]>();
    for (const c of classesDraft) {
      if (c.id !== 0) m.set(c.id, c.color);
    }
    return m;
  }, [classesDraft]);

  // Track container resize
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setContainerHeight(entry.contentRect.height);
    });
    ro.observe(el);
    setContainerHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // rAF-throttled scroll handler: at most one setScrollTop per frame,
  // so fast wheel/drag scrolling on huge lists doesn't queue up hundreds
  // of renders that each re-run the virtual window computation.
  const scrollRafRef = useRef<number | null>(null);
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    if (scrollRafRef.current != null) return;
    const top = e.currentTarget.scrollTop;
    scrollRafRef.current = requestAnimationFrame(() => {
      setScrollTop(top);
      scrollRafRef.current = null;
    });
  }, []);
  useEffect(() => () => {
    if (scrollRafRef.current != null) cancelAnimationFrame(scrollRafRef.current);
  }, []);

  // Keep a ref of the current filteredImages so the scroll-into-view effect
  // doesn't re-run just because a new array identity flowed in (e.g. when
  // perImageClasses updates but the order/ids are the same).
  const filteredImagesRef = useRef(filteredImages);
  filteredImagesRef.current = filteredImages;

  // Scroll active item into view when the *active id* changes.
  useEffect(() => {
    if (!activeImageId || !listRef.current) return;
    const idx = filteredImagesRef.current.findIndex((img) => img.id === activeImageId);
    if (idx < 0) return;
    const el = listRef.current;
    const itemTop = idx * ITEM_HEIGHT;
    const itemBottom = itemTop + ITEM_HEIGHT;
    if (itemTop < el.scrollTop) {
      el.scrollTop = itemTop;
    } else if (itemBottom > el.scrollTop + el.clientHeight) {
      el.scrollTop = itemBottom - el.clientHeight;
    }
  }, [activeImageId]);

  // Virtual window calculation
  const useVirtual = filteredImages.length > VIRTUAL_THRESHOLD;
  const totalHeight = filteredImages.length * ITEM_HEIGHT;
  const startIdx = useVirtual ? Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN) : 0;
  const endIdx = useVirtual
    ? Math.min(filteredImages.length, Math.ceil((scrollTop + containerHeight) / ITEM_HEIGHT) + OVERSCAN)
    : filteredImages.length;
  const visibleItems = filteredImages.slice(startIdx, endIdx);

  const renderItem = useCallback((item: ImageItem, index: number) => {
    const classIds = perImageClasses.get(item.id);
    const hasAnnotation = !!classIds && classIds.length > 0;
    const isClean = !!item.annotation?.markedClean;
    const style: React.CSSProperties = useVirtual
      ? { cursor: "pointer", position: "absolute", top: (startIdx + index) * ITEM_HEIGHT, left: 0, right: 0, height: ITEM_HEIGHT }
      : { cursor: "pointer" };
    return (
      <div
        key={item.id}
        className={`card list-item-flat ${activeImageId === item.id ? "active" : ""} ${selectedIds.has(item.id) ? "selected" : ""} ${hasAnnotation ? "has-mask" : ""} ${isClean ? "is-clean" : ""}`}
        data-image-id={item.id}
        role="option"
        aria-selected={activeImageId === item.id}
        onClick={(event) => handleSelectClick(item, event, true)}
        style={style}
      >
        <div className="image-list-row">
          <div className="image-list-name" title={item.annotation?.synthetic ? `${t("augment.syntheticBadge")}: ${item.name}` : item.name}>
            {item.annotation?.synthetic ? (
              <span className="synth-badge" aria-label="synthetic">
                <AugmentIcon size={12} />
              </span>
            ) : null}
            {item.name}
          </div>
          {isClean && (
            <span className="clean-badge" title="No defects (verified)">
              ✓OK
              {onUnmarkClean && (
                <span
                  className="clean-badge-remove"
                  title={t("classPanel.clearOk")}
                  onClick={(e) => { e.stopPropagation(); onUnmarkClean(item.id); }}
                >×</span>
              )}
            </span>
          )}
        </div>
        {hasAnnotation && (
          <div className="image-list-class-dots">
            {classIds.map((cid) => {
              const color = classColorMap.get(cid);
              return color ? (
                <span
                  key={cid}
                  className="class-dot"
                  style={{ background: `rgb(${color[0]},${color[1]},${color[2]})` }}
                />
              ) : null;
            })}
          </div>
        )}
      </div>
    );
  }, [activeImageId, selectedIds, perImageClasses, classColorMap, handleSelectClick, onUnmarkClean, useVirtual, startIdx, t]);

  return (
    <aside className="side-panel">
      <div className="sidebar-toolbar">
        <button
          className="ghost compact augment-button"
          onClick={() => { setAugmentError(null); setAugmentDialogOpen(true); }}
          disabled={!projectId || filteredImages.length === 0}
          title={t("augment.tooltip")}
          data-desc={t("augment.tooltip")}
          data-desc-pos="bottom"
          data-tutorial-step="annotate-augment"
        >
          <AugmentIcon size={14} />
          <span>{t("augment.button")}</span>
        </button>
        <button className="primary compact" onClick={() => setRenameDialogOpen(true)} title={t("imageList.addImages")} data-desc={t("imageList.addImages")} data-desc-pos="bottom" data-tutorial-step="add-images">+</button>
        <button className="ghost" onClick={selectAllFiltered} title={t("imageList.selectAllDesc")} data-desc={t("imageList.selectAllDesc")} data-desc-pos="bottom">{t("imageList.selectAll")}</button>
        <button className="danger" onClick={handleDeleteSelected} disabled={selectedIds.size === 0} title={t("imageList.deleteDesc")} data-desc={t("imageList.deleteDesc")} data-desc-pos="bottom">
          {selectedIds.size > 0 ? `${t("imageList.delete")}(${selectedIds.size})` : t("imageList.delete")}
        </button>
      </div>
      <RenameImportDialog open={renameDialogOpen} onClose={() => setRenameDialogOpen(false)} onImport={handleRenameImport} />
      <AugmentDialog
        open={augmentDialogOpen}
        classes={classesDraft}
        onClose={() => setAugmentDialogOpen(false)}
        onConfirm={handleAugmentConfirm}
        running={augmentRunning}
        lastError={augmentError}
      />
      <div className="image-list-panel-meta">
        <span>{filteredImages.length} / {datasetStats.total} {t("imageList.images")}</span>
        <span>{selectedIds.size > 0 ? `${selectedIds.size} ${t("imageList.selected")}` : t("imageList.multiSelect")}</span>
      </div>
      {prefetchMessage ? (
        <div className="image-list-prefetch-hint" aria-live="polite">{prefetchMessage}</div>
      ) : null}
      <div
        ref={listRef}
        className={`list list-compact image-list-dropzone ${isListDragActive ? "drag-active" : ""}`}
        style={{ marginTop: 4 }}
        role="listbox"
        aria-label="Annotate images"
        tabIndex={0}
        onScroll={handleScroll}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            onMoveSelection("down", event.shiftKey);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            onMoveSelection("up", event.shiftKey);
          } else if (event.key === "Delete" && selectedIds.size > 0) {
            event.preventDefault();
            handleDeleteSelected();
          } else if (event.key === "Escape" && selectedIds.size > 0 && onClearSelection) {
            event.preventDefault();
            onClearSelection();
          }
        }}
        onDragEnter={handleListDragEnter}
        onDragOver={handleListDragOver}
        onDragLeave={handleListDragLeave}
        onDrop={handleListDrop}
      >
        {filteredImages.length === 0 && (
          <div className="image-list-empty">
            {datasetStats.total === 0
              ? t("imageList.emptyHint").split("\n").map((line, i) => <div key={i}>{line}</div>)
              : t("imageList.noFilterMatch")}
          </div>
        )}
        {useVirtual ? (
          <div style={{ position: "relative", height: totalHeight, minHeight: "100%" }}>
            {visibleItems.map((item, i) => renderItem(item, i))}
          </div>
        ) : (
          filteredImages.map((item, i) => renderItem(item, i))
        )}
      </div>
      <div className="dataset-stats-grid" style={{ marginTop: 10 }}>
        <div className="dataset-stat-item"><div className="dataset-stat-value">{datasetStats.total}</div><div className="dataset-stat-label">{t("imageList.images")}</div></div>
        <div className="dataset-stat-item"><div className="dataset-stat-value">{datasetStats.withMask}</div><div className="dataset-stat-label">{t("imageList.mask")}</div></div>
      </div>
      {classPixelStats.length > 0 && (
        <div className="dataset-class-stats">
          {classPixelStats.filter((cs) => classesDraft.some((c) => c.id === cs.classId)).map((cs) => (
            <div key={cs.classId} className="dataset-class-row">
              <span className="dataset-class-marker" style={{ background: `rgb(${cs.color[0]}, ${cs.color[1]}, ${cs.color[2]})` }} />
              <span className="dataset-class-name">{cs.name}</span>
              <span className="dataset-class-pct">{(cs.ratio * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
});
