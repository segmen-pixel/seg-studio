// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n";

type ImageItem = {
  id: string;
  name: string;
  filename: string;
  set: "none" | "train" | "val" | "test";
  width: number;
  height: number;
  annotation?: {
    hasMask: boolean;
    hasForeground?: boolean;
    markedClean?: boolean;
    revision: number;
    lastSavedAt?: string | null;
  };
};

type ImageListPanelProps = {
  images: ImageItem[];
  filterSet: "all" | ImageItem["set"];
  onFilterChange: (value: "all" | ImageItem["set"]) => void;
  keyboardActive?: boolean;
  activeImageId: string | null;
  selectedIds: Set<string>;
  onSelectedIdsChange: (ids: Set<string>) => void;
  onSelectImage: (item: ImageItem) => void;
  onApplyPredToLabel?: (imageId: string) => void;
  onBulkApplyPredToLabel?: (imageIds: string[]) => Promise<boolean>;
  onClearOkLabels?: (imageIds: string[]) => void;
  onMoveSelection: (delta: number) => void;
  onRefresh: () => void;
  perImageClassIds: Map<string, number[]>;
  classColorMap: Map<number, [number, number, number]>;
  visibleCount: number;
  totalCount: number;
};

export type { ImageItem };

export default React.memo(function ImageListPanel({
  images,
  filterSet,
  onFilterChange,
  keyboardActive,
  activeImageId,
  selectedIds,
  onSelectedIdsChange,
  onSelectImage,
  onApplyPredToLabel,
  onBulkApplyPredToLabel,
  onClearOkLabels,
  onMoveSelection,
  onRefresh,
  perImageClassIds,
  classColorMap,
  visibleCount,
  totalCount,
}: ImageListPanelProps) {
  const { t } = useI18n();
  const listRef = useRef<HTMLDivElement>(null);
  const filteredImages = images.filter((item) => filterSet === "all" || item.set === filterSet);

  // Review mode: check images one by one, then apply predictions to labels in bulk
  const [reviewMode, setReviewMode] = useState(false);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [applying, setApplying] = useState(false);
  const checkedCount = filteredImages.filter((i) => checkedIds.has(i.id)).length;

  const toggleChecked = useCallback((id: string) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  function handleReviewStart() {
    setCheckedIds(new Set());
    if (selectedIds.size > 0) onSelectedIdsChange(new Set());
    setReviewMode(true);
    listRef.current?.focus();
  }

  function handleReviewCancel() {
    setReviewMode(false);
    setCheckedIds(new Set());
  }

  async function handleReviewConfirm() {
    if (!onBulkApplyPredToLabel || applying) return;
    const ids = filteredImages.filter((i) => checkedIds.has(i.id)).map((i) => i.id);
    if (ids.length === 0) return;
    setApplying(true);
    try {
      const done = await onBulkApplyPredToLabel(ids);
      if (done) {
        setReviewMode(false);
        setCheckedIds(new Set());
      }
    } finally {
      setApplying(false);
    }
  }

  function handleItemClick(item: ImageItem, e: React.MouseEvent) {
    if (reviewMode) {
      onSelectImage(item);
      (e.currentTarget.parentElement as HTMLDivElement | null)?.focus();
      return;
    }
    if (e.ctrlKey || e.metaKey) {
      const next = new Set(selectedIds);
      if (next.has(item.id)) next.delete(item.id); else next.add(item.id);
      onSelectedIdsChange(next);
      return;
    }
    if (e.shiftKey) {
      // Use activeImageId as anchor for range selection
      const anchor = activeImageId;
      if (anchor) {
        const ids = filteredImages.map((i) => i.id);
        const a = ids.indexOf(anchor);
        const b = ids.indexOf(item.id);
        if (a >= 0 && b >= 0) {
          const [lo, hi] = a < b ? [a, b] : [b, a];
          const next = new Set(selectedIds);
          for (let i = lo; i <= hi; i++) next.add(ids[i]!);
          onSelectedIdsChange(next);
          return;
        }
      }
    }
    if (selectedIds.size > 0) onSelectedIdsChange(new Set());
    onSelectImage(item);
    (e.currentTarget.parentElement as HTMLDivElement | null)?.focus();
  }

  useEffect(() => {
    if (!activeImageId || !listRef.current) return;
    const el = listRef.current.querySelector(`[data-image-id="${activeImageId}"]`);
    if (el) (el as HTMLElement).scrollIntoView({ block: "nearest" });
  }, [activeImageId]);

  // Review-mode shortcuts that work regardless of which element has focus:
  // Up/Down moves through images, Space toggles the active image's checkbox.
  // The list's own onKeyDown calls preventDefault, so defaultPrevented guards
  // against double-handling when the list is focused.
  useEffect(() => {
    if (!reviewMode || !keyboardActive) return;
    function handleKey(event: KeyboardEvent) {
      if (event.defaultPrevented) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const el = document.activeElement as HTMLElement | null;
      if (el) {
        const tag = el.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable) return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        onMoveSelection(1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        onMoveSelection(-1);
      } else if (event.key === " ") {
        // Don't hijack Space from a focused button (it activates the button)
        if (el?.tagName === "BUTTON" || !activeImageId) return;
        event.preventDefault();
        toggleChecked(activeImageId);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [reviewMode, keyboardActive, activeImageId, onMoveSelection, toggleChecked]);

  return (
    <aside className="side-panel">
      <div className="sidebar-toolbar">
        <select value={filterSet} onChange={(e) => onFilterChange(e.target.value as typeof filterSet)} data-desc={t("imageList.filter.desc")}>
          <option value="all">All</option>
          <option value="train">Train</option>
          <option value="val">Val</option>
          <option value="test">Test</option>
          <option value="none">Unassigned</option>
        </select>
        <button className="ghost" onClick={onRefresh} data-desc={t("imageList.refreshDesc")} data-desc-pos="bottom">Refresh</button>
        {onBulkApplyPredToLabel && !reviewMode && (
          <button
            className="ghost"
            style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)" }}
            onClick={handleReviewStart}
            disabled={filteredImages.length === 0}
            title={t("imageList.reviewStartTitle")}
            data-desc={t("imageList.reviewStart.desc")}
          >
            {t("imageList.reviewStart")}
          </button>
        )}
        {reviewMode && (
          <>
            <button
              className="ghost"
              style={{ fontSize: 11, fontWeight: 600, color: "var(--accent)" }}
              onClick={handleReviewConfirm}
              disabled={applying || checkedCount === 0}
              title={t("imageList.reviewConfirmTitle")}
              data-desc={t("imageList.reviewConfirm.desc")}
            >
              {applying ? t("imageList.bulkApplyProgress") : `${t("imageList.reviewConfirm")} (${checkedCount})`}
            </button>
            <button
              className="ghost"
              style={{ fontSize: 11 }}
              onClick={handleReviewCancel}
              disabled={applying}
              title={t("imageList.reviewCancel")}
              data-desc={t("imageList.reviewCancel.desc")}
            >
              {t("imageList.reviewCancel")}
            </button>
          </>
        )}
        {selectedIds.size > 0 && onClearOkLabels && (() => {
          const okIds = Array.from(selectedIds).filter((id) => {
            const img = images.find((i) => i.id === id);
            return img?.annotation?.hasMask && img.annotation.hasForeground === false;
          });
          return okIds.length > 0 ? (
            <button
              className="ghost"
              style={{ fontSize: 11, fontWeight: 600, color: "var(--warning, #e57373)" }}
              onClick={() => onClearOkLabels(okIds)}
              title={t("imageList.clearOkTitle")}
              data-desc={t("imageList.clearOk.desc")}
            >
              {t("imageList.clearOk")} ({okIds.length})
            </button>
          ) : null;
        })()}
        {selectedIds.size > 0 && (
          <button
            className="ghost"
            style={{ fontSize: 11, padding: "2px 6px" }}
            onClick={() => onSelectedIdsChange(new Set())}
            title={t("imageList.clearSelection")}
            data-desc={t("imageList.clearSelection.desc")}
          >&times;</button>
        )}
      </div>
      <div className="results-image-list-meta">
        <span>{visibleCount}/{totalCount} images</span>
        <span>{reviewMode ? "Up/Down move / Space check" : "Up/Down to move"}</span>
      </div>
      <div
        ref={listRef}
        className="list list-compact results-image-list"
        style={{ marginTop: 4 }}
        role="listbox"
        aria-label="Prediction images"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            onMoveSelection(1);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            onMoveSelection(-1);
          } else if (reviewMode && event.key === " ") {
            event.preventDefault();
            if (activeImageId) toggleChecked(activeImageId);
          }
        }}
      >
        {filteredImages.map((item) => {
          const isSelected = selectedIds.has(item.id);
          return (
          <div
            key={item.id}
            className={`card list-item-flat ${activeImageId === item.id ? "active" : ""}${isSelected ? " selected" : ""}`}
            data-image-id={item.id}
            role="option"
            aria-selected={activeImageId === item.id || isSelected}
            onClick={(event) => handleItemClick(item, event)}
            onDoubleClick={reviewMode ? undefined : () => onApplyPredToLabel?.(item.id)}
            style={{ cursor: "pointer", background: isSelected ? "color-mix(in srgb, var(--accent) 15%, transparent)" : undefined }}
          >
            <div className="image-list-row">
              {reviewMode && (
                <input
                  type="checkbox"
                  checked={checkedIds.has(item.id)}
                  onClick={(e) => e.stopPropagation()}
                  onChange={() => toggleChecked(item.id)}
                  tabIndex={-1}
                  aria-label={`${t("imageList.reviewConfirmTitle")}: ${item.name}`}
                  style={{ margin: 0, flexShrink: 0, accentColor: "var(--accent)" }}
                />
              )}
              {item.set && item.set !== "none" && (
                <span
                  className={`image-list-set-badge image-list-set-${item.set}`}
                  title={item.set}
                >
                  {item.set === "train" ? "Tr" : item.set === "val" ? "Va" : "Ts"}
                </span>
              )}
              <span className="image-list-name" title={item.name}>{item.name}</span>
            </div>
            {perImageClassIds.has(item.id) && (
              <div className="image-list-class-dots">
                {(perImageClassIds.get(item.id) ?? []).map((cid) => {
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
        );})}
        {filteredImages.length === 0 && (
          <div className="muted results-image-list-empty">No images in this filter.</div>
        )}
      </div>
    </aside>
  );
});
