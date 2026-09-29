// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n";
import type { ItemVerdict } from "../api";
import {
  VERDICT_LABEL_KEY,
  VERDICT_ORDER,
  pctOf,
  verdictNameOf,
  type VerdictName,
} from "./components/verdictFace";
import { VerdictMark } from "./components/VerdictMark";
import type { ClassItem } from "./types";

/** Keep the end of the name permanently visible. Files are often named like
 *  "part_cam1_0001234.png": end-truncation eats the serial and the extension,
 *  which are the only characters that tell one row from the next, and leaves
 *  the identical "part_cam1_000" prefix on screen. So the head truncates and
 *  the tail is pinned. Costs nothing -- it is the same span, split in two. */
const NAME_TAIL = 8;

function splitName(name: string): [string, string] {
  if (name.length <= NAME_TAIL + 4) return [name, ""];
  return [name.slice(0, -NAME_TAIL), name.slice(-NAME_TAIL)];
}

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
  /** set by prelabel or an agent's mark_review: a person should look */
  draft?: boolean;
  draftRun?: string;
  draftReason?: string;
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
  classes: ClassItem[];
  /** The rows to draw, already filtered. It comes from the parent rather than
   *  being re-derived here because Up/Down walks the parent's copy: two
   *  filters with the same intent still drift, and then the keyboard moves to
   *  a row that is not on screen. */
  visibleImages: ImageItem[];
  classFilter: Set<number>;
  onClassFilterChange: (next: Set<number>) => void;
  /** Per-image result at the current slider position, keyed by image id.
   *  A missing entry is the "unjudged" state and is drawn as such -- it used
   *  to render as a blank slot, which read as "fine". */
  verdicts?: Record<string, ItemVerdict>;
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
  classes,
  visibleImages,
  classFilter,
  onClassFilterChange,
  verdicts,
  visibleCount,
  totalCount,
}: ImageListPanelProps) {
  const { t } = useI18n();
  const anyVerdict = !!verdicts && Object.keys(verdicts).length > 0;
  const listRef = useRef<HTMLDivElement>(null);
  const filteredImages = visibleImages;

  // Review mode: check images one by one, then apply predictions to labels in bulk
  const [reviewMode, setReviewMode] = useState(false);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [applying, setApplying] = useState(false);
  const checkedCount = filteredImages.filter((i) => checkedIds.has(i.id)).length;

  // The strip is the confidence slider's readout. Without it the slider
  // silently repaints many small marks and expects the eye to catch the
  // difference; how many are missed, before and after the move, is the
  // number someone actually tunes against. It costs nothing per row, which
  // is why it can afford to be here at all.
  const verdictCounts = useMemo(() => {
    const c: Record<VerdictName, number> = {
      missed: 0, over: 0, detected: 0, clean: 0, unjudged: 0,
    };
    for (const item of filteredImages) c[verdictNameOf(verdicts?.[item.id])] += 1;
    return c;
  }, [filteredImages, verdicts]);

  // Where the class dots went. Their real job was "show me the images with
  // class 3" and they were doing it passively, from a client-side score cache
  // that could disagree with the row's own server-computed verdict -- which is
  // how a red dot ended up beside a blank verdict in the same row. As a
  // header control the same data filters instead of asserting, and it can say
  // nothing honestly while the cache is still cold.
  const classChips = useMemo(() => {
    const counts = new Map<number, number>();
    for (const ids of perImageClassIds.values()) {
      for (const id of ids) counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    return classes
      .filter((c) => c.id !== 0 && counts.has(c.id))
      .map((c) => ({ id: c.id, name: c.name, color: c.color, count: counts.get(c.id) ?? 0 }));
  }, [classes, perImageClassIds]);

  const toggleClassFilter = useCallback((id: number) => {
    const next = new Set(classFilter);
    if (next.has(id)) next.delete(id); else next.add(id);
    onClassFilterChange(next);
  }, [classFilter, onClassFilterChange]);

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
          <option value="all">{t("imageList.filter.all")}</option>
          <option value="train">{t("imageList.filter.train")}</option>
          <option value="val">{t("imageList.filter.val")}</option>
          <option value="test">{t("imageList.filter.test")}</option>
          <option value="none">{t("imageList.filter.none")}</option>
        </select>
        <button className="ghost" onClick={onRefresh} data-desc={t("imageList.refreshDesc")} data-desc-pos="bottom">{t("imageList.refresh")}</button>
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
      {anyVerdict && (
        <div className="results-verdict-strip">
          {VERDICT_ORDER.map((name) => (
            <span key={name} className="results-verdict-seg" title={t(VERDICT_LABEL_KEY[name])}>
              <VerdictMark verdict={name} />
              <b>{verdictCounts[name]}</b>
            </span>
          ))}
        </div>
      )}
      {(classChips.length > 0 || anyVerdict) && (
        <div className="results-class-filter">
          {classChips.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`results-class-chip${classFilter.has(c.id) ? " on" : ""}`}
              onClick={() => toggleClassFilter(c.id)}
              aria-pressed={classFilter.has(c.id)}
              data-desc={t("results.classFilter.desc")}
            >
              <i style={{ background: `rgb(${c.color[0]},${c.color[1]},${c.color[2]})` }} />
              {c.id} {c.name}
              <b>{c.count}</b>
            </button>
          ))}
          {anyVerdict && <span className="results-rate-head">{t("results.rateColumn")}</span>}
        </div>
      )}
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
          const [head, tail] = splitName(item.name);
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
              <span className="results-row-name" title={item.name}>
                <span className="results-row-name-head">{head}</span>
                {tail && <span className="results-row-name-tail">{tail}</span>}
              </span>
              {anyVerdict && (() => {
                // Shape carries the verdict, the number carries how much of the
                // annotation was covered. The pair rides at the END of the row,
                // not beside the set badge: "Va" is the same vermilion and the
                // same rounded rectangle as "missed", so side by side they read
                // as one marker. The name is flex:1, so the mark and the number
                // still land in a straight column down the whole list.
                const vr = verdicts?.[item.id];
                const name = verdictNameOf(vr);
                const rate = !vr || vr.match_rate === null || vr.match_rate === undefined
                  ? null
                  : pctOf(vr.match_rate);
                // Over-detection is area that sits off the annotation. It stays
                // in the tooltip and on the preview pill and never enters this
                // column, so the column means exactly one thing on every row.
                const excess = vr && vr.over_rate && vr.over_rate > 0.005
                  ? ` / ${t("results.verdict.overArea")} ${pctOf(vr.over_rate)}` : "";
                const label = t(VERDICT_LABEL_KEY[name]);
                return (
                  <>
                    <VerdictMark verdict={name} label={label} />
                    <span
                      className="results-row-rate"
                      title={rate ? `${label} / ${t("results.verdict.match")} ${rate}${excess}` : label}
                    >
                      {/* An em dash is a rate that does not exist: with no
                          annotated foreground, coverage is undefined, and 0%
                          or 100% would both be inventions. That is not only
                          the clean rows -- an image with no annotation the
                          model still painted on is "over" with no rate.
                          Blank is reserved for unjudged, where the value is
                          not undefined but simply not computed yet, and the
                          slash in the mark slot says so. */}
                      {rate ?? (name === "unjudged" ? "" : "\u2014")}
                    </span>
                  </>
                );
              })()}
            </div>
          </div>
        );})}
        {filteredImages.length === 0 && (
          <div className="muted results-image-list-empty">{t("imageList.noFilterMatch")}</div>
        )}
      </div>
    </aside>
  );
});
