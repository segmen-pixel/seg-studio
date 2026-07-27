// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import type { ClassItem } from "../../store";
import type { PrepareReport } from "../hooks/useImageList";
import { useI18n } from "../../i18n";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ClassPanelProps = {
  classesDraft: ClassItem[];
  activeClassId: number;
  setActiveClass: (id: number) => void;

  // class actions
  addClass: () => void;
  updateClass: (idx: number, patch: Partial<ClassItem>) => void;
  handleDeleteClass: () => void;
  clearClassById: (id: number) => void;
  // batch clear across the image-list multi-selection
  selectedCount?: number;
  onClearClassSelected?: (id: number) => void;
  // mark clean
  onMarkClean?: () => void;
  onUnmarkClean?: () => void;
  isClean?: boolean;
  // report
  prepareReport: PrepareReport;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const ClassPanel = React.memo(function ClassPanel({
  classesDraft, activeClassId, setActiveClass,
  addClass, updateClass, handleDeleteClass, clearClassById,
  selectedCount = 0, onClearClassSelected,
  onMarkClean, onUnmarkClean, isClean,
  prepareReport,
}: ClassPanelProps) {
  const { t } = useI18n();
  return (
    <>
      <div data-tutorial-step="annotate-classes">
        <div className="row class-actions-row" style={{ marginTop: 4 }}>
          <button className="ghost" onClick={addClass}>{t("classPanel.add")}</button>
          <button className="danger" onClick={handleDeleteClass} disabled={activeClassId === 0}>{t("classPanel.delete")}</button>
        </div>
        <div className="list class-list-window">
          <div className={`card ${isClean ? "active" : ""}`}>
            <div className="class-item-row" onClick={onMarkClean} style={{ cursor: "pointer" }}
              title={t("annotate.markClean.desc")}>
              <span className="color-swatch" style={{ width: 22, height: 22 }}>
                <span style={{ background: "transparent", border: "2px solid var(--text-muted)", borderRadius: 3, width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700 }}>✓</span>
              </span>
              <span className="class-name-input" style={{ pointerEvents: "none", fontSize: 13, display: "block" }}>{t("annotate.markClean")}</span>
              {isClean && onUnmarkClean && (
                <button className="ghost class-clear-btn" onClick={(e) => { e.stopPropagation(); onUnmarkClean(); }}>
                  {t("classPanel.clearMarks")}
                </button>
              )}
              {!isClean && <span style={{ fontSize: 10, opacity: 0.5, whiteSpace: "nowrap" }}>Shift+C</span>}
            </div>
          </div>
          {classesDraft.map((cls, idx) => {
            if (cls.id === 0) return null;
            const colorHex = `#${cls.color.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
            return (
              <div key={cls.id} className={`card ${activeClassId === cls.id ? "active" : ""}`}>
                <div className="class-item-row" onClick={() => setActiveClass(cls.id)}>
                  <label className="color-swatch" title="Pick color">
                    <input
                      type="color" value={colorHex}
                      onChange={(e) => {
                        const hex = e.target.value.replace("#", "");
                        const color: [number, number, number] = [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
                        updateClass(idx, { color });
                      }}
                    />
                    <span style={{ background: `rgb(${cls.color[0]}, ${cls.color[1]}, ${cls.color[2]})` }} />
                  </label>
                  <input value={cls.name} onChange={(e) => updateClass(idx, { name: e.target.value })} className="class-name-input" />
                  {selectedCount > 0 && onClearClassSelected ? (
                    <button
                      className="ghost class-clear-btn"
                      title={t("classPanel.clearSelectedDesc").replace("{n}", String(selectedCount))}
                      onClick={() => onClearClassSelected(cls.id)}
                    >
                      {t("classPanel.clearMarks")}({selectedCount})
                    </button>
                  ) : (
                    <button className="ghost class-clear-btn" onClick={() => { setActiveClass(cls.id); clearClassById(cls.id); }}>
                      {t("classPanel.clearMarks")}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div>
        {prepareReport && (
          <div className="muted" style={{ marginTop: 6 }}>
            prepared {prepareReport.with_mask} | train {prepareReport.train_count} | val {prepareReport.val_count}
            {prepareReport.auto_val_from_train_count ? ` | auto val ${prepareReport.auto_val_from_train_count}` : ""}
          </div>
        )}
      </div>
    </>
  );
});
