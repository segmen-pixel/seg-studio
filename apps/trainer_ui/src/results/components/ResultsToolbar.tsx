// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState, useRef, useCallback } from "react";
import type { ClassItem } from "../types";
import { useI18n } from "../../i18n";

type Props = {
  overlayVisible: boolean;
  setOverlayVisible: (fn: (v: boolean) => boolean) => void;
  showRegionLabels: boolean;
  setShowRegionLabels: (fn: (v: boolean) => boolean) => void;
  imageBrightness: number;
  setImageBrightness: (v: number) => void;
  imageContrast: number;
  setImageContrast: (v: number) => void;
  overlayAlpha: number;
  setOverlayAlpha: (fn: (v: number) => number) => void;
  heatmapMode: "none" | "confidence" | "class" | "error";
  setHeatmapMode: (fn: (m: "none" | "confidence" | "class" | "error") => "none" | "confidence" | "class" | "error") => void;
  heatmapClassId: number;
  setHeatmapClassId: (v: number) => void;
  /** Instance runs have no per-pixel probs — the heatmap tools are hidden. */
  isInstanceRun?: boolean;
  effectiveClasses: ClassItem[];
  showCount: boolean;
  setShowCount: (fn: (v: boolean) => boolean) => void;
  showArea: boolean;
  setShowArea: (fn: (v: boolean) => boolean) => void;
  calibration: { pixelDist: number; realDist: number; unit: string } | null;
  setCalibration: (v: null) => void;
  previewScale: number;
  width: number;
  height: number;
  hiddenClassIds: Set<number>;
  setHiddenClassIds: (fn: (prev: Set<number>) => Set<number>) => void;
  showGtOutline: boolean;
  setShowGtOutline: (fn: (v: boolean) => boolean) => void;
  instanceHighlight: boolean;
  setInstanceHighlight: (fn: (v: boolean) => boolean) => void;
  predOverlayPattern: "none" | "tint" | "hatch" | "dots" | "crosshatch" | "fine-dots";
  setPredOverlayPattern: (v: "none" | "tint" | "hatch" | "dots" | "crosshatch" | "fine-dots") => void;
  fitPreview: () => void;
};

export default React.memo(function ResultsToolbar({
  overlayVisible,
  setOverlayVisible,
  showRegionLabels,
  setShowRegionLabels,
  imageBrightness,
  setImageBrightness,
  imageContrast,
  setImageContrast,
  overlayAlpha,
  setOverlayAlpha,
  heatmapMode,
  setHeatmapMode,
  heatmapClassId,
  setHeatmapClassId,
  isInstanceRun = false,
  effectiveClasses,
  showCount,
  setShowCount,
  instanceHighlight,
  setInstanceHighlight,
  showArea,
  setShowArea,
  calibration,
  setCalibration,
  previewScale,
  width,
  height,
  hiddenClassIds,
  setHiddenClassIds,
  showGtOutline,
  setShowGtOutline,
  predOverlayPattern,
  setPredOverlayPattern,
  fitPreview,
}: Props) {
  const { t } = useI18n();
  const [patternMenuOpen, setPatternMenuOpen] = useState(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout>>();
  const openMenu = useCallback(() => { clearTimeout(closeTimer.current); setPatternMenuOpen(true); }, []);
  const closeMenu = useCallback(() => { closeTimer.current = setTimeout(() => setPatternMenuOpen(false), 150); }, []);
  return (
    <>
      {/* ── Left toolbar: view controls ── */}
      <div className="overlay-tools left" onPointerDown={(e) => e.stopPropagation()}>
        <button className="tool-icon" onClick={fitPreview} title={t("results.toolbar.fitToScreen")}>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2" /><path d="M8 8h4M16 8h-4M8 16h4M16 16h-4" /></svg>
        </button>
        <button className={`tool-icon ${!overlayVisible ? "active" : ""}`} onClick={() => setOverlayVisible((v) => !v)}
          title={overlayVisible ? t("results.toolbar.hideAnnotation") : t("results.toolbar.showAnnotation")}>
          {overlayVisible
            ? <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z" /><circle cx="12" cy="12" r="3" /></svg>
            : <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" /><path d="M1 1l22 22" /></svg>}
        </button>
        <button className={`tool-icon ${imageBrightness !== 100 || imageContrast !== 100 ? "active" : ""}`}
          onClick={() => { if (imageBrightness !== 100 || imageContrast !== 100) { setImageBrightness(100); setImageContrast(100); } else { setImageBrightness(120); } }}
          title={t("results.toolbar.brightness")}>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>
        </button>
        {/* Opacity */}
            <button className="tool-icon" title={t("results.toolbar.opacity").replace("{pct}", String(Math.round((overlayAlpha / 255) * 100)))}
              onClick={() => setOverlayAlpha((v) => v >= 180 ? 80 : v >= 80 ? 220 : 140)}>
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" opacity="0.4" /><path d="M12 2a10 10 0 0 1 0 20" fill="currentColor" stroke="none" /></svg>
            </button>
            <div className="left-toolbar-info">{Math.round((overlayAlpha / 255) * 100)}%</div>
        {/* Zoom */}
        {width > 0 && height > 0 && (
          <div className="left-toolbar-info">{Math.round(previewScale * 100)}%</div>
        )}
      </div>

      {/* ── Right toolbar: analysis tools ── */}
      <div className="overlay-tools right" onPointerDown={(e) => e.stopPropagation()}>
        {/* Prediction overlay with pattern menu */}
        <div className={`gt-compare-wrap${patternMenuOpen ? " open" : ""}`}
          onMouseEnter={openMenu}
          onMouseLeave={closeMenu}>
          <button className={`tool-icon ${showGtOutline ? "active" : ""}`}
            onClick={() => setShowGtOutline((v) => !v)}
            title={t("results.toolbar.prediction")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="3" strokeDasharray="4 3" />
              <circle cx="12" cy="12" r="4" fill="currentColor" opacity="0.3" />
            </svg>
          </button>
          {showGtOutline && patternMenuOpen && (
            <div className="gt-pattern-menu" onMouseEnter={openMenu} onMouseLeave={closeMenu}>
              {([
                ["hatch", t("results.toolbar.patternHatch"), <svg key="h" viewBox="0 0 16 16" width="14" height="14"><line x1="0" y1="16" x2="16" y2="0" stroke="currentColor" strokeWidth="1.5"/><line x1="-4" y1="12" x2="12" y2="-4" stroke="currentColor" strokeWidth="1.5"/><line x1="4" y1="20" x2="20" y2="4" stroke="currentColor" strokeWidth="1.5"/></svg>],
                ["fine-dots", t("results.toolbar.patternDots"), <svg key="f" viewBox="0 0 16 16" width="14" height="14"><circle cx="4" cy="4" r="1.5" fill="currentColor"/><circle cx="12" cy="4" r="1.5" fill="currentColor"/><circle cx="8" cy="8" r="1.5" fill="currentColor"/><circle cx="4" cy="12" r="1.5" fill="currentColor"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>],
                ["crosshatch", t("results.toolbar.patternCrosshatch"), <svg key="c" viewBox="0 0 16 16" width="14" height="14"><line x1="0" y1="16" x2="16" y2="0" stroke="currentColor" strokeWidth="1.2"/><line x1="0" y1="0" x2="16" y2="16" stroke="currentColor" strokeWidth="1.2"/><line x1="8" y1="0" x2="8" y2="16" stroke="currentColor" strokeWidth="1.2"/></svg>],
              ] as const).map(([val, label, icon]) => (
                <button key={val}
                  className={`gt-pattern-btn${predOverlayPattern === val ? " active" : ""}`}
                  onClick={() => setPredOverlayPattern(predOverlayPattern === val ? "none" : val)}
                  title={label}>
                  {icon}<span className="gt-pattern-label">{label}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        {/* Labels */}
        <button className={`tool-icon ${!showRegionLabels ? "active" : ""}`} onClick={() => setShowRegionLabels((v) => !v)}
          title={t("results.toolbar.labelsHint").replace("{label}", showRegionLabels ? t("results.toolbar.hideLabels") : t("results.toolbar.showLabels"))}>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 7h16M4 12h10M4 17h6" />
            {!showRegionLabels && <path d="M2 2l20 20" strokeWidth="2" />}
          </svg>
        </button>
        {/* Heatmaps (semantic runs only) */}
        {!isInstanceRun && (
          <>
            <button className={`tool-icon ${heatmapMode === "confidence" ? "active" : ""}`}
              onClick={() => setHeatmapMode((m) => m === "confidence" ? "none" : "confidence")}
              title={t("results.toolbar.confidenceHeatmap")}>
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z"/><circle cx="12" cy="12" r="5"/></svg>
            </button>
            <button className={`tool-icon ${heatmapMode === "class" ? "active" : ""}`}
              onClick={() => setHeatmapMode((m) => m === "class" ? "none" : "class")}
              title={t("results.toolbar.classHeatmap")}>
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M3 3h7v7H3V3zm11 0h7v7h-7V3zM3 14h7v7H3v-7zm11 0h7v7h-7v-7z"/></svg>
            </button>
            <button className={`tool-icon ${heatmapMode === "error" ? "active" : ""}`}
              onClick={() => setHeatmapMode((m) => m === "error" ? "none" : "error")}
              title={t("results.toolbar.errorHeatmap")}>
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
            </button>
            {heatmapMode === "class" && (
              <select className="heatmap-class-select" style={{ fontSize: 10, width: 28, padding: 1 }}
                value={heatmapClassId} onChange={(e) => setHeatmapClassId(parseInt(e.target.value, 10))}>
                {effectiveClasses.map((c) => (<option key={c.id} value={c.id}>{c.name}</option>))}
              </select>
            )}
          </>
        )}
        {/* Detection highlight (instance runs): blue wash + per-object colors */}
        {isInstanceRun && (
          <button className={`tool-icon ${instanceHighlight ? "active" : ""}`}
            onClick={() => setInstanceHighlight((v) => !v)}
            title={t("results.toolbar.detectionHighlight")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="7" cy="8" r="3" fill="currentColor" opacity=".5" stroke="none" />
              <circle cx="16" cy="9" r="4" fill="currentColor" opacity=".25" stroke="none" />
              <circle cx="11" cy="16" r="3.5" fill="currentColor" opacity=".7" stroke="none" />
              <circle cx="7" cy="8" r="3" /><circle cx="16" cy="9" r="4" /><circle cx="11" cy="16" r="3.5" />
            </svg>
          </button>
        )}
        {/* Measure */}
        <button className={`tool-icon ${showCount ? "active" : ""}`}
          onClick={() => setShowCount((v) => !v)} title={t("results.toolbar.regionCount")}>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M5 3h2v18H5zm6 6h2v12h-2zm6-3h2v15h-2z"/></svg>
        </button>
        <button className={`tool-icon ${showArea ? "active" : ""}`}
          onClick={() => setShowArea((v) => !v)} title={calibration ? t("results.toolbar.areaUnit").replace("{unit}", calibration.unit) : t("results.toolbar.areaPx")}>
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="1" /><rect x="8" y="8" width="8" height="8" fill="currentColor" opacity=".3" stroke="none" /></svg>
        </button>
        {calibration && showArea && (
          <div className="left-toolbar-info" style={{ cursor: "pointer", color: "#4fc3f7" }} onClick={() => setCalibration(null)} title={t("results.toolbar.reset")}>
            {calibration.unit}
          </div>
        )}
        {/* Class legend */}
        {effectiveClasses.filter((c) => c.id > 0).length > 0 && (
          <div style={{ marginTop: 4, display: "flex", flexDirection: "column", gap: 2 }}>
            {effectiveClasses.filter((c) => c.id > 0).map((cls) => {
              const hidden = hiddenClassIds.has(cls.id);
              return (
                <button key={cls.id} className="tool-icon"
                  style={{ display: "flex", alignItems: "center", gap: 4, padding: "2px 4px", opacity: hidden ? 0.35 : 1, width: "auto" }}
                  title={(hidden ? t("results.toolbar.classShow") : t("results.toolbar.classHide")).replace("{name}", cls.name)}
                  onClick={() => setHiddenClassIds((prev) => { const next = new Set(prev); if (next.has(cls.id)) next.delete(cls.id); else next.add(cls.id); return next; })}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, flexShrink: 0, background: `rgb(${cls.color[0]},${cls.color[1]},${cls.color[2]})` }} />
                  <span style={{ fontSize: 10, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 60 }}>{cls.name}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
});
