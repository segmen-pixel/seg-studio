// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState, useEffect, useCallback, useRef } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../../i18n";
import { TOOL_LABELS } from "../annotatorTypes";
import type { ToolId } from "../annotatorTypes";
import { ToolIcon } from "../ToolIcon";
import type { Tool } from "../../store";
import { TiledViewer } from "./TiledViewer";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type RegionLabel = {
  classId: number;
  cx: number;
  topY: number;
  count: number;
  name: string;
  color: [number, number, number];
};

export type CanvasAreaProps = {
  // canvas refs
  containerRef: React.Ref<HTMLDivElement>;
  imageCanvasRef: React.Ref<HTMLCanvasElement>;
  overlayCanvasRef: React.Ref<HTMLCanvasElement>;
  uiCanvasRef: React.Ref<HTMLCanvasElement>;

  // pointer handlers
  handlePointerDown: (e: React.PointerEvent) => void;
  handlePointerMove: (e: React.PointerEvent) => void;
  handlePointerUp: (e: React.PointerEvent) => void;

  // tool state
  tool: Tool;
  setTool: (tool: Tool) => void;
  brushSize: number;
  setBrushSize: (v: number) => void;

  // viewport state
  width: number;
  height: number;
  scale: number;
  offsetX: number;
  offsetY: number;
  isPanning: boolean;
  spacePressed: boolean;

  // image adjustments
  imageBrightness: number;
  setImageBrightness: (v: number) => void;
  imageContrast: number;
  setImageContrast: (v: number) => void;

  // overlay
  overlayAlpha: number;
  setOverlayAlpha: (v: number) => void;
  setView: (scale: number, offsetX: number, offsetY: number) => void;

  // region labels
  regionLabels: RegionLabel[];

  // Measure
  measureDistance: string | null;
  setMeasureStart: (v: [number, number] | null) => void;
  setMeasureEnd: (v: [number, number] | null) => void;

  // SAM / SpotDetect / CrackTrace cancel (for tool switch)
  handleSamCancel: () => void;
  handleSpotCancel: () => void;
  handleCrackCancel: () => void;
  samRefCurrent: unknown;
  spotDetectRefCurrent: unknown;
  crackTraceRefCurrent: unknown;

  // overlay toggle
  overlayVisible: boolean;
  setOverlayVisible: (v: boolean) => void;
  prevOverlayAlphaRef: React.MutableRefObject<number>;

  // actions
  handleUndo: () => void;
  handleRedo: () => void;
  handleFit: () => void;

  // recipe import (context menu)
  recipeInputRef: React.RefObject<HTMLInputElement | null>;
  handleRecipeImport: (file: File) => void;
  projectId: string | null;
  isRecipeRunning: boolean;
  onMarkClean?: () => void;

  activeImageId: string | null;
  activeImageName: string | null;

  // description mode
  descMode: boolean;
  gpuBusy: boolean;

  // Tiled viewer (for large images)
  dziUrl?: string | null;
  maskTileBaseUrl?: string | null;
  lut: Uint8ClampedArray;
  activeClassId: number;
};

// ---------------------------------------------------------------------------
// Tool descriptions (shown in desc-mode)
// ---------------------------------------------------------------------------
// TOOL_DESCS keys match i18n "toolDesc.<id>"
const TOOL_DESC_IDS = ["brush", "eraser", "move", "wand", "sam", "sambox", "spotdetect", "superpixel", "cracktrace"] as const;

// ---------------------------------------------------------------------------
// Generic slider popover (opacity, zoom, etc.)
// ---------------------------------------------------------------------------
function SliderPopover({ value, min, max, displayValue, icon, onChange, title }: {
  value: number; min: number; max: number;
  displayValue: string;
  icon: React.ReactNode;
  onChange: (v: number) => void;
  title: string;
}) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const hideTimer = useRef<ReturnType<typeof setTimeout>>();

  const show = useCallback(() => {
    clearTimeout(hideTimer.current);
    const el = triggerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    // popup to the right of trigger (image side)
    setPos({ top: rect.top + rect.height / 2, left: rect.right + 8 });
    setOpen(true);
  }, []);
  const hide = useCallback(() => { hideTimer.current = setTimeout(() => setOpen(false), 200); }, []);
  const cancelHide = useCallback(() => { clearTimeout(hideTimer.current); }, []);

  return (
    <>
      <div className="slider-popover-wrap" onPointerEnter={show} onPointerLeave={hide}
           onPointerDown={(e) => e.stopPropagation()}>
        <button ref={triggerRef} className="tool-icon slider-popover-trigger" type="button" title={title}>
          {icon}
          <span className="slider-popover-val">{displayValue}</span>
        </button>
      </div>
      {open && createPortal(
        <div className="slider-popover-popup" style={{ top: pos.top, left: pos.left }}
             onPointerEnter={cancelHide} onPointerLeave={hide} onPointerDown={(e) => e.stopPropagation()}>
          <input type="range" min={min} max={max} value={value}
                 onChange={(e) => onChange(parseInt(e.target.value, 10))} />
          <span className="slider-popover-popup-val">{displayValue}</span>
        </div>,
        document.body,
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Brush size popover (hover-to-expand slider)
// ---------------------------------------------------------------------------
function BrushSizePopover({ brushSize, setBrushSize, title }: {
  brushSize: number;
  setBrushSize: (v: number) => void;
  title: string;
}) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const hideTimer = useRef<ReturnType<typeof setTimeout>>();

  const show = useCallback(() => {
    clearTimeout(hideTimer.current);
    const el = triggerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    // place popup to the right of trigger (image side)
    setPos({ top: rect.top + rect.height / 2, left: rect.right + 8 });
    setOpen(true);
  }, []);

  const hide = useCallback(() => {
    hideTimer.current = setTimeout(() => setOpen(false), 200);
  }, []);

  const cancelHide = useCallback(() => {
    clearTimeout(hideTimer.current);
  }, []);

  return (
    <>
      <div
        className="brush-size-popover"
        onPointerDown={(event) => event.stopPropagation()}
        onPointerEnter={show}
        onPointerLeave={hide}
      >
        <button ref={triggerRef} className="brush-size-trigger" type="button" title={title}>
          <span
            className="brush-size-dot"
            style={{
              width: Math.max(4, Math.min(18, brushSize * 0.4)),
              height: Math.max(4, Math.min(18, brushSize * 0.4)),
            }}
          />
          <span className="brush-size-label">{brushSize}</span>
        </button>
      </div>
      {open && createPortal(
        <div
          className="brush-size-popup brush-size-popup-open"
          style={{ top: pos.top, left: pos.left }}
          onPointerEnter={cancelHide}
          onPointerLeave={hide}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <input
            type="range" min={2} max={200} value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value, 10))}
            aria-label="Brush size"
          />
          <span className="brush-size-popup-val">{brushSize}px</span>
        </div>,
        document.body,
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const CanvasArea = React.memo(function CanvasArea({
  containerRef, imageCanvasRef, overlayCanvasRef, uiCanvasRef,
  handlePointerDown, handlePointerMove, handlePointerUp,
  tool, setTool, brushSize, setBrushSize,
  width, height, scale, offsetX, offsetY,
  isPanning, spacePressed,
  imageBrightness, setImageBrightness, imageContrast, setImageContrast,
  overlayAlpha, setOverlayAlpha, setView,
  regionLabels,
  measureDistance, setMeasureStart, setMeasureEnd,
  handleSamCancel, handleSpotCancel, handleCrackCancel,
  samRefCurrent, spotDetectRefCurrent, crackTraceRefCurrent,
  overlayVisible, setOverlayVisible, prevOverlayAlphaRef,
  handleUndo, handleRedo, handleFit,
  recipeInputRef, handleRecipeImport, projectId, isRecipeRunning,
  onMarkClean,
  activeImageId,
  activeImageName,
  descMode,
  gpuBusy,
  dziUrl,
  maskTileBaseUrl,
  lut,
  activeClassId,
}: CanvasAreaProps) {
  const { lang, t } = useI18n();
  // ---- Image visibility toggle ----
  const [imageHidden, setImageHidden] = useState(false);
  const [showRegionLabels, setShowRegionLabels] = useState(true);

  // ---- Context menu for model assist button ----
  const [qlCtxMenu, setQlCtxMenu] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    if (!qlCtxMenu) return;
    const close = () => setQlCtxMenu(null);
    window.addEventListener("click", close);
    window.addEventListener("contextmenu", close);
    return () => { window.removeEventListener("click", close); window.removeEventListener("contextmenu", close); };
  }, [qlCtxMenu]);

  // ---- Context menu state ----
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);

  // Close context menu on any click outside
  useEffect(() => {
    if (!ctxMenu) return;
    const close = () => setCtxMenu(null);
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [ctxMenu]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    // SAM uses right-click for BG points, superpixel/cracktrace for deselect — let those through
    if (tool === "sam" || tool === "superpixel" || tool === "cracktrace") {
      e.preventDefault();
      return;
    }
    e.preventDefault();
    setCtxMenu({ x: e.clientX, y: e.clientY });
  }, [tool]);

  const handleToolSwitch = useCallback((id: ToolId) => {
    if (samRefCurrent && id !== "sam" && id !== "sambox") handleSamCancel();
    if (spotDetectRefCurrent && id !== "spotdetect") handleSpotCancel();
    if (crackTraceRefCurrent && id !== "cracktrace") handleCrackCancel();
    setTool(id as ToolId);
    setCtxMenu(null);
  }, [samRefCurrent, spotDetectRefCurrent, crackTraceRefCurrent, handleSamCancel, handleSpotCancel, handleCrackCancel, setTool]);

  const handleOverlayToggle = useCallback(() => {
    if (overlayVisible) {
      prevOverlayAlphaRef.current = overlayAlpha;
      setOverlayAlpha(0);
      setOverlayVisible(false);
    } else {
      setOverlayAlpha(prevOverlayAlphaRef.current || 140);
      setOverlayVisible(true);
    }
  }, [overlayVisible, overlayAlpha, setOverlayAlpha, setOverlayVisible, prevOverlayAlphaRef]);
  return (
    <div className={`canvas-area${descMode ? " desc-mode" : ""}`}>
      {activeImageName && (
        <div className="canvas-image-title" title={activeImageName}>
          {activeImageName}
          {width > 0 && height > 0 && <span className="canvas-image-size">{width}×{height}</span>}
        </div>
      )}
      <div
        className="canvas-stack"
        ref={containerRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onContextMenu={handleContextMenu}
        style={{ cursor: isPanning ? "grabbing" : spacePressed ? "grab" : tool === "move" ? "grab" : (tool === "measure" || tool === "sam" || tool === "sambox" || tool === "bucket") ? "crosshair" : "default" }}
      >
        {dziUrl ? (
          /* Large image: OpenSeadragon tiled viewer */
          <TiledViewer
            dziUrl={dziUrl}
            maskTileBaseUrl={maskTileBaseUrl ?? null}
            width={width}
            height={height}
            lut={lut}
            activeClassId={activeClassId}
            tool={tool}
            brushSize={brushSize}
          />
        ) : (
          /* Standard image: existing canvas stack */
          <div
            className="canvas-inner"
            style={{
              width: width || 0, height: height || 0,
              transform: `translate(${offsetX}px, ${offsetY}px) scale(${scale})`,
              transformOrigin: "top left",
            }}
          >
            <canvas ref={imageCanvasRef} style={{ ...(imageHidden ? { opacity: 0 } : undefined), ...((imageBrightness !== 100 || imageContrast !== 100) ? { filter: `brightness(${imageBrightness / 100}) contrast(${imageContrast / 100})` } : undefined) }} />
            <canvas ref={overlayCanvasRef} style={{ opacity: 1 }} />
            <canvas ref={uiCanvasRef} />
          </div>
        )}
        {overlayVisible && showRegionLabels && regionLabels.map((lbl, i) => {
          const imgL = offsetX;
          const imgT = offsetY;
          const imgR = offsetX + width * scale;
          const imgB = offsetY + height * scale;
          const rawX = offsetX + lbl.cx * scale;
          const rawY = offsetY + lbl.topY * scale;
          const lblH = 18;
          const lblW = 60;
          const x = Math.max(imgL + lblW / 2, Math.min(imgR - lblW / 2, rawX));
          const aboveY = rawY - lblH - 2;
          const y = aboveY >= imgT ? rawY : Math.min(imgB - lblH, rawY + lblH);
          const placeBelow = aboveY < imgT;
          return (
            <div
              key={`${lbl.classId}-${i}`}
              style={{
                position: "absolute", left: x, top: y,
                transform: placeBelow ? "translate(-50%, 2px)" : "translate(-50%, -100%) translateY(-2px)",
                fontSize: 11, fontWeight: 600, color: "#fff",
                background: `rgba(${lbl.color[0]}, ${lbl.color[1]}, ${lbl.color[2]}, 0.85)`,
                padding: "1px 5px", borderRadius: 3, pointerEvents: "none",
                whiteSpace: "nowrap", lineHeight: "16px",
                textShadow: "0 1px 2px rgba(0,0,0,0.6)",
              }}
            >
              {lbl.name} ({lbl.count.toLocaleString()}px)
            </div>
          );
        })}
        {/* Left action toolbar */}
        <div className="overlay-tools left" onPointerDown={(event) => event.stopPropagation()}>
          <button className="tool-icon" onClick={handleUndo} title="Undo" data-desc={t("canvas.undo")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M8 7L4 11l4 4" /><path d="M5 11h8a6 6 0 1 1 0 12" /></svg>
          </button>
          <button className="tool-icon" onClick={handleRedo} title="Redo" data-desc={t("canvas.redo")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M16 7l4 4-4 4" /><path d="M19 11H11a6 6 0 1 0 0 12" /></svg>
          </button>
          <button className="tool-icon" onClick={handleFit} title="Fit" data-desc={t("canvas.fit")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2" /><path d="M8 8h4M16 8h-4M8 16h4M16 16h-4" /></svg>
          </button>
          <button className={`tool-icon ${!overlayVisible ? "active" : ""}`} onClick={handleOverlayToggle}
            title={overlayVisible ? "Hide overlay" : "Show overlay"} data-desc={t("canvas.overlayToggle")}>
            {overlayVisible
              ? <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z" /><circle cx="12" cy="12" r="3" /></svg>
              : <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" /><path d="M1 1l22 22" /></svg>}
          </button>
          <button className={`tool-icon ${!showRegionLabels ? "active" : ""}`} onClick={() => setShowRegionLabels((v) => !v)}
            title={showRegionLabels ? t("canvas.hideLabels") : t("canvas.showLabels")} data-desc={t("canvas.labelsToggle")}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 7h16M4 12h10M4 17h6" />
              {!showRegionLabels && <path d="M2 2l20 20" strokeWidth="2" />}
            </svg>
          </button>
          <button className={`tool-icon ${imageHidden ? "active" : ""}`} onClick={() => setImageHidden((v) => !v)}
            title={imageHidden ? "Show image" : "Hide image"} data-desc={t("canvas.imageToggle")}>
            {imageHidden
              ? <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 3l18 18" strokeWidth="2" /></svg>
              : <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="M21 15l-5-5L5 21" /></svg>}
          </button>
          <SliderPopover value={imageBrightness} min={50} max={300}
            displayValue={`${imageBrightness}%`}
            icon={<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" /></svg>}
            onChange={(v) => setImageBrightness(v)}
            title={t("bottomBar.brightnessContrast")} />
          <SliderPopover value={overlayAlpha} min={0} max={220}
            displayValue={`${Math.round((overlayAlpha / 255) * 100)}%`}
            icon={<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" opacity="0.4" /><path d="M12 2a10 10 0 0 1 0 20" fill="currentColor" stroke="none" /></svg>}
            onChange={(v) => { setOverlayAlpha(v); if (v > 0 && !overlayVisible) setOverlayVisible(true); if (v === 0 && overlayVisible) setOverlayVisible(false); }}
            title={t("bottomBar.opacity")} />
          <SliderPopover value={Math.round(scale * 100)} min={25} max={400}
            displayValue={`${Math.round(scale * 100)}%`}
            icon={<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" /><path d="M11 8v6M8 11h6" /></svg>}
            onChange={(v) => setView(v / 100, offsetX, offsetY)}
            title={t("bottomBar.zoom")} />
          {width > 0 && height > 0 && (
            <div className="left-toolbar-info">{width}&times;{height}</div>
          )}
        </div>
        <div className="overlay-tools right" data-tutorial-step="annotate-tools">
          {TOOL_LABELS.filter((item) => item.id !== "bucket" && item.id !== "measure").map((item) => {
            const needsGpu = item.id === "sam" || item.id === "sambox" || item.id === "cracktrace" || item.id === "superpixel";
            const blocked = needsGpu && gpuBusy;
            const toolDescId = TOOL_DESC_IDS.find((id) => id === item.id);
            return (
              <button
                key={item.id}
                className={`tool-icon ${tool === item.id ? "active" : ""}${blocked ? " gpu-blocked" : ""}`}
                onClick={() => { if (!blocked) handleToolSwitch(item.id); }}
                onPointerDown={(event) => event.stopPropagation()}
                disabled={blocked}
                title={blocked ? t("canvas.gpuBlocked").replace("{tool}", lang === "ja" ? item.labelJa : item.label) : t("canvas.toolTitle").replace("{tool}", lang === "ja" ? item.labelJa : item.label).replace("{shortcut}", item.shortcut)}
                {...(toolDescId ? { "data-desc": t(`toolDesc.${toolDescId}`) } : {})}
              >
                <ToolIcon id={item.id} />
              </button>
            );
          })}
          <BrushSizePopover brushSize={brushSize} setBrushSize={setBrushSize} title={t("canvas.brushSizeDesc")} />
        </div>
      </div>
      {measureDistance && (
        <div className="section">
          <div className="section-title">{t("bottomBar.measure")}</div>
          <div className="row">
            <button className="ghost" onClick={() => { setMeasureStart(null); setMeasureEnd(null); }}>{t("bottomBar.clearMeasure")}</button>
            <div className="muted">{t("bottomBar.distance")}: {measureDistance}px</div>
          </div>
        </div>
      )}
      {/* ---- Right-click context menu ---- */}
      {ctxMenu && (
        <div
          style={{
            position: "fixed", left: ctxMenu.x, top: ctxMenu.y, zIndex: 9999,
            background: "#2a2a2a", border: "1px solid #555", borderRadius: 6,
            padding: "4px 0", minWidth: 180, boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
          }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          {TOOL_LABELS.filter((item) => item.id === "bucket" || item.id === "measure").map((item) => (
            <button
              key={item.id}
              onClick={() => handleToolSwitch(item.id)}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                width: "100%", padding: "6px 12px", border: "none",
                background: tool === item.id ? "rgba(255,255,255,0.12)" : "transparent",
                color: "#eee", fontSize: 13, cursor: "pointer", textAlign: "left",
              }}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "rgba(255,255,255,0.1)"; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = tool === item.id ? "rgba(255,255,255,0.12)" : "transparent"; }}
            >
              <span style={{ width: 20, height: 20, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
                <ToolIcon id={item.id} />
              </span>
              <span style={{ flex: 1 }}>{lang === "ja" ? item.labelJa : item.label}</span>
              <span style={{ color: "#888", fontSize: 11 }}>{item.shortcut}</span>
            </button>
          ))}
          <div style={{ borderTop: "1px solid #555", margin: "4px 0" }} />
          <input ref={recipeInputRef as React.RefObject<HTMLInputElement>} type="file" accept=".json" style={{ display: "none" }} onChange={(e) => { const file = e.target.files?.[0]; if (file) handleRecipeImport(file); e.target.value = ""; }} />
          <button
            onClick={() => { setCtxMenu(null); recipeInputRef.current?.click(); }}
            disabled={!projectId || isRecipeRunning}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              width: "100%", padding: "6px 12px", border: "none",
              background: "transparent", color: !projectId || isRecipeRunning ? "#666" : "#eee",
              fontSize: 13, cursor: !projectId || isRecipeRunning ? "default" : "pointer", textAlign: "left",
            }}
            onMouseEnter={(e) => { if (projectId && !isRecipeRunning) (e.target as HTMLElement).style.background = "rgba(255,255,255,0.1)"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
          >
            <span style={{ width: 20, height: 20, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>
            </span>
            <span style={{ flex: 1 }}>Import Recipe</span>
          </button>
          <button
            onClick={() => { setCtxMenu(null); onMarkClean?.(); }}
            disabled={!projectId}
            data-desc={t("annotate.markClean.desc")}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              width: "100%", padding: "6px 12px", border: "none",
              background: "transparent", color: !projectId ? "#666" : "#eee",
              fontSize: 13, cursor: !projectId ? "default" : "pointer", textAlign: "left",
            }}
            onMouseEnter={(e) => { if (projectId) (e.target as HTMLElement).style.background = "rgba(255,255,255,0.1)"; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
          >
            <span style={{ width: 20, height: 20, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </span>
            <span style={{ flex: 1 }}>{t("annotate.markClean")}</span>
            <span style={{ fontSize: 11, color: "#888" }}>Shift+C</span>
          </button>
        </div>
      )}
    </div>
  );
});
