// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState, useEffect, useRef } from "react";
import type { Tool } from "../../store";
import type { Recipe } from "../../api";
import { samListModels } from "../../api";
import type { SpotDetectRefValue, SamRefValue, SuperpixelRefValue, CrackTraceRefValue } from "../hooks/useDrawingEvents";
import type { SamModelId } from "../annotatorContext";
import { refilterSpot, refilterColorSpot } from "../hooks/toolActions";
import { useI18n } from "../../i18n";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AiAssistPanelProps = {
  // general state
  projectId: string | null;
  activeImageId: string | null;
  tool: Tool;
  width: number;
  height: number;
  activeClassId: number;

  // recipe state
  activeRecipe: Recipe | null;
  recipePreview: Uint8Array | null;
  isRecipeRunning: boolean;
  handleRecipePreview: () => void;
  handleAutoLabelPreview: (classId: number) => void;
  handleRecipeConfirm: () => void;
  handleRecipeCancel: () => void;
  handleRecipeApplyAll: () => void;

  // SAM state
  samModel: SamModelId;
  setSamModel: (v: SamModelId) => void;
  samRefCurrent: SamRefValue;
  handleSamConfirm: () => void;
  handleSamCancel: () => void;

  // Spot Detect state
  spotSensitivity: number;
  setSpotSensitivity: (v: number) => void;
  spotCount: number;
  setSpotCount: (v: number) => void;
  spotDetectRefCurrent: SpotDetectRefValue;
  spotPhase: "idle" | "sample" | "detect";
  handleSpotConfirm: () => void;
  handleSpotCancel: () => void;
  handleSpotRunDetect: () => void;
  colorTolerance: number;
  setColorTolerance: (v: number) => void;

  assistPreview: Uint8Array | null;
  setAssistPreview: (v: Uint8Array | null) => void;
  setStatus: (msg: string) => void;

  // Superpixel state
  superpixelRefCurrent: SuperpixelRefValue;
  nSegments: number;
  setNSegments: (v: number) => void;
  handleSuperpixelConfirm: () => void;
  handleSuperpixelCancel: () => void;
  handleSuperpixelRecompute: () => void;

  // Crack Trace state
  crackTraceRefCurrent: CrackTraceRefValue;
  crackSensitivity: number;
  setCrackSensitivity: (v: number) => void;
  crackWidth: number;
  setCrackWidth: (v: number) => void;
  handleCrackConfirm: () => void;
  handleCrackCancel: () => void;
  handleCrackRecompute: () => void;
};

// ---------------------------------------------------------------------------
// CrackTracePanel (debounced sliders)
// ---------------------------------------------------------------------------

function CrackTracePanel({ crackTraceRefCurrent, crackSensitivity, setCrackSensitivity, crackWidth, setCrackWidth, handleCrackConfirm, handleCrackCancel, handleCrackRecompute, assistPreview }: {
  crackTraceRefCurrent: NonNullable<CrackTraceRefValue> | null;
  crackSensitivity: number; setCrackSensitivity: (v: number) => void;
  crackWidth: number; setCrackWidth: (v: number) => void;
  handleCrackConfirm: () => void; handleCrackCancel: () => void;
  handleCrackRecompute: () => void; assistPreview: Uint8Array | null;
}) {
  const recomputeTimerRef = useRef<number | null>(null);
  const recomputeFnRef = useRef(handleCrackRecompute);
  recomputeFnRef.current = handleCrackRecompute;

  useEffect(() => () => {
    if (recomputeTimerRef.current !== null) window.clearTimeout(recomputeTimerRef.current);
  }, []);

  const scheduleRecompute = () => {
    if (!crackTraceRefCurrent || crackTraceRefCurrent.loading) return;
    if (recomputeTimerRef.current !== null) window.clearTimeout(recomputeTimerRef.current);
    recomputeTimerRef.current = window.setTimeout(() => {
      recomputeTimerRef.current = null;
      recomputeFnRef.current();
    }, 300);
  };

  const { t } = useI18n();
  const nCracks = crackTraceRefCurrent?.nCracks ?? 0;
  return (
    <div>
      <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>{t("aiAssist.crackTrace")}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div className="muted" style={{ fontSize: 11 }}>{t("aiAssist.sensitivity")} <span style={{ float: "right" }}>{crackSensitivity}</span></div>
        <input type="range" min={1} max={100} step={1} value={crackSensitivity} aria-label="Crack sensitivity" onPointerDown={(e) => e.stopPropagation()} onChange={(e) => { setCrackSensitivity(parseInt(e.target.value)); scheduleRecompute(); }} />
        <div className="muted" style={{ fontSize: 11 }}>{t("aiAssist.width")} <span style={{ float: "right" }}>{crackWidth}px</span></div>
        <input type="range" min={0} max={20} step={1} value={crackWidth} aria-label="Crack width" onPointerDown={(e) => e.stopPropagation()} onChange={(e) => { setCrackWidth(parseInt(e.target.value)); scheduleRecompute(); }} />
        <div className="muted" style={{ fontSize: 11 }}>
          {!crackTraceRefCurrent
            ? "Click image to detect cracks"
            : crackTraceRefCurrent.loading
              ? "Recomputing..."
              : `${crackTraceRefCurrent.selections.size}/${nCracks} selected — Left: select, Right/Shift: deselect`}
        </div>
        {crackTraceRefCurrent && !crackTraceRefCurrent.loading && (
          <div style={{ display: "flex", gap: 4 }}>
            <button className="primary" style={{ flex: 1 }} onClick={handleCrackConfirm} disabled={!assistPreview || crackTraceRefCurrent.selections.size === 0} data-desc={t("aiAssist.crackConfirm.desc")}>{t("aiAssist.confirm")}</button>
            <button className="ghost" style={{ flex: 1 }} onClick={handleCrackCancel} data-desc={t("aiAssist.crackCancel.desc")}>{t("aiAssist.cancel")}</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const AiAssistPanel = React.memo(function AiAssistPanel({
  projectId: _projectId, activeImageId, tool, width, height, activeClassId,
  activeRecipe, recipePreview, isRecipeRunning,
  handleRecipePreview, handleAutoLabelPreview, handleRecipeConfirm, handleRecipeCancel, handleRecipeApplyAll,
  samModel, setSamModel, samRefCurrent,
  handleSamConfirm, handleSamCancel,
  spotSensitivity, setSpotSensitivity, spotCount, setSpotCount,
  spotDetectRefCurrent, spotPhase,
  handleSpotConfirm, handleSpotCancel, handleSpotRunDetect,
  colorTolerance, setColorTolerance,
  assistPreview, setAssistPreview, setStatus,
  superpixelRefCurrent, nSegments, setNSegments,
  handleSuperpixelConfirm, handleSuperpixelCancel, handleSuperpixelRecompute,
  crackTraceRefCurrent, crackSensitivity, setCrackSensitivity, crackWidth, setCrackWidth,
  handleCrackConfirm, handleCrackCancel, handleCrackRecompute,
}: AiAssistPanelProps) {
  const { t } = useI18n();
  const samModelLabels: Record<SamModelId, string> = {
    mobile_sam: "MobileSAM", sam2_tiny: "SAM2 Tiny", sam2_small: "SAM2 Small",
    tinysam: "TinySAM", efficient_sam_ti: "EfficientSAM-Ti",
  };
  const samModelInfo: Record<string, { label: string; desc: string }> = Object.fromEntries(
    (Object.keys(samModelLabels) as SamModelId[]).map((k) => [k, { label: samModelLabels[k], desc: t(`sam.${k}`) }])
  );
  const [samModels, setSamModels] = useState<{ id: string; label: string }[]>([
    { id: "mobile_sam", label: "MobileSAM" },
    { id: "sam2_tiny", label: "SAM2 Tiny" },
  ]);
  const [samDropOpen, setSamDropOpen] = useState(false);
  const [samHover, setSamHover] = useState<string | null>(null);
  const samDropRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    samListModels().then((models) => {
      setSamModels(models.filter((m) => m.checkpoint_exists).map((m) => ({ id: m.id, label: samModelInfo[m.id]?.label ?? m.id })));
    }).catch((e: unknown) => console.warn("AiAssistPanel: SAM model list failed:", e));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // Close dropdown on outside click
  useEffect(() => {
    if (!samDropOpen) return;
    const h = (e: MouseEvent) => { if (samDropRef.current && !samDropRef.current.contains(e.target as Node)) setSamDropOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [samDropOpen]);
  return (
    <>
      {/* ---- Auto label (propose from existing annotations) ---- */}
      {!recipePreview && (
        <div>
          <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>{t("aiAssist.autoLabel")}</div>
          <button
            className="ghost"
            style={{ width: "100%" }}
            onClick={() => handleAutoLabelPreview(activeClassId)}
            disabled={!activeImageId || isRecipeRunning || !width || activeClassId <= 0}
            title={t("aiAssist.autoLabel.title")}
            data-desc={t("aiAssist.autoLabel.desc")}
          >
            {isRecipeRunning ? t("aiAssist.computing") : t("aiAssist.autoLabel")}
          </button>
        </div>
      )}

      {/* ---- Recipe ---- */}
      {(activeRecipe || recipePreview) && (
        <div>
          <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>{t("aiAssist.recipe")}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {activeRecipe && (<div className="muted" style={{ fontSize: 11 }}>Active: {activeRecipe.name} ({activeRecipe.rules.length} rules)</div>)}
            {activeRecipe && !recipePreview && (
              <div style={{ display: "flex", gap: 4 }}>
                <button className="primary" style={{ flex: 1 }} onClick={handleRecipePreview} disabled={!activeImageId || isRecipeRunning || !width} data-desc={t("aiAssist.preview.desc")}>{isRecipeRunning ? t("aiAssist.computing") : t("aiAssist.preview")}</button>
                <button className="ghost" style={{ flex: 1 }} onClick={handleRecipeApplyAll} disabled={isRecipeRunning} data-desc={t("aiAssist.applyAll.desc")}>{t("aiAssist.applyAll")}</button>
              </div>
            )}
            {recipePreview && (
              <div style={{ display: "flex", gap: 4 }}>
                <button className="primary" style={{ flex: 1 }} onClick={handleRecipeConfirm} data-desc={t("aiAssist.confirm.desc")}>{t("aiAssist.confirm")}</button>
                <button className="ghost" style={{ flex: 1 }} onClick={handleRecipeCancel} data-desc={t("aiAssist.cancel.desc")}>{t("aiAssist.cancel")}</button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---- SAM ---- */}
      {(tool === "sam" || tool === "sambox") && (
        <div>
          <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }} data-desc={t("sam.sectionDesc")}>{t("aiAssist.samSegment")}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="muted" style={{ fontSize: 11 }}>{t("aiAssist.model")}</div>
            <div ref={samDropRef} style={{ position: "relative" }} onPointerDown={(e) => e.stopPropagation()}>
              <button
                onClick={() => setSamDropOpen(!samDropOpen)}
                aria-expanded={samDropOpen}
                aria-haspopup="listbox"
                aria-label="SAM model selection"
                style={{ width: "100%", fontSize: 12, padding: "3px 6px", background: "#333", color: "#eee", border: "1px solid #555", borderRadius: 3, textAlign: "left", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center" }}
              >
                <span>{samModelInfo[samModel]?.label ?? samModel}</span>
                <span style={{ fontSize: 9, marginLeft: 4, opacity: .5 }}>{samDropOpen ? "\u25B2" : "\u25BC"}</span>
              </button>
              {samDropOpen && (
                <div role="listbox" aria-label="SAM models" style={{ position: "absolute", top: "100%", left: 0, right: 0, zIndex: 100, background: "#2a2a2a", border: "1px solid #555", borderRadius: 4, marginTop: 2, boxShadow: "0 4px 12px rgba(0,0,0,.5)", overflow: "visible" }}>
                  {samModels.map((m) => (
                    <div
                      key={m.id}
                      role="option"
                      aria-selected={samModel === m.id}
                      tabIndex={0}
                      onClick={() => { setSamModel(m.id as SamModelId); setSamDropOpen(false); setSamHover(null); }}
                      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSamModel(m.id as SamModelId); setSamDropOpen(false); setSamHover(null); } }}
                      onMouseEnter={() => setSamHover(m.id)}
                      onMouseLeave={() => setSamHover(null)}
                      style={{ padding: "5px 8px", fontSize: 12, cursor: "pointer", background: samModel === m.id ? "rgba(79,195,247,.15)" : samHover === m.id ? "rgba(255,255,255,.07)" : "transparent", color: samModel === m.id ? "#4fc3f7" : "#eee", position: "relative" }}
                    >
                      {m.label}
                      {samHover === m.id && samModelInfo[m.id] && (
                        <div style={{ position: "absolute", bottom: "calc(100% + 6px)", left: 0, right: 0, background: "#1a1a2e", color: "#cde", border: "1px solid #4fc3f7", borderRadius: 6, padding: "6px 10px", fontSize: 11, lineHeight: 1.4, zIndex: 101, boxShadow: "0 2px 8px rgba(0,0,0,.4)", pointerEvents: "none" }}>
                          {samModelInfo[m.id].desc}
                          <div style={{ position: "absolute", bottom: -5, left: 16, width: 8, height: 8, background: "#1a1a2e", borderRight: "1px solid #4fc3f7", borderBottom: "1px solid #4fc3f7", transform: "rotate(45deg)" }} />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="muted" style={{ fontSize: 11 }}>{tool === "sam" ? t("aiAssist.samFgBg") : t("aiAssist.samBox")}</div>
            {samRefCurrent && (
              <div style={{ display: "flex", gap: 4 }}>
                <button className="primary" style={{ flex: 1 }} onClick={handleSamConfirm} disabled={!assistPreview} data-desc={t("aiAssist.samConfirm.desc")}>{t("aiAssist.confirm")}</button>
                <button className="ghost" style={{ flex: 1 }} onClick={handleSamCancel} data-desc={t("aiAssist.samCancel.desc")}>{t("aiAssist.cancel")}</button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---- Spot Detect ---- */}
      {tool === "spotdetect" && (
        <div>
          <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>{t("aiAssist.spotDetect")}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {/* Phase: sample — user paints spot samples */}
            {spotPhase !== "detect" && (
              <>
                <div className="muted" style={{ fontSize: 11 }}>
                  {spotPhase === "sample" ? t("annotate.spot.sampleDone") : t("annotate.spot.paintFirst")}
                </div>
                <button className="primary" onClick={handleSpotRunDetect} disabled={spotPhase === "idle"}>
                  {t("annotate.spot.detect")}
                </button>
              </>
            )}
            {/* Phase: detect — results shown, can refine */}
            {spotPhase === "detect" && (
              <>
                <div className="muted" style={{ fontSize: 11 }}>
                  {spotDetectRefCurrent?.mode === "color" ? "ΔE Tolerance" : t("aiAssist.sensitivity")}
                  <span style={{ float: "right" }}>{spotSensitivity.toFixed(1)}</span>
                </div>
                <input type="range"
                  min={spotDetectRefCurrent?.mode === "color" ? 3 : 1}
                  max={spotDetectRefCurrent?.mode === "color" ? 80 : 50}
                  step={spotDetectRefCurrent?.mode === "color" ? 1 : 0.5}
                  value={spotSensitivity}
                  aria-label="Spot detection sensitivity"
                  onPointerDown={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setSpotSensitivity(val);
                    if (!spotDetectRefCurrent) return;
                    const rw = spotDetectRefCurrent.downscaleWidth ?? width;
                    const rh = spotDetectRefCurrent.downscaleHeight ?? height;
                    let result: { mask: Uint8Array; count: number };
                    if (spotDetectRefCurrent.mode === "color" && spotDetectRefCurrent.colorDistMap) {
                      result = refilterColorSpot(spotDetectRefCurrent.colorDistMap, rw, rh, val, activeClassId, spotDetectRefCurrent.sizeRange);
                    } else {
                      result = refilterSpot(spotDetectRefCurrent.scores, spotDetectRefCurrent.std, spotDetectRefCurrent.extrema, rw, rh, val, activeClassId, spotDetectRefCurrent.imageData, spotDetectRefCurrent.targetLab, colorTolerance, spotDetectRefCurrent.clickIdx, spotDetectRefCurrent.sizeRange);
                    }
                    if (spotDetectRefCurrent.downscaleWidth) {
                      const full = new Uint8Array(width * height);
                      const sx = width / rw, sy = height / rh;
                      for (let y = 0; y < height; y++) {
                        const srcY = Math.min(rh - 1, Math.round(y / sy));
                        for (let x = 0; x < width; x++) {
                          full[y * width + x] = result.mask[srcY * rw + Math.min(rw - 1, Math.round(x / sx))];
                        }
                      }
                      setAssistPreview(full);
                    } else {
                      setAssistPreview(result.mask);
                    }
                    setSpotCount(result.count);
                    setStatus(`Spot Detect: ${result.count} spots`);
                  }}
                />
                <div className="muted" style={{ fontSize: 11 }}>
                  {t("annotate.spot.result").replace("{n}", String(spotCount))}
                </div>
                <div style={{ display: "flex", gap: 4 }}>
                  <button className="primary" style={{ flex: 1 }} onClick={handleSpotConfirm} disabled={!assistPreview} data-desc={t("aiAssist.detectConfirm.desc")}>{t("aiAssist.confirm")}</button>
                  <button className="ghost" style={{ flex: 1 }} onClick={handleSpotCancel} data-desc={t("aiAssist.detectCancel.desc")}>{t("aiAssist.cancel")}</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ---- Superpixel ---- */}
      {tool === "superpixel" && (
        <div>
          <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>{t("aiAssist.superpixel")}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="muted" style={{ fontSize: 11 }}>{t("aiAssist.segments")} <span style={{ float: "right" }}>{nSegments}</span></div>
            <input type="range" min={200} max={2000} step={50} value={nSegments} aria-label="Superpixel segments" onPointerDown={(e) => e.stopPropagation()} onChange={(e) => setNSegments(parseInt(e.target.value))} />
            <div className="muted" style={{ fontSize: 11 }}>
              {!superpixelRefCurrent
                ? t("aiAssist.clickSuperpixel")
                : superpixelRefCurrent.loading
                  ? t("aiAssist.computing")
                  : `${superpixelRefCurrent.selections.size} segments selected — Left: select, Right/Shift: deselect`}
            </div>
            {superpixelRefCurrent && !superpixelRefCurrent.loading && (
              <div style={{ display: "flex", gap: 4 }}>
                <button className="primary" style={{ flex: 1 }} onClick={handleSuperpixelConfirm} disabled={!assistPreview || superpixelRefCurrent.selections.size === 0} data-desc={t("aiAssist.superpixelConfirm.desc")}>{t("aiAssist.confirm")}</button>
                <button className="ghost" style={{ flex: 1 }} onClick={handleSuperpixelCancel} data-desc={t("aiAssist.superpixelCancel.desc")}>{t("aiAssist.cancel")}</button>
              </div>
            )}
            {superpixelRefCurrent && !superpixelRefCurrent.loading && (
              <button className="ghost" style={{ width: "100%", fontSize: 11 }} onClick={handleSuperpixelRecompute} data-desc={t("aiAssist.recompute.desc")}>{t("aiAssist.recompute")}</button>
            )}
          </div>
        </div>
      )}

      {/* ---- Crack Trace ---- */}
      {tool === "cracktrace" && <CrackTracePanel
        crackTraceRefCurrent={crackTraceRefCurrent}
        crackSensitivity={crackSensitivity} setCrackSensitivity={setCrackSensitivity}
        crackWidth={crackWidth} setCrackWidth={setCrackWidth}
        handleCrackConfirm={handleCrackConfirm} handleCrackCancel={handleCrackCancel}
        handleCrackRecompute={handleCrackRecompute} assistPreview={assistPreview}
      />}

    </>
  );
});
