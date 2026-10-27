// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState, useEffect, useRef } from "react";
import type { Tool } from "../../store";
import type { Recipe } from "../../api";
import { samListModels } from "../../api";
import { samLevelPreview } from "../hooks/toolActions";
import type { SpotDetectRefValue, SamRefValue, SuperpixelRefValue, CrackTraceRefValue } from "../hooks/useDrawingEvents";
import type { SamModelId } from "../annotatorContext";
import type { TrainRunItem } from "../../utils";
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
  handleRecipeConfirm: () => void;
  handleRecipeCancel: () => void;
  handleRecipeApplyAll: () => void;

  // pre-label state (draft from a trained run)
  prelabelRuns: TrainRunItem[];
  prelabelRunId: string | null;
  setPrelabelRunId: (runId: string) => void;
  prelabelPreviewActive: boolean;
  isPrelabelRunning: boolean;
  prelabelUnannotated: number | null;
  handlePrelabelPreview: () => void;
  handlePrelabelConfirm: () => void;
  handlePrelabelCancel: () => void;
  handlePrelabelAll: () => void;

  // SAM state
  samModel: SamModelId;
  setSamModel: (v: SamModelId) => void;
  samRefCurrent: SamRefValue;
  handleSamConfirm: () => void;
  handleSamInvert: () => void;
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
  handleSpotSensitivity: (sensitivity: number) => void;

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
  projectId: _projectId, activeImageId, tool, width, height: _height, activeClassId,
  activeRecipe, recipePreview, isRecipeRunning,
  handleRecipePreview, handleRecipeConfirm, handleRecipeCancel, handleRecipeApplyAll,
  prelabelRuns, prelabelRunId, setPrelabelRunId, prelabelPreviewActive, isPrelabelRunning,
  prelabelUnannotated, handlePrelabelPreview, handlePrelabelConfirm, handlePrelabelCancel,
  handlePrelabelAll,
  samModel, setSamModel, samRefCurrent,
  handleSamConfirm, handleSamInvert, handleSamCancel,
  spotSensitivity, setSpotSensitivity, spotCount, setSpotCount: _setSpotCount,
  spotDetectRefCurrent, spotPhase,
  handleSpotConfirm, handleSpotCancel, handleSpotRunDetect, handleSpotSensitivity,
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
  // Which of SAM's nested answers is on screen. Kept here rather than read
  // from the ref because switching has to re-render the buttons, and the ref
  // is mutated in place.
  const [samLevel, setSamLevel] = useState<string | null>(null);
  const samLevels = samRefCurrent?.levels ?? null;
  // Deliberately keyed on the candidate array alone: it is a fresh array per
  // prediction, which is exactly when the choice should reset. Adding
  // activeLevel would fire on every button press and fight the click handler.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { setSamLevel(samRefCurrent?.activeLevel ?? null); }, [samLevels]);
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
      {/* ---- Draft from a trained run ---- */}
      {prelabelRuns.length > 0 && (
        <div>
          <div className="section-title" style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: 8 }}>{t("prelabel.title")}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <select
              aria-label={t("prelabel.run")}
              value={prelabelRunId ?? ""}
              onChange={(e) => setPrelabelRunId(e.target.value)}
              disabled={isPrelabelRunning}
              style={{ fontSize: 11, width: "100%" }}
            >
              {prelabelRuns.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {(run.model_name ?? run.run_id) + (typeof run.best_f1 === "number" ? `  F1 ${run.best_f1.toFixed(3)}` : "")}
                </option>
              ))}
            </select>
            {!prelabelPreviewActive && (
              <div style={{ display: "flex", gap: 4 }}>
                <button className="primary" style={{ flex: 1 }} onClick={handlePrelabelPreview} disabled={!activeImageId || isPrelabelRunning || !width} data-desc={t("prelabel.thisImage.desc")}>{isPrelabelRunning ? t("aiAssist.computing") : t("prelabel.thisImage")}</button>
                <button className="ghost" style={{ flex: 1 }} onClick={handlePrelabelAll} disabled={isPrelabelRunning || prelabelUnannotated === 0} data-desc={t("prelabel.allImages.desc")}>{t("prelabel.allImages")}</button>
              </div>
            )}
            {prelabelPreviewActive && (
              <div style={{ display: "flex", gap: 4 }}>
                <button className="primary" style={{ flex: 1 }} onClick={handlePrelabelConfirm} data-desc={t("aiAssist.confirm.desc")}>{t("aiAssist.confirm")}</button>
                <button className="ghost" style={{ flex: 1 }} onClick={handlePrelabelCancel} data-desc={t("aiAssist.cancel.desc")}>{t("aiAssist.cancel")}</button>
              </div>
            )}
            {prelabelUnannotated !== null && (
              <div className="muted" style={{ fontSize: 11 }}>{t("prelabel.unannotated").replace("{n}", String(prelabelUnannotated))}</div>
            )}
          </div>
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
            <div className="muted" style={{ fontSize: 11 }}>{tool === "sam" ? t("aiAssist.samClick") : t("aiAssist.samBox")}</div>
            {samLevels && samLevels.length > 1 && (
              <>
                <div className="muted" style={{ fontSize: 11 }}>{t("aiAssist.samGranularity")}</div>
                <div style={{ display: "flex", gap: 4 }}>
                  {samLevels.map((lv) => (
                    <button
                      key={lv.level}
                      className={lv.level === samLevel ? "primary" : "ghost"}
                      style={{ flex: 1, fontSize: 11 }}
                      aria-pressed={lv.level === samLevel}
                      onClick={() => {
                        if (samRefCurrent) samRefCurrent.activeLevel = lv.level;
                        setSamLevel(lv.level);
                        setAssistPreview(samLevelPreview(lv.mask, activeClassId));
                        setStatus(`SAM: ${lv.level}, ${lv.area.toLocaleString()} px`);
                      }}
                      data-desc={t(`aiAssist.samLevel.${lv.level}` as Parameters<typeof t>[0])}
                    >
                      {lv.level}
                    </button>
                  ))}
                </div>
              </>
            )}
            {samRefCurrent && (
              <div className="assist-actions-row">
                <button className="primary" onClick={handleSamConfirm} disabled={!assistPreview} data-desc={t("aiAssist.samConfirm.desc")}>{t("aiAssist.confirm")}</button>
                <button className="ghost" onClick={handleSamInvert} disabled={!assistPreview} data-desc={t("aiAssist.samInvert.desc")}>{t("aiAssist.invert")}</button>
                <button className="ghost" onClick={handleSamCancel} data-desc={t("aiAssist.samCancel.desc")}>{t("aiAssist.cancel")}</button>
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
                {/* One scale in both modes: the server multiplies the spread of
                    its scores by this, so a higher value keeps fewer spots. The
                    range is the one the server holds it to, sent with the answer. */}
                <div className="muted" style={{ fontSize: 11 }}>
                  {t("aiAssist.spotThreshold")}
                  <span style={{ float: "right" }}>{spotSensitivity}</span>
                </div>
                <input type="range"
                  min={spotDetectRefCurrent?.sensitivityRange?.[0] ?? 1}
                  max={spotDetectRefCurrent?.sensitivityRange?.[1] ?? 60}
                  step={1}
                  value={spotSensitivity}
                  aria-label={t("aiAssist.spotThreshold")}
                  onPointerDown={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    setSpotSensitivity(val);
                    // The server re-thresholds from its cached score map; the
                    // hook lets the drag settle before asking.
                    handleSpotSensitivity(val);
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
