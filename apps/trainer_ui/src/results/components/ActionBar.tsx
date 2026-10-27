// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

import type { OperatingSweep, SweptOperatingPoint } from "../../api";

/** One named threshold from the run's sweep (metrics.json `operating_points`). */
export type OperatingPoint = {
  threshold: number;
  rule: string;
  basis: string;
  f1: number;
  precision: number;
  recall: number;
  instance_recall?: number;
  instances_found?: number;
  instances_total?: number;
  /** Counting runs: objects agreed on, a different judgement from the
   *  defect-region coverage the instance_* fields report. */
  objects_found?: number;
  objects_total?: number;
};

/** The presets, in trade-off order: miss least -> balanced -> over-call least. */
const PRESETS = [
  { key: "recall_first", label: "results.opPreset.recall", desc: "results.opPreset.recall.desc" },
  { key: "balanced", label: "results.opPreset.balanced", desc: "results.opPreset.balanced.desc" },
  { key: "precision_first", label: "results.opPreset.precision", desc: "results.opPreset.precision.desc" },
] as const;

type Props = {
  activeRunId: string | null;
  isInferring: boolean;
  currentImageNotInferred: boolean;
  inferredRuns: Map<string, Set<string>>;
  confidenceThreshold: number;
  setConfidenceThreshold: (v: number) => void;
  operatingPoints?: Record<string, OperatingPoint> | null;
  /** Points swept over confidence AND min_area from this run's predictions.
   *  Preferred over operatingPoints, which knows only about the threshold. */
  sweptPoints?: OperatingSweep | null;
  ppMinArea: number;
  setPpMinArea: (v: number) => void;
  ppMaxArea: number;
  setPpMaxArea: (v: number) => void;
  ppApplyAll: boolean;
  handleRunInference: () => void;
  onOpenExport?: () => void;
  handleStopInference: () => void;
  handleApplyPostprocessAll: () => void;
  handleClearPostprocessAll: () => void;
  handleRestoreCache: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  t: (key: any) => string;
};

export default React.memo(function ActionBar({
  activeRunId,
  isInferring,
  currentImageNotInferred,
  inferredRuns,
  confidenceThreshold,
  setConfidenceThreshold,
  operatingPoints,
  sweptPoints,
  ppMinArea,
  setPpMinArea,
  ppMaxArea,
  setPpMaxArea,
  ppApplyAll,
  handleApplyPostprocessAll,
  handleClearPostprocessAll,
  t,
}: Props) {
  // The slider is in whole percent, the sweep in steps of 0.02, so a preset
  // lands on an integer without rounding drift.
  const pct = (op: OperatingPoint) => Math.round(op.threshold * 100);
  const available = PRESETS
    .map((p) => ({ ...p, op: operatingPoints?.[p.key] }))
    .filter((p): p is typeof p & { op: OperatingPoint } => p.op != null);

  // Defects the balanced point finds. A preset that finds fewer is trading
  // away detections, not just tightening a mask, and on an inspection line
  // that is the one cost that must not be discovered by hovering: the
  // precision-first point can reach precision 1.0 by keeping only the two
  // defects it is surest of.
  // Defect regions on a semantic run, counted objects on a counting one. The
  // two are measured differently, but the question the warning answers is the
  // same: does this preset find fewer things than balanced does. Whichever
  // the run recorded is the one read; a run with neither shows no warning
  // rather than a warning built on a zero.
  const foundOf = (op: OperatingPoint | undefined): number | undefined =>
    op == null ? undefined
      : typeof op.instances_found === "number" ? op.instances_found
      : typeof op.objects_found === "number" ? op.objects_found
      : undefined;
  const totalOf = (op: OperatingPoint): number | undefined =>
    typeof op.instances_total === "number" ? op.instances_total
      : typeof op.objects_total === "number" ? op.objects_total
      : undefined;
  const foundLabelKey = (op: OperatingPoint) =>
    typeof op.instances_total === "number"
      ? "results.metrics.defectRecall" as const
      : "results.metrics.objectsFound" as const;
  const balancedFound = foundOf(operatingPoints?.balanced);
  const missesMore = (op: OperatingPoint) => {
    const f = foundOf(op);
    return typeof balancedFound === "number" && typeof f === "number"
      && f < balancedFound;
  };

  /** What the preset costs, in the numbers it was chosen on. */
  const presetTitle = (op: OperatingPoint) => {
    const parts = [
      `thr ${op.threshold.toFixed(2)}`,
      `P ${op.precision.toFixed(3)}`,
      `R ${op.recall.toFixed(3)}`,
    ];
    const total = totalOf(op);
    if (total != null) {
      parts.push(`${t(foundLabelKey(op))} ${foundOf(op) ?? 0}/${total}`);
    }
    if (missesMore(op)) {
      const key = typeof op.instances_total === "number"
        ? "results.opPreset.missesMore" as const
        : "results.opPreset.missesMoreObjects" as const;
      parts.push(t(key).replace("{n}", String((balancedFound ?? 0) - (foundOf(op) ?? 0))));
    }
    parts.push(`F1 ${op.f1.toFixed(3)}`);
    return parts.join("  ·  ");
  };


  /** One row of buttons, whichever source produced them.
   *
   *  The swept points win wherever this run has predictions: they carry an
   *  area as well as a threshold, they are counted in defects rather than
   *  pixels, and they were measured on this run instead of on the val set
   *  during training. The metrics presets remain for a run nobody has
   *  predicted yet, and set the threshold only -- which is all they know. */
  type PresetView = {
    key: string; label: string; desc: string; title: string;
    thresholdPct: number;
    /** null when the preset has no opinion about the area filter. */
    minArea: number | null;
    lost?: number;
    atCeiling?: boolean;
    activeAtZero?: boolean;
    warn?: string | null;
  };

  const sweptViews: PresetView[] = PRESETS
    .map((p) => ({ p, sp: sweptPoints?.points?.[p.key] }))
    .filter((x): x is { p: typeof PRESETS[number]; sp: SweptOperatingPoint } => x.sp != null)
    .map(({ p, sp }) => {
      const total = sp.detected + sp.missed;
      return {
        key: p.key, label: p.label, desc: p.desc,
        thresholdPct: sp.threshold_pct,
        minArea: sp.min_area,
        lost: sp.defects_lost_to_area,
        // Sitting on the largest filter this threshold may carry. The server
        // decides it against its own ladder; recomputing it here from
        // min_area >= ceiling read "not at the cap" for the setting that was
        // in fact the largest one on offer.
        atCeiling: sp.at_area_ceiling,
        warn: total > 0 && sp.detected < total
          ? `${t("results.metrics.defectRecall")} ${sp.detected}/${total}` : null,
        title: [
          `thr ${sp.threshold_pct}%`,
          `min area ${sp.min_area}px`,
          `P ${sp.precision.toFixed(3)}`,
          `F1 ${sp.f1.toFixed(3)}`,
          `${t("results.metrics.defectRecall")} ${sp.detected}/${total}`,
          `${t("results.verdict.over")} ${sp.over}`,
        ].join("  \u00b7  "),
      };
    });

  const metricViews: PresetView[] = available.map(({ key, label, desc, op }) => ({
    key, label, desc,
    thresholdPct: pct(op),
    minArea: null,
    activeAtZero: key === "balanced",
    warn: missesMore(op) ? `${foundOf(op)}/${totalOf(op)}` : null,
    title: presetTitle(op),
  }));

  const views = sweptViews.length > 0 ? sweptViews : metricViews;

  return (
    <>
      {/* Named operating points. Absent on runs with neither a recorded sweep
          nor any predictions, in which case only the raw slider shows. */}
      {views.length > 0 && (
        <div style={{ marginBottom: 6 }}>
          <div
            data-desc={t("results.opPreset.desc")}
            style={{ fontSize: 11, fontWeight: 600, color: "var(--ink)", marginBottom: 4 }}
          >
            {t("results.opPreset.title")}
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {views.map((v) => {
              // A slider at 0 means "the threshold the run ships", which is the
              // balanced point by construction -- so show it as the active one
              // rather than as no selection at all.
              const active = (confidenceThreshold === v.thresholdPct
                  || (confidenceThreshold === 0 && v.activeAtZero))
                && (v.minArea == null || ppMinArea === v.minArea);
              return (
                <button
                  key={v.key}
                  onClick={() => {
                    setConfidenceThreshold(v.thresholdPct);
                    // A preset that set only the threshold would leave whatever
                    // area filter happened to be in the box multiplying into
                    // the numbers it was chosen on.
                    if (v.minArea != null) setPpMinArea(v.minArea);
                  }}
                  data-desc={t(v.desc)}
                  title={v.title}
                  style={{
                    flex: 1,
                    fontSize: 11,
                    padding: "4px 2px",
                    cursor: "pointer",
                    borderRadius: 4,
                    border: active ? "2px solid var(--accent)" : "1px solid #555",
                    fontWeight: active ? 700 : 400,
                  }}
                >
                  {t(v.label)}
                  {v.minArea != null && v.minArea > 0 && (
                    <span style={{ display: "block", fontSize: 10 }}>
                      {"\u2265"}{v.minArea}px
                    </span>
                  )}
                  {v.warn && (
                    /* Marked by symbol AND text, never by colour alone. */
                    <span style={{ display: "block", fontSize: 10, fontWeight: 700 }}>
                      {"\u26A0 "}{v.warn}
                    </span>
                  )}
                  {typeof v.lost === "number" && v.lost > 0 && (
                    <span style={{ display: "block", fontSize: 10, fontWeight: 700 }}>
                      {"\u26A0 "}
                      {t("results.opPreset.defectsLost").replace("{n}", String(v.lost))}
                    </span>
                  )}
                  {v.atCeiling && (
                    <span style={{ display: "block", fontSize: 10 }}>
                      {t("results.opPreset.areaCapped")}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          {sweptViews.length > 0 && sweptPoints && (
            <div className="muted" style={{ fontSize: 10, marginTop: 3, lineHeight: 1.35 }}>
              {t("results.opPreset.basis")
                .replace("{n}", String(sweptPoints.annotated_images))}
              {!sweptPoints.guarded && ` ${t("results.opPreset.unguarded")}`}
            </div>
          )}
        </div>
      )}
      {/* Confidence threshold slider */}
        <div style={{ marginBottom: 4, fontSize: 11 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
            <span data-desc={t("results.confidence.desc")} style={{ fontSize: 15, fontWeight: 600, color: "var(--ink)" }}>Confidence</span>
            <span style={{ fontSize: 16, fontWeight: 700, color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>{confidenceThreshold}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseInt(e.target.value, 10))}
            style={{ width: "100%" }}
          />
        </div>
      {/* Size filter. Open, and next to Confidence rather than folded behind a
          summary: it is not an advanced option, it is the other half of the
          operating point. The presets set it, and a preset that moves a
          control nobody can see is a preset that changes something in secret.
          Two columns because min and max are read as a pair. */}
        <div style={{ marginBottom: 8 }}>
          <div
            data-desc={t("results.sizeFilter.desc")}
            style={{ fontSize: 15, fontWeight: 600, color: "var(--ink)", marginBottom: 4 }}
          >
            {t("results.sizeFilter")}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
              <span className="muted" style={{ fontSize: 11 }} data-desc={t("results.ppMinArea.desc")}>
                Min (px)
              </span>
              <input
                type="number" min={0} step={10} value={ppMinArea}
                onChange={(e) => setPpMinArea(Math.max(0, parseInt(e.target.value, 10) || 0))}
                style={{ width: "100%", fontSize: 12, padding: "4px 8px", minWidth: 0 }}
              />
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
              <span className="muted" style={{ fontSize: 11 }} data-desc={t("results.ppMaxArea.desc")}>
                Max (px)
              </span>
              <input
                type="number" min={0} step={10} value={ppMaxArea}
                onChange={(e) => setPpMaxArea(Math.max(0, parseInt(e.target.value, 10) || 0))}
                style={{ width: "100%", fontSize: 12, padding: "4px 8px", minWidth: 0 }}
              />
            </label>
          </div>
          {/* The filter is live for the image on screen; this is the batch
              path, which is why it still says "All". */}
          <button
            className={`ghost${ppApplyAll ? " active" : ""}`}
            style={{ fontSize: 11, width: "100%", marginTop: 6 }}
            onClick={ppApplyAll ? handleClearPostprocessAll : handleApplyPostprocessAll}
            data-desc={t("results.ppApply.desc")}
          >
            {ppApplyAll ? "Applied (All)" : "Apply (All)"}
          </button>
        </div>
      {/* Not-inferred CTA state */}
      {currentImageNotInferred && !isInferring && (
        <div className="state-card results-not-inferred-card" data-testid="not-inferred-state" style={{ marginBottom: 12 }}>
          {activeRunId && (inferredRuns.get(activeRunId)?.size ?? 0) > 0 ? (
            <>
              <div className="state-card-title">{t("results.cachedTitle")}</div>
              <div className="state-card-copy">
                {t("results.cachedCopy")}
              </div>
            </>
          ) : (
            <>
              <div className="state-card-title">{t("results.notInferredTitle")}</div>
              <div className="state-card-copy">
                {t("results.notInferredCopy")}
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
});
