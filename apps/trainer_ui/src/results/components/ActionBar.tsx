// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

type Props = {
  activeRunId: string | null;
  isInferring: boolean;
  currentImageNotInferred: boolean;
  inferredRuns: Map<string, Set<string>>;
  confidenceThreshold: number;
  setConfidenceThreshold: (v: number) => void;
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
  ppMinArea,
  setPpMinArea,
  ppMaxArea,
  setPpMaxArea,
  ppApplyAll,
  handleRunInference,
  onOpenExport,
  handleStopInference,
  handleApplyPostprocessAll,
  handleClearPostprocessAll,
  handleRestoreCache,
  t,
}: Props) {
  return (
    <>
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
      {/* Post-Processing */}
        <details style={{ marginBottom: 8 }}>
          <summary style={{ fontSize: 12, cursor: "pointer", color: "#aaa", userSelect: "none" }}>Post-Processing</summary>
          <div className="card" style={{ flexDirection: "column", alignItems: "stretch", gap: 6, marginTop: 6, padding: "8px 10px" }}>
            <div className="row" style={{ justifyContent: "space-between", gap: 8 }}>
              <div className="muted" style={{ fontSize: 11 }} data-desc={t("results.ppMinArea.desc")}>Min Area (px)</div>
              <input type="number" min={0} step={10} value={ppMinArea} onChange={(e) => setPpMinArea(Math.max(0, parseInt(e.target.value, 10) || 0))} style={{ width: 70, fontSize: 11 }} />
            </div>
            <div className="row" style={{ justifyContent: "space-between", gap: 8 }}>
              <div className="muted" style={{ fontSize: 11 }} data-desc={t("results.ppMaxArea.desc")}>Max Area (px)</div>
              <input type="number" min={0} step={10} value={ppMaxArea} onChange={(e) => setPpMaxArea(Math.max(0, parseInt(e.target.value, 10) || 0))} style={{ width: 70, fontSize: 11 }} />
            </div>
            <div className="row" style={{ gap: 4 }}>
              <button
                className={`ghost${ppApplyAll ? " active" : ""}`}
                style={{ fontSize: 11, flex: 1 }}
                onClick={ppApplyAll ? handleClearPostprocessAll : handleApplyPostprocessAll}
                data-desc={t("results.ppApply.desc")}
              >
                {ppApplyAll ? "Applied (All)" : "Apply (All)"}
              </button>
            </div>
          </div>
        </details>
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
