// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import type { ClassItem, MetricsPayload } from "../types";

type DetectedItem = {
  id: number;
  name: string;
  color: [number, number, number] | number[];
  ratio: number;
  count?: number;
  meanConfidence: number | null;
};

type Props = {
  visiblePredictionClasses: DetectedItem[];
  effectiveClasses: ClassItem[];
  confidenceThreshold: number;
  confidenceIndex: Uint8Array;
  maskIndex: Uint8Array;
  gtMetrics: Map<number, { f1: number; precision: number; recall: number; iou: number }> | null;
  liveStats: { fg_ratio: number; per_class_mean_confidence?: Record<string, number> } | null;
  isPredictionLoading: boolean;
  predictionLoadingLabel: string;
  notInferred: boolean;
  metrics: MetricsPayload | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  t: (key: any) => string;
};

export default React.memo(function PredictionDetail({
  visiblePredictionClasses,
  effectiveClasses,
  confidenceThreshold,
  confidenceIndex,
  maskIndex,
  gtMetrics,
  liveStats,
  isPredictionLoading,
  predictionLoadingLabel,
  notInferred,
  metrics,
  t,
}: Props) {
  return (
    <>
      {/* ---- Image Prediction ---- */}
      <div className="section" style={{ marginTop: 12 }}>
        <div className="section-title">{t("training.image")} Prediction{confidenceThreshold > 0 ? ` (≥${confidenceThreshold}%)` : ""}</div>
        {/* Pixel-level confidence histogram for current image */}
        {confidenceIndex.length > 0 && maskIndex.length > 0 && (() => {
          const BIN_COUNT = 10;
          const bins = new Array(BIN_COUNT).fill(0);
          let fgPixels = 0;
          for (let i = 0; i < maskIndex.length; i++) {
            if (maskIndex[i] === 0) continue;
            fgPixels++;
            const conf = (confidenceIndex[i] ?? 0) / 255;
            bins[Math.min(BIN_COUNT - 1, Math.floor(conf * BIN_COUNT))]++;
          }
          if (fgPixels === 0) return null;
          const maxBin = Math.max(1, ...bins);
          const threshBin = Math.floor(confidenceThreshold / 100 * BIN_COUNT);
          return (
            <div style={{ marginBottom: 8 }}>
              <div className="muted" style={{ fontSize: 10, marginBottom: 2 }}>
                {t("results.metrics.imagePixelConfDist").replace("{px}", fgPixels.toLocaleString())}
              </div>
              <div style={{ display: "flex", width: "100%", height: 56 }}>
                <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", fontSize: 8, color: "#888", width: 20, flexShrink: 0, textAlign: "right", paddingRight: 3 }}>
                  <span>{maxBin}</span><span>{Math.ceil(maxBin / 2)}</span><span>0</span>
                </div>
                <div style={{ position: "relative", flex: 1, minWidth: 0, height: "100%" }}>
                  {[0, 0.5, 1].map((v) => (
                    <div key={v} style={{ position: "absolute", left: 0, right: 0, bottom: `${v * 100}%`, borderBottom: "1px solid rgba(255,255,255,0.06)", pointerEvents: "none" }} />
                  ))}
                  {confidenceThreshold > 0 && (
                    <div style={{ position: "absolute", bottom: 0, top: 0, left: `${confidenceThreshold}%`, borderLeft: "2px dashed rgba(79,195,247,0.7)", pointerEvents: "none", zIndex: 2 }}>
                      <span style={{ position: "absolute", top: -1, left: 3, fontSize: 8, color: "#4fc3f7", whiteSpace: "nowrap" }}>{confidenceThreshold}%</span>
                    </div>
                  )}
                  <div style={{ display: "flex", alignItems: "flex-end", width: "100%", height: "100%" }}>
                    {bins.map((count, i) => {
                      const pct = (i + 0.5) * 10;
                      const h = count > 0 ? Math.max(2, (count / maxBin) * 52) : 0;
                      const color = pct >= 80 ? "#4caf50" : pct >= 50 ? "#ff9800" : "#f44336";
                      return (
                        <div key={i} style={{ flex: 1, minWidth: 0, height: h, background: color, borderRadius: 1, marginRight: 1, opacity: i < threshBin ? 0.25 : 0.8 }}
                          title={`${i * 10}%-${(i + 1) * 10}%: ${count.toLocaleString()} px`} />
                      );
                    })}
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#888", marginTop: 1, marginLeft: 20 }}>
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>
          );
        })()}
        {/* GT comparison metrics */}
        {gtMetrics && gtMetrics.size > 0 && (
          <div className="card" style={{ marginBottom: 8, flexDirection: "column", alignItems: "stretch", overflow: "visible" }}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
              <div style={{ fontWeight: 600, fontSize: 12 }}>vs GT (annotate)</div>
            </div>
            <div style={{ overflowX: "auto", minWidth: 0 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, minWidth: 220 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "2px 4px" }}>Class</th>
                    <th style={{ textAlign: "right", padding: "2px 4px" }}>F1</th>
                    <th style={{ textAlign: "right", padding: "2px 4px" }}>Prec</th>
                    <th style={{ textAlign: "right", padding: "2px 4px" }}>Rec</th>
                    <th style={{ textAlign: "right", padding: "2px 4px" }}>IoU</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from(gtMetrics.entries())
                    .sort(([a], [b]) => a - b)
                    .map(([classId, m]) => {
                      const cls = effectiveClasses.find((c) => c.id === classId);
                      return (
                        <tr key={`gt-${classId}`}>
                          <td style={{ padding: "2px 4px" }}>
                            <div className="row" style={{ gap: 4 }}>
                              <div style={{ width: 10, height: 10, borderRadius: 3, border: "1px solid var(--border)", background: cls ? `rgb(${cls.color[0]}, ${cls.color[1]}, ${cls.color[2]})` : "#888" }} />
                              {cls?.name ?? `class${classId}`}
                            </div>
                          </td>
                          <td style={{ textAlign: "right", padding: "2px 4px" }}>{m.f1.toFixed(3)}</td>
                          <td style={{ textAlign: "right", padding: "2px 4px" }}>{m.precision.toFixed(3)}</td>
                          <td style={{ textAlign: "right", padding: "2px 4px" }}>{m.recall.toFixed(3)}</td>
                          <td style={{ textAlign: "right", padding: "2px 4px" }}>{m.iou.toFixed(3)}</td>
                        </tr>
                      );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
        <div className="list">
          <>
            {visiblePredictionClasses.map((item) => {
              const gtM = gtMetrics?.get(item.id);
              return (
                <div key={item.id} className="card">
                  <div className="row" style={{ gap: 8 }}>
                    <div style={{ width: 14, height: 14, borderRadius: 4, border: "2px solid var(--border)", background: `rgb(${item.color[0]}, ${item.color[1]}, ${item.color[2]})` }} />
                    <div>{item.name}</div>
                  </div>
                  <div className="muted">
                    {"count" in item && typeof item.count === "number"
                      ? `${(item.ratio * 100).toFixed(2)}% · ${item.count.toLocaleString()}${gtM ? ` · F1 ${gtM.f1.toFixed(3)}` : ""}`
                      : `${((liveStats?.per_class_mean_confidence?.[String(item.id)] ?? item.meanConfidence ?? 0) * 100).toFixed(1)}% confidence${item.ratio > 0 ? ` · ${((liveStats?.per_class_mean_confidence ? (liveStats.fg_ratio ?? 0) : item.ratio) * 100).toFixed(2)}% FG` : ""}${gtM ? ` · F1 ${gtM.f1.toFixed(3)}` : ""}`}
                  </div>
                </div>
              );
            })}
            {visiblePredictionClasses.length === 0 && (
              isPredictionLoading ? (
                <div className="state-card" data-tone="loading">
                  <div className="state-card-title">Loading prediction details</div>
                  <div className="state-card-copy">{predictionLoadingLabel || "Fetching the score and overlay for this image."}</div>
                </div>
              ) : notInferred ? (
                <div className="state-card">
                  <div className="state-card-title">{t("results.notInferredTitle")}</div>
                  <div className="state-card-copy">
                    {t("results.notInferredCopy")}
                  </div>
                </div>
              ) : (
                // Prediction exists but nothing exceeds the threshold —
                // an all-background result, not a missing inference.
                <div className="state-card">
                  <div className="state-card-title">{t("results.noDetectionTitle")}</div>
                  <div className="state-card-copy">
                    {t("results.noDetectionCopy")}
                  </div>
                </div>
              )
            )}
          </>
        </div>
      </div>

      {/* ---- Confusion Matrix (collapsed) ---- */}
      {metrics?.metrics && (() => {
        const confusionMatrix = metrics.metrics["confusion_matrix_val"] as number[][] | undefined;
        if (!confusionMatrix || confusionMatrix.length === 0) return null;
        const n = confusionMatrix.length;
        const maxVal = Math.max(1, ...confusionMatrix.flat());
        return (
          <details>
            <summary>Confusion Matrix</summary>
            <table style={{ borderCollapse: "collapse", fontSize: 11, width: "100%", marginTop: 8 }}>
              <thead>
                <tr>
                  <th style={{ padding: "2px 4px" }}></th>
                  {Array.from({ length: n }, (_, i) => {
                    const cls = effectiveClasses.find((c) => c.id === i);
                    return <th key={`cm-h-${i}`} style={{ padding: "2px 4px", textAlign: "center" }}>{cls?.name ?? `c${i}`}</th>;
                  })}
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, ri) => {
                  const cls = effectiveClasses.find((c) => c.id === ri);
                  return (
                    <tr key={`cm-r-${ri}`}>
                      <td style={{ padding: "2px 4px", fontWeight: 600 }}>{cls?.name ?? `c${ri}`}</td>
                      {row.map((val, ci) => {
                        const ratio = val / maxVal;
                        const isDiag = ri === ci;
                        const bg = isDiag ? `rgba(0,180,80,${Math.min(0.8, ratio * 0.8)})` : `rgba(220,40,40,${Math.min(0.8, ratio * 0.8)})`;
                        return (
                          <td key={`cm-${ri}-${ci}`} style={{ textAlign: "center", padding: "2px 4px", background: val > 0 ? bg : "transparent", color: ratio > 0.5 ? "#fff" : "inherit", borderRadius: 2 }}>
                            {val > 0 ? Math.round(val) : ""}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </details>
        );
      })()}

      {/* ---- All Metrics & Config (collapsed) ---- */}
      {metrics?.metrics && (() => {
        const m = metrics.metrics;
        const hiddenKeys = new Set([
          "per_class_f1_train", "per_class_f1_val",
          "per_class_precision_train", "per_class_precision_val",
          "per_class_recall_train", "per_class_recall_val",
          "per_class_iou_train", "per_class_iou_val",
          "confusion_matrix_val",
          "dataset_stats", "auto_tuned", "class_weights",
          "best_F1_val", "best_mIoU_val", "best_epoch", "loss", "F1_val", "mIoU_val",
          // f1_curve is a {threshold, f1}[] sweep used programmatically by
          // MetricsSection for the live-F1 lookup; rendering it raw produces
          // a long "[object Object],[object Object],..." string.
          "f1_curve",
        ]);
        const entries = Object.entries(m).filter(([key]) => !hiddenKeys.has(key));

        // Render any value safely: numbers get fixed precision, primitives
        // stringify as-is, and arrays / objects fall back to compact JSON
        // (truncated) so future array-valued metrics never leak as
        // "[object Object]".
        const renderValue = (value: unknown): string => {
          if (typeof value === "number") return value.toFixed(4);
          if (value == null) return String(value);
          if (typeof value === "string" || typeof value === "boolean") return String(value);
          if (Array.isArray(value)) {
            return value.length === 0 ? "[]" : `[${value.length} items]`;
          }
          if (typeof value === "object") {
            const s = JSON.stringify(value);
            return s.length > 80 ? `${s.slice(0, 77)}…` : s;
          }
          return String(value);
        };
        if (entries.length === 0 && !metrics.config) return null;
        return (
          <details>
            <summary>All Metrics & Config</summary>
            <div style={{ marginTop: 8 }}>
              {entries.length > 0 && (
                <div className="list">
                  {entries.map(([key, value]) => (
                    <div key={key} className="card">
                      <div>{key}</div>
                      <div>{renderValue(value)}</div>
                    </div>
                  ))}
                </div>
              )}
              {metrics.config && (
                <pre style={{ margin: "8px 0 0", whiteSpace: "pre-wrap", fontSize: 11 }}>
                  {JSON.stringify(metrics.config, null, 2)}
                </pre>
              )}
            </div>
          </details>
        );
      })()}
    </>
  );
});
