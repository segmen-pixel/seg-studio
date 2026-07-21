// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import type { ClassItem, MetricsPayload, PredictionScore } from "../types";
import type { ImageItem } from "../ImageListPanel";

type Props = {
  metrics: MetricsPayload | null;
  activeRunId: string | null;
  predictBackend: "onnx" | "coreml";
  scoreCacheRef: React.RefObject<Map<string, PredictionScore>> | React.MutableRefObject<Map<string, PredictionScore>>;
  images: ImageItem[];
  confidenceThreshold: number;
  effectiveClasses: ClassItem[];
  pixelHist: { bins: number[]; counts: number[]; total_pixels: number } | null;
  cacheVersion: number;
};

/* ── Small reusable histogram renderer ── */
function MiniHistogram({
  title,
  bins,
  maxBin,
  xLabels,
  barColor,
  barOpacity,
  tooltipFn,
  thresholdPct,
}: {
  title: string;
  bins: number[];
  maxBin: number;
  xLabels: string[];
  barColor: (binIndex: number, binCount: number) => string;
  barOpacity?: (binIndex: number) => number;
  tooltipFn: (binIndex: number, count: number) => string;
  thresholdPct?: number;
}) {
  return (
    <>
      <div className="muted" style={{ fontSize: 10, marginTop: 8, marginBottom: 2 }}>
        {title}
      </div>
      <div style={{ display: "flex", width: "100%", height: 80 }}>
        {/* Y-axis */}
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", fontSize: 8, color: "#888", width: 20, flexShrink: 0, textAlign: "right", paddingRight: 3 }}>
          <span>{maxBin}</span><span>{Math.ceil(maxBin / 2)}</span><span>0</span>
        </div>
        {/* Chart area */}
        <div style={{ position: "relative", flex: 1, minWidth: 0, height: "100%" }}>
          {[0, 0.5, 1].map((v) => (
            <div key={v} style={{ position: "absolute", left: 0, right: 0, bottom: `${v * 100}%`, borderBottom: "1px solid rgba(255,255,255,0.06)", pointerEvents: "none" }} />
          ))}
          {thresholdPct != null && thresholdPct > 0 && (
            <div style={{ position: "absolute", bottom: 0, top: 0, left: `${thresholdPct}%`, borderLeft: "2px dashed rgba(79,195,247,0.7)", pointerEvents: "none", zIndex: 2 }}>
              <span style={{ position: "absolute", top: -1, left: 3, fontSize: 8, color: "#4fc3f7", whiteSpace: "nowrap" }}>{thresholdPct.toFixed(0)}%</span>
            </div>
          )}
          <div style={{ display: "flex", alignItems: "flex-end", width: "100%", height: "100%" }}>
            {bins.map((count, i) => {
              const h = count > 0 ? Math.max(2, (count / maxBin) * 76) : 0;
              return (
                <div
                  key={i}
                  style={{ flex: 1, minWidth: 0, height: h, background: barColor(i, bins.length), borderRadius: 1, marginRight: 1, opacity: barOpacity ? barOpacity(i) : 0.8 }}
                  title={tooltipFn(i, count)}
                />
              );
            })}
          </div>
        </div>
      </div>
      {/* X-axis */}
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#888", marginTop: 1, marginLeft: 20 }}>
        {xLabels.map((l, i) => <span key={i}>{l}</span>)}
      </div>
    </>
  );
}

export default React.memo(function MetricsSection({
  metrics,
  activeRunId,
  predictBackend,
  scoreCacheRef,
  images,
  confidenceThreshold,
  effectiveClasses,
  pixelHist,
  cacheVersion: _cacheVersion,
}: Props) {
  const { t } = useI18n();
  // ── Segmentation inference chart data ──
  const chartScores: { id: string; name: string; fgConf: number }[] = [];
  if (activeRunId && scoreCacheRef.current) {
    const prefix = `${predictBackend}:${activeRunId}:`;
    for (const [key, score] of scoreCacheRef.current.entries()) {
      if (!key.startsWith(prefix)) continue;
      const imgId = key.slice(prefix.length).replace(/:tta$/, "");
      const img = images.find((item) => item.id === imgId);
      chartScores.push({ id: imgId, name: img?.name ?? imgId.slice(0, 8), fgConf: score.foreground_mean_confidence });
    }
  }
  chartScores.sort((a, b) => a.fgConf - b.fgConf);
  const avgConf = chartScores.length > 0 ? chartScores.reduce((s, x) => s + x.fgConf, 0) / chartScores.length : 0;

  // Training metrics
  const m = metrics?.metrics;
  // F1 curve from val-set threshold sweep: look up live F1 for current slider position.
  // Fallback to best_F1_val when curve missing or slider at 0.
  const f1Curve = m?.["f1_curve"] as Array<{ threshold: number; f1: number }> | undefined;
  const liveF1: number | null = (() => {
    if (!f1Curve || f1Curve.length === 0 || confidenceThreshold === 0) return null;
    const target = confidenceThreshold / 100;
    let nearest = f1Curve[0];
    let minDist = Math.abs(nearest.threshold - target);
    for (const pt of f1Curve) {
      const d = Math.abs(pt.threshold - target);
      if (d < minDist) { minDist = d; nearest = pt; }
    }
    return nearest.f1;
  })();
  const bestF1 = liveF1 != null
    ? liveF1.toFixed(3)
    : (m && typeof m["best_F1_val"] === "number" ? (m["best_F1_val"] as number).toFixed(3) : null);
  const bestMIoU = m && typeof m["best_mIoU_val"] === "number" ? (m["best_mIoU_val"] as number).toFixed(3) : null;
  const bestEpoch = m && typeof m["best_epoch"] === "number" ? String(m["best_epoch"]) : null;
  const loss = m && typeof m["loss"] === "number" ? (m["loss"] as number).toFixed(4) : null;

  const perClassF1Val = m?.["per_class_f1_val"] as Record<string, number> | undefined;
  const perClassPrecVal = m?.["per_class_precision_val"] as Record<string, number> | undefined;
  const perClassRecVal = m?.["per_class_recall_val"] as Record<string, number> | undefined;
  const perClassIoUVal = m?.["per_class_iou_val"] as Record<string, number> | undefined;
  const classIdsInMetrics = new Set<string>();
  [perClassF1Val, perClassPrecVal, perClassRecVal, perClassIoUVal].forEach((obj) => {
    if (obj) Object.keys(obj).forEach((k) => classIdsInMetrics.add(k));
  });
  const sortedClassIds = Array.from(classIdsInMetrics).sort((a, b) => Number(a) - Number(b));

  const hasSegData = bestF1 != null || chartScores.length > 0;
  if (!hasSegData) return null;

  return (
    <div className="section" style={{ marginTop: 12 }}>
      <div className="section-title">{t("results.metrics.allImageScores")}</div>

      {/* ── Segmentation: Training score grid ── */}
      {bestF1 && (
        <div className="score-grid">
          <span className="score-label">F1</span>
          <span className="score-val best">{bestF1}</span>
          <span className="score-label">mIoU</span>
          <span className="score-val">{bestMIoU ?? "\u2014"}</span>
          <span className="score-label">Ep</span>
          <span className="score-val">{bestEpoch ?? "\u2014"}</span>
          <span className="score-label">Loss</span>
          <span className="score-val">{loss ?? "\u2014"}</span>
        </div>
      )}
      {/* ── Segmentation: Confidence distribution histogram ── */}
      {chartScores.length > 0 && (() => {
        const BIN_COUNT = 10;
        const bins = new Array(BIN_COUNT).fill(0);
        for (const s of chartScores) {
          const idx = Math.min(BIN_COUNT - 1, Math.floor(s.fgConf * BIN_COUNT));
          bins[idx]++;
        }
        const maxBin = Math.max(1, ...bins);
        const threshBin = Math.floor(confidenceThreshold / 100 * BIN_COUNT);
        return (
          <>
            <div className="muted" style={{ fontSize: 10, marginTop: 8, marginBottom: 2 }}>
              {t("results.metrics.fgConfDist").replace("{n}", String(chartScores.length)).replace("{avg}", (avgConf * 100).toFixed(1))}
            </div>
            <div style={{ display: "flex", width: "100%", height: 80 }}>
              {/* Y-axis */}
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", fontSize: 8, color: "#888", width: 20, flexShrink: 0, textAlign: "right", paddingRight: 3 }}>
                <span>{maxBin}</span><span>{Math.ceil(maxBin / 2)}</span><span>0</span>
              </div>
              {/* Histogram */}
              <div style={{ position: "relative", flex: 1, minWidth: 0, height: "100%" }}>
                {[0, 0.5, 1].map((v) => (
                  <div key={v} style={{ position: "absolute", left: 0, right: 0, bottom: `${v * 100}%`, borderBottom: "1px solid rgba(255,255,255,0.06)", pointerEvents: "none" }} />
                ))}
                {/* Confidence threshold line */}
                {confidenceThreshold > 0 && (
                  <div style={{
                    position: "absolute", bottom: 0, top: 0,
                    left: `${confidenceThreshold}%`,
                    borderLeft: "2px dashed rgba(79,195,247,0.7)",
                    pointerEvents: "none", zIndex: 2,
                  }}>
                    <span style={{ position: "absolute", top: -1, left: 3, fontSize: 8, color: "#4fc3f7", whiteSpace: "nowrap" }}>{confidenceThreshold}%</span>
                  </div>
                )}
                <div style={{ display: "flex", alignItems: "flex-end", width: "100%", height: "100%" }}>
                  {bins.map((count, i) => {
                    const pct = (i + 0.5) * (100 / BIN_COUNT);
                    const h = count > 0 ? Math.max(2, (count / maxBin) * 76) : 0;
                    const color = pct >= 80 ? "#4caf50" : pct >= 50 ? "#ff9800" : "#f44336";
                    const belowThreshold = i < threshBin;
                    return (
                      <div
                        key={i}
                        style={{
                          flex: 1, minWidth: 0, height: h, background: color,
                          borderRadius: 1, marginRight: 1,
                          opacity: belowThreshold ? 0.25 : 0.8,
                        }}
                        title={t("results.metrics.binTooltip").replace("{range}", `${i * 10}%-${(i + 1) * 10}%`).replace("{count}", String(count))}
                      />
                    );
                  })}
                </div>
              </div>
            </div>
            {/* X-axis */}
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#888", marginTop: 1, marginLeft: 20 }}>
              <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
            </div>
            <div className="muted" style={{ fontSize: 9, display: "flex", gap: 8, marginTop: 2 }}>
              <span style={{ color: "#f44336" }}>● &lt;50%</span>
              <span style={{ color: "#ff9800" }}>● 50-80%</span>
              <span style={{ color: "#4caf50" }}>● &gt;80%</span>
              {confidenceThreshold > 0 && <span style={{ color: "#4fc3f7" }}>┆ threshold</span>}
            </div>
          </>
        );
      })()}
      {/* ── Segmentation: Pixel-level confidence histogram ── */}
      {pixelHist && pixelHist.total_pixels > 0 && (() => {
        const { bins: binEdges, counts } = pixelHist;
        const maxCount = Math.max(1, ...counts);
        const BIN_COUNT = counts.length;
        return (
          <>
            <div className="muted" style={{ fontSize: 10, marginTop: 10, marginBottom: 2 }}>
              {t("results.metrics.pixelConfDist").replace("{px}", (pixelHist.total_pixels / 1000000).toFixed(1))}
            </div>
            <div style={{ display: "flex", width: "100%", height: 80 }}>
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", fontSize: 8, color: "#888", width: 28, flexShrink: 0, textAlign: "right", paddingRight: 3 }}>
                <span>{maxCount > 1000000 ? `${(maxCount/1000000).toFixed(1)}M` : maxCount > 1000 ? `${(maxCount/1000).toFixed(0)}K` : maxCount}</span>
                <span>0</span>
              </div>
              <div style={{ position: "relative", flex: 1, minWidth: 0, height: "100%" }}>
                {confidenceThreshold > 0 && (
                  <div style={{
                    position: "absolute", bottom: 0, top: 0,
                    left: `${confidenceThreshold}%`,
                    borderLeft: "2px dashed rgba(79,195,247,0.7)",
                    pointerEvents: "none", zIndex: 2,
                  }} />
                )}
                <div style={{ display: "flex", alignItems: "flex-end", width: "100%", height: "100%" }}>
                  {counts.map((count, i) => {
                    const pct = ((binEdges[i] + binEdges[i + 1]) / 2) * 100;
                    const h = count > 0 ? Math.max(1, (count / maxCount) * 76) : 0;
                    const color = pct >= 80 ? "#66bb6a" : pct >= 50 ? "#ffb74d" : "#ef5350";
                    return (
                      <div
                        key={i}
                        style={{ flex: 1, minWidth: 0, height: h, background: color, opacity: 0.8, marginRight: BIN_COUNT > 30 ? 0 : 1 }}
                        title={`${(binEdges[i] * 100).toFixed(0)}-${(binEdges[i + 1] * 100).toFixed(0)}%: ${count.toLocaleString()} px`}
                      />
                    );
                  })}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#888", marginTop: 1, marginLeft: 28 }}>
              <span>0%</span><span>50%</span><span>100%</span>
            </div>
          </>
        );
      })()}
      {/* Per-class metrics table */}
      {sortedClassIds.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, marginTop: 8 }}>
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
            {sortedClassIds.map((id) => {
              const classId = Number(id);
              const cls = effectiveClasses.find((c) => c.id === classId);
              return (
                <tr key={`pcm-${id}`}>
                  <td style={{ padding: "2px 4px" }}>
                    <div className="row" style={{ gap: 4 }}>
                      <div style={{ width: 10, height: 10, borderRadius: 3, border: "1px solid var(--border)", background: cls ? `rgb(${cls.color[0]}, ${cls.color[1]}, ${cls.color[2]})` : "#888" }} />
                      {cls?.name ?? `class${id}`}
                    </div>
                  </td>
                  <td style={{ textAlign: "right", padding: "2px 4px" }}>{(perClassF1Val?.[id] ?? 0).toFixed(3)}</td>
                  <td style={{ textAlign: "right", padding: "2px 4px" }}>{(perClassPrecVal?.[id] ?? 0).toFixed(3)}</td>
                  <td style={{ textAlign: "right", padding: "2px 4px" }}>{(perClassRecVal?.[id] ?? 0).toFixed(3)}</td>
                  <td style={{ textAlign: "right", padding: "2px 4px" }}>{(perClassIoUVal?.[id] ?? 0).toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
});
