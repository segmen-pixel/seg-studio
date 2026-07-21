// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import { useI18n } from "../../i18n";
import {
  runHeatmapConfidenceUrl,
  runHeatmapClassUrl,
  runHeatmapErrorUrl,
  annotateImageUrl,
} from "../../api";
import type { PredictionScore, ClassInfo } from "../types";

export default function ScorePanel({
  scoreData,
  scoreLoading,
  scoreProgress,
  scoreClasses,
  scoreImageNames,
  scoreTotalImages,
  scoreSortKey,
  scoreSortAsc,
  onSortChange,
  onReload,
  selectedRunIdForLogs,
  hasModel,
  projectId,
}: {
  scoreData: Map<string, PredictionScore>;
  scoreLoading: boolean;
  scoreProgress: string;
  scoreClasses: ClassInfo[];
  scoreImageNames: Map<string, string>;
  scoreTotalImages: number;
  scoreSortKey: "name" | "confidence" | "fg_ratio";
  scoreSortAsc: boolean;
  onSortChange: (key: "name" | "confidence" | "fg_ratio") => void;
  onReload: () => void;
  selectedRunIdForLogs: string | null;
  hasModel: boolean;
  projectId: string;
}) {
  const { t } = useI18n();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [heatmapClassId, setHeatmapClassId] = useState<number>(() =>
    scoreClasses.length > 0 ? scoreClasses[0].id : 1,
  );
  if (!selectedRunIdForLogs) {
    return <div className="score-panel"><div className="muted">{t("training.selectRun")}</div></div>;
  }
  if (!hasModel) {
    return <div className="score-panel"><div className="muted">{t("training.noModelYet")}</div></div>;
  }
  if (scoreLoading) {
    return <div className="score-panel"><div className="muted">{scoreProgress || "Loading..."}</div></div>;
  }
  if (scoreData.size === 0) {
    return (
      <div className="score-panel">
        <div className="muted">{t("training.noScoreData")}</div>
        <button className="ghost" onClick={onReload} style={{ marginTop: 8 }} data-desc={t("training.runScoreDesc")}>Run Score</button>
      </div>
    );
  }

  // Summary
  const entries = Array.from(scoreData.entries());
  const avgConf = entries.reduce((s, [, v]) => s + v.mean_confidence, 0) / entries.length;
  const avgFgConf = entries.reduce((s, [, v]) => s + v.foreground_mean_confidence, 0) / entries.length;
  const avgFgRatio = entries.reduce((s, [, v]) => s + v.foreground_ratio, 0) / entries.length;

  // Per-class aggregation
  const classAgg = new Map<string, { sum: number; count: number }>();
  for (const [, score] of entries) {
    if (!score.per_class_mean_confidence) continue;
    for (const [cid, conf] of Object.entries(score.per_class_mean_confidence)) {
      const prev = classAgg.get(cid) ?? { sum: 0, count: 0 };
      prev.sum += conf;
      prev.count += 1;
      classAgg.set(cid, prev);
    }
  }
  const classBarData = scoreClasses.map((c) => {
    const agg = classAgg.get(String(c.id));
    return { id: c.id, name: c.name, color: c.color, avg: agg ? agg.sum / agg.count : 0 };
  }).filter((d) => d.avg > 0);

  // SVG bar chart
  const barW = 400;
  const barH = Math.max(80, classBarData.length * 28 + 20);
  const barMaxVal = Math.max(...classBarData.map((d) => d.avg), 0.01);
  const labelW = 80;
  const chartW = barW - labelW - 40;

  // Sorted image table
  const sortedEntries = [...entries].sort((a, b) => {
    const nameA = scoreImageNames.get(a[0]) ?? a[0];
    const nameB = scoreImageNames.get(b[0]) ?? b[0];
    let cmp = 0;
    if (scoreSortKey === "name") cmp = nameA.localeCompare(nameB);
    else if (scoreSortKey === "confidence") cmp = a[1].mean_confidence - b[1].mean_confidence;
    else cmp = a[1].foreground_ratio - b[1].foreground_ratio;
    return scoreSortAsc ? cmp : -cmp;
  });

  const sortArrow = (key: typeof scoreSortKey) =>
    scoreSortKey === key ? (scoreSortAsc ? " \u25B2" : " \u25BC") : "";

  return (
    <div className="score-panel">
      <div className="score-panel-toolbar">
        <button className="ghost" onClick={onReload} style={{ fontSize: 12 }} data-desc={t("training.reloadScoreDesc")}>Reload</button>
      </div>
      <div className="score-summary-grid">
        <div className="score-summary-item">
          <div className="score-summary-label">Mean Conf</div>
          <div className="score-summary-value">{(avgConf * 100).toFixed(1)}%</div>
        </div>
        <div className="score-summary-item">
          <div className="score-summary-label">FG Conf</div>
          <div className="score-summary-value">{(avgFgConf * 100).toFixed(1)}%</div>
        </div>
        <div className="score-summary-item">
          <div className="score-summary-label">FG Ratio</div>
          <div className="score-summary-value">{(avgFgRatio * 100).toFixed(2)}%</div>
        </div>
        <div className="score-summary-item">
          <div className="score-summary-label">Images</div>
          <div className="score-summary-value">
            {entries.length}{scoreTotalImages > entries.length ? ` / ${scoreTotalImages}` : ""}
          </div>
          {scoreTotalImages > entries.length && (
            <div className="score-summary-warn">{scoreTotalImages - entries.length} failed</div>
          )}
        </div>
      </div>
      {classBarData.length > 0 && (
        <div className="score-bar-chart">
          <div className="score-bar-chart-title">Per-Class Mean Confidence</div>
          <svg width="100%" viewBox={`0 0 ${barW} ${barH}`} style={{ maxWidth: barW }}>
            {classBarData.map((d, i) => {
              const y = 10 + i * 28;
              const w = (d.avg / barMaxVal) * chartW;
              const rgb = `rgb(${d.color[0]},${d.color[1]},${d.color[2]})`;
              return (
                <g key={d.id}>
                  <text x={labelW - 4} y={y + 15} textAnchor="end" fontSize={11} fill="var(--ink)">
                    {d.name}
                  </text>
                  <rect x={labelW} y={y + 2} width={Math.max(w, 2)} height={18} rx={3} fill={rgb} opacity={0.85} />
                  <text x={labelW + w + 6} y={y + 15} fontSize={10} fill="var(--muted)">
                    {(d.avg * 100).toFixed(1)}%
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}
      <div className="score-image-table-wrap">
        <table className="score-image-table">
          <thead>
            <tr>
              <th onClick={() => onSortChange("name")} style={{ cursor: "pointer" }}>
                {t("training.image")}{sortArrow("name")}
              </th>
              <th onClick={() => onSortChange("confidence")} style={{ cursor: "pointer" }}>
                {t("training.confidence")}{sortArrow("confidence")}
              </th>
              <th onClick={() => onSortChange("fg_ratio")} style={{ cursor: "pointer" }}>
                {t("training.fgRatio")}{sortArrow("fg_ratio")}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedEntries.map(([id, score]) => (
              <tr
                key={id}
                className={selectedImage === id ? "score-image-row selected" : "score-image-row"}
                onClick={() => setSelectedImage(selectedImage === id ? null : id)}
                style={{ cursor: "pointer" }}
              >
                <td title={id}>{scoreImageNames.get(id) ?? id.slice(0, 12)}</td>
                <td>{(score.mean_confidence * 100).toFixed(1)}%</td>
                <td>{(score.foreground_ratio * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selectedImage && selectedRunIdForLogs && (
        <div className="score-heatmap-detail">
          <div className="score-heatmap-header">
            <span className="score-heatmap-title">
              {scoreImageNames.get(selectedImage) ?? selectedImage}
            </span>
            <button className="ghost" onClick={() => setSelectedImage(null)} style={{ fontSize: 11, padding: "2px 6px" }}>✕</button>
          </div>
          <div className="score-heatmap-grid">
            <div className="score-heatmap-item">
              <div className="score-heatmap-label">Original</div>
              <img
                src={annotateImageUrl(projectId, (scoreImageNames.get(selectedImage) ?? selectedImage))}
                alt="original"
                className="score-heatmap-img"
              />
            </div>
            <div className="score-heatmap-item">
              <div className="score-heatmap-label">{t("training.confidence")}</div>
              <img
                src={runHeatmapConfidenceUrl(projectId, selectedRunIdForLogs, selectedImage)}
                alt="confidence"
                className="score-heatmap-img"
              />
            </div>
            <div className="score-heatmap-item">
              <div className="score-heatmap-label">
                Class
                <select
                  value={heatmapClassId}
                  onChange={(e) => setHeatmapClassId(Number(e.target.value))}
                  className="score-heatmap-class-select"
                >
                  {scoreClasses.map((c) => (
                    <option key={c.id} value={c.id}>{c.name}</option>
                  ))}
                </select>
              </div>
              <img
                src={runHeatmapClassUrl(projectId, selectedRunIdForLogs, selectedImage, heatmapClassId)}
                alt="class probability"
                className="score-heatmap-img"
              />
            </div>
            <div className="score-heatmap-item">
              <div className="score-heatmap-label">Error</div>
              <img
                src={runHeatmapErrorUrl(projectId, selectedRunIdForLogs, selectedImage)}
                alt="error map"
                className="score-heatmap-img"
              />
            </div>
          </div>
          <div className="score-heatmap-legend">
            <span style={{ color: "#00c800" }}>■ Correct</span>
            <span style={{ color: "#dc2828" }}>■ False Positive</span>
            <span style={{ color: "#ffc800" }}>■ False Negative</span>
          </div>
        </div>
      )}
    </div>
  );
}
