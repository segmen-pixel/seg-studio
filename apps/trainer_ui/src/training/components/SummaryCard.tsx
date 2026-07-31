// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import type { TrainRunItem } from "../../utils";
import type { ClassInfo } from "../types";
import type { TranslationKey } from "../../i18n";
import { parseApiDate } from "../../time";

export type SummaryCardProps = {
  bestRun: TrainRunItem | null;
  f1Delta: number | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  bestMetrics: any;
  convergenceStatus: "converged" | "overfit" | "undertrained" | null;
  perClassF1: { classId: number; f1: number }[];
  scoreClasses: ClassInfo[];
  dsStats: { num_train?: number; fg_ratio?: number; num_active_classes?: number } | null;
  effectiveRuns: TrainRunItem[];
  completedRunCount: number;
  modelReadyCount: number;
  selectedRunLabel: string;
  selectedRunSummary: string;
  onOpenResults?: (runId: string, label: string) => void;
  lang: string;
  t: (key: TranslationKey) => string;
};

export default function SummaryCard({
  bestRun,
  f1Delta,
  bestMetrics,
  convergenceStatus,
  perClassF1,
  scoreClasses,
  dsStats,
  effectiveRuns,
  completedRunCount,
  selectedRunLabel,
  selectedRunSummary,
  onOpenResults,
  lang,
  t,
}: SummaryCardProps) {
  const [convHelpOpen, setConvHelpOpen] = useState(false);
  return (
    <div className="training-summary-card">
      {bestRun ? (
        <>
          {/* Zone A: model name + F1 ring gauge */}
          <div className="summary-zone-hero">
            <div className="summary-hero-identity">
              <div className="training-summary-label">{lang === "ja" ? "ベストモデル" : "Best Model"}</div>
              <div className="training-summary-title">{bestRun.model_name || `Run ${bestRun.run_id.slice(0, 8)}`}</div>
            </div>
            <div className="summary-f1-gauge" style={{ "--f1-pct": typeof bestRun.best_f1 === "number" ? Math.round(bestRun.best_f1 * 100) : 0 } as React.CSSProperties}>
              <div className="summary-f1-inner">
                <span className="summary-hero-value">{typeof bestRun.best_f1 === "number" ? bestRun.best_f1.toFixed(3) : "—"}</span>
                {f1Delta !== null && (
                  <span className={`summary-delta ${f1Delta >= 0 ? "up" : "down"}`}>
                    {f1Delta >= 0 ? "+" : ""}{f1Delta.toFixed(3)}
                  </span>
                )}
                <span className="summary-metric-label" title={t("metric.f1.tooltip")}>F1</span>
              </div>
            </div>
          </div>
          {/* Zone B: secondary metrics */}
          <div className="summary-zone-metrics">
            <div className="summary-kv-row">
              <span className="summary-kv-label" title={t("metric.miou.tooltip")}>mIoU</span>
              <span className="summary-kv-value">{typeof bestRun.best_miou === "number" ? bestRun.best_miou.toFixed(3) : "—"}</span>
            </div>
            {bestRun.created_at && (() => {
              // Naive UTC from the API: normalize before reading local fields.
              const d = parseApiDate(bestRun.created_at);
              if (!d) return null;
              const hh = String(d.getHours()).padStart(2, "0");
              const mm = String(d.getMinutes()).padStart(2, "0");
              return (
                <div className="summary-kv-row">
                  <span className="summary-kv-label">{lang === "ja" ? "開始" : "Started"}</span>
                  <span className="summary-kv-value">{hh}:{mm}</span>
                </div>
              );
            })()}
            {convergenceStatus && (
              <div className="summary-kv-row">
                <span className="summary-kv-label">{lang === "ja" ? "収束" : "Status"}</span>
                <span className="summary-conv-group">
                <span className={`summary-convergence-pill summary-badge-${convergenceStatus}`}>
                  {convergenceStatus === "converged" ? "✓ " : convergenceStatus === "overfit" ? "⚠ " : "… "}
                  {convergenceStatus === "converged" ? (lang === "ja" ? "収束" : "Converged")
                    : convergenceStatus === "overfit" ? (lang === "ja" ? "過学習" : "Overfit")
                    : (lang === "ja" ? "学習不足" : "Undertrained")}
                </span>
                <button type="button" className="summary-help-icon" aria-label={lang === "ja" ? "収束判定の説明" : "About convergence status"} onClick={() => setConvHelpOpen(true)}>?</button>
                </span>
              </div>
            )}
          </div>
          {/* Zone C: per-class F1 bars */}
          <div className="summary-zone-bars">
            {perClassF1.length > 0 ? (
              <>
                <div className="summary-class-bars-title">{lang === "ja" ? "クラス別 F1" : "Per-Class F1"}</div>
                {perClassF1.map((c) => {
                  const cls = scoreClasses.find((cl) => cl.id === c.classId);
                  return (
                    <div key={c.classId} className="summary-class-row">
                      <span className="summary-class-name">{cls?.name ?? (c.classId === 0 ? (lang === "ja" ? "背景" : "BG") : `class${c.classId}`)}</span>
                      <div className="summary-class-bar-track">
                        <div
                          className="summary-class-bar-fill"
                          style={{
                            width: `${Math.round(c.f1 * 100)}%`,
                            background: c.f1 >= 0.7 ? "#30d158" : c.f1 >= 0.4 ? "#f0a040" : "#ff453a",
                          }}
                        />
                      </div>
                      <span className="summary-class-val">{c.f1.toFixed(2)}</span>
                    </div>
                  );
                })}
              </>
            ) : (
              <div className="summary-class-bars-title" style={{ color: "var(--muted)" }}>—</div>
            )}
          </div>
          {/* Zone D: Confusion Matrix mini heatmap */}
          {bestMetrics?.confusion_matrix_val ? (() => {
            const rawMatrix = bestMetrics.confusion_matrix_val as number[][];
            const numClasses = Math.min(rawMatrix.length, 5);
            const matrix = rawMatrix.slice(0, numClasses).map(row => row.slice(0, numClasses));
            const classLabels = matrix.map((_, i) => {
              const cls = scoreClasses.find((cl) => cl.id === i);
              return cls?.name ?? (i === 0 ? (lang === "ja" ? "背景" : "BG") : `c${i}`);
            });
            return (
              <div className="summary-zone-cm">
                <div className="summary-class-bars-title">{lang === "ja" ? "混同行列" : "Confusion Matrix"}</div>
                <div className="summary-cm-grid" style={{ gridTemplateColumns: `auto repeat(${numClasses}, 1fr)` }}>
                  <span className="summary-cm-corner" />
                  {classLabels.map((l, i) => <span key={`h${i}`} className="summary-cm-header">{l}</span>)}
                  {matrix.map((row, i) => {
                    const rowTotal = row.reduce((a, b) => a + b, 0);
                    return (
                      <React.Fragment key={`r${i}`}>
                        <span className="summary-cm-label">{classLabels[i]}</span>
                        {row.map((val, j) => {
                          const pct = rowTotal > 0 ? val / rowTotal : 0;
                          const isDiag = i === j;
                          return (
                            <span key={`c${i}-${j}`} className="summary-cm-cell" style={{
                              background: isDiag
                                ? `rgba(48, 209, 88, ${pct * 0.6 + 0.1})`
                                : pct > 0 ? `rgba(255, 69, 58, ${pct * 0.6 + 0.05})` : "transparent",
                            }}>
                              {(pct * 100).toFixed(pct > 0 && pct < 0.01 ? 1 : 0)}{pct > 0 ? "%" : ""}
                            </span>
                          );
                        })}
                      </React.Fragment>
                    );
                  })}
                </div>
              </div>
            );
          })() : <div className="summary-zone-cm" />}
          {/* Zone E: Calibration & Threshold info */}
          <div className="summary-zone-cal">
            <div className="summary-class-bars-title">{lang === "ja" ? "キャリブレーション" : "Calibration"}</div>
            {(() => {
              const optThresh = bestMetrics?.optimal_threshold as number | undefined;
              const optF1 = bestMetrics?.optimal_threshold_f1 as number | undefined;
              const ece = bestMetrics?.ece as number | undefined;
              if (typeof optF1 !== "number" && typeof ece !== "number") {
                return <div style={{ color: "var(--muted)", fontSize: 11 }}>—</div>;
              }
              const eceColor = typeof ece === "number"
                ? ece < 0.05 ? "#30d158" : ece < 0.15 ? "#f0a040" : "#ff453a"
                : "#888";
              return (
                <>
                  {typeof optF1 === "number" && (
                    <div className="summary-cal-row">
                      <span className="summary-cal-label">Opt F1</span>
                      <span className="summary-cal-value">{optF1.toFixed(3)}</span>
                    </div>
                  )}
                  {typeof optThresh === "number" && (
                    <div className="summary-cal-row">
                      <span className="summary-cal-label">Threshold</span>
                      <span className="summary-cal-value">{optThresh.toFixed(2)}</span>
                    </div>
                  )}
                  {typeof ece === "number" && (
                    <div className="summary-cal-row">
                      <span className="summary-cal-label">ECE</span>
                      <span className="summary-cal-value" style={{ color: eceColor }}>{ece.toFixed(4)}</span>
                    </div>
                  )}
                  {typeof ece === "number" && (
                    <div className="summary-cal-row">
                      <span className="summary-cal-label" />
                      <div className="summary-ece-bar-track">
                        <div className="summary-ece-bar-fill" style={{ width: `${Math.min(100, ece * 500)}%`, background: eceColor }} />
                      </div>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
          {convHelpOpen && (
            <div className="training-mode-help-overlay" onClick={() => setConvHelpOpen(false)}>
              <div className="training-mode-help-popup" onClick={(e) => e.stopPropagation()}>
                <button className="ghost" style={{ position: "absolute", top: 4, right: 8, fontSize: 16 }} onClick={() => setConvHelpOpen(false)}>×</button>
                <h3 style={{ margin: "0 0 10px", fontSize: 15 }}>{lang === "ja" ? "「収束」判定について" : "About the convergence status"}</h3>
                <p style={{ fontSize: 12, color: "var(--muted)", margin: "0 0 10px", lineHeight: 1.6 }}>{lang === "ja" ? "学習が十分かを、ベスト epoch の位置と train/val の差から自動判定します。" : "A heuristic that flags whether training finished cleanly, from the best-epoch position and the train/val gap."}</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12, lineHeight: 1.6 }}>
                  <div><strong style={{ color: "#30d158" }}>{lang === "ja" ? "✓ 収束" : "✓ Converged"}</strong>{lang === "ja" ? "：検証スコアが頭打ちに達してから終了（理想）。" : ": validation score plateaued before training ended (ideal)."}</div>
                  <div><strong style={{ color: "var(--ink)" }}>{lang === "ja" ? "… 学習不足" : "… Undertrained"}</strong>{lang === "ja" ? "：ベスト epoch が最終 epoch の直近（最後の 2 epoch 以内）。まだ精度が伸びている途中で打ち切られた状態。epoch 数を増やすか早期終了を有効にすると改善します。" : ": best epoch is within the last 2 of the run — the model was still improving when training stopped. Increase epochs or enable early stopping."}</div>
                </div>
              </div>
            </div>
          )}
          {/* Footer: model name + chips + CTA */}
          <div className="training-summary-footer">
            {bestRun.has_model && onOpenResults && (
              <button
                className="primary training-summary-cta"
                onClick={() => onOpenResults(bestRun.run_id, bestRun.model_name ?? `Run ${bestRun.run_id.slice(0, 8)}`)}
                data-testid="open-results-cta"
              >
                {t("projects.openResults")}
              </button>
            )}
            <span className="training-chip">{effectiveRuns.length} {t("training.runs")}</span>
            <span className="training-chip">{completedRunCount} {t("training.completed")}</span>
            {dsStats && (
              <>
                <span className="training-chip">{dsStats.num_train ?? "?"} {lang === "ja" ? "枚" : "imgs"}</span>
                {typeof dsStats.fg_ratio === "number" && (
                  <span className={`training-chip ${dsStats.fg_ratio < 0.03 ? "accent" : ""}`}>
                    FG {(dsStats.fg_ratio * 100).toFixed(1)}%
                  </span>
                )}
              </>
            )}
            {bestMetrics && typeof (bestMetrics as any).arch === "string" && (
              <span className="training-chip">{(bestMetrics as any).arch}</span>
            )}
            {bestMetrics && typeof (bestMetrics as any).best_epoch === "number" && (
              <span className="training-chip">E{(bestMetrics as any).best_epoch}{typeof (bestMetrics as any).dataset_stats?.epochs === "number" ? `/${(bestMetrics as any).dataset_stats.epochs}` : ""}</span>
            )}
            {bestMetrics && typeof (bestMetrics as any).patch_size === "number" && (
              <span className="training-chip">p{(bestMetrics as any).patch_size}</span>
            )}
          </div>
        </>
      ) : (
        <div className="training-summary-main">
          <div className="training-summary-title">{selectedRunLabel}</div>
          <div className="training-summary-copy">{selectedRunSummary}</div>
        </div>
      )}
    </div>
  );
}
