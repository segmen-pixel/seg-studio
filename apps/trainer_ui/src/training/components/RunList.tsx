// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import MonitorState from "./MonitorState";
import TrainingModeBadge from "./TrainingModeBadge";
import type { TrainRunItem } from "../../utils";
import type { OpenVinoPrecision } from "../../api/training";
import type { TranslationKey } from "../../i18n";
import { formatApiDate, parseApiDate } from "../../time";

type RunSortKey = "date" | "f1" | "name";

export type RunListProps = {
  effectiveRuns: TrainRunItem[];
  selectedRunIdForLogs: string | null;
  setSelectedRunIdForLogs: (id: string) => void;
  runsReady: boolean;
  runSortKey: RunSortKey;
  setRunSortKey: (key: RunSortKey) => void;
  trainingProgress: { current: number; total: number; pct: number; bestF1: number | null } | null;
  stoppingRunIds: Set<string>;
  lockedRunIds?: string[];
  viewedRunIds?: string[];
  descMode?: boolean;
  exportingModel: boolean;
  exportingOpenVino?: boolean;
  onStopRun: (runId: string) => void;
  onDeleteRun: (runId: string) => void;
  onOptimizeRun: (runId: string) => void;
  onExportModel: () => void;
  onExportOpenVino?: (precision: OpenVinoPrecision) => void;
  onOpenResults?: (runId: string, label: string) => void;
  moveRunSelection: (offset: number) => void;
  selectedRun: TrainRunItem | null;
  modelReadyCount: number;
  exportFormat?: "coreml" | "coreml-updatable" | "onnx";
  lang: string;
  t: (key: TranslationKey) => string;
};

export default function RunList({
  effectiveRuns,
  selectedRunIdForLogs,
  setSelectedRunIdForLogs,
  runsReady,
  runSortKey,
  setRunSortKey,
  trainingProgress,
  stoppingRunIds,
  lockedRunIds,
  viewedRunIds,
  descMode,
  exportingModel,
  exportingOpenVino,
  onStopRun,
  onDeleteRun,
  onOptimizeRun,
  onExportModel,
  onExportOpenVino,
  onOpenResults,
  moveRunSelection,
  selectedRun,
  modelReadyCount,
  exportFormat,
  lang,
  t,
}: RunListProps) {
  const sortKeys: RunSortKey[] = ["date", "f1", "name"];

  return (
    <div className="section training-runs-section" style={{ marginBottom: 0 }} data-tutorial-step="training-runs">
      <div className="training-models-header">
        <div>
          <div className="section-title">{t("training.models")}</div>
          <div className="training-section-meta">
            {!runsReady
              ? t("training.syncingRuns")
              : effectiveRuns.length === 0
                ? t("training.noRuns")
                : `${effectiveRuns.length} ${t("training.runs")} · ${modelReadyCount} ${t("training.ready")}`}
          </div>
        </div>
        <div className="training-models-actions">
          <button
            className="models-action-btn models-sort-btn"
            onClick={() => {
              const next = sortKeys[(sortKeys.indexOf(runSortKey) + 1) % sortKeys.length];
              setRunSortKey(next);
            }}
            title={t("training.sortRuns")}
            data-desc={t("training.sortRuns.desc")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 6h18M6 12h12M9 18h6"/>
            </svg>
            {runSortKey === "date" ? t("training.sortDate") : runSortKey === "f1" ? "F1" : t("training.sortName")}
          </button>
          <button
            className="models-action-btn"
            onClick={onExportModel}
            disabled={!selectedRunIdForLogs || exportingModel || !effectiveRuns.find((r) => r.run_id === selectedRunIdForLogs)?.has_model}
            title={`Export ${(exportFormat ?? "coreml").toUpperCase()}`}
            data-desc={t("training.export.desc")}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
          </button>
          {onExportOpenVino && (
            <div className="models-action-dropdown">
              <button
                className="models-action-btn"
                disabled={
                  !selectedRunIdForLogs ||
                  exportingOpenVino ||
                  !effectiveRuns.find((r) => r.run_id === selectedRunIdForLogs)?.has_model
                }
                title={t("training.export.openvino")}
                data-desc={t("training.export.openvino.desc")}
              >
                {/* Intel-like glyph: square with corner cut, hints at edge/CPU target */}
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 4h12l4 4v12H4z"/>
                  <path d="M16 4v4h4"/>
                  <path d="M8 12h8M8 16h5"/>
                </svg>
              </button>
              <div className="models-action-dropdown-menu" role="menu">
                <button
                  className="models-action-dropdown-item"
                  onClick={() => onExportOpenVino("fp32")}
                  disabled={
                    !selectedRunIdForLogs ||
                    exportingOpenVino ||
                    !effectiveRuns.find((r) => r.run_id === selectedRunIdForLogs)?.has_model
                  }
                  role="menuitem"
                >
                  {t("training.export.openvino.fp32")}
                </button>
                <button
                  className="models-action-dropdown-item"
                  onClick={() => onExportOpenVino("fp16")}
                  disabled={
                    !selectedRunIdForLogs ||
                    exportingOpenVino ||
                    !effectiveRuns.find((r) => r.run_id === selectedRunIdForLogs)?.has_model
                  }
                  role="menuitem"
                >
                  {t("training.export.openvino.fp16")}
                </button>
                <button
                  className="models-action-dropdown-item"
                  onClick={() => onExportOpenVino("int8")}
                  disabled={
                    !selectedRunIdForLogs ||
                    exportingOpenVino ||
                    !effectiveRuns.find((r) => r.run_id === selectedRunIdForLogs)?.has_model
                  }
                  role="menuitem"
                >
                  {t("training.export.openvino.int8")}
                </button>
              </div>
            </div>
          )}
          <button
            className="models-action-btn models-action-btn-danger"
            onClick={() => selectedRunIdForLogs && onDeleteRun(selectedRunIdForLogs)}
            disabled={!selectedRunIdForLogs || (lockedRunIds?.includes(selectedRunIdForLogs) ?? false)}
            title={selectedRunIdForLogs && lockedRunIds?.includes(selectedRunIdForLogs) ? t("training.lockedInResults") : t("training.delete")}
            data-desc={selectedRunIdForLogs && lockedRunIds?.includes(selectedRunIdForLogs) ? t("training.lockedInResults") : t("training.deleteRun.desc")}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
            </svg>
          </button>
        </div>
      </div>
      {trainingProgress && (
        <div className="training-progress-bar-wrap">
          <div className="training-progress-info">
            <span>{t("training.epoch")} {trainingProgress.current}/{trainingProgress.total}</span>
            {trainingProgress.bestF1 !== null && (
              <span>{t("training.bestF1")}: {trainingProgress.bestF1.toFixed(4)}</span>
            )}
            <span>{trainingProgress.pct.toFixed(0)}%</span>
          </div>
          <div className="training-progress-track">
            <div
              className="training-progress-fill"
              style={{ width: `${trainingProgress.pct}%` }}
            />
          </div>
        </div>
      )}
      <div
        className="training-run-list"
        role="listbox"
        aria-label="Training runs"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            moveRunSelection(1);
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            moveRunSelection(-1);
          } else if (
            event.key === "Enter" &&
            selectedRun &&
            selectedRun.has_model &&
            (selectedRun.status === "completed" || selectedRun.status === "stopped" || selectedRun.status === "done") &&
            onOpenResults
          ) {
            event.preventDefault();
            onOpenResults(selectedRun.run_id, selectedRun.model_name ?? `Run ${selectedRun.run_id.slice(0, 8)}`);
          }
        }}
      >
        {!runsReady ? (
          <MonitorState
            title={t("training.syncingRuns")}
            copy={t("training.syncingRuns")}
            tone="loading"
          />
        ) : effectiveRuns.length === 0 ? (
          <MonitorState
            title={t("training.noRuns")}
            copy={t("training.startFirst")}
          />
        ) : effectiveRuns.map((run) => {
          const isLocked = lockedRunIds?.includes(run.run_id) ?? false;
          return (
          <div
            key={run.run_id}
            className={`card ${selectedRunIdForLogs === run.run_id ? "active" : ""}${run.status === "running" && !stoppingRunIds.has(run.run_id) ? " run-running" : ""}${stoppingRunIds.has(run.run_id) ? " run-stopping" : ""}${run.status === "reserved" ? " run-reserved" : ""}${run.status === "failed" ? " run-failed" : ""}${run.status === "stopped" ? " run-stopped" : ""}${isLocked ? " run-locked" : ""}`}
            role="option"
            aria-selected={selectedRunIdForLogs === run.run_id}
            onClick={() => setSelectedRunIdForLogs(run.run_id)}
            style={{ cursor: "pointer" }}
          >
            <div>
              {(run.status === "completed" || run.status === "stopped") && run.has_model ? (
                <svg className="run-check-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-2)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              ) : stoppingRunIds.has(run.run_id) ? (
                <span className="run-stopping-badge">{t("training.stoppingBadge")}</span>
              ) : run.status === "running" ? (
                <span className="run-pulse-dot" />
              ) : run.status === "preparing" || run.status === "starting" ? (
                <span className="run-pulse-dot run-preparing" />
              ) : run.status === "reserved" ? (
                <span className="run-reserved-badge">{t("training.queueBadge")}{run.queue_position ? ` #${run.queue_position}` : ""}</span>
              ) : run.status === "failed" ? (
                <svg className="run-failed-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef5350" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              ) : null}
              <TrainingModeBadge mode={run.training_mode} />
              {run.fold_index != null && run.total_folds != null && (
                <span className="run-fold-badge" title={`CV fold ${(run.fold_index ?? 0) + 1} of ${run.total_folds}`}>
                  fold {(run.fold_index ?? 0) + 1}/{run.total_folds}
                </span>
              )}
              {run.iter_index != null && run.iter_max_iters != null && (
                <span className="run-iter-badge" title={`Iterative pass ${(run.iter_index ?? 0) + 1} of ${run.iter_max_iters}`}>
                  iter {(run.iter_index ?? 0) + 1}/{run.iter_max_iters}
                </span>
              )}
              {run.model_name || run.run_id.slice(0, 8)}{run.fp16 && <span style={{ color: "#4fc3f7", fontSize: 10, marginLeft: 4 }}>⚡</span>}
              {(() => {
                // API timestamps are naive UTC. parseApiDate normalizes them
                // before we read local clock fields; a bare new Date() would
                // render the UTC clock as if it were local time (-9h in JST).
                const d = parseApiDate(run.created_at);
                if (!d) return null;
                const hh = String(d.getHours()).padStart(2, "0");
                const mm = String(d.getMinutes()).padStart(2, "0");
                return (
                  <span
                    style={{ marginLeft: 6, fontSize: 11, opacity: 0.6 }}
                    // created_at is stamped when the run row is created, which for a
                    // queued run is the reservation time, not the moment training began
                    // (nothing on the row records that). Label it for what it is.
                    title={`${lang === "ja" ? "作成" : "Created"}: ${formatApiDate(run.created_at)}`}
                  >
                    {hh}:{mm}
                  </span>
                );
              })()}
            </div>
            {typeof run.best_f1 === "number" && (run.status === "completed" || run.status === "stopped" || run.status === "done") && run.has_model ? (
              <span className="run-f1-chip" style={{ color: run.best_f1 >= 0.7 ? "#30d158" : run.best_f1 >= 0.4 ? "#f0a040" : "#ff453a" }}>F1 {run.best_f1.toFixed(3)}</span>
            ) : <span />}
            <div className="row" style={{ gap: 2 }}>
              <button
                className="models-action-btn models-action-btn-danger"
                onClick={(event) => {
                  event.stopPropagation();
                  void onStopRun(run.run_id);
                }}
                disabled={run.status !== "running" && run.status !== "reserved" && run.status !== "preparing" && run.status !== "starting"}
                title={run.status === "reserved" ? t("training.cancelReserved") : t("training.abortRun")}
                data-desc={t("training.abortRunDesc")}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="4" y="4" width="16" height="16" rx="2"/>
                </svg>
              </button>
              <button
                className="models-action-btn models-action-btn-danger"
                onClick={(event) => {
                  event.stopPropagation();
                  void onDeleteRun(run.run_id);
                }}
                disabled={isLocked}
                title={isLocked ? t("training.lockedInResults") : t("training.delete")}
                data-desc={isLocked ? t("training.lockedInResults") : t("training.deleteRun.desc")}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                </svg>
              </button>
              {(run.status === "completed" || run.status === "stopped") && run.has_model && !run.fp16 && (
                <button
                  className="models-action-btn"
                  onClick={(event) => {
                    event.stopPropagation();
                    void onOptimizeRun(run.run_id);
                  }}
                  title={t("training.runs.optimizeTitle")}
                  data-desc={t("training.runs.optimizeDesc")}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                  </svg>
                </button>
              )}
              {(run.status === "completed" || run.status === "stopped" || run.status === "done") && run.has_model && onOpenResults && (
                <button
                  className={`models-action-btn run-result-cta${descMode && viewedRunIds && !viewedRunIds.includes(run.run_id) ? " results-btn-attention" : ""}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onOpenResults(run.run_id, run.model_name ?? `Run ${run.run_id.slice(0, 8)}`);
                  }}
                  title={t("projects.openResults")}
                  data-desc={t("projects.openResults")}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/>
                  </svg>
                  <span>{lang === "ja" ? "結果を見る" : "Results"}</span>
                </button>
              )}
            </div>
          </div>
          );
        })}
        {runsReady && effectiveRuns.length === 0 && <div className="muted">{t("training.noRuns")}</div>}
      </div>
    </div>
  );
}
