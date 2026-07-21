// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { fetchRuns } from "../api";
import { useTrainingStore } from "../store";
import {
  pickDefaultRunId,
  type TrainRunItem
} from "../utils";
import HyperparameterForm from "./HyperparameterForm";
import { useI18n } from "../i18n";
import type { ClassInfo } from "./types";
import MonitorState from "./components/MonitorState";
import ScorePanel from "./components/ScorePanel";
import SummaryCard from "./components/SummaryCard";
import RunList from "./components/RunList";
import { useLogStream } from "./hooks/useLogStream";
import { useTrainingScore } from "./hooks/useTrainingScore";
import { useBestRunInsights } from "./hooks/useBestRunInsights";
import { useTorchDevice } from "./hooks/useTorchDevice";
import { useModelSearch } from "./hooks/useModelSearch";
import { useTrainStart } from "./hooks/useTrainStart";
import { useRunActions } from "./hooks/useRunActions";
import LogsPanel from "./components/LogsPanel";

type TrainingProps = {
  projectId: string | null;
  active?: boolean;
  onOpenResults?: (runId: string, label: string) => void;
  valRatio?: number;
  testRatio?: number;
  exportFormat?: "coreml" | "coreml-updatable" | "onnx";
  showToast?: (msg: string) => void;
  lockedRunIds?: string[];
  viewedRunIds?: string[];
  descMode?: boolean;
};

export default React.memo(function Training({
  projectId,
  active,
  onOpenResults,
  valRatio,
  testRatio,
  exportFormat,
  showToast,
  lockedRunIds,
  viewedRunIds,
  descMode,
}: TrainingProps) {
  const { t, lang } = useI18n();
  const runs = useTrainingStore((s) => s.runs);
  const runsProjectId = useTrainingStore((s) => s.runsProjectId);
  const sharedSetRuns = useTrainingStore((s) => s.setRuns);
  const setStatus = showToast ?? (() => {});
  // Ref-based stale guard: loadRuns responses from a previous project are discarded.
  const currentProjectRef = useRef(projectId);
  currentProjectRef.current = projectId;
  // Torch device state + GPU-busy polling (extracted to useTorchDevice)
  const { torchState, updatingTorchDevice, handleTorchDeviceChange, gpuBusy, deviceSummary } =
    useTorchDevice(active, setStatus, t);
  void deviceSummary; // kept available; not currently rendered here
  const [selectedRunIdForLogs, setSelectedRunIdForLogs] = useState<string | null>(null);
  const monitorGridRef = useRef<HTMLDivElement>(null);
  const [monitorSplit, setMonitorSplit] = useState<number>(() => {
    const v = parseFloat(localStorage.getItem("seg-monitor-split") ?? "");
    return Number.isFinite(v) && v >= 28 && v <= 72 ? v : 55;
  });
  const startMonitorDrag = (e: React.MouseEvent) => {
    e.preventDefault();
    const grid = monitorGridRef.current;
    if (!grid) return;
    let latest = monitorSplit;
    const onMove = (ev: MouseEvent) => {
      const rect = grid.getBoundingClientRect();
      latest = Math.max(28, Math.min(72, ((ev.clientX - rect.left) / rect.width) * 100));
      setMonitorSplit(latest);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      try { localStorage.setItem("seg-monitor-split", String(Math.round(latest))); } catch { /* noop */ }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };
  const [hasHydratedRuns, setHasHydratedRuns] = useState(false);
  const [trainingMode, setTrainingMode] = useState<"standard" | "quick" | "transfer" | null>(null);
  // Model-search state + combo-library stats (extracted to useModelSearch)
  const {
    isSearchingModel,
    modelSearchResult,
    libraryStats,
    anchorElapsedSec,
    handleModelSearch,
    handleCancelModelSearch,
    handleRecalibrateETAs,
    resetModelSearch,
  } = useModelSearch(projectId, lang, t);

  // Report modal

  // Monitor tabs: logs vs score
  const [monitorTab, setMonitorTab] = useState<"logs" | "score">("logs");
  const score = useTrainingScore(projectId, runs, setStatus, t);

  // Run list sort: date (default), f1, name
  type RunSortKey = "date" | "f1" | "name";
  const [runSortKey, setRunSortKey] = useState<RunSortKey>("date");

  // Fetch classes on project change (needed for summary card class names)
  useEffect(() => {
    score.refreshClasses();
  }, [projectId, score.refreshClasses]);

  // Runs are "ready" only when they belong to the current project AND the fetch has completed.
  // This prevents stale data from a previous project from being rendered during project switch.
  const runsReady = hasHydratedRuns && runsProjectId === projectId;
  // Effective runs: suppress stale runs from a different project entirely.
  // Memoised so the reference is stable across renders (avoids useEffect dep churn).
  const effectiveRuns = useMemo(() => {
    const base = runsProjectId === projectId ? runs : [];
    if (base.length === 0) return base;
    const sorted = [...base];
    if (runSortKey === "f1") {
      sorted.sort((a, b) => (b.best_f1 ?? -1) - (a.best_f1 ?? -1));
    } else if (runSortKey === "name") {
      sorted.sort((a, b) => (a.model_name ?? a.run_id).localeCompare(b.model_name ?? b.run_id));
    }
    // "date" keeps the original order (already sorted by updated_at desc in loadRuns)
    return sorted;
  }, [runsProjectId, projectId, runs, runSortKey]);

  async function loadRuns(id: string) {
    try {
      const items = (await fetchRuns(id)) as TrainRunItem[];
      // Discard response if the user switched projects before the fetch returned.
      if (currentProjectRef.current !== id) return;
      const sorted = [...items].sort((a, b) => {
        const aTime = Date.parse(a.updated_at || a.created_at || "") || 0;
        const bTime = Date.parse(b.updated_at || b.created_at || "") || 0;
        return bTime - aTime;
      });
      // Avoid empty-flash: if server returned empty but we have a running
      // placeholder/real run, keep it until server catches up.
      if (sorted.length === 0 && isStartingTrain) return;
      // Server data replaces all placeholders — no __preparing_ survive
      sharedSetRuns(id, sorted);
      // Clear stoppingRunIds for runs that are no longer "running"
      setStoppingRunIds((prev) => {
        if (prev.size === 0) return prev;
        const next = new Set(prev);
        for (const rid of prev) {
          const run = sorted.find((r) => r.run_id === rid);
          if (!run || run.status !== "running") {
            next.delete(rid);
          }
        }
        return next.size === prev.size ? prev : next;
      });
    } catch (err) {
      setStatus(`Runs failed: ${(err as Error).message}`);
    } finally {
      setHasHydratedRuns(true);
    }
  }

  // Train-start flow: local placeholder run (extracted to useTrainStart)
  const {
    isStartingTrain,
    prepareReport,
    setPrepareReport,
    handleStartTrain,
  } = useTrainStart({
    projectId, valRatio, testRatio,
    effectiveRuns, sharedSetRuns, loadRuns, setSelectedRunIdForLogs,
    setRunLogsError: (msg) => logStream.setRunLogsError(msg),
    setStatus, lang, t,
  });

  // Per-run actions (stop / optimize / delete / export) — extracted to useRunActions
  const {
    stoppingRunIds,
    setStoppingRunIds,
    exportingModel,
    exportingOpenVino,
    handleStopRun,
    handleOptimizeRun,
    handleDeleteRun,
    handleExportModel,
    handleExportOpenVino,
  } = useRunActions({
    projectId, effectiveRuns, runs, sharedSetRuns, loadRuns,
    selectedRunIdForLogs, setSelectedRunIdForLogs,
    setRunLogsText: (msg) => logStream.setRunLogsText(msg),
    exportFormat, setStatus, t,
  });

  const loadScoreData = () => score.loadScoreData(selectedRunIdForLogs);

  useEffect(() => {
    setSelectedRunIdForLogs(null);
    logStream.setRunLogsText("");
    logStream.setRunLogsError("");
    setStatus("");
    setPrepareReport(null);
    setHasHydratedRuns(false);
    resetModelSearch();
    if (!projectId) return;
    void loadRuns(projectId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Also mark hydrated when the centralized poller in App.tsx has already
  // populated the store for this project (e.g. tab switch back to Training).
  useEffect(() => {
    if (runsProjectId === projectId && runs.length > 0) {
      setHasHydratedRuns(true);
    }
  }, [runs, runsProjectId, projectId]);

  // Runs polling is handled centrally by App.tsx → useTrainingStore

  useEffect(() => {
    if (!active) return;
    if (effectiveRuns.length === 0) {
      setSelectedRunIdForLogs(null);
      logStream.setRunLogsText("");
      return;
    }
    if (!selectedRunIdForLogs || !effectiveRuns.some((run) => run.run_id === selectedRunIdForLogs)) {
      setSelectedRunIdForLogs(pickDefaultRunId(effectiveRuns));
    }
  }, [active, effectiveRuns, selectedRunIdForLogs]); // eslint-disable-line react-hooks/exhaustive-deps

  // Log streaming hook
  const logStream = useLogStream(projectId, active, selectedRunIdForLogs, runs, sharedSetRuns);

  const trainingProgress = useMemo(() => {
    if (!logStream.runLogsText || !selectedRunIdForLogs) return null;
    const selectedRun = runs.find((r) => r.run_id === selectedRunIdForLogs);
    if (!selectedRun || selectedRun.status !== "running") return null;
    // Find all "Epoch X/Y" occurrences and use the last one
    const epochMatches = [...logStream.runLogsText.matchAll(/Epoch (\d+)\/(\d+)/g)];
    if (epochMatches.length === 0) return null;
    const last = epochMatches[epochMatches.length - 1]!;
    const current = parseInt(last[1]!, 10);
    const total = parseInt(last[2]!, 10);
    // Extract best val F1 — look for the latest "val F1: X.XXXX"
    const f1Matches = [...logStream.runLogsText.matchAll(/val F1: ([\d.]+)/g)];
    const bestF1 = f1Matches.length > 0
      ? Math.max(...f1Matches.map((m) => parseFloat(m[1]!)))
      : null;
    const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
    return { current, total, pct, bestF1 };
  }, [logStream.runLogsText, selectedRunIdForLogs, runs]);
  const selectedRun = effectiveRuns.find((run) => run.run_id === selectedRunIdForLogs) ?? null;
  const completedRunCount = effectiveRuns.filter((run) => run.status === "completed").length;
  const modelReadyCount = effectiveRuns.filter((run) => run.has_model).length;
  const selectedRunLabel = !runsReady
    ? t("training.syncingRuns")
    : selectedRun?.model_name || (selectedRun ? `Run ${selectedRun.run_id.slice(0, 8)}` : t("training.noRuns"));
  const selectedRunSummary = !runsReady
    ? t("training.syncingRuns")
    : !effectiveRuns.length
    ? t("training.startFirst")
    : !selectedRun
      ? t("training.selectRun")
      : selectedRun.status === "running"
        ? t("training.inProgress")
        : selectedRun.status === "reserved"
          ? t("training.queued")
          : selectedRun.has_model
            ? t("training.readyToInspect")
            : t("training.finishedNoExport");
  // Best-run insight state (best-by-F1, metrics, delta, convergence, per-class F1)
  const { bestRun, bestMetrics, f1Delta, convergenceStatus, perClassF1, dsStats } =
    useBestRunInsights(projectId, effectiveRuns);
  function moveRunSelection(offset: number) {
    if (effectiveRuns.length === 0) return;
    const currentIndex = effectiveRuns.findIndex((run) => run.run_id === selectedRunIdForLogs);
    const nextIndex = currentIndex < 0
      ? 0
      : Math.max(0, Math.min(effectiveRuns.length - 1, currentIndex + offset));
    const nextRun = effectiveRuns[nextIndex];
    if (nextRun) setSelectedRunIdForLogs(nextRun.run_id);
  }

  if (!projectId) {
    return <div className="muted">{t("projects.noProject")}</div>;
  }
  const hasRunningRun = gpuBusy;

  return (
    <div className="training-layout">
      <HyperparameterForm
        projectId={projectId}
        isStartingTrain={isStartingTrain}
        hasRunningRun={hasRunningRun}
        onStartTrain={handleStartTrain}
        torchState={torchState}
        onTorchDeviceChange={handleTorchDeviceChange}
        updatingTorchDevice={updatingTorchDevice}
        showToast={setStatus}
        runs={effectiveRuns}
        prepareReport={prepareReport}
        onModelSearch={handleModelSearch}
        onCancelModelSearch={handleCancelModelSearch}
        isSearching={isSearchingModel}
        libraryStats={libraryStats}
        trainingMode={trainingMode}
        onTrainingModeChange={setTrainingMode}
      />
      <SummaryCard
        bestRun={bestRun}
        f1Delta={f1Delta}
        bestMetrics={bestMetrics}
        convergenceStatus={convergenceStatus}
        perClassF1={perClassF1}
        scoreClasses={score.scoreClasses}
        dsStats={dsStats}
        effectiveRuns={effectiveRuns}
        completedRunCount={completedRunCount}
        modelReadyCount={modelReadyCount}
        selectedRunLabel={selectedRunLabel}
        selectedRunSummary={selectedRunSummary}
        onOpenResults={onOpenResults}
        lang={lang}
        t={t}
      />
      <div className="training-monitor-grid" ref={monitorGridRef} style={{ gridTemplateColumns: `minmax(0, ${monitorSplit}fr) 12px minmax(0, ${100 - monitorSplit}fr)` }}>
        <RunList
          effectiveRuns={effectiveRuns}
          selectedRunIdForLogs={selectedRunIdForLogs}
          setSelectedRunIdForLogs={setSelectedRunIdForLogs}
          runsReady={runsReady}
          runSortKey={runSortKey}
          setRunSortKey={setRunSortKey}
          trainingProgress={trainingProgress}
          stoppingRunIds={stoppingRunIds}
          lockedRunIds={lockedRunIds}
          viewedRunIds={viewedRunIds}
          descMode={descMode}
          exportingModel={exportingModel}
          exportingOpenVino={exportingOpenVino}
          onStopRun={handleStopRun}
          onDeleteRun={handleDeleteRun}
          onOptimizeRun={handleOptimizeRun}
          onExportModel={handleExportModel}
          onExportOpenVino={handleExportOpenVino}
          onOpenResults={onOpenResults}
          moveRunSelection={moveRunSelection}
          selectedRun={selectedRun}
          modelReadyCount={modelReadyCount}
          exportFormat={exportFormat}
          lang={lang}
          t={t}
        />
        <div
          className="monitor-splitter"
          role="separator"
          aria-orientation="vertical"
          title={lang === "ja" ? "ドラッグで幅を調整" : "Drag to resize"}
          onMouseDown={startMonitorDrag}
        />
        <div className="section training-logs-section" style={{ marginBottom: 0 }}>
          <div className="training-monitor-tabs">
            <button
              className={monitorTab === "logs" ? "active" : ""}
              onClick={() => setMonitorTab("logs")}
              data-desc={t("training.logs")}
            >
              {selectedRunIdForLogs ? `${t("training.logs")} ${selectedRunIdForLogs.slice(0, 8)}` : t("training.logs")}
            </button>
            <button
              className={monitorTab === "score" ? "active" : ""}
              data-desc={t("training.score")}
              onClick={() => {
                setMonitorTab("score");
                if (selectedRunIdForLogs && score.scoreRunId !== selectedRunIdForLogs && !score.scoreLoading) {
                  void loadScoreData();
                }
              }}
            >
              {t("training.score")}
            </button>
          </div>
          {monitorTab === "logs" ? (
            <LogsPanel
              selectedRunIdForLogs={selectedRunIdForLogs}
              selectedRun={selectedRun}
              logStream={logStream}
              modelSearchResult={modelSearchResult}
              isSearchingModel={isSearchingModel}
              anchorElapsedSec={anchorElapsedSec}
              onRecalibrateETAs={handleRecalibrateETAs}
              lang={lang}
              t={t}
            />
          ) : (
            <ScorePanel
              scoreData={score.scoreData}
              scoreLoading={score.scoreLoading}
              scoreProgress={score.scoreProgress}
              scoreClasses={score.scoreClasses}
              scoreImageNames={score.scoreImageNames}
              scoreTotalImages={score.scoreTotalImages}
              scoreSortKey={score.scoreSortKey}
              scoreSortAsc={score.scoreSortAsc}
              onSortChange={(key) => {
                if (key === score.scoreSortKey) {
                  score.setScoreSortAsc(!score.scoreSortAsc);
                } else {
                  score.setScoreSortKey(key);
                  score.setScoreSortAsc(key === "name");
                }
              }}
              onReload={loadScoreData}
              selectedRunIdForLogs={selectedRunIdForLogs}
              hasModel={!!effectiveRuns.find((r) => r.run_id === selectedRunIdForLogs)?.has_model}
              projectId={projectId}
            />
          )}
        </div>
      </div>
    </div>
  );
})
