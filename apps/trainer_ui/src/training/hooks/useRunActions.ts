// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useState } from "react";
import {
  deleteRun,
  exportCoreML,
  exportCoreMLUpdatable,
  exportOnnx,
  exportOpenVino,
  type OpenVinoPrecision,
  optimizeRun,
  stopRun,
} from "../../api";
import type { TrainRunItem } from "../../utils";
import type { TranslationKey } from "../../i18n";

type UseRunActionsArgs = {
  projectId: string | null;
  effectiveRuns: TrainRunItem[];
  runs: TrainRunItem[];
  sharedSetRuns: (projectId: string, runs: TrainRunItem[]) => void;
  loadRuns: (id: string) => Promise<void>;
  selectedRunIdForLogs: string | null;
  setSelectedRunIdForLogs: (id: string | null) => void;
  setRunLogsText: (msg: string) => void;
  exportFormat?: "coreml" | "coreml-updatable" | "onnx";
  setStatus: (msg: string) => void;
  t: (key: TranslationKey) => string;
};

// Extracted verbatim from Training.tsx (pre-OSS refactor): per-run actions —
// stop, speed-optimize, delete, and the CoreML/ONNX/
// OpenVINO exports — plus their stopping/exporting UI state.
export function useRunActions({
  projectId,
  effectiveRuns,
  runs,
  sharedSetRuns,
  loadRuns,
  selectedRunIdForLogs,
  setSelectedRunIdForLogs,
  setRunLogsText,
  exportFormat,
  setStatus,
  t,
}: UseRunActionsArgs) {
  // Track runs that have been requested to stop (show "停止中" until status changes)
  const [stoppingRunIds, setStoppingRunIds] = useState<Set<string>>(new Set());
  const [exportingModel, setExportingModel] = useState(false);
  const [exportingOpenVino, setExportingOpenVino] = useState(false);

  async function handleStopRun(runId: string) {
    if (!projectId) return;
    const ok = window.confirm(t("training.confirmStop").replace("{runId}", runId.slice(0, 8)));
    if (!ok) return;
    try {
      await stopRun(projectId, runId);
      setStoppingRunIds((prev) => new Set(prev).add(runId));
      setStatus(t("training.stoppingToast"));
      await loadRuns(projectId);
    } catch (err) {
      setStatus(t("training.stopError").replace("{msg}", (err as Error).message));
    }
  }

  async function handleOptimizeRun(runId: string) {
    if (!projectId) return;
    setStatus(t("training.runs.optimizing"));
    try {
      const result = await optimizeRun(projectId, runId);
      setStatus(t("training.runs.optimizeDone").replace("{name}", result.model_name));
    } catch (err) {
      setStatus(t("training.runs.optimizeFailed").replace("{msg}", (err as Error).message));
    }
  }

  async function handleDeleteRun(runId: string) {
    if (!projectId) return;
    const ok = window.confirm(t("training.confirmDeleteRun").replace("{runId}", runId.slice(0, 8)));
    if (!ok) return;
    try {
      await deleteRun(projectId, runId);
      setStatus(`Deleted run ${runId.slice(0, 8)}.`);
    } catch (err) {
      const msg = (err as Error).message || "";
      if (msg.includes("not found") || msg.includes("404")) {
        // Run already gone from DB — just refresh the list
        setStatus(`Run ${runId.slice(0, 8)} not found — removing from list.`);
      } else {
        setStatus(`Delete failed: ${msg}`);
        return;
      }
    }
    if (selectedRunIdForLogs === runId) {
      setSelectedRunIdForLogs(null);
      setRunLogsText("");
    }
    // Immediately remove from local list for instant UI feedback
    sharedSetRuns(projectId, runs.filter((r) => r.run_id !== runId));
    // Then refresh from server
    await loadRuns(projectId);
  }

  async function handleExportModel() {
    if (!projectId || !selectedRunIdForLogs || exportingModel) return;
    setExportingModel(true);
    try {
      const fmt = exportFormat ?? "coreml";
      if (fmt === "onnx") {
        await exportOnnx(projectId, selectedRunIdForLogs);
      } else if (fmt === "coreml-updatable") {
        await exportCoreMLUpdatable(projectId, selectedRunIdForLogs);
      } else {
        await exportCoreML(projectId, selectedRunIdForLogs);
      }
      setStatus(`Exported (${fmt}) run ${selectedRunIdForLogs.slice(0, 8)}.`);
    } catch (err) {
      setStatus(`Export failed: ${(err as Error).message}`);
    } finally {
      setExportingModel(false);
    }
  }

  async function handleExportOpenVino(precision: OpenVinoPrecision) {
    if (!projectId || !selectedRunIdForLogs || exportingOpenVino) return;
    setExportingOpenVino(true);
    try {
      await exportOpenVino(projectId, selectedRunIdForLogs, precision);
      setStatus(`Exported OpenVINO (${precision}) run ${selectedRunIdForLogs.slice(0, 8)}.`);
    } catch (err) {
      setStatus(`OpenVINO export failed: ${(err as Error).message}`);
    } finally {
      setExportingOpenVino(false);
    }
  }

  return {
    stoppingRunIds,
    setStoppingRunIds,
    exportingModel,
    exportingOpenVino,
    handleStopRun,
    handleOptimizeRun,
    handleDeleteRun,
    handleExportModel,
    handleExportOpenVino,
  };
}
