// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useRef, useState } from "react";
import {
  prepareAnnotateDataset,
  startTraining,
} from "../../api";
import type { TrainRunItem } from "../../utils";
import type { TranslationKey } from "../../i18n";
import type { StartPhase } from "../components/TrainStartDialog";

type UseTrainStartArgs = {
  projectId: string | null;
  valRatio?: number;
  testRatio?: number;
  effectiveRuns: TrainRunItem[];
  sharedSetRuns: (projectId: string, runs: TrainRunItem[]) => void;
  loadRuns: (id: string) => Promise<void>;
  setSelectedRunIdForLogs: (id: string | null) => void;
  setRunLogsError: (msg: string) => void;
  setStatus: (msg: string) => void;
  lang: string;
  t: (key: TranslationKey) => string;
};

// Extracted verbatim from Training.tsx (pre-OSS refactor): the train-start
// flow — local start with a "preparing" placeholder run
// path through the GPU dialog, and the dataset prepare report.
export function useTrainStart({
  projectId,
  valRatio,
  testRatio,
  effectiveRuns,
  sharedSetRuns,
  loadRuns,
  setSelectedRunIdForLogs,
  setRunLogsError,
  setStatus,
  lang,
  t,
}: UseTrainStartArgs) {
  const [isStartingTrain, setIsStartingTrain] = useState(false);
  // Drives TrainStartDialog. The wait before the first training log is dataset
  // preparation, which is minutes on a large project; a line of text in the run
  // list reads as the click having been ignored.
  const [startProgress, setStartProgress] = useState<null | {
    phase: StartPhase;
    detail: string;
    startedAt: number;
  }>(null);
  const [prepareReport, setPrepareReport] = useState<null | {
    train_count: number;
    val_count: number;
    with_mask: number;
    auto_val_from_train_count?: number;
  }>(null);

  async function handleStartTrain(payload: Record<string, unknown>) {
    if (!projectId || isStartingTrain) return;

    setIsStartingTrain(true);
    const startedAt = Date.now();
    setStartProgress({ phase: "preparing", detail: "", startedAt });
    // Immediately add a placeholder run so the user sees feedback
    const placeholderId = `__preparing_${Date.now()}`;
    const modelLabel = payload.model_name as string || "";
    const placeholderRun: TrainRunItem = {
      run_id: placeholderId,
      status: "preparing",
      model_name: modelLabel || (lang === "ja" ? "準備中..." : "Preparing..."),
    };
    // Helper to update placeholder status text
    const updatePlaceholder = (status: string, detail: string) => {
      const updated = { ...placeholderRun, status: status as any, model_name: detail };
      sharedSetRuns(projectId!, [updated, ...effectiveRuns]);
    };

    sharedSetRuns(projectId, [placeholderRun, ...effectiveRuns]);
    setSelectedRunIdForLogs(placeholderId);
    try {
      updatePlaceholder("preparing", lang === "ja" ? "データセット準備中..." : "Preparing dataset...");
      const prep = await prepareAnnotateDataset(projectId, valRatio, testRatio);
      if (prep?.report) {
        setPrepareReport(prep.report);
        const r = prep.report;
        const detail = lang === "ja"
          ? `データセット準備完了 (train=${r.train_count}, val=${r.val_count})`
          : `Dataset ready (train=${r.train_count}, val=${r.val_count})`;
        updatePlaceholder("preparing", detail);
        setStartProgress((p) => (p ? { ...p, detail } : p));
      }

      updatePlaceholder("starting", lang === "ja" ? "GPU確保・学習設定中..." : "Claiming GPU...");
      setStartProgress((p) => (p ? { ...p, phase: "claiming" } : p));
      const started = await startTraining(projectId, payload);
      const isReserved = started && typeof started === "object" && "status" in started && started.status === "reserved";
      setStatus(isReserved ? t("training.queuedToast") : t("training.startedToast"));
      // Queued behind another job: keep the dialog up, since there is still
      // nothing to watch. Actually running: the log panel takes over.
      setStartProgress((p) => (p && isReserved ? { ...p, phase: "queued", detail: "" } : null));
      setRunLogsError("");
      const startedRunId =
        started && typeof started === "object" && "run_id" in started && typeof started.run_id === "string"
          ? started.run_id
          : null;
      if (startedRunId) {
        const realRun: TrainRunItem = {
          run_id: startedRunId,
          status: isReserved ? "reserved" : "running",
          model_name: (started as any).model_name || modelLabel || (lang === "ja" ? "学習中..." : "Training..."),
        };
        sharedSetRuns(projectId!, [realRun, ...effectiveRuns]);
        setSelectedRunIdForLogs(startedRunId);
      }
      await loadRuns(projectId);
    } catch (err) {
      const raw = (err as Error).message;
      try {
        const parsed = JSON.parse(raw);
        const detail = parsed?.detail;
        if (detail?.run_id && typeof detail.run_id === "string") {
          setStatus(`Train already running: ${detail.run_id.slice(0, 8)}`);
          setStartProgress(null);
          setSelectedRunIdForLogs(detail.run_id);
          await loadRuns(projectId);
          return;
        }
      } catch (_parseErr) {
        // Keep original error text when payload is not JSON.
      }
      setStatus(`Train failed: ${raw}`);
      setStartProgress({ phase: "failed", detail: raw, startedAt });
    } finally {
      setIsStartingTrain(false);
    }
  }


  return {
    isStartingTrain,
    prepareReport,
    setPrepareReport,
    handleStartTrain,
    startProgress,
    dismissStartProgress: useCallback(() => setStartProgress(null), []),
  };
}
