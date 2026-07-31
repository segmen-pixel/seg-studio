// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef, useState } from "react";
import { API_BASE, fetchRuns } from "../../api";
import { useTrainingStore } from "../../store";
import type { TrainRunItem } from "../../utils";
import type { TrainProgressInfo, TrainingStatusInfo, TabId } from "../types";
import { parseApiDate } from "../../time";
import { visibleInterval } from "./useVisibleInterval";

/**
 * Global training/inference status polling + per-project runs polling.
 */
export function useGlobalPolling(
  selectedProjectId: string | null,
  activeTab: TabId,
  lang: string,
) {
  const [trainProgress, setTrainProgress] = useState<TrainProgressInfo | null>(null);
  const [gpuBusy, setGpuBusy] = useState(false);
  const [trainProjectId, setTrainProjectId] = useState<string | null>(null);
  const [inferStatus, setInferStatus] = useState("");
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusInfo>({
    state: "idle", running: 0, total: 0, percent: null, etaMinutes: null,
  });

  // Global status polling (5s)
  const prevPollRef = useRef<{
    gpuBusy: boolean; pct: number; epoch: number; total: number;
    projectId: string | null; inferKey: string;
  }>({ gpuBusy: false, pct: -1, epoch: -1, total: -1, projectId: null, inferKey: "" });

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/train/global-status`);
        if (res.ok) {
          const d = await res.json();
          const p = d.progress;
          const prev = prevPollRef.current;
          const newBusy = !!d.gpu_busy;
          const newPct = p?.pct ?? -1;
          const newEpoch = p?.epoch ?? -1;
          const newTotal = p?.total_epochs ?? -1;
          const newProjId = d.project_id ?? null;
          if (newBusy !== prev.gpuBusy) { setGpuBusy(newBusy); prev.gpuBusy = newBusy; }
          if (newPct !== prev.pct || newEpoch !== prev.epoch || newTotal !== prev.total) {
            setTrainProgress(p ?? null);
            prev.pct = newPct; prev.epoch = newEpoch; prev.total = newTotal;
          }
          if (newProjId !== prev.projectId) { setTrainProjectId(newProjId); prev.projectId = newProjId; }
          const inf = d.inference;
          const inferKey = inf?.active ? `${inf.completed}/${inf.total}` : "";
          if (inferKey !== prev.inferKey) {
            prev.inferKey = inferKey;
            if (inf?.active) {
              setInferStatus(`${lang === "ja" ? "推論中" : "Inferring"} ${inf.completed}/${inf.total}`);
            } else if (prev.inferKey && !inferKey) {
              setInferStatus((cur) => cur.startsWith("推論中") || cur.startsWith("Inferring") ? "" : cur);
            }
          }
        }
      } catch { /* ignore */ }
    };
    poll();
    return visibleInterval(poll, 5_000);
  }, [lang]);

  // Per-project runs polling (15s training / 30s idle)
  const pollIntervalMs = trainingStatus.state === "running" ? 15000 : 30000;
  useEffect(() => {
    const { setRuns, clearRuns } = useTrainingStore.getState();
    async function pollTrainingStatus() {
      if (!selectedProjectId) {
        setTrainingStatus({ state: "idle", running: 0, total: 0, percent: null, etaMinutes: null });
        clearRuns();
        return;
      }
      try {
        const rawRuns = (await fetchRuns(selectedProjectId)) as TrainRunItem[];
        const sorted = [...rawRuns].sort((a, b) => {
          const aTime = parseApiDate(a.updated_at || a.created_at)?.getTime() ?? 0;
          const bTime = parseApiDate(b.updated_at || b.created_at)?.getTime() ?? 0;
          return bTime - aTime;
        });
        setRuns(selectedProjectId, sorted);
        const runningRuns = sorted.filter((r) => r.status === "running");
        setTrainingStatus({
          state: runningRuns.length > 0 ? "running" : "idle",
          running: runningRuns.length,
          total: sorted.length,
          percent: null,
          etaMinutes: null,
        });
      } catch {
        setTrainingStatus({ state: "error", running: 0, total: 0, percent: null, etaMinutes: null });
      }
    }
    pollTrainingStatus();
    return visibleInterval(pollTrainingStatus, pollIntervalMs);
  }, [selectedProjectId, activeTab, pollIntervalMs]);

  return {
    trainProgress, gpuBusy, trainProjectId,
    inferStatus, setInferStatus,
    trainingStatus,
  } as const;
}
