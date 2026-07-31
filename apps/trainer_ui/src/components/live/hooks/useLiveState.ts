// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, API_ORIGIN } from "../../../api";
import { useI18n } from "../../../i18n";
import { parseApiDate } from "../../../time";
import type { ProjectOption, RunOption, SessionInfo, InferenceResult, InferenceStats, CameraState } from "../types";

export function useLiveState(
  parentProjectId: string,
  parentProjectName: string | undefined,
  active: boolean,
  targetRunId: string | undefined,
  toast: (msg: string) => void,
) {
  const { t } = useI18n();
  const [allProjects, setAllProjects] = useState<ProjectOption[]>([]);
  const [localProjectId, setLocalProjectId] = useState(parentProjectId);
  const [runs, setRuns] = useState<RunOption[]>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [results, setResults] = useState<InferenceResult[]>([]);
  const [stats, setStats] = useState<InferenceStats>({ total: 0, ok: 0, ng: 0, avgMs: 0 });
  const [switchHint, setSwitchHint] = useState<string | null>(null);

  // Effective project ID: local selection overrides parent
  const effectiveProjectId = localProjectId || parentProjectId;

  // On mount: clean up orphan backend camera/session (e.g. after page reload)
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_ORIGIN}/v2/camera/status`);
        if (res.ok) {
          const { state } = await res.json();
          if (state !== "IDLE") {
            console.info("LiveInspection: cleaning up orphan camera (state=%s)", state);
            await fetch(`${API_ORIGIN}/v2/camera/stop`, { method: "POST" });
          }
        }
      } catch { /* ignore */ }
      try {
        const res = await fetch(`${API_ORIGIN}/v2/session/status`);
        if (res.ok) {
          const data = await res.json();
          if (data.active_session || data.loaded) {
            console.info("LiveInspection: cleaning up orphan session");
            await fetch(`${API_ORIGIN}/v2/session/stop`, { method: "POST" });
          }
        }
      } catch { /* ignore */ }
    })();
  }, []);

  // Sync parent project selection when it changes (only if no local override)
  useEffect(() => {
    if (parentProjectId && !localProjectId) {
      setLocalProjectId(parentProjectId);
    }
  }, [parentProjectId]);

  // Fetch all projects for dropdown
  useEffect(() => {
    if (!active) return;
    fetch(`${API_BASE}/projects`)
      .then((r) => r.json())
      .then((data: any[]) => {
        setAllProjects(data.map((p) => ({ id: p.id, name: p.name })));
      })
      .catch((err) => { console.warn("LiveInspection: fetch projects failed:", err); });
  }, [active]);

  // Fetch runs on project change
  useEffect(() => {
    if (!effectiveProjectId || !active) return;
    setRuns([]);
    setSelectedRun("");
    fetch(`${API_BASE}/projects/${effectiveProjectId}/train/runs`)
      .then((r) => r.json())
      .then((data: any[]) => {
        const completed = data
          .filter((r) => r.status === "completed")
          .map((r) => {
            const name = r.model_name || r.name || r.run_id.slice(0, 8);
            const f1 = r.best_f1 != null ? `F1=${(r.best_f1 as number).toFixed(3)}` : "";
            const created = parseApiDate(r.created_at);
            const date = created ? created.toLocaleDateString("ja-JP", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
            const parts = [name, f1, date].filter(Boolean);
            return {
              run_id: r.run_id,
              label: parts.join(" | "),
              model_name: r.model_name,
              best_f1: r.best_f1,
              best_miou: r.best_miou,
              created_at: r.created_at,
            };
          });
        setRuns(completed);
        if (completed.length > 0) {
          setSelectedRun(completed[0].run_id);
        }
      })
      .catch((err) => { console.warn("LiveInspection: fetch runs failed:", err); });
  }, [effectiveProjectId, active]);

  // Auto-select run when navigating from Results tab
  const appliedTargetRef = useRef<string>("");
  useEffect(() => {
    if (!targetRunId || targetRunId === appliedTargetRef.current || runs.length === 0) return;
    const match = runs.find((r) => r.run_id === targetRunId);
    if (!match) return;
    if (session) {
      if (selectedRun !== targetRunId) {
        const targetLabel = match.label;
        setSwitchHint(t("live.switchHint").replace("{label}", targetLabel));
      }
    } else {
      appliedTargetRef.current = targetRunId;
      setSelectedRun(targetRunId);
      setSwitchHint(null);
    }
  }, [targetRunId, runs, session, selectedRun, t]);

  // Start session (with GPU busy check)
  const startSession = useCallback(async (cameraState: CameraState, setCameraState: (s: CameraState) => void) => {
    if (!selectedRun) return;
    // Check if GPU is busy with training
    try {
      const statusRes = await fetch(`${API_BASE}/train/global-status`);
      if (statusRes.ok) {
        const status = await statusRes.json();
        if (status.gpu_busy) {
          toast("GPU is busy with training.");
          return;
        }
      }
    } catch (err) { console.warn("LiveInspection: GPU status check failed:", err); }
    setConnecting(true);
    try {
      const res = await fetch(`${API_ORIGIN}/v2/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: effectiveProjectId, run_id: selectedRun }),
      });
      if (!res.ok) throw new Error(await res.text());
      const info: SessionInfo = await res.json();
      setSession(info);
      setResults([]);
      setStats({ total: 0, ok: 0, ng: 0, avgMs: 0 });
      toast(`Session started (${info.device}, warmup ${Math.round(info.warmup_ms)}ms)`);

      // If camera is running, attach inference
      if (cameraState === "PREVIEW") {
        try {
          await fetch(`${API_ORIGIN}/v2/camera/attach`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ project_id: effectiveProjectId, run_id: selectedRun }),
          });
          setCameraState("INSPECT");
        } catch (e) {
          console.error("Camera attach failed:", e);
          toast(`Camera attach failed: ${(e as Error).message}`);
        }
      }
    } catch (e) {
      console.error("Session start failed:", e);
      toast(`Session start failed: ${(e as Error).message}`);
    } finally {
      setConnecting(false);
    }
  }, [effectiveProjectId, selectedRun, toast]);

  // Stop session
  const stopSession = useCallback(async (
    cameraState: CameraState,
    setCameraState: (s: CameraState) => void,
  ) => {
    // Detach inference from camera if attached
    if (cameraState === "INSPECT") {
      try {
        await fetch(`${API_ORIGIN}/v2/camera/detach`, { method: "POST" });
        setCameraState("PREVIEW");
      } catch (err) { console.warn("LiveInspection: camera detach failed:", err); }
    }

    try {
      await fetch(`${API_ORIGIN}/v2/session/stop`, { method: "POST" });
    } catch (err) { console.warn("LiveInspection: session stop failed:", err); }
    setSession(null);
    setSwitchHint(null);
    if (targetRunId && targetRunId !== appliedTargetRef.current) {
      appliedTargetRef.current = targetRunId;
      setSelectedRun(targetRunId);
    }
  }, [targetRunId]);

  const handleProjectChange = useCallback((projectId: string) => {
    setLocalProjectId(projectId);
    setRuns([]);
    setSelectedRun("");
  }, []);

  const addResult = useCallback((result: InferenceResult) => {
    setResults((prev) => {
      const dropped = prev.length >= 199 ? prev.slice(0, prev.length - 199) : [];
      for (const old of dropped) { if (old.imageUrl) URL.revokeObjectURL(old.imageUrl); }
      return [...prev.slice(-199), result];
    });
    setStats((prev) => {
      const total = prev.total + 1;
      const ok = prev.ok + (result.judgement === "OK" ? 1 : 0);
      const ng = prev.ng + (result.judgement === "NG" ? 1 : 0);
      const avgMs = (prev.avgMs * prev.total + (result.latency_ms.total || 0)) / total;
      return { total, ok, ng, avgMs };
    });
  }, []);

  const clearResults = useCallback(() => {
    for (const r of results) { if (r.imageUrl) URL.revokeObjectURL(r.imageUrl); }
    setResults([]);
    setStats({ total: 0, ok: 0, ng: 0, avgMs: 0 });
  }, [results]);

  const selectedProjectName = allProjects.find((p) => p.id === effectiveProjectId)?.name ?? parentProjectName;

  return {
    allProjects,
    effectiveProjectId,
    runs,
    selectedRun,
    setSelectedRun,
    session,
    connecting,
    results,
    stats,
    switchHint,
    selectedProjectName,
    startSession,
    stopSession,
    handleProjectChange,
    addResult,
    clearResults,
  };
}
