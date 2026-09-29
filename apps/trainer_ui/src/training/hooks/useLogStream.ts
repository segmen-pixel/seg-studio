// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef, useState } from "react";
import { fetchRunLogs } from "../../api";
import { useTrainingStore } from "../../store";
import type { TrainRunItem } from "../../utils";

export type LogStreamResult = {
  runLogsText: string;
  setRunLogsText: React.Dispatch<React.SetStateAction<string>>;
  runLogsError: string;
  setRunLogsError: React.Dispatch<React.SetStateAction<string>>;
  isRunLogsLoading: boolean;
  setIsRunLogsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  logPreRef: React.RefObject<HTMLPreElement>;
};

export function useLogStream(
  projectId: string | null,
  active: boolean | undefined,
  selectedRunIdForLogs: string | null,
  runs: TrainRunItem[],
  sharedSetRuns: (projectId: string, runs: TrainRunItem[]) => void,
): LogStreamResult {
  const [runLogsText, setRunLogsText] = useState("");
  const [runLogsError, setRunLogsError] = useState("");
  const [isRunLogsLoading, setIsRunLogsLoading] = useState(false);

  const logOffsetRef = useRef(0);
  const logPreRef = useRef<HTMLPreElement>(null);

  // Auto-scroll log panel to bottom when new log text arrives
  useEffect(() => {
    const el = logPreRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [runLogsText]);

  // Track which run/project the cached logs belong to so we can preserve them on tab re-entry
  const logsForRunRef = useRef<string | null>(null);
  const logsForProjectRef = useRef<string | null>(null);
  // Stable ref for run status — avoids including `runs` in the log effect deps
  const runsRef = useRef(runs);
  useEffect(() => { runsRef.current = runs; }, [runs]);

  useEffect(() => {
    if (!projectId || !active || !selectedRunIdForLogs) {
      // Tab hidden or no run selected: stop fetching but KEEP log content to avoid flicker on re-entry
      setRunLogsError("");
      return;
    }
    const runChanged =
      logsForRunRef.current !== selectedRunIdForLogs ||
      logsForProjectRef.current !== projectId;
    if (runChanged) {
      // Switched to a different run or project — reset cached logs
      setRunLogsText("");
      setRunLogsError("");
      logOffsetRef.current = 0;
      logsForRunRef.current = selectedRunIdForLogs;
      logsForProjectRef.current = projectId;
    }
    const selectedRunForLogs = runsRef.current.find((run) => run.run_id === selectedRunIdForLogs) ?? null;
    const isTerminalRun = !!selectedRunForLogs && ["completed", "failed", "stopped"].includes(selectedRunForLogs.status);
    const shouldUseLiveUpdates = !!selectedRunForLogs && !isTerminalRun;
    let disposed = false;
    let firstLoad = logOffsetRef.current === 0;

    // --- HTTP polling fallback ---
    let pollTimer: number | undefined;
    const pollLoad = async () => {
      // Skip loading indicator for terminal runs — they resolve instantly and
      // showing "Loading logs" for a completed run with no log file is misleading.
      if (firstLoad && !isTerminalRun) setIsRunLogsLoading(true);
      try {
        const offset = logOffsetRef.current;
        const payload = await fetchRunLogs(projectId, selectedRunIdForLogs, offset > 0 ? offset : undefined);
        if (disposed) return;
        const newText = payload.log || "";
        const total = payload.total ?? 0;
        if (offset > 0 && total >= offset) {
          if (newText.length > 0) setRunLogsText((prev) => prev + newText);
          logOffsetRef.current = total;
        } else {
          setRunLogsText(newText);
          logOffsetRef.current = total;
        }
        setRunLogsError("");
      } catch (err) {
        if (disposed) return;
        const msg = (err as Error).message || "";
        // Network errors (server crash, connection refused) get a friendlier message
        if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("ERR_CONNECTION_REFUSED")) {
          setRunLogsError("Server connection lost. The backend may have crashed -- check server.log for details.");
        } else {
          setRunLogsError(`Log fetch failed: ${msg}`);
        }
      } finally {
        if (!disposed && firstLoad) { setIsRunLogsLoading(false); firstLoad = false; }
      }
    };

    function startPollingFallback() {
      if (disposed || pollTimer) return;
      void pollLoad();
      pollTimer = window.setInterval(() => { void pollLoad(); }, 2000);
    }

    // Safety timeout: clear loading indicator after 5s even if fetch/WS hangs
    const loadingTimeout = window.setTimeout(() => {
      if (!disposed) setIsRunLogsLoading(false);
    }, 5000);

    // WebSocket is opened AFTER the initial HTTP load completes, with
    // ?offset=<already-fetched-bytes> so the server streams only the delta.
    // Otherwise the WS replays the entire log file on connect, which gets
    // appended on top of what pollLoad already set => duplicated output.
    let ws: WebSocket | null = null;
    const openWebSocket = () => {
      if (disposed || !shouldUseLiveUpdates) return;
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsBase = `${wsProtocol}//${window.location.host}/api/v1/ws/train/${projectId}/${selectedRunIdForLogs}`;
      const currentOffset = logOffsetRef.current;
      const wsUrl = currentOffset > 0 ? `${wsBase}?offset=${currentOffset}` : wsBase;
      try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
          if (firstLoad) { setIsRunLogsLoading(false); firstLoad = false; }
        };
        ws.onmessage = (event) => {
          if (disposed) return;
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "log" && msg.data) {
              setRunLogsText((prev) => prev + msg.data);
              if (typeof msg.total === "number") logOffsetRef.current = msg.total;
            } else if (msg.type === "status" && msg.status && msg.status !== "unknown") {
              // "unknown" means the server has no record of this run (e.g. a
              // mocked or externally-managed run) — never clobber known state
              // (such a run would drop out of the status-polling filter).
              // Update run status in shared store
              const { runs: currentRuns } = useTrainingStore.getState();
              const updated = currentRuns.map((r) =>
                r.run_id === selectedRunIdForLogs ? { ...r, status: msg.status } : r
              );
              if (JSON.stringify(updated) !== JSON.stringify(currentRuns)) {
                sharedSetRuns(projectId, updated);
              }
            }
            // type === "done" is handled by onclose
          } catch (err) {
            console.warn("Training: WS message parse error:", err);
          }
        };
        ws.onerror = () => {
          // WebSocket failed, fall back to polling
          if (!disposed) startPollingFallback();
        };
        ws.onclose = () => {
          ws = null;
          // If not disposed and not terminal, fall back to polling
          if (!disposed && !pollTimer) startPollingFallback();
        };
      } catch (err) {
        console.warn("Training: WebSocket construction failed:", err);
        startPollingFallback();
      }
    };

    // Run the HTTP prefix load, then open the WS once we know the offset.
    void pollLoad().then(() => { if (!disposed) openWebSocket(); });

    return () => {
      disposed = true;
      window.clearTimeout(loadingTimeout);
      if (pollTimer) window.clearInterval(pollTimer);
      if (ws) { try { ws.close(); } catch (err) { console.warn("Training: WS close error:", err); } }
      setIsRunLogsLoading(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, active, selectedRunIdForLogs]);

  return { runLogsText, setRunLogsText, runLogsError, setRunLogsError, isRunLogsLoading, setIsRunLogsLoading, logPreRef };
}
