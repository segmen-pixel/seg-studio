// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useRef, useState } from "react";
import { fetchFleetStatus, type FleetItem, type QueueItem } from "../api";
import { visibleInterval } from "../app/hooks/useVisibleInterval";

/** Format large numbers compactly: 70000 → "70K" */
function _fmtK(n: number): string {
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return String(n);
}

type Props = {
  onJump: (projectId: string, tab: "training") => void;
  activeProjectId: string | null;
};

export default function FloatingTrainingWidget({ onJump, activeProjectId }: Props) {
  const [items, setItems] = useState<FleetItem[]>([]);
  const [queueCount, setQueueCount] = useState(0);
  const [queueItems, setQueueItems] = useState<QueueItem[]>([]);
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // Poll fleet status every 3s
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await fetchFleetStatus();
        if (cancelled) return;
        setItems(data.items);
        setQueueCount(data.queue_count);
        setQueueItems(data.queue ?? []);
      } catch { /* ignore */ }
    };
    poll();
    const stop = visibleInterval(poll, 3_000);
    return () => { cancelled = true; stop(); };
  }, []);

  // Close panel on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const busyItems = items.filter((i) => i.busy);
  const hasRunning = busyItems.length > 0;

  // Build tooltip
  const tooltip = hasRunning
    ? `Training: ${busyItems.length}${queueCount > 0 ? ` / Queue: ${queueCount}` : ""}`
    : queueCount > 0
    ? `Queue: ${queueCount}`
    : "";

  const handleLaneClick = useCallback((item: FleetItem) => {
    if (item.project_id) {
      onJump(item.project_id, "training");
      setOpen(false);
    }
  }, [onJump]);

  // Hide entirely when nothing is happening
  if (!hasRunning && queueCount === 0) return null;

  return (
    <div className={`train-fab ${open ? "open" : ""} ${hasRunning ? "running" : ""}`} ref={panelRef}>
      <button
        className="train-fab-button"
        onClick={() => setOpen((p) => !p)}
        title={tooltip}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="4" width="16" height="16" rx="2" />
          <line x1="8" y1="10" x2="16" y2="10" />
          <line x1="8" y1="14" x2="13" y2="14" />
        </svg>
        {hasRunning && (
          <span className="train-fab-count">{busyItems.length}</span>
        )}
      </button>
      {open && (
        <div className="train-fab-panel">
          <div className="train-fab-panel-header">
            <span className="train-fab-panel-title">Training Pulse</span>
            <span className="train-fab-panel-badge">{busyItems.length} GPU{busyItems.length !== 1 ? "s" : ""}</span>
          </div>
          {items.map((item) => {
            if (!item.busy && queueCount === 0) return null;
            const isCurrent = item.project_id === activeProjectId;
            return (
              <button
                className={`train-fab-lane ${isCurrent ? "current" : ""}`}
                key={item.device_id}
                onClick={() => handleLaneClick(item)}
                disabled={!item.project_id}
              >
                <div className="train-fab-lane-top">
                  <span className="train-fab-lane-device">{item.device_id.replace("cuda:", "CUDA:")}</span>
                  {item.progress_pct != null && (
                    <span className="train-fab-lane-pct">{item.progress_pct}%</span>
                  )}
                </div>
                <div className="train-fab-lane-title">
                  {item.busy
                    ? item.project_name ?? "Unknown project"
                    : "Idle"}
                </div>
                {item.busy && (
                  <div className="train-fab-mini-progress">
                    <span style={{ width: `${item.progress_pct ?? 6}%` }} />
                  </div>
                )}
                <div className="train-fab-lane-meta">
                  {item.busy && item.epoch != null && item.total_epochs != null
                    ? item.progress_unit === "step"
                      ? `Step ${_fmtK(item.epoch)}/${_fmtK(item.total_epochs)}`
                      : `Epoch ${item.epoch}/${item.total_epochs}`
                    : item.busy
                    ? "Starting..."
                    : ""}
                  {isCurrent && <span className="train-fab-lane-cta">Back to Run</span>}
                  {!isCurrent && item.project_id && <span className="train-fab-lane-cta">Open</span>}
                </div>
              </button>
            );
          })}
          {queueCount > 0 && (
            <div className="train-fab-queue-section">
              <div className="train-fab-queue-header">
                Queue ({queueCount})
              </div>
              {queueItems.map((q) => (
                <button
                  key={q.run_id}
                  className="train-fab-queue-item"
                  onClick={() => { onJump(q.project_id, "training"); setOpen(false); }}
                >
                  <span className="train-fab-queue-pos">#{q.position}</span>
                  <span className="train-fab-queue-name">{q.project_name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
