// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useRef, useState } from "react";
import { fetchAgentRuns, type AgentRun } from "../api/agent";
import { visibleInterval } from "../app/hooks/useVisibleInterval";
import { useI18n } from "../i18n";

type Props = {
  onJump: (projectId: string, tab: "annotate") => void;
  activeProjectId: string | null;
};

/**
 * Which project the labelling model is working on, from anywhere in the app.
 *
 * The Assist panel lives inside the annotate tab and returns null off it, so
 * moving to another project left no sign that a run was still going -- and a
 * run does keep going: the server drives it whether or not a screen is
 * listening. This sits in the same corner stack as the training widget, is
 * absent when nothing is running, and says which image is being worked on.
 *
 * A run that is waiting for an answer is the state worth seeing: nothing moves
 * until somebody replies, and from another project there was no way to know.
 * It is marked by a glyph and a word, not by a colour alone.
 */
export default function FloatingAssistWidget({ onJump, activeProjectId }: Props) {
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await fetchAgentRuns();
        if (!cancelled) setRuns(data.items ?? []);
      } catch { /* ignore */ }
    };
    poll();
    const stop = visibleInterval(poll, 3_000);
    return () => { cancelled = true; stop(); };
  }, []);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const handleLaneClick = useCallback((run: AgentRun) => {
    onJump(run.project_id, "annotate");
    setOpen(false);
  }, [onJump]);

  if (runs.length === 0) return null;

  const waiting = runs.filter((r) => r.waiting_for_reply).length;
  const tooltip = waiting > 0
    ? t("assistWidget.tooltipWaiting").replace("{n}", String(waiting))
    : t("assistWidget.tooltipRunning").replace("{n}", String(runs.length));

  return (
    <div className={`assist-fab ${open ? "open" : ""} ${waiting > 0 ? "waiting" : "running"}`} ref={panelRef}>
      <button className="assist-fab-button" onClick={() => setOpen((p) => !p)} title={tooltip}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3v3" />
          <rect x="4" y="6" width="16" height="12" rx="3" />
          <circle cx="9" cy="12" r="1.4" fill="currentColor" stroke="none" />
          <circle cx="15" cy="12" r="1.4" fill="currentColor" stroke="none" />
        </svg>
        <span className="assist-fab-count">{runs.length}</span>
      </button>
      {open && (
        <div className="assist-fab-panel">
          <div className="assist-fab-panel-header">
            <span className="assist-fab-panel-title">{t("assistWidget.title")}</span>
            <span className="assist-fab-panel-badge">
              {t("assistWidget.runCount").replace("{n}", String(runs.length))}
            </span>
          </div>
          {runs.map((run) => {
            const isCurrent = run.project_id === activeProjectId;
            const state = run.waiting_for_reply ? "waiting" : run.paused ? "paused" : "running";
            return (
              <button
                className={`assist-fab-lane ${isCurrent ? "current" : ""} ${state}`}
                key={run.project_id}
                onClick={() => handleLaneClick(run)}
              >
                <div className="assist-fab-lane-top">
                  <span className={`assist-fab-state ${state}`}>
                    <span className="assist-fab-state-glyph" aria-hidden="true">
                      {state === "waiting" ? "?" : state === "paused" ? "\u2016" : "\u25cf"}
                    </span>
                    {state === "waiting"
                      ? t("assistWidget.stateWaiting")
                      : state === "paused"
                      ? t("assistWidget.statePaused")
                      : t("assistWidget.stateRunning")}
                  </span>
                  <span className="assist-fab-lane-turns">
                    {t("assistWidget.turns").replace("{n}", String(run.turns))}
                  </span>
                </div>
                <div className="assist-fab-lane-title">
                  {run.project_name ?? t("assistWidget.unknownProject")}
                </div>
                <div className="assist-fab-lane-meta">
                  <span className="assist-fab-lane-item">
                    {run.item_id ? t("assistWidget.onImage").replace("{id}", run.item_id) : ""}
                  </span>
                  <span className="assist-fab-lane-cta">
                    {isCurrent ? t("assistWidget.backToRun") : t("assistWidget.open")}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
