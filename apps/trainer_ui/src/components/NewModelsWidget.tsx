// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useRef, useState } from "react";
import { fetchRecentCompletions, type CompletedRunItem } from "../api";
import { useI18n } from "../i18n";
import { parseApiDate } from "../time";

const SEEN_KEY = "seg-studio-seen-models";

function loadSeen(): Set<string> {
  try {
    const raw = localStorage.getItem(SEEN_KEY);
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch {
    return new Set();
  }
}

function saveSeen(s: Set<string>) {
  // Keep max 200 entries to avoid unbounded growth
  const arr = [...s];
  localStorage.setItem(SEEN_KEY, JSON.stringify(arr.slice(-200)));
}

type Props = {
  onJump: (projectId: string, runId: string, label: string) => void;
};

export default function NewModelsWidget({ onJump }: Props) {
  const { t } = useI18n();
  const [items, setItems] = useState<CompletedRunItem[]>([]);
  const [seen, setSeen] = useState<Set<string>>(loadSeen);
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // Poll recent completions every 10s
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await fetchRecentCompletions();
        if (cancelled) return;
        setItems(data.items);
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, 10_000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  // Close on outside click
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

  const unseen = items.filter((i) => !seen.has(i.run_id));

  const handleClick = useCallback((item: CompletedRunItem) => {
    const next = new Set(seen);
    next.add(item.run_id);
    setSeen(next);
    saveSeen(next);
    const label = item.model_name ?? item.project_name;
    onJump(item.project_id, item.run_id, label);
    setOpen(false);
  }, [seen, onJump]);

  const handleDismissAll = useCallback(() => {
    const next = new Set(seen);
    for (const item of unseen) next.add(item.run_id);
    setSeen(next);
    saveSeen(next);
  }, [seen, unseen]);

  if (unseen.length === 0) return null;

  return (
    <div className={`new-models-fab ${open ? "open" : ""}`} ref={panelRef}>
      <button
        className="new-models-fab-button"
        onClick={() => setOpen((p) => !p)}
        title={t("widget.newModels.title")}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
        <span className="new-models-fab-count">{unseen.length}</span>
      </button>
      {open && (
        <div className="new-models-fab-panel">
          <div className="new-models-fab-header">
            <span className="new-models-fab-title">{t("widget.newModels.title")}</span>
            <button className="new-models-fab-dismiss" onClick={handleDismissAll}>
              {t("widget.newModels.dismissAll")}
            </button>
          </div>
          {unseen.map((item) => (
            <button
              className="new-models-fab-item"
              key={item.run_id}
              onClick={() => handleClick(item)}
            >
              <div className="new-models-fab-item-top">
                <span className="new-models-fab-item-project">{item.project_name}</span>
                {item.best_f1 != null && (
                  <span className="new-models-fab-item-f1">F1 {(item.best_f1 * 100).toFixed(1)}%</span>
                )}
              </div>
              <div className="new-models-fab-item-bottom">
                <span className="new-models-fab-item-model">{item.model_name ?? item.run_id.slice(0, 8)}</span>
                <span className="new-models-fab-item-time">
                  {_formatRelative(item.completed_at)}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function _formatRelative(iso: string): string {
  // API timestamps carry no timezone suffix, so a bare new Date() reads them
  // as local time and reports a just-finished model as hours old.
  const parsed = parseApiDate(iso);
  if (!parsed) return "";
  const diff = Date.now() - parsed.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}
