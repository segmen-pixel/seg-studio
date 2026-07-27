// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTrainingStore } from "../../store";
import type { OpenResultTab, TabId } from "../types";

/**
 * Manages per-project result tabs (open/close/lock) with sessionStorage + localStorage persistence.
 */
export function useResultTabs(
  selectedProjectId: string | null,
  setActiveTab: React.Dispatch<React.SetStateAction<TabId>>,
) {
  const resultTabsMapRef = useRef<Record<string, OpenResultTab[]>>(
    (() => { try { return JSON.parse(sessionStorage.getItem("seg-result-tabs-map") ?? "{}"); } catch { return {}; } })()
  );
  const tabsOwnerRef = useRef<string | null>(sessionStorage.getItem("seg-project"));
  const restoredProjectId = sessionStorage.getItem("seg-project");

  const [openResultTabs, setOpenResultTabs] = useState<OpenResultTab[]>(() => {
    const session = (restoredProjectId ? resultTabsMapRef.current[restoredProjectId] : null) ?? [];
    try {
      const lockedMap: Record<string, OpenResultTab[]> = JSON.parse(localStorage.getItem("seg-locked-result-tabs") ?? "{}");
      const locked = restoredProjectId ? (lockedMap[restoredProjectId] ?? []) : [];
      if (locked.length === 0) return session;
      const ids = new Set(session.map((t) => t.runId));
      const merged = [...session];
      for (const lt of locked) {
        if (!ids.has(lt.runId)) merged.push({ ...lt, locked: true });
      }
      return merged;
    } catch { return session; }
  });

  const [viewedRunIds, setViewedRunIds] = useState<string[]>(() => {
    try {
      const map = JSON.parse(localStorage.getItem("seg-viewed-result-runs") ?? "{}");
      return restoredProjectId ? (map[restoredProjectId] ?? []) : [];
    } catch { return []; }
  });

  const resultTabsScrollRef = useRef<HTMLDivElement>(null);
  const activeResultBtnRef = useRef<HTMLButtonElement>(null);

  // Migrate old format
  useEffect(() => {
    const old = sessionStorage.getItem("seg-result-tabs");
    if (old) sessionStorage.removeItem("seg-result-tabs");
  }, []);

  const persistResultTabs = useCallback((tabs: OpenResultTab[]) => {
    const owner = tabsOwnerRef.current;
    if (owner) resultTabsMapRef.current[owner] = tabs;
    sessionStorage.setItem("seg-result-tabs-map", JSON.stringify(resultTabsMapRef.current));
    if (owner) {
      const locked = tabs.filter((t) => t.locked);
      try {
        const lockedMap: Record<string, OpenResultTab[]> = JSON.parse(localStorage.getItem("seg-locked-result-tabs") ?? "{}");
        if (locked.length > 0) {
          lockedMap[owner] = locked;
        } else {
          delete lockedMap[owner];
        }
        localStorage.setItem("seg-locked-result-tabs", JSON.stringify(lockedMap));
      } catch { /* ignore */ }
    }
  }, []);

  const markRunViewed = useCallback((runId: string) => {
    setViewedRunIds((prev) => {
      if (prev.includes(runId)) return prev;
      const next = [...prev, runId];
      const owner = tabsOwnerRef.current;
      if (owner) {
        try {
          const map: Record<string, string[]> = JSON.parse(localStorage.getItem("seg-viewed-result-runs") ?? "{}");
          map[owner] = next;
          localStorage.setItem("seg-viewed-result-runs", JSON.stringify(map));
        } catch { /* ignore */ }
      }
      return next;
    });
  }, []);

  const openResultTab = useCallback((runId: string, label: string) => {
    setOpenResultTabs((prev) => {
      const next = prev.some((t) => t.runId === runId) ? prev : [...prev, { runId, label }];
      persistResultTabs(next);
      return next;
    });
    markRunViewed(runId);
    const tabId: TabId = `result:${runId}`;
    sessionStorage.setItem("seg-tab", tabId);
    setActiveTab(tabId);
  }, [persistResultTabs, setActiveTab, markRunViewed]);

  const closeResultTab = useCallback((runId: string) => {
    setOpenResultTabs((prev) => {
      const tab = prev.find((t) => t.runId === runId);
      if (tab?.locked) return prev;
      const next = prev.filter((t) => t.runId !== runId);
      persistResultTabs(next);
      return next;
    });
    const tabId: TabId = `result:${runId}`;
    setActiveTab((prev) => prev === tabId ? "training" : prev);
  }, [persistResultTabs, setActiveTab]);

  const toggleResultTabLock = useCallback((runId: string) => {
    setOpenResultTabs((prev) => {
      const next = prev.map((t) => t.runId === runId ? { ...t, locked: !t.locked } : t);
      persistResultTabs(next);
      return next;
    });
  }, [persistResultTabs]);

  const lockedRunIds = useMemo(
    () => openResultTabs.filter((t) => t.locked).map((t) => t.runId),
    [openResultTabs],
  );

  // Restore result tabs when project changes
  useEffect(() => {
    tabsOwnerRef.current = selectedProjectId;
    const session = selectedProjectId ? resultTabsMapRef.current[selectedProjectId] ?? [] : [];
    let restored = session;
    try {
      const lockedMap: Record<string, OpenResultTab[]> = JSON.parse(localStorage.getItem("seg-locked-result-tabs") ?? "{}");
      const locked = selectedProjectId ? (lockedMap[selectedProjectId] ?? []) : [];
      if (locked.length > 0) {
        const ids = new Set(session.map((t) => t.runId));
        const merged = [...session];
        for (const lt of locked) {
          if (!ids.has(lt.runId)) merged.push({ ...lt, locked: true });
        }
        restored = merged;
      }
    } catch { /* ignore */ }
    setOpenResultTabs(restored);
    try {
      const map = JSON.parse(localStorage.getItem("seg-viewed-result-runs") ?? "{}");
      setViewedRunIds(selectedProjectId ? (map[selectedProjectId] ?? []) : []);
    } catch { setViewedRunIds([]); }
    setActiveTab((prev) => {
      if (!prev.startsWith("result:")) return prev;
      const runId = prev.slice("result:".length);
      return restored.some((t) => t.runId === runId) ? prev : "training";
    });
  }, [selectedProjectId, setActiveTab]);

  // Auto-close result tabs for deleted runs
  const runs = useTrainingStore((s) => s.runs);
  useEffect(() => {
    if (runs.length === 0) return;
    const runIds = new Set(runs.map((r) => r.run_id));
    setOpenResultTabs((prev) => {
      if (prev.length === 0) return prev;
      const filtered = prev.filter((t) => t.locked || runIds.has(t.runId));
      if (filtered.length === prev.length) return prev;
      persistResultTabs(filtered);
      return filtered;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs]);

  return {
    openResultTabs, openResultTab, closeResultTab, toggleResultTabLock,
    lockedRunIds, viewedRunIds, resultTabsScrollRef, activeResultBtnRef,
  } as const;
}
