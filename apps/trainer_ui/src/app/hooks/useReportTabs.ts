// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OpenReportTab, TabId } from "../types";

/**
 * Manages per-project report tabs (open/close/lock) with sessionStorage + localStorage persistence.
 * Mirrors useResultTabs but keyed by reportId.
 */
export function useReportTabs(
  selectedProjectId: string | null,
  setActiveTab: React.Dispatch<React.SetStateAction<TabId>>,
) {
  const tabsMapRef = useRef<Record<string, OpenReportTab[]>>(
    (() => { try { return JSON.parse(sessionStorage.getItem("seg-report-tabs-map") ?? "{}"); } catch { return {}; } })()
  );
  const ownerRef = useRef<string | null>(sessionStorage.getItem("seg-project"));
  const restoredProjectId = sessionStorage.getItem("seg-project");

  const [openReportTabs, setOpenReportTabs] = useState<OpenReportTab[]>(() => {
    const session = (restoredProjectId ? tabsMapRef.current[restoredProjectId] : null) ?? [];
    try {
      const lockedMap: Record<string, OpenReportTab[]> = JSON.parse(localStorage.getItem("seg-locked-report-tabs") ?? "{}");
      const locked = restoredProjectId ? (lockedMap[restoredProjectId] ?? []) : [];
      if (locked.length === 0) return session;
      const ids = new Set(session.map((t) => t.reportId));
      const merged = [...session];
      for (const lt of locked) {
        if (!ids.has(lt.reportId)) merged.push({ ...lt, locked: true });
      }
      return merged;
    } catch { return session; }
  });

  const persist = useCallback((tabs: OpenReportTab[]) => {
    const owner = ownerRef.current;
    if (owner) tabsMapRef.current[owner] = tabs;
    sessionStorage.setItem("seg-report-tabs-map", JSON.stringify(tabsMapRef.current));
    if (owner) {
      const locked = tabs.filter((t) => t.locked);
      try {
        const lockedMap: Record<string, OpenReportTab[]> = JSON.parse(localStorage.getItem("seg-locked-report-tabs") ?? "{}");
        if (locked.length > 0) lockedMap[owner] = locked;
        else delete lockedMap[owner];
        localStorage.setItem("seg-locked-report-tabs", JSON.stringify(lockedMap));
      } catch { /* ignore */ }
    }
  }, []);

  const openReportTab = useCallback((reportId: string, runId: string, label: string) => {
    setOpenReportTabs((prev) => {
      const next = prev.some((t) => t.reportId === reportId) ? prev : [...prev, { reportId, runId, label }];
      persist(next);
      return next;
    });
    const tabId: TabId = `report:${reportId}`;
    sessionStorage.setItem("seg-tab", tabId);
    setActiveTab(tabId);
  }, [persist, setActiveTab]);

  const closeReportTab = useCallback((reportId: string) => {
    setOpenReportTabs((prev) => {
      const tab = prev.find((t) => t.reportId === reportId);
      if (tab?.locked) return prev;
      const next = prev.filter((t) => t.reportId !== reportId);
      persist(next);
      return next;
    });
    const tabId: TabId = `report:${reportId}`;
    setActiveTab((prev) => (prev === tabId ? "training" : prev));
  }, [persist, setActiveTab]);

  const toggleReportTabLock = useCallback((reportId: string) => {
    setOpenReportTabs((prev) => {
      const next = prev.map((t) => (t.reportId === reportId ? { ...t, locked: !t.locked } : t));
      persist(next);
      return next;
    });
  }, [persist]);

  const lockedReportIds = useMemo(
    () => openReportTabs.filter((t) => t.locked).map((t) => t.reportId),
    [openReportTabs],
  );

  // Restore report tabs when the project changes
  useEffect(() => {
    ownerRef.current = selectedProjectId;
    const session = selectedProjectId ? tabsMapRef.current[selectedProjectId] ?? [] : [];
    let restored = session;
    try {
      const lockedMap: Record<string, OpenReportTab[]> = JSON.parse(localStorage.getItem("seg-locked-report-tabs") ?? "{}");
      const locked = selectedProjectId ? (lockedMap[selectedProjectId] ?? []) : [];
      if (locked.length > 0) {
        const ids = new Set(session.map((t) => t.reportId));
        const merged = [...session];
        for (const lt of locked) {
          if (!ids.has(lt.reportId)) merged.push({ ...lt, locked: true });
        }
        restored = merged;
      }
    } catch { /* ignore */ }
    setOpenReportTabs(restored);
    setActiveTab((prev) => {
      if (!prev.startsWith("report:")) return prev;
      const reportId = prev.slice("report:".length);
      return restored.some((t) => t.reportId === reportId) ? prev : "training";
    });
  }, [selectedProjectId, setActiveTab]);

  return {
    openReportTabs, openReportTab, closeReportTab, toggleReportTabLock, lockedReportIds,
  } as const;
}
