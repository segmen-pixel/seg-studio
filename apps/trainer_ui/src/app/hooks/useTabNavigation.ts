// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useRef } from "react";
import { BASE_TABS, type OpenResultTab, type TabId } from "../types";

/**
 * Manages tab switching, keyboard navigation, and auto-scroll.
 * activeTab / setActiveTab are owned by the caller (App) to avoid circular deps.
 */
export function useTabNavigation(
  activeTab: TabId,
  setActiveTab: React.Dispatch<React.SetStateAction<TabId>>,
  showInspectTab: boolean,
  openResultTabs: OpenResultTab[],
  annotatorSaveRef: React.RefObject<(() => Promise<void>) | null>,
) {
  const activeResultBtnRef = useRef<HTMLButtonElement>(null);

  const switchTab = useCallback((tab: TabId) => {
    setActiveTab((prev) => {
      if (tab === prev) return prev;
      if (annotatorSaveRef.current) {
        const save = annotatorSaveRef.current;
        Promise.race([
          save(),
          new Promise((_, reject) => setTimeout(() => reject(new Error("save timeout")), 5000)),
        ]).catch((e: unknown) => console.warn("App: tab-switch auto-save failed:", e));
      }
      sessionStorage.setItem("seg-tab", tab);
      return tab;
    });
  }, [setActiveTab, annotatorSaveRef]);

  const navigateTab = useCallback((direction: -1 | 1) => {
    setActiveTab((prev) => {
      const allTabs: TabId[] = [
        ...BASE_TABS,
        ...(showInspectTab ? ["inspect" as TabId] : []),
        ...openResultTabs.map((t) => `result:${t.runId}` as TabId),
      ];
      const idx = allTabs.indexOf(prev);
      if (idx < 0) return prev;
      const next = (idx + direction + allTabs.length) % allTabs.length;
      const nextTab = allTabs[next];
      sessionStorage.setItem("seg-tab", nextTab);
      return nextTab;
    });
  }, [setActiveTab, showInspectTab, openResultTabs]);

  // Keyboard: Ctrl+Arrow to navigate tabs
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if ((e.target as HTMLElement)?.isContentEditable) return;
      if (e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey) {
        if (e.key === "ArrowLeft") { e.preventDefault(); navigateTab(-1); }
        else if (e.key === "ArrowRight") { e.preventDefault(); navigateTab(1); }
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigateTab]);

  // Auto-scroll active result tab into view
  useEffect(() => {
    if (activeTab.startsWith("result:") && activeResultBtnRef.current) {
      activeResultBtnRef.current.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    }
  }, [activeTab]);

  return { switchTab, navigateTab, activeResultBtnRef } as const;
}
