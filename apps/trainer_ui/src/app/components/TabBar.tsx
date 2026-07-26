// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useRef } from "react";
import { useI18n } from "../../i18n";
import { BASE_TABS, type OpenResultTab, type OpenReportTab, type TabId } from "../types";

type TabBarProps = {
  activeTab: TabId;
  switchTab: (tab: TabId) => void;
  showInspectTab: boolean;
  openResultTabs: OpenResultTab[];
  toggleResultTabLock: (runId: string) => void;
  closeResultTab: (runId: string) => void;
  openReportTabs: OpenReportTab[];
  toggleReportTabLock: (reportId: string) => void;
  closeReportTab: (reportId: string) => void;
  activeResultBtnRef: React.RefObject<HTMLButtonElement>;
  guideTab?: TabId | null;
};

export default React.memo(function TabBar({
  activeTab, switchTab, showInspectTab,
  openResultTabs, toggleResultTabLock, closeResultTab,
  openReportTabs, toggleReportTabLock, closeReportTab,
  activeResultBtnRef, guideTab,
}: TabBarProps) {
  const { t } = useI18n();
  const resultTabsScrollRef = useRef<HTMLDivElement>(null);

  return (
    <div className="tabs-row">
      <div className={`tabs tabs-fixed${showInspectTab ? " tabs-fixed-4" : ""}`}>
        {BASE_TABS.map((tab) => (
          <button
            key={tab}
            data-tutorial-step={`${tab}-tab`}
            className={`${activeTab === tab ? "active" : ""}${guideTab === tab ? " tab-guide-pulse" : ""}`}
            onClick={() => switchTab(tab)}
          >
            {t(`tab.${tab}` as "tab.projects" | "tab.annotate" | "tab.training")}
          </button>
        ))}
        {showInspectTab && (
          <button
            className={`${activeTab === "inspect" ? "active" : ""}${guideTab === "inspect" ? " tab-guide-pulse" : ""}`}
            onClick={() => switchTab("inspect")}
          >
            {t("tab.inspect")}
          </button>
        )}
      </div>
      <div
        className={`tabs-result-scroll${openResultTabs.length >= 8 ? " tabs-density-3" : openResultTabs.length >= 5 ? " tabs-density-2" : openResultTabs.length >= 3 ? " tabs-density-1" : ""}`}
        ref={resultTabsScrollRef}
      >
        <div className="tabs">
          {openResultTabs.map((rt) => {
            const tabId: TabId = `result:${rt.runId}`;
            return (
              <button
                key={tabId}
                className={`tab-result ${activeTab === tabId ? "active" : ""}${rt.locked ? " tab-locked" : ""}`}
                onClick={() => switchTab(tabId)}
                ref={activeTab === tabId ? activeResultBtnRef : undefined}
              >
                <span className="tab-result-label">{rt.label}</span>
                <span
                  className={`tab-lock-btn${rt.locked ? " locked" : ""}`}
                  onClick={(e) => { e.stopPropagation(); toggleResultTabLock(rt.runId); }}
                  title={rt.locked ? "Unlock" : "Lock"}
                >
                  {rt.locked ? "\u{1F512}" : "\u{1F513}"}
                </span>
                {!rt.locked && (
                  <span className="tab-close-btn" onClick={(e) => { e.stopPropagation(); closeResultTab(rt.runId); }} title="Close">&times;</span>
                )}
              </button>
            );
          })}
          {openReportTabs.map((rt) => {
            const tabId: TabId = `report:${rt.reportId}`;
            return (
              <button
                key={tabId}
                className={`tab-result tab-report ${activeTab === tabId ? "active" : ""}${rt.locked ? " tab-locked" : ""}`}
                onClick={() => switchTab(tabId)}
              >
                <span className="tab-result-label">📄 {rt.label}</span>
                <span
                  className={`tab-lock-btn${rt.locked ? " locked" : ""}`}
                  onClick={(e) => { e.stopPropagation(); toggleReportTabLock(rt.reportId); }}
                  title={rt.locked ? "Unlock" : "Lock"}
                >
                  {rt.locked ? "\u{1F512}" : "\u{1F513}"}
                </span>
                {!rt.locked && (
                  <span className="tab-close-btn" onClick={(e) => { e.stopPropagation(); closeReportTab(rt.reportId); }} title="Close">&times;</span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
});
