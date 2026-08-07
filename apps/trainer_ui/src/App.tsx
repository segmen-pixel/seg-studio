// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ProjectsPanel from "./components/ProjectsPanel";
import type { SettingsValues } from "./components/SettingsDialog";

// Lazy-loaded tab components (code-split per tab)
const Annotator = React.lazy(() => import("./annotate/Annotator"));
const Training = React.lazy(() => import("./training/Training"));
const Results = React.lazy(() => import("./results/Results"));
const LiveInspection = React.lazy(() => import("./components/LiveInspection"));
const AboutDialog = React.lazy(() => import("./components/AboutDialog"));
const SettingsDialog = React.lazy(() => import("./components/SettingsDialog"));

// These are always visible (FAB stack), keep eager
import FloatingTrainingWidget from "./components/FloatingTrainingWidget";
import NewModelsWidget from "./components/NewModelsWidget";
import { useI18n } from "./i18n";

import {
  useTheme,
  useToast,
  useGlobalPolling,
  useResultTabs,
  useReportTabs,
  useSettings,
  useTabNavigation,
  useApiConnection,
  useGuideStep,
  useTutorial,
} from "./app/hooks";
import TutorialOverlay from "./app/components/TutorialOverlay";

import AppHeader from "./app/components/AppHeader";
import TabBar from "./app/components/TabBar";
import ReportViewer from "./reports/ReportViewer";
import StatusToast from "./app/components/StatusToast";
import StartupWarnings from "./app/components/StartupWarnings";

import { BASE_TABS, type TabId } from "./app/types";

export default function App() {
  const { lang, t } = useI18n();

  // --- Core hooks ---
  const { themeMode, cycleTheme } = useTheme();
  const toast = useToast();
  const api = useApiConnection(toast.showToast);
  const settings = useSettings();

  // activeTab is lifted here to break circular dep between useResultTabs ↔ useTabNavigation
  const [activeTab, setActiveTab] = useState<TabId>(() => {
    const saved = sessionStorage.getItem("seg-tab");
    if (!saved) return "projects";
    if ((BASE_TABS as readonly string[]).includes(saved)) return saved as TabId;
    if (saved.startsWith("result:")) return saved as TabId;
    if (saved.startsWith("report:")) return saved as TabId;
    return "projects";
  });

  const annotatorSaveRef = useRef<(() => Promise<void>) | null>(null);
  const resultTabs = useResultTabs(api.selectedProjectId, setActiveTab);
  const reportTabs = useReportTabs(api.selectedProjectId, setActiveTab);
  const tabNav = useTabNavigation(activeTab, setActiveTab, settings.showInspectTab, resultTabs.openResultTabs, annotatorSaveRef);
  const polling = useGlobalPolling(api.selectedProjectId, activeTab, lang);
  const guideTab = useGuideStep({
    descMode: settings.descMode,
    activeTab,
    selectedProjectId: api.selectedProjectId,
    projects: api.projects,
    projectPreviews: api.projectPreviews,
    viewedRunIds: resultTabs.viewedRunIds,
  });
  const tutorial = useTutorial(activeTab, tabNav.switchTab);

  // --- Local state ---
  const [aboutOpen, setAboutOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inspectTargetRun, setInspectTargetRun] = useState<string | undefined>(undefined);
  const [pendingEditImageId, setPendingEditImageId] = useState<string | null>(null);

  // Prevent browser from opening dropped files (global safety net)
  useEffect(() => {
    const prevent = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => { window.removeEventListener("dragover", prevent); window.removeEventListener("drop", prevent); };
  }, []);

  // Refresh projects when revisiting projects tab
  useEffect(() => {
    if (activeTab === "projects" && api.apiStatus === "connected") {
      void api.refreshProjects(api.projectsSummaryReady);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, api.apiStatus]);

  const handleEditImage = useCallback((imageId: string) => {
    setPendingEditImageId(imageId);
    tabNav.switchTab("annotate");
  }, [tabNav]);

  const goToInspect = useCallback((runId: string) => {
    setInspectTargetRun(runId);
    tabNav.switchTab("inspect");
  }, [tabNav]);

  const openProjectWorkspace = useCallback((projectId: string, tab: "annotate" | "training") => {
    api.setSelectedProjectId(projectId);
    sessionStorage.setItem("seg-project", projectId);
    tabNav.switchTab(tab);
  }, [api, tabNav]);

  const settingsValues: SettingsValues = settings.settingsValues;
  const handleSettingsChange = settings.handleSettingsChange;

  // --- Instant tooltips (body-level): rich data-desc in desc-mode, else the
  //     element's native title — shown immediately, with the browser title
  //     suppressed during hover so there is no delayed / double tooltip. ---
  useEffect(() => {
    let tip: HTMLDivElement | null = null;
    let activeEl: HTMLElement | null = null;
    const restore = (el: HTMLElement | null) => {
      if (el && el.dataset.nativeTitle != null) {
        el.setAttribute("title", el.dataset.nativeTitle);
        el.removeAttribute("data-native-title");
      }
    };
    const show = (e: MouseEvent) => {
      const el = (e.target as HTMLElement).closest?.("[data-desc],[title]") as HTMLElement | null;
      if (!el || el === activeEl) return;
      restore(activeEl);
      activeEl = el;
      const desc = settings.descMode ? el.getAttribute("data-desc") : null;
      const compact = !desc;
      let text = desc;
      if (!text) {
        text = el.getAttribute("title");
        if (text) { el.dataset.nativeTitle = text; el.removeAttribute("title"); }
      }
      if (!text) { activeEl = null; return; }
      if (!tip) { tip = document.createElement("div"); document.body.appendChild(tip); }
      tip.className = compact ? "desc-tooltip desc-tooltip-compact visible" : "desc-tooltip visible";
      tip.textContent = text;
      const r = el.getBoundingClientRect();
      const pos = el.getAttribute("data-desc-pos") || "top";
      // measure tooltip
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let top: number, left: number;
      if (pos === "bottom") { top = r.bottom + 8; left = r.left + r.width / 2 - tw / 2; }
      else if (pos === "right") { top = r.top + r.height / 2 - th / 2; left = r.right + 8; }
      else if (pos === "left") { top = r.top + r.height / 2 - th / 2; left = r.left - tw - 8; }
      else { top = r.top - th - 8; left = r.left + r.width / 2 - tw / 2; }
      // clamp to viewport
      if (left < 4) left = 4;
      if (left + tw > window.innerWidth - 4) left = window.innerWidth - tw - 4;
      if (top < 4) { top = r.bottom + 8; } // flip to bottom
      if (top + th > window.innerHeight - 4) { top = r.top - th - 8; } // flip to top
      tip.style.top = `${top}px`;
      tip.style.left = `${left}px`;
    };
    const hide = (e: MouseEvent) => {
      if (!activeEl) return;
      const to = e.relatedTarget as Node | null;
      if (to && activeEl.contains(to)) return;
      if (tip) tip.classList.remove("visible");
      restore(activeEl);
      activeEl = null;
    };
    document.addEventListener("mouseover", show);
    document.addEventListener("mouseout", hide);
    return () => {
      document.removeEventListener("mouseover", show);
      document.removeEventListener("mouseout", hide);
      restore(activeEl);
      if (tip) { tip.remove(); tip = null; }
    };
  }, [settings.descMode]);

  // --- Render ---
  return (
    <div className={`app-shell tab-${activeTab.startsWith("result:") ? "results" : activeTab.startsWith("report:") ? "report" : activeTab}${settings.descMode ? " desc-mode" : ""}`}>
      <div className="app-top">
        <AppHeader
          currentProject={api.currentProject}
          activeTabIsProjects={activeTab === "projects"}
          themeMode={themeMode}
          cycleTheme={cycleTheme}
          descMode={settings.descMode}
          setDescMode={settings.setDescMode}
          onOpenAbout={() => { setAboutOpen(true); api.refreshHealthInfo(); }}
          onOpenSettings={() => setSettingsOpen(true)}
          onRestartTutorial={tutorial.restart}
        />
        <TabBar
          activeTab={activeTab}
          switchTab={tabNav.switchTab}
          showInspectTab={settings.showInspectTab}
          openResultTabs={resultTabs.openResultTabs}
          toggleResultTabLock={resultTabs.toggleResultTabLock}
          closeResultTab={resultTabs.closeResultTab}
          openReportTabs={reportTabs.openReportTabs}
          toggleReportTabLock={reportTabs.toggleReportTabLock}
          closeReportTab={reportTabs.closeReportTab}
          activeResultBtnRef={tabNav.activeResultBtnRef}
          guideTab={guideTab}
        />
      </div>
      <main className="app-content">
        <Suspense fallback={<div className="muted" style={{ padding: 32, textAlign: "center" }}>Loading...</div>}>
        <section className={`panel ${activeTab === "annotate" ? "panel-tight" : ""}`}>
          <div className={`tab-panel ${activeTab === "projects" ? "active" : ""}`}>
            <ProjectsPanel
              projects={api.projects}
              setProjects={api.setProjects}
              selectedProjectId={api.selectedProjectId}
              setSelectedProjectId={api.setSelectedProjectId}
              projectPreviews={api.projectPreviews}
              setProjectPreviews={api.setProjectPreviews}
              projectsLoading={api.projectsLoading}
              projectsSummaryReady={api.projectsSummaryReady}
              apiStatus={api.apiStatus}
              currentProject={api.currentProject}
              currentProjectPreview={api.currentProjectPreview}
              openProjectWorkspace={openProjectWorkspace}
              showToast={toast.showToast}
            />
          </div>
          <div className={`tab-panel ${activeTab === "annotate" ? "active" : ""}`}>
            <Annotator
              projectId={api.selectedProjectId}
              projects={api.projects}
              onProjectChange={api.setSelectedProjectId}
              active={activeTab === "annotate"}
              saveRef={annotatorSaveRef}
              previewStyle={settings.previewStyle}
              setPreviewStyle={settings.setPreviewStyle}
              showToast={toast.showToast}
              descMode={settings.descMode}
              pendingImageId={pendingEditImageId}
              onPendingImageHandled={() => setPendingEditImageId(null)}
            />
          </div>
          <div className={`tab-panel ${activeTab === "training" ? "active" : ""}`}>
            <Training
              projectId={api.selectedProjectId}
              active={activeTab === "training"}
              onOpenResults={resultTabs.openResultTab}
              valRatio={settings.valRatio}
              testRatio={settings.testRatio}
              exportFormat={settings.exportFormat}
              showToast={toast.showToast}
              lockedRunIds={resultTabs.lockedRunIds}
              viewedRunIds={resultTabs.viewedRunIds}
              descMode={settings.descMode}
            />
          </div>
          {settings.showInspectTab && (
          <div className={`tab-panel ${activeTab === "inspect" ? "active" : ""}`}>
            <LiveInspection
              projectId={api.selectedProjectId ?? ""}
              projectName={api.projects.find((p) => p.id === api.selectedProjectId)?.name}
              active={activeTab === "inspect"}
              targetRunId={inspectTargetRun}
              showToast={toast.showToast}
            />
          </div>
          )}
          {resultTabs.openResultTabs.map((rt) => {
            const tabId: TabId = `result:${rt.runId}`;
            return (
              <div key={tabId} className={`tab-panel ${activeTab === tabId ? "active" : ""}`}>
                <Results
                  projectId={api.selectedProjectId}
                  projects={api.projects}
                  onProjectChange={api.setSelectedProjectId}
                  active={activeTab === tabId}
                  showToast={toast.showToast}
                  onInferStatus={polling.setInferStatus}
                  runId={rt.runId}
                  onClose={() => resultTabs.closeResultTab(rt.runId)}
                  onGoInspect={goToInspect}
                  onOpenReport={reportTabs.openReportTab}
                />
              </div>
            );
          })}
          {reportTabs.openReportTabs.map((rt) => {
            const tabId: TabId = `report:${rt.reportId}`;
            return (
              <div key={tabId} className={`tab-panel ${activeTab === tabId ? "active" : ""}`}>
                <ReportViewer
                  projectId={api.selectedProjectId}
                  reportId={rt.reportId}
                  runLabel={rt.label}
                  active={activeTab === tabId}
                />
              </div>
            );
          })}
        </section>
        </Suspense>
        <button
          className="tab-nav-arrow tab-nav-arrow-left"
          onClick={() => tabNav.navigateTab(-1)}
          aria-label="Previous tab"
          tabIndex={-1}
          data-desc={t("projects.prevTab")}
          data-desc-pos="right"
        >
          <svg width="20" height="36" viewBox="0 0 20 36" fill="none">
            <path d="M16 4 L4 18 L16 32" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <button
          className="tab-nav-arrow tab-nav-arrow-right"
          onClick={() => tabNav.navigateTab(1)}
          aria-label="Next tab"
          tabIndex={-1}
          data-desc={t("projects.nextTab")}
          data-desc-pos="left"
        >
          <svg width="20" height="36" viewBox="0 0 20 36" fill="none">
            <path d="M4 4 L16 18 L4 32" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </main>
      <StartupWarnings
        warnings={api.startupWarnings}
        onDismiss={() => api.setStartupWarnings([])}
      />
      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} healthInfo={api.healthInfo} />
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        values={settingsValues}
        onChange={handleSettingsChange}
        showToast={toast.showToast}
        onLibraryChanged={() => window.dispatchEvent(new Event("library-changed"))}
      />
      <div className="fab-stack">
        <NewModelsWidget
          onJump={(projectId, runId, label) => {
            api.setSelectedProjectId(projectId);
            sessionStorage.setItem("seg-project", projectId);
            resultTabs.openResultTab(runId, label);
          }}
        />
        <FloatingTrainingWidget
          activeProjectId={api.selectedProjectId}
          onJump={(projectId, tab) => openProjectWorkspace(projectId, tab)}
        />
        <StatusToast
          toastMsg={toast.toastMsg}
          toastCopied={toast.toastCopied}
          gpuBusy={polling.gpuBusy}
          inferStatus={polling.inferStatus}
          trainProgress={polling.trainProgress}
          trainProjectId={polling.trainProjectId}
          projects={api.projects}
          onToastClick={toast.handleToastClick}
          onMouseEnter={toast.handleToastHoverEnter}
          onMouseLeave={toast.handleToastHoverLeave}
        />
      </div>
      {tutorial.active && tutorial.currentStep && (
        <TutorialOverlay
          step={tutorial.currentStep}
          stepIndex={tutorial.stepIndex}
          totalSteps={tutorial.totalSteps}
          lang={lang}
          onNext={tutorial.next}
          onBack={tutorial.back}
          onSkip={tutorial.skip}
          onChooseMode={tutorial.chooseMode}
        />
      )}
    </div>
  );
}
