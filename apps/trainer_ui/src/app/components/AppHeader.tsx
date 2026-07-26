// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import { fetchHealth, type Project } from "../../api";
import GpuMonitor from "../../components/GpuMonitor";
import type { ThemeMode } from "../types";

declare const __APP_VERSION__: string;

type AppHeaderProps = {
  currentProject: Project | null;
  activeTabIsProjects: boolean;
  // Theme
  themeMode: ThemeMode;
  cycleTheme: () => void;
  // Desc mode
  descMode: boolean;
  setDescMode: (fn: (v: boolean) => boolean) => void;
  // About
  onOpenAbout: () => void;
  // Settings
  onOpenSettings: () => void;
  // Tutorial
  onRestartTutorial?: () => void;
};

export default React.memo(function AppHeader({
  currentProject, activeTabIsProjects,
  themeMode, cycleTheme,
  descMode, setDescMode,
  onOpenAbout,
  onOpenSettings, onRestartTutorial,
}: AppHeaderProps) {
  const { lang, setLang, t } = useI18n();

  return (
    <header>
      <div className="app-brand">
        <h1>
          Seg-Studio <span className="app-version">v{__APP_VERSION__}</span>
        </h1>
        <span className="header-project-name">{!activeTabIsProjects && currentProject ? currentProject.name : ""}</span>
        <div className="header-actions">
          <GpuMonitor />
          <button
            className={`theme-toggle${descMode ? " active" : ""}`}
            onClick={() => setDescMode((v) => !v)}
            title={t("header.descMode")}
            data-desc={t("header.descMode.desc")}
            data-desc-pos="bottom"
            style={descMode ? { color: "var(--accent)" } : undefined}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </button>
          {onRestartTutorial && (
            <button
              className="theme-toggle"
              onClick={onRestartTutorial}
              title={t("tutorial.replay.title")}
              data-desc={t("tutorial.replay.desc")}
              data-desc-pos="bottom"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none" />
              </svg>
            </button>
          )}
          <button
            className="theme-toggle"
            onClick={onOpenAbout}
            title={t("header.about")}
            data-desc={t("header.about.desc")}
            data-desc-pos="bottom"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
          </button>
          <button
            className="theme-toggle"
            onClick={onOpenSettings}
            title={t("header.settings")}
            data-desc={t("header.settings.desc")}
            data-desc-pos="bottom"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
          </button>
          <button
            className="theme-toggle"
            onClick={cycleTheme}
            aria-label={`Theme: ${themeMode}`}
            title={`Theme: ${themeMode}`}
            data-desc={t("header.theme.desc")}
            data-desc-pos="bottom"
          >
            {themeMode === "dark" ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            ) : themeMode === "light" ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
              </svg>
            )}
          </button>
          <button
            className="theme-toggle lang-toggle"
            onClick={() => setLang(lang === "ja" ? "en" : "ja")}
            title={t("header.lang.desc")}
            data-desc={t("header.lang.desc")}
            data-desc-pos="bottom"
          >
            <span className="lang-toggle-label">{lang === "ja" ? "EN" : "JP"}</span>
          </button>
        </div>
      </div>
    </header>
  );
});
