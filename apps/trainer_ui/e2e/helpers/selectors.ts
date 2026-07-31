// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Centralised CSS selectors — single source of truth for all E2E tests.
 * When the UI changes a class name, update it here only.
 */

export const SEL = {
  // ── App shell ──
  appShell: ".app-shell",
  appShellTab: (tab: string) => `.app-shell.tab-${tab}` as const,

  // ── Header ──
  appHeader: ".app-header",
  headerProjectName: ".header-project-name",
  appBrandTitle: ".app-brand h1",

  // ── API status ──
  apiStatusBadge: ".api-status-badge.connecting",
  apiConnectingBanner: ".api-connecting-banner",

  // -- Startup warnings --
  // Shares .settings-overlay with the settings dialog, hence the testid.
  startupWarnings: '[data-testid="startup-warnings"]',

  // ── Tabs ──
  tabButtons: ".tabs-fixed button",
  tabsRow: ".tabs-row",
  tabResult: ".tab-result",
  tabReport: ".tab-report",

  // ── Projects ──
  projectTile: ".project-tile",
  projectTileActive: ".project-tile.active",
  projectsLoadingCard: ".projects-loading-card",
  projectsCreateRow: ".projects-create-row",
  projectsCurrentTitle: ".projects-current-title",
  projectGrid: ".project-grid",

  // ── Annotate ──
  canvasStack: ".canvas-stack",
  imageListItem: ".image-list-dropzone .card.list-item-flat",
  classListCard: ".class-list-window .card",
  classNameInput: ".class-list-window .card .class-name-input",
  classColor: ".class-color",
  classSwatch: ".class-swatch",
  overlayToolsRight: ".overlay-tools.right",
  annotateLayout: ".annotate-layout",
  annotatorBusyOverlay: ".annotator-busy-overlay",

  // ── Training ──
  trainingLayout: ".training-layout",
  trainingConfigSection: ".training-config-section",
  trainingHyperToggle: ".training-hyper-toggle",
  trainingGroupBasic: "#hyperparameter-panel",
  trainingRunList: ".training-run-list .card",
  trainingChip: ".training-chip",
  trainingEmptyState: ".training-empty-state",
  trainingParamGroupInput: ".training-param-group input",
  trainingModeBtn: ".training-mode-btn",
  instanceSection: "[data-testid='instance-section']",
  instancePreviewBtn: "[data-testid='btn-instance-preview']",

  // ── Results ──
  resultsLayout: ".results-layout",

  // ── Settings ──
  settingsOverlay: ".settings-overlay",

  // ── Live Inspection ──
  inspectContainer: ".inspect-container",
  inspectEmpty: ".inspect-empty",
  inspectSessionRow: ".inspect-session-row",

  // ── Toast ──
  toast: ".toast-message",
} as const;
