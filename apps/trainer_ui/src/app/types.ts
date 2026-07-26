// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors

/**
 * App-level shared types used across hooks and components.
 */

export const BASE_TABS = ["projects", "annotate", "training"] as const;
export type TabId = (typeof BASE_TABS)[number] | "inspect" | `result:${string}` | `report:${string}`;
export type OpenResultTab = { runId: string; label: string; locked?: boolean };
export type OpenReportTab = { reportId: string; runId: string; label: string; locked?: boolean };
export type ThemeMode = "light" | "dark" | "system";

// Single source of truth lives next to the fetch helper in api/training.ts;
// re-exported here so app-level imports keep working.

export type TrainProgressInfo = {
  pct: number;
  epoch: number;
  total_epochs: number;
  unit?: "epoch" | "step";
};

export type TrainingStatusInfo = {
  state: string;
  running: number;
  total: number;
  percent: null | number;
  etaMinutes: null | number;
};

export type StartupWarning = {
  level: string;
  title: string;
  message: string;
};
