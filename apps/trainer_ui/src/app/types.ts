// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors

/**
 * App-level shared types used across hooks and components.
 */

/** The tasks a run can train for, in the order the API declares them.
 *
 *  The API list additionally carries "anomaly", which was removed in 0.9.7
 *  and survives only so a stale client gets a sentence it can act on rather
 *  than a schema error; nothing here may select it. Held against
 *  app/core/training_modes.py by test_task_modes_mirror.py, because nothing
 *  else links a TypeScript union to a Python tuple. */
export const TASK_MODES = ["standard", "instance"] as const;
export type TaskMode = (typeof TASK_MODES)[number];

/** A task id from somewhere that outlives a deploy -- sessionStorage today.
 *
 *  Returns null for anything unrecognised rather than trusting a cast. A
 *  stale value used to reach the strip as a mode no card matches: nothing
 *  rendered active, and Start stayed enabled because it only checks that
 *  SOME task is chosen. */
export function normalizeTaskMode(raw: string | null): TaskMode | null {
  if (!raw) return null;
  return (TASK_MODES as readonly string[]).includes(raw) ? (raw as TaskMode) : null;
}

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
