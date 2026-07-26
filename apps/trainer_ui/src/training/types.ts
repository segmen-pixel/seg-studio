// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors

export type PredictionScore = {
  /** True when this entry is a presence-only stub written by
   *  refreshPredictionStatus to recover per-image class chips, NOT a real
   *  score.json. Every numeric field on a stub is 0, so anything that
   *  AGGREGATES scores must skip it -- the "FG Confidence distribution" panel
   *  averaged them and reported a fleet-wide mean of whatever fraction of the
   *  images happened to be stubs.
   *  NOTE: this type is duplicated in results/types.ts; keep both in step. */
  presence_only?: boolean;
  mean_confidence: number;
  foreground_mean_confidence: number;
  foreground_ratio: number;
  per_class_mean_confidence?: Record<string, number>;
};

export type ClassInfo = { id: number; name: string; color: number[] };
