// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors

export type ClassItem = {
  id: number;
  name: string;
  color: [number, number, number];
  active: boolean;
};

export const DEFAULT_CLASS_COLORS: Record<number, [number, number, number]> = {
  1: [255, 0, 0],
  2: [0, 122, 255],
  3: [0, 200, 120],
  4: [255, 180, 0],
  5: [213, 94, 0],
};

export function fallbackColorForClass(id: number): [number, number, number] {
  const known = DEFAULT_CLASS_COLORS[id];
  if (known) return known;
  const hue = (id * 47) % 360;
  const c = 0.75;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = 0.1;
  let r = 0;
  let g = 0;
  let b = 0;
  if (hue < 60) [r, g, b] = [c, x, 0];
  else if (hue < 120) [r, g, b] = [x, c, 0];
  else if (hue < 180) [r, g, b] = [0, c, x];
  else if (hue < 240) [r, g, b] = [0, x, c];
  else if (hue < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

export type MetricsPayload = {
  metrics: null | Record<string, unknown>;
  config: null | Record<string, unknown>;
};

export type PredictionScore = {
  /** True when this entry is a presence-only stub written by
   *  refreshPredictionStatus to recover per-image class chips, NOT a real
   *  score.json. Every numeric field on a stub is 0, so anything that
   *  AGGREGATES scores must skip it -- the "FG Confidence distribution" panel
   *  averaged them in and reported a mean dragged toward 0.
   *  NOTE: this type is duplicated in training/types.ts; keep both in step. */
  presence_only?: boolean;
  backend: string;
  item_id: string;
  mean_confidence: number;
  foreground_mean_confidence: number;
  background_mean_confidence: number;
  foreground_ratio: number;
  max_confidence: number;
  min_confidence: number;
  per_class_mean_confidence?: Record<string, number>;
  inference_ms?: number;
  inference_device?: string;
};

export type RegionLabel = {
  classId: number;
  cx: number;
  topY: number;
  count: number;
  name: string;
  color: [number, number, number];
};
