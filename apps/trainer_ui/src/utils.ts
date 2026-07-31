// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Shared types and utility functions used across tabs.
 */

export type TrainRunItem = {
  run_id: string;
  status: string;
  model_name?: string | null;
  has_model?: boolean;
  best_f1?: number | null;
  best_miou?: number | null;
  queue_position?: number | null;
  optimized_from?: string | null;
  fp16?: boolean;
  active_class_ids?: number[] | null;
  inference_threshold?: number | null;
  training_mode?: string | null;
  fold_index?: number | null;
  total_folds?: number | null;
  cv_group_id?: string | null;
  iter_index?: number | null;
  iter_max_iters?: number | null;
  iter_group_id?: string | null;
  created_at?: string;
  updated_at?: string;
};

export const DEFAULT_OUTPUT_STRIDE = 2;
export const VALID_OUTPUT_STRIDES = [1, 2, 4] as const;

/** Snap a dimension value up to the nearest multiple of stride (min 64). */
export function snapToStride(value: number, stride: number): number {
  const base = Math.max(64, Math.round(value));
  const s = Math.max(1, stride);
  const rem = base % s;
  return rem === 0 ? base : base + (s - rem);
}

/** Pick the best default run ID: prefer running, then first in list. */
export function pickDefaultRunId(items: TrainRunItem[]): string | null {
  const running = items.find((item) => item.status === "running");
  if (running) return running.run_id;
  return items.length > 0 ? items[0]?.run_id ?? null : null;
}
