// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors

export type PredictionScore = {
  mean_confidence: number;
  foreground_mean_confidence: number;
  foreground_ratio: number;
  per_class_mean_confidence?: Record<string, number>;
};

export type ClassInfo = { id: number; name: string; color: number[] };
