// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors

export type Region = {
  class: string;
  class_id: number;
  area_px: number;
  bbox: [number, number, number, number];
  confidence: number;
};

export type InferenceResult = {
  type: string;
  frame_id: string;
  judgement: "OK" | "NG";
  defect_found: boolean;
  regions: Region[];
  summary: { fg_ratio: number; max_confidence: number; num_defects: number };
  latency_ms: { decode?: number; preprocess?: number; inference?: number; postprocess?: number; total?: number };
  result_id: string;
  /** Client-side thumbnail URL (Object URL from dropped/selected file) */
  imageUrl?: string;
  /** Base64-encoded RGBA PNG of defect mask overlay */
  mask_png_b64?: string;
  /** Client-side Object URL created from mask_png_b64 */
  maskUrl?: string;
};

export type SessionInfo = {
  status: string;
  session_id: string;
  model_id: string;
  device: string;
  warmup_ms: number;
};

export type RunOption = {
  run_id: string;
  label: string;
  model_name?: string;
  best_f1?: number | null;
  best_miou?: number | null;
  created_at?: string;
};

export type ProjectOption = {
  id: string;
  name: string;
};

export type CameraConfig = {
  device_id: number | string;
  width: number;
  height: number;
  fps: number;
  preview_max_width: number;
  preview_fps: number;
};

export const DEFAULT_CAMERA_CONFIG: CameraConfig = {
  device_id: 0,
  width: 640,
  height: 480,
  fps: 30,
  preview_max_width: 640,
  preview_fps: 15,
};

export type AreaUnit = "px" | "mm" | "cm" | "m";

export type InferenceStats = {
  total: number;
  ok: number;
  ng: number;
  avgMs: number;
};

export type CameraState = "IDLE" | "PREVIEW" | "INSPECT";
