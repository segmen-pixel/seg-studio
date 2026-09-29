// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

export type FleetItem = {
  device_id: string;
  busy: boolean;
  project_id: string | null;
  run_id: string | null;
  project_name: string | null;
  progress_pct: number | null;
  epoch: number | null;
  total_epochs: number | null;
  progress_unit: "epoch" | "step" | null;
  queue_count: number;
  memory_mb: number | null;
};

export type QueueItem = {
  position: number;
  run_id: string;
  project_id: string;
  project_name: string;
  created_at: string;
};

export async function startTraining(projectId: string, payload: unknown) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export type InstancePreviewResponse = {
  samples: { image: string; n_instances: number }[];
  class_id: number;
  n_sources: number;
  n_cutouts: number;
  n_bg_plates: number;
  area_band: [number, number];
};

/** Export an instance run's RF-DETR-Seg checkpoint to the serving registry (fp32 ONNX). */
export async function exportInstanceOnnx(projectId: string, runId: string) {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/export/instance-onnx`,
    { method: "POST" },
  );
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

/** Compose a few synthetic instance samples server-side (training form preview). */
export async function fetchInstancePreview(
  projectId: string,
  params: Record<string, unknown>,
): Promise<InstancePreviewResponse> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/instance-preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}


export async function fetchRuns(projectId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchRunLogs(projectId: string, runId: string, offset?: number): Promise<{ log: string; total?: number }> {
  const qs = typeof offset === "number" && offset > 0 ? `?offset=${offset}` : "";
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/logs${qs}`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function stopRun(projectId: string, runId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/stop`, {
    method: "POST"
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchRunMetrics(projectId: string, runId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/metrics`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchRunSplits(
  projectId: string,
  runId: string,
): Promise<{ splits: Record<string, "train" | "val" | "test"> }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/splits`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function deleteRun(projectId: string, runId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}`, {
    method: "DELETE"
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchPixelHistogram(projectId: string, runId: string, backend = "onnx", bins = 50): Promise<{ bins: number[]; counts: number[]; total_pixels: number }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/pixel-histogram?backend=${backend}&bins=${bins}`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function optimizeRun(projectId: string, runId: string): Promise<{ status: string; run_id: string; model_name: string }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/optimize`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function exportOnnx(projectId: string, runId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/export/onnx?run_id=${runId}`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function exportCoreML(projectId: string, runId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/export/coreml`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  const blob = await res.blob();
  const cd = res.headers.get("content-disposition") || "";
  const match = cd.match(/filename="?([^";]+)"?/);
  const fname = match?.[1] || "model.mlmodel";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = fname; a.click();
  URL.revokeObjectURL(url);
  return { downloaded: fname };
}


export async function exportCoreMLUpdatable(projectId: string, runId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/export/coreml-updatable`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  const blob = await res.blob();
  const cd = res.headers.get("content-disposition") || "";
  const match = cd.match(/filename="?([^";]+)"?/);
  const fname = match?.[1] || "model_updatable.mlmodel";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = fname; a.click();
  URL.revokeObjectURL(url);
  return { downloaded: fname };
}


export type OpenVinoPrecision = "fp32" | "fp16" | "int8";

export async function exportOpenVino(
  projectId: string,
  runId: string,
  precision: OpenVinoPrecision = "fp32",
) {
  const url = `${API_BASE}/projects/${projectId}/train/runs/${runId}/export/openvino?precision=${precision}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  const blob = await res.blob();
  // RFC 5987: prefer filename*=UTF-8''... if present (non-ASCII project names);
  // fall back to filename="..." then a hardcoded default.
  const cd = res.headers.get("content-disposition") || "";
  const utf8Match = cd.match(/filename\*=UTF-8''([^;]+)/i);
  let fname = utf8Match ? decodeURIComponent(utf8Match[1]) : "";
  if (!fname) {
    const asciiMatch = cd.match(/filename="?([^";]+)"?/);
    fname = asciiMatch?.[1] || `model_openvino_${precision}.zip`;
  }
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl; a.download = fname; a.click();
  URL.revokeObjectURL(objectUrl);
  return { downloaded: fname };
}


export async function fetchFleetStatus(): Promise<{ items: FleetItem[]; queue_count: number; queue: QueueItem[] }> {
  const res = await fetch(`${API_BASE}/train/fleet-status`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export type CompletedRunItem = {
  run_id: string;
  project_id: string;
  project_name: string;
  model_name: string | null;
  status: string;
  best_f1: number | null;
  best_miou: number | null;
  completed_at: string;
};

export async function fetchRecentCompletions(): Promise<{ items: CompletedRunItem[] }> {
  const res = await fetch(`${API_BASE}/train/recent-completions`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchGlobalTrainingStatus(device?: string): Promise<{
  gpu_busy: boolean;
  progress: { epoch: number; total_epochs: number; pct: number } | null;
  device?: string | null;
  owner_kind?: string | null;
  owner_id?: string | null;
  inference?: { active: boolean; project_id: string; run_id: string; total: number; completed: number } | null;
}> {
  const params = new URLSearchParams();
  if (device) params.set("device", device);
  const qs = params.toString();
  const res = await fetch(`${API_BASE}/train/global-status${qs ? `?${qs}` : ""}`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}
