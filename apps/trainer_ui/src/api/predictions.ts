// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

function _buildPredictParams(backend?: string, tta?: boolean, force?: boolean, readonly?: boolean): string {
  const params = new URLSearchParams();
  if (backend) params.set("backend", backend);
  if (tta) params.set("tta", "true");
  if (force) params.set("force", "true");
  if (readonly) params.set("readonly", "true");
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

export async function fetchPredictionStatus(
  projectId: string, runId: string, backend?: string, tta?: boolean,
): Promise<{ predicted: string[]; count: number; per_image_classes?: Record<string, number[]> }> {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/status${_buildPredictParams(backend, tta)}`
  );
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export function runPredictMaskUrl(projectId: string, runId: string, imageId: string, backend?: string, tta?: boolean, force?: boolean, readonly?: boolean) {
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}.png${_buildPredictParams(backend, tta, force, readonly)}`;
}

export function runPredictConfidenceUrl(projectId: string, runId: string, imageId: string, backend?: string, tta?: boolean, force?: boolean, readonly?: boolean) {
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/confidence.png${_buildPredictParams(backend, tta, force, readonly)}`;
}

/** Instance-run overlay (server-rendered fills + numbered badges). */
export function runInstanceOverlayUrl(
  projectId: string, runId: string, imageId: string,
  readonly?: boolean, mode?: "class" | "instance",
) {
  const params = new URLSearchParams();
  if (readonly) params.set("readonly", "true");
  if (mode) params.set("mode", mode);
  const q = params.toString();
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/overlay.png${q ? "?" + q : ""}`;
}

export type InstanceRle = { size: [number, number]; counts: number[] };
export type InstanceItem = {
  id: number; conf: number; bbox: [number, number, number, number];
  area: number; rle: InstanceRle;
  class_id?: number; class_name?: string; centroid?: [number, number];
};
export type InstancePrediction = {
  instances: InstanceItem[]; count: number; threshold: number; dedup_iou: number;
  // Multi-class runs report a count per class; single-class runs still
  // send `count` so older consumers keep working.
  counts_by_class?: Record<string, number>;
  class_names?: Record<string, string>;
};

/** Fetch instances.json for an instance run (readonly = never triggers inference). */
export async function fetchRunInstances(
  projectId: string, runId: string, imageId: string, readonly: boolean = true,
): Promise<InstancePrediction> {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/instances.json${readonly ? "?readonly=true" : ""}`,
  );
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

function _withHeatmapFilters(
  baseParams: string,
  threshold?: number,
  minArea?: number,
  maxArea?: number,
): string {
  // threshold is a fraction (0..1); each filter is omitted at 0 / undefined
  // so the server keeps using its legacy cached PNG when nothing is set.
  const extra: string[] = [];
  if (threshold !== undefined && threshold > 0) extra.push(`threshold=${threshold.toFixed(2)}`);
  if (minArea !== undefined && minArea > 0) extra.push(`min_area=${Math.floor(minArea)}`);
  if (maxArea !== undefined && maxArea > 0) extra.push(`max_area=${Math.floor(maxArea)}`);
  if (extra.length === 0) return baseParams;
  const sep = baseParams ? "&" : "?";
  return `${baseParams}${sep}${extra.join("&")}`;
}

export function runHeatmapConfidenceUrl(projectId: string, runId: string, imageId: string, backend?: string, tta?: boolean, threshold?: number, minArea?: number, maxArea?: number) {
  const params = _withHeatmapFilters(_buildPredictParams(backend, tta), threshold, minArea, maxArea);
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/heatmap/confidence.png${params}`;
}

export function runHeatmapClassUrl(projectId: string, runId: string, imageId: string, classId: number, backend?: string, tta?: boolean, threshold?: number, minArea?: number, maxArea?: number) {
  const params = _withHeatmapFilters(_buildPredictParams(backend, tta), threshold, minArea, maxArea);
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/heatmap/class/${classId}.png${params}`;
}

export function runHeatmapErrorUrl(projectId: string, runId: string, imageId: string, backend?: string, tta?: boolean) {
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/heatmap/error.png${_buildPredictParams(backend, tta)}`;
}

export function runPostprocessMaskUrl(
  projectId: string,
  runId: string,
  imageId: string,
  opts: {
    confidenceThreshold?: number;
    minAreaPx?: number;
    maxAreaPx?: number;
    backend?: string;
    tta?: boolean;
    readonly?: boolean;
  } = {},
): string {
  const params = new URLSearchParams();
  if (opts.backend) params.set("backend", opts.backend);
  if (opts.tta) params.set("tta", "true");
  if (opts.confidenceThreshold && opts.confidenceThreshold > 0)
    params.set("confidence_threshold", String(opts.confidenceThreshold));
  if (opts.minAreaPx && opts.minAreaPx > 0)
    params.set("min_area_px", String(opts.minAreaPx));
  if (opts.maxAreaPx && opts.maxAreaPx > 0)
    params.set("max_area_px", String(opts.maxAreaPx));
  if (opts.readonly) params.set("readonly", "true");
  const qs = params.toString();
  return `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/postprocess.png${qs ? `?${qs}` : ""}`;
}


export async function crackTrace(
  projectId: string,
  itemId: string,
  sensitivity: number = 25,
  widthPx: number = 0,
): Promise<{ label_map_b64: string; n_cracks: number; time_ms: number; crack_map_cached: boolean }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}/crack-trace`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sensitivity, width_px: widthPx }),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function crackTraceAdaptive(
  projectId: string,
  itemId: string,
  clickX: number,
  clickY: number,
  sensitivity: number = 25,
  widthPx: number = 0,
): Promise<{ label_map_b64: string | null; n_cracks: number; time_ms: number; crack_map_cached: boolean }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}/crack-trace/adaptive`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ click_x: clickX, click_y: clickY, sensitivity, width_px: widthPx }),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function superpixelMap(
  projectId: string,
  itemId: string,
  nSegments: number = 500,
): Promise<{ segments_b64: string; boundaries_b64: string; n_segments: number; time_ms: number }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}/superpixel-map`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n_segments: nSegments }),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function samSegment(
  projectId: string,
  itemId: string,
  points: [number, number][] | null,
  labels: number[] | null,
  model?: string,
  box?: [number, number, number, number] | null,
): Promise<{ mask: string; score: number; predict_time_ms: number }> {
  const payload: Record<string, unknown> = { model: model ?? "mobile_sam" };
  if (points && labels) { payload.points = points; payload.labels = labels; }
  if (box) { payload.box = box; }
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}/sam-segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function samListModels(): Promise<{ id: string; checkpoint_exists: boolean; loaded: boolean }[]> {
  const res = await fetch(`${API_BASE}/sam/models`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchRunPredictScore(projectId: string, runId: string, imageId: string, backend?: string, tta?: boolean, force?: boolean, readonly?: boolean) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/${encodeURIComponent(imageId)}/score${_buildPredictParams(backend, tta, force, readonly)}`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

/**
 * Stream batch prediction results as NDJSON.
 * Calls the backend pipeline endpoint that overlaps I/O with GPU inference.
 * @param onResult callback for each completed item
 * @param signal optional AbortSignal for cancellation
 */
export async function fetchRunPredictBatch(
  projectId: string,
  runId: string,
  itemIds: string[],
  backend: string,
  tta: boolean,
  force: boolean,
  onResult: (result: { item_id: string; status: string; score?: unknown; detail?: string; total_ms?: number }) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/batch`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_ids: itemIds, backend, tta, force }),
      signal,
    },
  );
  if (!res.ok) throw await parseApiError(res);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onResult(JSON.parse(trimmed));
      } catch {
        // skip malformed lines
      }
    }
  }
  // Process remaining buffer
  if (buffer.trim()) {
    try {
      onResult(JSON.parse(buffer.trim()));
    } catch {
      // skip
    }
  }
}
