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

export type ItemVerdict = {
  /** null when the image has no annotation: nothing to be right or wrong against. */
  verdict: "detected" | "missed" | "over" | "clean" | null;
  detected: number;
  missed: number;
  over: number;
  /** Fraction of the annotated area the prediction covered. Null when the
   *  image has no annotated foreground -- that is "no rate", not 0%. */
  match_rate: number | null;
  /** Fraction of the predicted area that sits off the annotation. Reported
   *  next to match_rate because area match alone cannot see a false alarm:
   *  predicting the whole frame would score 100%. */
  over_rate: number | null;
  gt_area: number;
  gt_covered: number;
  pred_area: number;
  pred_off_gt: number;
};


/** One point on the (confidence, min_area) surface, chosen by the server.
 *
 *  Unlike metrics.json's operating points these carry an area as well as a
 *  threshold, they are counted in defects rather than pixels, and they are
 *  measured on this run's own predictions -- so a run that finished before the
 *  sweep existed still gets them, and a run with no predictions gets none. */
export type SweptOperatingPoint = {
  threshold_pct: number;
  min_area: number;
  /** Largest area filter this threshold may carry: the smallest predicted
   *  component sitting on a defect. Null when nothing overlaps one. */
  min_area_ceiling: number | null;
  /** No larger filter is permitted here. Decided against the ladder, not by
   *  comparing min_area to the ceiling -- the ladder is geometric and a rung
   *  almost never lands exactly on it. */
  at_area_ceiling: boolean;
  detected: number;
  missed: number;
  over: number;
  f1: number;
  precision: number;
  /** Defects found without the area filter and not found with it. The guard
   *  keeps this at 0 for every preset it can. */
  defects_lost_to_area: number;
};

export type OperatingSweep = {
  points: Partial<Record<"recall_first" | "balanced" | "precision_first", SweptOperatingPoint>>;
  images: number;
  /** A point chosen over four images is not a recommendation. */
  annotated_images: number;
  grid: number;
  /** False means no setting was free of cost and the presets may spend defects. */
  guarded: boolean;
  min_area_clipped: boolean;
  min_area_max: number;
  coverage: number;
};

/** Sweep this run's predictions for named operating points. Arithmetic over
 *  files already on disk -- no inference, no model load. */
export async function fetchOperatingPoints(
  projectId: string, runId: string,
  opts: { backend?: string; tta?: boolean; coverage?: number } = {},
): Promise<OperatingSweep> {
  const params = new URLSearchParams();
  if (opts.backend) params.set("backend", opts.backend);
  if (opts.tta) params.set("tta", "true");
  if (opts.coverage) params.set("coverage", String(opts.coverage));
  const qs = params.toString();
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/operating-points${qs ? `?${qs}` : ""}`
  );
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

/** Per-image detected/missed/over-detected at one confidence-slider position.
 *
 *  One request re-judges the whole run: the components were cut when the
 *  prediction was written, so the server only sums histogram tails. */
export async function fetchRunVerdicts(
  projectId: string, runId: string,
  opts: { backend?: string; tta?: boolean; threshold?: number; minArea?: number; maxArea?: number } = {},
): Promise<{
  verdicts: Record<string, ItemVerdict>;
  counts: Record<string, number>;
  threshold: number;
  coverage: number;
  /** Predictions still without components. Non-zero means the server is
   *  building them and asking again shortly returns more. */
  pending: number;
}> {
  const params = new URLSearchParams();
  if (opts.backend) params.set("backend", opts.backend);
  if (opts.tta) params.set("tta", "true");
  if (opts.threshold) params.set("threshold", String(opts.threshold));
  if (opts.minArea) params.set("min_area", String(opts.minArea));
  if (opts.maxArea) params.set("max_area", String(opts.maxArea));
  const qs = params.toString();
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/predict/verdicts${qs ? `?${qs}` : ""}`
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
  /** Drop instances scoring below this (0..1) from the drawn overlay. The
   *  count chips filter the same field at the same value, so the picture and
   *  the number stay in step as the confidence slider moves. Omitted at 0,
   *  which is the cached, unfiltered artifact. */
  score?: number,
) {
  const params = new URLSearchParams();
  if (readonly) params.set("readonly", "true");
  if (mode) params.set("mode", mode);
  if (score && score > 0) params.set("score", String(score));
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

export type SpotDetectResult = {
  /** The spots, as a single-channel PNG whose pixels are the class id. */
  mask: string | null;
  count: number;
  sensitivity: number;
  class_id: number;
  mode: "dog" | "color";
  channel: string;
  best_snr: number;
  size_range: [number, number];
  color_tolerance: number;
  width: number;
  height: number;
  time_ms: number;
  /** The score map is kept per image, so slider moves skip the expensive part. */
  scores_cached: boolean;
  mark_recall?: number | null;
  /** The range a sensitivity sent back is held to; higher keeps fewer spots. */
  sensitivity_range?: [number, number];
  /** Set instead of a mask when the painted example could not be read. */
  why?: string;
};

/**
 * Find the spots that look like the painted one.
 *
 * `markPoints` is the pixels the user painted, as [x, y] pairs -- the example,
 * not the answer. Leave `sensitivity` out on the first call and the server
 * picks the tightest threshold that still recovers that example; pass it back
 * for slider moves.
 */
export async function spotDetect(
  projectId: string,
  itemId: string,
  markPoints: number[][],
  classId: number,
  sensitivity?: number,
): Promise<SpotDetectResult> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}/spot-detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mark_points: markPoints,
      class_id: classId,
      ...(sensitivity !== undefined ? { sensitivity } : {}),
    }),
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

/**
 * Ask SAM for the object under the given points and/or inside the box.
 *
 * Every point is a positive prompt -- a place on the object -- which is the
 * only kind the server accepts, so the labels it wants are always 1.
 */
export async function samSegment(
  projectId: string,
  itemId: string,
  points: [number, number][] | null,
  model?: string,
  box?: [number, number, number, number] | null,
): Promise<{
  mask: string;
  score: number;
  predict_time_ms: number;
  /** Every nested candidate for this click, finest first. */
  levels?: { level: string; mask: string; score: number; area: number }[];
  /** The one `mask` holds: what picking the highest score would have given. */
  default_level?: string;
}> {
  const payload: Record<string, unknown> = { model: model ?? "mobile_sam" };
  if (points && points.length) { payload.points = points; payload.labels = points.map(() => 1); }
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

// ---------------------------------------------------------------------------
// Pre-labelling: adopt a finished run's predictions as draft annotations
// ---------------------------------------------------------------------------

export type PrelabelCandidates = {
  total: number;
  unannotated: number;
  annotated: number;
};

export async function fetchPrelabelCandidates(
  projectId: string,
  runId: string,
): Promise<PrelabelCandidates> {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/prelabel/candidates`,
  );
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export type PrelabelOutcome = "written" | "skipped" | "empty" | "failed";

export type PrelabelProgress = {
  item_id?: string;
  outcome?: PrelabelOutcome;
  /** Why this one image failed. */
  detail?: string;
  /** The run itself cannot predict: the stream ends here, nothing was written. */
  error?: string;
  done: number;
  total: number;
  summary?: Record<PrelabelOutcome, number>;
};

/** Stream draft adoption, one callback per image plus a final summary line. */
export async function prelabelRun(
  projectId: string,
  runId: string,
  onProgress: (progress: PrelabelProgress) => void,
  options?: { itemIds?: string[]; overwrite?: boolean; signal?: AbortSignal },
): Promise<void> {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/train/runs/${runId}/prelabel`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_ids: options?.itemIds ?? null,
        overwrite: options?.overwrite ?? false,
      }),
      signal: options?.signal,
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
        onProgress(JSON.parse(trimmed));
      } catch {
        // skip malformed lines
      }
    }
  }
  if (buffer.trim()) {
    try {
      onProgress(JSON.parse(buffer.trim()));
    } catch {
      // skip
    }
  }
}
