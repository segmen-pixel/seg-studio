// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError, assertFileSize, MAX_UPLOAD_BYTES, getZipImportLimitBytes } from "./shared";

export async function fetchClasses(projectId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/classes`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function saveClasses(projectId: string, payload: unknown) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/classes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchClassReconcile(projectId: string): Promise<{ orphan_ids: number[]; details: Record<string, number> }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/classes/reconcile`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function postClassReconcile(projectId: string): Promise<{ added: Array<{ id: number; name: string; color: number[]; active: boolean }>; orphan_ids: number[] }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/classes/reconcile`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function purgeClass(projectId: string, classId: number) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/classes/${classId}/purge`, {
    method: "POST"
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

// Short-lived in-memory cache so multiple hooks (useImageList,
// useInferenceEngine, useTrainingScore, useRecipe) don't each issue their
// own /datasets/annotate request for the same project at mount time. The
// payload can be multi-hundred KB on large projects and the backend walks
// the project directory on every call. `sync=true` callers skip the cache
// so they always get fresh data. Mutations below invalidate the cache.
type _AnnotateCacheEntry = { data: unknown; expiresAt: number; inflight?: Promise<unknown> };
const _annotateItemsCache = new Map<string, _AnnotateCacheEntry>();
const _ANNOTATE_ITEMS_TTL_MS = 10_000;

export function invalidateAnnotateItemsCache(projectId?: string): void {
  if (projectId) _annotateItemsCache.delete(projectId);
  else _annotateItemsCache.clear();
}

export async function fetchAnnotateItems(projectId: string, sync = false) {
  if (!sync) {
    const hit = _annotateItemsCache.get(projectId);
    if (hit) {
      if (hit.inflight) return hit.inflight;
      if (hit.expiresAt > Date.now()) return hit.data;
    }
  }
  const qs = sync ? "?sync=true" : "?sync=false";
  const req = (async () => {
    const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate${qs}`);
    if (!res.ok) throw await parseApiError(res);
    return res.json();
  })();
  if (!sync) {
    _annotateItemsCache.set(projectId, { data: undefined, expiresAt: 0, inflight: req });
    try {
      const data = await req;
      _annotateItemsCache.set(projectId, { data, expiresAt: Date.now() + _ANNOTATE_ITEMS_TTL_MS });
      return data;
    } catch (err) {
      _annotateItemsCache.delete(projectId);
      throw err;
    }
  }
  const data = await req;
  // Refresh the cache so subsequent non-sync callers see the newer data.
  _annotateItemsCache.set(projectId, { data, expiresAt: Date.now() + _ANNOTATE_ITEMS_TTL_MS });
  return data;
}

export async function uploadAnnotateImages(
  projectId: string,
  files: FileList | File[],
  onProgress?: (uploaded: number, total: number) => void,
) {
  const allFiles = Array.from(files);
  const total = allFiles.length;

  // Dynamic batching driven purely by each file's size — no user
  // settings needed. Pack files until either the combined payload
  // reaches MAX_BATCH_BYTES or the per-request file count reaches
  // MAX_BATCH_FILES, whichever comes first. This way:
  //   • Tiny files (e.g. Places365 256px JPGs at ~15 KB) fill up to
  //     MAX_BATCH_FILES per request → far fewer HTTP round-trips.
  //   • Big files (e.g. 4K photos at ~2 MB) hit MAX_BATCH_BYTES
  //     first → each request stays manageable for the server's
  //     per-request PNG re-encode loop.
  const MAX_BATCH_BYTES = 10 * 1024 * 1024; // 10 MB combined payload per request
  const MAX_BATCH_FILES = 100;              // server re-encodes each to PNG
  const CONCURRENCY = 6;                    // max simultaneous POSTs

  const batches: File[][] = [];
  let current: File[] = [];
  let currentBytes = 0;
  for (const f of allFiles) {
    assertFileSize(f);
    const fits =
      current.length < MAX_BATCH_FILES &&
      (current.length === 0 || currentBytes + f.size <= MAX_BATCH_BYTES);
    if (!fits) {
      if (current.length > 0) batches.push(current);
      current = [];
      currentBytes = 0;
    }
    current.push(f);
    currentBytes += f.size;
  }
  if (current.length > 0) batches.push(current);

  let uploaded = 0;
  let lastResult: unknown = null;

  // Send batches with limited concurrency
  for (let i = 0; i < batches.length; i += CONCURRENCY) {
    const chunk = batches.slice(i, i + CONCURRENCY);
    const promises = chunk.map(async (batch) => {
      const form = new FormData();
      for (const file of batch) {
        form.append("files", file);
      }
      const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/upload`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw await parseApiError(res);
      return res.json();
    });
    const results = await Promise.all(promises);
    lastResult = results[results.length - 1];
    uploaded += chunk.reduce((sum, b) => sum + b.length, 0);
    onProgress?.(Math.min(uploaded, total), total);
  }
  invalidateAnnotateItemsCache(projectId);
  return lastResult;
}

export async function importAnnotateZip(projectId: string, file: File) {
  const maxBytes = getZipImportLimitBytes();
  assertFileSize(file, maxBytes);
  const maxGb = Math.max(1, Math.round(maxBytes / (1024 * 1024 * 1024)));
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/import_zip?max_gb=${maxGb}`, {
    method: "POST",
    body: form
  });
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

export async function uploadVideoFrames(
  projectId: string,
  file: File,
  interval: number,
  onProgress?: (msg: string) => void,
  // Localized "uploading video..." message supplied by the caller — this API
  // module has no access to the i18n context.
  uploadingMsg?: string,
): Promise<{ status: string; frame_count: number; interval: number }> {
  const form = new FormData();
  form.append("file", file);
  if (uploadingMsg) onProgress?.(uploadingMsg);
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/datasets/annotate/upload-video?interval=${interval}`,
    { method: "POST", body: form },
  );
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

export async function prepareAnnotateDataset(
  projectId: string,
  valRatio?: number,
  testRatio?: number,
) {
  const params = new URLSearchParams();
  if (valRatio !== undefined) params.set("val_ratio", String(valRatio));
  if (testRatio !== undefined) params.set("test_ratio", String(testRatio));
  const qs = params.toString();
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/datasets/annotate/prepare${qs ? "?" + qs : ""}`,
    { method: "POST" }
  );
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function deleteAnnotateItem(projectId: string, itemId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}`, {
    method: "DELETE"
  });
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

// Bulk delete: one request, one index.json write, O(N) instead of O(N²).
// Server still has a sane upper bound for index.json / file-system work, so
// we chunk on the client side too to let the UI update progress instead of
// staring at a 30-second spinner on huge deletes.
export async function deleteAnnotateItemsBulk(
  projectId: string,
  itemIds: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<{ deleted: number; not_found: number; remaining: number }> {
  const CHUNK = 2000;
  let deleted = 0;
  let notFound = 0;
  let remaining = 0;
  for (let i = 0; i < itemIds.length; i += CHUNK) {
    const batch = itemIds.slice(i, i + CHUNK);
    const res = await fetch(
      `${API_BASE}/projects/${projectId}/datasets/annotate/bulk-delete`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: batch }),
      },
    );
    if (!res.ok) throw await parseApiError(res);
    const payload = (await res.json()) as {
      deleted: number; not_found: number; remaining: number;
    };
    deleted += payload.deleted;
    notFound += payload.not_found;
    remaining = payload.remaining; // last response wins
    onProgress?.(Math.min(i + batch.length, itemIds.length), itemIds.length);
  }
  invalidateAnnotateItemsCache(projectId);
  return { deleted, not_found: notFound, remaining };
}

export async function exportAnnotateAnnotations(projectId: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/annotations/export`, {
    method: "POST"
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export function exportDatasetUrl(projectId: string, resizeScale?: number): string {
  const params = resizeScale != null && resizeScale < 1.0 ? `?resize_scale=${resizeScale}` : "";
  return `${API_BASE}/projects/${projectId}/datasets/export${params}`;
}

export interface FgAnalysisResult {
  num_masks_analyzed: number;
  num_components: number;
  p25_fg_area_px: number;
  p50_fg_area_px: number;
  recommended_scale: number;
  mean_image_size: [number, number];
  scale_details: Array<{ scale: number; p25_area_at_scale: number; safe: boolean }>;
}

export async function fetchFgAnalysis(projectId: string): Promise<FgAnalysisResult> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/fg-analysis`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export interface ResizeCloneResult {
  project_id: string;
  name: string;
  original_size: [number, number];
  train_size: [number, number];
  image_count: number;
}

export async function resizeCloneProject(projectId: string, resizeScale: number): Promise<ResizeCloneResult> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/resize-clone?resize_scale=${resizeScale}`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export function annotateImageUrl(projectId: string, filename: string) {
  return `${API_BASE}/projects/${projectId}/datasets/annotate/images/${encodeURIComponent(filename)}`;
}

export function thumbnailUrl(projectId: string, filename: string) {
  return `${API_BASE}/projects/${projectId}/datasets/annotate/images/${encodeURIComponent(filename)}/thumbnail`;
}

export function tilesDziUrl(projectId: string, imageId: string) {
  return `${API_BASE}/projects/${projectId}/tiles/${encodeURIComponent(imageId)}.dzi`;
}

export function maskTileBaseUrl(projectId: string, imageId: string) {
  return `${API_BASE}/projects/${projectId}/tiles/${encodeURIComponent(imageId)}/mask`;
}

export async function fetchTileInfo(projectId: string, imageId: string): Promise<{ tiled: boolean }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/tiles/${encodeURIComponent(imageId)}/info`);
  if (!res.ok) return { tiled: false };
  return res.json();
}

export function annotateMaskUrl(projectId: string, imageId: string) {
  return `${API_BASE}/projects/${projectId}/datasets/annotate/masks/${encodeURIComponent(imageId)}.png`;
}

export async function markImagesClean(projectId: string, imageIds: string[]): Promise<{ marked: number }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/mark-clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_ids: imageIds }),
  });
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

export async function unmarkImagesClean(projectId: string, imageIds: string[]): Promise<{ unmarked: number }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/unmark-clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_ids: imageIds }),
  });
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

export async function clearClassFromImages(
  projectId: string, imageIds: string[], classId: number,
): Promise<{ updated: number; skipped: number }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/clear-class`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_ids: imageIds, class_id: classId }),
  });
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

export type LightingVariant = "daytime" | "evening" | "night";

export type AugmentAnnotateParams = {
  count: number;
  perlin_strength?: number;
  color_jitter?: number;
  defects_per_image?: [number, number];
  /** 0 or undefined = all classes; positive = restrict synthesis to that class id. */
  class_id?: number;
  /** Either / both synthesis modes. `count` applies per mode — both on = 2*count. */
  modes?: { perlin?: boolean; lighting?: boolean };
  /** Subset of ["daytime","evening","night"]; only used when modes.lighting is true. */
  lighting_variants?: LightingVariant[];
  /** When true, "Mark Clean" images join the Perlin paste-target host pool. */
  use_clean_hosts?: boolean;
  seed?: number;
};

export async function augmentAnnotate(
  projectId: string,
  params: AugmentAnnotateParams,
): Promise<{ generated: number; items: Array<Record<string, unknown>> }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/augment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw await parseApiError(res);
  invalidateAnnotateItemsCache(projectId);
  return res.json();
}

export async function autoLabel(
  projectId: string,
  itemId: string,
  classId: number,
  erodePct: number = 5.0,
  iterations: number = 3
): Promise<Blob> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/annotate/${encodeURIComponent(itemId)}/auto_label`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_id: classId, erode_pct: erodePct, iterations })
  });
  if (!res.ok) throw await parseApiError(res);
  return res.blob();
}
