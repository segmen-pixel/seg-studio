// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError, assertFileSize, MAX_RECIPE_BYTES } from "./shared";

export type Recipe = {
  id: string;
  version: number;
  name: string;
  description?: string;
  rules: Array<{
    class_id: number;
    class_name?: string;
    steps: Array<{ type: string; params: Record<string, unknown>; combine?: string }>;
  }>;
};

export async function uploadRecipe(projectId: string, file: File): Promise<{ status: string; recipe_id: string; recipe: Recipe }> {
  assertFileSize(file, MAX_RECIPE_BYTES);
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/projects/${projectId}/recipes/import`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchActiveRecipe(projectId: string): Promise<{ recipe: Recipe | null }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/recipes/active`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function previewRecipe(
  projectId: string,
  itemId: string,
): Promise<{ mask_base64: string; width: number; height: number; fg_pixels: number; fg_ratio: number }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/recipes/preview/${encodeURIComponent(itemId)}`, {
    method: "POST",
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function applyRecipe(
  projectId: string,
  itemIds?: string[],
): Promise<{ status: string; applied: number; skipped: number; recipe_id: string }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/recipes/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(itemIds ? { item_ids: itemIds } : {}),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}


