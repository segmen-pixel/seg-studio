// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Label management, post-processing, and cache restoration actions
 * extracted from useInferenceEngine.
 */
import {
  runPredictMaskUrl,
  runPostprocessMaskUrl,
  unmarkImagesClean,
} from "../../api";
import type { ImageItem } from "../ImageListPanel";
import type { useResultsState } from "./useResultsState";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TFn = (key: any) => string;
type State = ReturnType<typeof useResultsState>;

type ResultActionsArgs = {
  s: State;
  projectId: string | null;
  setStatus: (msg: string) => void;
  t: TFn;
  loadDataset: (id: string) => Promise<void>;
  fetchGrayRaw: (url: string, w: number, h: number) => Promise<Uint8Array>;
};

export function useResultActions({ s, projectId, setStatus, t, loadDataset, fetchGrayRaw }: ResultActionsArgs) {

  function patchImageAnnotations(patches: Map<string, Record<string, unknown>>) {
    s.setImages((prev) =>
      prev.map((img) => {
        const ann = patches.get(img.id);
        return ann ? { ...img, annotation: { ...img.annotation, ...ann } as ImageItem["annotation"] } : img;
      }),
    );
  }

  async function applyPredToLabel(imageId: string) {
    if (!projectId || !s.activeRunId) return;
    if (!window.confirm(t("imageList.applyPredToLabelConfirm"))) return;
    const maskUrl = runPredictMaskUrl(projectId, s.activeRunId, imageId, s.predictBackend);
    try {
      const res = await fetch(maskUrl);
      if (!res.ok) throw new Error(`Fetch mask failed: ${res.status}`);
      const blob = await res.blob();
      const pngBlob = new Blob([blob], { type: "image/png" });
      const formData = new FormData();
      formData.append("file", pngBlob, `${imageId}.png`);
      const putRes = await fetch(`/api/v1/projects/${projectId}/datasets/annotate/masks/${encodeURIComponent(imageId)}.png`, {
        method: "PUT",
        body: formData,
      });
      if (!putRes.ok) {
        const detail = await putRes.text().catch(() => "");
        throw new Error(`Save mask failed: ${putRes.status} ${detail}`);
      }
      const result = await putRes.json();
      if (result.annotation) {
        patchImageAnnotations(new Map([[imageId, result.annotation]]));
      }
      setStatus(t("imageList.applyPredToLabelDone"));
    } catch (err) {
      setStatus(`${t("imageList.applyPredToLabelFail")}: ${(err as Error).message}`);
    }
  }

  async function bulkApplyPredToLabel(imageIds: string[]): Promise<boolean> {
    if (!projectId || !s.activeRunId || imageIds.length === 0) return false;
    if (!window.confirm(t("imageList.bulkApplyConfirm").replace("{count}", String(imageIds.length)))) return false;
    let ok = 0;
    let fail = 0;
    const patches = new Map<string, Record<string, unknown>>();
    for (const imageId of imageIds) {
      try {
        const maskUrl = runPredictMaskUrl(projectId, s.activeRunId, imageId, s.predictBackend);
        const res = await fetch(maskUrl);
        if (!res.ok) throw new Error(`${res.status}`);
        const blob = await res.blob();
        const pngBlob = new Blob([blob], { type: "image/png" });
        const formData = new FormData();
        formData.append("file", pngBlob, `${imageId}.png`);
        const putRes = await fetch(`/api/v1/projects/${projectId}/datasets/annotate/masks/${encodeURIComponent(imageId)}.png`, {
          method: "PUT",
          body: formData,
        });
        if (!putRes.ok) throw new Error(`${putRes.status}`);
        const result = await putRes.json();
        if (result.annotation) patches.set(imageId, result.annotation);
        ok++;
        setStatus(`${t("imageList.bulkApplyProgress")} ${ok}/${imageIds.length}`);
      } catch {
        fail++;
      }
    }
    if (patches.size > 0) patchImageAnnotations(patches);
    s.setSelectedIds(new Set());
    if (fail === 0) {
      setStatus(t("imageList.bulkApplyDone").replace("{count}", String(ok)));
    } else {
      setStatus(t("imageList.bulkApplyPartial").replace("{ok}", String(ok)).replace("{fail}", String(fail)));
    }
    return true;
  }

  async function clearOkLabels(imageIds: string[]) {
    if (!projectId || imageIds.length === 0) return;
    if (!window.confirm(t("imageList.clearOkConfirm").replace("{count}", String(imageIds.length)))) return;
    try {
      const result = await unmarkImagesClean(projectId, imageIds);
      s.setSelectedIds(new Set());
      setStatus(t("imageList.clearOkDone").replace("{count}", String(result.unmarked)));
      loadDataset(projectId);
    } catch (err) {
      setStatus(`${t("imageList.clearOkFail")}: ${(err as Error).message}`);
    }
  }

  async function handleApplyPostprocess() {
    if (!projectId || !s.activeRunId || !s.activeImageId || s.width === 0 || s.height === 0) return;
    const needsPostprocess = s.ppMinArea > 0 || s.ppMaxArea > 0 || s.confidenceThreshold > 0;
    if (!needsPostprocess) {
      s.setPpApplyAll(false);
      const key = s.makeCacheKey(s.activeRunId, s.activeImageId, s.predictBackend, s.ttaEnabled);
      const cachedMask = s.cacheRef.current.get(key);
      if (cachedMask && cachedMask.length === s.width * s.height) {
        s.setMaskIndex(cachedMask);
      }
      return;
    }
    try {
      setStatus("Applying post-processing...");
      const url = runPostprocessMaskUrl(projectId, s.activeRunId, s.activeImageId, {
        confidenceThreshold: s.confidenceThreshold / 100,
        minAreaPx: s.ppMinArea,
        maxAreaPx: s.ppMaxArea,
        backend: s.predictBackend,
        tta: s.ttaEnabled,
      });
      const ppMask = await fetchGrayRaw(url, s.width, s.height);
      s.setMaskIndex(ppMask);
    } catch (err) {
      setStatus(`Post-process failed: ${(err as Error).message}`);
    }
  }

  function handleApplyPostprocessAll() {
    s.setPpApplyAll(true);
    handleApplyPostprocess();
  }

  function handleClearPostprocessAll() {
    s.setPpApplyAll(false);
    if (!s.activeRunId || !s.activeImageId) return;
    const key = s.makeCacheKey(s.activeRunId, s.activeImageId, s.predictBackend, s.ttaEnabled);
    const cachedMask = s.cacheRef.current.get(key);
    if (cachedMask && cachedMask.length === s.width * s.height) {
      s.setMaskIndex(cachedMask);
    }
  }

  async function handleRestoreCache() {
    if (!projectId || !s.activeRunId) return;
    s.cacheRef.current.clear();
    s.confidenceCacheRef.current.clear();
    s.scoreCacheRef.current.clear();
    s.setCacheVersion((v) => v + 1);
  }

  return {
    applyPredToLabel,
    bulkApplyPredToLabel,
    clearOkLabels,
    handleApplyPostprocess,
    handleApplyPostprocessAll,
    handleClearPostprocessAll,
    handleRestoreCache,
  };
}
