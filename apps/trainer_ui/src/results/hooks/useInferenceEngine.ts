// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useRef } from "react";
import {
  API_BASE,
  annotateImageUrl,
  annotateMaskUrl,
  fetchRunPredictScore,
  fetchRunPredictBatch,
  runPredictMaskUrl,
  runPredictConfidenceUrl,
  runPostprocessMaskUrl,
  fetchPredictionStatus,
  fetchAnnotateItems,
  fetchClasses,
  fetchRunMetrics,
  fetchRunSplits,
  fetchPixelHistogram,
} from "../../api";
import type { ImageItem } from "../ImageListPanel";
import type { PredictionScore, ClassItem } from "../types";
import { DEFAULT_CLASS_COLORS } from "../types";
import type { useResultsState } from "./useResultsState";
import { useResultActions } from "./useResultActions";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TFn = (key: any) => string;

type State = ReturnType<typeof useResultsState>;

/**
 * All data-fetching, inference-engine and side-effect logic extracted from
 * the Results component.  Receives the full state bag so it can read/write
 * every piece of state without prop-drilling individual setters.
 */
export function useInferenceEngine(
  s: State,
  projectId: string | null,
  active: boolean | undefined,
  showToast: ((msg: string) => void) | undefined,
  onInferStatus: ((msg: string) => void) | undefined,
  runIdProp: string | undefined,
  t: TFn,
  onGoInspect: ((runId: string) => void) | undefined,
) {
  const setStatus = showToast ?? (() => {});

  // ── Helpers ──

  async function fetchGrayRaw(url: string, w: number, h: number) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const img = new Image();
    const objUrl = URL.createObjectURL(blob);
    const data = await new Promise<ImageData>((resolve, reject) => {
      img.onload = () => {
        URL.revokeObjectURL(objUrl);
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(img, 0, 0, w, h);
        resolve(ctx.getImageData(0, 0, w, h));
      };
      img.onerror = () => { URL.revokeObjectURL(objUrl); reject(new Error("Image decode failed")); };
      img.src = objUrl;
    });
    const out = new Uint8Array(w * h);
    for (let i = 0; i < out.length; i += 1) {
      out[i] = data.data[i * 4] ?? 0;
    }
    return out;
  }

  async function fetchMaskRaw(imageId: string, w: number, h: number, runId: string, tta: boolean = false, force: boolean = false, readonlyMode: boolean = false) {
    return fetchGrayRaw(runPredictMaskUrl(projectId as string, runId, imageId, s.predictBackend, tta, force, readonlyMode), w, h);
  }

  async function fetchConfidenceRaw(imageId: string, w: number, h: number, runId: string, tta: boolean = false, force: boolean = false, readonlyMode: boolean = false) {
    return fetchGrayRaw(runPredictConfidenceUrl(projectId as string, runId, imageId, s.predictBackend, tta, force, readonlyMode), w, h);
  }

  async function fetchScoreRaw(imageId: string, runId: string, tta: boolean = false, force: boolean = false, readonlyMode: boolean = false) {
    return fetchRunPredictScore(projectId as string, runId, imageId, s.predictBackend, tta, force, readonlyMode) as Promise<PredictionScore>;
  }

  // ── Public actions ──

  function selectImage(item: ImageItem) {
    if (!projectId) return;
    s.setActiveImageId(item.id);
    s.setImageUrl(annotateImageUrl(projectId, item.filename));
    s.setWidth(item.width);
    s.setHeight(item.height);

    const key = s.activeRunId ? s.makeCacheKey(s.activeRunId, item.id, s.predictBackend, s.ttaEnabled) : null;
    const cachedMask = key ? s.cacheRef.current.get(key) : null;
    const cachedConfidence = key ? s.confidenceCacheRef.current.get(key) : null;
    const cachedScore = key ? s.scoreCacheRef.current.get(key) ?? null : null;
    if (cachedMask && cachedMask.length === item.width * item.height) {
      s.setMaskIndex(cachedMask);
      if (cachedConfidence && cachedConfidence.length === item.width * item.height) {
        s.setConfidenceIndex(cachedConfidence);
      } else {
        s.setConfidenceIndex(new Uint8Array(item.width * item.height));
      }
      s.setPredictionScore(cachedScore);
      // Auto-apply post-processing if "Apply to All" is active
      if (s.ppApplyAll && s.activeRunId) {
        const needsPp = s.ppMinArea > 0 || s.ppMaxArea > 0 || s.confidenceThreshold > 0;
        if (needsPp) {
          const ppUrl = runPostprocessMaskUrl(projectId, s.activeRunId, item.id, {
            confidenceThreshold: s.confidenceThreshold / 100,
            minAreaPx: s.ppMinArea,
            maxAreaPx: s.ppMaxArea,
            backend: s.predictBackend,
            tta: s.ttaEnabled,
            readonly: true,
          });
          fetchGrayRaw(ppUrl, item.width, item.height)
            .then((ppMask) => { s.setMaskIndex(ppMask); })
            .catch(() => { /* keep raw mask */ });
        }
      }
    } else {
      s.setMaskIndex(new Uint8Array(item.width * item.height));
      s.setConfidenceIndex(new Uint8Array(item.width * item.height));
      s.setPredictionScore(null);
    }
  }

  // The train/val/test split the ACTIVE RUN actually used (from the run's
  // per_image_metrics.json). Overlaid onto the image list for display so
  // the set badges/filter reflect the selected model's training, not the
  // per-item manual metadata (which has no assignment UI).
  const runSplitMapRef = useRef<Map<string, ImageItem["set"]>>(new Map());

  function applyRunSplits(items: ImageItem[]): ImageItem[] {
    const map = runSplitMapRef.current;
    return items.map((item) => {
      const set = map.get(item.id) ?? "none";
      return item.set === set ? item : { ...item, set };
    });
  }

  async function loadDataset(id: string, forceReselect: boolean = false, silent: boolean = false) {
    if (!silent) s.setIsDatasetLoading(true);
    try {
      const data = await fetchAnnotateItems(id);
      if (s.currentProjectRef.current !== id) return;
      const items: ImageItem[] = applyRunSplits(data.items || []);
      s.setImages(items);
      const firstVisibleItem = items.find((item) => s.filterSet === "all" || item.set === s.filterSet) ?? items[0];
      const target =
        (!forceReselect && s.activeImageId ? items.find((item) => item.id === s.activeImageId) : undefined)
        ?? firstVisibleItem;
      if (target) {
        if (!(silent && target.id === s.activeImageId)) {
          selectImage(target);
        }
      } else {
        s.setActiveImageId(null);
        s.setImageUrl(null);
        s.setWidth(0);
        s.setHeight(0);
        s.setMaskIndex(new Uint8Array());
        s.setConfidenceIndex(new Uint8Array());
        s.setPredictionScore(null);
      }
    } catch (err) {
      setStatus(`Dataset failed: ${(err as Error).message}`);
    } finally {
      if (!silent && s.currentProjectRef.current === id) {
        s.setIsDatasetLoading(false);
      }
    }
  }

  async function loadClasses(id: string) {
    try {
      const payload = await fetchClasses(id);
      if (s.currentProjectRef.current !== id) return;
      const list: ClassItem[] = (payload.classes || []).map((cls: ClassItem) => {
        const fallback = DEFAULT_CLASS_COLORS[cls.id];
        return { ...cls, color: cls.color ?? fallback };
      });
      if (!list.find((c) => c.id === 0)) {
        list.unshift({ id: 0, name: "background", color: [0, 0, 0], active: true });
      }
      s.setClasses(list);
    } catch (err) {
      setStatus(`Classes failed: ${(err as Error).message}`);
    }
  }

  async function loadMetrics(id: string, runId: string) {
    try {
      const data = await fetchRunMetrics(id, runId);
      if (s.currentProjectRef.current !== id) return;
      s.setMetrics(data);
      setStatus("");
    } catch (err) {
      setStatus(`Metrics failed: ${(err as Error).message}`);
    }
  }

  /** Patch annotation for specific image IDs without reloading the full dataset. */
  const actions = useResultActions({ s, projectId, setStatus, t, loadDataset, fetchGrayRaw });

  async function handleRunInference(targetRunId?: string) {
    if (!projectId) { console.warn("[Infer] no projectId"); return; }
    const runId = targetRunId ?? s.activeRunId;
    if (!runId) {
      setStatus(t("results.selectModel"));
      return;
    }
    if (s.images.length === 0) {
      setStatus(t("results.noImages"));
      return;
    }
    if (s.isInferring) return;

    const targetRun = s.runs.find((r) => r.run_id === runId);

    if (targetRun?.inference_threshold != null) {
      s.setConfidenceThreshold(Math.round(targetRun.inference_threshold * 100));
    }
    const abort = new AbortController();
    s.inferAbortRef.current = abort;
    s.setIsInferring(true);
    try {
      s.setActiveRunId(runId);
      const progressMsg = (n: number) => `${t("results.inferring")} ${n}/${s.images.length}`;
      s.setInferProgress(`0/${s.images.length}`);
      onInferStatus?.(progressMsg(0));
      let success = 0;
      let lastError = "";
      let done = 0;
      const itemIds = s.images.map((img) => img.id);

      try {
        await fetchRunPredictBatch(
          projectId,
          runId,
          itemIds,
          s.predictBackend,
          s.ttaEnabled,
          true,
          (result) => {
            done++;
            if (result.status === "ok" && result.score) {
              s.scoreCacheRef.current.set(
                s.makeCacheKey(runId, result.item_id, s.predictBackend, s.ttaEnabled),
                result.score as PredictionScore,
              );
              success++;
            } else {
              lastError = result.detail || "unknown error";
              console.warn(`[Infer] ${result.item_id} failed:`, lastError);
            }
            s.setInferProgress(`${done}/${s.images.length}`);
            onInferStatus?.(progressMsg(done));
            if (done % 5 === 0) s.setCacheVersion((v) => v + 1);
          },
          abort.signal,
        );
      } catch (err) {
        if (!abort.signal.aborted) {
          lastError = (err as Error).message || "batch request failed";
          console.warn("[Infer] batch failed:", lastError);
        }
      }

      s.setInferProgress("");
      onInferStatus?.("");
      s.setCacheVersion((v) => v + 1);
      if (success > 0) {
        s.setInferredRuns((prev) => {
          const next = new Map(prev);
          const prevSet = next.get(runId) ?? new Set<string>();
          const merged = new Set(prevSet);
          for (const [key] of s.scoreCacheRef.current.entries()) {
            const prefix = `${s.predictBackend}:${runId}:`;
            if (key.startsWith(prefix)) {
              merged.add(key.slice(prefix.length).replace(/:tta$/, ""));
            }
          }
          next.set(runId, merged);
          return next;
        });
      }
      const stopped = abort.signal.aborted;
      const msg = stopped
        ? `${t("results.inferStopped")} (${success}/${s.images.length}).`
        : success
          ? `${t("results.inferComplete")} (${success}/${s.images.length}).`
          : `${t("results.inferFailed")}: ${lastError || "all images failed"}`;
      setStatus(msg);

      if (s.activeImageId) {
        const key = s.makeCacheKey(runId, s.activeImageId, s.predictBackend, s.ttaEnabled);
        const cachedScore = s.scoreCacheRef.current.get(key);
        s.setPredictionScore(cachedScore ?? null);
        const activeItem = s.images.find((item) => item.id === s.activeImageId);
        if (activeItem && cachedScore) {
          try {
            const [mask, confidence] = await Promise.all([
              fetchMaskRaw(s.activeImageId, activeItem.width, activeItem.height, runId, s.ttaEnabled, false, true),
              fetchConfidenceRaw(s.activeImageId, activeItem.width, activeItem.height, runId, s.ttaEnabled, false, true),
            ]);
            s.cacheRef.current.set(key, mask);
            s.confidenceCacheRef.current.set(key, confidence);
            s.setMaskIndex(mask);
            s.setConfidenceIndex(confidence);
          } catch {
            // Keep the batch result even if the preview refresh fails.
          }
        }
      }
    } finally {
      s.inferAbortRef.current = null;
      s.setIsInferring(false);
      onInferStatus?.("");
      const finishedRun = s.runs.find((r) => r.run_id === runId);
      if (finishedRun?.inference_threshold != null) {
        s.setConfidenceThreshold(Math.round(finishedRun.inference_threshold * 100));
      }
    }
  }

  function handleStopInference() {
    s.inferAbortRef.current?.abort();
  }

  const refreshPredictionStatus = useCallback((pid: string) => {
    void (async () => {
      const currentRuns = s.runsRef.current;
      const modelRuns = currentRuns.filter((r) => r.has_model);
      const results = await Promise.allSettled(
        modelRuns.map((run) => fetchPredictionStatus(pid, run.run_id, s.predictBackend, s.ttaEnabled)),
      );
      const newMap = new Map<string, Set<string>>();
      results.forEach((result, i) => {
        if (result.status === "fulfilled") {
          const status = result.value;
          const runId = modelRuns[i]!.run_id;
          newMap.set(runId, status.count > 0 ? new Set(status.predicted) : new Set());
          if (status.per_image_classes) {
            const prefix = `${s.predictBackend}:${runId}:`;
            for (const [imgId, classIds] of Object.entries(status.per_image_classes)) {
              const key = `${prefix}${imgId}`;
              if (!s.scoreCacheRef.current.has(key)) {
                const pcmc: Record<string, number> = {};
                for (const cid of classIds as number[]) pcmc[String(cid)] = 1;
                s.scoreCacheRef.current.set(key, {
                  backend: s.predictBackend, item_id: imgId,
                  mean_confidence: 0, foreground_mean_confidence: 0,
                  background_mean_confidence: 0, foreground_ratio: 0,
                  max_confidence: 0, min_confidence: 0,
                  per_class_mean_confidence: pcmc,
                } as PredictionScore);
              }
            }
          }
        }
      });
      const wasFirstLoad = !s.predStatusLoadedRef.current;
      s.predStatusLoadedRef.current = true;
      s.setCacheVersion((v) => v + 1);
      s.setInferredRuns((prev) => {
        if (newMap.size !== prev.size) return newMap;
        for (const [runId, newSet] of newMap) {
          const prevSet = prev.get(runId);
          if (!prevSet || prevSet.size !== newSet.size) return newMap;
          for (const id of newSet) if (!prevSet.has(id)) return newMap;
        }
        return prev;
      });
      if (wasFirstLoad) s.setCacheVersion((v) => v + 1);
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.predictBackend, s.ttaEnabled]);

  function handleMoveImageSelection(delta: number) {
    if (!s.filteredImages.length) return;
    let index = s.filteredImages.findIndex((item) => item.id === s.activeImageId);
    if (index < 0) {
      index = delta > 0 ? -1 : s.filteredImages.length;
    }
    const nextIndex = Math.min(s.filteredImages.length - 1, Math.max(0, index + delta));
    const nextItem = s.filteredImages[nextIndex];
    if (!nextItem) return;
    selectImage(nextItem);
    requestAnimationFrame(() => {
      const node = document.querySelector(`[data-image-id="${nextItem.id}"]`) as HTMLElement | null;
      node?.scrollIntoView({ block: "nearest" });
    });
  }

  // ── Effects ──

  // Sync activeRunId when runId prop changes
  useEffect(() => {
    if (runIdProp) s.setActiveRunId(runIdProp);
  }, [runIdProp]);

  // Load the active run's train/val/test split and overlay it onto the
  // image list (empty map -> every image shows as unassigned).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const map = new Map<string, ImageItem["set"]>();
      if (projectId && s.activeRunId) {
        try {
          const { splits } = await fetchRunSplits(projectId, s.activeRunId);
          for (const [itemId, split] of Object.entries(splits)) {
            if (split === "train" || split === "val" || split === "test") map.set(itemId, split);
          }
        } catch { /* run without split info — keep the map empty */ }
      }
      if (cancelled) return;
      runSplitMapRef.current = map;
      s.setImages((prev) => prev.map((item) => {
        const set = map.get(item.id) ?? "none";
        return item.set === set ? item : { ...item, set };
      }));
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, s.activeRunId]);

  // Apply inference_threshold from train_config
  useEffect(() => {
    if (!s.activeRunId) return;
    const run = s.runs.find((r) => r.run_id === s.activeRunId);
    if (run?.inference_threshold != null) {
      s.setConfidenceThreshold(Math.round(run.inference_threshold * 100));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.activeRunId]);

  // Recover inference status after hard reload
  useEffect(() => {
    if (!projectId || !s.activeRunId) return;
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/predict/batch-status?project_id=${projectId}&run_id=${s.activeRunId}`);
        if (!res.ok || cancelled) return;
        const d = await res.json();
        if (d.active && !cancelled) {
          s.setIsInferring(true);
          s.setInferProgress(`${d.completed}/${d.total}`);
          onInferStatus?.(`${t("results.inferring")} ${d.completed}/${d.total}`);
          const poll = setInterval(async () => {
            try {
              const r2 = await fetch(`${API_BASE}/predict/batch-status?project_id=${projectId}&run_id=${s.activeRunId}`);
              if (!r2.ok) { clearInterval(poll); return; }
              const d2 = await r2.json();
              if (d2.active) {
                s.setInferProgress(`${d2.completed}/${d2.total}`);
                onInferStatus?.(`${t("results.inferring")} ${d2.completed}/${d2.total}`);
              } else {
                clearInterval(poll);
                s.setIsInferring(false);
                s.setInferProgress("");
                onInferStatus?.("");
                // Clear stale front-end caches so fresh results are fetched
                s.cacheRef.current.clear();
                s.confidenceCacheRef.current.clear();
                s.scoreCacheRef.current.clear();
                s.setCacheVersion((v) => v + 1);
              }
            } catch { clearInterval(poll); }
          }, 3000);
          return () => clearInterval(poll);
        }
      } catch { /* non-critical */ }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, s.activeRunId]);

  // Load dataset + classes on project change
  useEffect(() => {
    s.setActiveImageId(null);
    s.cacheRef.current.clear();
    s.confidenceCacheRef.current.clear();
    s.scoreCacheRef.current.clear();
    s.gtMaskCacheRef.current.clear();
    s.predStatusLoadedRef.current = false;
    if (!projectId) return;
    loadDataset(projectId);
    loadClasses(projectId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Filter change → re-select image
  useEffect(() => {
    if (!projectId) return;
    if (s.filteredImages.length === 0) {
      s.setActiveImageId(null);
      s.setImageUrl(null);
      s.setWidth(0);
      s.setHeight(0);
      s.setMaskIndex(new Uint8Array());
      s.setConfidenceIndex(new Uint8Array());
      s.setPredictionScore(null);
      return;
    }
    if (s.activeImageId && s.filteredImages.some((item) => item.id === s.activeImageId)) {
      return;
    }
    selectImage(s.filteredImages[0]!);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, s.filteredImages, s.activeImageId]);

  // Tab activation / re-entry
  useEffect(() => {
    if (!projectId || !active) return;
    const isSameProjectReentry = s.prevActiveProjectRef.current === projectId;
    s.prevActiveProjectRef.current = projectId;
    if (isSameProjectReentry) {
      void loadDataset(projectId, false, true);
      void loadClasses(projectId);
    } else if (s.images.length === 0) {
      void loadDataset(projectId);
      void loadClasses(projectId);
    }
    refreshPredictionStatus(projectId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, active]);

  // When runs list changes, refresh prediction status
  useEffect(() => {
    if (!projectId || s.runs.length === 0) return;
    refreshPredictionStatus(projectId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.runs]);

  // Load metrics
  useEffect(() => {
    if (!projectId || !s.activeRunId) return;
    loadMetrics(projectId, s.activeRunId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, s.activeRunId]);

  // Pixel histogram
  useEffect(() => {
    s.setPixelHist(null);
    if (!projectId || !s.activeRunId) return;
    fetchPixelHistogram(projectId, s.activeRunId, s.predictBackend)
      .then(s.setPixelHist)
      .catch(() => s.setPixelHist(null));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, s.activeRunId, s.predictBackend]);

  // Prediction load on image switch
  useEffect(() => {
    if (!s.activeImageId || !projectId || s.width === 0 || s.height === 0) {
      s.setIsPredictionLoading(false);
      s.setPredictionLoadingLabel("");
      return;
    }
    const activeRun = s.runsRef.current.find((r) => r.run_id === s.activeRunId);
    if (!s.activeRunId || !activeRun?.has_model) {
      if (activeRun && !activeRun.has_model) setStatus("No model checkpoint for this run.");
      s.setMaskIndex(new Uint8Array(s.width * s.height));
      s.setConfidenceIndex(new Uint8Array(s.width * s.height));
      s.setPredictionScore(null);
      s.setIsPredictionLoading(false);
      s.setPredictionLoadingLabel("");
      return;
    }
    let cancelled = false;
    const key = s.makeCacheKey(s.activeRunId, s.activeImageId, s.predictBackend, s.ttaEnabled);
    const cachedMask = s.cacheRef.current.get(key);
    const cachedConfidence = s.confidenceCacheRef.current.get(key);
    const cachedScore = s.scoreCacheRef.current.get(key);
    if (cachedMask && cachedMask.length === s.width * s.height) {
      s.setMaskIndex(cachedMask);
      s.setConfidenceIndex(
        cachedConfidence && cachedConfidence.length === s.width * s.height
          ? cachedConfidence
          : new Uint8Array(s.width * s.height)
      );
      s.setPredictionScore(cachedScore ?? null);
      s.setIsPredictionLoading(false);
      s.setPredictionLoadingLabel("");
      return;
    }
    const knownPredicted = s.inferredRunsRef.current.get(s.activeRunId);
    const statusNotYetLoaded = !s.predStatusLoadedRef.current;
    if (statusNotYetLoaded || (knownPredicted !== undefined && !knownPredicted.has(s.activeImageId))) {
      s.setMaskIndex(new Uint8Array(s.width * s.height));
      s.setConfidenceIndex(new Uint8Array(s.width * s.height));
      s.setPredictionScore(null);
      s.setIsPredictionLoading(false);
      s.setPredictionLoadingLabel("");
      return;
    }

    s.setMaskIndex(new Uint8Array(s.width * s.height));
    s.setConfidenceIndex(new Uint8Array(s.width * s.height));
    s.setPredictionScore(cachedScore ?? null);
    s.setIsPredictionLoading(true);
    s.setPredictionLoadingLabel("Loading prediction...");
    void (async () => {
      try {
        const scoreP = cachedScore
          ? Promise.resolve(cachedScore)
          : fetchScoreRaw(s.activeImageId!, s.activeRunId!, s.ttaEnabled, false, true);
        const maskP = fetchMaskRaw(s.activeImageId!, s.width, s.height, s.activeRunId!, s.ttaEnabled, false, true);
        const confP = fetchConfidenceRaw(s.activeImageId!, s.width, s.height, s.activeRunId!, s.ttaEnabled, false, true);
        const results = await Promise.allSettled([scoreP, maskP, confP]);
        if (cancelled) return;
        const scoreResult = results[0];
        const maskResult = results[1];
        const confResult = results[2];
        const allFailed = scoreResult.status === "rejected"
          && maskResult.status === "rejected"
          && confResult.status === "rejected";
        if (scoreResult.status === "fulfilled") {
          const score = scoreResult.value;
          s.scoreCacheRef.current.set(key, score);
          s.setPredictionScore(score);
          s.setInferredRuns((prev) => {
            const next = new Map(prev);
            const runPredicted = new Set(next.get(s.activeRunId!) ?? []);
            runPredicted.add(s.activeImageId!);
            next.set(s.activeRunId!, runPredicted);
            return next;
          });
        } else {
          s.setPredictionScore(null);
        }
        if (maskResult.status === "fulfilled" && confResult.status === "fulfilled") {
          const mask = maskResult.value;
          const confidence = confResult.value;
          s.cacheRef.current.set(key, mask);
          s.confidenceCacheRef.current.set(key, confidence);
          s.setMaskIndex(mask);
          s.setConfidenceIndex(confidence);
        }
        if (allFailed) {
          s.setIsPredictionLoading(false);
          s.setPredictionLoadingLabel("");
        }
      } catch {
        if (!cancelled) setStatus("Prediction unavailable.");
      } finally {
        if (!cancelled) {
          s.setIsPredictionLoading(false);
          s.setPredictionLoadingLabel("");
        }
      }
    })();

    // Fire-and-forget: preload score/mask/confidence for next 2 images
    if (s.filteredImages.length > 1) {
      const curIdx = s.filteredImages.findIndex((item) => item.id === s.activeImageId);
      if (curIdx >= 0) {
        const adjacentIndices = [curIdx + 1, curIdx + 2].filter(
          (i) => i >= 0 && i < s.filteredImages.length,
        );
        for (const adjIdx of adjacentIndices) {
          const adjItem = s.filteredImages[adjIdx]!;
          const adjKnown = s.inferredRunsRef.current.get(s.activeRunId!);
          if (adjKnown !== undefined && !adjKnown.has(adjItem.id)) continue;
          const adjKey = s.makeCacheKey(s.activeRunId!, adjItem.id, s.predictBackend, s.ttaEnabled);
          if (s.scoreCacheRef.current.has(adjKey) && s.cacheRef.current.has(adjKey)) continue;
          const adjRunId = s.activeRunId!;
          void (async () => {
            if (cancelled) return;
            try {
              if (!s.scoreCacheRef.current.has(adjKey)) {
                const score = await fetchScoreRaw(adjItem.id, adjRunId, s.ttaEnabled, false, true);
                if (!cancelled) s.scoreCacheRef.current.set(adjKey, score);
              }
            } catch { /* no cached prediction */ }
            try {
              if (!cancelled && !s.cacheRef.current.has(adjKey)) {
                const [mask, confidence] = await Promise.all([
                  fetchMaskRaw(adjItem.id, adjItem.width, adjItem.height, adjRunId, s.ttaEnabled, false, true),
                  fetchConfidenceRaw(adjItem.id, adjItem.width, adjItem.height, adjRunId, s.ttaEnabled, false, true),
                ]);
                if (!cancelled) {
                  s.cacheRef.current.set(adjKey, mask);
                  s.confidenceCacheRef.current.set(adjKey, confidence);
                }
              }
            } catch { /* no cached prediction */ }
          })();
        }
      }
    }

    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.activeImageId, projectId, s.width, s.height, s.activeRunId, s.predictBackend, s.ttaEnabled, s.cacheVersion]);

  // Load GT mask
  const activeImgForGt = s.images.find((item) => item.id === s.activeImageId) ?? null;
  useEffect(() => {
    if (!projectId || !s.activeImageId || s.width === 0 || s.height === 0) {
      s.setGtMaskIndex(new Uint8Array());
      return;
    }
    if (!activeImgForGt?.annotation?.hasMask) {
      s.setGtMaskIndex(new Uint8Array());
      return;
    }
    const cacheKey = `${projectId}:${s.activeImageId}`;
    const cached = s.gtMaskCacheRef.current.get(cacheKey);
    if (cached !== undefined) {
      s.setGtMaskIndex(cached ?? new Uint8Array());
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const gt = await fetchGrayRaw(annotateMaskUrl(projectId, s.activeImageId!), s.width, s.height);
        if (!cancelled) {
          s.gtMaskCacheRef.current.set(cacheKey, gt);
          s.setGtMaskIndex(gt);
        }
      } catch {
        if (!cancelled) {
          s.gtMaskCacheRef.current.set(cacheKey, null);
          s.setGtMaskIndex(new Uint8Array());
        }
      }
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, s.activeImageId, s.width, s.height]);

  // Keyboard navigation
  useEffect(() => {
    if (!active) return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented) return;
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const activeEl = document.activeElement as HTMLElement | null;
      if (activeEl) {
        const tag = activeEl.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || activeEl.isContentEditable) {
          return;
        }
      }
      if (!s.filteredImages.length) return;
      let index = s.filteredImages.findIndex((item) => item.id === s.activeImageId);
      if (index < 0) index = 0;
      const nextIndex =
        event.key === "ArrowDown"
          ? Math.min(s.filteredImages.length - 1, index + 1)
          : Math.max(0, index - 1);
      if (nextIndex === index) return;
      event.preventDefault();
      const nextItem = s.filteredImages[nextIndex];
      if (!nextItem) return;
      selectImage(nextItem);
      requestAnimationFrame(() => {
        const node = document.querySelector(`[data-image-id="${nextItem.id}"]`) as HTMLElement | null;
        node?.scrollIntoView({ block: "nearest" });
      });
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.filteredImages, s.activeImageId, active]);

  return {
    selectImage,
    loadDataset,
    loadClasses,
    applyPredToLabel: actions.applyPredToLabel,
    bulkApplyPredToLabel: actions.bulkApplyPredToLabel,
    clearOkLabels: actions.clearOkLabels,
    handleRunInference,
    handleStopInference,
    handleApplyPostprocess: actions.handleApplyPostprocess,
    handleApplyPostprocessAll: actions.handleApplyPostprocessAll,
    handleClearPostprocessAll: actions.handleClearPostprocessAll,
    handleRestoreCache: actions.handleRestoreCache,
    handleMoveImageSelection,
    refreshPredictionStatus,
    setStatus,
  };
}
