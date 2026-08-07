// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useMemo, useRef, useState } from "react";
import { useTrainingStore } from "../../store";
import { useI18n } from "../../i18n";
import type { ImageItem } from "../ImageListPanel";
import type { MeasureMode, AreaUnit, Calibration } from "../MeasurementPanel";
import {
  runHeatmapConfidenceUrl,
  runHeatmapClassUrl,
  runHeatmapErrorUrl,
  runInstanceOverlayUrl,
  type InstancePrediction,
} from "../../api";
import {
  type ClassItem,
  type MetricsPayload,
  type PredictionScore,
  type RegionLabel,
  DEFAULT_CLASS_COLORS,
  fallbackColorForClass,
} from "../types";
import { countRegionsPerClass, areaPerClass } from "../MeasurementPanel";

export function useResultsState(
  projectId: string | null,
  runId: string | undefined,
) {
  const { t } = useI18n();
  const storeRuns = useTrainingStore((s) => s.runs);
  const runsProjectId = useTrainingStore((s) => s.runsProjectId);
  const runs = useMemo(
    () => (runsProjectId === projectId ? storeRuns : []),
    [runsProjectId, projectId, storeRuns],
  );
  const [activeRunId, setActiveRunId] = useState<string | null>(runId ?? null);
  // Fixed inference options: backend is ONNX-only in the UI; TTA is
  // intentionally not exposed (too slow for interactive use).
  const predictBackend: "onnx" | "coreml" = "onnx";
  const ttaEnabled = false;
  const [ppMinArea, setPpMinArea] = useState(0);
  const [ppMaxArea, setPpMaxArea] = useState(0);
  const [ppApplyAll, setPpApplyAll] = useState(false);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [images, setImages] = useState<ImageItem[]>([]);
  const [activeImageId, setActiveImageId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [filterSet, setFilterSet] = useState<"all" | ImageItem["set"]>("all");
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [overlayAlpha, setOverlayAlpha] = useState(140);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);
  const [pixelHist, setPixelHist] = useState<{ bins: number[]; counts: number[]; total_pixels: number } | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [maskIndex, setMaskIndex] = useState<Uint8Array>(new Uint8Array());
  const [confidenceIndex, setConfidenceIndex] = useState<Uint8Array>(new Uint8Array());
  const [predictionScore, setPredictionScore] = useState<PredictionScore | null>(null);
  const [gtMaskIndex, setGtMaskIndex] = useState<Uint8Array>(new Uint8Array());
  const [isDatasetLoading, setIsDatasetLoading] = useState(false);
  const [isPredictionLoading, setIsPredictionLoading] = useState(false);
  const [predictionLoadingLabel, setPredictionLoadingLabel] = useState("");

  const cacheRef = useRef<Map<string, Uint8Array>>(new Map());
  const confidenceCacheRef = useRef<Map<string, Uint8Array>>(new Map());
  const scoreCacheRef = useRef<Map<string, PredictionScore>>(new Map());
  const gtMaskCacheRef = useRef<Map<string, Uint8Array | null>>(new Map());
  const [cacheVersion, setCacheVersion] = useState(0);

  // Instance runs (v0.9.8): per-image instances.json + server-rendered overlay
  const [instanceData, setInstanceData] = useState<InstancePrediction | null>(null);
  const instanceCacheRef = useRef<Map<string, InstancePrediction | null>>(new Map());

  const [heatmapMode, setHeatmapMode] = useState<"none" | "confidence" | "class" | "error">("none");
  const [heatmapClassId, setHeatmapClassId] = useState(1);
  const [showCount, setShowCount] = useState(false);
  const [showArea, setShowArea] = useState(false);
  const measureMode: MeasureMode = showCount ? "count" : showArea ? "area" : "none";
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [unitMenuPos, setUnitMenuPos] = useState<{ x: number; y: number } | null>(null);
  const [calibrating, setCalibrating] = useState<{ unit: AreaUnit; p1: [number, number] | null } | null>(null);
  const calibLineRef = useRef<SVGLineElement | null>(null);
  const [imageBrightness, setImageBrightness] = useState(100);
  const [imageContrast, setImageContrast] = useState(100);
  const [hiddenClassIds, setHiddenClassIds] = useState<Set<number>>(new Set());

  const [inferredRuns, setInferredRuns] = useState<Map<string, Set<string>>>(new Map());
  const predStatusLoadedRef = useRef(false);
  const runsRef = useRef(runs);
  runsRef.current = runs;
  const inferredRunsRef = useRef(inferredRuns);
  inferredRunsRef.current = inferredRuns;
  const [isInferring, setIsInferring] = useState(false);
  const [inferProgress, setInferProgress] = useState("");
  const inferAbortRef = useRef<AbortController | null>(null);
  const currentProjectRef = useRef(projectId);
  currentProjectRef.current = projectId;

  const prevActiveProjectRef = useRef<string | null>(null);

  const [regionLabels, setRegionLabels] = useState<RegionLabel[]>([]);
  const [showRegionLabels, setShowRegionLabels] = useState(true);
  const regionWorkerRef = useRef<Worker | null>(null);
  const regionReqIdRef = useRef(0);

  // Canvas / preview refs and state — declared early so handleCalibrationClick can reference them
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const gtOutlineCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showGtOutline, setShowGtOutline] = useState(true);
  // Detection-highlight view: blue background wash + one vivid colour
  // per detected object (per-instance, not per-class).
  const [instanceHighlight, setInstanceHighlight] = useState(false);
  const [predOverlayPattern, setPredOverlayPattern] = useState<"none" | "tint" | "hatch" | "dots" | "crosshatch" | "fine-dots">("crosshatch");
  const previewRef = useRef<HTMLDivElement | null>(null);
  const [previewScale, setPreviewScale] = useState(1);
  const [previewOffset, setPreviewOffset] = useState({ x: 0, y: 0 });
  const panRef = useRef<{ startX: number; startY: number; ox: number; oy: number } | null>(null);

  const makeCacheKey = (runId: string, imageId: string, backend: "onnx" | "coreml", tta: boolean = false) =>
    `${backend}:${runId}:${imageId}${tta ? ":tta" : ""}`;

  const filteredImages = useMemo(
    () => images.filter((item) => filterSet === "all" || item.set === filterSet),
    [images, filterSet],
  );

  const activeImage = useMemo(
    () => images.find((item) => item.id === activeImageId) ?? null,
    [images, activeImageId],
  );

  // Instance-run detection: the run list carries training_mode; the run's
  // train_config.json (via metrics payload) is the fallback for stale lists.
  const isInstanceRun = useMemo(() => {
    const activeRun = runs.find((r) => r.run_id === activeRunId);
    if (activeRun?.training_mode) return activeRun.training_mode === "instance";
    return metrics?.config?.["training_mode"] === "instance";
  }, [runs, activeRunId, metrics]);

  const instanceOverlayUrl = useMemo(() => {
    if (!isInstanceRun || !projectId || !activeRunId || !activeImageId || !instanceData) return null;
    return runInstanceOverlayUrl(projectId, activeRunId, activeImageId, true,
      instanceHighlight ? "instance" : "class");
  }, [isInstanceRun, projectId, activeRunId, activeImageId, instanceData, instanceHighlight]);

  const presentClassIds = useMemo(() => {
    const ids = new Set<number>();
    for (let i = 0; i < maskIndex.length; i += 1) {
      const id = maskIndex[i] ?? 0;
      if (id > 0) ids.add(id);
    }
    if (ids.size === 0 && predictionScore?.per_class_mean_confidence) {
      Object.keys(predictionScore.per_class_mean_confidence)
        .map(Number)
        .filter((id) => id > 0)
        .forEach((id) => ids.add(id));
    }
    return Array.from(ids).sort((a, b) => a - b);
  }, [maskIndex, predictionScore]);

  const heatmapUrl = useMemo(() => {
    if (isInstanceRun) return null; // semantic-only artifact (no probs for rfdetr)
    if (!projectId || !activeRunId || !activeImageId || heatmapMode === "none") return null;
    const b = predictBackend;
    const t = ttaEnabled;
    // Confidence slider is 0..100 (%); server expects 0..1 fraction.
    const thr = confidenceThreshold > 0 ? confidenceThreshold / 100 : undefined;
    const mn = ppMinArea > 0 ? ppMinArea : undefined;
    const mx = ppMaxArea > 0 ? ppMaxArea : undefined;
    if (heatmapMode === "confidence") return runHeatmapConfidenceUrl(projectId, activeRunId, activeImageId, b, t, thr, mn, mx);
    if (heatmapMode === "class") return runHeatmapClassUrl(projectId, activeRunId, activeImageId, heatmapClassId, b, t, thr, mn, mx);
    if (heatmapMode === "error") return runHeatmapErrorUrl(projectId, activeRunId, activeImageId, b, t);
    return null;
  }, [isInstanceRun, projectId, activeRunId, activeImageId, heatmapMode, heatmapClassId, predictBackend, ttaEnabled, confidenceThreshold, ppMinArea, ppMaxArea]);

  const effectiveClasses = useMemo(() => {
    const byId = new Map<number, ClassItem>();
    classes.forEach((cls) => {
      const color = cls.color ?? fallbackColorForClass(cls.id);
      byId.set(cls.id, { ...cls, color });
    });
    presentClassIds.forEach((id) => {
      if (!byId.has(id)) {
        byId.set(id, {
          id,
          name: `class${id}`,
          color: fallbackColorForClass(id),
          active: true,
        });
      }
    });
    return Array.from(byId.values()).sort((a, b) => a.id - b.id);
  }, [classes, presentClassIds]);

  const lut = useMemo(() => {
    const table = new Uint8ClampedArray(256 * 4);
    for (let i = 0; i < 256; i += 1) table[i * 4 + 3] = 0;
    effectiveClasses.forEach((cls) => {
      const base = cls.id * 4;
      table[base] = cls.color[0];
      table[base + 1] = cls.color[1];
      table[base + 2] = cls.color[2];
      table[base + 3] = cls.id === 0 || hiddenClassIds.has(cls.id) ? 0 : overlayAlpha;
    });
    return table;
  }, [effectiveClasses, overlayAlpha, hiddenClassIds]);

  const perImageClassIds = useMemo(() => {
    const map = new Map<string, number[]>();
    if (!activeRunId) return map;
    const prefix = `${predictBackend}:${activeRunId}:`;
    for (const [key, score] of scoreCacheRef.current.entries()) {
      if (!key.startsWith(prefix)) continue;
      const imageId = key.slice(prefix.length).replace(/:tta$/, "");
      const classIds = score.per_class_mean_confidence
        ? Object.keys(score.per_class_mean_confidence).map(Number).filter((id) => id > 0).sort((a, b) => a - b)
        : [];
      map.set(imageId, classIds);
    }
    return map;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRunId, predictBackend, cacheVersion]);

  const classColorMap = useMemo(() => {
    const m = new Map<number, [number, number, number]>();
    for (const c of effectiveClasses) {
      if (c.id !== 0) m.set(c.id, c.color);
    }
    return m;
  }, [effectiveClasses]);

  // The threshold the server already applied when it produced this mask.
  const shippedThresholdPct = useMemo(() => {
    const run = runs.find((r) => r.run_id === activeRunId);
    const v = run?.inference_threshold;
    return v != null ? Math.round(v * 100) : null;
  }, [runs, activeRunId]);

  const effectiveMaskIndex = useMemo(() => {
    if (confidenceThreshold === 0 || confidenceIndex.length !== maskIndex.length) return maskIndex;
    // Do not apply the run's own threshold a second time. The server thresholds
    // at model-output resolution and NEAREST-upsamples the mask, but the
    // confidence PNG is INTER_LINEAR-upsampled: at output_stride 2 each
    // destination pixel is 0.75 of its own source pixel plus 0.25 of a
    // neighbour, so re-testing it here strips a half-source-pixel ring off
    // every blob and deletes single-source-pixel blobs outright. The slider is
    // auto-armed to the run's threshold, so that was the DEFAULT state, and it
    // made every number computed from effectiveMaskIndex -- fg_ratio, region
    // counts, class areas, the vs-GT scores -- smaller than the server's for
    // the identical prediction. Only filter when the user has raised the
    // slider above what the server already enforced.
    if (shippedThresholdPct != null && confidenceThreshold <= shippedThresholdPct) return maskIndex;
    const thresholdByte = Math.round((confidenceThreshold / 100) * 255);
    const result = new Uint8Array(maskIndex.length);
    for (let i = 0; i < maskIndex.length; i++) {
      if ((maskIndex[i] ?? 0) !== 0 && (confidenceIndex[i] ?? 0) >= thresholdByte) {
        result[i] = maskIndex[i]!;
      }
    }
    return result;
  }, [maskIndex, confidenceIndex, confidenceThreshold, shippedThresholdPct]);

  const liveStats = useMemo(() => {
    if (effectiveMaskIndex.length === 0 || width === 0 || height === 0) return null;
    const total = width * height;
    let fgCount = 0;
    let confSum = 0;
    let fgConfSum = 0;
    let fgConfCount = 0;
    const hasConf = confidenceIndex.length === effectiveMaskIndex.length;
    const perClassConf = new Map<number, { sum: number; count: number }>();
    for (let i = 0; i < effectiveMaskIndex.length; i++) {
      const classId = effectiveMaskIndex[i] ?? 0;
      const conf = hasConf ? (confidenceIndex[i] ?? 0) : 255;
      confSum += conf;
      if (classId === 0) continue;
      fgCount++;
      fgConfSum += conf;
      fgConfCount++;
      const entry = perClassConf.get(classId);
      if (entry) { entry.sum += conf; entry.count++; }
      else perClassConf.set(classId, { sum: conf, count: 1 });
    }
    return {
      fg_ratio: fgCount / total,
      mean_confidence: total > 0 ? confSum / total / 255 : 0,
      fg_mean_confidence: fgConfCount > 0 ? fgConfSum / fgConfCount / 255 : 0,
      per_class_mean_confidence: Object.fromEntries(
        Array.from(perClassConf.entries()).map(([id, { sum, count }]) => [String(id), sum / count / 255])
      ),
    };
  }, [effectiveMaskIndex, confidenceIndex, width, height]);

  const gtMetrics = useMemo(() => {
    if (gtMaskIndex.length === 0 || effectiveMaskIndex.length === 0 ||
        gtMaskIndex.length !== effectiveMaskIndex.length) return null;
    const activeRun = runs.find((r) => r.run_id === activeRunId);
    const activeIds = activeRun?.active_class_ids;
    const validFg = activeIds ? new Set(activeIds.filter((id) => id > 0)) : null;
    let hasGtFg = false;
    for (let i = 0; i < gtMaskIndex.length; i++) {
      const v = gtMaskIndex[i] ?? 0;
      if (v > 0 && (!validFg || validFg.has(v))) { hasGtFg = true; break; }
    }
    if (!hasGtFg) return null;
    const allClasses = new Set<number>();
    for (let i = 0; i < gtMaskIndex.length; i++) {
      const gtRaw = gtMaskIndex[i] ?? 0;
      const gt = (gtRaw > 0 && validFg && !validFg.has(gtRaw)) ? 0 : gtRaw;
      const pred = effectiveMaskIndex[i] ?? 0;
      if (gt > 0) allClasses.add(gt);
      if (pred > 0) allClasses.add(pred);
    }
    if (allClasses.size === 0) return null;
    const perClass = new Map<number, { tp: number; fp: number; fn: number }>();
    for (const id of allClasses) perClass.set(id, { tp: 0, fp: 0, fn: 0 });
    for (let i = 0; i < gtMaskIndex.length; i++) {
      const gtRaw2 = gtMaskIndex[i] ?? 0;
      const gt = (gtRaw2 > 0 && validFg && !validFg.has(gtRaw2)) ? 0 : gtRaw2;
      const pred = effectiveMaskIndex[i] ?? 0;
      for (const id of allClasses) {
        const isGt = gt === id;
        const isPred = pred === id;
        const entry = perClass.get(id)!;
        if (isGt && isPred) entry.tp++;
        else if (!isGt && isPred) entry.fp++;
        else if (isGt && !isPred) entry.fn++;
      }
    }
    const results = new Map<number, { f1: number; precision: number; recall: number; iou: number }>();
    for (const [id, { tp, fp, fn }] of perClass) {
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
      const iou = tp + fp + fn > 0 ? tp / (tp + fp + fn) : 0;
      results.set(id, { f1, precision, recall, iou });
    }
    return results;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gtMaskIndex, effectiveMaskIndex, activeRunId, runs]);

  const regionCounts = useMemo(() => {
    if (!showCount) return new Map<number, number>();
    if (isInstanceRun) {
      // Instance runs count from instances.json (dedup-aware), not union-find.
      if (!instanceData || instanceData.count === 0) return new Map<number, number>();
      const byClass = instanceData.counts_by_class;
      if (byClass && Object.keys(byClass).length > 0) {
        return new Map<number, number>(
          Object.entries(byClass).map(([cid, n]) => [Number(cid), n]));
      }
      const classId = presentClassIds[0]
        ?? effectiveClasses.find((c) => c.id > 0)?.id
        ?? 1;
      return new Map<number, number>([[classId, instanceData.count]]);
    }
    if (effectiveMaskIndex.length === 0 || width === 0) return new Map<number, number>();
    return countRegionsPerClass(effectiveMaskIndex, width, height);
  }, [showCount, isInstanceRun, instanceData, presentClassIds, effectiveClasses, effectiveMaskIndex, width, height]);

  const classAreas = useMemo(() => {
    if (!showArea || effectiveMaskIndex.length === 0) return new Map<number, number>();
    return areaPerClass(effectiveMaskIndex);
  }, [showArea, effectiveMaskIndex]);

  const handleCalibrationClick = useCallback((e: React.PointerEvent) => {
    if (!calibrating || !previewRef.current) return;
    const rect = previewRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left - previewOffset.x) / previewScale;
    const y = (e.clientY - rect.top - previewOffset.y) / previewScale;
    if (!calibrating.p1) {
      setCalibrating({ ...calibrating, p1: [x, y] });
    } else {
      const [x1, y1] = calibrating.p1;
      const pixelDist = Math.sqrt((x - x1) ** 2 + (y - y1) ** 2);
      if (pixelDist < 2) return;
      const input = prompt(
        t("results.calibrate.prompt").replace("{unit}", calibrating.unit),
        "1"
      );
      if (input) {
        const realDist = parseFloat(input);
        if (realDist > 0) {
          setCalibration({ pixelDist, realDist, unit: calibrating.unit });
        }
      }
      setCalibrating(null);
    }
  }, [calibrating, previewOffset.x, previewOffset.y, previewScale, t]);

  const currentImageNotInferred = useMemo(() => {
    if (!activeImageId || !activeRunId) return false;
    const activeRun = runs.find((r) => r.run_id === activeRunId);
    if (!activeRun?.has_model) return false;
    if (isPredictionLoading) return false;
    // If this image was already inferred (tracked in inferredRuns), it's not "not inferred"
    const known = inferredRuns.get(activeRunId);
    if (known?.has(activeImageId)) return false;
    return predictionScore === null;
  }, [activeImageId, activeRunId, runs, isPredictionLoading, predictionScore, inferredRuns]);

  const detected = useMemo(() => {
    const fallbackFromScore = () => {
      const src = liveStats?.per_class_mean_confidence ?? predictionScore?.per_class_mean_confidence;
      if (!src) return [];
      return Object.entries(src)
        .map(([rawId, confidence]) => {
          const id = Number(rawId);
          const cls = effectiveClasses.find((c) => c.id === id);
          return {
            id,
            name: cls?.name ?? `class${id}`,
            color: cls?.color ?? [180, 180, 180] as [number, number, number],
            ratio: liveStats?.fg_ratio ?? predictionScore?.foreground_ratio ?? 0,
            meanConfidence: confidence,
          };
        })
        .filter((item) => item.id > 0)
        .sort((a, b) => b.meanConfidence - a.meanConfidence);
    };
    if (width === 0 || height === 0 || effectiveMaskIndex.length === 0) {
      return fallbackFromScore();
    }
    const counts = new Map<number, number>();
    for (let i = 0; i < effectiveMaskIndex.length; i += 1) {
      const id = effectiveMaskIndex[i] ?? 0;
      if (id === 0) continue;
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    if (counts.size === 0) {
      if (confidenceThreshold > 0) return [];
      return fallbackFromScore();
    }
    const total = width * height;
    return Array.from(counts.entries())
      .map(([id, count]) => {
        const cls = effectiveClasses.find((c) => c.id === id);
        return {
          id,
          name: cls?.name ?? `class${id}`,
          color: cls?.color ?? [180, 180, 180] as [number, number, number],
          count,
          ratio: total ? count / total : 0,
          meanConfidence: liveStats?.per_class_mean_confidence?.[String(id)]
            ?? predictionScore?.per_class_mean_confidence?.[String(id)] ?? null,
        };
      })
      .sort((a, b) => (b.count ?? 0) - (a.count ?? 0));
  }, [effectiveMaskIndex, confidenceThreshold, width, height, effectiveClasses, liveStats, predictionScore]);

  const scoreOnlyClasses = useMemo(() => {
    if (!predictionScore && !liveStats) return [];
    const src = liveStats?.per_class_mean_confidence ?? predictionScore?.per_class_mean_confidence;
    const perClassItems = src
      ? Object.entries(src)
          .map(([rawId, confidence]) => {
            const id = Number(rawId);
            const cls = effectiveClasses.find((c) => c.id === id);
            return {
              id,
              name: cls?.name ?? `class${id}`,
              color: cls?.color ?? [180, 180, 180] as [number, number, number],
              meanConfidence: confidence,
              ratio: liveStats?.fg_ratio ?? predictionScore?.foreground_ratio ?? 0,
            };
          })
          .filter((item) => item.id > 0)
      : [];
    if (perClassItems.length > 0) {
      return perClassItems.sort((a, b) => b.meanConfidence - a.meanConfidence);
    }
    const fgRatio = liveStats?.fg_ratio ?? predictionScore?.foreground_ratio ?? 0;
    const fgConf = liveStats?.fg_mean_confidence ?? predictionScore?.foreground_mean_confidence ?? 0;
    if (fgRatio <= 0 && fgConf <= 0) {
      return [];
    }
    const foregroundClass =
      effectiveClasses.find((cls) => cls.id > 0) ??
      {
        id: 1,
        name: "foreground",
        color: fallbackColorForClass(1),
        active: true,
      };
    return [
      {
        id: foregroundClass.id,
        name: foregroundClass.name,
        color: foregroundClass.color,
        meanConfidence: fgConf,
        ratio: fgRatio,
      },
    ];
  }, [effectiveClasses, predictionScore, liveStats]);

  const visiblePredictionClasses = detected.length > 0
    ? detected
    : confidenceThreshold > 0 ? [] : scoreOnlyClasses;

  return {
    // Runs
    runs,
    activeRunId, setActiveRunId,
    // Backend / TTA
    predictBackend, ttaEnabled,
    // Post-processing
    ppMinArea, setPpMinArea,
    ppMaxArea, setPpMaxArea,
    ppApplyAll, setPpApplyAll,
    // Metrics
    metrics, setMetrics,
    // Images
    images, setImages,
    activeImageId, setActiveImageId,
    selectedIds, setSelectedIds,
    filterSet, setFilterSet,
    activeImage,
    filteredImages,
    // Classes
    classes, setClasses,
    effectiveClasses,
    lut,
    presentClassIds,
    perImageClassIds,
    classColorMap,
    // Overlay
    overlayAlpha, setOverlayAlpha,
    overlayVisible, setOverlayVisible,
    // Confidence
    confidenceThreshold, setConfidenceThreshold,
    pixelHist, setPixelHist,
    // Image data
    imageUrl, setImageUrl,
    width, setWidth,
    height, setHeight,
    maskIndex, setMaskIndex,
    confidenceIndex, setConfidenceIndex,
    effectiveMaskIndex,
    predictionScore, setPredictionScore,
    gtMaskIndex, setGtMaskIndex,
    // Loading states
    isDatasetLoading, setIsDatasetLoading,
    isPredictionLoading, setIsPredictionLoading,
    predictionLoadingLabel, setPredictionLoadingLabel,
    // Caches
    cacheRef, confidenceCacheRef, scoreCacheRef, gtMaskCacheRef,
    cacheVersion, setCacheVersion,
    // Instance runs
    isInstanceRun,
    instanceData, setInstanceData,
    instanceCacheRef,
    instanceOverlayUrl,
    // Heatmap
    heatmapMode, setHeatmapMode,
    heatmapClassId, setHeatmapClassId,
    heatmapUrl,
    // Measure
    showCount, setShowCount,
    showArea, setShowArea,
    measureMode,
    calibration, setCalibration,
    unitMenuPos, setUnitMenuPos,
    calibrating, setCalibrating,
    calibLineRef,
    handleCalibrationClick,
    regionCounts, classAreas,
    // Brightness / contrast
    imageBrightness, setImageBrightness,
    imageContrast, setImageContrast,
    hiddenClassIds, setHiddenClassIds,
    // Inference state
    inferredRuns, setInferredRuns,
    predStatusLoadedRef,
    runsRef, inferredRunsRef,
    isInferring, setIsInferring,
    inferProgress, setInferProgress,
    inferAbortRef,
    currentProjectRef,
    prevActiveProjectRef,
    // Computed
    liveStats,
    gtMetrics,
    currentImageNotInferred,
    detected,
    scoreOnlyClasses,
    visiblePredictionClasses,
    // Canvas
    imageCanvasRef, overlayCanvasRef, gtOutlineCanvasRef, previewRef,
    showGtOutline, setShowGtOutline,
    instanceHighlight, setInstanceHighlight,
    predOverlayPattern, setPredOverlayPattern,
    previewScale, setPreviewScale,
    previewOffset, setPreviewOffset,
    panRef,
    // Helpers
    makeCacheKey,
    // Region labels
    regionLabels, setRegionLabels,
    showRegionLabels, setShowRegionLabels,
    regionWorkerRef, regionReqIdRef,
  };
}
