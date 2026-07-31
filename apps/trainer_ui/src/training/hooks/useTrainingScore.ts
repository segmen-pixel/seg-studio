// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useState, useCallback } from "react";
import {
  fetchAnnotateItems,
  fetchClasses,
  fetchRunPredictScore,
} from "../../api";
import type { PredictionScore, ClassInfo } from "../types";
import type { TrainRunItem } from "../../utils";

export type ScoreSortKey = "name" | "confidence" | "fg_ratio";

export function useTrainingScore<T extends (key: any) => string>(
  projectId: string | null,
  runs: TrainRunItem[],
  showToast: (msg: string) => void,
  t: T,
) {
  const [scoreData, setScoreData] = useState<Map<string, PredictionScore>>(new Map());
  const [scoreLoading, setScoreLoading] = useState(false);
  const [scoreProgress, setScoreProgress] = useState("");
  const [scoreRunId, setScoreRunId] = useState<string | null>(null);
  const [scoreClasses, setScoreClasses] = useState<ClassInfo[]>([]);
  const [scoreImageNames, setScoreImageNames] = useState<Map<string, string>>(new Map());
  const [scoreTotalImages, setScoreTotalImages] = useState(0);
  const [scoreSortKey, setScoreSortKey] = useState<ScoreSortKey>("name");
  const [scoreSortAsc, setScoreSortAsc] = useState(true);

  const loadScoreData = useCallback(async (selectedRunId: string | null) => {
    if (!projectId || !selectedRunId) return;
    const selectedRun = runs.find((r) => r.run_id === selectedRunId);
    if (!selectedRun?.has_model) {
      showToast(t("training.noModelYet"));
      return;
    }
    setScoreLoading(true);
    setScoreData(new Map());
    setScoreProgress(t("training.loadingImages"));
    try {
      const [itemsRes, classesRes] = await Promise.all([
        fetchAnnotateItems(projectId),
        fetchClasses(projectId),
      ]);
      const items: Array<{ id: string; name: string; filename: string }> = itemsRes.items || [];
      const classes: ClassInfo[] = (classesRes.classes || []).filter((c: ClassInfo) => c.id !== 0);
      setScoreClasses(classes);
      const nameMap = new Map<string, string>();
      for (const item of items) nameMap.set(item.id, item.name);
      setScoreImageNames(nameMap);
      setScoreRunId(selectedRunId);
      setScoreTotalImages(items.length);

      const results = new Map<string, PredictionScore>();
      let done = 0;
      let failed = 0;
      const CONCURRENCY = 1;
      const queue = [...items];
      const workers = Array.from({ length: Math.min(CONCURRENCY, queue.length) }, async () => {
        while (queue.length > 0) {
          const item = queue.shift()!;
          try {
            const score = await fetchRunPredictScore(projectId, selectedRunId, item.id);
            results.set(item.id, score);
          } catch (err) {
            console.warn("Training: fetch score failed for item", item.id, err);
            failed++;
          }
          done++;
          setScoreProgress(`Running ${done}/${items.length}${failed > 0 ? ` (${failed} failed)` : ""}...`);
        }
      });
      await Promise.all(workers);
      setScoreData(new Map(results));
      setScoreProgress("");
    } catch (err) {
      showToast(`Score failed: ${(err as Error).message}`);
      setScoreProgress("");
    } finally {
      setScoreLoading(false);
    }
  }, [projectId, runs, showToast, t]);

  const refreshClasses = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await fetchClasses(projectId);
      const classes: ClassInfo[] = (res.classes || []);
      setScoreClasses(classes);
    } catch (e: unknown) {
      console.warn("Training: fetch classes failed:", e);
    }
  }, [projectId]);

  return {
    scoreData,
    scoreLoading,
    scoreProgress,
    scoreRunId,
    scoreClasses,
    scoreImageNames,
    scoreTotalImages,
    scoreSortKey,
    scoreSortAsc,
    setScoreSortKey,
    setScoreSortAsc,
    loadScoreData,
    refreshClasses,
  };
}
