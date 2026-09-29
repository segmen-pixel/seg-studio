// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useMemo } from "react";
import { useTrainingStore } from "../../store";
import type { Project } from "../../api";
import type { TabId } from "../types";

type PreviewMap = Record<string, { thumbUrl: string | null; imageCount: number; maskCount: number }>;

const MIN_ANNOTATED_IMAGES = 10;

/**
 * Compute the tab the user should look at next, based on project state.
 * Returns `null` when desc-mode is off or the user has no obvious next step.
 * Used to pulse the target tab as a hands-on guide.
 */
export function useGuideStep(params: {
  descMode: boolean;
  activeTab: TabId;
  selectedProjectId: string | null;
  projects: Project[];
  projectPreviews: PreviewMap;
  viewedRunIds: string[];
}): TabId | null {
  const { descMode, activeTab, selectedProjectId, projects, projectPreviews, viewedRunIds } = params;
  const runs = useTrainingStore((s) => s.runs);
  const runsProjectId = useTrainingStore((s) => s.runsProjectId);

  return useMemo<TabId | null>(() => {
    if (!descMode) return null;

    let target: TabId | null = null;
    if (!selectedProjectId || !projects.some((p) => p.id === selectedProjectId)) {
      target = "projects";
    } else {
      const preview = projectPreviews[selectedProjectId];
      if (!preview || preview.imageCount === 0) {
        target = "projects";
      } else if (preview.maskCount < MIN_ANNOTATED_IMAGES) {
        target = "annotate";
      } else if (runsProjectId === selectedProjectId) {
        const completedRuns = runs.filter(
          (r) =>
            (r.status === "completed" || r.status === "stopped" || r.status === "done") && r.has_model,
        );
        if (completedRuns.length === 0) {
          target = "training";
        } else if (completedRuns.some((r) => !viewedRunIds.includes(r.run_id))) {
          target = "training";
        }
      }
    }

    if (target === null) return null;
    // Don't pulse the tab the user is already looking at.
    if (activeTab === target) return null;
    return target;
  }, [descMode, activeTab, selectedProjectId, projects, projectPreviews, viewedRunIds, runs, runsProjectId]);
}
