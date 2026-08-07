// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useState } from "react";
import {
  fetchHealth,
  fetchProjectsSummary,
  thumbnailUrl,
  type Project,
  type ProjectSummary,
  type HealthInfo,
} from "../../api";
import { formatError, useI18n } from "../../i18n";
import type { StartupWarning } from "../types";

/**
 * Manages API connection lifecycle, project loading, and startup warnings.
 */
export function useApiConnection(showToast: (msg: string) => void) {
  const { lang } = useI18n();
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(
    () => sessionStorage.getItem("seg-project"),
  );
  const [projectPreviews, setProjectPreviews] = useState<Record<string, { thumbUrl: string | null; imageCount: number; maskCount: number }>>({});
  const [apiStatus, setApiStatus] = useState<"connecting" | "connected" | "error">("connecting");
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsSummaryReady, setProjectsSummaryReady] = useState(false);
  const [startupWarnings, setStartupWarnings] = useState<StartupWarning[]>([]);
  const [healthInfo, setHealthInfo] = useState<HealthInfo | null>(null);

  function applyProjectsSummary(summaries: ProjectSummary[]) {
    setProjects(summaries);
    const previews: Record<string, { thumbUrl: string | null; imageCount: number; maskCount: number }> = {};
    for (const s of summaries) {
      previews[s.id] = {
        thumbUrl: s.first_filename ? thumbnailUrl(s.id, s.first_filename) : null,
        imageCount: s.image_count,
        maskCount: s.mask_count,
      };
    }
    setProjectPreviews(previews);
  }

  const refreshProjects = useCallback(async (silent = false) => {
    if (!silent) setProjectsLoading(true);
    try {
      const summaries = await fetchProjectsSummary();
      applyProjectsSummary(summaries);
      setProjectsSummaryReady(true);
      const first = summaries[0];
      if (first) {
        setSelectedProjectId((prev) => prev ?? first.id);
      }
      setApiStatus("connected");
      showToast("");
    } catch (err) {
      setApiStatus((prev) => prev === "connected" ? "error" : prev);
      if (!silent) showToast(`API error: ${formatError(err, lang)}`);
    } finally {
      if (!silent) setProjectsLoading(false);
    }
  }, [showToast, lang]);

  // Initial connection loop
  useEffect(() => {
    let cancelled = false;
    async function connectLoop() {
      while (!cancelled) {
        try {
          await fetchHealth();
          if (cancelled) return;
          setApiStatus("connected");
          showToast("");
          // Check startup warnings
          try {
            const statusRes = await fetch("/startup-status");
            if (statusRes.ok) {
              const status = await statusRes.json();
              if (Array.isArray(status.warnings) && status.warnings.length > 0) {
                const dismissed = sessionStorage.getItem("seg_startup_warnings_dismissed");
                const toShow = dismissed ? status.warnings.filter((w: any) => w.level === "error") : status.warnings;
                if (toShow.length > 0) setStartupWarnings(toShow);
              }
            }
          } catch { /* non-critical */ }
          await refreshProjects();
          return;
        } catch {
          if (cancelled) return;
          setApiStatus("connecting");
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    }
    connectLoop();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist selectedProjectId
  useEffect(() => {
    if (selectedProjectId) {
      sessionStorage.setItem("seg-project", selectedProjectId);
    }
  }, [selectedProjectId]);

  const currentProject = projects.find((p) => p.id === selectedProjectId) ?? null;
  const currentProjectPreview = currentProject ? projectPreviews[currentProject.id] ?? null : null;

  const refreshHealthInfo = useCallback(async () => {
    try { setHealthInfo(await fetchHealth()); } catch { setHealthInfo(null); }
  }, []);

  return {
    projects, setProjects,
    selectedProjectId, setSelectedProjectId,
    projectPreviews, setProjectPreviews,
    apiStatus, projectsLoading, projectsSummaryReady,
    startupWarnings, setStartupWarnings,
    healthInfo, refreshHealthInfo,
    currentProject, currentProjectPreview,
    refreshProjects,
  } as const;
}
