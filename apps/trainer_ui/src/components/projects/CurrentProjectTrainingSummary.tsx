// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useI18n } from "../../i18n";
import { useTrainingStore } from "../../store";

// Extracted verbatim from ProjectsPanel.tsx (pre-OSS refactor).
/**
 * Compact training summary for the currently selected project card.
 *
 * Subscribes to the shared training store (populated by useGlobalPolling at
 * 15-30s cadence). Stale-guards on runsProjectId so a freshly selected project
 * does not briefly show the previous project's stats while the next poll
 * fetches its runs.
 */
export function CurrentProjectTrainingSummary({ projectId }: { projectId: string }) {
  const { t } = useI18n();
  const runs = useTrainingStore((s) => s.runs);
  const runsProjectId = useTrainingStore((s) => s.runsProjectId);

  if (runsProjectId !== projectId) return null;

  const total = runs.length;
  if (total === 0) {
    return (
      <div className="projects-current-training muted">
        {t("projects.training.noRuns")}
      </div>
    );
  }

  const running = runs.filter((r) => r.status === "running").length;
  const bestF1 = runs.reduce<number | null>((acc, r) => {
    const v = r.best_f1;
    if (typeof v !== "number" || Number.isNaN(v)) return acc;
    return acc === null || v > acc ? v : acc;
  }, null);
  const bestMIoU = runs.reduce<number | null>((acc, r) => {
    const v = r.best_miou;
    if (typeof v !== "number" || Number.isNaN(v)) return acc;
    return acc === null || v > acc ? v : acc;
  }, null);
  // runs are sorted newest-first by useGlobalPolling, so [0] is the last
  // training activity. Falls back across updated_at -> created_at.
  const latestIso = runs[0]?.updated_at || runs[0]?.created_at || null;

  const countLabel = running > 0
    ? t("projects.training.countWithRunning").replace("{n}", String(total)).replace("{running}", String(running))
    : t("projects.training.count").replace("{n}", String(total));

  const parts: string[] = [countLabel];
  if (latestIso) {
    parts.push(t("projects.training.last").replace("{rel}", _formatRel(latestIso)));
  }
  if (bestF1 !== null) {
    const miouPart = bestMIoU !== null ? ` / mIoU ${bestMIoU.toFixed(3)}` : "";
    parts.push(`${t("projects.training.bestF1")} ${bestF1.toFixed(3)}${miouPart}`);
  } else if (latestIso) {
    parts.push(t("projects.training.noF1Yet"));
  }

  return (
    <div className="projects-current-training" title={latestIso ?? undefined}>
      {parts.join(" · ")}
    </div>
  );
}

function _formatRel(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(diff) || diff < 0) return "—";
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default CurrentProjectTrainingSummary;
