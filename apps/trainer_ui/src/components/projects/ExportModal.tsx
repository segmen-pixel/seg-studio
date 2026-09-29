// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import {
  exportDatasetUrl,
  fetchFgAnalysis,
  fetchProjectsSummary,
  resizeCloneProject,
  type FgAnalysisResult,
  type Project,
  type ProjectSummary,
} from "../../api";
import { useI18n } from "../../i18n";

type ExportModalProps = {
  project: Project;
  onClose: () => void;
  exportBusy: boolean;
  setExportBusy: (busy: boolean) => void;
  applyProjectsSummary: (summaries: ProjectSummary[]) => void;
  showToast: (msg: string) => void;
};

// Extracted verbatim from ProjectsPanel.tsx (pre-OSS refactor): the per-project
// export dialog — original-size ZIP download or FG-analysis-guided resize clone.
// The scale / analysis state lives here: the dialog unmounts on close, which
// matches the old open-time reset (scale=100, result=null) exactly.
const ExportModal: React.FC<ExportModalProps> = ({
  project,
  onClose,
  exportBusy,
  setExportBusy,
  applyProjectsSummary,
  showToast,
}) => {
  const { t } = useI18n();
  const [exportFgLoading, setExportFgLoading] = useState(false);
  const [exportFgResult, setExportFgResult] = useState<FgAnalysisResult | null>(null);
  const [exportScale, setExportScale] = useState(100);

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel export-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 420, minHeight: "auto" }}>
        <div className="settings-header">
          <h2>{t("projects.exportModal.title")}</h2>
          <button className="settings-close-btn" onClick={onClose}>&times;</button>
        </div>
        <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              className={`models-action-btn${exportScale === 100 && !exportFgResult ? " active" : ""}`}
              style={{ flex: 1, padding: "8px 0", borderRadius: 6, fontWeight: exportScale === 100 && !exportFgResult ? 600 : 400, background: exportScale === 100 && !exportFgResult ? "var(--accent)" : "var(--panel-secondary, var(--panel))", color: exportScale === 100 && !exportFgResult ? "#fff" : "var(--text)" }}
              onClick={() => { setExportScale(100); setExportFgResult(null); }}
            >
              {t("projects.exportModal.originalSize")}
            </button>
            <button
              className="models-action-btn"
              style={{ flex: 1, padding: "8px 0", borderRadius: 6, fontWeight: exportFgResult ? 600 : 400, background: exportFgResult ? "var(--accent)" : "var(--panel-secondary, var(--panel))", color: exportFgResult ? "#fff" : "var(--text)" }}
              disabled={exportFgLoading}
              onClick={async () => {
                if (exportFgResult) return;
                setExportFgLoading(true);
                try {
                  const result = await fetchFgAnalysis(project.id);
                  setExportFgResult(result);
                  setExportScale(Math.round(result.recommended_scale * 100));
                } catch (err) {
                  showToast(t("projects.exportModal.analyzeFailed").replace("{msg}", (err as Error).message));
                } finally {
                  setExportFgLoading(false);
                }
              }}
            >
              {exportFgLoading ? t("projects.exportModal.analyzing") : t("projects.exportModal.shrink")}
            </button>
          </div>

          {exportFgResult && (() => {
            const fg = exportFgResult;
            const [mw, mh] = fg.mean_image_size;
            return (
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                <div style={{ marginBottom: 8, color: "var(--text-secondary)" }}>
                  {t("projects.exportModal.fgSummary").replace("{n}", String(fg.num_components)).replace("{px}", String(fg.p25_fg_area_px))}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {fg.scale_details.map((d) => {
                    const pct = Math.round(d.scale * 100);
                    const selected = exportScale === pct;
                    const recommended = pct === Math.round(fg.recommended_scale * 100);
                    return (
                      <label
                        key={pct}
                        style={{
                          display: "flex", alignItems: "center", gap: 8,
                          padding: "5px 8px", borderRadius: 5, cursor: "pointer",
                          background: selected ? "var(--accent-bg, rgba(59,130,246,0.1))" : "transparent",
                          border: selected ? "1px solid var(--accent)" : "1px solid transparent",
                        }}
                      >
                        <input
                          type="radio" name="export-scale" value={pct}
                          checked={selected}
                          onChange={() => setExportScale(pct)}
                          style={{ accentColor: "var(--accent)" }}
                        />
                        <span style={{ fontWeight: recommended ? 600 : 400 }}>
                          {pct}%
                          {recommended && ` ${t("projects.exportModal.recommended")}`}
                        </span>
                        <span style={{ color: "var(--text-secondary)", marginLeft: "auto", fontSize: 12 }}>
                          {Math.round(mw * d.scale)}x{Math.round(mh * d.scale)}
                        </span>
                        <span style={{ fontSize: 12, color: d.safe ? "var(--success, #22c55e)" : "var(--danger, #ef4444)" }}>
                          {d.safe ? "OK" : "NG"}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })()}

          <button
            style={{
              padding: "10px 0", borderRadius: 6, border: "none", cursor: "pointer",
              background: "var(--accent)", color: "#fff", fontWeight: 600, fontSize: 14,
            }}
            disabled={exportBusy}
            onClick={async () => {
              const proj = project;
              onClose();
              setExportBusy(true);
              try {
                if (exportScale < 100) {
                  // Resize clone: create new project directly
                  showToast(t("projects.exportModal.resizing").replace("{pct}", String(exportScale)));
                  const result = await resizeCloneProject(proj.id, exportScale / 100);
                  showToast(t("projects.exportModal.resizeDone").replace("{name}", result.name).replace("{count}", String(result.image_count)));
                  const summaries = await fetchProjectsSummary();
                  applyProjectsSummary(summaries);
                } else {
                  // Original size: ZIP download
                  showToast(t("projects.exporting"));
                  const res = await fetch(exportDatasetUrl(proj.id));
                  if (!res.ok) throw new Error(`${res.status}`);
                  const blob = await res.blob();
                  const cd = res.headers.get("content-disposition") || "";
                  const match = cd.match(/filename="?([^";]+)"?/);
                  const filename = match?.[1] || `${proj.name || proj.id}.zip`;
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url; a.download = filename; a.click();
                  URL.revokeObjectURL(url);
                  showToast(t("projects.exportDone"));
                }
              } catch (err) {
                showToast(`${t("projects.exportFailed")}: ${(err as Error).message}`);
              } finally {
                setExportBusy(false);
              }
            }}
          >
            {exportScale < 100 ? t("projects.exportModal.createResized").replace("{pct}", String(exportScale)) : t("projects.exportModal.zipExport")}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExportModal;
