// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useState } from "react";
import type { TranslationKey } from "../i18n";
import { deleteReport, generateReport, listReports, type ReportGenerateRequest, type ReportListItem } from "../api/reports";

type ReportModalProps = {
  open: boolean;
  onClose: () => void;
  projectId: string;
  runId: string;
  runLabel: string;
  onGenerated?: (reportId: string, runId: string, label: string) => void;
  t: (key: TranslationKey) => string;
  lang: string;
};

export default function ReportModal({
  open,
  onClose,
  projectId,
  runId,
  runLabel,
  onGenerated,
  lang,
}: ReportModalProps) {
  const [reportType, setReportType] = useState<"model_eval" | "batch">("model_eval");
  const [includeLearningCurves, setIncludeLearningCurves] = useState(true);
  const [includeConfusionMatrix, setIncludeConfusionMatrix] = useState(true);
  const [includeThresholdAnalysis, setIncludeThresholdAnalysis] = useState(true);
  const [includeHardCases, setIncludeHardCases] = useState(true);
  const [includeInstanceRecall, setIncludeInstanceRecall] = useState(true);
  const [hardCaseTopN, setHardCaseTopN] = useState(10);
  // Batch reports: defect-count threshold in %; "" = run's optimal threshold
  const [confidenceThresholdPct, setConfidenceThresholdPct] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pastReports, setPastReports] = useState<ReportListItem[] | null>(null);

  // Load previously generated reports whenever the modal opens
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    listReports(projectId)
      .then((items) => { if (!cancelled) setPastReports(items); })
      .catch(() => { if (!cancelled) setPastReports([]); });
    return () => { cancelled = true; };
  }, [open, projectId]);

  if (!open) return null;

  const ja = lang === "ja";

  const handleOpenPast = (item: ReportListItem) => {
    onGenerated?.(item.report_id, item.run_id, item.run_id === runId ? runLabel : item.run_id.slice(0, 8));
    onClose();
  };

  const handleDeleteReport = async (reportId: string) => {
    if (!window.confirm(ja ? "このレポートを削除しますか？" : "Delete this report?")) return;
    try {
      await deleteReport(projectId, reportId);
      setPastReports((prev) => prev ? prev.filter((r) => r.report_id !== reportId) : prev);
    } catch (err: any) {
      setError(err?.displayMessage ?? err?.message ?? "Delete failed");
    }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    try {
      const payload: ReportGenerateRequest = {
        run_id: runId,
        report_type: reportType,
        formats: ["html"],
        lang,
        options: {
          include_learning_curves: includeLearningCurves,
          include_confusion_matrix: includeConfusionMatrix,
          include_threshold_analysis: includeThresholdAnalysis,
          include_hard_cases: includeHardCases,
          include_instance_recall: includeInstanceRecall,
          hard_case_top_n: hardCaseTopN,
          confidence_threshold: (() => {
            const v = parseFloat(confidenceThresholdPct);
            return Number.isFinite(v) ? Math.max(0, Math.min(100, v)) / 100 : null;
          })(),
        },
      };
      const res = await generateReport(projectId, payload);
      onGenerated?.(res.report_id, runId, runLabel);
      onClose();
    } catch (err: any) {
      setError(err?.displayMessage ?? err?.message ?? "Report generation failed");
    } finally {
      setGenerating(false);
    }
  };

  const sections: { key: string; label: string; checked: boolean; set: (v: boolean) => void }[] = [
    { key: "lc", label: ja ? "学習曲線" : "Learning Curves", checked: includeLearningCurves, set: setIncludeLearningCurves },
    { key: "cm", label: ja ? "混同行列" : "Confusion Matrix", checked: includeConfusionMatrix, set: setIncludeConfusionMatrix },
    { key: "th", label: ja ? "閾値・較正分析" : "Threshold & Calibration", checked: includeThresholdAnalysis, set: setIncludeThresholdAnalysis },
    { key: "hc", label: ja ? "Hard Case分析" : "Hard Case Analysis", checked: includeHardCases, set: setIncludeHardCases },
    { key: "ir", label: "Instance-level Recall", checked: includeInstanceRecall, set: setIncludeInstanceRecall },
  ];

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" style={{ maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{ja ? "レポート生成" : "Generate Report"}</h2>
          <button className="ghost" onClick={onClose}>&times;</button>
        </div>

        <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)" }}>
          <div className="muted" style={{ fontSize: "0.85rem" }}>Run: <code>{runLabel}</code></div>
        </div>

        <section style={{ padding: "16px 20px" }}>
          <div style={{ marginBottom: 14 }}>
            <label style={{ fontWeight: 600, fontSize: "0.9rem" }}>{ja ? "レポートタイプ" : "Report Type"}</label>
            <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
              <button className={reportType === "model_eval" ? "primary" : "ghost"} style={{ flex: 1, padding: "8px 12px", borderRadius: 6, border: "1px solid var(--border)" }} onClick={() => setReportType("model_eval")}>{ja ? "モデル評価" : "Model Evaluation"}</button>
              <button className={reportType === "batch" ? "primary" : "ghost"} style={{ flex: 1, padding: "8px 12px", borderRadius: 6, border: "1px solid var(--border)" }} onClick={() => setReportType("batch")}>{ja ? "検査バッチ" : "Batch Inspection"}</button>
            </div>
          </div>

          {reportType === "model_eval" && (
            <div style={{ marginBottom: 4 }}>
              <label style={{ fontWeight: 600, fontSize: "0.9rem" }}>{ja ? "含めるセクション" : "Sections to include"}</label>
              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 6 }}>
                {sections.map((sec) => (
                  <label key={sec.key} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                    <input type="checkbox" checked={sec.checked} onChange={(e) => sec.set(e.target.checked)} />
                    {sec.label}
                  </label>
                ))}
                {includeHardCases && (
                  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.85rem", marginLeft: 22 }}>
                    {ja ? "Hard Case表示数:" : "Hard cases to show:"}
                    <input type="number" min={1} max={50} value={hardCaseTopN} onChange={(e) => setHardCaseTopN(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))} style={{ width: 60, padding: "2px 6px" }} />
                  </label>
                )}
              </div>
            </div>
          )}

          {reportType === "batch" && (
            <div style={{ marginBottom: 4 }}>
              <label style={{ fontWeight: 600, fontSize: "0.9rem" }}>{ja ? "判定しきい値" : "Confidence threshold"}</label>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontSize: "0.85rem" }}>
                <input
                  type="number" min={0} max={100} step={1}
                  placeholder={ja ? "自動" : "auto"}
                  value={confidenceThresholdPct}
                  onChange={(e) => setConfidenceThresholdPct(e.target.value)}
                  style={{ width: 80, padding: "2px 6px" }}
                />
                <span>%</span>
                <span className="muted">{ja ? "空欄 = 学習時の最適しきい値" : "blank = run's optimal threshold"}</span>
              </div>
            </div>
          )}

          <div className="muted" style={{ fontSize: "0.8rem", marginTop: 14 }}>
            {ja
              ? "生成後、結果タブの隣にプレビュータブが開きます（ブラウザ印刷で PDF 保存可）。"
              : "Opens a preview tab next to the results tab. Use browser print to save as PDF."}
          </div>
        </section>

        {pastReports && pastReports.length > 0 && (
          <section style={{ padding: "0 20px 12px", borderTop: "1px solid var(--border)" }}>
            <label style={{ fontWeight: 600, fontSize: "0.9rem", display: "block", margin: "12px 0 6px" }}>
              {ja ? "過去のレポート" : "Past reports"} ({pastReports.length})
            </label>
            <div style={{ maxHeight: 160, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
              {pastReports.map((r) => (
                <div key={r.report_id} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.85rem" }}>
                  <span className="muted" style={{ whiteSpace: "nowrap" }}>{(r.created_at ?? "").slice(0, 16).replace("T", " ")}</span>
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {r.report_type === "model_eval" ? (ja ? "モデル評価" : "Model Eval") : (ja ? "検査バッチ" : "Batch")} · {r.run_id.slice(0, 8)}
                  </span>
                  <button className="ghost" style={{ fontSize: 12 }} onClick={() => handleOpenPast(r)}>{ja ? "開く" : "Open"}</button>
                  <button className="ghost" style={{ fontSize: 12, color: "var(--warning, #e57373)" }} onClick={() => handleDeleteReport(r.report_id)}>{ja ? "削除" : "Delete"}</button>
                </div>
              ))}
            </div>
          </section>
        )}

        {error && (
          <div style={{ padding: "8px 20px", color: "#dc2626", fontSize: "0.85rem" }}>{error}</div>
        )}

        <div style={{ padding: "12px 20px", borderTop: "1px solid var(--border)", display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button className="ghost" onClick={onClose} disabled={generating}>{ja ? "キャンセル" : "Cancel"}</button>
          <button className="primary" onClick={handleGenerate} disabled={generating} style={{ minWidth: 140 }}>
            {generating ? (ja ? "生成中..." : "Generating...") : (ja ? "生成してプレビュー" : "Generate & Preview")}
          </button>
        </div>
      </div>
    </div>
  );
}
