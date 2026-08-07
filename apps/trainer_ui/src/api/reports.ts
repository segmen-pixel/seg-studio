// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

export type ReportFileInfo = {
  filename: string;
  format: string;
  size_bytes: number;
};

export type ReportGenerateResponse = {
  report_id: string;
  report_type: string;
  files: ReportFileInfo[];
  status: string;
  created_at: string;
};

export type ReportListItem = {
  report_id: string;
  report_type: string;
  run_id: string;
  files: ReportFileInfo[];
  created_at: string;
};

export type ReportOptions = {
  hard_case_top_n?: number;
  include_instance_recall?: boolean;
  include_hard_cases?: boolean;
  include_learning_curves?: boolean;
  include_confusion_matrix?: boolean;
  include_threshold_analysis?: boolean;
  confidence_threshold?: number | null;
};

export type ReportGenerateRequest = {
  run_id: string;
  report_type: "model_eval" | "batch";
  formats: string[];
  lang?: string;
  options?: ReportOptions;
};

export async function generateReport(
  projectId: string,
  payload: ReportGenerateRequest,
): Promise<ReportGenerateResponse> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/reports/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function listReports(projectId: string): Promise<ReportListItem[]> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/reports`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export function getReportFileUrl(
  projectId: string,
  reportId: string,
  filename: string,
): string {
  return `${API_BASE}/projects/${projectId}/reports/${reportId}/${filename}`;
}

export async function deleteReport(projectId: string, reportId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/reports/${reportId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw await parseApiError(res);
}
