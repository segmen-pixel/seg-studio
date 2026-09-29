// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useMemo, useRef } from "react";
import { useI18n } from "../i18n";
import { getReportFileUrl } from "../api/reports";

export default React.memo(function ReportViewer({
  projectId,
  reportId,
  runLabel,
}: {
  projectId: string | null;
  reportId: string;
  runLabel: string;
  active?: boolean;
}) {
  const { lang } = useI18n();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const ja = lang === "ja";
  // Cache-buster: ensures a freshly (re)generated report is fetched, not a stale cache
  // (also avoids stale X-Frame-Options responses cached before the SAMEORIGIN fix).
  const cacheBust = useMemo(() => `${reportId}-${Date.now()}`, [reportId]);
  const url = projectId ? `${getReportFileUrl(projectId, reportId, "report.html")}?t=${cacheBust}` : "";

  // The report is a single self-contained file -- charts and images are all
  // data: URIs -- so saving it is enough to keep or send the whole thing.
  // The .html route deliberately serves without Content-Disposition so the
  // iframe above can render it, which leaves the download attribute as the
  // way to save rather than navigate. API_BASE is window.location.origin,
  // so this is always same-origin and the attribute is always honoured.
  const downloadName = useMemo(() => {
    // Keep the run label readable (it is routinely Japanese) and drop only
    // what a filesystem actually rejects.
    const safe = (runLabel || "").replace(/[\\/:*?"<>|]/g, "_").trim().slice(0, 60);
    return safe ? `report_${safe}_${reportId}.html` : `report_${reportId}.html`;
  }, [runLabel, reportId]);

  const handlePrint = () => {
    const win = iframeRef.current?.contentWindow;
    if (win) {
      win.focus();
      win.print();
    }
  };

  if (!projectId) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
        <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>{ja ? "レポート" : "Report"}: {runLabel}</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
          <button className="ghost results-header-btn" onClick={handlePrint} title={ja ? "印刷 / PDF保存" : "Print / Save as PDF"}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
            {ja ? "印刷" : "Print"}
          </button>
          <a className="ghost results-header-btn" href={url} download={downloadName} title={ja ? "HTML ファイルとして保存 (画像を含む単一ファイル)" : "Save as an HTML file (single file, images included)"} style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 6 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            {ja ? "保存" : "Save"}
          </a>
          <a className="ghost results-header-btn" href={url} target="_blank" rel="noopener noreferrer" title={ja ? "ブラウザの別タブで開く" : "Open in a browser tab"} style={{ textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 6 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
            {ja ? "別タブ" : "Open"}
          </a>
        </div>
      </div>
      <iframe
        ref={iframeRef}
        src={url}
        title={`report-${reportId}`}
        data-testid="report-iframe"
        style={{ flex: 1, width: "100%", border: 0, background: "#fff", minHeight: 0 }}
      />
    </div>
  );
});
