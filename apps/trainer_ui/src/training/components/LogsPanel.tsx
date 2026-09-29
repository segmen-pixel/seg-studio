// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import MonitorState from "./MonitorState";
import type { TrainRunItem } from "../../utils";
import type { TranslationKey } from "../../i18n";

export type LogsPanelProps = {
  selectedRunIdForLogs: string | null;
  selectedRun: TrainRunItem | null;
  logStream: {
    runLogsText: string;
    runLogsError: string;
    isRunLogsLoading: boolean;
    logPreRef: React.Ref<HTMLPreElement>;
  };
  lang: string;
  t: (key: TranslationKey) => string;
};

// Extracted verbatim from Training.tsx (pre-OSS refactor): the "logs" monitor
// panel, the phase progress bar, and the streaming log pane.
export default function LogsPanel({
  selectedRunIdForLogs,
  selectedRun,
  logStream,
  lang,
  t,
}: LogsPanelProps) {
  return (
    <div className="train-log-panel">
      {!selectedRunIdForLogs ? (
        <MonitorState
          title={t("training.selectRun")}
          copy={t("training.selectRun")}
        />
      ) : logStream.isRunLogsLoading ? (
        <MonitorState
          title={t("training.logs.loading")}
          copy={t("training.logs.loading.copy")}
          tone="loading"
        />
      ) : logStream.runLogsError ? (
        <MonitorState
          title={t("training.logs.unavailable")}
          copy={logStream.runLogsError}
          tone="error"
        />
      ) : !logStream.runLogsText ? (() => {
        return (
          <MonitorState
            title={
              selectedRunIdForLogs?.startsWith("__preparing_")
                ? (lang === "ja" ? "学習準備中" : "Preparing")
                : selectedRun?.status && ["completed", "failed", "stopped"].includes(selectedRun.status)
                  ? t("training.logs.noSaved")
                  : (lang === "ja" ? "ログ待機中" : "Waiting for the first log line")
            }
            copy={
              selectedRunIdForLogs?.startsWith("__preparing_")
                ? (selectedRun?.model_name || (lang === "ja" ? "データセット準備・GPU確保・学習設定を行っています..." : "Preparing dataset, claiming GPU, configuring training..."))
                : selectedRun?.status && ["completed", "failed", "stopped"].includes(selectedRun.status)
                  ? t("training.logs.noSaved.copy")
                  : (lang === "ja" ? "学習は開始されましたが、まだログが出力されていません。" : "Training has started, but the backend has not emitted a log line yet.")
            }
            tone={selectedRunIdForLogs?.startsWith("__preparing_") ? "loading" : undefined}
          />
        );
      })() : (
        <>
          {/* Phase progress bar for training preparation */}
          {(() => {
            const phaseMatch = logStream.runLogsText.match(/\[PHASE (\d+)\/(\d+)\]/g);
            if (!phaseMatch || logStream.runLogsText.includes("Epoch ")) return null;
            const last = phaseMatch[phaseMatch.length - 1];
            const m = last.match(/\[PHASE (\d+)\/(\d+)\]/);
            if (!m) return null;
            const current = parseInt(m[1], 10);
            const total = parseInt(m[2], 10);
            const pct = Math.round((current / total) * 100);
            const labels = ["", t("training.phases.seg1"), t("training.phases.seg2"), t("training.phases.seg3"), t("training.phases.seg4"), t("training.phases.seg5"), t("training.phases.seg6")];
            const label = labels[current] || `Phase ${current}`;
            return (
              <div style={{ padding: "6px 8px", background: "var(--bg-secondary, #1e1e2e)", borderBottom: "1px solid var(--border, #333)", fontSize: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span>{lang === "ja" ? "学習準備中" : "Preparing"}: {label}</span>
                  <span>{current}/{total}</span>
                </div>
                <div style={{ height: 4, background: "var(--bg-tertiary, #2a2a3e)", borderRadius: 2, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${pct}%`, background: "#f0a040", borderRadius: 2, transition: "width 0.3s" }} />
                </div>
              </div>
            );
          })()}
          <pre ref={logStream.logPreRef} className="train-log-pre">{logStream.runLogsText}</pre>
        </>
      )}
    </div>
  );
}
