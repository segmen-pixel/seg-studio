// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import type { Project } from "../../api";
import type { TrainProgressInfo } from "../types";

function _fmtK(n: number): string {
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
  return String(n);
}

type StatusToastProps = {
  toastMsg: string;
  toastCopied: boolean;
  gpuBusy: boolean;
  inferStatus: string;
  trainProgress: TrainProgressInfo | null;
  trainProjectId: string | null;
  projects: Project[];
  onToastClick: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
};

export default React.memo(function StatusToast({
  toastMsg, toastCopied, gpuBusy, inferStatus,
  trainProgress, trainProjectId, projects,
  onToastClick, onMouseEnter, onMouseLeave,
}: StatusToastProps) {
  const { t } = useI18n();

  return (
    <div
      className={`tabs-status${/fail|error/i.test(toastMsg) ? " toast-error" : ""}${!toastMsg && (gpuBusy || inferStatus) ? " training-active" : ""}`}
      onClick={onToastClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{ cursor: toastMsg ? "pointer" : undefined }}
      title={toastMsg ? "Click to copy" : ""}
    >
      {toastCopied ? "✓ Copied" : toastMsg || (inferStatus
        ? <><span className="train-spinner" />{inferStatus}</>
        : gpuBusy && trainProgress
        ? <>
            <span className="train-spinner" />
            {`${projects.find(p => p.id === trainProjectId)?.name ?? t("tab.training")} — ${trainProgress.pct}% (${trainProgress.unit === "step" ? `Step ${_fmtK(trainProgress.epoch)}/${_fmtK(trainProgress.total_epochs)}` : `Epoch ${trainProgress.epoch}/${trainProgress.total_epochs}`})`}
          </>
        : gpuBusy ? <><span className="train-spinner" />{projects.find(p => p.id === trainProjectId)?.name ?? t("tab.training")}...</> : "")}
    </div>
  );
});
