// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

export default function MonitorState({
  title,
  copy,
  tone = "neutral",
  progress,
  actionLabel,
  onAction,
}: {
  title: string;
  copy: string;
  tone?: "neutral" | "loading" | "error";
  progress?: number;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="state-card" data-tone={tone}>
      <div className="state-card-title">{title}</div>
      <div className="state-card-copy">{copy}</div>
      {typeof progress === "number" && (
        <div className="state-progress" aria-hidden="true">
          <div className="state-progress-fill" style={{ width: `${progress}%` }} />
        </div>
      )}
      {actionLabel && onAction && (
        <div className="state-card-actions">
          <button className="ghost" onClick={onAction}>{actionLabel}</button>
        </div>
      )}
    </div>
  );
}
