// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import { useI18n } from "../../i18n";
import ModeHelpDialog from "./ModeHelpDialog";
import TrainingModeBadge from "./TrainingModeBadge";

import type { TaskMode as TrainingMode } from "../../app/types";

type ModeSelectButtonsProps = {
  trainingMode: TrainingMode | null;
  onTrainingModeChange?: (mode: TrainingMode) => void;
};

const DEFECT_MODES: readonly TrainingMode[] = ["standard"];

/**
 * Task selection buttons (what to inspect), with SVG illustrations + help.
 *
 * The top level is organized by task: defect detection and count inspection.
 * How a defect-detection model is trained (standard / quick / transfer) is a
 * strategy, not a task, and lives in the detailed-settings panel — all three
 * share one pipeline and the wire format is unchanged. Picking the defect
 * task selects the "standard" strategy until the panel says otherwise.
 */
export default React.memo(function ModeSelectButtons({
  trainingMode,
  onTrainingModeChange,
}: ModeSelectButtonsProps) {
  const { t } = useI18n();
  const [modeHelpOpen, setModeHelpOpen] = useState<TrainingMode | null>(null);
  const isDefect = trainingMode != null && DEFECT_MODES.includes(trainingMode);

  return (
    <div className="training-mode-block">
      <div className="training-mode-select">
        {/* Defect detection task: annotate → train → brain */}
        <div className="training-mode-btn-wrap">
        <span className="training-mode-help" onClick={(e) => { e.stopPropagation(); setModeHelpOpen(modeHelpOpen === "standard" ? null : "standard"); }}>?</span>
        <button
          className={`training-mode-btn ${isDefect ? "active" : ""}`}
          onClick={() => { if (!isDefect) onTrainingModeChange?.("standard"); }}
          type="button"
          data-mode="standard"
          data-desc={t("training.task.defect.btn.desc")}
        >
          <svg className="training-mode-illustration" viewBox="0 0 140 50" fill="none" aria-hidden="true">
            <defs>
              <linearGradient id="std-bg1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#64b5f6" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#42a5f5" stopOpacity="0.08" />
              </linearGradient>
              <linearGradient id="std-bg2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#66bb6a" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#43a047" stopOpacity="0.08" />
              </linearGradient>
              <linearGradient id="std-bg3" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#d55e00" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#e91e63" stopOpacity="0.15" />
              </linearGradient>
            </defs>
            {/* Step 1: Annotation */}
            <rect x="6" y="4" width="28" height="24" rx="6" fill="url(#std-bg1)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.25" />
            <path d="M12 11h12M12 15h16M12 19h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.35" />
            <rect x="22" y="9" width="8" height="5" rx="1.5" fill="#42a5f5" opacity="0.5" />
            <circle cx="26" cy="22" r="2.5" fill="#42a5f5" opacity="0.6" />
            {/* Arrow 1 */}
            <path d="M39 16h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M47 13l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            {/* Step 2: Training — Robot */}
            <line x1="70" y1="3" x2="70" y2="7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.4" />
            <circle cx="70" cy="2.5" r="1.5" fill="#66bb6a" opacity="0.5" />
            <rect x="60" y="7" width="20" height="16" rx="4" fill="url(#std-bg2)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.3" />
            <rect x="63" y="11" width="4" height="3" rx="1" fill="#66bb6a" opacity="0.6" />
            <rect x="73" y="11" width="4" height="3" rx="1" fill="#66bb6a" opacity="0.6" />
            <path d="M65 18h10M66 20h8" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.3" />
            <rect x="57" y="11" width="2" height="6" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            <rect x="81" y="11" width="2" height="6" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            {/* Arrow 2 */}
            <path d="M89 16h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M97 13l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            {/* Step 3: Defect detection result */}
            <rect x="104" y="4" width="28" height="24" rx="6" fill="url(#std-bg3)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.2" />
            <rect x="107" y="7" width="22" height="14" rx="2" fill="currentColor" fillOpacity="0.06" />
            <rect x="110" y="9" width="8" height="5" rx="1.5" fill="#e91e63" opacity="0.45" />
            <rect x="113" y="16" width="6" height="3" rx="1" fill="#d55e00" opacity="0.4" />
            <circle cx="124" cy="12" r="3" fill="#e91e63" opacity="0.35" />
            <rect x="109" y="8" width="10" height="7" rx="2" fill="none" stroke="#e91e63" strokeWidth="0.8" strokeOpacity="0.5" strokeDasharray="1.5 1.5" />
            {/* Labels */}
            <text x="20" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.annotate")}</text>
            <text x="70" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.train")}</text>
            <text x="118" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.detect")}</text>
          </svg>
          <span><TrainingModeBadge mode={isDefect && trainingMode ? trainingMode : "standard"} /> {t("training.task.defect")}</span>
        </button>
        </div>

        {/* Count inspection task: masks → synthesized instances → numbered count */}
        <div className="training-mode-btn-wrap">
        <span className="training-mode-help" onClick={(e) => { e.stopPropagation(); setModeHelpOpen(modeHelpOpen === "instance" ? null : "instance"); }}>?</span>
        <button
          className={`training-mode-btn ${trainingMode === "instance" ? "active" : ""}`}
          onClick={() => onTrainingModeChange?.("instance")}
          type="button"
          data-mode="instance"
          data-desc={t("training.mode.instance.btn.desc")}
        >
          <svg className="training-mode-illustration" viewBox="0 0 140 50" fill="none" aria-hidden="true">
            <defs>
              <linearGradient id="in-bg1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#64b5f6" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#42a5f5" stopOpacity="0.08" />
              </linearGradient>
              <linearGradient id="in-bg2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#009e73" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#009e73" stopOpacity="0.08" />
              </linearGradient>
            </defs>
            {/* Step 1: Semantic masks (same paint as standard) */}
            <rect x="6" y="4" width="28" height="24" rx="6" fill="url(#in-bg1)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.25" />
            <ellipse cx="14" cy="12" rx="4.5" ry="2.5" fill="#42a5f5" opacity="0.55" transform="rotate(-24 14 12)" />
            <ellipse cx="25" cy="15" rx="4.5" ry="2.5" fill="#42a5f5" opacity="0.55" transform="rotate(18 25 15)" />
            <ellipse cx="17" cy="22" rx="4.5" ry="2.5" fill="#42a5f5" opacity="0.55" transform="rotate(-6 17 22)" />
            {/* Arrow 1 */}
            <path d="M39 16h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M47 13l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            {/* Step 2: Training — Robot (bluish-green accents) */}
            <line x1="70" y1="3" x2="70" y2="7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.4" />
            <circle cx="70" cy="2.5" r="1.5" fill="#009e73" opacity="0.6" />
            <rect x="60" y="7" width="20" height="16" rx="4" fill="url(#in-bg2)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.3" />
            <rect x="63" y="11" width="4" height="3" rx="1" fill="#009e73" opacity="0.7" />
            <rect x="73" y="11" width="4" height="3" rx="1" fill="#009e73" opacity="0.7" />
            <path d="M65 18h10M66 20h8" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.3" />
            <rect x="57" y="11" width="2" height="6" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            <rect x="81" y="11" width="2" height="6" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            {/* Arrow 2 */}
            <path d="M89 16h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M97 13l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            {/* Step 3: Numbered instances */}
            <rect x="104" y="4" width="28" height="24" rx="6" fill="url(#in-bg2)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.2" />
            <ellipse cx="112" cy="13" rx="4.5" ry="2.5" fill="none" stroke="#d55e00" strokeWidth="1.1" transform="rotate(-20 112 13)" />
            <ellipse cx="124" cy="12" rx="4.5" ry="2.5" fill="none" stroke="#56b4e9" strokeWidth="1.1" transform="rotate(14 124 12)" />
            <ellipse cx="118" cy="22" rx="4.5" ry="2.5" fill="none" stroke="#009e73" strokeWidth="1.1" transform="rotate(-4 118 22)" />
            <text x="112" y="15" textAnchor="middle" fontSize="6" fontFamily="sans-serif" fontWeight="bold" fill="#d55e00">1</text>
            <text x="124" y="14" textAnchor="middle" fontSize="6" fontFamily="sans-serif" fontWeight="bold" fill="#56b4e9">2</text>
            <text x="118" y="24" textAnchor="middle" fontSize="6" fontFamily="sans-serif" fontWeight="bold" fill="#009e73">3</text>
            {/* Labels */}
            <text x="20" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.annotate")}</text>
            <text x="70" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.train")}</text>
            <text x="118" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.count")}</text>
          </svg>
          <span><TrainingModeBadge mode="instance" /> {t("training.task.instance")}</span>
        </button>
        </div>
      </div>

      {/* Mode help popup */}
      {modeHelpOpen && (
        <ModeHelpDialog mode={modeHelpOpen} onClose={() => setModeHelpOpen(null)} />
      )}
    </div>
  );
});
