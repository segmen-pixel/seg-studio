// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import { useI18n } from "../../i18n";
import ModeHelpDialog from "./ModeHelpDialog";
import TrainingModeBadge from "./TrainingModeBadge";

type TrainingMode = "standard" | "quick" | "transfer";

type ModeSelectButtonsProps = {
  trainingMode: TrainingMode | null;
  onTrainingModeChange?: (mode: TrainingMode) => void;
  libraryStats?: { total_profiles: number; total_projects: number } | null;
};

/** Training mode selection buttons with SVG illustrations + help dialog. */
export default React.memo(function ModeSelectButtons({
  trainingMode,
  onTrainingModeChange,
  libraryStats,
}: ModeSelectButtonsProps) {
  const { t } = useI18n();
  const [modeHelpOpen, setModeHelpOpen] = useState<TrainingMode | null>(null);

  return (
    <>
      <div className="training-mode-select">
        {/* Standard: annotate → train → brain */}
        <div className="training-mode-btn-wrap">
        <span className="training-mode-help" onClick={(e) => { e.stopPropagation(); setModeHelpOpen(modeHelpOpen === "standard" ? null : "standard"); }}>?</span>
        <button
          className={`training-mode-btn ${trainingMode === "standard" ? "active" : ""}`}
          onClick={() => onTrainingModeChange?.("standard")}
          type="button"
          data-desc={t("training.mode.standard.btn.desc")}
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
          <span><TrainingModeBadge mode="standard" /> {t("training.mode.standard")}</span>
        </button>
        </div>
        {/* Quick: labeled → AI → mask */}
        <div className="training-mode-btn-wrap">
        <span className="training-mode-help" onClick={(e) => { e.stopPropagation(); setModeHelpOpen(modeHelpOpen === "quick" ? null : "quick"); }}>?</span>
        <button
          className={`training-mode-btn ${trainingMode === "quick" ? "active" : ""}`}
          onClick={() => onTrainingModeChange?.("quick")}
          type="button"
          data-desc={t("training.mode.quick.btn.desc")}
        >
          <svg className="training-mode-illustration" viewBox="0 0 140 50" fill="none" aria-hidden="true">
            <defs>
              <linearGradient id="qk-bg1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#64b5f6" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#42a5f5" stopOpacity="0.05" />
              </linearGradient>
              <linearGradient id="qk-bg2" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#ff9800" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#f57c00" stopOpacity="0.1" />
              </linearGradient>
            </defs>
            {/* Left: stack of images */}
            <rect x="6" y="2" width="22" height="16" rx="3" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.15" />
            <rect x="10" y="6" width="22" height="16" rx="3" fill="currentColor" fillOpacity="0.08" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.18" />
            <rect x="14" y="10" width="22" height="16" rx="3" fill="url(#qk-bg1)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.3" />
            <rect x="17" y="13" width="10" height="5" rx="1.5" fill="#42a5f5" opacity="0.5" />
            <rect x="19" y="20" width="6" height="3" rx="1" fill="#66bb6a" opacity="0.45" />
            <circle cx="33" cy="12" r="3.5" fill="#42a5f5" opacity="0.6" />
            <path d="M31.5 12l1 1 2-2" stroke="#fff" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
            {/* Arrow */}
            <path d="M42 18h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M50 15l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            {/* Center: Robot */}
            <line x1="70" y1="3" x2="70" y2="7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.4" />
            <circle cx="70" cy="2.5" r="1.5" fill="#ff9800" opacity="0.5" />
            <rect x="60" y="7" width="20" height="16" rx="4" fill="url(#qk-bg2)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.3" />
            <rect x="63" y="11" width="4" height="3" rx="1" fill="#ff9800" opacity="0.6" />
            <rect x="73" y="11" width="4" height="3" rx="1" fill="#ff9800" opacity="0.6" />
            <path d="M65 18h10M66 20h8" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.3" />
            <rect x="57" y="11" width="2" height="6" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            <rect x="81" y="11" width="2" height="6" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            {/* Arrow */}
            <path d="M88 18h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M96 15l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            {/* Right: Output mask */}
            <rect x="104" y="8" width="22" height="18" rx="4" fill="url(#qk-bg1)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.3" />
            <rect x="108" y="12" width="14" height="6" rx="1.5" fill="#42a5f5" opacity="0.5" />
            <rect x="110" y="20" width="8" height="3" rx="1" fill="#66bb6a" opacity="0.5" />
            <path d="M130 8l1 2.5 2.5 1-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z" fill="#fdd835" opacity="0.55" />
            {/* Labels */}
            <text x="25" y="42" textAnchor="middle" fontSize="7" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.labeled")}</text>
            <text x="70" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.train")}</text>
            <text x="115" y="42" textAnchor="middle" fontSize="7.5" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.mask")}</text>
          </svg>
          <span><TrainingModeBadge mode="quick" /> {t("training.mode.quick")}</span>
        </button>
        </div>
        {/* Transfer: brain → arrow → brain with sparkle */}
        <div className="training-mode-btn-wrap">
        <span className="training-mode-help" onClick={(e) => { e.stopPropagation(); setModeHelpOpen(modeHelpOpen === "transfer" ? null : "transfer"); }}>?</span>
        <button
          className={`training-mode-btn ${trainingMode === "transfer" ? "active" : ""}`}
          onClick={() => onTrainingModeChange?.("transfer")}
          type="button"
          data-desc={t("training.mode.transfer.btn.desc")}
        >
          <svg className="training-mode-illustration" viewBox="0 0 140 50" fill="none" aria-hidden="true">
            <defs>
              <linearGradient id="tf-bg1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#78909c" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#546e7a" stopOpacity="0.06" />
              </linearGradient>
              <linearGradient id="tf-bg2" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#d55e00" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#e91e63" stopOpacity="0.12" />
              </linearGradient>
            </defs>
            {/* Left: Model database */}
            <rect x="22" y="2" width="30" height="28" rx="5" fill="url(#tf-bg1)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.25" />
            <circle cx="31" cy="10" r="4" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.15" />
            <circle cx="43" cy="10" r="4" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.15" />
            <circle cx="31" cy="22" r="4" fill="currentColor" fillOpacity="0.06" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.15" />
            <circle cx="43" cy="22" r="4.5" fill="#d55e00" fillOpacity="0.2" stroke="#d55e00" strokeWidth="1" strokeOpacity="0.5" />
            <path d="M42 20.5v3M41 22h4" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" opacity="0.5" />
            <circle cx="49" cy="6" r="3" fill="#42a5f5" opacity="0.5" />
            <path d="M51 8l2 2" stroke="#fff" strokeWidth="1" strokeLinecap="round" />
            {/* Arrow */}
            <path d="M58 16h14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" opacity="0.3" />
            <path d="M70 13l4 3-4 3" fill="currentColor" fillOpacity="0.4" stroke="none" />
            <path d="M48 22h10" stroke="currentColor" strokeWidth="0.8" strokeLinecap="round" strokeDasharray="2 2" opacity="0.25" />
            {/* Right: Robot */}
            <line x1="94" y1="1" x2="94" y2="5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.4" />
            <circle cx="94" cy="0.5" r="1.5" fill="#d55e00" opacity="0.6" />
            <rect x="82" y="5" width="24" height="20" rx="5" fill="url(#tf-bg2)" stroke="currentColor" strokeWidth="1.2" strokeOpacity="0.3" />
            <rect x="86" y="10" width="5" height="4" rx="1.5" fill="#d55e00" opacity="0.55" />
            <rect x="97" y="10" width="5" height="4" rx="1.5" fill="#d55e00" opacity="0.55" />
            <path d="M88 18h12M89 21h10" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.3" />
            <rect x="79" y="11" width="2" height="7" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            <rect x="107" y="11" width="2" height="7" rx="1" fill="currentColor" fillOpacity="0.15" stroke="currentColor" strokeWidth="0.8" strokeOpacity="0.2" />
            {/* Sparkles */}
            <path d="M112 4l1 2.5 2.5 1-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z" fill="#fdd835" opacity="0.6" />
            <path d="M116 16l.7 1.5 1.5.6-1.5.6-.7 1.5-.7-1.5-1.5-.6 1.5-.6z" fill="#fdd835" opacity="0.4" />
            {/* Labels */}
            <text x="37" y="42" textAnchor="middle" fontSize="7" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.modelDb")}</text>
            <text x="94" y="42" textAnchor="middle" fontSize="7" fontFamily="sans-serif" fill="currentColor" opacity="0.7">{t("training.illust.bestModel")}</text>
          </svg>
          <span><TrainingModeBadge mode="transfer" /> {t("training.mode.transfer")}{libraryStats ? ` (${libraryStats.total_profiles})` : ""}</span>
        </button>
        </div>
      </div>

      {/* Mode help popup */}
      {modeHelpOpen && (
        <ModeHelpDialog mode={modeHelpOpen} onClose={() => setModeHelpOpen(null)} />
      )}
    </>
  );
});
