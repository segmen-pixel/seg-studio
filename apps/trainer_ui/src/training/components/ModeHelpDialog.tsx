// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n, type TranslationKey } from "../../i18n";

type TrainingMode = "standard" | "quick" | "transfer" | "instance";

type ModeHelpDialogProps = {
  mode: TrainingMode;
  onClose: () => void;
};

type Step = {
  glyph: string;     // single emoji-style icon; works without an icon font
  titleKey: TranslationKey;
  descKey: TranslationKey;
};

const STEPS: Record<TrainingMode, Step[]> = {
  standard: [
    { glyph: "🖌", titleKey: "training.modeHelp.standard.step1.title", descKey: "training.modeHelp.standard.step1.desc" },
    { glyph: "🤖", titleKey: "training.modeHelp.standard.step2.title", descKey: "training.modeHelp.standard.step2.desc" },
    { glyph: "✨", titleKey: "training.modeHelp.standard.step3.title", descKey: "training.modeHelp.standard.step3.desc" },
  ],
  quick: [
    { glyph: "🏷", titleKey: "training.modeHelp.quick.step1.title", descKey: "training.modeHelp.quick.step1.desc" },
    { glyph: "⚡", titleKey: "training.modeHelp.quick.step2.title", descKey: "training.modeHelp.quick.step2.desc" },
    { glyph: "✨", titleKey: "training.modeHelp.quick.step3.title", descKey: "training.modeHelp.quick.step3.desc" },
  ],
  transfer: [
    { glyph: "🧠", titleKey: "training.modeHelp.transfer.step1.title", descKey: "training.modeHelp.transfer.step1.desc" },
    { glyph: "🔧", titleKey: "training.modeHelp.transfer.step2.title", descKey: "training.modeHelp.transfer.step2.desc" },
    { glyph: "🌟", titleKey: "training.modeHelp.transfer.step3.title", descKey: "training.modeHelp.transfer.step3.desc" },
  ],
  instance: [
    { glyph: "🖌", titleKey: "training.modeHelp.instance.step1.title", descKey: "training.modeHelp.instance.step1.desc" },
    { glyph: "🧩", titleKey: "training.modeHelp.instance.step2.title", descKey: "training.modeHelp.instance.step2.desc" },
    { glyph: "🔢", titleKey: "training.modeHelp.instance.step3.title", descKey: "training.modeHelp.instance.step3.desc" },
  ],
};

/**
 * Modal popup explaining each training mode.
 *
 * Previously rendered tightly-packed SVG diagrams whose 9px Japanese labels
 * crowded into each other ("セグメンテーション" + "自動マスク生成" rendered
 * as overlapping text on the standard mode card). Replaced with an HTML
 * 3-step flex layout so each step's title + description can wrap naturally
 * without colliding with its neighbours.
 */
export default React.memo(function ModeHelpDialog({ mode, onClose }: ModeHelpDialogProps) {
  const { t } = useI18n();
  const steps = STEPS[mode];
  const summaryKey = `training.modeHelp.${mode}.summary` as TranslationKey;

  return (
    <div className="training-mode-help-overlay" onClick={onClose}>
      <div className="training-mode-help-popup" onClick={(e) => e.stopPropagation()}>
        <button
          className="ghost"
          style={{ position: "absolute", top: 4, right: 8, fontSize: 16 }}
          onClick={onClose}
          aria-label="close"
        >
          ×
        </button>
        <h3 style={{ margin: "0 0 12px", fontSize: 15 }}>
          {t(`training.mode.${mode}` as TranslationKey)}
        </h3>
        <div className="training-mode-help-steps">
          {steps.map((step, i) => (
            <React.Fragment key={step.titleKey}>
              <div className="training-mode-help-step">
                <div className="training-mode-help-step-glyph" aria-hidden="true">{step.glyph}</div>
                <div className="training-mode-help-step-title">{t(step.titleKey)}</div>
                <div className="training-mode-help-step-desc">{t(step.descKey)}</div>
              </div>
              {i < steps.length - 1 && (
                <div className="training-mode-help-arrow" aria-hidden="true">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
        <p className="training-mode-help-summary">{t(summaryKey)}</p>
      </div>
    </div>
  );
});
