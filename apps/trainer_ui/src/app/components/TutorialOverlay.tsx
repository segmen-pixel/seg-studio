// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useLayoutEffect, useState } from "react";
import type { TutorialMode, TutorialStep } from "../tutorialSteps";

const PAD = 8;  // space between highlighted element and spotlight border
const GAP = 14; // space between spotlight and tooltip
const TOOLTIP_WIDTH = 480;         // spotlight tooltip (with target)
const TOOLTIP_CENTER_WIDTH = 640;  // centered modal (welcome / text-only steps)

type Rect = { top: number; left: number; width: number; height: number };

type Props = {
  step: TutorialStep;
  stepIndex: number;
  totalSteps: number;
  lang: "ja" | "en";
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
  onChooseMode: (mode: TutorialMode) => void;
};

export default function TutorialOverlay({
  step, stepIndex, totalSteps, lang,
  onNext, onBack, onSkip, onChooseMode,
}: Props) {
  const [targetRect, setTargetRect] = useState<Rect | null>(null);

  // Track the target element across resizes / scrolls / tab switches.
  useLayoutEffect(() => {
    if (!step.targetSelector) { setTargetRect(null); return; }

    let raf = 0;
    const measure = () => {
      const el = document.querySelector(step.targetSelector!) as HTMLElement | null;
      if (!el) { setTargetRect(null); return; }
      const r = el.getBoundingClientRect();
      setTargetRect({
        top: r.top - PAD,
        left: r.left - PAD,
        width: r.width + PAD * 2,
        height: r.height + PAD * 2,
      });
    };
    measure();

    // Re-measure on resize/scroll and repeatedly for a short window (handles tab-switch animations).
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    const start = Date.now();
    const tick = () => {
      measure();
      if (Date.now() - start < 700) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [step.targetSelector, stepIndex]);

  const tooltip = useTooltipPosition(step, targetRect);

  // Keyboard shortcuts — Enter=next, ←/→=back/next, Esc=skip, 1/2/3=mode on welcome.
  //
  // Arrow keys and Esc must work even when an input on the host page has
  // focus — the Training tab has many form inputs, so opening the tutorial
  // there used to leave the user stuck because ← / → were being eaten by
  // the focused input. Enter still defers to the input so users can finish
  // typing without accidentally advancing the tutorial.
  useEffect(() => {
    const isTypingTarget = (el: HTMLElement | null) =>
      !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (step.isModeSelect) {
        if (e.key === "1") { e.preventDefault(); onChooseMode("beginner"); }
        else if (e.key === "2") { e.preventDefault(); onChooseMode("intermediate"); }
        else if (e.key === "3") { e.preventDefault(); onChooseMode("expert"); }
        else if (e.key === "Escape") { e.preventDefault(); onSkip(); }
        return;
      }
      // Tutorial-priority keys: always handled, regardless of focused input.
      if (e.key === "ArrowRight") { e.preventDefault(); onNext(); return; }
      if (e.key === "ArrowLeft") { if (stepIndex > 0) { e.preventDefault(); onBack(); } return; }
      if (e.key === "Escape") { e.preventDefault(); onSkip(); return; }
      // Enter still yields to inputs so the user can confirm a value being typed.
      if (isTypingTarget(target)) return;
      if (e.key === "Enter") { e.preventDefault(); onNext(); }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [step, stepIndex, onNext, onBack, onSkip, onChooseMode]);

  const title = lang === "ja" ? step.titleJa : step.titleEn;
  const body = lang === "ja" ? step.bodyJa : step.bodyEn;
  const nextLabel = stepIndex === totalSteps - 1
    ? (lang === "ja" ? "完了" : "Finish")
    : (lang === "ja" ? "次へ" : "Next");
  const backLabel = lang === "ja" ? "戻る" : "Back";
  const skipLabel = lang === "ja" ? "スキップ" : "Skip";

  return (
    <div className="tutorial-root" role="dialog" aria-modal="true" aria-label="Tutorial">
      {/* Backdrop with cutout for the spotlight (box-shadow trick). */}
      {targetRect ? (
        <div
          className="tutorial-spotlight"
          style={{
            top: targetRect.top,
            left: targetRect.left,
            width: targetRect.width,
            height: targetRect.height,
          }}
        />
      ) : (
        <div className="tutorial-backdrop-plain" />
      )}

      {/* Tooltip / modal */}
      <div
        className={`tutorial-tooltip${targetRect ? "" : " tutorial-tooltip-center"}`}
        style={tooltip}
      >
        <div className="tutorial-tooltip-header">
          <div className="tutorial-step-counter">{stepIndex + 1} / {totalSteps}</div>
          <button className="tutorial-skip" onClick={onSkip} title="Esc">
            {skipLabel} <span className="tutorial-kbd">Esc</span>
          </button>
        </div>
        <h4 className="tutorial-tooltip-title">{title}</h4>
        <p className="tutorial-tooltip-body">{body}</p>
        {step.isModeSelect ? (
          <div className="tutorial-mode-select">
            <ModeButton
              mode="beginner"
              hotkey="1"
              label={lang === "ja" ? "初級" : "Beginner"}
              sub={lang === "ja" ? "最小ステップ" : "Fastest path"}
              onClick={onChooseMode}
            />
            <ModeButton
              mode="intermediate"
              hotkey="2"
              label={lang === "ja" ? "中級" : "Intermediate"}
              sub={lang === "ja" ? "AIツール+拡張" : "AI tools & aug"}
              onClick={onChooseMode}
            />
            <ModeButton
              mode="expert"
              hotkey="3"
              label={lang === "ja" ? "エキスパート" : "Expert"}
              sub={lang === "ja" ? "全機能" : "All features"}
              onClick={onChooseMode}
            />
          </div>
        ) : (
          <div className="tutorial-actions">
            <button
              className="tutorial-btn tutorial-btn-secondary"
              onClick={onBack}
              disabled={stepIndex === 0}
              title="←"
            >
              {backLabel} <span className="tutorial-kbd">←</span>
            </button>
            <button className="tutorial-btn tutorial-btn-primary" onClick={onNext} title="Enter">
              {nextLabel} <span className="tutorial-kbd">Enter</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ModeButton({ mode, hotkey, label, sub, onClick }: {
  mode: TutorialMode;
  hotkey?: string;
  label: string;
  sub: string;
  onClick: (mode: TutorialMode) => void;
}) {
  return (
    <button className={`tutorial-mode-btn tutorial-mode-${mode}`} onClick={() => onClick(mode)}>
      <span className="tutorial-mode-label">
        {label}
        {hotkey && <span className="tutorial-kbd">{hotkey}</span>}
      </span>
      <span className="tutorial-mode-sub">{sub}</span>
    </button>
  );
}

/**
 * Compute tooltip position so it sits next to the spotlight, clamped to viewport.
 * No target → centered modal.
 */
function useTooltipPosition(step: TutorialStep, target: Rect | null): React.CSSProperties {
  const [style, setStyle] = useState<React.CSSProperties>({});

  useEffect(() => {
    if (!target) {
      setStyle({ top: "50%", left: "50%", transform: "translate(-50%, -50%)", width: TOOLTIP_CENTER_WIDTH });
      return;
    }

    const tw = TOOLTIP_WIDTH;
    const th = 240;
    const placement = step.placement ?? "bottom";
    const vpW = window.innerWidth, vpH = window.innerHeight;

    let top: number, left: number;
    if (placement === "top") {
      top = target.top - th - GAP;
      left = target.left + target.width / 2 - tw / 2;
    } else if (placement === "left") {
      top = target.top + target.height / 2 - th / 2;
      left = target.left - tw - GAP;
    } else if (placement === "right") {
      top = target.top + target.height / 2 - th / 2;
      left = target.left + target.width + GAP;
    } else if (placement === "center") {
      top = vpH / 2 - th / 2;
      left = vpW / 2 - tw / 2;
    } else { // bottom
      top = target.top + target.height + GAP;
      left = target.left + target.width / 2 - tw / 2;
    }

    if (top < 8) top = target.top + target.height + GAP;
    if (top + th > vpH - 8) top = Math.max(8, target.top - th - GAP);
    if (left < 8) left = 8;
    if (left + tw > vpW - 8) left = vpW - tw - 8;

    setStyle({ top, left, width: tw });
  }, [step, target]);

  return style;
}
