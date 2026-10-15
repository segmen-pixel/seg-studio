// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getStepsForMode, type TutorialMode, type TutorialStep } from "../tutorialSteps";
import type { TabId } from "../types";

const STORAGE_KEY = "seg-tutorial-state";

type PersistedState = {
  completed: boolean;
  skipped: boolean;
  lastStep: number;
  mode: TutorialMode | null;
};

function readPersisted(): PersistedState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { completed: false, skipped: false, lastStep: 0, mode: null };
    const parsed = JSON.parse(raw);
    return {
      completed: !!parsed.completed,
      skipped: !!parsed.skipped,
      lastStep: Number.isFinite(parsed.lastStep) ? parsed.lastStep : 0,
      mode: parsed.mode === "beginner" || parsed.mode === "intermediate" || parsed.mode === "expert" ? parsed.mode : null,
    };
  } catch {
    return { completed: false, skipped: false, lastStep: 0, mode: null };
  }
}

function writePersisted(state: PersistedState) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* ignore */ }
}

/**
 * Orchestrates the hands-on tutorial overlay. Auto-starts on first launch and
 * advances itself when the user performs the expected action (tab switch / click),
 * but every step also has a manual Next button. Steps are filtered by the mode
 * the user picks on the welcome screen.
 */
export function useTutorial(activeTab: TabId, switchTab: (tab: TabId) => void) {
  const [active, setActive] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [mode, setMode] = useState<TutorialMode>("beginner");
  const initialized = useRef(false);

  // First-launch auto-start
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    const st = readPersisted();
    if (!st.completed && !st.skipped) {
      setActive(true);
      setStepIndex(0);
      if (st.mode) setMode(st.mode);
    }
  }, []);

  const steps: TutorialStep[] = useMemo(() => getStepsForMode(mode), [mode]);
  const totalSteps = steps.length;
  const currentStep = active ? steps[stepIndex] ?? null : null;

  const persistNow = useCallback((patch: Partial<PersistedState>) => {
    const prev = readPersisted();
    writePersisted({ ...prev, ...patch });
  }, []);

  const complete = useCallback(() => {
    setActive(false);
    persistNow({ completed: true, skipped: false, lastStep: totalSteps, mode });
  }, [totalSteps, mode, persistNow]);

  const skip = useCallback(() => {
    setActive(false);
    persistNow({ completed: false, skipped: true, lastStep: stepIndex, mode });
  }, [stepIndex, mode, persistNow]);

  const next = useCallback(() => {
    setStepIndex((prev) => {
      const n = prev + 1;
      if (n >= totalSteps) {
        setActive(false);
        persistNow({ completed: true, skipped: false, lastStep: totalSteps, mode });
        return prev;
      }
      persistNow({ completed: false, skipped: false, lastStep: n, mode });
      return n;
    });
  }, [totalSteps, mode, persistNow]);

  const back = useCallback(() => {
    setStepIndex((prev) => {
      const n = Math.max(0, prev - 1);
      persistNow({ completed: false, skipped: false, lastStep: n, mode });
      return n;
    });
  }, [mode, persistNow]);

  const restart = useCallback(() => {
    setStepIndex(0);
    setActive(true);
    persistNow({ completed: false, skipped: false, lastStep: 0, mode: null });
  }, [persistNow]);

  /** Called from the welcome step's mode-select buttons. */
  const chooseMode = useCallback((m: TutorialMode) => {
    setMode(m);
    // Advance to step 1 after mode is set (skip welcome).
    setStepIndex(1);
    persistNow({ completed: false, skipped: false, lastStep: 1, mode: m });
  }, [persistNow]);

  // Required-tab auto-switch
  useEffect(() => {
    if (!currentStep || !currentStep.requireTab) return;
    if (activeTab !== currentStep.requireTab) {
      switchTab(currentStep.requireTab);
    }
  }, [currentStep, activeTab, switchTab]);

  // Auto-advance on tabEnter.
  //
  // Only fire when the active tab actually CHANGES to the target. Without
  // this guard, navigating back (← key) onto a tabEnter step while the
  // user is already on the target tab — which is exactly what happens at
  // the "go to Training tab" step (#14) after auto-advance moved them to
  // #15 — would re-fire next() on every render and bounce the user
  // straight forward again. The prevTabRef captures the activeTab the
  // last time this effect ran so we can distinguish "tab just changed"
  // from "we re-rendered with the same tab".
  const prevTabRef = useRef<TabId>(activeTab);
  useEffect(() => {
    const prevTab = prevTabRef.current;
    prevTabRef.current = activeTab;
    if (!currentStep) return;
    if (currentStep.advanceOn.type !== "tabEnter") return;
    if (activeTab !== currentStep.advanceOn.tabId) return;
    if (prevTab === activeTab) return;
    next();
  }, [currentStep, activeTab, next]);

  // Auto-advance on click
  useEffect(() => {
    if (!currentStep || currentStep.advanceOn.type !== "click") return;
    const selector = currentStep.advanceOn.selector;
    const handler = (e: MouseEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.(selector);
      if (el) next();
    };
    document.addEventListener("click", handler, true);
    return () => document.removeEventListener("click", handler, true);
  }, [currentStep, next]);

  // onEnterClickSelector — programmatically click an element when the step
  // becomes active (used to auto-open dialogs whose contents the next steps
  // will spotlight, e.g. AugmentDialog for the Perlin walkthrough). Fired
  // once per step activation; the ref guards against re-firing if the same
  // step's effect re-runs (e.g. on tab switch).
  const lastAutoClickStepRef = useRef<string | null>(null);
  useEffect(() => {
    if (!currentStep || !currentStep.onEnterClickSelector) return;
    if (lastAutoClickStepRef.current === currentStep.id) return;
    lastAutoClickStepRef.current = currentStep.id;
    // Defer one frame so the spotlight has rendered first.
    const selector = currentStep.onEnterClickSelector;
    const raf = requestAnimationFrame(() => {
      const el = document.querySelector(selector) as HTMLElement | null;
      el?.click();
    });
    return () => cancelAnimationFrame(raf);
  }, [currentStep]);

  return {
    active,
    mode,
    stepIndex,
    totalSteps,
    currentStep,
    next,
    back,
    skip,
    complete,
    restart,
    chooseMode,
  } as const;
}
