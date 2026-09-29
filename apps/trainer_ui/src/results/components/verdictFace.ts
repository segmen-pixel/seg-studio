// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type { ItemVerdict } from "../../api";
import type { TranslationKey } from "../../i18n";

/** The five states a row can be in, worst first.
 *
 *  "unjudged" is a state, not the absence of one. A run predicted before
 *  summaries existed -- or one the server is still backfilling -- used to
 *  render as an empty slot, which read as "fine". It now gets its own mark.
 */
export type VerdictName = "missed" | "over" | "detected" | "clean" | "unjudged";

export const VERDICT_ORDER: VerdictName[] = [
  "missed",
  "over",
  "detected",
  "clean",
  "unjudged",
];

export const VERDICT_CLASS: Record<VerdictName, string> = {
  missed: "verdict-missed",
  over: "verdict-over",
  detected: "verdict-detected",
  clean: "verdict-clean",
  unjudged: "verdict-unjudged",
};

/** i18n keys, kept beside the marks so a mark and its name cannot drift. */
export const VERDICT_LABEL_KEY: Record<VerdictName, TranslationKey> = {
  missed: "results.verdict.missed",
  over: "results.verdict.over",
  detected: "results.verdict.detected",
  clean: "results.verdict.clean",
  unjudged: "results.verdict.unjudged",
};

/** A missing verdict is "unjudged", so every row is drawable and an empty
 *  slot means a rendering bug rather than a state. */
export function verdictNameOf(result: ItemVerdict | undefined): VerdictName {
  return (result?.verdict ?? "unjudged") as VerdictName;
}

export const pctOf = (v: number) => `${Math.round(v * 100)}%`;
