// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

import { VERDICT_CLASS, type VerdictName } from "./verdictFace";

/** One mark, five silhouettes, no two sharing an outline.
 *
 *  Why shapes and not characters: the old set was the glyphs U+2715 U+25B3
 *  U+25CF U+25CB, which on a Windows box are served by three different font
 *  families -- so they sat at different heights and weights from row to row,
 *  and could fall back to tofu on a machine with no CJK font. Worse, a font
 *  offers no lever but fill, and fill is what produced a filled circle for
 *  "detected" beside a hollow circle for "clean": two marks nobody can
 *  separate at 11px.
 *
 *  The separations here are all coarse and pre-attentive, so they survive
 *  12px and they survive a reader who cannot use hue at all:
 *    square    corners, and the only mark that fills its box
 *    triangle  the only silhouette converging to a point at the top
 *    circle    the only closed curve, and the only circle left in the row
 *    bar       one-dimensional, 0 degrees
 *    slash     one-dimensional, 45 degrees
 *  Ink mass descends 121 / 66 / 38 / 33 / 24 px^2, so the state most worth
 *  catching is the heaviest and the ordinary majority is the quietest.
 */

const GEOMETRY: Record<VerdictName, (p: Record<string, unknown>) => React.ReactElement> = {
  missed: (p) => <rect x={0.5} y={0.5} width={11} height={11} rx={1.5} {...p} />,
  over: (p) => <path d="M6 0.6 L11.6 11.4 L0.4 11.4 Z" strokeLinejoin="round" {...p} />,
  detected: (p) => <circle cx={6} cy={6} r={3.5} {...p} />,
  clean: (p) => <rect x={0.5} y={4.5} width={11} height={3} rx={1.5} {...p} />,
  unjudged: (p) => <line x1={1.6} y1={10.4} x2={10.4} y2={1.6} strokeLinecap="round" {...p} />,
};

/** Okabe-Ito, on the light panel. The outlines are not decoration and must
 *  not be optimised away: #E69F00 on white is 2.25:1 and fails WCAG 1.4.11
 *  for a graphical object; the darker edge (4.07:1) is what carries it. The
 *  unjudged grey is #8A8A8A (3.45:1) rather than a paler one for the same
 *  reason. Colour rides on top of the silhouette here -- it is reinforcement,
 *  never the only channel. */
const SOLID: Record<VerdictName, Record<string, unknown>> = {
  missed: { fill: "#D55E00", stroke: "#A94900", strokeWidth: 0.75 },
  over: { fill: "#E69F00", stroke: "#A87400", strokeWidth: 1 },
  detected: { fill: "#009E73", stroke: "#007355", strokeWidth: 0.75 },
  clean: { fill: "#6B6B6B" },
  unjudged: { fill: "none", stroke: "#8A8A8A", strokeWidth: 2.2 },
};

/** For the pill over the preview, whose background is already the verdict
 *  colour: a #D55E00 square on a #D55E00 pill is an invisible mark. Same
 *  silhouettes, knocked out in white. */
const ON_COLOR: Record<VerdictName, Record<string, unknown>> = {
  missed: { fill: "#fff" },
  over: { fill: "#fff" },
  detected: { fill: "#fff" },
  clean: { fill: "#fff" },
  unjudged: { fill: "none", stroke: "#fff", strokeWidth: 2.2 },
};

export function VerdictMark({
  verdict,
  label,
  variant = "solid",
}: {
  verdict: VerdictName;
  /** Given only where the mark is the sole carrier of the state. Where a text
   *  label sits beside it, the mark is decorative and stays out of the tree. */
  label?: string;
  variant?: "solid" | "onColor";
}) {
  const paint = variant === "onColor" ? ON_COLOR[verdict] : SOLID[verdict];
  return (
    <svg
      className={`verdict-mark ${VERDICT_CLASS[verdict]}`}
      viewBox="0 0 12 12"
      role={label ? "img" : undefined}
      aria-hidden={label ? undefined : true}
      aria-label={label}
    >
      {GEOMETRY[verdict](paint)}
    </svg>
  );
}

