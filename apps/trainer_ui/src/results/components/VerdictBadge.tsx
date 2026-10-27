// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

import type { ItemVerdict } from "../../api";
import { VERDICT_CLASS, VERDICT_LABEL_KEY, pctOf as pct, verdictNameOf } from "./verdictFace";
import { VerdictMark } from "./VerdictMark";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TFn = (key: any) => string;

export function VerdictBadge({ result, t }: { result: ItemVerdict | undefined; t: TFn }) {
  // No annotation means no verdict. Showing a rate there would be a claim
  // about an image nothing was ever checked against. The list still draws a
  // slash for that row -- every row has to account for itself -- but a pill
  // over the image is an assertion, and there is nothing to assert here.
  if (!result || result.verdict === null) return null;
  const name = verdictNameOf(result);

  // The headline is the area actually covered, not the instance tally. By
  // count, a defect the model found 90% of reads exactly like one it never
  // touched -- and a single stray pixel left behind while annotating is a
  // whole "missed instance" that flips the image's label on its own.
  //
  // When there is no rate the headline names the state instead. It used to say
  // "clean" there, which is a different claim: match_rate is null whenever
  // gt_area is 0, and that is a fact about the ANNOTATION, not the verdict. An
  // unannotated image the model painted on is over-detection -- the server
  // reserves "clean" for nothing annotated AND nothing predicted -- so the
  // pill read "no anomaly, 100% excess area" beside a mark it was already
  // drawing, correctly, as the over-detection triangle.
  const matched = result.match_rate;
  const over = result.over_rate ?? 0;
  const rate = matched === null || matched === undefined ? null : pct(matched);
  const headline = rate === null
    ? t(VERDICT_LABEL_KEY[name])
    : `${t("results.verdict.match")} ${rate}`;

  // The area figure and the instance count answer different questions. A
  // prediction that traces a defect well but spills a little past the drawn
  // outline scores a few percent of excess area while being nobody's idea of
  // a false alarm: it has no over-detected components at all. Label the area
  // as area, and keep the counts for the tooltip.
  const breakdown = `${t("results.verdict.detected")} ${result.detected} / `
    + `${t("results.verdict.missed")} ${result.missed} / `
    + `${t("results.verdict.over")} ${result.over}`;

  return (
    <div className={`results-verdict-badge ${VERDICT_CLASS[name]}`} title={breakdown}>
      {/* Knocked out in white: the pill background is already the verdict
          colour, so the solid variant would be an invisible mark. */}
      <VerdictMark verdict={name} variant="onColor" />
      <span className="results-verdict-label">{headline}</span>
      {over > 0.005 && (
        <span className="results-verdict-over">
          {t("results.verdict.overArea")} {pct(over)}
        </span>
      )}
    </div>
  );
}
