# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Named operating points swept over BOTH knobs the results tab exposes.

The run's metrics.json already carries operating points, but they are a
threshold-only sweep measured on the val set while training. Two consequences:
they cannot answer for the area filter, which is the other control on the
panel, and they cannot be recomputed for a run that already exists.

This sweeps (confidence, min_area) over the prediction summaries instead --
the same components ``detection_verdict`` cuts once at zero confidence and
stores a 256-bin histogram for, so area at any threshold is a suffix sum. No
inference is re-run and no model is loaded; the whole thing is arithmetic over
files already on disk.

Counts, not pixels. The threshold-only curves are pixel F1, but ``min_area``
is a component-level operation and the presets already explain themselves in
defects ("misses 9 of 20"). Mixing the two units in one objective would make
the winner depend on which one moved.

THE GUARD. Raising ``min_area`` is the cheapest way to make false alarms go
away, and on a line that inspects for tiny defects it is also the cheapest way
to throw real ones away. So candidates are scored against the same threshold
with the filter off, and any (confidence, min_area) that turns even one found
defect into a missed one is not eligible for the recall-first or balanced
points. The precision-first point may spend defects -- that is what it is for
-- but it has to say how many. The bound is measured, not derived: a formula
over the smallest annotation would miss the case where two components together
cover one defect and only one of them is small.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .detection_verdict import (
    SUMMARY_SUFFIX,
    _suffix,
    conf_bin_for_percent,
    evaluate_summary,
    load_prepared_summary,
)

#: Percent steps for the confidence axis. The slider is integer percent; 2%
#: keeps the grid at 51 points, which is the difference between a sweep that
#: returns while someone waits and one that does not.
THRESHOLD_STEP_PCT = 2

#: Geometric, because the useful resolution is at the small end: the gap
#: between 5 and 10 px decides whether a speck survives, the gap between 640
#: and 1280 decides nothing anyone tunes for. Grown from the data rather than
#: fixed -- a 5120 px ceiling is most of a 640x480 frame and a rounding error
#: on a 2560x1920 one, and a fixed ladder silently clips the answer on the
#: second. See build_min_area_ladder.
_LADDER_BASE: tuple[int, ...] = (0, 5, 10, 20, 40, 80, 160, 320, 640)


def build_min_area_ladder(summaries: list[dict[str, Any]]) -> tuple[int, ...]:
    """Geometric steps up to the largest predicted component in the run.

    Past that every component is filtered and the surface is flat, so the
    ladder ends there: an optimum "beyond the end" is then impossible rather
    than unreported.
    """
    largest = 0
    for summary in summaries:
        # load_prepared_summary replaces each histogram with its suffix table,
        # so summing the list is not the pixel count -- it is the sum of every
        # suffix, which put the top of the ladder past the size of the image.
        # Area at bin 0 is the first suffix, and _suffix knows which form the
        # summary is in.
        prepared = bool(summary.get("prepared"))
        for comp in summary.get("pred") or []:
            area = _suffix(comp["hist"], 0, prepared)
            if area > largest:
                largest = area
    ladder = [a for a in _LADDER_BASE if a <= largest]
    step = _LADDER_BASE[-1] * 2 if _LADDER_BASE[-1] else 1
    while step <= largest:
        ladder.append(step)
        step *= 2
    if not ladder:
        ladder = [0]
    return tuple(sorted(set(ladder)))


def _count_f1(detected: int, missed: int, over: int) -> float:
    """F1 over defects, not pixels: 2d / (2d + missed + over)."""
    denom = 2 * detected + missed + over
    return (2 * detected / denom) if denom else 0.0


def _precision(detected: int, over: int) -> float:
    """Defect precision, and 0.0 where nothing was called at all.

    Predicting nothing raises no false alarms, so a ratio that treats 0/0 as
    perfect makes "turn the model off" the cleanest setting on the surface --
    which is what the first version of this picked.
    """
    denom = detected + over
    return (detected / denom) if denom else 0.0


def smallest_defect_component(
    summaries: list[dict[str, Any]], *, bin_: int,
) -> int | None:
    """Area of the smallest predicted component sitting on a defect.

    The floor the sweep is allowed to recommend. "No defect was lost" is a
    weaker statement than it sounds: a defect covered by two components
    survives losing one of them, so a filter can discard real evidence and
    still score clean. This is the direct question instead -- how small is the
    smallest blob that is actually on a defect -- and a floor above it is a
    standing instruction to ignore something the size of a real find.

    None when nothing overlaps a defect at this threshold, which leaves the
    caller with no evidence to bound the filter by.
    """
    smallest: int | None = None
    for summary in summaries:
        prepared = bool(summary.get("prepared"))
        for comp in summary.get("pred") or []:
            on_gt_hist = comp.get("on_gt_hist")
            if on_gt_hist is None:
                continue
            if _suffix(on_gt_hist, bin_, prepared) <= 0:
                continue
            area = _suffix(comp["hist"], bin_, prepared)
            if area > 0 and (smallest is None or area < smallest):
                smallest = area
    return smallest


def _totals(summaries: list[dict[str, Any]], *, bin_: int, coverage: float, min_area: int
            ) -> tuple[int, int, int]:
    detected = missed = over = 0
    for summary in summaries:
        r = evaluate_summary(
            summary, min_conf_bin=bin_, coverage=coverage, min_area=min_area,
        )
        detected += r["detected"]
        missed += r["missed"]
        over += r["over"]
    return detected, missed, over


def load_run_summaries(pred_dir: Path) -> list[dict[str, Any]]:
    """Every prepared summary in a prediction directory.

    Images with no annotation are kept but contribute nothing: evaluate_summary
    counts neither defects nor false alarms where there is no ground truth to
    be right or wrong against.
    """
    out: list[dict[str, Any]] = []
    if not pred_dir.exists():
        return out
    for path in sorted(pred_dir.glob(f"*{SUMMARY_SUFFIX}")):
        item_id = path.name[: -len(SUMMARY_SUFFIX)]
        summary = load_prepared_summary(pred_dir, item_id)
        if summary is not None:
            out.append(summary)
    return out


def sweep_operating_points(
    summaries: list[dict[str, Any]],
    *,
    coverage: float = 0.5,
    threshold_step_pct: int = THRESHOLD_STEP_PCT,
    min_areas: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Sweep both axes and name three points on the surface.

    Returns the three presets plus the grid size and how many annotated images
    the answer is based on -- a point chosen over four images is not a
    recommendation, and the caller has to be able to say so.
    """
    if not summaries:
        return {"points": {}, "images": 0, "annotated_images": 0, "grid": 0}

    annotated = sum(1 for s in summaries if s.get("gt") is not None)
    pcts = list(range(0, 101, max(1, threshold_step_pct)))
    if min_areas is None:
        min_areas = build_min_area_ladder(summaries)
    areas = tuple(sorted(set(int(a) for a in min_areas if int(a) >= 0)))

    # baseline[pct] = counts with the area filter off, i.e. what the threshold
    # alone achieves. Every candidate is judged against its own column.
    baseline: dict[int, tuple[int, int, int]] = {}
    grid: list[dict[str, Any]] = []
    for pct in pcts:
        bin_ = conf_bin_for_percent(float(pct))
        base = _totals(summaries, bin_=bin_, coverage=coverage, min_area=0)
        baseline[pct] = base
        # The floor this threshold is allowed to carry. Without it the sweep
        # walks straight into the trap it exists to avoid: at a low threshold
        # components merge into large blobs, so a huge min_area costs nothing
        # measurable on THIS annotated set while telling the line to ignore
        # anything smaller than a postage stamp on every future image.
        ceiling = smallest_defect_component(summaries, bin_=bin_)
        for area in areas:
            if ceiling is not None and area > ceiling:
                continue
            if area == 0:
                detected, missed, over = base
            else:
                detected, missed, over = _totals(
                    summaries, bin_=bin_, coverage=coverage, min_area=area,
                )
            grid.append({
                "threshold_pct": pct,
                "min_area": area,
                "min_area_ceiling": ceiling,
                # "the filter cannot go higher here", decided against the
                # ladder rather than against the ceiling. The ladder is
                # geometric, so a rung almost never lands exactly on the
                # ceiling: comparing min_area to it says "not at the cap" for
                # a setting that is, in fact, the largest one available.
                "at_area_ceiling": (
                    ceiling is not None
                    and any(a > area for a in areas)
                    and not any(area < a <= ceiling for a in areas)
                ),
                "detected": detected,
                "missed": missed,
                "over": over,
                "f1": _count_f1(detected, missed, over),
                "precision": _precision(detected, over),
                # Defects that were found without the filter and are not found
                # with it. This is the number the guard is about.
                "defects_lost_to_area": max(0, missed - base[1]),
            })

    safe = [g for g in grid if g["defects_lost_to_area"] == 0]
    pool = safe or grid

    # The same rules the threshold-only presets use, on a surface instead of a
    # curve. No target constant: "recall >= 0.90" is met by every setting on an
    # easy task and by none on a hard one, and the preset stops meaning
    # anything in both directions. "The best this model can do" is defined for
    # every run.
    best_detected = max(g["detected"] for g in pool)
    best_precision = max(g["precision"] for g in pool)

    # Among settings that score identically, prefer the one that leans least
    # on min_area. Confidence and area can reach the same numbers here, and
    # they are not equally safe to lean on: confidence is measured per pixel
    # and reversible, while an area floor is a standing instruction to ignore
    # anything small on a line whose whole job is finding small things. Ties
    # therefore spend the safe knob first. Only then the house tie-break on
    # threshold -- toward the lower one, so a preset and the run's shipped
    # default cannot disagree about the same score.
    def rank(*primary):
        return lambda g: (*[k(g) for k in primary], g["min_area"], g["threshold_pct"])

    points = {
        # Find every defect this model can find at any setting; of those, the
        # quietest.
        "recall_first": min(
            (g for g in pool if g["detected"] == best_detected),
            key=rank(lambda g: -g["precision"]),
        ),
        "balanced": min(pool, key=rank(lambda g: -g["f1"])),
        # The mirror: hold precision at its attainable best, keep the most
        # defects of those settings.
        "precision_first": min(
            (g for g in pool if g["precision"] == best_precision),
            key=rank(lambda g: -g["detected"]),
        ),
    }
    ladder_max = max(areas) if areas else 0
    return {
        "points": {k: v for k, v in points.items() if v is not None},
        "images": len(summaries),
        "annotated_images": annotated,
        "grid": len(grid),
        # False means no setting on the surface was free of cost, so the
        # presets were chosen from the whole grid and may spend defects. The
        # caller has to say so rather than presenting them as safe.
        "guarded": bool(safe),
        # A preset sitting on the largest filter we tried is a preset whose
        # real optimum may be past the end of the ladder. Say so instead of
        # letting a clipped answer look like a chosen one.
        "min_area_clipped": any(
            p["min_area"] == ladder_max and ladder_max > 0 for p in points.values()
        ),
        "min_area_max": ladder_max,
        "coverage": coverage,
    }
