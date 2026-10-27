# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Inference-threshold search over accumulated per-threshold TP/FP/FN stats.

Extracted from metrics.py during the pre-OSS refactor; metrics.py
re-exports these names for backward compatibility.
"""
from __future__ import annotations

import numpy as np

# Candidate thresholds for the optimal FG threshold search. Applied to the
# summed foreground probability, which is a real probability in [0, 1] -- see
# sliding_window.blend_accumulated_probs for the period when it was not.
THRESHOLD_CANDIDATES = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02..0.98 step 0.02

#: FPR the AUPRO integral stops at. MVTec publishes at 0.3 because an anomaly
#: map sweeps its whole score range; a trained segmentation model with
#: saturated probabilities may not reach that FPR at any threshold in the grid
#: above. The score therefore travels with `fpr_max` and `covered`, and a
#: curve that falls short of the cap is reported as falling short rather than
#: being held flat to it.
AUPRO_FPR_CAP = 0.3


def gt_present_classes(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> list[int]:
    """Foreground classes present in the GT -- the same set at every threshold.

    ``tp[c] + fn[c]`` is the number of GT pixels of class c (a GT pixel is
    either matched or missed), so it does not move with the threshold. Fixing
    the set once is what keeps the sweep comparable.

    Without this the macro average was taken over "classes with any pixel
    activity" (finalize_metrics skips a class whose ``2*tp + fp + fn`` is zero),
    so a class that is declared but absent from the evaluated GT -- ordinary
    mid-project, when a label exists but has not been drawn in the val split --
    dropped OUT of the average as soon as a high threshold stopped predicting
    it. A macro over fewer classes scores HIGHER, so the sweep preferred high
    thresholds for a denominator reason rather than an accuracy one. The winner
    is written to train_config.json as ``inference_threshold`` and shipped to
    serving, and the bias is toward missing defects.
    """
    classes = [
        cls for cls in range(num_classes)
        if cls != ignore_index and cls != 0
    ]
    if not per_threshold_stats:
        return classes
    tp, _fp, fn = next(iter(per_threshold_stats.values()))
    present = [cls for cls in classes if (tp[cls] + fn[cls]) > 0]
    # No foreground anywhere in the GT: keep the declared set rather than
    # averaging over nothing, so every threshold scores 0 and the sweep is a
    # tie instead of a meaningless ranking.
    return present or classes


def macro_f1_over(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    classes: list[int],
) -> float:
    """Macro F1 over a FIXED class list. A class with no activity scores 0.0.

    Scoring it 0 rather than dropping it is the point: a class that exists in
    the GT and is predicted nowhere has recall 0, and that is what the average
    should say.
    """
    if not classes:
        return 0.0
    vals: list[float] = []
    for cls in classes:
        denom = 2.0 * float(tp[cls]) + float(fp[cls]) + float(fn[cls])
        vals.append(float(2.0 * float(tp[cls]) / denom) if denom > 0 else 0.0)
    return float(np.mean(vals))


def macro_precision_recall_over(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    classes: list[int],
) -> tuple[float, float]:
    """Macro precision and recall over the SAME fixed class list as macro_f1_over.

    A class with no activity scores 0.0 on both, for the reason spelled out in
    :func:`macro_f1_over`: a class that is in the GT and predicted nowhere has
    recall 0, and dropping it would flatter the average.

    These are the two numbers the operator actually trades against each other
    when moving the inference threshold, and until now the sweep computed only
    their harmonic mean -- so the results view could show F1 moving without
    showing WHICH side of it moved.
    """
    if not classes:
        return 0.0, 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for cls in classes:
        t = float(tp[cls])
        pred_pos = t + float(fp[cls])
        gt_pos = t + float(fn[cls])
        precisions.append(t / pred_pos if pred_pos > 0 else 0.0)
        recalls.append(t / gt_pos if gt_pos > 0 else 0.0)
    return float(np.mean(precisions)), float(np.mean(recalls))


def find_optimal_threshold(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> tuple[float, float]:
    """Find the FG threshold that maximizes foreground macro-F1.

    The macro is taken over a class set fixed once by
    :func:`gt_present_classes`, so every candidate threshold is scored on the
    same denominator.

    Returns ``(best_threshold, best_f1)``.
    """
    classes = gt_present_classes(per_threshold_stats, num_classes, ignore_index)
    best_t = 0.5
    best_f1 = -1.0
    for t in sorted(per_threshold_stats.keys()):
        tp, fp, fn = per_threshold_stats[t]
        f1 = macro_f1_over(tp, fp, fn, classes)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def build_f1_curve(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> list[dict[str, float]]:
    """Per-threshold macro F1 with the precision and recall it is made of.

    ``[{"threshold": t, "f1": .., "precision": .., "recall": ..}, ...]`` sorted
    by t. The name is kept for the readers that only want ``f1``; the two extra
    keys cost nothing, because the TP/FP/FN the F1 comes from are already
    accumulated per threshold.
    """
    classes = gt_present_classes(per_threshold_stats, num_classes, ignore_index)
    curve: list[dict[str, float]] = []
    for t in sorted(per_threshold_stats.keys()):
        tp, fp, fn = per_threshold_stats[t]
        precision, recall = macro_precision_recall_over(tp, fp, fn, classes)
        curve.append({
            "threshold": float(t),
            "f1": macro_f1_over(tp, fp, fn, classes),
            "precision": precision,
            "recall": recall,
        })
    return curve


#: Floats compared for "is this the same score" inside the operating-point
#: search. The scores are means of ratios, so two thresholds that are genuinely
#: tied can differ in the last bits.
_TIE_EPS = 1e-9


def _merged_curve(
    f1_curve: list[dict],
    pro_curve: list[dict] | None,
) -> list[dict]:
    """One dict per threshold carrying whatever both curves measured for it."""
    by_threshold: dict[float, dict] = {
        round(float(p["threshold"]), 6): dict(p) for p in f1_curve
    }
    for point in pro_curve or []:
        key = round(float(point["threshold"]), 6)
        if key in by_threshold:
            by_threshold[key].update(
                {k: v for k, v in point.items() if k != "threshold"}
            )
    return [by_threshold[k] for k in sorted(by_threshold)]


def build_operating_points(
    f1_curve: list[dict],
    pro_curve: list[dict] | None = None,
    *,
    unit: str = "pixel",
) -> dict[str, dict]:
    """Named thresholds an operator can pick instead of dragging a slider.

    Computed here rather than in the UI so that one rule serves the results
    view, the report and serving alike, and so the rule is covered by the
    Python suite rather than by eyeballing a slider.

    - ``balanced`` -- highest macro F1. The same point the run already ships as
      ``optimal_threshold``, including its tie-break toward the LOWER
      threshold, so the preset and the shipped default never disagree.
    - ``recall_first`` -- how far the threshold can be RAISED while still
      finding every defect this model finds at any threshold. Among the tied
      thresholds it takes the one with the best precision, i.e. the cleanest
      point that misses nothing extra.
    - ``precision_first`` -- the mirror: how far it can be LOWERED while the
      precision stays at its attainable best.

    There is deliberately no target constant. A fixed "recall >= 0.90" is met
    by every threshold on an easy task and by none on a hard one, and in both
    cases the preset stops meaning anything; "the best this model can do" is
    defined for every run. Each entry reports the metrics AT its threshold, so
    the cost of the choice is visible rather than implied.

    ``recall_first`` keys on defect-level recall when the run measured it
    (``instances_found``, exact integers) and falls back to pixel recall
    otherwise -- ``basis`` says which, because the two answer different
    questions and an operator reading "no defect missed" deserves to know
    whether that was counted in defects or in pixels.

    *unit* reaches the ``basis`` strings only. The rule is the same whatever
    the curve counted, but a counting run measures objects and not pixels, and
    an operator reading ``pixel_precision`` on one would be told the wrong
    thing about what the number is a ratio of.
    """
    points = _merged_curve(f1_curve, pro_curve)
    if not points:
        return {}
    # A curve from before the sweep recorded precision and recall has neither,
    # and every rule below would then compare 0.0 against 0.0 and hand back the
    # last threshold in the grid with total confidence. Returning nothing is the
    # honest answer: the caller shows the raw slider instead of three presets
    # that mean nothing. Silent fallbacks in this file have cost whole runs before.
    if not all(
        isinstance(p.get("precision"), (int, float)) and isinstance(p.get("recall"), (int, float))
        for p in points
    ):
        return {}

    by_instance = all("instances_found" in p for p in points) and any(
        int(p.get("instances_total", 0)) > 0 for p in points
    )

    def found(point: dict) -> float:
        return float(point["instances_found"]) if by_instance else float(point.get("recall", 0.0))

    def entry(point: dict, rule: str, basis: str) -> dict:
        out = {
            "threshold": float(point["threshold"]),
            "rule": rule,
            "basis": basis,
            "f1": float(point.get("f1", 0.0)),
            "precision": float(point.get("precision", 0.0)),
            "recall": float(point.get("recall", 0.0)),
        }
        if "instance_recall" in point:
            out["instance_recall"] = float(point["instance_recall"])
            out["instances_found"] = int(point.get("instances_found", 0))
            out["instances_total"] = int(point.get("instances_total", 0))
        # Counting runs measure objects rather than defect regions, and the
        # two are not the same judgement (one is count agreement, the other
        # is "half of this region was covered"). They travel under separate
        # names so a caller cannot read one as the other.
        if "objects_total" in point:
            out["objects_found"] = int(point.get("objects_found", 0))
            out["objects_total"] = int(point["objects_total"])
        return out

    # Ascending-scan-with-strict-> keeps the lowest threshold among ties, which
    # is what find_optimal_threshold does; -threshold in the key reproduces it.
    balanced = max(points, key=lambda p: (float(p.get("f1", 0.0)), -float(p["threshold"])))

    best_recall = max(found(p) for p in points)
    recall_tied = [p for p in points if found(p) >= best_recall - _TIE_EPS]
    recall_first = max(
        recall_tied,
        key=lambda p: (float(p.get("precision", 0.0)), float(p["threshold"])),
    )

    best_precision = max(float(p.get("precision", 0.0)) for p in points)
    precision_tied = [
        p for p in points if float(p.get("precision", 0.0)) >= best_precision - _TIE_EPS
    ]
    precision_first = min(
        precision_tied,
        key=lambda p: (-found(p), float(p["threshold"])),
    )

    recall_basis = "instance_recall" if by_instance else f"{unit}_recall"
    return {
        "balanced": entry(balanced, "max_f1", f"{unit}_f1"),
        "recall_first": entry(recall_first, "max_recall_then_precision", recall_basis),
        "precision_first": entry(precision_first, "max_precision_then_recall", f"{unit}_precision"),
    }
