# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Per-Region Overlap: how much of each defect was found, region by region.

Pixel F1 charges a defect for its own outline. A blob of ~2000px has a
perimeter of ~160px, so a mask that is right everywhere except one pixel of
boundary already loses several points, and one that is off by two or three
loses ten. On a task whose foreground is under one percent of the image that
is most of the error budget -- and none of it is a missed defect.

PRO asks a different question: of THIS region's pixels, what share did the
prediction cover? Averaged over regions rather than pixels, so a large defect
cannot drown out a small one, which is the failure the pixel metrics have on
a dataset with one big blob and several small ones.

PRO alone is not a score. Predicting the whole image scores 1.0. It is read
against the false-positive rate at the same threshold, and the pair is what
AUPRO integrates: the area under PRO-vs-FPR up to an FPR the operator is
willing to accept, normalised by that cap so it lands in [0, 1].

The cap is not a constant here. MVTec's published AUPRO uses 0.3 because an
anomaly map sweeps its whole score range; a trained segmentation model with
saturated probabilities may never produce an FPR that high at any threshold,
and normalising by a cap the curve never reaches divides by mostly-empty
area. So the achieved FPR range travels with the number, and a cap the data
does not reach is reported as such rather than silently folded in.

Regions come from the ground truth, once per image: which pixels form one
defect does not depend on the threshold being tested. Background is never a
region -- it would arrive as one enormous component that the model also
predicts, a guaranteed perfect score on every image (the same trap
_compute_instance_recall documents in report_builders.py).
"""
from __future__ import annotations

import numpy as np


def gt_regions(target: np.ndarray, class_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Label the connected components of *class_id* in *target*.

    Returns ``(labels, areas)`` where ``labels`` is 0 for everything that is
    not this class and 1..n for the components, and ``areas[i]`` is the pixel
    count of component ``i + 1``. An empty class gives ``(zeros, empty)``.
    """
    binary = target == int(class_id)
    if not binary.any():
        return np.zeros(target.shape, dtype=np.int32), np.zeros(0, dtype=np.int64)
    try:
        from scipy.ndimage import label as _label
    except ImportError:  # pragma: no cover - scipy is a hard dep of the trainer
        return np.zeros(target.shape, dtype=np.int32), np.zeros(0, dtype=np.int64)
    labels, n = _label(binary)
    if n == 0:
        return np.zeros(target.shape, dtype=np.int32), np.zeros(0, dtype=np.int64)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    return labels.astype(np.int32), areas


#: Covered fraction at which a GT defect counts as FOUND rather than missed.
#: The same 0.5 the model-evaluation report already uses
#: (``_compute_instance_recall`` in report_builders.py), measured the same way:
#: the denominator is the instance's own area, so a model that finds more
#: defects elsewhere in the image cannot push this one into "missed". Two
#: numbers both called instance recall that disagree would be worse for an
#: operator than only having the one.
INSTANCE_DETECT_COVERAGE = 0.5


def region_overlaps(labels: np.ndarray, areas: np.ndarray,
                    predicted: np.ndarray) -> tuple[float, int, int]:
    """Per-region covered fractions: their sum, the region count, the found count.

    ``predicted`` is a boolean mask of the pixels called this class at the
    threshold under test. The denominator is the region's own area, never the
    union with the prediction: a model that finds MORE defects elsewhere in
    the image must not lower this region's score.

    The third value counts the regions covered at least
    ``INSTANCE_DETECT_COVERAGE`` -- the defects the model FOUND, where PRO
    reports how MUCH of them it found. It rides along here because the covered
    fractions are already in hand: a defect-level recall over the whole
    threshold sweep costs no second pass over the image.
    """
    if areas.size == 0:
        return 0.0, 0, 0
    hit = labels[predicted]
    covered = np.bincount(hit.ravel(), minlength=areas.size + 1)[1:]
    fractions = covered / areas
    return (
        float(np.sum(fractions)),
        int(areas.size),
        int(np.count_nonzero(fractions >= INSTANCE_DETECT_COVERAGE)),
    )


def build_pro_curve(pro_sum: dict, pro_count: dict, fp_sum: dict,
                    negatives: int, hit_sum: dict | None = None) -> list[dict]:
    """One point per threshold: ``{threshold, pro, fpr}``, ascending threshold.

    ``pro_sum`` / ``pro_count`` accumulate over regions, ``fp_sum`` over
    background pixels called foreground, and *negatives* is how many
    background pixels were evaluated in total.

    When *hit_sum* is supplied -- regions covered at least
    ``INSTANCE_DETECT_COVERAGE``, accumulated the same way -- each point also
    carries ``instance_recall`` with the raw ``instances_found`` /
    ``instances_total`` it came from. The counts travel with the ratio because
    "6 of 7 defects" and "0.857" are read differently by an operator, and
    because integer counts let a caller compare two thresholds exactly.
    Omitted entirely when *hit_sum* is None, so a reader can tell a run that
    did not measure it from one that measured zero.
    """
    curve = []
    for t in sorted(pro_sum):
        n = pro_count.get(t, 0)
        point = {
            "threshold": float(t),
            "pro": float(pro_sum[t] / n) if n else 0.0,
            "fpr": float(fp_sum.get(t, 0.0) / negatives) if negatives else 0.0,
        }
        if hit_sum is not None:
            found = int(hit_sum.get(t, 0))
            point["instance_recall"] = float(found / n) if n else 0.0
            point["instances_found"] = found
            point["instances_total"] = int(n)
        curve.append(point)
    return curve


def compute_aupro(curve: list[dict], fpr_cap: float) -> dict:
    """Normalised area under PRO-vs-FPR up to *fpr_cap*.

    Returns the score together with the FPR the curve actually reached, so a
    number produced on a task that never gets near the cap can be recognised
    as one. ``covered`` is the fraction of the cap the curve spans; an AUPRO
    with a small ``covered`` is an extrapolation, not a measurement.
    """
    points = sorted(
        ({"fpr": p["fpr"], "pro": p["pro"]} for p in curve),
        key=lambda p: (p["fpr"], p["pro"]),
    )
    if not points or fpr_cap <= 0:
        return {"aupro": 0.0, "fpr_cap": float(fpr_cap), "fpr_max": 0.0, "covered": 0.0}
    fpr_max = points[-1]["fpr"]
    xs = [0.0]
    ys = [0.0]
    for p in points:
        x = min(p["fpr"], fpr_cap)
        # A curve is a function of FPR; keep the best PRO seen at each x.
        if x == xs[-1]:
            ys[-1] = max(ys[-1], p["pro"])
        else:
            xs.append(x)
            ys.append(max(p["pro"], ys[-1]))
        if p["fpr"] >= fpr_cap:
            break
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    # Beyond the last measured point the curve is unknown. Holding the last
    # PRO flat to the cap is the optimistic reading, so it is NOT done here:
    # the area stops where the measurement stops and `covered` says so.
    return {
        "aupro": float(area / fpr_cap),
        "fpr_cap": float(fpr_cap),
        "fpr_max": float(fpr_max),
        "covered": float(min(fpr_max, fpr_cap) / fpr_cap),
    }
