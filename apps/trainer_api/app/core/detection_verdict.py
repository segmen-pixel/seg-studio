# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""One definition of detected / missed / over-detected, for every caller.

The report already asked "did the model find this defect": for each connected
component in the ground truth, the fraction of it the prediction covered, and
a hit at >= 0.5. It never asked the mirror question -- a predicted blob that
lands where no defect is -- so over-detection had no definition anywhere.

The results tab needs both, and it needs them to move with the confidence
slider. Re-thresholding and re-labelling every image on every slider tick is
not affordable, so components are cut once, at zero confidence, and each one
carries a 256-bin histogram of the prediction confidence over its pixels. Area
and overlap at any threshold are then suffix sums over those bins: exact, and
O(bins) instead of O(pixels).

The rates this reports are the familiar ones by another name: match_rate is
pixel recall over the annotated area, and over_rate is one minus pixel
precision. useResultsState.ts computes the same two, per class, for the one
image on screen, to fill the "vs GT (annotate)" table -- it has the pixel
buffers in hand there and needs no round trip. This module is the authority
for the badge and for anything that has to answer for images the browser has
not loaded, which is why it does not read from there. The two agree today;
if the definition here changes, that table is what to check against.

The one thing this does not model is a component SPLITTING as the threshold
rises -- a dumbbell whose bridge fades before its ends do stays one component
here. Survival and area stay exact; only the count of a splitting blob is
conservative (one, not two). Modelling splits means re-labelling per threshold,
which is the cost this design exists to avoid.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

#: Confidence is stored as uint8 in ``{item}.confidence.png``; one bin each.
CONF_BINS = 256

#: Fraction of a component that has to line up with its counterpart. Matches
#: the instance-recall threshold the report has always used.
DEFAULT_COVERAGE = 0.5


#: Sits next to ``{item}.score.json`` in the predictions directory.
SUMMARY_SUFFIX = ".components.json"


def summary_path(pred_dir: Path, item_id: str) -> Path:
    return pred_dir / f"{item_id}{SUMMARY_SUFFIX}"


def write_item_summary(
    pred_dir: Path,
    item_id: str,
    project_id: str,
    pred_mask: np.ndarray,
    conf_u8: np.ndarray | None,
    fg_class_ids: list[int],
) -> Path | None:
    """Cut the components for one prediction and store them beside it.

    Returns None when there is nothing to compare against, or when the stored
    annotation does not line up with the prediction -- a verdict from
    mismatched shapes would be worse than no verdict.
    """
    from .paths import annotate_masks_dir

    gt_path = annotate_masks_dir(project_id) / f"{item_id}.png"
    gt_mask = None
    if gt_path.exists():
        try:
            import cv2

            from segcore.image_io import imread as _seg_imread  # non-ASCII ids on Windows
            gt_mask = _seg_imread(gt_path, cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            _logger.warning("could not read the annotation for %s: %s", item_id, exc)
            gt_mask = None
    if gt_mask is not None:
        if gt_mask.ndim > 2:
            gt_mask = gt_mask[:, :, 0]
        if gt_mask.shape != pred_mask.shape:
            _logger.warning(
                "annotation %s is %s but the prediction is %s; no verdict",
                item_id, gt_mask.shape, pred_mask.shape,
            )
            gt_mask = None

    summary = build_component_summary(gt_mask, pred_mask, conf_u8, fg_class_ids)
    path = summary_path(pred_dir, item_id)
    path.write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
    return path


def ensure_item_summary(
    pred_dir: Path,
    item_id: str,
    project_id: str,
    pred_png: Path,
    conf_png: Path,
    fg_class_ids: list[int],
) -> bool:
    """Build the summary from artifacts already on disk, if it is missing.

    Predictions written before this existed have a mask and a confidence map
    and no components, and the serving path returns them straight from cache
    without ever reaching the code that writes one -- so those images could
    never earn a verdict, however many times inference was re-run. Reading the
    two PNGs back costs far less than predicting again.

    Returns True when a summary was written.
    """
    if summary_path(pred_dir, item_id).exists():
        return False
    try:
        import cv2

        from segcore.image_io import imread as _seg_imread  # non-ASCII ids on Windows
        pred_mask = _seg_imread(pred_png, cv2.IMREAD_UNCHANGED)
        conf_u8 = _seg_imread(conf_png, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        _logger.warning("could not reread the prediction for %s: %s", item_id, exc)
        return False
    if pred_mask is None:
        return False
    if pred_mask.ndim > 2:
        pred_mask = pred_mask[:, :, 0]
    if conf_u8 is not None and conf_u8.ndim > 2:
        conf_u8 = conf_u8[:, :, 0]
    if conf_u8 is not None and conf_u8.shape != pred_mask.shape:
        conf_u8 = None
    write_item_summary(pred_dir, item_id, project_id, pred_mask, conf_u8, fg_class_ids)
    return True


def load_summary(pred_dir: Path, item_id: str) -> dict[str, Any] | None:
    """Read one summary from disk. Unprepared; see load_prepared_summary."""
    path = summary_path(pred_dir, item_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


#: Prepared summaries keyed by path, validated against the file's mtime and
#: size. Judging a whole run re-reads every summary, and the slider asks again
#: on every settle -- without this, moving it re-parses the entire run's JSON.
_PREPARED_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_PREPARED_CACHE_LOCK = threading.Lock()
#: Bounded so a long session over many runs cannot grow without limit. Cleared
#: wholesale rather than by LRU: refilling is one read per image and the cost
#: only lands on the first slider move after it.
_PREPARED_CACHE_MAX = 4096


def load_prepared_summary(pred_dir: Path, item_id: str) -> dict[str, Any] | None:
    path = summary_path(pred_dir, item_id)
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    stamp = (st.st_mtime, st.st_size)
    with _PREPARED_CACHE_LOCK:
        hit = _PREPARED_CACHE.get(key)
        if hit is not None and (hit[0], hit[1]) == stamp:
            return hit[2]
    raw = load_summary(pred_dir, item_id)
    if raw is None:
        return None
    prepared = prepare_summary(raw)
    with _PREPARED_CACHE_LOCK:
        if len(_PREPARED_CACHE) >= _PREPARED_CACHE_MAX:
            _PREPARED_CACHE.clear()
        _PREPARED_CACHE[key] = (stamp[0], stamp[1], prepared)
    return prepared


def conf_bin_for_percent(pct: float) -> int:
    """Map the UI's 0-100 slider onto a confidence bin.

    The slider is applied as ``pct / 100`` against a float confidence that was
    already quantised to uint8 on the way to disk, so this rounds the same way
    the stored artifact did.
    """
    pct = max(0.0, min(100.0, float(pct)))
    return int(round(pct / 100.0 * (CONF_BINS - 1)))


def _histogram(conf_values: np.ndarray) -> list[int]:
    counts = np.bincount(conf_values.ravel(), minlength=CONF_BINS)
    return [int(v) for v in counts[:CONF_BINS]]


def _suffix(counts: list[int], lo: int, prepared: bool) -> int:
    """Pixels whose confidence bin is >= lo.

    On a raw histogram this sums a slice, which copies up to CONF_BINS ints
    per component per query -- and the slider asks again on every tick. A
    prepared summary carries the cumulative-from-the-top table instead, so
    the same answer is one index.
    """
    if prepared:
        return counts[lo] if 0 <= lo < len(counts) else 0
    if lo <= 0:
        return sum(counts)
    return sum(counts[lo:])


def _suffix_table(hist: list[int]) -> list[int]:
    """Cumulative counts from the top bin down; ``table[b]`` is "bin >= b"."""
    table = [0] * (CONF_BINS + 1)
    run = 0
    for b in range(CONF_BINS - 1, -1, -1):
        run += hist[b] if b < len(hist) else 0
        table[b] = run
    return table


def prepare_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Trade a little memory for a threshold query that does no arithmetic.

    The stored form stays a plain histogram: it is smaller on disk and it is
    what a human reading the file would expect. This is the in-memory shape.
    """
    if summary.get("prepared"):
        return summary
    out: dict[str, Any] = {"conf_bins": summary.get("conf_bins", CONF_BINS), "prepared": True}
    gt = summary.get("gt")
    out["gt"] = None if gt is None else [
        {"class_id": g["class_id"], "area": g["area"], "hit_hist": _suffix_table(g["hit_hist"])}
        for g in gt
    ]
    out["pred"] = [
        {
            "class_id": c["class_id"],
            "hist": _suffix_table(c["hist"]),
            "on_gt_hist": None if c.get("on_gt_hist") is None else _suffix_table(c["on_gt_hist"]),
        }
        for c in (summary.get("pred") or [])
    ]
    return out


def build_component_summary(
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    conf_u8: np.ndarray | None,
    fg_class_ids: list[int],
) -> dict[str, Any]:
    """Cut both sides into components once and record their confidence.

    Args:
        gt_mask: Ground-truth class-id mask, or None when the image is not
            annotated -- then no verdict is possible and only the predicted
            side is recorded.
        pred_mask: Predicted class-id mask (argmax), same shape.
        conf_u8: Per-pixel foreground confidence, uint8. None means "treat
            every predicted pixel as fully confident", which is what the
            report wants: it scores the shipped mask, not a slider position.
        fg_class_ids: Active class ids. Background (0) is dropped here rather
            than trusted to each caller -- it is active in every project and
            arrives as one enormous component that the model also predicts,
            a guaranteed hit on every image.

    Returns:
        A JSON-serialisable summary. ``gt`` is None for an unannotated image.
    """
    ids = [int(c) for c in fg_class_ids if int(c) != 0]
    if conf_u8 is None:
        conf_u8 = np.full(pred_mask.shape, CONF_BINS - 1, dtype=np.uint8)

    from scipy.ndimage import label as ndimage_label

    gt_components: list[dict[str, Any]] | None = None if gt_mask is None else []
    pred_components: list[dict[str, Any]] = []

    for cid in ids:
        pred_binary = pred_mask == cid
        gt_binary = None if gt_mask is None else (gt_mask == cid)

        if gt_binary is not None and gt_binary.any():
            labeled, n = ndimage_label(gt_binary)
            for inst in range(1, n + 1):
                inst_mask = labeled == inst
                # How much of THIS instance the model covered. The denominator
                # is the instance's own area; predictions elsewhere in the
                # image must not enter it, or finding more defects would push
                # the others into "missed".
                covered = inst_mask & pred_binary
                gt_components.append({
                    "class_id": cid,
                    "area": int(inst_mask.sum()),
                    # Confidence over the covered part only: the rest of the
                    # instance was not predicted as this class at any threshold.
                    "hit_hist": _histogram(conf_u8[covered]),
                })

        if pred_binary.any():
            labeled, n = ndimage_label(pred_binary)
            for comp in range(1, n + 1):
                comp_mask = labeled == comp
                on_gt = comp_mask if gt_binary is None else (comp_mask & gt_binary)
                pred_components.append({
                    "class_id": cid,
                    "hist": _histogram(conf_u8[comp_mask]),
                    # Of this component's pixels, the ones that sit on ground
                    # truth of the same class. None when there is no GT to
                    # compare against.
                    "on_gt_hist": None if gt_binary is None else _histogram(conf_u8[on_gt]),
                })

    return {
        "conf_bins": CONF_BINS,
        "gt": gt_components,
        "pred": pred_components,
    }


def instance_coverages(
    summary: dict[str, Any], *, min_conf_bin: int = 0,
) -> list[dict[str, Any]]:
    """Per ground-truth instance: its area and the fraction covered.

    The report needs the individual numbers to list its worst misses; the
    verdict only needs the counts. Both read the same components.
    """
    prepared = bool(summary.get("prepared"))
    out: list[dict[str, Any]] = []
    for inst in summary.get("gt") or []:
        area = int(inst.get("area") or 0)
        if area <= 0:
            continue
        hit = _suffix(inst["hit_hist"], min_conf_bin, prepared)
        out.append({
            "class_id": int(inst["class_id"]),
            "area": area,
            "coverage": hit / area,
        })
    return out


def evaluate_summary(
    summary: dict[str, Any],
    *,
    min_conf_bin: int = 0,
    coverage: float = DEFAULT_COVERAGE,
    min_area: int = 0,
    max_area: int = 0,
) -> dict[str, Any]:
    """Count detected / missed / over-detected at one slider position.

    Args:
        summary: From build_component_summary.
        min_conf_bin: Confidence bin the slider sits on; pixels below it are
            background.
        coverage: Fraction of a component that has to line up.
        min_area: Drop predicted components smaller than this, as the results
            tab's area post-filter does. 0 disables.
        max_area: Drop predicted components larger than this. 0 disables.

    Returns:
        ``verdict`` is a single label, ordered by what a reviewer has to act
        on first: a missed defect outranks a false alarm, which outranks a
        clean hit. ``None`` when the image has no ground truth, because
        nothing here can be judged then.
    """
    prepared = bool(summary.get("prepared"))
    gt = summary.get("gt")
    detected = 0
    missed = 0
    over = 0
    gt_area = 0
    gt_covered = 0
    pred_area = 0
    pred_off_gt = 0

    if gt is not None:
        for inst in gt:
            area = int(inst.get("area") or 0)
            if area <= 0:
                continue
            hit = _suffix(inst["hit_hist"], min_conf_bin, prepared)
            gt_area += area
            gt_covered += hit
            if hit / area >= coverage:
                detected += 1
            else:
                missed += 1

    for comp in summary.get("pred") or []:
        area = _suffix(comp["hist"], min_conf_bin, prepared)
        if area <= 0:
            continue  # nothing of this component survives the threshold
        if min_area > 0 and area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        on_gt_hist = comp.get("on_gt_hist")
        if on_gt_hist is None:
            continue  # no ground truth: nothing to call it wrong against
        on_gt = _suffix(on_gt_hist, min_conf_bin, prepared)
        pred_area += area
        pred_off_gt += area - on_gt
        if on_gt / area < coverage:
            over += 1

    if gt is None:
        verdict = None
    elif missed:
        verdict = "missed"
    elif over:
        verdict = "over"
    elif detected:
        verdict = "detected"
    else:
        # No defects annotated and none predicted: a good part, correctly
        # passed. Calling that "detected" would read as a found defect.
        verdict = "clean"

    return {
        "verdict": verdict,
        "detected": detected,
        "missed": missed,
        "over": over,
        # Area rates rather than instance counts, because counts overstate.
        # One stray pixel left behind while annotating is a whole "missed
        # instance" and flips the image's label, but it is one pixel in a
        # thousand by area -- and a defect the model covered 90% of reads as
        # missed exactly like one it never touched.
        "gt_area": gt_area,
        "gt_covered": gt_covered,
        "match_rate": (gt_covered / gt_area) if gt_area else None,
        # Area match alone cannot see a false alarm: predicting the whole
        # frame would score 100%. This is the other half.
        "pred_area": pred_area,
        "pred_off_gt": pred_off_gt,
        "over_rate": (pred_off_gt / pred_area) if pred_area else None,
    }
