# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Evaluation loops: full-image loader, sliding-window and per-image metrics.

Core metric math lives in metrics_core, calibration in metrics_calibration
and threshold search in metrics_threshold; their names are re-exported here
for backward compatibility.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint_adapter import _resolve_active_class_ids, _suppress_inactive_logits

# ---------------------------------------------------------------------------
# Core metric math, calibration and threshold search live in dedicated
# modules since the pre-OSS refactor; the names are re-exported here so
# existing importers (train.py, trainer_api, scripts, tests) stay unchanged.
# ---------------------------------------------------------------------------
from .metrics_calibration import (  # noqa: F401
    _CAL_N_BINS,
    accumulate_calibration_bins,
    compute_ece,
    draw_reliability_diagram,
)
from .metrics_core import (  # noqa: F401
    accumulate_confusion_matrix,
    accumulate_f1_stats,
    compute_miou,
    finalize_f1,
    finalize_metrics,
)
from .metrics_threshold import (  # noqa: F401
    THRESHOLD_CANDIDATES,
    build_f1_curve,
    find_optimal_threshold,
)
from .prediction_rules import prediction_from_probs
from .split_utils import _find_by_stem


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    ignore_index: int,
    include_background: bool = False,
    active_class_ids: list[int] | None = None,
    compute_confusion: bool = False,
    threshold_search: bool = False,
) -> tuple[float, float, dict, dict, dict, dict, np.ndarray | None, dict | None]:
    """Evaluate model on a data loader.

    When *threshold_search* is True an additional pass over candidate FG
    thresholds is performed on the softmax probabilities and the optimal
    threshold / F1 is returned in the 8th element (dict with keys
    ``optimal_threshold``, ``optimal_threshold_f1``, and calibration stats).
    """
    if len(loader) == 0:
        return 0.0, 0.0, {}, {}, {}, {}, None, None
    miou_list: list[float] = []
    total_tp = np.zeros(num_classes, dtype="float64")
    total_fp = np.zeros(num_classes, dtype="float64")
    total_fn = np.zeros(num_classes, dtype="float64")
    total_cm: np.ndarray | None = np.zeros((num_classes, num_classes), dtype="float64") if compute_confusion else None

    # Per-threshold accumulators (only when threshold_search is enabled)
    thresh_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    if threshold_search:
        thresh_stats = {
            t: (np.zeros(num_classes, "float64"), np.zeros(num_classes, "float64"), np.zeros(num_classes, "float64"))
            for t in THRESHOLD_CANDIDATES
        }
    # Calibration bins (collected alongside threshold search)
    cal_correct = np.zeros(_CAL_N_BINS, dtype="float64")
    cal_confidence = np.zeros(_CAL_N_BINS, dtype="float64")
    cal_count = np.zeros(_CAL_N_BINS, dtype="float64")

    with torch.no_grad():
        device = next(model.parameters()).device
        resolved_active_ids = _resolve_active_class_ids(num_classes, active_class_ids)
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            masks = batch[1].to(device, non_blocking=True)
            logits = _suppress_inactive_logits(model(images), resolved_active_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            # Compute softmax probs for threshold search and calibration
            probs_np = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(preds.shape[0]):
                pred = preds[i]
                tgt = targets[i]
                prob_i = probs_np[i]  # (C, H, W)
                miou_list.append(
                    compute_miou(pred, tgt, num_classes, ignore_index, include_background=include_background)
                )
                tp, fp, fn = accumulate_f1_stats(pred, tgt, num_classes, ignore_index, include_background)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if total_cm is not None:
                    total_cm += accumulate_confusion_matrix(pred, tgt, num_classes, ignore_index)
                # Threshold search accumulation
                if thresh_stats is not None:
                    for t, (t_tp, t_fp, t_fn) in thresh_stats.items():
                        pred_t = prediction_from_probs(prob_i, fg_threshold=t)
                        s_tp, s_fp, s_fn = accumulate_f1_stats(pred_t, tgt, num_classes, ignore_index, include_background)
                        t_tp += s_tp
                        t_fp += s_fp
                        t_fn += s_fn
                # Calibration accumulation
                bc, bf, bn = accumulate_calibration_bins(prob_i, tgt, ignore_index)
                cal_correct += bc
                cal_confidence += bf
                cal_count += bn

    miou = float(np.mean(miou_list)) if miou_list else 0.0
    f1_macro, per_class_f1, per_class_precision, per_class_recall, per_class_iou = finalize_metrics(
        total_tp, total_fp, total_fn, num_classes, ignore_index, include_background
    )
    threshold_info: dict | None = None
    if thresh_stats is not None:
        best_t, best_f1 = find_optimal_threshold(thresh_stats, num_classes, ignore_index)
        f1_curve = build_f1_curve(thresh_stats, num_classes, ignore_index)
        ece = compute_ece(cal_correct, cal_confidence, cal_count)
        threshold_info = {
            "optimal_threshold": best_t,
            "optimal_threshold_f1": best_f1,
            "f1_curve": f1_curve,
            "ece": ece,
            "cal_bins": (cal_correct, cal_confidence, cal_count),
        }
    else:
        # Still return calibration even without threshold search
        ece = compute_ece(cal_correct, cal_confidence, cal_count)
        threshold_info = {
            "ece": ece,
            "cal_bins": (cal_correct, cal_confidence, cal_count),
        }
    return miou, f1_macro, per_class_f1, per_class_precision, per_class_recall, per_class_iou, total_cm, threshold_info


def compute_per_image_metrics(
    model: nn.Module,
    dataset,
    num_classes: int,
    ignore_index: int,
    device,
    include_background: bool = False,
    active_class_ids: list[int] | None = None,
    batch_size: int = 4,
    save_predictions_dir=None,
) -> dict[str, dict]:
    """Evaluate ``model`` on every item in ``dataset`` and return a per-image
    metric dict keyed by image stem (dataset.split_ids[stem_idx]).

    Each value is
    ``{"per_class": {cls: {tp, fp, fn, f1, prec, rec, iou}},
       "macro_f1": ..., "macro_prec": ..., "macro_rec": ..., "macro_iou": ...}``.
    ``macro_*`` averages across the foreground classes actually present in the
    image; classes absent from both prediction and target are skipped so a
    clean-image doesn\'t drag the average to zero.

    Used by the iterative hard-mining loop: `train.py` calls this once at the
    end of training on the best model, dumps the result to
    ``per_image_metrics.json``, and picks the bottom-N images (by prec or rec)
    as the hard set for the next iteration.

    When ``save_predictions_dir`` is passed, the same forward pass also
    materialises the ``.png / .confidence.png / .probs.npy / .score.json``
    quartet the UI expects, so each iterative run leaves a populated
    predictions/ directory without the user having to click 推論実行 on
    every chained iteration.
    """
    import torch
    from torch.utils.data import DataLoader
    prior_return_meta = getattr(dataset, "return_meta", False)
    dataset.return_meta = True
    try:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            drop_last=False,
        )
        resolved_active_ids = _resolve_active_class_ids(num_classes, active_class_ids)
        per_image: dict[str, dict] = {}
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device, non_blocking=True)
                masks = batch[1].to(device, non_blocking=True)
                metas = batch[2]
                logits = _suppress_inactive_logits(model(images), resolved_active_ids)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C, H, W)
                targets = masks.cpu().numpy()
                # metas is a dict of tensors when return_meta=True
                stem_idxs = metas["stem_idx"] if isinstance(metas, dict) else metas
                for i in range(preds.shape[0]):
                    stem_idx = int(stem_idxs[i]) if hasattr(stem_idxs, "__getitem__") else int(stem_idxs)
                    stem = dataset.split_ids[stem_idx]
                    tp, fp, fn = accumulate_f1_stats(
                        preds[i], targets[i], num_classes, ignore_index, include_background,
                    )
                    target_i = targets[i]
                    probs_i = probs[i]  # (C, H, W)
                    per_class: dict[str, dict] = {}
                    f1s: list[float] = []
                    precs: list[float] = []
                    recs: list[float] = []
                    ious: list[float] = []
                    fg_confs_at_gt: list[float] = []
                    for cls in range(num_classes):
                        if cls == ignore_index:
                            continue
                        if not include_background and cls == 0:
                            continue
                        denom = tp[cls] + fp[cls] + fn[cls]
                        if denom == 0:
                            continue
                        pd_ = tp[cls] + fp[cls]
                        rd_ = tp[cls] + fn[cls]
                        prec = float(tp[cls] / pd_) if pd_ > 0 else 0.0
                        rec = float(tp[cls] / rd_) if rd_ > 0 else 0.0
                        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                        iou = float(tp[cls] / denom)
                        # Mean prob[cls] over pixels where the GT is cls —
                        # how confidently the model votes for the correct
                        # class on true-foreground pixels. Only computed
                        # when the class is actually present in the GT.
                        _gt_mask = target_i == cls
                        _n_gt = int(_gt_mask.sum())
                        if _n_gt > 0:
                            fg_conf_at_gt = float(probs_i[cls][_gt_mask].mean())
                        else:
                            fg_conf_at_gt = 0.0
                        per_class[str(int(cls))] = {
                            "tp": int(tp[cls]),
                            "fp": int(fp[cls]),
                            "fn": int(fn[cls]),
                            "f1": f1,
                            "prec": prec,
                            "rec": rec,
                            "iou": iou,
                            "fg_conf_at_gt": fg_conf_at_gt,
                            "n_gt_pixels": _n_gt,
                        }
                        f1s.append(f1)
                        precs.append(prec)
                        recs.append(rec)
                        ious.append(iou)
                        if _n_gt > 0:
                            fg_confs_at_gt.append(fg_conf_at_gt)
                    per_image[stem] = {
                        "per_class": per_class,
                        "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
                        "macro_prec": float(sum(precs) / len(precs)) if precs else 0.0,
                        "macro_rec": float(sum(recs) / len(recs)) if recs else 0.0,
                        "macro_iou": float(sum(ious) / len(ious)) if ious else 0.0,
                        # None when the image has no GT foreground for any
                        # tracked class (a clean image), so downstream sort
                        # can distinguish "no data" from "low confidence".
                        "mean_fg_conf_at_gt": (float(sum(fg_confs_at_gt) / len(fg_confs_at_gt))
                                                if fg_confs_at_gt else None),
                    }
                    if save_predictions_dir is not None:
                        try:
                            import json as _pred_json

                            from PIL import Image as _PILImage
                            pred_root = save_predictions_dir
                            pred_root.mkdir(parents=True, exist_ok=True)
                            pred_i = preds[i].astype("uint8")
                            probs_i_arr = probs_i  # (C, H, W)
                            conf_i = probs_i_arr.max(axis=0)  # (H, W)
                            _PILImage.fromarray(pred_i, mode="L").save(pred_root / f"{stem}.png")
                            _PILImage.fromarray(
                                (conf_i * 255).clip(0, 255).astype("uint8"), mode="L"
                            ).save(pred_root / f"{stem}.confidence.png")
                            np.save(pred_root / f"{stem}.probs.npy", probs_i_arr.astype("float16"))
                            fg_mask_i = pred_i > 0
                            n_fg = int(fg_mask_i.sum())
                            n_bg = int((~fg_mask_i).sum())
                            fg_mean_c = float(conf_i[fg_mask_i].mean()) if n_fg > 0 else 0.0
                            bg_mean_c = float(conf_i[~fg_mask_i].mean()) if n_bg > 0 else 0.0
                            per_cls_conf = {}
                            for _cls in range(num_classes):
                                if _cls == ignore_index:
                                    continue
                                _cm = pred_i == _cls
                                if _cm.any():
                                    per_cls_conf[str(int(_cls))] = float(probs_i_arr[_cls][_cm].mean())
                            score = {
                                "artifact_version": 4,
                                "backend": "torch",
                                "item_id": stem,
                                "inference_input_size": [int(pred_i.shape[1]), int(pred_i.shape[0])],
                                "inference_ms": 0.0,
                                "inference_device": str(device),
                                "mean_confidence": float(conf_i.mean()),
                                "foreground_mean_confidence": fg_mean_c,
                                "background_mean_confidence": bg_mean_c,
                                "foreground_ratio": float(fg_mask_i.mean()),
                                "max_confidence": float(conf_i.max()),
                                "min_confidence": float(conf_i.min()),
                                "per_class_mean_confidence": per_cls_conf,
                                "origin": "iterative_auto",
                            }
                            (pred_root / f"{stem}.score.json").write_text(
                                _pred_json.dumps(score, indent=2), encoding="utf-8"
                            )
                        except Exception as _pred_err:
                            # Predictions are an auxiliary side-effect; if the
                            # write fails (e.g. disk full mid-batch) we still
                            # want per_image_metrics to land, so we log and go on.
                            import logging as _log_pred
                            _log_pred.getLogger(__name__).warning(
                                "iterative predictions save failed for %s: %s", stem, _pred_err
                            )
        return per_image
    finally:
        dataset.return_meta = prior_return_meta


def compute_per_image_metrics_sw(
    model: nn.Module,
    images_dir,
    masks_dir,
    split_map: dict,
    patch_size: int,
    sw_stride: int,
    num_classes: int,
    output_stride: int,
    ignore_index: int,
    normalize: dict,
    device,
    include_background: bool = False,
    active_class_ids: list[int] | None = None,
    fg_threshold: float | None = None,
    save_predictions_dir=None,
    log_fn=None,
) -> dict[str, dict]:
    """Per-image metrics via sliding-window inference at native resolution.

    Same return shape as :func:`compute_per_image_metrics`, plus a
    ``"split"`` key taken from *split_map* (stem -> "train"|"val"|"test").

    The DataLoader-based sibling feeds the model whatever the dataset
    serves: under sliding-window validation the val/train_eval datasets
    are None so the hook fell back to train_ds — random augmented
    patches — and the val split was never evaluated at all, while
    test_ds resized whole images down to input_size. All three disagree
    with the production prediction engine, which runs patch/stride
    sliding-window at native resolution. This variant runs that same SW
    inference, so hard-mining picks, iter-stop macros and the auto-saved
    predictions/ artifacts all describe what 推論実行 actually shows.

    ``fg_threshold`` should be the run's resolved inference threshold so
    the argmax->background suppression matches the engine default view.
    """
    from pathlib import Path as _Path

    import cv2
    import torch

    from segcore.image_io import imread as _imread

    from .prediction_rules import prediction_from_probs
    from .sliding_window import sliding_window_predict

    _log = log_fn if log_fn is not None else (lambda _m: None)
    images_dir = _Path(images_dir)
    masks_dir = _Path(masks_dir)
    if not isinstance(device, torch.device):
        device = torch.device(device)
    resolved_active_ids = _resolve_active_class_ids(num_classes, active_class_ids)
    per_image: dict[str, dict] = {}
    model.eval()
    _n_done = 0
    for stem, split_name in split_map.items():
        img_path = None
        for ext in (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        img = _imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        gt = None
        for ext in (".png", ".bmp", ".tiff"):
            p = masks_dir / f"{stem}{ext}"
            if p.exists():
                gt = _imread(str(p), cv2.IMREAD_GRAYSCALE)
                break
        if gt is None:
            gt = np.zeros((orig_h, orig_w), dtype="uint8")

        with torch.no_grad():
            _, probs = sliding_window_predict(
                model, img, patch_size, sw_stride,
                num_classes, output_stride, normalize,
                active_class_ids=resolved_active_ids,
                device=device,
            )
        # probs: (C, H/os, W/os). Threshold like the prediction engine so
        # the metric describes the default UI view of this run.
        pred_small = prediction_from_probs(probs, fg_threshold=fg_threshold)
        pred_full = cv2.resize(
            pred_small.astype("uint8"), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype("int64")

        gt_i = gt.astype("int64")
        tp, fp, fn = accumulate_f1_stats(
            pred_full, gt_i, num_classes, ignore_index, include_background,
        )
        per_class: dict[str, dict] = {}
        f1s: list[float] = []
        precs: list[float] = []
        recs: list[float] = []
        ious: list[float] = []
        fg_confs_at_gt: list[float] = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            if not include_background and cls == 0:
                continue
            denom = tp[cls] + fp[cls] + fn[cls]
            if denom == 0:
                continue
            pd_ = tp[cls] + fp[cls]
            rd_ = tp[cls] + fn[cls]
            prec = float(tp[cls] / pd_) if pd_ > 0 else 0.0
            rec = float(tp[cls] / rd_) if rd_ > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            iou = float(tp[cls] / denom)
            _gt_mask = gt_i == cls
            _n_gt = int(_gt_mask.sum())
            if _n_gt > 0:
                conf_cls_full = cv2.resize(
                    probs[cls].astype("float32"), (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                fg_conf_at_gt = float(conf_cls_full[_gt_mask].mean())
            else:
                fg_conf_at_gt = 0.0
            per_class[str(int(cls))] = {
                "tp": int(tp[cls]),
                "fp": int(fp[cls]),
                "fn": int(fn[cls]),
                "f1": f1,
                "prec": prec,
                "rec": rec,
                "iou": iou,
                "fg_conf_at_gt": fg_conf_at_gt,
                "n_gt_pixels": _n_gt,
            }
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)
            ious.append(iou)
            if _n_gt > 0:
                fg_confs_at_gt.append(fg_conf_at_gt)
        # Confidence-weighted damage masses for hard-mining priority.
        # fp_conf_mass sums how far ABOVE the threshold background pixels
        # scored (confident ghosts survive threshold raises — the most
        # dangerous FPs); fn_conf_mass sums how far BELOW it GT pixels
        # scored (deep misses need training the most). Raw pixel counts
        # rank a barely-over-threshold blob above a smaller confident one.
        _thr_m = float(fg_threshold) if fg_threshold is not None else 0.5
        fg_prob_small = probs[1:].sum(axis=0)
        gt_small = cv2.resize(
            ((gt_i > 0) & (gt_i != ignore_index)).astype("uint8"),
            (fg_prob_small.shape[1], fg_prob_small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        pred_fg_small = pred_small > 0
        fp_conf_mass = float(np.maximum(fg_prob_small - _thr_m, 0.0)[pred_fg_small & ~gt_small].sum())
        fn_conf_mass = float(np.maximum(_thr_m - fg_prob_small, 0.0)[~pred_fg_small & gt_small].sum())
        per_image[stem] = {
            "per_class": per_class,
            "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "macro_prec": float(sum(precs) / len(precs)) if precs else 0.0,
            "macro_rec": float(sum(recs) / len(recs)) if recs else 0.0,
            "macro_iou": float(sum(ious) / len(ious)) if ious else 0.0,
            "mean_fg_conf_at_gt": (float(sum(fg_confs_at_gt) / len(fg_confs_at_gt))
                                    if fg_confs_at_gt else None),
            "fp_conf_mass": fp_conf_mass,
            "fn_conf_mass": fn_conf_mass,
            "split": split_name,
        }

        if save_predictions_dir is not None:
            try:
                import json as _pred_json
                pred_root = _Path(save_predictions_dir)
                pred_root.mkdir(parents=True, exist_ok=True)
                pred_u8 = pred_full.astype("uint8")
                # Same confidence semantics as the prediction engine:
                # summed foreground probability, not max-class prob.
                conf_small = probs[1:, :, :].sum(axis=0).astype("float32")
                conf_full = cv2.resize(
                    conf_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
                )
                conf_full = np.clip(conf_full, 0.0, 1.0)
                _png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
                ok, pred_buf = cv2.imencode(".png", pred_u8, _png_params)
                (pred_root / f"{stem}.png").write_bytes(pred_buf.tobytes())
                conf_u8 = np.clip(conf_full * 255.0, 0, 255).astype("uint8")
                ok, conf_buf = cv2.imencode(".png", conf_u8, _png_params)
                (pred_root / f"{stem}.confidence.png").write_bytes(conf_buf.tobytes())
                np.save(pred_root / f"{stem}.probs.npy", probs.astype("float16"))
                fg_mask_i = pred_u8 > 0
                n_fg = int(fg_mask_i.sum())
                n_bg = int((~fg_mask_i).sum())
                fg_mean_c = float(conf_full[fg_mask_i].mean()) if n_fg > 0 else 0.0
                bg_mean_c = float(conf_full[~fg_mask_i].mean()) if n_bg > 0 else 0.0
                per_cls_conf = {}
                for _cls in range(num_classes):
                    if _cls == ignore_index:
                        continue
                    _cm = pred_u8 == _cls
                    if _cm.any():
                        per_cls_conf[str(int(_cls))] = float(conf_full[_cm].mean())
                score = {
                    "artifact_version": 4,
                    "backend": "torch",
                    "item_id": stem,
                    "inference_input_size": [int(orig_w), int(orig_h)],
                    "inference_ms": 0.0,
                    "inference_device": str(device),
                    "mean_confidence": float(conf_full.mean()),
                    "foreground_mean_confidence": fg_mean_c,
                    "background_mean_confidence": bg_mean_c,
                    "foreground_ratio": float(fg_mask_i.mean()),
                    "max_confidence": float(conf_full.max()),
                    "min_confidence": float(conf_full.min()),
                    "per_class_mean_confidence": per_cls_conf,
                    "origin": "iterative_auto",
                    "sliding_window": {"patch_size": int(patch_size), "stride": int(sw_stride)},
                }
                (pred_root / f"{stem}.score.json").write_text(
                    _pred_json.dumps(score, indent=2), encoding="utf-8"
                )
            except Exception as _pred_err:
                import logging as _log_pred
                _log_pred.getLogger(__name__).warning(
                    "iterative SW predictions save failed for %s: %s", stem, _pred_err
                )
        _n_done += 1
    _log(f"Per-image SW eval: {_n_done}/{len(split_map)} images (patch={patch_size}, stride={sw_stride})\n")
    return per_image



def evaluate_sliding_window(
    model: nn.Module,
    images_dir: Path,
    masks_dir: Path,
    split_ids: list[str],
    patch_size: int,
    sw_stride: int,
    num_classes: int,
    output_stride: int,
    ignore_index: int,
    normalize: dict,
    include_background: bool = False,
    active_class_ids: list[int] | None = None,
    compute_confusion: bool = False,
    stop_flag: Callable[[], bool] | None = None,
    relabel_ignore_as_bg: bool = False,
    threshold_search: bool = False,
) -> tuple:
    """Evaluate with sliding-window inference at original resolution.

    When *threshold_search* is True, candidate FG thresholds are evaluated
    and the optimal threshold / F1 is returned in the 8th element (dict with
    keys ``optimal_threshold`` and ``optimal_threshold_f1``).
    """
    from .sliding_window import sliding_window_predict

    if not split_ids:
        return 0.0, 0.0, {}, {}, {}, {}, None, None

    miou_list: list[float] = []
    total_tp = np.zeros(num_classes, dtype="float64")
    total_fp = np.zeros(num_classes, dtype="float64")
    total_fn = np.zeros(num_classes, dtype="float64")
    total_cm: np.ndarray | None = (
        np.zeros((num_classes, num_classes), dtype="float64") if compute_confusion else None
    )

    # Per-threshold accumulators (only when threshold_search is enabled)
    thresh_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    if threshold_search:
        thresh_stats = {
            t: (np.zeros(num_classes, "float64"), np.zeros(num_classes, "float64"), np.zeros(num_classes, "float64"))
            for t in THRESHOLD_CANDIDATES
        }
    # Calibration bins
    cal_correct = np.zeros(_CAL_N_BINS, dtype="float64")
    cal_confidence = np.zeros(_CAL_N_BINS, dtype="float64")
    cal_count = np.zeros(_CAL_N_BINS, dtype="float64")

    device = next(model.parameters()).device

    # --- Background prefetcher ------------------------------------------
    # The original loop serialised image I/O (open + decode + mask resize)
    # with GPU compute (sliding-window forward). On large val sets (~800
    # images at 1800×1500) that pinned the GPU at 0% util during disk
    # reads. Run the I/O in a daemon thread and hand the main loop a
    # queue of already-decoded (stem, img_np, tgt) tuples. PIL's decode
    # releases the GIL, so the thread makes real progress while torch
    # kernels run on the main thread.
    #
    # Ported from feature/large-dataset commit e6f3909 (one of the good
    # algorithmic ideas from that experiment).
    import queue as _queue
    import threading as _threading

    _STOP = object()
    _prefetch: _queue.Queue = _queue.Queue(maxsize=2)
    _prefetch_should_stop = _threading.Event()

    def _prefetch_worker() -> None:
        for _pstem in split_ids:
            if _prefetch_should_stop.is_set():
                break
            if stop_flag and stop_flag():
                break
            try:
                img_path = _find_by_stem(images_dir, _pstem)
                img = Image.open(img_path).convert("RGB")
                img_np_local = np.asarray(img)
                try:
                    mask_path = _find_by_stem(masks_dir, _pstem)
                    mask_pil = Image.open(mask_path).convert("L")
                except FileNotFoundError:
                    mask_pil = Image.new("L", img.size, 0)
                H_local, W_local = img_np_local.shape[:2]
                out_w = W_local // output_stride
                out_h = H_local // output_stride
                mask_pil = mask_pil.resize((out_w, out_h), resample=Image.NEAREST)
                tgt_local = np.asarray(mask_pil).astype("int64")
                if relabel_ignore_as_bg:
                    tgt_local[tgt_local == ignore_index] = 0
                _prefetch.put((_pstem, img_np_local, tgt_local))
            except Exception as _prefetch_err:
                _prefetch.put(_prefetch_err)
                break
        _prefetch.put(_STOP)

    _prefetch_thread = _threading.Thread(target=_prefetch_worker, daemon=True)
    _prefetch_thread.start()

    with torch.no_grad():
        try:
            while True:
                if stop_flag and stop_flag():
                    _prefetch_should_stop.set()
                    break
                item = _prefetch.get()
                if item is _STOP:
                    break
                if isinstance(item, Exception):
                    _prefetch_should_stop.set()
                    raise item
                stem, img_np, tgt = item

                pred, probs = sliding_window_predict(
                    model, img_np, patch_size, sw_stride,
                    num_classes, output_stride, normalize,
                    active_class_ids=active_class_ids, device=device,
                    stop_flag=stop_flag,
                )

                miou_list.append(
                    compute_miou(pred, tgt, num_classes, ignore_index, include_background=include_background)
                )
                tp, fp, fn = accumulate_f1_stats(pred, tgt, num_classes, ignore_index, include_background)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if total_cm is not None:
                    total_cm += accumulate_confusion_matrix(pred, tgt, num_classes, ignore_index)
                # Threshold search accumulation
                if thresh_stats is not None and probs is not None:
                    for t, (t_tp, t_fp, t_fn) in thresh_stats.items():
                        pred_t = prediction_from_probs(probs, fg_threshold=t)
                        s_tp, s_fp, s_fn = accumulate_f1_stats(pred_t, tgt, num_classes, ignore_index, include_background)
                        t_tp += s_tp
                        t_fp += s_fp
                        t_fn += s_fn
                # Calibration accumulation
                if probs is not None:
                    bc, bf, bn = accumulate_calibration_bins(probs, tgt, ignore_index)
                    cal_correct += bc
                    cal_confidence += bf
                    cal_count += bn
        finally:
            _prefetch_should_stop.set()
            # Drain leftover items so the worker can exit cleanly.
            try:
                while not _prefetch.empty():
                    _prefetch.get_nowait()
            except Exception:
                pass

    miou = float(np.mean(miou_list)) if miou_list else 0.0
    f1_macro, per_class_f1, per_class_precision, per_class_recall, per_class_iou = finalize_metrics(
        total_tp, total_fp, total_fn, num_classes, ignore_index, include_background,
    )
    threshold_info: dict | None = None
    ece = compute_ece(cal_correct, cal_confidence, cal_count)
    if thresh_stats is not None:
        best_t, best_f1 = find_optimal_threshold(thresh_stats, num_classes, ignore_index)
        f1_curve = build_f1_curve(thresh_stats, num_classes, ignore_index)
        threshold_info = {
            "optimal_threshold": best_t,
            "optimal_threshold_f1": best_f1,
            "f1_curve": f1_curve,
            "ece": ece,
            "cal_bins": (cal_correct, cal_confidence, cal_count),
        }
    else:
        threshold_info = {
            "ece": ece,
            "cal_bins": (cal_correct, cal_confidence, cal_count),
        }
    return miou, f1_macro, per_class_f1, per_class_precision, per_class_recall, per_class_iou, total_cm, threshold_info
