# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Post-training finalization: per-image metrics and the iterative-mining
decision. Extracted verbatim from train() during the pre-OSS refactor.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from ..tiling_geometry import default_patch_stride
from .hard_mining import _damage_key
from .iterative_mining import _dataset_micro_prf
from .layout import PRED_DIRNAME
from .metrics import compute_per_image_metrics, compute_per_image_metrics_sw


def run_per_image_and_iterative(
    model,
    config,
    prepared_dir: Path,
    run_dir: Path,
    num_classes: int,
    device: torch.device,
    resolved_active_ids: list[int],
    val_threshold_info: dict | None,
    train_ds,
    train_eval_ds,
    val_ds,
    test_ds,
    log_fn: Callable[[str], None],
) -> None:
    """Load the best epoch's weights, run one inference pass over
    train+val(+test), and dump per_image_metrics.json. In iterative mode
    with unmet targets, also write iterative_hard_ids.json for the runner
    to fan out the next iteration. Never raises: metrics-side errors must
    not fail the training run."""
    try:
        _best_ckpt = run_dir / "model.pt"
        if _best_ckpt.exists():
            model.load_state_dict(
                torch.load(str(_best_ckpt), map_location=device, weights_only=True)
            )
            model.eval()
        _pred_root = run_dir / PRED_DIRNAME
        _per_image_all: dict = {}
        _sw_patch = int(config.patch_size or 0)
        if _sw_patch > 0:
            # Sliding-window per-image eval at native resolution — the same
            # inference the prediction engine (推論実行) runs. The DataLoader
            # path below fed random augmented train patches under SW
            # validation (val_ds/train_eval_ds are None there) and resized
            # test images to input_size, so hard-mining picks and iter-stop
            # macros judged something production never shows.
            _split_map: dict = {}
            for _sname in ("train", "val", "test"):
                _sf = prepared_dir / "splits" / f"{_sname}.txt"
                if _sf.exists():
                    for _line in _sf.read_text(encoding="utf-8").split():
                        if _line.strip():
                            _split_map[_line.strip()] = _sname
            _fg_thr = None
            # Iterations > 0 evaluate at the threshold inherited from the
            # chain (written into this run's train_config.json at launch)
            # so per-iteration scores stay comparable.
            if (bool(getattr(config, "iterative_mode", False))
                    and int(getattr(config, "iter_index", 0) or 0) > 0):
                try:
                    _fg_thr = json.loads(
                        (run_dir / "train_config.json").read_text(encoding="utf-8")
                    ).get("inference_threshold")
                except (json.JSONDecodeError, OSError):
                    _fg_thr = None
            if _fg_thr is None and val_threshold_info is not None and "optimal_threshold" in val_threshold_info:
                _fg_thr = float(val_threshold_info["optimal_threshold"])
            _fg_thr = float(_fg_thr) if _fg_thr is not None else None
            # Same stride rule as the prediction engine (patch * 3/4,
            # aligned to output_stride) so scores match 推論実行 exactly.
            _eval_stride = default_patch_stride(_sw_patch)
            _eval_stride = max(
                config.output_stride,
                _eval_stride - _eval_stride % config.output_stride,
            )
            _per_image_all = compute_per_image_metrics_sw(
                model,
                prepared_dir / "images",
                prepared_dir / "masks",
                _split_map,
                _sw_patch,
                _eval_stride,
                num_classes,
                config.output_stride,
                config.ignore_index,
                config.normalize,
                device,
                include_background=False,
                active_class_ids=resolved_active_ids,
                fg_threshold=_fg_thr,
                save_predictions_dir=_pred_root,
                log_fn=log_fn,
            )
        else:
            # Full-image training (no patches): the production engine also
            # infers by resizing to input_size here, so the dataset-based
            # path matches production and stays.
            _train_eval_target = train_eval_ds if train_eval_ds is not None else train_ds
            if _train_eval_target is not None:
                _train_per_image = compute_per_image_metrics(
                    model, _train_eval_target, num_classes, config.ignore_index, device,
                    include_background=False, active_class_ids=resolved_active_ids,
                    save_predictions_dir=_pred_root,
                )
                for _stem, _entry in _train_per_image.items():
                    _per_image_all[_stem] = {**_entry, "split": "train"}
            if val_ds is not None:
                _val_per_image = compute_per_image_metrics(
                    model, val_ds, num_classes, config.ignore_index, device,
                    include_background=False, active_class_ids=resolved_active_ids,
                    save_predictions_dir=_pred_root,
                )
                for _stem, _entry in _val_per_image.items():
                    _per_image_all[_stem] = {**_entry, "split": "val"}
            if test_ds is not None:
                # Test set is a holdout: predictions + metrics are produced
                # (so the results view isn't blank) but hard mining skips
                # split="test" to keep the holdout honest.
                _test_per_image = compute_per_image_metrics(
                    model, test_ds, num_classes, config.ignore_index, device,
                    include_background=False, active_class_ids=resolved_active_ids,
                    save_predictions_dir=_pred_root,
                )
                for _stem, _entry in _test_per_image.items():
                    _per_image_all[_stem] = {**_entry, "split": "test"}
        if _per_image_all:
            (run_dir / "per_image_metrics.json").write_text(
                json.dumps(_per_image_all, indent=2), encoding="utf-8"
            )
            log_fn(f"Per-image metrics: saved for {len(_per_image_all)} images\n")

        # Iterative hard mining decision (only when the run was launched in
        # iterative mode; the runner passes these via config).
        _target_recall = float(getattr(config, "target_recall", 0.0) or 0.0)
        _target_precision = float(getattr(config, "target_precision", 0.0) or 0.0)
        _iterative_on = bool(getattr(config, "iterative_mode", False))
        if _iterative_on and (_target_recall > 0 or _target_precision > 0):
            # Compute the iter-stop macros from per_image_all — every
            # annotated image (train + val + test) contributes. A 5-image
            # val macro is too noisy to trust as a chain-stop signal for
            # small projects; the whole point of the completion hook is
            # already to inference all 38 images, so we might as well use
            # them. Hard-mining still filters to train+val below to keep
            # the test holdout honest.
            _target_confidence = float(getattr(config, "target_confidence", 0.0) or 0.0)
            # Dataset-level micro P/R (the UI 全画像スコア aggregation).
            # Confidence stays per-defect-image: mean prob at GT pixels.
            _val_macro_prec, _val_macro_rec = _dataset_micro_prf(_per_image_all)
            _conf_pool = [v["mean_fg_conf_at_gt"] for v in _per_image_all.values()
                          if v.get("mean_fg_conf_at_gt") is not None]
            _val_macro_conf = (sum(_conf_pool) / len(_conf_pool)) if _conf_pool else 0.0
            _prec_ok = _val_macro_prec >= _target_precision
            _rec_ok = _val_macro_rec >= _target_recall
            _conf_ok = (_target_confidence <= 0.0) or (_val_macro_conf >= _target_confidence)
            if not (_prec_ok and _rec_ok and _conf_ok):
                _hard_ids: set[str] = set()
                # Only train+val entries are eligible; test is the honest
                # holdout and must not leak into the next iteration's train.
                _hard_pool = {
                    _sid: _v for _sid, _v in _per_image_all.items()
                    if _v.get("split") in ("train", "val")
                }
                _n_hard = max(3, len(_hard_pool) // 5)
                if not _prec_ok:
                    # Rank by confidence-weighted FP mass (sum of prob above
                    # the threshold over FP pixels): a small confidently
                    # wrong ghost outranks a large barely-over-threshold
                    # one, because it survives threshold raises. Falls back
                    # to raw FP pixel counts for legacy metric payloads.
                    _by_fp = sorted(
                        _hard_pool.items(),
                        key=lambda kv: _damage_key(kv[1], "fp"),
                        reverse=True,
                    )
                    for _sid, _v in _by_fp[:_n_hard]:
                        if _damage_key(_v, "fp") > 0:
                            _hard_ids.add(_sid)
                if not _rec_ok:
                    # Symmetric for misses: deep FNs (prob far below the
                    # threshold at GT pixels) need training the most.
                    _by_fn = sorted(
                        _hard_pool.items(),
                        key=lambda kv: _damage_key(kv[1], "fn"),
                        reverse=True,
                    )
                    for _sid, _v in _by_fn[:_n_hard]:
                        if _damage_key(_v, "fn") > 0:
                            _hard_ids.add(_sid)
                if not _conf_ok:
                    # Sort ascending on mean_fg_conf_at_gt so the least
                    # confident "barely-correct" images (Rec passes but
                    # prob at GT ≈ threshold) become the hard set —
                    # the loop we care about for FP/TP separation.
                    _low_conf = sorted(
                        (kv for kv in _hard_pool.items()
                         if kv[1].get("mean_fg_conf_at_gt") is not None),
                        key=lambda kv: kv[1]["mean_fg_conf_at_gt"],
                    )
                    for _sid, _ in _low_conf[:_n_hard]:
                        _hard_ids.add(_sid)
                (run_dir / "iterative_hard_ids.json").write_text(
                    json.dumps({
                        "hard_ids": sorted(_hard_ids),
                        "judge_metric": "dataset_micro_prf",
                        "val_macro_prec": _val_macro_prec,
                        "val_macro_rec": _val_macro_rec,
                        "val_macro_conf": _val_macro_conf,
                        "target_precision": _target_precision,
                        "target_recall": _target_recall,
                        "target_confidence": _target_confidence,
                    }, indent=2),
                    encoding="utf-8",
                )
                log_fn(
                    f"Iterative: dataset Prec={_val_macro_prec:.3f}/Rec={_val_macro_rec:.3f}"
                    f"/Conf={_val_macro_conf:.3f} under target "
                    f"({_target_precision:.2f}/{_target_recall:.2f}/{_target_confidence:.2f}); "
                    f"flagged {len(_hard_ids)} hard IDs for the next iteration\n"
                )
            else:
                log_fn(
                    f"Iterative: dataset Prec={_val_macro_prec:.3f}/Rec={_val_macro_rec:.3f}"
                    f"/Conf={_val_macro_conf:.3f} meets target; no next iteration.\n"
                )
    except Exception as _pim_err:
        # Do not fail the whole training run over a metrics-side error.
        log_fn(f"WARN: per-image metrics / iterative check failed: {_pim_err}\n")
