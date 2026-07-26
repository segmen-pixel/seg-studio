# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""RF-DETR-Seg fine-tune wrapper for instance-mode training.

Runs inside the training child process. rfdetr is imported lazily with an
install hint (same pattern as the OpenVINO backend) so environments without
the dependency fail with a clear message instead of an import crash.

Writes into the run dir:
  rfdetr/                  — checkpoints + lightning metrics.csv
  metrics.json             — translated metrics for the UI
  instance_inference.json  — checkpoint / threshold / dedup contract for predict
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .count import count_instances_by_class


def _write_json_atomic(path: Path, data: Any, **dump_kwargs) -> None:
    """Write JSON via a temp file + os.replace so readers never see partials."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, **dump_kwargs)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

# RF-DETR-Seg checkpoints are Apache-2.0 across all Seg sizes (license
# trail: dev commit 2dd655b, verified against the upstream repo 2026-07-20).
# The non-Seg large detection variants and the "plus"-tier models are under
# a more restrictive (non-Apache) license and are deliberately NOT mapped
# here — do not add them without re-running the license check.
# "nano" is no longer offered for new training (trainer_api's
# INSTANCE_MODEL_SIZES) but stays mapped so checkpoints from earlier runs
# remain loadable for prediction and export.
_MODEL_CLASSES = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}

#: The square input each size takes. Composition uses it as the patch size so a
#: composed canvas is already the model's input and nothing is resized away;
#: sliding-window inference tiles at the same size, so the object reaches the
#: model at the size the camera gave it in both.
_MODEL_RESOLUTION = {
    "nano": 384,
    "small": 384,
    "medium": 432,
    "large": 432,
}


def model_resolution(model_size: str) -> int:
    """Input size of *model_size*, defaulting to the medium/large value."""
    return int(_MODEL_RESOLUTION.get(str(model_size).lower(), 432))
_THRESHOLD_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
_DEDUP_IOU = 0.7
# Calibration evaluates the threshold grid against every prediction's masks;
# on full-resolution photos the pairwise mask-IoU dedup dominates wall time
# (tens of minutes on 16MP sources). IoU is scale-invariant for blob-sized
# masks, so calibration shrinks them to this bound first (measured: identical
# counts, minutes -> seconds).
_CALIB_MASK_MAX_SIDE = 1024


def shrink_masks_for_iou(masks: list, max_side: int = _CALIB_MASK_MAX_SIDE) -> list:
    """Downscale binary masks (nearest) so their long side is <= max_side.

    Only IoU *ratios* between the masks are consumed downstream, so scaling
    every mask identically preserves dedup and count decisions.
    """
    if not masks:
        return masks
    import cv2
    import numpy as np

    h, w = masks[0].shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return masks
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return [
        cv2.resize(np.asarray(m).astype("uint8"), (nw, nh),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
        for m in masks
    ]


def resolve_num_workers() -> int:
    """DataLoader workers for the rfdetr fine-tune (default 0).

    Verified 2026-07-23: workers>0 works in a fresh training process
    (the earlier crash came from a second train() in one process), but a
    1-epoch probe measured only ~6% gain for workers=2 on the reference
    box, so the safe default stays 0 (each Windows worker costs GBs of
    commit charge). Set SEG_INSTANCE_NUM_WORKERS to experiment; planner
    integration is tracked separately.
    """
    raw = os.environ.get("SEG_INSTANCE_NUM_WORKERS", "").strip()
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


def _load_model_class(model_size: str):
    name = _MODEL_CLASSES.get(model_size)
    if name is None:
        raise ValueError(f"unknown instance_model_size: {model_size!r}")
    try:
        import rfdetr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "rfdetr is not installed. Instance-mode training requires it: "
            "pip install \"rfdetr[train]\""
        ) from exc
    return getattr(rfdetr, name)


def _parse_metrics_csv(csv_path: Path) -> dict[str, Any]:
    """Pick the best val epoch from lightning's metrics.csv."""
    if not csv_path.exists():
        return {}
    best: dict[str, Any] = {}
    best_map = -1.0
    last_epoch = 0
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                last_epoch = max(last_epoch, int(float(row.get("epoch") or 0)))
                v = row.get("val/segm_mAP_50_95")
                if not v:
                    continue
                m = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(m) or m <= best_map:
                continue
            best_map = m

            def _f(key: str) -> float | None:
                raw = row.get(key)
                try:
                    val = float(raw) if raw not in (None, "") else None
                except (TypeError, ValueError):
                    return None
                return None if val is None or math.isnan(val) else val

            best = {
                "segm_mAP_50_95_val": m,
                "segm_mAP_50_val": _f("val/segm_mAP_50"),
                "mAP_50_95_val": _f("val/mAP_50_95"),
                "AR_val": _f("val/mAR"),
                "F1_val": _f("val/F1"),
                "best_epoch": int(float(row.get("epoch") or 0)),
            }
    best["epochs_effective"] = last_epoch + 1
    return best


def read_epoch_val_metrics(
    csv_path: Path, after_epoch: int = -1,
) -> list[dict[str, Any]]:
    """Per-epoch validation rows from lightning's metrics.csv.

    Returns rows with ``epoch > after_epoch`` sorted by epoch, each as
    ``{"epoch", "segm_map", "segm_map50", "f1"}`` (metric values may be
    None when the column is absent). The parent training monitor uses this
    to stream per-epoch progress into the run log while the child trains —
    the child's own stdout (rich progress bars) is never forwarded.
    """
    if not csv_path.exists():
        return []
    rows: dict[int, dict[str, Any]] = {}
    train_loss: dict[int, float] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                epoch = int(float(row.get("epoch") or 0))
            except (TypeError, ValueError):
                continue

            def _f(key: str) -> float | None:
                v = row.get(key)
                try:
                    val = float(v) if v not in (None, "") else None
                except (TypeError, ValueError):
                    return None
                return None if val is None or math.isnan(val) else val

            tl = _f("train/loss")
            if tl is not None:
                train_loss[epoch] = tl  # keep the last train row per epoch
            m = _f("val/segm_mAP_50_95")
            if m is None or epoch <= after_epoch:
                continue
            rows[epoch] = {
                "epoch": epoch,
                "segm_map": m,
                "segm_map50": _f("val/segm_mAP_50"),
                "f1": _f("val/F1"),
            }
    for e, r in rows.items():
        r["train_loss"] = train_loss.get(e)
    return [rows[e] for e in sorted(rows)]


def _calibrate_threshold(
    model_cls, checkpoint: Path, dataset_dir: Path, log_fn: Callable[[str], None],
    patch_size: int | None = None,
) -> tuple[float, int | None, int]:
    """Calibrate the count threshold on the real validation images.

    Predict once per image at the lowest grid threshold, then evaluate every
    grid value by confidence filtering + dedup. Returns
    (threshold, exact_matches, n_images), with exact_matches None when there
    were no real validation images to calibrate against -- distinguishing "no
    threshold matched" from "this never ran", which a plain 0 does not.

    *patch_size* must be the one inference will tile at. Counting the
    validation photos a different way than production counts them would
    optimise the threshold for a pipeline that never runs: the model sees each
    object several times larger through a patch than through a whole
    2560x2048 frame resized to its 384 input.
    """
    from PIL import Image

    val_dir = dataset_dir / "valid"
    ann = json.loads((val_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    # Per-category GT counts: with several classes an image only counts as
    # exact when EVERY class matches, so a model that trades screws for
    # nuts cannot look calibrated.
    categories = [int(c["id"]) for c in ann.get("categories", [])] or [1]
    gt_counts: dict[str, dict[int, int]] = {}
    for im in ann["images"]:
        if not im["file_name"].startswith("real_"):
            continue
        per_class = {cid: 0 for cid in categories}
        for a in ann["annotations"]:
            if a["image_id"] == im["id"]:
                per_class[int(a.get("category_id", 1))] = (
                    per_class.get(int(a.get("category_id", 1)), 0) + 1)
        gt_counts[im["file_name"]] = per_class
    if not gt_counts:
        log_fn(f"[instance] WARNING: no real validation images — the count "
               f"threshold is NOT calibrated and stays at the grid minimum "
               f"{_THRESHOLD_GRID[0]}. Every annotated image had a region "
               f"outside the single-object area band, or none reached the "
               f"validation split.\n")
        return _THRESHOLD_GRID[0], None, 0

    # Full-resolution real photos make this loop slow (each detection mask is
    # upsampled to the source resolution), so report progress as it runs —
    # a silent multi-minute phase reads as a hang in the run log.
    log_fn(f"[instance] [PHASE 2b] calibrating count threshold on "
           f"{len(gt_counts)} real validation images (slow on full-res photos)\n")
    model = model_cls(pretrain_weights=str(checkpoint))
    if patch_size:
        log_fn(f"[instance] calibrating over {patch_size}px tiles, as inference "
               f"will run\n")
    preds = {}
    for i, fn in enumerate(gt_counts, start=1):
        img = Image.open(val_dir / fn)
        if patch_size:
            from .tiled import predict_tiled_masks, sdk_tile_predict
            masks, confs, classes, _plan = predict_tiled_masks(
                img, sdk_tile_predict(model, _THRESHOLD_GRID[0]), int(patch_size),
                iou_threshold=_DEDUP_IOU)
        else:
            det = model.predict(img, threshold=_THRESHOLD_GRID[0])
            masks = list(det.mask) if det.mask is not None else []
            confs = [float(c) for c in det.confidence]
            classes = (list(det.class_id)
                       if getattr(det, "class_id", None) is not None
                       else [0] * len(masks))
        # SDK class ids are 0-based model indices; COCO categories start at
        # 1, and gt_counts below is keyed by category id.
        cids = [int(c) + 1 for c in classes]
        preds[fn] = (shrink_masks_for_iou(list(masks)), confs, cids)
        if i % 10 == 0 or i == len(gt_counts):
            log_fn(f"[instance] calibration predict {i}/{len(gt_counts)}\n")

    def _exact(fn: str, thr: float) -> bool:
        masks, confs, cids = preds[fn]
        got = count_instances_by_class(masks, confs, cids, thr, _DEDUP_IOU)
        return all(got.get(cid, 0) == n for cid, n in gt_counts[fn].items())

    best_thr, best_ok = _THRESHOLD_GRID[0], -1
    for thr in _THRESHOLD_GRID:
        ok = sum(1 for fn in gt_counts if _exact(fn, thr))
        log_fn(f"[instance] thr={thr:.2f} val exact {ok}/{len(gt_counts)}\n")
        if ok > best_ok:
            best_ok, best_thr = ok, thr
    return best_thr, best_ok, len(gt_counts)


def train_instance(
    dataset_dir: Path,
    run_dir: Path,
    params: dict[str, Any],
    log_fn: Callable[[str], None],
    stop_flag: Callable[[], bool],
) -> None:
    model_size = str(params.get("model_size", "small"))
    model_cls = _load_model_class(model_size)
    out_dir = run_dir / "rfdetr"
    if stop_flag():
        return

    workers = resolve_num_workers()
    log_fn(f"[instance] fine-tuning RF-DETR-Seg {model_size} "
           f"(epochs={params['epochs']}, batch={params['batch_size']}, "
           f"workers={workers})\n")
    model = model_cls()
    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        grad_accum_steps=int(params.get("grad_accum_steps", 2)),
        lr=float(params.get("lr", 1e-4)),
        num_workers=workers,
        output_dir=str(out_dir),
    )
    if workers > 0:
        # Spawned workers re-import torch (~10s each on Windows);
        # persistent workers pay that once instead of every epoch.
        train_kwargs["persistent_workers"] = True
    model.train(**train_kwargs)

    # Release the trainer's model before calibration loads its own copy from
    # the checkpoint — otherwise both live on the GPU at once.
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if stop_flag():
        return

    metrics = _parse_metrics_csv(out_dir / "metrics.csv")
    # checkpoint_best_total aggregates regular+ema; glob order makes it last.
    ckpts = sorted(out_dir.glob("checkpoint_best*.pth"))
    if not ckpts:
        # Without a checkpoint the run is unusable: fail loudly instead of
        # writing a contract whose inference can never run.
        raise RuntimeError(
            "rfdetr training finished without producing checkpoint_best*.pth "
            f"under {out_dir} — marking the run failed")
    checkpoint = ckpts[-1]

    threshold, exact_ok, exact_n = _THRESHOLD_GRID[0], None, None
    if not stop_flag():
        threshold, exact_ok, exact_n = _calibrate_threshold(
            model_cls, checkpoint, dataset_dir, log_fn,
            patch_size=params.get("patch_size"))
        metrics["count_exact_val"] = exact_ok
        metrics["count_exact_val_n"] = exact_n

    stats_file = dataset_dir / "stats.json"
    if stats_file.exists():
        metrics["dataset_stats"] = json.loads(stats_file.read_text(encoding="utf-8"))
    metrics["training_mode"] = "instance"
    metrics["instance_model_size"] = model_size
    _write_json_atomic(run_dir / "metrics.json", metrics, indent=1, sort_keys=True)

    # Contract written last, atomically, and only after the checkpoint above
    # was verified to exist — its presence is what flags "model available".
    _write_json_atomic(run_dir / "instance_inference.json", {
        "checkpoint": checkpoint.name,
        "threshold": threshold,
        # False means the grid minimum was used because there was nothing to
        # calibrate against. Serving cannot tell the two apart from the number
        # alone, and a project whose every annotated image held a touching
        # pair used to ship an unmeasured 0.3 looking exactly like a measured
        # one.
        "threshold_calibrated": exact_ok is not None,
        "dedup_iou": _DEDUP_IOU,
        "model_size": model_size,
        # Set when the composites were patch-sized at native scale: inference
        # has to tile at this size, because the model never saw a whole frame
        # resized down. Absent means whole-plate composition and the single
        # resized pass. Getting this wrong is silent -- the model runs, and
        # every object is simply the wrong size -- so it travels with the
        # contract rather than being configured separately at inference.
        "patch_size": params.get("patch_size"),
        # Multi-class bookkeeping: the model predicts contiguous COCO
        # category ids, so inference needs the mapping back to the project's
        # semantic class ids (and their names for display).
        "class_ids": [int(c) for c in params.get("class_ids", [1])],
        "class_names": dict(params.get("class_names", {})),
        "coco_category_of": dict(params.get("coco_category_of", {})),
    }, indent=1)
    log_fn(f"[instance] metrics written (best segm mAP "
           f"{metrics.get('segm_mAP_50_95_val')}, thr={threshold})\n")
