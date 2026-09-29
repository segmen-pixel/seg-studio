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

from ..training.metrics_threshold import build_operating_points
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
#:
#: Fallback only. The live value comes from the SDK's own config (see
#: model_resolution) because this copy drifted: it claimed 384 for nano, which
#: takes 312, and 432 for large, which takes 504. Composition doubles this
#: number, so a wrong entry silently breaks the one invariant the patch mode
#: exists to hold -- that the canvas is the model's input and the object
#: reaches it at capture scale. `large` is selectable, so that one shipped.
_MODEL_RESOLUTION = {
    "nano": 312,
    "small": 384,
    "medium": 432,
    "large": 504,
}


def model_resolution(model_size: str) -> int:
    """Input size of *model_size*, read from the SDK when it is importable.

    Building the config is cheap -- it is a plain settings object, no weights
    are loaded -- so the authoritative number is available at the one moment
    it matters. The table above is the answer when rfdetr is absent, which is
    every environment that never installed the instance extra.
    """
    key = str(model_size).lower()
    name = _MODEL_CLASSES.get(key)
    if name is not None:
        try:
            import rfdetr.config as _rf_config

            return int(getattr(_rf_config, f"{name}Config")().resolution)
        except Exception:
            pass
    return int(_MODEL_RESOLUTION.get(key, 432))
_THRESHOLD_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
_DEDUP_IOU = 0.7
#: Epochs between COCO evaluations during the fine-tune.
#:
#: rfdetr evaluates every epoch over the whole validation split, and does it
#: twice: on_validation_batch_end runs a second forward pass through the EMA
#: weights so both metric families can be logged. Semantic runs already
#: evaluate on a 5-epoch cadence and nothing here asked for a finer one.
#:
#: The trade is checkpoint granularity. BestModelCallback is a no-op on an
#: epoch whose evaluation was skipped, so the best checkpoint is chosen among
#: evaluated epochs only, and a run stopped before the first one has nothing
#: to keep (write_run_contract reports that instead of raising). The final
#: epoch always evaluates.
_EVAL_INTERVAL = 5
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


def loader_steps_per_epoch(dataset_dir: Path, batch_size: int) -> int | None:
    """Batches the train loader yields per epoch, or None when unreadable.

    Counted from the COCO manifest the composer just wrote, because that is
    the population the loader iterates. Only used to cap worker count, so an
    unreadable manifest degrades to "do not cap" rather than failing.
    """
    if batch_size <= 0:
        return None
    try:
        with open(dataset_dir / "train" / "_annotations.coco.json",
                  encoding="utf-8") as fh:
            images = json.load(fh).get("images") or []
    except (OSError, ValueError, AttributeError):
        return None
    if not images:
        return None
    return max(1, math.ceil(len(images) / batch_size))


def plan_num_workers(steps_per_epoch: int | None = None) -> tuple[int, list[str]]:
    """DataLoader workers for the rfdetr fine-tune, sized against this host.

    Returns ``(num_workers, reasoning)``. The count itself is decided by
    ``runtime.plan_instance_workers``, which sits in the same module as the
    semantic path's planner so both DataLoader sizing policies stay in one
    auditable place and both answer against measured host resources -- a
    constant here would be tuned to whichever machine the author had.

    Escape hatches, in priority order, matching the semantic path:
      ``SEG_INSTANCE_NUM_WORKERS=<n>``  use exactly n (0 disables workers)
      ``SEG_DISABLE_AUTO_PLAN=1``       skip planning, use 0

    Verified 2026-07-23: workers>0 is safe in a fresh training process. The
    crash that once justified a hard 0 came from calling train() twice in
    one process, which this path cannot do -- the subprocess entry point
    (trainer_api's _instance_train_subprocess_worker) calls train_instance
    once and exits.

    Measured 2026-08-19 on the reference box, identical config across arms:
    66.5 s/epoch at 0 workers, 57.9 s at 2, 59.1 s at 4 (-13% at the knee).
    The remaining wall time is main-thread bound, not decode bound, which is
    why the planner's ceiling is low rather than "as many as fit".
    """
    raw = os.environ.get("SEG_INSTANCE_NUM_WORKERS", "").strip()
    if raw:
        try:
            n = max(0, int(raw))
        except ValueError:
            return 0, [f"workers: SEG_INSTANCE_NUM_WORKERS={raw!r} is not an "
                       f"integer -> 0"]
        return n, [f"workers: SEG_INSTANCE_NUM_WORKERS={raw!r} -> {n}"]

    if os.environ.get("SEG_DISABLE_AUTO_PLAN") == "1":
        return 0, ["workers: SEG_DISABLE_AUTO_PLAN=1 -> 0"]

    try:
        from ..runtime import ProcessRegistry, plan_instance_workers, probe_host

        host = probe_host()
        try:
            peers = ProcessRegistry().others_total_ram()
        except Exception:
            peers = 0
        return plan_instance_workers(host, steps_per_epoch=steps_per_epoch,
                                     peers_ram_bytes=peers)
    except Exception as exc:  # pragma: no cover - host-dependent
        # The plan is an optimisation; never let probing failure stop a run.
        return 0, [f"workers: host probe failed ({type(exc).__name__}: {exc})"
                   f" -> 0"]


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


def _build_rfdetr(model_cls, *, device: str | None = None, **kwargs):
    """Instantiate an rfdetr model class with AMP gated by the shared policy.

    rfdetr defaults to ``amp=True`` and resolves ``amp_dtype="auto"`` to fp16
    on anything below Ampere -- exactly the combination
    segcore.training.amp_policy exists to keep away from this project's
    models. Nothing here overrode it, so instance training and every path that
    reloads its checkpoint ran fp16 autocast on cards that cannot do it.

    Measured on a GTX 1650 Max-Q (Turing, cc 7.5), RF-DETR-Seg nano over the
    same composed dataset at batch 2 / grad_accum 8: one epoch took 8.8 min
    with the SDK default and 4.0 min with AMP off. The correctness half
    weighs more than the speed -- on the semantic side the same card returns
    all-NaN logits under fp16 autocast once a forward batch reaches four.

    An explicit ``amp=`` from the caller still wins; this only supplies the
    default the SDK should have had.
    """
    import torch

    from segcore.training.amp_policy import amp_supported

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs.setdefault("amp", amp_supported(resolved))
    return model_cls(**kwargs)


def build_model(model_size: str, *, device: str | None = None, **kwargs):
    """``_build_rfdetr`` by size name, for callers outside this module."""
    return _build_rfdetr(_load_model_class(model_size), device=device, **kwargs)


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


def count_curve_point(
    threshold: float,
    predicted: dict[str, dict[int, int]],
    truth: dict[str, dict[int, int]],
    categories: list[int],
) -> dict[str, float]:
    """One point of the count sweep, measured in objects rather than pixels.

    An object counted beyond the true number is a false positive and one short
    of it is a miss, so precision and recall answer "did it over-call" and
    "did it under-call" -- the two ways a counting line goes wrong, which a
    single exact-match tally cannot separate.

    ``exact_images`` keeps the older, stricter question (did EVERY class in
    this image match) because that is what picks the threshold the run ships.
    The two are reported side by side rather than one being derived from the
    other: a threshold can improve precision while matching fewer images
    exactly, and an operator choosing a preset deserves to see that.

    Counts are agreements, not matches. Nothing here checks WHERE an object
    was, so this is a different judgement from the semantic runs' defect
    recall, which asks whether half of a specific region was covered.
    """
    exact = 0
    found = over = missed = 0
    for name, truth_counts in truth.items():
        got = predicted.get(name, {})
        if all(got.get(cid, 0) == n for cid, n in truth_counts.items()):
            exact += 1
        # The union of categories, so a class predicted into an image that
        # never had one still costs precision: truth is zero-filled per
        # manifest, and with two manifests they need not list the same ones.
        for cid in categories:
            n_true = int(truth_counts.get(cid, 0))
            n_pred = int(got.get(cid, 0))
            found += min(n_pred, n_true)
            over += max(0, n_pred - n_true)
            missed += max(0, n_true - n_pred)
    precision = found / (found + over) if found + over else 0.0
    recall = found / (found + missed) if found + missed else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "threshold": float(threshold),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "objects_found": found,
        "objects_total": found + missed,
        "exact_images": exact,
    }


def _calibrate_threshold(
    model_cls, checkpoint: Path, dataset_dir: Path, log_fn: Callable[[str], None],
    patch_size: int | None = None,
    object_span_px: int | None = None,
) -> tuple[float, int | None, int, list[dict[str, float]]]:
    """Calibrate the count threshold on the real validation images.

    Predict once per image at the lowest grid threshold, then evaluate every
    grid value by confidence filtering + dedup. Returns
    (threshold, exact_matches, n_images, curve), with exact_matches None when
    there were no real validation images to calibrate against -- distinguishing
    "no threshold matched" from "this never ran", which a plain 0 does not.

    *curve* is what the sweep measured at every grid value and used to be
    thrown away: the loop already counted each image at each threshold, so the
    precision/recall the results view needs cost no extra inference. It is
    counted in objects -- an over-count is a false positive and an under-count
    a miss -- which is a different judgement from the semantic runs' defect
    recall ("was half of this region covered"), and so travels under different
    names.

    *patch_size* must be the one inference will tile at. Counting the
    validation photos a different way than production counts them would
    optimise the threshold for a pipeline that never runs: the model sees each
    object several times larger through a patch than through a whole
    2560x2048 frame resized to its 384 input.
    """
    from PIL import Image

    from .compose import REAL_ANNOTATIONS_NAME

    val_dir = dataset_dir / "valid"
    # Photos larger than one inference tile travel in their own manifest,
    # deliberately kept out of the COCO the detector trains and evaluates on
    # (see compose_dataset_split). They are exactly the ones calibration wants:
    # whole frames, counted the way production counts them. Both files are read
    # so a dataset composed before the manifest existed still calibrates.
    annotation_files = [val_dir / "_annotations.coco.json",
                        val_dir / REAL_ANNOTATIONS_NAME]
    # Per-category GT counts: with several classes an image only counts as
    # exact when EVERY class matches, so a model that trades screws for
    # nuts cannot look calibrated.
    categories: list[int] = []
    gt_counts: dict[str, dict[int, int]] = {}
    for ann_path in annotation_files:
        if not ann_path.exists():
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = [int(c["id"]) for c in ann.get("categories", [])] or [1]
        for cid in cats:
            if cid not in categories:
                categories.append(cid)
        # Image ids restart in each file, so annotations are grouped per file
        # and only the file name carries across.
        by_image: dict[int, dict[int, int]] = {}
        for a in ann.get("annotations", []):
            cid = int(a.get("category_id", 1))
            slot = by_image.setdefault(int(a["image_id"]), {})
            slot[cid] = slot.get(cid, 0) + 1
        for im in ann.get("images", []):
            if not im["file_name"].startswith("real_"):
                continue
            per_class = {cid: 0 for cid in cats}
            per_class.update(by_image.get(int(im["id"]), {}))
            gt_counts[im["file_name"]] = per_class
    categories = categories or [1]
    if not gt_counts:
        log_fn(f"[instance] WARNING: no real validation images — the count "
               f"threshold is NOT calibrated and stays at the grid minimum "
               f"{_THRESHOLD_GRID[0]}. Every annotated image had a region "
               f"outside the single-object area band, or none reached the "
               f"validation split.\n")
        return _THRESHOLD_GRID[0], None, 0, []

    # Full-resolution real photos make this loop slow (each detection mask is
    # upsampled to the source resolution), so report progress as it runs —
    # a silent multi-minute phase reads as a hang in the run log.
    log_fn(f"[instance] [PHASE 2b] calibrating count threshold on "
           f"{len(gt_counts)} real validation images (slow on full-res photos)\n")
    model = _build_rfdetr(model_cls, pretrain_weights=str(checkpoint))
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
                iou_threshold=_DEDUP_IOU, object_span_px=object_span_px)
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

    curve: list[dict[str, float]] = []
    best_thr, best_ok = _THRESHOLD_GRID[0], -1
    for thr in _THRESHOLD_GRID:
        predicted = {
            fn: count_instances_by_class(masks, confs, cids, thr, _DEDUP_IOU)
            for fn, (masks, confs, cids) in preds.items()
        }
        point = count_curve_point(thr, predicted, gt_counts, categories)
        curve.append(point)
        ok = int(point["exact_images"])
        log_fn(f"[instance] thr={thr:.2f} val exact {ok}/{len(gt_counts)}  "
               f"P {point['precision']:.3f} R {point['recall']:.3f} "
               f"objects {int(point['objects_found'])}/"
               f"{int(point['objects_total'])}\n")
        if ok > best_ok:
            best_ok, best_thr = ok, thr
    return best_thr, best_ok, len(gt_counts), curve


def early_stopping_patience_epochs(patience_epochs: Any) -> int:
    """The patience to hand RF-DETR, in epochs. Passed through, not converted.

    RF-DETR documents patience as a count of *evaluations* and states that the
    epochs ``eval_interval`` suppresses are skipped rather than counted, so
    this used to divide by the interval. In practice it does not work that
    way: the callback reads ``trainer.callback_metrics``, which
    Lightning carries forward between epochs, so on a suppressed epoch it sees
    the previous evaluation's number again, scores it as "no improvement", and
    spends a patience step anyway. A 20-epoch run configured for 10 epochs of
    patience stopped 2 epochs after its best.

    So patience is spent per epoch and the configured number goes through
    unchanged. Do not reintroduce the division, whatever the SDK docstring
    says -- it makes the real patience shorter than asked for by exactly the
    evaluation interval.

    Zero, None and negatives all mean "never stop early", which is how to ask
    for the full epoch count now that this is on by default.
    """
    try:
        epochs = int(patience_epochs)
    except (TypeError, ValueError):
        return 0
    return epochs if epochs > 0 else 0


def _write_split_map(run_dir: Path, stats: dict[str, Any]) -> int:
    """Record which split each source image landed in, for the results view.

    The semantic runs write this file from their finalize pass, and the API's
    /splits endpoint reads it without caring which trainer produced it. An
    instance run that skipped it left every row in the results list labelled
    as if no run had ever used the image -- the same images the run had just
    trained on, shown as unassigned.

    Only the split is written. There are no per-image scores here because the
    detector never scores a source image directly: it trains on composites cut
    from them. A key that is absent is honestly absent rather than zero.
    """
    splits: dict[str, dict[str, str]] = {}
    for key, split in (("train_source_ids", "train"), ("val_source_ids", "val")):
        for item_id in stats.get(key) or []:
            splits[str(item_id)] = {"split": split}
    if splits:
        _write_json_atomic(run_dir / "per_image_metrics.json", splits, indent=1)
    return len(splits)


def write_run_contract(
    dataset_dir: Path,
    run_dir: Path,
    params: dict[str, Any],
    log_fn: Callable[[str], None],
    *,
    calibrate: bool = True,
) -> bool:
    """Write metrics.json + instance_inference.json for the run in *run_dir*.

    Split out of train_instance because a stopped run needs it too. rfdetr
    exposes no in-training stop hook, so stopping terminates the child inside
    model.train() and every line after it — including this one — never ran in
    that process. The checkpoints were on disk and the run still reported no
    model, because instance_inference.json is what _instance_model_ok() looks
    for. The parent calls this after the terminate so an early stop keeps the
    epochs the user already paid for.

    *calibrate* False ships the grid minimum instead of sweeping for it, for
    callers that cannot afford to load the model back onto the GPU.

    Returns False when there is no checkpoint to describe yet. That is a
    failure for a completed run and an ordinary early stop for a stopped one,
    so the caller decides which it was.
    """
    model_size = str(params.get("model_size", "small"))
    out_dir = run_dir / "rfdetr"
    metrics = _parse_metrics_csv(out_dir / "metrics.csv")
    # checkpoint_best_total aggregates regular+ema; glob order makes it last.
    ckpts = sorted(out_dir.glob("checkpoint_best*.pth"))
    if not ckpts:
        return False
    checkpoint = ckpts[-1]

    threshold, exact_ok = _THRESHOLD_GRID[0], None
    if calibrate:
        model_cls = _load_model_class(model_size)
        threshold, exact_ok, exact_n, count_curve = _calibrate_threshold(
            model_cls, checkpoint, dataset_dir, log_fn,
            patch_size=params.get("patch_size"),
            # Same number the contract records, so calibration stitches
            # fragments exactly where inference will.
            object_span_px=params.get("object_span_px"))
        metrics["count_exact_val"] = exact_ok
        metrics["count_exact_val_n"] = exact_n
        if count_curve:
            metrics["count_curve"] = count_curve
            # The same function the semantic runs call, so there is one rule
            # for what "miss least" means and it is tested once. The unit only
            # reaches the basis strings: this sweep counted objects.
            metrics["operating_points"] = build_operating_points(
                count_curve, unit="count")
            # The shipped threshold under the name every reader of a
            # metrics.json already knows. Without it the results view has no
            # operating point to read its row at, and the presets have nothing
            # to show as currently selected.
            metrics["optimal_threshold"] = float(threshold)

    stats_file = dataset_dir / "stats.json"
    if stats_file.exists():
        metrics["dataset_stats"] = json.loads(stats_file.read_text(encoding="utf-8"))
        n_split = _write_split_map(run_dir, metrics["dataset_stats"])
        if n_split:
            log_fn(f"[instance] split recorded for {n_split} source images\n")
    metrics["training_mode"] = "instance"
    metrics["instance_model_size"] = model_size
    _write_json_atomic(run_dir / "metrics.json", metrics, indent=1, sort_keys=True)

    # Contract written last, atomically, and only after the checkpoint above
    # was verified to exist — its presence is what flags "model available".
    _write_json_atomic(run_dir / "instance_inference.json", {
        "checkpoint": checkpoint.name,
        "threshold": threshold,
        # False means the grid minimum was used rather than measured: either
        # there was nothing to calibrate against, or the run was stopped and
        # the caller chose not to pay for the sweep. Serving cannot tell any
        # of those apart from the number alone, and a project whose every
        # annotated image held a touching pair used to ship an unmeasured 0.3
        # looking exactly like a measured one.
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
        # Longest annotated object, in source pixels. Compared against the
        # tile overlap this is what says whether a count can be trusted; see
        # instance_training._warn_if_patch_too_small.
        "object_span_px": params.get("object_span_px"),
        # Multi-class bookkeeping: the model predicts contiguous COCO
        # category ids, so inference needs the mapping back to the project's
        # semantic class ids (and their names for display).
        "class_ids": [int(c) for c in params.get("class_ids", [1])],
        "class_names": dict(params.get("class_names", {})),
        "coco_category_of": dict(params.get("coco_category_of", {})),
    }, indent=1)
    log_fn(f"[instance] metrics written (best segm mAP "
           f"{metrics.get('segm_mAP_50_95_val')}, thr={threshold})\n")
    return True


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

    workers, worker_reasoning = plan_num_workers(
        steps_per_epoch=loader_steps_per_epoch(dataset_dir,
                                               int(params["batch_size"])))
    log_fn(f"[instance] fine-tuning RF-DETR-Seg {model_size} "
           f"(epochs={params['epochs']}, batch={params['batch_size']}, "
           f"workers={workers})\n")
    for _line in worker_reasoning:
        log_fn(f"[instance]   - {_line}\n")
    device = params.get("device")
    # Same line the semantic runs print, from the same policy: the two must
    # not disagree about precision, and until now nobody said which one this
    # was using.
    from segcore.training.amp_policy import amp_status_line

    log_fn(f"[instance] {amp_status_line(device)}\n")
    model = _build_rfdetr(model_cls, device=device)
    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        grad_accum_steps=int(params.get("grad_accum_steps", 2)),
        lr=float(params.get("lr", 1e-4)),
        num_workers=workers,
        output_dir=str(out_dir),
        # rfdetr's multi_scale does not actually vary the scale here: with
        # do_random_resize_via_padding left False it keeps only the largest
        # candidate, which for a 384-input model is 504. Training fed the
        # model 504 while validation, predict() and the tiled inference path
        # all use the config resolution of 384 -- so composition sized its
        # canvas at 2x384 for the clean 2:1 this mode exists to hold, and
        # training quietly took 1.52:1 instead, reaching the model with every
        # object 1.31x larger than inference will ever show it. Pinning the
        # scale to the model resolution restores the invariant; 42% fewer
        # training pixels per epoch is the side effect, not the reason.
        multi_scale=False,
        eval_interval=_EVAL_INTERVAL,
    )
    if workers > 0:
        # Spawned workers re-import torch (~10s each on Windows);
        # persistent workers pay that once instead of every epoch.
        train_kwargs["persistent_workers"] = True
    patience = early_stopping_patience_epochs(
        params.get("early_stopping_patience"))
    if patience:
        train_kwargs["early_stopping"] = True
        train_kwargs["early_stopping_patience"] = patience
        log_fn(f"[instance] early stopping: {patience} epochs without a better "
               f"segmentation mAP ends the run\n")
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

    if not write_run_contract(dataset_dir, run_dir, params, log_fn):
        # Without a checkpoint the run is unusable: fail loudly instead of
        # writing a contract whose inference can never run.
        raise RuntimeError(
            "rfdetr training finished without producing checkpoint_best*.pth "
            f"under {out_dir} — marking the run failed")
