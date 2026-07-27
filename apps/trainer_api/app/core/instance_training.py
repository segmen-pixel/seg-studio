# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance-mode training phases (docs/design_instance_segmentation_v098.md).

Called from run_training_job inside its try block: composes the synthetic
COCO dataset from the project's semantic masks, then fine-tunes RF-DETR-Seg
in a child process. Failures raise — the shared handler marks the run
failed; success falls through to the shared completed-status update.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from .annotate_index import load_annotate_index
from .paths import annotate_images_dir, annotate_masks_dir
from .torch_device import touch_torch_device_claim
from .training_workers import _TRAIN_EXIT_OK, _instance_train_subprocess_worker

# rfdetr/lightning has no in-training stop hook; after the stop file has been
# ignored for this long the child is terminated (num_workers=0, so no
# DataLoader worker processes are orphaned by the terminate).
_STOP_GRACE_SECONDS = 30.0
_MIN_SOURCE_IMAGES = 4

# Trainable model sizes. "nano" was dropped in v0.9.8: on the reference
# counting workload it reached only 0.92 segm mAP50 where small/medium reach
# 0.94/0.99, which is not enough for exact-count use. Existing nano
# checkpoints stay loadable for prediction (segcore keeps the class mapping).
#: Patch size composition and inference both use, unless a run overrides it.
#:
#: 768 = twice the 384 model input, so a patch is halved to reach the model on
#: a clean 2:1 rather than a fractional resize. Measured on a real 2560x2048
#: project with 110px screws, counting 4 photos against their annotation:
#:
#:     patch 384   63 tiles   36, 36, 36, 36   mean error 3.2   1.6 s
#:     patch 768   20 tiles   40, 40, 40, 40   mean error 0.8   0.5 s
#:     (truth      39, 39, 40, 39)
#:
#: The larger patch is both more accurate and three times faster: fewer tiles
#: means fewer seams, and every seam costs a detection that has to be dropped
#: as clipped. It also leaves an overlap wider than the object without any
#: special stride, so the geometry works without tuning.
#:
#: Sources smaller than this compose and infer as a single padded patch, which
#: is the behaviour they had before tiling existed -- so a default this large
#: turns tiling on only where there is something to tile.
DEFAULT_PATCH_SIZE = 768

INSTANCE_MODEL_SIZES = ("small", "medium", "large")

# Measured on RTX 3090 with 1-epoch probes; the numbers below are the
# measured peak plus headroom for the CUDA context + allocator fragmentation.
#   small  (2026-07-22): peak reserved 6.5 GiB @ b8, 3.6 @ b4, 2.2 @ b2
#   medium (2026-07-22): peak reserved 14.1 GiB @ b8 (alloc 12.7)
# "large" has no measurement yet — it is not auto-fitted (see below).
_VRAM_REQUIRED_GIB: dict[str, dict[int, float]] = {
    "small": {8: 8.0, 4: 5.5, 2: 3.5},
    "medium": {8: 16.0, 4: 9.5, 2: 6.0},
}
_VRAM_FITTED_SIZES = tuple(_VRAM_REQUIRED_GIB)


def _fit_batch_to_vram(batch: int, grad_accum: int, budget_gib: float,
                       model_size: str = "small") -> tuple[int, int]:
    """Halve the batch (doubling grad-accum) until it fits *budget_gib*.

    Keeps the effective batch (batch x accum) constant. Floor is batch 2 —
    below the b2 requirement training is not supported. Model sizes without
    a measured table are returned unchanged.

    *budget_gib* is what the GPU will actually lend us, not its nameplate
    size: see instance_vram_budget_gib. The two headrooms involved are
    different things and do stack -- the policy one is what the OS and driver
    hold back (2 GiB of a Windows card is simply not ours), the table's own is
    allocator fragmentation inside our process.
    """
    table = _VRAM_REQUIRED_GIB.get(model_size)
    if table is None:
        return batch, grad_accum

    if budget_gib <= 0:
        return batch, grad_accum

    def required(b: int) -> float:
        if b >= 8:
            return table[8]
        if b >= 4:
            return table[4]
        return table[2]

    while batch > 2 and required(batch) > budget_gib:
        batch = max(2, batch // 2)
        grad_accum *= 2
    return batch, grad_accum


def instance_vram_budget_gib(gpu_total_gib: float, is_wddm: bool) -> float:
    """What of a GPU this size is actually available to a training run.

    Uses the same policy the semantic auto-config VRAM check applies
    (segcore.auto_select.vram_predictor.SafetyConfig): a fraction of the card
    plus a fixed headroom, larger on Windows because WDDM lets the OS and the
    compositor hold memory a CUDA process never sees.

    Instance mode used the card's nameplate total instead, so on the same
    Windows machine it would accept a batch the semantic path had already
    judged too large -- 24.0 GiB against 22.1 on a 24 GiB card. One policy,
    imported rather than restated, because two copies of a safety margin drift
    and the optimistic copy is the one that OOMs.
    """
    from segcore.auto_select.vram_predictor import SafetyConfig

    return SafetyConfig().budget_mb(gpu_total_gib * 1024.0, is_wddm) / 1024.0


def validate_instance_config(config: dict[str, Any]) -> None:
    """Request-time validation for instance mode. Raises HTTPException(400)."""
    size = str(config.get("instance_model_size", "small"))
    if size not in INSTANCE_MODEL_SIZES:
        raise HTTPException(
            status_code=400,
            detail="instance_model_size must be one of "
                   + ", ".join(repr(s) for s in INSTANCE_MODEL_SIZES))
    if int(config.get("instance_objects_min", 4)) > int(config.get("instance_objects_max", 8)):
        raise HTTPException(status_code=400,
                            detail="instance_objects_min must be <= instance_objects_max")
    band_min = config.get("instance_area_band_min")
    band_max = config.get("instance_area_band_max")
    if (band_min is None) != (band_max is None):
        raise HTTPException(status_code=400,
                            detail="instance_area_band_min and instance_area_band_max "
                                   "must be set together")
    if band_min is not None and int(band_min) > int(band_max):
        raise HTTPException(status_code=400,
                            detail="instance_area_band_min must be <= instance_area_band_max")
    if bool(config.get("iterative_mode")) or int(config.get("k_folds", 1) or 1) > 1:
        raise HTTPException(status_code=400,
                            detail="instance mode does not support iterative_mode or k_folds")


def load_class_table(classes_file: Path) -> list[dict]:
    """Parsed classes.json entries ([] when missing or corrupt)."""
    if not classes_file.exists():
        return []
    try:
        data = json.loads(classes_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [c for c in data.get("classes", []) if isinstance(c, dict)]


def resolve_class_ids(run_path: Path, config: dict[str, Any]) -> list[int]:
    """Semantic classes this run counts.

    Every active class is counted (owner decision 2026-07-23) — one model
    reports each class separately. ``instance_class_id`` is still honoured
    when set, so older runs and API clients that pinned one class keep
    their previous behaviour.
    """
    cid = config.get("instance_class_id")
    if cid:
        return [int(cid)]
    ids = [int(c.get("id", 0)) for c in load_class_table(run_path / "classes.json")
           if int(c.get("id", 0)) != 0 and c.get("active", True)]
    return sorted(set(ids)) or [1]


def class_name_map(run_path: Path) -> dict[int, str]:
    """{class_id: display name} from the run's classes.json.

    Background (id 0) is excluded: it is never a counted instance class,
    and leaving it in makes a mislabelled detection read as "background"
    instead of standing out as wrong.
    """
    return {int(c["id"]): str(c.get("name") or f"class{int(c['id'])}")
            for c in load_class_table(run_path / "classes.json")
            if c.get("id") is not None and int(c["id"]) != 0}


def _resolve_class_id(run_path: Path, config: dict[str, Any]) -> int:
    """First counted class — the legacy single-class entry point."""
    return resolve_class_ids(run_path, config)[0]


def _load_sources(project_id: str, class_ids: int | list[int],
                  log_fn: Callable[[str], None]):
    """Sources as (item_id, image_bgr, label_mask).

    The label mask keeps the semantic class ids (restricted to
    ``class_ids``), so the composer can build a multi-class dataset; other
    classes and the 255 ignore value are cleared to background.
    """
    img_dir = annotate_images_dir(project_id)
    mask_dir = annotate_masks_dir(project_id)
    sources = []
    for item in load_annotate_index(project_id).get("items", []):
        item_id = item.get("id")
        if not item_id:
            continue
        mask_path = mask_dir / f"{item_id}.png"
        img_path = next(img_dir.glob(f"{item_id}.*"), None)
        if not mask_path.exists() or img_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        wanted = [class_ids] if isinstance(class_ids, int) else list(class_ids)
        keep = np.isin(mask, wanted)
        if not keep.any():
            continue
        fg = np.where(keep, mask, 0).astype("uint8")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        sources.append((str(item_id), img, fg))
    wanted = [class_ids] if isinstance(class_ids, int) else list(class_ids)
    log_fn(f"[instance] annotated sources for classes {wanted}: {len(sources)}\n")
    return sources


def run_instance_phases(
    project_id: str,
    run_id: str,
    run_path: Path,
    logs_path: Path,
    config: dict[str, Any],
    stop_event,
    log_fn: Callable[[str], None],
) -> None:
    from segcore.instseg.compose import ComposeConfig, compose_dataset_split
    from segcore.instseg.train_rfdetr import model_resolution

    log_fn("[instance] [PHASE 1/3] 合成データセット作成 (composing synthetic dataset)\n")
    class_ids = resolve_class_ids(run_path, config)
    names = class_name_map(run_path)
    sources = _load_sources(project_id, class_ids, log_fn)
    if len(sources) < _MIN_SOURCE_IMAGES:
        raise RuntimeError(
            f"instance mode needs at least {_MIN_SOURCE_IMAGES} annotated images "
            f"containing one of classes {class_ids}, found {len(sources)}")

    band_min = config.get("instance_area_band_min")
    band_max = config.get("instance_area_band_max")
    # Patch mode: compose canvases at the model's own input size using cutouts
    # at native scale, the way the semantic side trains on patches cropped at
    # capture resolution. Whole-plate canvases are the source resolution and the
    # detector resizes its input, so a 2560x2048 photo arrives at 432 and a
    # 110px screw becomes 18px before the model sees it. A patch-sized canvas
    # is already the input, so nothing is resized away -- measured on a real
    # project, the same object reaches the model at 101.6px instead of 18.4px.
    #
    # It only pays when the source is much larger than the input; on a 512px
    # project the resize is 0.84x and there is nothing to recover. That case
    # costs nothing to leave on, though: an image smaller than the patch is
    # composed and inferred as a single padded tile, which is what it did
    # before. So this is on by default, and inference tiles to match -- the
    # patch travels in the contract precisely so the two cannot disagree.
    _model_size = str(config.get("instance_model_size", "small"))
    _patch = config.get("instance_patch_size", DEFAULT_PATCH_SIZE)
    if _patch in ("", "0", 0):
        # Explicit off: compose whole plates and infer in one resized pass.
        _patch = None
    elif _patch is None:
        _patch = DEFAULT_PATCH_SIZE
    elif str(_patch).lower() in ("auto", "true", "1"):
        _patch = model_resolution(_model_size) * 2
    else:
        _patch = int(_patch)
    cfg = ComposeConfig(
        n_train=int(config.get("instance_n_train", 500)),
        n_val=int(config.get("instance_n_val", 80)),
        objects_min=int(config.get("instance_objects_min", 4)),
        objects_max=int(config.get("instance_objects_max", 8)),
        stack_pair_prob=float(config.get("instance_stack_pair_prob", 0.55)),
        seed=int(config.get("instance_seed", 42)),
        area_band=(int(band_min), int(band_max)) if band_min and band_max else None,
        patch_size=_patch,
    )
    if _patch:
        log_fn(f"[instance] patch mode: composing {_patch}x{_patch} canvases at native "
               f"scale; inference must tile at the same size\n")
    dataset_dir = run_path / "instseg_dataset"
    # Leakage-safe path: sources are split train/val BEFORE material
    # extraction, so validation imagery never feeds the training composition.
    stats = compose_dataset_split(
        sources, dataset_dir, cfg,
        progress_fn=lambda m: log_fn(f"[instance] {m}\n"),
        class_names=names)
    (dataset_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    log_fn(f"[instance] dataset composed: {stats}\n")
    if stop_event.is_set():
        return

    log_fn("[instance] [PHASE 2/3] RF-DETR-Seg fine-tune (subprocess)\n")
    stop_file = run_path / ".stop"
    if stop_file.exists():
        stop_file.unlink()
    batch = int(config.get("batch_size", 8))
    grad_accum = int(config.get("instance_grad_accum", 2))
    model_size = str(config.get("instance_model_size", "small"))
    resolved_device = str(config.get("resolved_torch_device") or config.get("torch_device") or "")
    if resolved_device.startswith("cuda"):
        if model_size in _VRAM_FITTED_SIZES:
            try:
                import torch

                index = int(resolved_device.split(":", 1)[1]) if ":" in resolved_device else 0
                total_gib = torch.cuda.get_device_properties(index).total_memory / 2 ** 30
                is_wddm = os.name == "nt"
                budget_gib = instance_vram_budget_gib(total_gib, is_wddm)
                need = _VRAM_REQUIRED_GIB[model_size]
                required = need.get(batch if batch in need else
                                    max(k for k in need if k <= max(batch, 2)), need[2])
                log_fn(f"[instance] VRAM: {total_gib:.1f} GiB GPU "
                       f"({'WDDM' if is_wddm else 'Linux'}); {model_size} at batch "
                       f"{batch} needs ~{required:.1f} GiB "
                       f"(budget {budget_gib:.1f} GiB)\n")
                fitted_batch, fitted_accum = _fit_batch_to_vram(
                    batch, grad_accum, budget_gib, model_size)
                if fitted_batch != batch:
                    log_fn(f"[instance] VRAM: batch {batch}->{fitted_batch}, "
                           f"grad_accum {grad_accum}->{fitted_accum} "
                           f"(effective batch kept)\n")
                    batch, grad_accum = fitted_batch, fitted_accum
            except Exception as err:
                # Logged, not swallowed: an unfitted batch is exactly the case
                # that OOMs an hour in, and a silent skip leaves no trace of
                # why the fit never happened.
                log_fn(f"[instance] VRAM: fit skipped ({err}) — using batch "
                       f"{batch} as-is\n")
        else:
            # No measured table for this size, so nothing can be fitted and
            # the requested batch goes through unchecked (see
            # _VRAM_REQUIRED_GIB).
            log_fn(f"[instance] VRAM: WARNING - no measurements for "
                   f"{model_size}, using requested batch {batch} unchecked; "
                   f"this may OOM. Lower the batch by hand if it does.\n")
    params = {
        "model_size": model_size,
        # Recorded so inference can match how the model was trained. Set means
        # the composites were patch-sized at native scale, so inference has to
        # tile at this size; absent/None means whole-plate composition and the
        # usual single resized pass.
        "patch_size": _patch,
        "class_ids": [int(c) for c in stats.get("class_ids", class_ids)],
        "class_names": {str(k): v for k, v in names.items()},
        "coco_category_of": stats.get("coco_category_of", {}),
        "epochs": int(config.get("epochs", 80)),
        "batch_size": batch,
        "grad_accum_steps": grad_accum,
        "lr": float(config.get("instance_lr", 1e-4)),
        # The scheduler's device claim must be the device rfdetr actually
        # uses; the child pins it via CUDA_VISIBLE_DEVICES before any CUDA init.
        "device": resolved_device,
    }
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_instance_train_subprocess_worker,
        args=(str(dataset_dir), str(run_path), params, str(logs_path), str(stop_file)),
    )
    proc.start()
    log_fn(f"[instance] training subprocess started (pid={proc.pid})\n")

    # Stream per-epoch progress into the run log, like semantic training.
    # The child's stdout (rich progress bars) is not forwarded; instead the
    # monitor tails lightning's metrics.csv and emits one line per epoch
    # whose validation pass has completed.
    from segcore.instseg.train_rfdetr import read_epoch_val_metrics

    metrics_csv = run_path / "rfdetr" / "metrics.csv"
    total_epochs = int(params["epochs"])
    # "Epoch N/M" (capital E) is load-bearing: training_status parses it from
    # the log tail to drive the UI progress bar, same as semantic runs.
    epoch_state: dict[str, float] = {"last": -1, "best": -1.0, "mtime": 0.0,
                                     "t_first": 0.0, "e_first": -1}

    def _fmt_dur(sec: float) -> str:
        h, m = int(sec // 3600), int(sec % 3600 // 60)
        return f"{h}h{m:02d}m" if h else f"{m}m"

    def _emit_new_epochs() -> None:
        try:
            mtime = metrics_csv.stat().st_mtime
        except OSError:
            return
        if mtime <= epoch_state["mtime"]:
            return
        epoch_state["mtime"] = mtime
        for r in read_epoch_val_metrics(metrics_csv, int(epoch_state["last"])):
            now = time.monotonic()
            if epoch_state["e_first"] < 0:
                epoch_state["t_first"], epoch_state["e_first"] = now, r["epoch"]
            star = ""
            if r["segm_map"] is not None and r["segm_map"] > epoch_state["best"]:
                epoch_state["best"] = r["segm_map"]
                star = "  *best"
            parts = []
            if r.get("train_loss") is not None:
                parts.append(f"loss={r['train_loss']:.2f}")
            parts.append(f"segm mAP50-95={r['segm_map']:.3f}")
            if r.get("segm_map50") is not None:
                parts.append(f"mAP50={r['segm_map50']:.3f}")
            if r.get("f1") is not None:
                parts.append(f"F1={r['f1']:.3f}")
            span = r["epoch"] - epoch_state["e_first"]
            if span >= 1:
                avg = (now - epoch_state["t_first"]) / span
                remaining = avg * max(0, total_epochs - (r["epoch"] + 1))
                parts.append(f"ETA {_fmt_dur(remaining)}")
            log_fn(f"[instance] Epoch {r['epoch'] + 1}/{total_epochs}: "
                   + " ".join(parts) + star + "\n")
            epoch_state["last"] = r["epoch"]

    stop_seen_at: float | None = None
    next_heartbeat = time.monotonic() + 5.0
    while proc.is_alive():
        if stop_event.is_set():
            if not stop_file.exists():
                stop_file.write_text("stop", encoding="utf-8")
            if stop_seen_at is None:
                stop_seen_at = time.monotonic()
            elif time.monotonic() - stop_seen_at > _STOP_GRACE_SECONDS:
                log_fn("[instance] stop requested — terminating training subprocess\n")
                proc.terminate()
                break
        now = time.monotonic()
        if resolved_device and now >= next_heartbeat:
            try:
                touch_torch_device_claim(resolved_device, owner_id=run_id, worker_pid=proc.pid)
            except Exception:
                pass
            next_heartbeat = now + 5.0
        _emit_new_epochs()
        proc.join(timeout=5)
    proc.join(timeout=30)
    # Final epochs land in metrics.csv right before the child exits — flush
    # whatever the in-loop polling has not reported yet.
    _emit_new_epochs()

    if stop_event.is_set():
        log_fn("[instance] stopped by user\n")
        return
    if proc.exitcode != _TRAIN_EXIT_OK:
        raise RuntimeError(f"instance training subprocess exited with code {proc.exitcode}")
    log_fn("[instance] [PHASE 3/3] 学習完了 (metrics.json written)\n")
