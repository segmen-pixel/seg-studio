# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Auto-config utilities — dataset stats, query profile, epoch budgeting.

Extracted from training_runner.py — pure refactoring, no behaviour change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Imported lazily at function scope at runtime; annotation-only here.
    from segcore.auto_select.schema import ProjectProfile


def compute_basic_stats_fallback(
    images_dir: Path,
    masks_dir: Path,
    report_path: Path | None = None,
) -> dict[str, float]:
    """Compute the basic dataset stats the ML combo predictor expects.

    Mirrors the fields the ablation library was built with: num_train/val/total,
    mean_width/height, fg_ratio, mean_fg_area_px, std_fg_area_px,
    mean_fg_ratio_per_image, fg_area_frac, num_active_classes,
    class_imbalance_ratio, log_num_train, log_img_pixels.

    Used when prepared_dir/dataset_stats.json is missing (which it currently
    is for seg-studio annotate flows).
    """
    import math as _math

    from PIL import Image as _Image

    if not (images_dir.exists() and masks_dir.exists()):
        return {}
    img_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp")
    )
    mask_files = sorted(
        f for f in masks_dir.iterdir()
        if f.suffix.lower() in (".png", ".bmp", ".tif")
    )
    if not img_files or not mask_files:
        return {}

    # Split counts from report.json if available
    num_train = len(img_files)
    num_val = 0
    num_total = len(img_files)
    if report_path is not None and report_path.exists():
        try:
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            num_train = int(rep.get("train_count", num_train))
            num_val = int(rep.get("val_count", num_val))
            num_total = int(rep.get("total_items", num_total))
        except Exception:
            pass

    widths, heights = [], []
    for p in img_files[:30]:
        try:
            with _Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue

    fg_areas: list[int] = []
    fg_ratios: list[float] = []
    class_pixel_counts: dict[int, int] = {}
    for mf in mask_files[:60]:
        try:
            arr = np.array(_Image.open(mf).convert("L"))
            total_px = int(arr.shape[0] * arr.shape[1])
            fg_mask = (arr > 0) & (arr != 255)
            fg_px = int(fg_mask.sum())
            if fg_px > 0:
                fg_areas.append(fg_px)
                fg_ratios.append(fg_px / total_px)
            for cls_id in np.unique(arr):
                if cls_id == 255:
                    continue
                class_pixel_counts[int(cls_id)] = class_pixel_counts.get(int(cls_id), 0) + int((arr == cls_id).sum())
        except Exception:
            continue

    mean_w = float(np.mean(widths)) if widths else 0.0
    mean_h = float(np.mean(heights)) if heights else 0.0
    img_px = mean_w * mean_h
    mean_fg_area = float(np.mean(fg_areas)) if fg_areas else 0.0
    std_fg_area = float(np.std(fg_areas, ddof=0)) if len(fg_areas) > 1 else 0.0
    fg_ratio = float(np.mean(fg_ratios)) if fg_ratios else 0.0

    all_counts = [c for c in class_pixel_counts.values() if c > 0]
    num_active_classes = float(len(class_pixel_counts))
    class_imbalance = 0.0
    if len(all_counts) >= 2:
        class_imbalance = max(all_counts) / max(1.0, min(all_counts))

    stats: dict[str, float] = {
        "num_train": float(num_train),
        "num_val": float(num_val),
        "num_total": float(num_total),
        "mean_width": mean_w,
        "mean_height": mean_h,
        "fg_ratio": fg_ratio,
        "mean_fg_area_px": mean_fg_area,
        "std_fg_area_px": std_fg_area,
        "mean_fg_ratio_per_image": fg_ratio,
        "num_active_classes": num_active_classes,
        "class_imbalance_ratio": class_imbalance,
    }
    if num_train > 0:
        stats["log_num_train"] = _math.log1p(num_train)
    if img_px > 0:
        stats["log_img_pixels"] = _math.log(img_px)
        if mean_fg_area > 0:
            stats["fg_area_frac"] = mean_fg_area / img_px
    return stats


def build_query_profile(
    project_id: str,
    images_dir: Path,
    masks_dir: Path,
    arch: str = "simpleunet",
    base_channels: int = 64,
    dataset_stats: dict | None = None,
) -> ProjectProfile:
    """Build a ProjectProfile query from images/masks directories.

    Used by the model-search API endpoint (explicit transfer learning).
    """
    from segcore.auto_select.schema import ProjectProfile, features_to_handcrafted

    if dataset_stats is None:
        dataset_stats = {}

    # If dataset_stats is empty, compute basic stats from masks directly
    if not dataset_stats and masks_dir.exists() and images_dir.exists():
        try:
            from PIL import Image
            img_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp"))
            mask_files = sorted(f for f in masks_dir.iterdir() if f.suffix.lower() in (".png", ".bmp", ".tif"))
            if img_files:
                sample_imgs = img_files[:30]
                widths, heights, fg_areas, fg_ratios = [], [], [], []
                for img_path in sample_imgs:
                    try:
                        im = Image.open(img_path)
                        w, h = im.size
                        widths.append(w)
                        heights.append(h)
                    except Exception:
                        continue
                n_with_fg = 0
                for mf in mask_files[:50]:
                    try:
                        marr = np.array(Image.open(mf).convert("L"))
                        fg_px = int(((marr > 0) & (marr != 255)).sum())
                        total_px = marr.shape[0] * marr.shape[1]
                        if fg_px > 0:
                            n_with_fg += 1
                            fg_areas.append(fg_px)
                            fg_ratios.append(fg_px / total_px)
                    except Exception:
                        continue
                dataset_stats = {
                    "num_train": len(img_files),
                    "num_total": len(img_files),
                    "mean_width": sum(widths) / len(widths) if widths else 0,
                    "mean_height": sum(heights) / len(heights) if heights else 0,
                    "fg_ratio": sum(fg_ratios) / len(fg_ratios) if fg_ratios else 0,
                    "mean_fg_area_px": sum(fg_areas) / len(fg_areas) if fg_areas else 0,
                    "num_active_classes": 2,
                }
        except Exception:
            pass

    try:
        from scripts.make_autoalgorithm.features import (
            compute_image_features,
            extract_basic_features,
        )
        basic = extract_basic_features(dataset_stats)
        if images_dir.exists() and masks_dir.exists():
            basic.update(compute_image_features(images_dir, masks_dir, max_samples=30))
        handcrafted = features_to_handcrafted(basic)
    except Exception:
        from segcore.auto_select.schema import HANDCRAFTED_KEYS
        handcrafted = np.zeros(len(HANDCRAFTED_KEYS), dtype=np.float32)
        basic = {}

    return ProjectProfile(
        project_id=project_id,
        run_id="query",
        arch=arch,
        base_channels=base_channels,
        handcrafted=handcrafted,
        meta={
            "num_train": basic.get("num_train", 0),
            "num_active_classes": basic.get("num_active_classes", 2),
            "mean_fg_area_px": basic.get("mean_fg_area_px", 0),
        },
    )


# Default from-scratch epoch budget used when no image size info is available.
_DEFAULT_SCRATCH_EPOCHS = 80


def _recommend_scratch_epochs(min_width: float | int | None) -> int:
    """Pick from-scratch epoch budget from image min_width.

    Derived from a 37-project ablation (wave6 sweep): min_width was the
    strongest predictor of the project's median best_epoch (spearman ρ=+0.50);
    other features (num_train, fg_ratio, mean_fg_area_px) added no signal once
    min_width was in the tree. A depth-2 decision tree converged on these two
    splits and stayed stable across LOOCV: deeper trees overfit (n=37).

      min_width < 1000 px  → 60 epochs  (small; ~33% projects still growing at 80)
      min_width < 2000 px  → 80 epochs  (mid; ~53% still growing — legacy default)
      else                 → 100 epochs (large; ~70% still growing — undershot)

    Returns the legacy default when min_width is missing/invalid so the caller
    can keep its old behaviour unchanged.

    See ``docs/auto-config-rationale.md`` for the full sweep design, dataset
    scope, paired-comparison statistics for other axes (lr, distill, loss),
    and the limitations of the rule.
    """
    try:
        mw = float(min_width) if min_width is not None else 0.0
    except (TypeError, ValueError):
        return _DEFAULT_SCRATCH_EPOCHS
    if mw <= 0:
        return _DEFAULT_SCRATCH_EPOCHS
    if mw < 1000:
        return 60
    if mw < 2000:
        return 80
    return 100


def save_training_profile(
    project_id: str,
    run_id: str,
    run_path: Path,
    prepared_dir: Path,
    config: dict,
    log_fn,
) -> None:
    """Save a feature profile after successful training for the transfer library."""
    try:
        from segcore.auto_select.profile_io import save_profile
        from segcore.auto_select.schema import ProjectProfile, features_to_handcrafted
    except ImportError:
        return

    metrics_path = run_path / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"
    try:
        from scripts.make_autoalgorithm.features import (
            compute_image_features,
            extract_basic_features,
        )
        dataset_stats = metrics.get("dataset_stats", {})
        basic = extract_basic_features(dataset_stats)
        if images_dir.exists() and masks_dir.exists():
            basic.update(compute_image_features(images_dir, masks_dir, max_samples=30))
        handcrafted = features_to_handcrafted(basic)
    except Exception:
        from segcore.auto_select.schema import HANDCRAFTED_KEYS
        handcrafted = np.zeros(len(HANDCRAFTED_KEYS), dtype=np.float32)
        basic = {}

    model_pt = run_path / "model.pt"
    profile = ProjectProfile(
        project_id=project_id,
        run_id=run_id,
        arch=str(config.get("arch", "simpleunet")),
        base_channels=int(config.get("base_channels", 64)),
        output_stride=int(config.get("output_stride", 2)),
        patch_size=int(config.get("patch_size", 256)),
        patches_per_image=int(config.get("patches_per_image", 8)),
        fg_patch_prob=float(config.get("fg_patch_prob", 0.5)),
        loss_type=str(config.get("loss_type", "focal")),
        distill_mode=str(config.get("distill_mode", "off")),
        best_f1=float(metrics.get("best_F1_val", 0)),
        best_miou=float(metrics.get("best_mIoU_val", 0)),
        best_epoch=int(metrics.get("best_epoch", 0)),
        total_epochs=int(metrics.get("total_epochs", metrics.get("epoch", 0))),
        handcrafted=handcrafted,
        meta={
            "num_train": basic.get("num_train", 0),
            "num_active_classes": basic.get("num_active_classes", 2),
            "mean_fg_area_px": basic.get("mean_fg_area_px", 0),
        },
        checkpoint_path=str(model_pt) if model_pt.exists() else "",
    )

    save_profile(profile, run_path)
    log_fn(f"Auto-select: profile saved to {run_path / 'feature_profile.npz'}\n")
