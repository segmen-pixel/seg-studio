#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""CLI training runner for seg-studio.

Usage:
    python scripts/cli_train.py --project <project_id> [options]

Examples:
    # Basic training
    python scripts/cli_train.py --project 09e4bc73-...

    # With distillation
    python scripts/cli_train.py --project 09e4bc73-... --distill feature

    # Custom hyperparameters
    python scripts/cli_train.py --project 09e4bc73-... --lr 5e-4 --loss focal --dice-weight 2.0

    # Sweep: try multiple configs
    python scripts/cli_train.py --project 09e4bc73-... --distill feature --sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# segcore + trainer_api are installed via `pip install -e`.
# Fallback: add to sys.path if not installed as packages.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
try:
    import segcore  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "packages" / "segcore"))
try:
    import app.core.config  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "apps" / "trainer_api"))

import numpy as np

from app.core.config import (
    FIXED_INPUT_SIZE,
    IGNORE_INDEX,
    NORMALIZE,
    OUTPUT_STRIDE,
    read_num_classes,
)
from app.core.paths import classes_path, pretrained_model_path, project_dir
from app.core.paths import prepared_dir as _prepared_dir_fn
from segcore.training.train import TrainConfig, train


def log_print(msg: str) -> None:
    print(msg, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CLI training runner")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loss", choices=["ce", "focal", "lovasz"], default="focal")
    parser.add_argument("--dice-weight", type=float, default=None, help="None=auto")
    parser.add_argument("--class-weight-strength", type=float, default=0.80)
    parser.add_argument("--bg-boost", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, nargs=2, default=None)
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size (0=no patching, use full image)")
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--fg-patch-prob", type=float, default=0.5)
    parser.add_argument("--sw-stride", type=int, default=0,
                        help="Sliding-window validation stride (0=auto: patch_size*3/4)")
    parser.add_argument("--annotation-patches", action="store_true", default=True,
                        help="Only sample patches around annotated regions (default: True)")
    parser.add_argument("--no-annotation-patches", action="store_true",
                        help="Disable annotation-based patch sampling")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--arch", choices=["simpleunet", "stdc", "deeplabv3plus"],
                        default="simpleunet", help="Model architecture")
    parser.add_argument("--no-se", action="store_true",
                        help="Disable SE (Squeeze-and-Excitation) attention blocks")
    parser.add_argument("--ohem-ratio", type=float, default=0.0,
                        help="OHEM: keep top N%% hardest pixels (0=off, 0.75=top 75%%)")
    parser.add_argument("--early-stopping", type=int, default=15,
                        help="Early stopping patience (0=disable)")
    parser.add_argument("--hnm-interval", type=int, default=5,
                        help="Hard negative mining interval in epochs (1=every epoch, 5=default)")
    parser.add_argument("--min-area", type=int, default=0,
                        help="Post-processing: min connected component area in pixels (0=off)")
    parser.add_argument("--output-stride", type=int, default=OUTPUT_STRIDE)
    parser.add_argument("--distill", choices=["off", "feature", "channel"], default="off")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-loss", choices=["smooth_l1", "mse", "cosine"], default="smooth_l1")
    parser.add_argument("--distill-tap", default="s1")
    parser.add_argument("--distill-teacher-dir", default=None,
                        help="Online distillation teacher selector (Apache-2.0 only). "
                             "Use 'dinov2_vitb14', 'dinov2_vitl14', or 'sam2.1_hiera_*'.")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained checkpoint if available (default: True)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--auto-batch", action="store_true",
                        help="Profile GPU VRAM and auto-set optimal batch_size (target ~80%% usage)")
    parser.add_argument("--no-auto-adjust", action="store_true",
                        help="Disable VRAM-based batch_size auto-adjustment (use --batch-size as-is)")
    parser.add_argument("--run-id", default=None, help="Custom run ID (default: auto-generate)")
    parser.add_argument("--tag", default="", help="Tag for this run (appended to log)")
    args = parser.parse_args()

    pid = args.project
    base = project_dir(pid)
    if not base.exists():
        print(f"ERROR: Project {pid} not found at {base}")
        sys.exit(1)

    prepared_dir = _prepared_dir_fn(pid)
    if not (prepared_dir / "images").exists():
        print(f"ERROR: No prepared dataset at {prepared_dir}. Run dataset prepare from GUI first.")
        sys.exit(1)

    # Resolve classes
    cls_file = classes_path(pid)
    if not cls_file.exists():
        print(f"ERROR: classes.json not found for project {pid}")
        sys.exit(1)
    classes = json.loads(cls_file.read_text(encoding="utf-8"))
    num_classes = read_num_classes(classes)
    active_class_ids = []
    for c in classes.get("classes", []):
        if c.get("id", 0) != 0 and c.get("active", True):
            active_class_ids.append(int(c["id"]))
    if not active_class_ids:
        # Fallback: all non-background classes up to num_classes
        active_class_ids = list(range(1, num_classes))

    # Resolve input size
    input_size = args.input_size or list(FIXED_INPUT_SIZE)

    # Auto-adjust batch_size based on VRAM (2-param model: base + per_sample * batch)
    from app.core.torch_device import _batch_limit_for_input, _cuda_free_memory_mb, _cuda_total_memory_mb
    memory_mb = _cuda_total_memory_mb(args.device)
    free_mb = _cuda_free_memory_mb(args.device)
    if memory_mb is not None and not args.no_auto_adjust:
        effective_inp = args.patch_size if args.patch_size > 0 else max(input_size)
        max_batch = _batch_limit_for_input(
            effective_inp, memory_mb,
            arch=args.arch,
            base_channels=args.base_channels,
            output_stride=args.output_stride,
            distill_mode=args.distill,
            free_mb=free_mb,
        )
        # Cap up: raise batch if VRAM allows (CLI always maximizes GPU usage)
        if args.batch_size != max_batch:
            print(f"Auto-adjust batch_size: {args.batch_size} -> {max_batch} "
                  f"(effective_input={effective_inp}, VRAM={memory_mb}MB, "
                  f"free={free_mb or '?'}MB, arch={args.arch})")
            # Scale patches_per_image proportionally to keep steps/epoch constant
            old_batch = args.batch_size
            new_patches = max(1, args.patches_per_image * max_batch // max(1, old_batch))
            if new_patches != args.patches_per_image:
                print(f"Auto-adjust patches_per_image: {args.patches_per_image} -> {new_patches} "
                      f"(keep steps/epoch constant)")
                args.patches_per_image = new_patches
            args.batch_size = max_batch

    # Auto-batch profiling (optional, more precise but slower)
    if args.auto_batch:
        from app.core.torch_device import _profile_max_batch_size
        # Use patch_size for profiling when patch training is enabled
        profile_input = [args.patch_size, args.patch_size] if args.patch_size > 0 else input_size
        print(f"Profiling VRAM on {args.device} for input={profile_input}...")
        profiled_batch = _profile_max_batch_size(
            device=args.device,
            input_size=profile_input,
            num_classes=num_classes,
            base_channels=args.base_channels,
            output_stride=args.output_stride,
            distill_mode=args.distill,
        )
        print(f"  Profiled max batch_size: {profiled_batch} (was {args.batch_size})")
        args.batch_size = profiled_batch

    # Resolve pretrained
    pretrained_ckpt = None
    if args.pretrained and not args.no_pretrained:
        p = pretrained_model_path(pid)
        if p.exists():
            pretrained_ckpt = str(p)

    # Resolve distillation teacher cache (skip if online distillation via --distill-teacher-dir)
    distill_teacher_cache = None
    if args.distill != "off" and not args.distill_teacher_dir:
        cache_dir = prepared_dir / "teacher_cache"
        if cache_dir.exists() and (cache_dir / "teacher_meta.json").exists():
            distill_teacher_cache = str(cache_dir)
        else:
            print(f"WARNING: distill={args.distill} but teacher cache not found at {cache_dir}")
            print("Run distill precompute first. Falling back to distill=off.")
            args.distill = "off"

    # Create run directory
    run_id = args.run_id or str(uuid.uuid4())
    run_path = base / "training" / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    # Estimate foreground ratio from masks
    from segcore.training.train import load_split_ids
    splits_dir = prepared_dir / "splits"
    train_ids = load_split_ids(splits_dir / "train.txt")
    val_ids = load_split_ids(splits_dir / "val.txt")

    from app.core.training_runner import estimate_foreground_ratio
    fg_ratio, fg_sampled = estimate_foreground_ratio(
        prepared_dir / "masks", train_ids, ignore_index=IGNORE_INDEX
    )

    print("=" * 60)
    print("CLI Training Runner")
    print("=" * 60)
    print(f"Project:        {pid[:12]}...")
    print(f"Run:            {run_id[:12]}...")
    print(f"Architecture:   {args.arch}")
    print(f"Input size:     {input_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"LR:             {args.lr}")
    print(f"Loss:           {args.loss}")
    print(f"Dice weight:    {args.dice_weight or 'auto'}")
    print(f"OHEM ratio:     {args.ohem_ratio or 'off'}")
    print(f"CW strength:    {args.class_weight_strength}")
    print(f"BG boost:       {args.bg_boost}")
    print(f"Patch size:     {args.patch_size}")
    print(f"Patches/img:    {args.patches_per_image}")
    print(f"FG patch prob:  {args.fg_patch_prob}")
    print(f"Annot. patches: {args.annotation_patches and not args.no_annotation_patches}")
    print("Context expand: 3.0 (fixed)")
    print(f"SW stride:      {args.sw_stride or 'auto'}")
    print("Crop FG:        off (fixed)")
    print(f"Early stop:     {args.early_stopping}")
    print(f"Pretrained:     {pretrained_ckpt is not None}")
    print(f"Distill:        {args.distill}")
    if args.distill != "off":
        print(f"  weight:       {args.distill_weight}")
        print(f"  loss:         {args.distill_loss}")
        print(f"  tap:          {args.distill_tap}")
    print(f"Train/Val:      {len(train_ids)}/{len(val_ids)}")
    print(f"FG ratio:       {fg_ratio*100:.2f}%")
    print(f"Num classes:    {num_classes}")
    print(f"Active classes: {active_class_ids}")
    print(f"Base channels:  {args.base_channels}")
    print(f"SE attention:   {not args.no_se}")
    print(f"Device:         {args.device}")
    if args.tag:
        print(f"Tag:            {args.tag}")
    print("=" * 60)
    print()

    config = TrainConfig(
        input_size=input_size,
        output_stride=args.output_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ignore_index=IGNORE_INDEX,
        normalize=NORMALIZE,
        crop_foreground=False,
        crop_scale=0.7,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        fg_patch_prob=args.fg_patch_prob,
        augment_enabled=True,
        augment_hflip_prob=0.5,
        augment_vflip_prob=0.0,
        augment_rotate90_prob=0.25,
        augment_brightness=0.15,
        augment_contrast=0.15,
        augment_noise_std=0.02,
        pretrained_checkpoint=pretrained_ckpt,
        use_class_weights=True,
        class_weight_strength=args.class_weight_strength,
        background_weight_boost=args.bg_boost,
        early_stopping_patience=args.early_stopping,
        min_epochs=5,
        active_class_ids=[0] + active_class_ids,
        device=args.device,
        foreground_ratio=fg_ratio,
        loss_type=args.loss,
        dice_weight=args.dice_weight,
        distill_mode=args.distill,
        distill_teacher_cache_dir=distill_teacher_cache,
        distill_feature_weight=args.distill_weight,
        distill_feature_loss=args.distill_loss,
        distill_feature_tap=args.distill_tap,
        base_channels=args.base_channels,
        use_se=not args.no_se,
        sw_stride=args.sw_stride,
        annotation_patches_only=args.annotation_patches and not args.no_annotation_patches,
        context_expand=3.0,
        arch=args.arch,
        ohem_ratio=args.ohem_ratio,
        hnm_interval=args.hnm_interval,
    )
    # Post-processing
    config.postprocess_min_area = args.min_area
    # Online distillation: set teacher model dir
    if args.distill_teacher_dir:
        config.distill_teacher_model_dir = args.distill_teacher_dir

    # Copy classes.json to run dir
    import shutil
    shutil.copy2(str(cls_file), str(run_path / "classes.json"))

    # Save train config
    config_dump = {
        "arch": args.arch,
        "num_classes": num_classes,
        "input_size": input_size,
        "output_stride": args.output_stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss_type": args.loss,
        "dice_weight": args.dice_weight,
        "class_weight_strength": args.class_weight_strength,
        "background_weight_boost": args.bg_boost,
        "distill_mode": args.distill,
        "distill_feature_weight": args.distill_weight,
        "pretrained_path": pretrained_ckpt,
        "base_channels": args.base_channels,
        "use_se": not args.no_se,
        "patch_size": args.patch_size,
        "patches_per_image": args.patches_per_image,
        "fg_patch_prob": args.fg_patch_prob,
        "annotation_patches_only": args.annotation_patches and not args.no_annotation_patches,
        "context_expand": 3.0,
        "crop_foreground": False,
        "crop_scale": 0.7,
        "sw_stride": args.sw_stride,
        "early_stopping_patience": args.early_stopping,
        "fg_ratio": fg_ratio,
        "tag": args.tag,
        "created_at": datetime.utcnow().isoformat(),
    }
    (run_path / "train_config.json").write_text(
        json.dumps(config_dump, indent=2), encoding="utf-8"
    )

    # Train!
    try:
        metrics = train(
            prepared_dir=prepared_dir,
            run_dir=run_path,
            num_classes=num_classes,
            config=config,
            log_fn=log_print,
        )
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best val F1:  {metrics.get('best_F1_val', 'N/A')}")
        print(f"Best epoch:   {metrics.get('best_epoch', 'N/A')}")
        print(f"Run dir:      {run_path}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
