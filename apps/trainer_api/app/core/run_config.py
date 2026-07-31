# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Readers for per-run training configuration (train_config.json).

Each helper reads a single field from a run directory's ``train_config.json``
and returns a validated value with a sensible default.  These are used by both
the prediction engine and the inference runtime to reconstruct the model /
sliding-window parameters that were used during training.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import FIXED_INPUT_SIZE, NUM_CLASSES, OUTPUT_STRIDE, read_num_classes


def _load_run_input_size(run_path: Path) -> tuple[int, int]:
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return int(FIXED_INPUT_SIZE[0]), int(FIXED_INPUT_SIZE[1])
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = payload.get("input_size")
        if isinstance(raw, list) and len(raw) == 2:
            w = int(raw[0])
            h = int(raw[1])
            if w > 0 and h > 0:
                return w, h
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return int(FIXED_INPUT_SIZE[0]), int(FIXED_INPUT_SIZE[1])


def _load_run_output_stride(run_path: Path) -> int:
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return int(OUTPUT_STRIDE)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = int(payload.get("output_stride", OUTPUT_STRIDE))
        if raw in (1, 2, 4):
            return raw
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return int(OUTPUT_STRIDE)


def _load_run_patch_size(run_path: Path) -> int:
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return 0
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = int(payload.get("patch_size", 0))
        if raw > 0:
            return raw
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return 0


def _load_run_configured_sw_stride(run_path: Path) -> int:
    """Return the sliding-window stride the run persisted, or 0 if it has none."""
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return 0
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = int(payload.get("sw_stride", 0))
        if raw > 0:
            return raw
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return 0


def _load_run_sw_stride(run_path: Path, *, realtime: bool = False) -> int:
    """Return the sliding-window stride for inference.

    Post-training stride optimisation measures the candidate strides on the
    run's own validation set and writes the winner back into train_config.json
    ("so inference can use them", train.py), and the shipped inference
    threshold is then calibrated at that same geometry.  Reading it back is
    what keeps inference on the geometry the run was tuned for.  The 3/4-patch
    default it replaces is only equivalent when the foreground is dense: run
    9d29d4f37df4 (fg 1.9%) measured F1 0.9714 at the stride it chose, 64,
    against 0.9176 at 192 -- 5.4 points thrown away by ignoring its own answer.

    ``realtime=True`` asks for the coarse ``patch_size * 3 // 4`` default
    instead.  The camera path runs the window on every frame and a 3x finer
    stride is 9x the patches; there the frame rate is the requirement.
    """
    patch_size = _load_run_patch_size(run_path)
    output_stride = _load_run_output_stride(run_path)
    if patch_size <= 0:
        return 0
    stride = 0 if realtime else _load_run_configured_sw_stride(run_path)
    if stride <= 0 or stride > patch_size:
        stride = patch_size * 3 // 4
    stride = max(output_stride, stride - stride % output_stride)
    return stride if stride > 0 else 0


def _load_run_base_channels(run_path: Path) -> int:
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return 32
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = int(payload.get("base_channels", 32))
        if 8 <= raw <= 128:
            return raw
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return 32


def _load_run_arch(run_path: Path) -> str:
    """Read model architecture from run's train_config.json."""
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return "simpleunet"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = payload.get("arch", "simpleunet")
        if raw in ("simpleunet", "stdc"):
            return raw
    except (json.JSONDecodeError, OSError):
        pass
    return "simpleunet"


def _load_run_train_size(run_path: Path) -> list[int] | None:
    """Read train_size [W, H] from train_config.json (set for resized projects).

    Falls back to resize_scale for backward compatibility with older projects.
    """
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        ts = payload.get("train_size")
        if ts is not None and len(ts) == 2:
            return [int(ts[0]), int(ts[1])]
        # Backward compat: old resize_scale format (ratio-based)
        raw = payload.get("resize_scale")
        if raw is not None:
            val = float(raw)
            if 0.1 <= val < 1.0:
                # Cannot determine absolute size from ratio alone;
                # return None so inference uses original size
                return None
    except (json.JSONDecodeError, OSError, ValueError, TypeError, IndexError):
        pass
    return None


def _load_run_inference_threshold(run_path: Path) -> float | None:
    """Read optimal inference threshold from train_config.json (set by training)."""
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw = payload.get("inference_threshold")
        if raw is not None:
            val = float(raw)
            if 0.0 < val < 1.0:
                return val
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return None


def _load_run_num_classes(run_path: Path) -> int:
    """Read num_classes from run's train_config.json, classes.json, or checkpoint."""
    config_path = run_path / "train_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            raw = payload.get("num_classes")
            if isinstance(raw, int) and raw > 0:
                return raw
        except (json.JSONDecodeError, OSError):
            pass
    cls_path = run_path / "classes.json"
    if cls_path.exists():
        try:
            classes = json.loads(cls_path.read_text(encoding="utf-8"))
            return read_num_classes(classes)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    # Fallback: infer from checkpoint head weight shape
    model_path = run_path / "model.pt"
    if model_path.exists():
        try:
            import torch
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            for key in ("head.weight", "head.bias", "head2.weight", "head2.bias"):
                if key in state:
                    return int(state[key].shape[0])
        except Exception:
            pass
    return NUM_CLASSES
