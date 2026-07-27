# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Checkpoint adaptation: convert pretrained weights to UNet format.

Currently supports STDC checkpoints only. For feature-level transfer from
other architectures, use the DINOv2 or SAM2 teachers in
``segcore.training.distill``.
"""
from __future__ import annotations

import torch


def _resolve_device(requested: str) -> tuple[torch.device, str]:
    req = (requested or "cpu").strip().lower()
    if req == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu"), "cpu"
        parts = req.split(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx < 0 or idx >= torch.cuda.device_count():
                return torch.device("cpu"), "cpu"
            return torch.device(f"cuda:{idx}"), f"cuda:{idx}"
        idx = torch.cuda.current_device()
        return torch.device(f"cuda:{idx}"), f"cuda:{idx}"
    return torch.device("cpu"), "cpu"


def _resolve_active_class_ids(num_classes: int, active_class_ids: list[int] | None) -> list[int]:
    if active_class_ids is None:
        return list(range(num_classes))
    resolved = sorted({int(v) for v in active_class_ids if 0 <= int(v) < num_classes})
    if 0 not in resolved:
        resolved.insert(0, 0)
    return resolved


def _suppress_inactive_logits(
    logits: torch.Tensor,
    active_class_ids: list[int],
) -> torch.Tensor:
    channels = logits.shape[1]
    mask = torch.ones(channels, dtype=torch.bool, device=logits.device)
    mask[active_class_ids] = False
    if not torch.any(mask):
        return logits
    logits = logits.clone()
    logits[:, mask, :, :] = -1e4
    return logits


def _repeat_or_truncate_channels(t: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    current = t.shape[dim]
    if current == target:
        return t
    if current > target:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice(0, target)
        return t[tuple(slices)]
    repeat = (target + current - 1) // current
    reps = [1] * t.dim()
    reps[dim] = repeat
    expanded = t.repeat(*reps)
    slices = [slice(None)] * expanded.dim()
    slices[dim] = slice(0, target)
    return expanded[tuple(slices)]


def _fit_conv_weight(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if src.dim() != 4 or len(target_shape) != 4:
        return None
    out_ch, in_ch, kh, kw = target_shape
    w = src.float()
    w = _repeat_or_truncate_channels(w, out_ch, dim=0)
    w = _repeat_or_truncate_channels(w, in_ch, dim=1)
    sh, sw = w.shape[2], w.shape[3]
    if (sh, sw) == (kh, kw):
        return w
    if sh >= kh and sw >= kw:
        top = (sh - kh) // 2
        left = (sw - kw) // 2
        return w[:, :, top : top + kh, left : left + kw]
    out = torch.zeros((out_ch, in_ch, kh, kw), dtype=w.dtype)
    top = (kh - sh) // 2
    left = (kw - sw) // 2
    out[:, :, top : top + sh, left : left + sw] = w
    return out


def _strip_common_prefix(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not raw_state:
        return raw_state
    keys = list(raw_state.keys())
    # Fast path: uniform prefix.
    for prefix in ("module.", "model.", "student."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix) :]: v for k, v in raw_state.items()}
    # Mixed checkpoints appear in practice (e.g. some keys prefixed, some not).
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state.items():
        out_key = key
        for prefix in ("module.", "model.", "student."):
            if out_key.startswith(prefix):
                out_key = out_key[len(prefix) :]
                break
        normalized[out_key] = tensor
    return normalized


def _build_stdc_to_unet_init(
    raw_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    num_classes: int,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    conv_map = {
        "enc1.0.weight": "stem.0.weight",
        "enc1.2.weight": "stem.3.weight",
        "enc2.0.weight": "stage2.0.block.6.weight",
        "enc2.2.weight": "stage2.1.block.6.weight",
        "enc3.0.weight": "stage3.1.block.6.weight",
        "enc3.2.weight": "head.0.weight",
    }
    for dst_key, src_key in conv_map.items():
        if dst_key not in model_state or src_key not in raw_state:
            continue
        fitted = _fit_conv_weight(raw_state[src_key], model_state[dst_key].shape)
        if fitted is not None:
            converted[dst_key] = fitted.to(dtype=model_state[dst_key].dtype)
    if "head.weight" in model_state and "head.3.weight" in raw_state:
        fitted_head = _fit_conv_weight(raw_state["head.3.weight"], model_state["head.weight"].shape)
        if fitted_head is not None:
            converted["head.weight"] = fitted_head.to(dtype=model_state["head.weight"].dtype)
    for key in model_state.keys():
        if key.endswith(".bias") and key not in converted:
            converted[key] = torch.zeros_like(model_state[key])
    return converted


# NOTE: For pretraining-style weight transfer use the STDC adapter above,
# or use feature distillation against DINOv2 / SAM2 via
# ``segcore.training.distill``.
