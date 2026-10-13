# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Knowledge distillation support for segcore training.

Supported online teachers:
  - DINOv2 (segcore.dinov2_loader: vendored model definition, published weights)
  - SAM2 image encoder (from the bundled SAM2 package)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeatureProjector(nn.Module):
    """Project student features to teacher feature space via 1x1 conv.

    Student e3 has 128 channels; teacher s1 has 64 channels.
    A single 1x1 convolution (no bias) aligns the channel dimensions
    while keeping spatial dimensions unchanged.
    """

    def __init__(self, student_ch: int = 128, teacher_ch: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def feature_distillation_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    projector: FeatureProjector,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Compute feature distillation loss between student and teacher.

    Args:
        student_feat: Student feature map [B, Cs, H, W].
        teacher_feat: Teacher feature map [B, Ct, H, W].
        projector: 1x1 conv projecting Cs -> Ct.
        loss_type: "smooth_l1", "mse", or "cosine".

    Returns:
        Scalar loss value.
    """
    projected = projector(student_feat)  # [B, Ct, H, W]
    teacher_feat = teacher_feat.to(projected.dtype)

    # Spatial alignment safety: interpolate if sizes differ
    if projected.shape[2:] != teacher_feat.shape[2:]:
        teacher_feat = F.interpolate(
            teacher_feat,
            size=projected.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    if loss_type == "mse":
        return F.mse_loss(projected, teacher_feat)
    elif loss_type == "cosine":
        return 1.0 - F.cosine_similarity(projected, teacher_feat, dim=1).mean()
    else:  # default: smooth_l1
        return F.smooth_l1_loss(projected, teacher_feat)


# ---------------------------------------------------------------------------
# Online teacher: shared constants
# ---------------------------------------------------------------------------

_TAP_TO_INDEX = {"s1": 0, "s2": 1, "s3": 2, "s4": 3}


# ---------------------------------------------------------------------------
# DINOv2 teacher (class-agnostic feature distillation, no training needed)
# ---------------------------------------------------------------------------


class _DINOv2Wrapper(nn.Module):
    """Wraps DINOv2 to expose a ``.hidden_states`` interface.

    ``online_teacher_features()`` calls ``model(images.half())`` and reads
    ``outputs.hidden_states[tap_idx]``.  This wrapper stores the DINOv2 model
    and returns a namespace with ``.hidden_states`` so the existing call-site
    works unchanged.
    """

    def __init__(self, dino_model: nn.Module, tap_idx: int = 0):
        super().__init__()
        self.dino = dino_model
        self.tap_idx = tap_idx
        # DINOv2 patch_size (needed for input padding)
        self.patch_size = getattr(dino_model, "patch_size", 14)

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        B, C, H, W = pixel_values.shape
        ps = self.patch_size
        # Pad to nearest multiple of patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="reflect")

        feats = self.dino.get_intermediate_layers(
            pixel_values.float(), n=4, reshape=True,
        )
        # Return namespace with .hidden_states = list of [B, C, H, W]
        return _HiddenStatesResult(list(feats))


class _HiddenStatesResult:
    __slots__ = ("hidden_states",)

    def __init__(self, hidden_states: list):
        self.hidden_states = hidden_states


# ---------------------------------------------------------------------------
# SAM2 teacher (class-agnostic feature distillation, no training needed)
# ---------------------------------------------------------------------------


class _SAM2Wrapper(nn.Module):
    """Wraps SAM2 image encoder to expose a ``.hidden_states`` interface.

    SAM2 image_encoder returns backbone_fpn: list of [B, 256, H, W] at 3 scales.
    We pick the scale that best matches s1 (1/8 = 32x32 for 256px input).
    """

    def __init__(self, image_encoder: nn.Module, fpn_level: int = 1):
        super().__init__()
        self.image_encoder = image_encoder
        self.fpn_level = fpn_level  # 0=64x64, 1=32x32, 2=16x16

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        out = self.image_encoder(pixel_values.float())
        fpn_feats = out["backbone_fpn"]
        # Return all FPN levels as hidden_states (tap_idx selects which one)
        return _HiddenStatesResult(fpn_feats)


def load_sam2_teacher(
    variant: str = "sam2.1_hiera_small",
    device: str = "cuda:0",
    tap: str = "s1",
) -> tuple[nn.Module, int]:
    """Load SAM2 as online teacher for feature distillation.

    Args:
        variant: SAM2 model variant.
            "sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_b+", "sam2.1_hiera_l"
        device: CUDA device string.
        tap: Feature tap point. Maps to FPN level:
            "s1"=level 0 (64x64, 1/4), "s2"=level 1 (32x32, 1/8), "s4"=level 2 (16x16, 1/16)

    Returns:
        (wrapped_model, teacher_channels)  — channels is always 256 for SAM2 FPN.
    """
    from pathlib import Path

    from sam2.build_sam import build_sam2

    ckpt_dir = Path(__file__).resolve().parents[3] / "models" / "sam_checkpoints"

    # Map variant to config + checkpoint
    config_map = {
        "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "sam2.1_hiera_b+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "sam2.1_hiera_l": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    }
    if variant not in config_map:
        raise ValueError(f"Unknown SAM2 variant: {variant}. Choose from {list(config_map.keys())}")

    config_file, ckpt_name = config_map[variant]
    ckpt_path = ckpt_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {ckpt_path}\n"
            f"Download from https://github.com/facebookresearch/sam2#download-checkpoints"
        )

    model = build_sam2(config_file, str(ckpt_path), device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # SAM2 FPN: tap mapping (all levels are 256ch)
    # s1 -> level 1 (32x32, matches student e3 at 1/8 scale)
    # s2 -> level 0 (64x64, higher res)
    # s4 -> level 2 (16x16, lower res)
    tap_to_fpn = {"s1": 1, "s2": 0, "s3": 2, "s4": 2}
    fpn_level = tap_to_fpn.get(tap, 1)

    wrapper = _SAM2Wrapper(model.image_encoder, fpn_level).to(device)
    teacher_ch = 256  # SAM2 FPN always outputs 256ch
    return wrapper, teacher_ch


def load_dinov2_teacher(
    variant: str = "dinov2_vitb14",
    device: str = "cuda:0",
    tap: str = "s1",
) -> tuple[nn.Module, int]:
    """Load DINOv2 as online teacher for feature distillation.

    No training data or class_map needed - uses frozen pretrained features.

    Args:
        variant: DINOv2 model variant.
            "dinov2_vitb14" (86M, 768ch) or "dinov2_vitl14" (300M, 1024ch).
        device: CUDA device string.
        tap: Feature tap point ("s1"=early, "s4"=deepest).
            All DINOv2 layers have the same channel dim.

    Returns:
        (wrapped_model, teacher_channels)

    Notes:
        The backbone is built from the Apache-2.0 model definition vendored
        in segcore, never through ``torch.hub.load`` (:mod:`segcore.dinov2_loader`
        says why). The weight comes from a packaged build's
        ``models/dinov2/``, else torch's checkpoint cache, else it is
        downloaded into that cache.
    """
    from ..dinov2_loader import bundled_weight_dirs
    from ..dinov2_loader import load as _load_dinov2

    model = _load_dinov2(variant, search_dirs=bundled_weight_dirs(), device=device)
    for p in model.parameters():
        p.requires_grad_(False)

    tap_idx = _TAP_TO_INDEX[tap]
    # DINOv2 ViT-B: 768, ViT-L: 1024, ViT-S: 384
    embed_dim = model.embed_dim
    wrapper = _DINOv2Wrapper(model, tap_idx).to(device)
    return wrapper, embed_dim


def online_teacher_features(
    teacher_model: nn.Module,
    images: torch.Tensor,
    tap: str = "s1",
) -> torch.Tensor:
    """Run teacher on a batch of images and extract features at the given tap.

    On CUDA OOM, automatically splits the batch into smaller chunks and
    retries (halving chunk size each time, minimum 1).

    Args:
        teacher_model: Frozen teacher model in eval mode.
        images: [B, 3, H, W] already ImageNet-normalized.
        tap: Which encoder stage to extract ("s1", "s2", "s3", "s4").

    Returns:
        Teacher features [B, C, Ht, Wt] in fp16.
    """
    tap_idx = _TAP_TO_INDEX[tap]
    B = images.shape[0]

    # Try full batch first
    try:
        with torch.no_grad():
            outputs = teacher_model(images.half())
        return outputs.hidden_states[tap_idx]
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or B <= 1:
            raise
        torch.cuda.empty_cache()

    # OOM: split into chunks, halving until it fits
    chunk_size = max(1, B // 2)
    while chunk_size >= 1:
        try:
            parts = []
            for i in range(0, B, chunk_size):
                with torch.no_grad():
                    out = teacher_model(images[i:i + chunk_size].half())
                parts.append(out.hidden_states[tap_idx])
            return torch.cat(parts, dim=0)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or chunk_size <= 1:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(1, chunk_size // 2)
    # Should not reach here, but just in case
    raise RuntimeError("Teacher forward OOM even with batch_size=1")
