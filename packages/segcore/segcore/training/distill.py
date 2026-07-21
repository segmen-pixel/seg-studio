# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Knowledge distillation support for segcore training.

Supported online teachers:
  - DINOv2 (loaded via torch.hub or a bundled snapshot)
  - SAM2 image encoder (from the bundled SAM2 package)
"""
from __future__ import annotations

from pathlib import Path

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
        teacher_feat: Teacher feature map [B, Ct, H, W] (float16 from cache).
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


def load_teacher_cache(
    cache_dir: Path,
    device: torch.device | str,
    tap_name: str = "s1",
) -> dict[str, torch.Tensor]:
    """Load all cached teacher features into memory.

    With ~21 images at ~300KB each, total is ~6MB - fits easily in GPU memory.

    Returns:
        Dict mapping image stem -> feature tensor [C, H, W] (float16 on device).
    """
    cache: dict[str, torch.Tensor] = {}
    suffix = f"_{tap_name}.pt"
    for pt_file in sorted(cache_dir.glob(f"*{suffix}")):
        stem = pt_file.stem[: -len(f"_{tap_name}")]
        feat = torch.load(pt_file, map_location=device, weights_only=True)
        cache[stem] = feat
    return cache


def get_teacher_batch(
    cache: dict[str, torch.Tensor],
    stems: list[str],
    device: torch.device | str,
) -> torch.Tensor | None:
    """Look up teacher features for a batch of stems.

    Returns:
        Stacked tensor [B, C, H, W] or None if any stem is missing.
    """
    feats = []
    for stem in stems:
        if stem not in cache:
            return None
        feats.append(cache[stem])
    return torch.stack(feats, dim=0).to(device)


def apply_augmentation_to_features(
    features: torch.Tensor,
    hflip: torch.Tensor,
    vflip: torch.Tensor,
    rot90_k: torch.Tensor,
) -> torch.Tensor:
    """Apply the same geometric augmentations to teacher features.

    Args:
        features: [B, C, H, W] teacher feature maps.
        hflip: [B] bool tensor - whether horizontal flip was applied.
        vflip: [B] bool tensor - whether vertical flip was applied.
        rot90_k: [B] int tensor - number of 90-degree rotations (0-3).

    Returns:
        Augmented features [B, C, H, W].
    """
    result = features.clone()
    for i in range(features.shape[0]):
        if hflip[i]:
            result[i] = result[i].flip(-1)  # flip W dimension
        if vflip[i]:
            result[i] = result[i].flip(-2)  # flip H dimension
        k = int(rot90_k[i].item())
        if k > 0:
            result[i] = torch.rot90(result[i], k=k, dims=(-2, -1))
    return result


# ---------------------------------------------------------------------------
# Channel-level distillation (GAP-based, spatially invariant)
# ---------------------------------------------------------------------------


class ChannelProjector(nn.Module):
    """Project student GAP vector to teacher GAP vector via Linear layer.

    Student e3 GAP has 128 dims; teacher s1 GAP has 64 dims.
    A single Linear (no bias) aligns the channel dimensions.
    """

    def __init__(self, student_ch: int = 128, teacher_ch: int = 64):
        super().__init__()
        self.proj = nn.Linear(student_ch, teacher_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def channel_distillation_loss(
    student_feat: torch.Tensor,   # [B, Cs, H, W]
    teacher_vec: torch.Tensor,    # [B, Ct]
    projector: ChannelProjector,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Compute channel-level distillation loss (GAP-based).

    Global Average Pools the student feature map, projects to teacher dim,
    then computes loss against teacher GAP vector.

    Args:
        student_feat: Student feature map [B, Cs, H, W].
        teacher_vec: Teacher GAP vector [B, Ct] (float16 from cache).
        projector: Linear projecting Cs -> Ct.
        loss_type: "smooth_l1", "mse", or "cosine".

    Returns:
        Scalar loss value.
    """
    student_gap = student_feat.mean(dim=(2, 3))  # [B, Cs]
    projected = projector(student_gap)             # [B, Ct]
    teacher_vec = teacher_vec.to(projected.dtype)
    if loss_type == "mse":
        return F.mse_loss(projected, teacher_vec)
    elif loss_type == "cosine":
        return 1.0 - F.cosine_similarity(projected, teacher_vec, dim=1).mean()
    else:  # default: smooth_l1
        return F.smooth_l1_loss(projected, teacher_vec)


def gap_teacher_cache(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert spatial teacher cache [C,H,W] to GAP vectors [C].

    Pre-computes GAP so the training loop only needs vector lookups.
    """
    gap_cache: dict[str, torch.Tensor] = {}
    for stem, feat in cache.items():
        gap_cache[stem] = feat.mean(dim=(1, 2))  # [C,H,W] -> [C]
    return gap_cache


def get_teacher_batch_vec(
    cache: dict[str, torch.Tensor],
    stems: list[str],
    device: torch.device | str,
) -> torch.Tensor | None:
    """Look up teacher GAP vectors for a batch of stems.

    Returns:
        Stacked tensor [B, C] or None if any stem is missing.
    """
    vecs = []
    for stem in stems:
        if stem not in cache:
            return None
        vecs.append(cache[stem])
    return torch.stack(vecs, dim=0).to(device)


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
        Only the Apache-2.0 pretrained weight is bundled with Seg-Studio.
        The model-definition source tree is fetched at first call via
        ``torch.hub.load('facebookresearch/dinov2', ...)``; we deliberately
        do not redistribute the upstream hub repo because recent versions
        carry non-commercial license fragments alongside the Apache code.
        For air-gapped deployment, place the Apache-2.0-licensed Python
        files under ``~/.cache/torch/hub/facebookresearch_dinov2_main/``
        manually.
    """
    from pathlib import Path as _Path

    import torch as _torch

    # The installer ships the weight at <root>/models/dinov2/<variant>_pretrain.pth.
    # The hub source tree is intentionally NOT bundled (see docstring above);
    # if a user has populated ~/.cache/torch/hub/facebookresearch_dinov2_main/
    # locally we honour it as an offline fallback path.
    _this_file = _Path(__file__).resolve()
    _root = _this_file.parent.parent.parent.parent  # distill.py -> training -> segcore -> packages -> root
    _bundled = _root / "models" / "dinov2" / f"{variant}_pretrain.pth"
    if not _bundled.exists():
        _bundled = _root.parent / "models" / "dinov2" / f"{variant}_pretrain.pth"

    # Optional offline path: only used when the user has manually populated
    # the torch hub cache with the Apache-2.0 files. Default install pulls
    # the source tree on demand from torch.hub.
    _hub_cache = _Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"

    if _bundled.exists() and _hub_cache.exists():
        # Fully offline: load definition from local hub cache + bundled weight.
        model = _torch.hub.load(
            str(_hub_cache), variant, pretrained=False, source="local",
        )
        state = _torch.load(str(_bundled), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    elif _bundled.exists():
        # Bundled weight is available; fetch the source tree via torch.hub
        # (cached in ~/.cache/torch/hub/ on first run).
        model = _torch.hub.load(
            "facebookresearch/dinov2", variant, pretrained=False,
        )
        state = _torch.load(str(_bundled), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        # No bundled weight: pull both source and weights via torch.hub.
        model = _torch.hub.load(
            "facebookresearch/dinov2", variant, pretrained=True,
        )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tap_idx = _TAP_TO_INDEX[tap]
    # DINOv2 ViT-B: 768, ViT-L: 1024, ViT-S: 384
    embed_dim = model.embed_dim
    wrapper = _DINOv2Wrapper(model, tap_idx).to(device)
    return wrapper, embed_dim


# ---------------------------------------------------------------------------
# Ensemble logits distillation (multi-scale teacher averaging)
# ---------------------------------------------------------------------------


def load_ensemble_logits_cache(
    cache_dir: Path,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Load precomputed ensemble logits into memory.

    Each file is ``{stem}_ensemble.pt`` containing [C, H, W] raw logits (float16).

    Returns:
        Dict mapping image stem -> logits tensor [C, H, W] on device.
    """
    cache: dict[str, torch.Tensor] = {}
    suffix = "_ensemble.pt"
    for pt_file in sorted(cache_dir.glob(f"*{suffix}")):
        stem = pt_file.stem[: -len("_ensemble")]
        feat = torch.load(pt_file, map_location=device, weights_only=True)
        cache[stem] = feat
    return cache


def get_ensemble_logits_batch(
    cache: dict[str, torch.Tensor],
    stems: list[str],
    crop_boxes: list[tuple[int, int, int, int]] | None,
    target_size: tuple[int, int],
    device: torch.device | str,
) -> torch.Tensor | None:
    """Look up ensemble logits for a batch, optionally cropping patch regions.

    Args:
        cache: stem -> [C, H, W] raw logits.
        stems: list of image stems for the batch.
        crop_boxes: per-sample (left, top, right, bottom) in original image
                    coordinates, or None for full-image mode.
        target_size: (out_h, out_w) to resize cropped logits to match student
                     output spatial dims.
        device: target device.

    Returns:
        [B, C, out_h, out_w] float32 tensor, or None if any stem missing.
    """
    logits_list = []
    out_h, out_w = target_size
    for i, stem in enumerate(stems):
        if stem not in cache:
            return None
        logits = cache[stem]  # [C, H, W] float16
        if crop_boxes is not None:
            left, top, right, bottom = crop_boxes[i]
            logits = logits[:, top:bottom, left:right]
        # Resize to student output spatial dims
        logits_resized = F.interpolate(
            logits.unsqueeze(0).float(),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        logits_list.append(logits_resized)
    return torch.stack(logits_list, dim=0).to(device)


def ensemble_logits_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 2.0,
    ignore_index: int = 255,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence loss between student and ensemble teacher soft labels.

    Teacher probabilities are pre-computed (already softmax-averaged across
    teachers in probability space). Only the student logits are temperature-
    scaled. Uses T^2 scaling to keep gradient magnitude consistent.

    Args:
        student_logits: [B, C, H, W] student raw logits.
        teacher_probs: [B, C, H, W] teacher soft-label probabilities (float32).
                       Already averaged across teachers in probability space.
        temperature: softmax temperature for student (default 2.0).
        ignore_index: pixels to exclude from loss.
        mask: optional [B, H, W] bool tensor (True = valid pixel).

    Returns:
        Scalar loss value.
    """
    T = float(temperature)

    # Spatial alignment
    if student_logits.shape[2:] != teacher_probs.shape[2:]:
        teacher_probs = F.interpolate(
            teacher_probs,
            size=student_logits.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    # Student: temperature-scaled log-softmax
    s_log = F.log_softmax(student_logits / T, dim=1)
    # Teacher: already probabilities, clamp to avoid log(0)
    t_soft = teacher_probs.clamp(min=1e-7)

    # KL(P_t || P_s) = sum(P_t * (log P_t - log P_s))
    kl = F.kl_div(s_log, t_soft, reduction="none")  # [B, C, H, W]
    kl = kl.sum(dim=1)  # [B, H, W]

    if mask is not None:
        kl = kl * mask.float()
        n_valid = mask.float().sum().clamp(min=1.0)
        return T * T * kl.sum() / n_valid
    else:
        return T * T * kl.mean()


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
