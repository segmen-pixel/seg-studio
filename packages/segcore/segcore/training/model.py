# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(arch: str, num_classes: int, output_stride: int,
                base_channels: int, deep_supervision: bool = False,
                **kwargs) -> nn.Module:
    """Instantiate a registered model by architecture name."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](
        num_classes=num_classes, output_stride=output_stride,
        base_channels=base_channels, deep_supervision=deep_supervision,
        **kwargs)


def distill_feature_channels(arch: str, base_channels: int) -> int:
    """Return the channel count of the 'e3' feature tap for distillation."""
    if arch == "simpleunet":
        return base_channels * 4
    elif arch == "stdc":
        return base_channels * 8
    return base_channels * 4


def _norm_layer(num_channels: int) -> nn.GroupNorm:
    """GroupNorm that works with any batch size including batch=1."""
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (CoreML-friendly)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)       # global avg pool
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


@register_model("simpleunet")
class SimpleUNet(nn.Module):
    """Lightweight UNet with skip connections, GroupNorm, and SE attention.

    Encoder: 3 blocks with MaxPool downsampling.
    Channel widths scale with base_channels: ch1=base, ch2=base*2, ch3=base*4.
    Default base_channels=32 gives 32->64->128 (~340K params).
    base_channels=64 gives 64->128->256 (~1.9M params at output_stride=2).
    Each encoder block has a Squeeze-and-Excitation attention module to
    weight feature channels by importance (improves texture/color discrimination).
    Decoder: 2 upsampling blocks with skip connections from encoder.
    Output: num_classes logits at the specified output_stride.

    Uses GroupNorm instead of BatchNorm for stable training at any batch size,
    including batch=1 on low-VRAM GPUs.
    """

    def __init__(self, num_classes: int, output_stride: int = 4, base_channels: int = 32,
                 use_se: bool = True, deep_supervision: bool = False):
        super().__init__()
        if output_stride not in (1, 2, 4):
            raise ValueError("output_stride must be one of {1, 2, 4}")
        self.output_stride = int(output_stride)
        self.use_se = bool(use_se)
        self.deep_supervision = bool(deep_supervision)

        ch1 = base_channels        # 32 or 64
        ch2 = base_channels * 2    # 64 or 128
        ch3 = base_channels * 4    # 128 or 256

        # Encoder
        self.enc1 = self._block(3, ch1)
        self.se1 = _SEBlock(ch1) if use_se else nn.Identity()
        self.enc2 = self._block(ch1, ch2)
        self.se2 = _SEBlock(ch2) if use_se else nn.Identity()
        self.enc3 = self._block(ch2, ch3)
        self.se3 = _SEBlock(ch3) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(ch3, ch2, kernel_size=2, stride=2)
        self.dec2 = self._block(ch2 + ch2, ch2)
        self.up1 = nn.ConvTranspose2d(ch2, ch1, kernel_size=2, stride=2)
        self.dec1 = self._block(ch1 + ch1, ch1)

        self.head = nn.Conv2d(ch1, num_classes, kernel_size=1)
        # Stride-specific heads: skip unnecessary decoder stages for efficiency
        if output_stride >= 2:
            self.head2 = nn.Conv2d(ch2, num_classes, kernel_size=1)
        if output_stride >= 4:
            self.head4 = nn.Conv2d(ch3, num_classes, kernel_size=1)

        # Deep supervision: auxiliary heads for intermediate decoder stages
        if deep_supervision:
            if output_stride == 1:
                # Aux from d2 (H/2 resolution)
                self.aux_head = nn.Conv2d(ch2, num_classes, kernel_size=1)
            elif output_stride == 2:
                # Aux from e3 (H/4 resolution)
                self.aux_head = nn.Conv2d(ch3, num_classes, kernel_size=1)

    def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Encoder
        e1 = self.se1(self.enc1(x))             # [B, ch1, H, W]
        e2 = self.se2(self.enc2(self.pool(e1)))  # [B, ch2, H/2, W/2]
        e3 = self.se3(self.enc3(self.pool(e2)))  # [B, ch3, H/4, W/4]

        # Decoder: skip unnecessary stages based on output_stride
        aux_logits: list[torch.Tensor] = []
        if self.output_stride == 4:
            # No decode needed — apply head directly to encoder output
            logits = self.head4(e3)              # [B, num_classes, H/4, W/4]
        elif self.output_stride == 2:
            # One upsample: e3 → d2 at H/2
            d2 = self.up2(e3)                    # [B, ch2, H/2, W/2]
            d2 = self._pad_to_match(d2, e2)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            logits = self.head2(d2)              # [B, num_classes, H/2, W/2]
            if self.deep_supervision and self.training:
                aux_logits.append(self.aux_head(e3))  # H/4
        else:
            # Full decode to H (output_stride == 1)
            d2 = self.up2(e3)                    # [B, ch2, H/2, W/2]
            d2 = self._pad_to_match(d2, e2)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)                    # [B, ch1, H, W]
            d1 = self._pad_to_match(d1, e1)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            logits = self.head(d1)               # [B, num_classes, H, W]
            if self.deep_supervision and self.training:
                aux_logits.append(self.aux_head(d2))  # H/2

        feat_dict = {"e1": e1, "e2": e2, "e3": e3}
        if aux_logits:
            feat_dict["aux_logits"] = aux_logits
        if return_features:
            return logits, feat_dict
        if aux_logits:
            return logits, feat_dict
        return logits

    @staticmethod
    def _pad_to_match(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pad src spatially to match target's H and W (handles odd dimensions)."""
        diff_h = target.shape[2] - src.shape[2]
        diff_w = target.shape[3] - src.shape[3]
        if diff_h == 0 and diff_w == 0:
            return src
        return F.pad(src, [0, diff_w, 0, diff_h])


# ---------------------------------------------------------------------------
# Auto-register additional architectures
# ---------------------------------------------------------------------------
import segcore.training.model_stdc  # noqa: F401,E402
