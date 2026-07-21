# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""DeepLabV3+ with MobileNetV3-Large encoder.

Multi-scale feature extraction via ASPP (Atrous Spatial Pyramid Pooling)
with a lightweight MobileNetV3-Large backbone. All normalization uses
GroupNorm for CoreML compatibility and batch=1 stability.

Reference: Chen et al., "Encoder-Decoder with Atrous Separable Convolution" (ECCV 2018)
           Howard et al., "Searching for MobileNetV3" (ICCV 2019)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from segcore.training.model import _norm_layer, _SEBlock, register_model

# ---------------------------------------------------------------------------
# MobileNetV3 building blocks
# ---------------------------------------------------------------------------

class _HSwish(nn.Module):
    """Hard-swish activation (JIT-traceable, no inplace issues)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=False) / 6.0


class _InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block with optional SE and h-swish."""

    def __init__(self, in_ch: int, out_ch: int, expand_ch: int,
                 kernel_size: int = 3, stride: int = 1,
                 use_se: bool = False, use_hswish: bool = False,
                 dilation: int = 1):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        layers = []
        # Expand
        if expand_ch != in_ch:
            layers.extend([
                nn.Conv2d(in_ch, expand_ch, 1, bias=False),
                _norm_layer(expand_ch),
                _HSwish() if use_hswish else nn.ReLU(inplace=True),
            ])
        # Depthwise
        layers.extend([
            nn.Conv2d(expand_ch, expand_ch, kernel_size, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=expand_ch, bias=False),
            _norm_layer(expand_ch),
            _HSwish() if use_hswish else nn.ReLU(inplace=True),
        ])
        # SE
        if use_se:
            layers.append(_SEBlock(expand_ch, reduction=4))
        # Project
        layers.extend([
            nn.Conv2d(expand_ch, out_ch, 1, bias=False),
            _norm_layer(out_ch),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _MobileNetV3Encoder(nn.Module):
    """MobileNetV3-Large encoder (simplified, GroupNorm).

    Outputs features at 1/4 (low-level) and 1/16 (high-level).
    Width scaled by width_mult = base_channels / 32.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        wm = base_channels / 32.0

        def c(ch: int) -> int:
            return _make_divisible(ch * wm)

        # Stem: 1/1 → 1/2
        self.stem = nn.Sequential(
            nn.Conv2d(3, c(16), 3, stride=2, padding=1, bias=False),
            _norm_layer(c(16)),
            _HSwish(),
        )

        # Stage 1: 1/2 → 1/4 (low-level features)
        self.stage1 = nn.Sequential(
            _InvertedResidual(c(16), c(16), c(16), kernel_size=3, stride=2, use_se=False),
            _InvertedResidual(c(16), c(24), c(64), kernel_size=3, stride=1, use_se=False),
            _InvertedResidual(c(24), c(24), c(72), kernel_size=3, stride=1, use_se=False),
        )
        self.low_channels = c(24)

        # Stage 2: 1/4 → 1/8
        self.stage2 = nn.Sequential(
            _InvertedResidual(c(24), c(40), c(72), kernel_size=5, stride=2, use_se=True, use_hswish=False),
            _InvertedResidual(c(40), c(40), c(120), kernel_size=5, stride=1, use_se=True, use_hswish=False),
            _InvertedResidual(c(40), c(40), c(120), kernel_size=5, stride=1, use_se=True, use_hswish=False),
        )

        # Stage 3: 1/8 → 1/16
        self.stage3 = nn.Sequential(
            _InvertedResidual(c(40), c(80), c(240), kernel_size=3, stride=2, use_se=False, use_hswish=True),
            _InvertedResidual(c(80), c(80), c(200), kernel_size=3, stride=1, use_se=False, use_hswish=True),
            _InvertedResidual(c(80), c(80), c(184), kernel_size=3, stride=1, use_se=False, use_hswish=True),
            _InvertedResidual(c(80), c(80), c(184), kernel_size=3, stride=1, use_se=False, use_hswish=True),
            _InvertedResidual(c(80), c(112), c(480), kernel_size=3, stride=1, use_se=True, use_hswish=True),
            _InvertedResidual(c(112), c(112), c(672), kernel_size=3, stride=1, use_se=True, use_hswish=True),
            _InvertedResidual(c(112), c(160), c(672), kernel_size=5, stride=1, use_se=True, use_hswish=True),
            _InvertedResidual(c(160), c(160), c(960), kernel_size=5, stride=1, use_se=True, use_hswish=True),
        )
        self.high_channels = c(160)

    def forward(self, x: torch.Tensor):
        s = self.stem(x)          # 1/2
        low = self.stage1(s)      # 1/4
        mid = self.stage2(low)    # 1/8
        high = self.stage3(mid)   # 1/16
        return low, high


# ---------------------------------------------------------------------------
# ASPP (Atrous Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------

class _ASPPConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool to 1x1 then resize back to x's spatial size.
        # Passing x.shape[2:] directly (not via an intermediate tuple
        # variable) avoids inserting an aten::Int cast that coremltools'
        # torch frontend rejects with "only 0-dimensional arrays can be
        # converted to Python scalars".
        out = self.pool(x)
        return F.interpolate(out, size=x.shape[2:],
                             mode="bilinear", align_corners=False)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with 5 branches."""

    def __init__(self, in_ch: int, out_ch: int = 256,
                 rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                _norm_layer(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(_ASPPConv(in_ch, out_ch, rate))
        modules.append(_ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(modules), out_ch, 1, bias=False),
            _norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))


# ---------------------------------------------------------------------------
# DeepLabV3+ Segmentation Head
# ---------------------------------------------------------------------------

@register_model("deeplabv3plus")
class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with MobileNetV3-Large encoder.

    ASPP on 1/16 features + decoder with low-level feature fusion from 1/4.
    Width scales with base_channels (width_mult = base_channels / 32).

    Args:
        num_classes: Number of output classes.
        output_stride: Output downsampling factor (1, 2, or 4).
        base_channels: Base channel width (32 → ~2.0M params, 64 → ~6.5M params).
    """

    def __init__(self, num_classes: int, output_stride: int = 2,
                 base_channels: int = 32, **kwargs):
        super().__init__()
        if output_stride not in (1, 2, 4):
            raise ValueError("output_stride must be one of {1, 2, 4}")
        self.output_stride = int(output_stride)

        # Encoder
        self.encoder = _MobileNetV3Encoder(base_channels)
        low_ch = self.encoder.low_channels
        high_ch = self.encoder.high_channels

        # ASPP
        self.aspp = _ASPP(high_ch, out_ch=256)

        # Decoder
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            _norm_layer(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            _norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            _norm_layer(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # See model_stdc._upsample_from_head for why we avoid reading
        # x.shape[2:] when computing the final upsample target.

        # Encoder: low=1/4, high=1/16
        low, high = self.encoder(x)

        # ASPP on high-level features
        aspp_out = self.aspp(high)  # [B, 256, H/16, W/16]

        # Upsample ASPP (1/16) to match low-level features (1/4).
        # Using scale_factor=4 instead of size=low.shape[2:] avoids an
        # aten::Int cast that coremltools cannot lower. For any input
        # size that is a multiple of 16 (the training configs always
        # enforce this via patch_size), the two formulations produce
        # an identical output tensor.
        aspp_up = F.interpolate(aspp_out, scale_factor=4.0,
                                mode="bilinear", align_corners=False)

        # Fuse low-level + ASPP
        low_proj = self.low_proj(low)
        fused = torch.cat([aspp_up, low_proj], dim=1)
        decoded = self.decoder(fused)  # [B, 256, H/4, W/4]

        logits = self.head(decoded)  # 1/4 (head_stride = 4)

        # Resize to target output_stride using a static scale_factor
        # instead of size=(h // output_stride, ...), which coremltools
        # cannot lower because it inserts aten::Int on the shape.
        from .model_stdc import _upsample_from_head
        logits = _upsample_from_head(logits, head_stride=4,
                                     output_stride=self.output_stride)

        if return_features:
            return logits, {"e3": aspp_out}
        return logits
