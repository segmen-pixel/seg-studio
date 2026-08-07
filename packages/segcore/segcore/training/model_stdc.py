# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Lightweight segmentation network registered under the ``stdc`` key.

A plain sequential 3x3 convolution encoder (stem plus three stages, widths
bc / bc*2 / bc*4 / bc*8) feeding a context module, a feature-fusion module
and a segmentation head. Optimized for real-time inference.
All normalization uses GroupNorm (CoreML/batch=1 compatible).

NOTE: the ``stdc`` registry key is historical and kept for checkpoint and
config compatibility. The encoder is a plain convolution stack; it does not
implement a dense-concatenation block.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from segcore.training.model import _norm_layer, _SEBlock, register_model


def _upsample_from_head(feat: torch.Tensor, *, head_stride: int,
                        output_stride: int) -> torch.Tensor:
    """Resample a feature map produced at ``1/head_stride`` resolution to
    the target ``1/output_stride`` resolution using a static scale_factor.

    Avoids reading the input tensor's dynamic shape, which would inject an
    ``aten::Int`` op that coremltools' torch frontend cannot lower.
    """
    if output_stride < head_stride:
        scale = head_stride // output_stride  # int
        return F.interpolate(
            feat, scale_factor=float(scale),
            mode="bilinear", align_corners=False,
        )
    if output_stride > head_stride:
        factor = output_stride // head_stride  # int
        return F.avg_pool2d(feat, kernel_size=factor, stride=factor)
    return feat


class _ConvBNReLU(nn.Module):
    """Conv2d + GroupNorm + ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.norm = _norm_layer(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _ContextModule(nn.Module):
    """Lightweight context aggregation via global pooling + local conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0),
        )
        self.local_branch = _ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.global_branch(x)
        g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        loc = self.local_branch(x)
        return g + loc


class _FFM(nn.Module):
    """Feature Fusion Module — merges low-level and high-level features."""

    def __init__(self, low_ch: int, high_ch: int, out_ch: int):
        super().__init__()
        self.conv_low = _ConvBNReLU(low_ch, out_ch, kernel_size=1, padding=0)
        self.conv_high = _ConvBNReLU(high_ch, out_ch, kernel_size=1, padding=0)
        self.se = _SEBlock(out_ch, reduction=4)
        self.fuse = _ConvBNReLU(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low = self.conv_low(low)
        high = self.conv_high(high)
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(high, size=low.shape[2:], mode="bilinear", align_corners=False)
        out = low + high
        out = self.se(out)
        out = self.fuse(out)
        return out


@register_model("stdc")
class STDCNet(nn.Module):
    """Lightweight segmentation network (registry key ``stdc``).

    4-stage plain convolution encoder (stem, then three stride-2 stages).
    Channel widths: bc, bc*2, bc*4, bc*8 (default bc=32: 32→64→128→256).
    Includes context module and feature fusion for segmentation.

    Args:
        num_classes: Number of output classes.
        output_stride: Output downsampling factor (1, 2, or 4).
        base_channels: Base channel width (scales all layers).
    """

    def __init__(self, num_classes: int, output_stride: int = 2,
                 base_channels: int = 32, deep_supervision: bool = False, **kwargs):
        super().__init__()
        if output_stride not in (1, 2, 4):
            raise ValueError("output_stride must be one of {1, 2, 4}")
        self.output_stride = int(output_stride)
        self.deep_supervision = bool(deep_supervision)

        ch1 = base_channels          # 32
        ch2 = base_channels * 2      # 64
        ch3 = base_channels * 4      # 128
        ch4 = base_channels * 8      # 256

        # Stem
        self.stem = nn.Sequential(
            _ConvBNReLU(3, ch1, kernel_size=3, stride=2, padding=1),
            _ConvBNReLU(ch1, ch1, kernel_size=3, stride=1, padding=1),
        )

        # Stage 1: 1/2 → 1/4
        self.stage1 = nn.Sequential(
            _ConvBNReLU(ch1, ch2, kernel_size=3, stride=2, padding=1),
            _ConvBNReLU(ch2, ch2, kernel_size=3, stride=1, padding=1),
        )

        # Stage 2: 1/4 → 1/8
        self.stage2_down = _ConvBNReLU(ch2, ch3, kernel_size=3, stride=2, padding=1)
        self.stage2_body = nn.Sequential(
            _ConvBNReLU(ch3, ch3, kernel_size=3, padding=1),
            _ConvBNReLU(ch3, ch3, kernel_size=3, padding=1),
        )

        # Stage 3: 1/8 → 1/16
        self.stage3_down = _ConvBNReLU(ch3, ch4, kernel_size=3, stride=2, padding=1)
        self.stage3_body = nn.Sequential(
            _ConvBNReLU(ch4, ch4, kernel_size=3, padding=1),
            _ConvBNReLU(ch4, ch4, kernel_size=3, padding=1),
        )

        # Context module on deepest features
        self.context = _ContextModule(ch4, ch4)

        # Feature fusion: merge stage2 (1/8) with context-enhanced stage3 (1/16)
        self.ffm = _FFM(low_ch=ch3, high_ch=ch4, out_ch=ch3)

        # Segmentation head
        self.head = nn.Sequential(
            _ConvBNReLU(ch3, ch3, kernel_size=3, padding=1),
            nn.Conv2d(ch3, num_classes, kernel_size=1),
        )

        # Deep supervision: auxiliary head on stage2 features (before fusion)
        if deep_supervision:
            self.aux_head = nn.Sequential(
                _ConvBNReLU(ch3, ch3, kernel_size=3, padding=1),
                nn.Conv2d(ch3, num_classes, kernel_size=1),
            )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # NOTE: intentionally avoid reading x.shape[2:] and computing
        # target_h/target_w as Python ints. Doing so inserts an aten::Int
        # op into the traced graph (reading x.shape → casting to int)
        # which coremltools' torch frontend cannot lower, causing
        # "TypeError: only 0-dimensional arrays can be converted to
        # Python scalars" during CoreML export. Using scale_factor
        # traces cleanly as aten::upsample_bilinear2d with static
        # float scales and is well-supported by coremltools.
        # For typical configs (input size a multiple of 8),
        # scale_factor * head_output = input_size // output_stride,
        # so the result is numerically identical to the old path.

        # Encoder
        s0 = self.stem(x)                    # 1/2
        s1 = self.stage1(s0)                 # 1/4
        s2 = self.stage2_body(self.stage2_down(s1))  # 1/8
        s3 = self.stage3_body(self.stage3_down(s2))  # 1/16

        # Context
        ctx = self.context(s3)               # 1/16

        # Fusion
        fused = self.ffm(s2, ctx)            # 1/8

        # Head
        logits = self.head(fused)            # 1/8  (head_stride = 8)

        # Upsample from 1/8 to 1/output_stride using a static
        # scale_factor. Works for output_stride ∈ {1, 2, 4, 8}
        # (the only values ever produced by train_config).
        logits = _upsample_from_head(logits, head_stride=8,
                                     output_stride=self.output_stride)

        # Deep supervision: auxiliary prediction from stage2 (also 1/8)
        aux_logits: list[torch.Tensor] = []
        if self.deep_supervision and self.training:
            aux = self.aux_head(s2)  # 1/8
            aux = _upsample_from_head(aux, head_stride=8,
                                      output_stride=self.output_stride)
            aux_logits.append(aux)

        feat_dict = {"e3": ctx}
        if aux_logits:
            feat_dict["aux_logits"] = aux_logits
        if return_features:
            return logits, feat_dict
        if aux_logits:
            return logits, feat_dict
        return logits
