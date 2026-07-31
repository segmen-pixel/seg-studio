"""Single source of truth for whether AMP (fp16 autocast) may be used.

Training and inference must agree. Turing (GTX 16xx / RTX 20xx, compute
capability 7.5) has no FP16 Tensor Cores, and on this project's model it
returns all-NaN logits once a forward batch reaches four tiles: batch 1 and 2
are clean, batch 4 and up are 100% NaN, in both fp32-benchmark modes.

Training gated on compute capability from the start. Sliding-window inference
did not -- it enabled AMP for any CUDA device. On a Turing box that meant
training in FP32 and evaluating in FP16 at a tile batch of ten, so every
probability came back NaN, argmax collapsed to background, and validation F1
read exactly 0.0000 while the weights were perfectly healthy. Measured on a
GTX 1650: the same checkpoint scored F1 0.0000 with AMP and F1 0.8759 without.

Keep this the only place that decides. A scale that differs between training
and inference does not raise anything; it just quietly changes the numbers.
"""

from __future__ import annotations

import torch

# FP16 Tensor Cores land with Ampere. Below that, autocast is a correctness
# risk on this model, not merely a performance question.
_MIN_AMP_MAJOR = 8


def amp_supported(device: torch.device | str | None) -> bool:
    """Return True when fp16 autocast is safe on ``device``."""
    if device is None:
        return False
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(dev)[0] >= _MIN_AMP_MAJOR


def amp_status_line(device: torch.device | str | None) -> str:
    """One-line explanation suitable for the training log."""
    if device is None:
        return "Mixed precision (AMP): disabled (no device)"
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "mps":
        return "Mixed precision (AMP): disabled (MPS backend, not yet stable)"
    if dev.type != "cuda" or not torch.cuda.is_available():
        return f"Mixed precision (AMP): disabled ({dev.type} backend)"
    major, minor = torch.cuda.get_device_capability(dev)
    if major >= _MIN_AMP_MAJOR:
        return f"Mixed precision (AMP): enabled (compute capability {major}.{minor})"
    return (f"Mixed precision (AMP): disabled (compute capability "
            f"{major}.{minor} < {_MIN_AMP_MAJOR}.0)")
