# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Sliding-window patch-based inference for full-resolution evaluation.

When training with random patches (e.g. 256x256 from 1280x960 images),
resizing the full image down to 256x256 for validation loses fine detail
and creates a scale mismatch.  This module runs the model on overlapping
patches at the *original* resolution and stitches the softmax predictions
back together by averaging overlapping regions.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def _estimate_tile_batch(patch_size: int, num_classes: int, safety: float = 0.55) -> int:
    """Estimate max tile batch size from available VRAM. Falls back to 16.

    Uses nvidia-smi subprocess to avoid initializing PyTorch CUDA context
    (which interferes with ORT CUDA EP on low-VRAM GPUs).

    Inference needs far less VRAM than training (no gradients, no optimizer
    state), so we use 55% of free memory with a generous cap.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 16
        # Pick the GPU with most free memory
        lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
        free_mb = max(int(ln) for ln in lines) if lines else 0
        if free_mb <= 0:
            return 16
        free_mem = free_mb * 1024 * 1024
        # Inference-only: input + encoder intermediates + output (no grads)
        bytes_per_pixel = (3 + 128 + num_classes * 2) * 4
        per_patch_bytes = patch_size * patch_size * bytes_per_pixel
        max_batch = int(free_mem * safety / max(per_patch_bytes, 1))
        # Clamp: min 4 (amortize per-call overhead), cap 64 (diminishing returns)
        return max(4, min(max_batch, 64))
    except Exception:
        return 16


def _ceil_to_stride(dim: int, patch_size: int, stride: int) -> int:
    """Return the smallest padded size >= *dim* so that patches tile evenly.

    After padding, the last patch start ``(padded - patch_size)`` must be a
    multiple of *stride* and ``>= 0``.
    """
    if dim <= patch_size:
        return patch_size
    # Number of strides needed: ceil((dim - patch_size) / stride)
    n = math.ceil((dim - patch_size) / stride)
    return n * stride + patch_size


def compute_patch_grid(
    H: int, W: int, patch_size: int, stride: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Compute padded dimensions and top-left positions for every patch.

    Returns:
        (H_pad, W_pad, positions)  where *positions* is a list of (y, x).
    """
    H_pad = _ceil_to_stride(H, patch_size, stride)
    W_pad = _ceil_to_stride(W, patch_size, stride)
    positions: list[tuple[int, int]] = []
    for y in range(0, H_pad - patch_size + 1, stride):
        for x in range(0, W_pad - patch_size + 1, stride):
            positions.append((y, x))
    return H_pad, W_pad, positions


def _softmax_np(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    denom = np.sum(exp_logits, axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp_logits / denom


def _normalize_logits_batch(logits: np.ndarray, num_classes: int) -> np.ndarray:
    """Normalize backend outputs to BCHW logits."""
    logits_np = np.asarray(logits, dtype="float32")
    if logits_np.ndim == 3:
        if logits_np.shape[0] == num_classes:
            logits_np = logits_np[None, ...]
        elif logits_np.shape[-1] == num_classes:
            logits_np = np.transpose(logits_np, (2, 0, 1))[None, ...]
        else:
            raise ValueError("logits output shape mismatch")
    elif logits_np.ndim == 4:
        if logits_np.shape[1] == num_classes:
            pass
        elif logits_np.shape[-1] == num_classes:
            logits_np = np.transpose(logits_np, (0, 3, 1, 2))
        else:
            raise ValueError("logits output shape mismatch")
    else:
        raise ValueError("logits output shape unsupported")
    return logits_np


# ---------------------------------------------------------------------------
# Decomposed functions for pipelined inference runtime
# ---------------------------------------------------------------------------


def build_sw_job(
    job_id: str,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    output_stride: int,
    num_classes: int,
    normalize: dict,
    active_class_ids: list[int] | None = None,
) -> dict:
    """Prepare all state needed for a sliding-window inference job.

    Returns a dict that can be passed to :func:`iter_chunks`,
    :func:`accumulate_chunk`, and :func:`finalize_job`.
    """
    H, W = image.shape[:2]

    # Reflect-pad all 4 sides so edge patches always have context
    margin = patch_size // 2
    padded = np.pad(
        image,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="reflect",
    )
    H_eff, W_eff = padded.shape[:2]

    H_pad, W_pad, positions = compute_patch_grid(H_eff, W_eff, patch_size, stride)

    # Additional tiling pad (bottom-right) if needed
    extra_b = H_pad - H_eff
    extra_r = W_pad - W_eff
    if extra_b > 0 or extra_r > 0:
        padded = np.pad(padded, ((0, extra_b), (0, extra_r), (0, 0)), mode="reflect")

    # Output accumulator at model-output resolution
    out_h = H_pad // output_stride
    out_w = W_pad // output_stride
    accum = np.zeros((num_classes, out_h, out_w), dtype="float32")
    count = np.zeros((1, out_h, out_w), dtype="float32")

    patch_out = patch_size // output_stride

    suppress_mask: np.ndarray | None = None
    if active_class_ids is not None:
        m = np.ones(num_classes, dtype=bool)
        for cid in active_class_ids:
            if 0 <= cid < num_classes:
                m[cid] = False
        if np.any(m):
            suppress_mask = m

    # Gaussian weighting for smoother tile blending
    sigma = patch_out / 4.0
    ax = np.arange(patch_out, dtype="float32") - patch_out / 2.0 + 0.5
    xx, yy = np.meshgrid(ax, ax)
    gauss_weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype("float32")

    return {
        "job_id": job_id,
        "orig_hw": (H, W),
        "positions": positions,
        "padded": padded,
        "accum": accum,
        "count": count,
        "gauss_weight": gauss_weight,
        "patch_out": patch_out,
        "margin": margin,
        "output_stride": output_stride,
        "num_classes": num_classes,
        "normalize": normalize,
        "active_class_ids": active_class_ids,
        "suppress_mask": suppress_mask,
        "_total_accumulated": 0,
    }


def iter_chunks(
    job_dict: dict,
    padded: np.ndarray,
    positions: list[tuple[int, int]],
    patch_size: int,
    normalize: dict,
    chunk_size: int = 16,
):
    """Lazily yield normalised patch chunks for inference.

    Yields dicts with keys ``job_id``, ``chunk_index``, ``positions``,
    ``batch_np``.  Patches are extracted on-the-fly so only one chunk
    lives in memory at a time.
    """
    mean = np.array(normalize["mean"], dtype="float32").reshape(1, 3, 1, 1)
    std = np.array(normalize["std"], dtype="float32").reshape(1, 3, 1, 1)

    for i in range(0, len(positions), chunk_size):
        chunk_positions = positions[i:i + chunk_size]
        batch_np = np.empty(
            (len(chunk_positions), 3, patch_size, patch_size), dtype="float32",
        )
        for j, (y, x) in enumerate(chunk_positions):
            patch = padded[y:y + patch_size, x:x + patch_size]
            batch_np[j] = patch.transpose(2, 0, 1)
        # Normalize
        batch_np *= (1.0 / 255.0)
        batch_np = (batch_np - mean) / std

        yield {
            "job_id": job_dict["job_id"],
            "chunk_index": i,
            "positions": chunk_positions,
            "batch_np": batch_np,
        }


def accumulate_chunk(
    job_dict: dict,
    chunk_positions: list[tuple[int, int]],
    probs_np: np.ndarray,
) -> bool:
    """Accumulate Gaussian-weighted probabilities for one chunk.

    Returns ``True`` when all positions have been accumulated.
    """
    accum = job_dict["accum"]
    count = job_dict["count"]
    gauss_weight = job_dict["gauss_weight"]
    patch_out = job_dict["patch_out"]
    output_stride = job_dict["output_stride"]

    weighted = probs_np * gauss_weight  # (B, C, pH, pW)
    for j, (y, x) in enumerate(chunk_positions):
        oy = y // output_stride
        ox = x // output_stride
        accum[:, oy:oy + patch_out, ox:ox + patch_out] += weighted[j]
        count[:, oy:oy + patch_out, ox:ox + patch_out] += gauss_weight

    job_dict["_total_accumulated"] += len(chunk_positions)
    return job_dict["_total_accumulated"] >= len(job_dict["positions"])


def finalize_job(job_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Average accumulated probabilities, crop, and argmax.

    Returns:
        (pred, avg_probs) — same format as
        :func:`sliding_window_predict_infer_fn`.
    """
    accum = job_dict["accum"]
    count = job_dict["count"]
    margin = job_dict["margin"]
    output_stride = job_dict["output_stride"]
    H, W = job_dict["orig_hw"]

    # Average overlapping regions
    count = np.maximum(count, 1.0)
    avg_probs = (accum / count).astype("float32")

    # Crop back to original output size (skip the reflect-pad margin)
    margin_out = margin // output_stride
    orig_out_h = H // output_stride
    orig_out_w = W // output_stride
    avg_probs = avg_probs[:, margin_out:margin_out + orig_out_h, margin_out:margin_out + orig_out_w]
    pred = np.argmax(avg_probs, axis=0).astype("int64")

    return pred, avg_probs


def sliding_window_predict_infer_fn(
    infer_fn: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    patch_size: int,
    stride: int,
    num_classes: int,
    output_stride: int,
    normalize: dict,
    active_class_ids: list[int] | None = None,
    infer_fn_returns_probs: bool = False,
    stop_flag: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sliding-window inference on a full-resolution image.

    Args:
        infer_fn: Callable receiving normalized BCHW float32 patches and
            returning BCHW logits (or probs if *infer_fn_returns_probs*).
        image: HWC uint8 numpy array (original resolution, no resize).
        patch_size: Spatial size of each square patch (must match training).
        stride: Step between patches (< patch_size for overlap).
        num_classes: Number of output classes.
        output_stride: Model output stride (e.g. 2 means output is half res).
        normalize: ``{"mean": [...], "std": [...]}`` per-channel stats.
        active_class_ids: If given, suppress inactive class logits.
        infer_fn_returns_probs: If True, *infer_fn* returns softmax probs
            directly (e.g. from TTA) — skip internal softmax and suppression.

    Returns:
        (pred, probs)
        - pred: ``[H_out, W_out]`` int64 argmax class predictions.
        - probs: ``[C, H_out, W_out]`` float32 averaged softmax probabilities.
        where ``H_out = H // output_stride``, ``W_out = W // output_stride``.
    """
    H, W = image.shape[:2]

    # Reflect-pad all 4 sides so edge patches always have context
    margin = patch_size // 2
    padded = np.pad(
        image,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="reflect",
    )
    H_eff, W_eff = padded.shape[:2]

    H_pad, W_pad, positions = compute_patch_grid(H_eff, W_eff, patch_size, stride)

    # Additional tiling pad (bottom-right) if needed
    extra_b = H_pad - H_eff
    extra_r = W_pad - W_eff
    if extra_b > 0 or extra_r > 0:
        padded = np.pad(padded, ((0, extra_b), (0, extra_r), (0, 0)), mode="reflect")

    # Output accumulator at model-output resolution
    out_h = H_pad // output_stride
    out_w = W_pad // output_stride
    accum = np.zeros((num_classes, out_h, out_w), dtype="float32")
    count = np.zeros((1, out_h, out_w), dtype="float32")

    mean = np.array(normalize["mean"], dtype="float32")
    std = np.array(normalize["std"], dtype="float32")
    patch_out = patch_size // output_stride

    suppress_mask: np.ndarray | None = None
    if active_class_ids is not None and not infer_fn_returns_probs:
        m = np.ones(num_classes, dtype=bool)
        for cid in active_class_ids:
            if 0 <= cid < num_classes:
                m[cid] = False
        if np.any(m):
            suppress_mask = m

    # Gaussian weighting for smoother tile blending
    sigma = patch_out / 4.0
    ax = np.arange(patch_out, dtype="float32") - patch_out / 2.0 + 0.5
    xx, yy = np.meshgrid(ax, ax)
    gauss_weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype("float32")

    # Adaptive batch size based on available VRAM (OOM-safe)
    tile_batch = _estimate_tile_batch(patch_size, num_classes)
    tile_batch = min(tile_batch, len(positions))

    # Pre-extract ALL patches at once (avoids per-batch Python loop overhead)
    all_patches = np.empty((len(positions), 3, patch_size, patch_size), dtype="float32")
    for i, (y, x) in enumerate(positions):
        patch = padded[y:y + patch_size, x:x + patch_size]
        all_patches[i] = patch.transpose(2, 0, 1)
    # In-place normalization (avoids allocating a second 1.18 GiB array)
    all_patches *= (1.0 / 255.0)
    mean_bcast = mean.reshape(1, 3, 1, 1)
    std_bcast = std.reshape(1, 3, 1, 1)
    all_patches -= mean_bcast
    all_patches /= std_bcast

    bi = 0
    while bi < len(positions):
        if stop_flag and stop_flag():
            break
        chunk = positions[bi:bi + tile_batch]
        batch_np = all_patches[bi:bi + len(chunk)]
        try:
            raw_out = _normalize_logits_batch(infer_fn(batch_np), num_classes)
        except Exception as e:
            if "out of memory" in str(e).lower() and tile_batch > 1:
                # OOM: halve batch size, clear GPU cache, retry this chunk
                tile_batch = max(1, tile_batch // 2)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue  # retry same `bi` with smaller batch
            raise
        if raw_out.shape[0] != len(chunk):
            raise ValueError("infer_fn batch dimension mismatch")
        if raw_out.shape[2] != patch_out or raw_out.shape[3] != patch_out:
            raise ValueError("infer_fn spatial output shape mismatch")
        if infer_fn_returns_probs:
            probs_np = raw_out
        else:
            if suppress_mask is not None:
                raw_out[:, suppress_mask, :, :] = -1e9
            probs_np = _softmax_np(raw_out, axis=1)

        # Vectorized accumulation: batch multiply with gauss_weight then scatter
        weighted = probs_np * gauss_weight  # (B, C, pH, pW)
        for j, (y, x) in enumerate(chunk):
            oy = y // output_stride
            ox = x // output_stride
            accum[:, oy:oy + patch_out, ox:ox + patch_out] += weighted[j]
            count[:, oy:oy + patch_out, ox:ox + patch_out] += gauss_weight
        bi += tile_batch

    # Average overlapping regions
    count = np.maximum(count, 1.0)
    avg_probs = (accum / count).astype("float32")

    # Crop back to original output size (skip the reflect-pad margin)
    margin_out = margin // output_stride
    orig_out_h = H // output_stride
    orig_out_w = W // output_stride
    avg_probs = avg_probs[:, margin_out:margin_out + orig_out_h, margin_out:margin_out + orig_out_w]
    pred = np.argmax(avg_probs, axis=0).astype("int64")

    return pred, avg_probs


def sliding_window_predict(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    num_classes: int,
    output_stride: int,
    normalize: dict,
    active_class_ids: list[int] | None = None,
    device: torch.device | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Torch model wrapper around :func:`sliding_window_predict_infer_fn`.

    AMP autocast is enabled on CUDA devices: SW predict only feeds argmax
    so FP16 numerical noise (~1e-3) is well below the per-pixel decision
    margin. Yields ~1.5-2x speedup on Ampere/Turing without measurable
    metric change.
    """
    if device is None:
        device = next(model.parameters()).device

    _use_amp = (device.type == "cuda")

    def infer_fn(batch_np: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            batch_t = torch.from_numpy(batch_np).to(device, non_blocking=True)
            if _use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(batch_t)
                logits = logits.float()  # cast back for stable softmax
            else:
                logits = model(batch_t)
            return logits.detach().cpu().numpy()

    return sliding_window_predict_infer_fn(
        infer_fn,
        image,
        patch_size,
        stride,
        num_classes,
        output_stride,
        normalize,
        active_class_ids=active_class_ids,
        stop_flag=stop_flag,
    )


# ---------------------------------------------------------------------------
# Public aliases — external callers (trainer_api, etc.) should use the
# un-underscored names. The underscored variants remain as canonical
# definitions so in-module references keep working.
# ---------------------------------------------------------------------------
softmax_np = _softmax_np
normalize_logits_batch = _normalize_logits_batch

__all__ = [
    "compute_patch_grid",
    "normalize_logits_batch",
    "softmax_np",
    "build_sw_job",
    "iter_chunks",
    "accumulate_chunk",
    "finalize_job",
    "sliding_window_predict_infer_fn",
    "sliding_window_predict",
]
