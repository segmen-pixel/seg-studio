# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Numeric inference helpers shared by the prediction backends.

Extracted verbatim from prediction_engine.py during the pre-OSS refactor:
the _SUPPRESS_LOGIT constant, numpy softmax, the dtype-matching ORT logits
runner, CHW prob resizing, the sliding-window eligibility rule, and the
6-pass TTA drivers (torch + ORT).
"""
from __future__ import annotations

import numpy as np

from segcore.training.sliding_window import normalize_logits_batch

# Logit penalty for inactive (suppressed) classes.  Applied before softmax to
# drive their probability to ~0.  Must be large enough to dominate any real
# logit value, but small enough to avoid float overflow.  -1e9 is safe for
# float32; CoreML may use float16 internally so we keep a single value that
# works for both.
_SUPPRESS_LOGIT = -1e9


def _softmax_np(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    denom = np.sum(exp_logits, axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp_logits / denom


def _ort_run_logits(session: object, input_name: str, output_name: str, batch_np: np.ndarray, num_classes: int) -> np.ndarray:
    # Match input dtype to what the model expects (FP16 or FP32)
    expected_type = session.get_inputs()[0].type
    if "float16" in expected_type:
        batch_np = np.ascontiguousarray(batch_np, dtype="float16")
    else:
        batch_np = np.ascontiguousarray(batch_np, dtype="float32")
    outputs = session.run([output_name], {input_name: batch_np})
    # Always return float32 for downstream processing
    result = outputs[0]
    if result.dtype != np.float32:
        result = result.astype(np.float32)
    return normalize_logits_batch(result, num_classes)


def _resize_probs_chw_np(probs_chw: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    import cv2
    if probs_chw.ndim != 3:
        raise ValueError("probs_chw must be CHW")
    c, h, w = probs_chw.shape
    if h == target_h and w == target_w:
        return probs_chw.astype("float32")
    resized = np.zeros((c, target_h, target_w), dtype="float32")
    for idx in range(c):
        resized[idx] = cv2.resize(probs_chw[idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized



def _should_use_sliding_window(patch_size: int, sw_stride: int, output_stride: int) -> bool:
    return (
        patch_size > 0
        and sw_stride > 0
        and patch_size % output_stride == 0
        and sw_stride % output_stride == 0
    )


def _tta_predict_torch(model, input_tensor, inactive_class_ids, device):
    """6-pass TTA: orig + hflip + vflip + rot90(1,2,3). Returns averaged probs."""
    import torch

    transforms = [
        ("orig", lambda x: x, lambda p: p),
        ("hflip", lambda x: torch.flip(x, [3]), lambda p: torch.flip(p, [3])),
        ("vflip", lambda x: torch.flip(x, [2]), lambda p: torch.flip(p, [2])),
        ("rot90_1", lambda x: torch.rot90(x, 1, [2, 3]), lambda p: torch.rot90(p, -1, [2, 3])),
        ("rot90_2", lambda x: torch.rot90(x, 2, [2, 3]), lambda p: torch.rot90(p, -2, [2, 3])),
        ("rot90_3", lambda x: torch.rot90(x, 3, [2, 3]), lambda p: torch.rot90(p, -3, [2, 3])),
    ]
    avg_probs = None
    with torch.inference_mode():
        for _name, aug_fn, inv_fn in transforms:
            augmented = aug_fn(input_tensor)
            logits = model(augmented)
            if inactive_class_ids:
                logits[:, inactive_class_ids, :, :] = _SUPPRESS_LOGIT
            probs = torch.softmax(logits, dim=1)
            reversed_probs = inv_fn(probs)
            avg_probs = reversed_probs if avg_probs is None else avg_probs + reversed_probs
    return avg_probs / len(transforms)


def _tta_predict_ort(
    session: object,
    input_name: str,
    output_name: str,
    batch_np: np.ndarray,
    *,
    num_classes: int,
    inactive_class_ids: list[int],
) -> np.ndarray:
    transforms = [
        (lambda x: x, lambda p: p),
        (lambda x: np.flip(x, axis=3), lambda p: np.flip(p, axis=3)),
        (lambda x: np.flip(x, axis=2), lambda p: np.flip(p, axis=2)),
        (lambda x: np.rot90(x, 1, axes=(2, 3)), lambda p: np.rot90(p, -1, axes=(2, 3))),
        (lambda x: np.rot90(x, 2, axes=(2, 3)), lambda p: np.rot90(p, -2, axes=(2, 3))),
        (lambda x: np.rot90(x, 3, axes=(2, 3)), lambda p: np.rot90(p, -3, axes=(2, 3))),
    ]
    avg_probs: np.ndarray | None = None
    for aug_fn, inv_fn in transforms:
        augmented = np.ascontiguousarray(aug_fn(batch_np), dtype="float32")
        logits = _ort_run_logits(session, input_name, output_name, augmented, num_classes)
        if inactive_class_ids:
            logits[:, inactive_class_ids, :, :] = _SUPPRESS_LOGIT
        probs = _softmax_np(logits, axis=1)
        reversed_probs = np.ascontiguousarray(inv_fn(probs), dtype="float32")
        avg_probs = reversed_probs if avg_probs is None else avg_probs + reversed_probs
    return avg_probs / float(len(transforms))
