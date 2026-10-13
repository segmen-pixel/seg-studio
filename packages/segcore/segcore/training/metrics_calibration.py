# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Confidence calibration: reliability bins, ECE and the reliability diagram.

Extracted from metrics.py during the pre-OSS refactor; metrics.py
re-exports these names for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Calibration bins for reliability diagram
_CAL_N_BINS = 10


def accumulate_calibration_bins(
    probs: np.ndarray,
    target: np.ndarray,
    ignore_index: int,
    n_bins: int = _CAL_N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate calibration stats from softmax probabilities.

    Args:
        probs: (C, H, W) softmax probabilities.
        target: (H, W) ground truth class indices.
        n_bins: number of confidence bins.

    Returns:
        (bin_correct, bin_confidence, bin_count) each of shape (n_bins,).
    """
    pred = np.argmax(probs, axis=0)
    # max confidence per pixel
    conf = np.max(probs, axis=0)
    valid = target != ignore_index
    pred_v = pred[valid].ravel()
    conf_v = conf[valid].ravel().astype("float64")
    tgt_v = target[valid].ravel()

    correct = (pred_v == tgt_v).astype("float64")

    bin_correct = np.zeros(n_bins, dtype="float64")
    bin_confidence = np.zeros(n_bins, dtype="float64")
    bin_count = np.zeros(n_bins, dtype="float64")

    bin_idx = np.clip((conf_v * n_bins).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = mask.sum()
        if cnt > 0:
            bin_count[b] = cnt
            bin_correct[b] = correct[mask].sum()
            bin_confidence[b] = conf_v[mask].sum()
    return bin_correct, bin_confidence, bin_count


def compute_ece(
    bin_correct: np.ndarray,
    bin_confidence: np.ndarray,
    bin_count: np.ndarray,
) -> float:
    """Compute Expected Calibration Error from accumulated bins."""
    total = bin_count.sum()
    if total == 0:
        return 0.0
    ece = 0.0
    for b in range(len(bin_count)):
        if bin_count[b] > 0:
            acc = bin_correct[b] / bin_count[b]
            avg_conf = bin_confidence[b] / bin_count[b]
            ece += (bin_count[b] / total) * abs(acc - avg_conf)
    return float(ece)


def draw_reliability_diagram(
    bin_correct: np.ndarray,
    bin_confidence: np.ndarray,
    bin_count: np.ndarray,
    ece: float,
    epoch: int,
    save_path: Path,
) -> None:
    """Draw a reliability diagram using PIL and save as PNG."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    n_bins = len(bin_count)
    W, H = 400, 400
    margin = 50
    plot_w = W - 2 * margin
    plot_h = H - 2 * margin

    img = PILImage.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Axes
    draw.rectangle([margin, margin, margin + plot_w, margin + plot_h], outline="black", width=1)
    # Diagonal (perfect calibration)
    draw.line([margin, margin + plot_h, margin + plot_w, margin], fill="#CCCCCC", width=1)

    bar_w = plot_w // n_bins
    for b in range(n_bins):
        if bin_count[b] > 0:
            acc = bin_correct[b] / bin_count[b]
            avg_conf = bin_confidence[b] / bin_count[b]
        else:
            acc = 0.0
            avg_conf = 0.0

        x0 = margin + b * bar_w
        x1 = x0 + bar_w - 2

        # Accuracy bar (blue)
        bar_h = int(acc * plot_h)
        y0 = margin + plot_h - bar_h
        draw.rectangle([x0, y0, x1, margin + plot_h], fill="#4477AA")

        # Confidence line (red tick)
        conf_y = margin + plot_h - int(avg_conf * plot_h)
        draw.line([x0, conf_y, x1, conf_y], fill="#CC3311", width=2)

    # Labels
    draw.text((margin, margin - 15), f"Epoch {epoch}  ECE={ece:.4f}", fill="black")
    draw.text((margin + plot_w // 2 - 30, margin + plot_h + 5), "Confidence", fill="black")
    # Y-axis ticks
    for i in range(0, 11, 2):
        y = margin + plot_h - int(i / 10 * plot_h)
        draw.text((margin - 25, y - 5), f"{i / 10:.1f}", fill="black")
    # X-axis ticks
    for i in range(0, 11, 2):
        x = margin + int(i / 10 * plot_w)
        draw.text((x - 5, margin + plot_h + 5), f"{i / 10:.1f}", fill="black")
    # Legend
    draw.rectangle([margin + plot_w - 100, margin + 5, margin + plot_w - 85, margin + 15], fill="#4477AA")
    draw.text((margin + plot_w - 80, margin + 5), "Accuracy", fill="black")
    draw.line([margin + plot_w - 100, margin + 25, margin + plot_w - 85, margin + 25], fill="#CC3311", width=2)
    draw.text((margin + plot_w - 80, margin + 20), "Confidence", fill="black")

    img.save(str(save_path))
