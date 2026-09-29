# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Lighting / time-of-day variation synthesis.

Transforms an existing labeled image to simulate a different lighting
condition (daytime / evening / night) while keeping the segmentation mask
untouched. Useful for outdoor inspection projects where collecting real
imagery across all lighting conditions is impractical.

The transforms are pure colour/contrast manipulations (HSV / gamma / white
balance / sensor-gain noise). No geometry change, so the existing mask is
reused 1-for-1.

Pipeline for one output sample:
  1. Pick a random labeled (image, mask) pair.
  2. Pick a variant ("daytime" | "evening" | "night") from the caller's
     allow-list, cycling fairly so every selected variant is represented.
  3. Apply the variant-specific colour transform.
  4. Return (synth_bgr, mask_u8, meta) in the same shape as
     synthesize_from_labeled so the caller can persist both kinds of
     synthesis through the same code path.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

# Variant → (R_mul, G_mul, B_mul, brightness_mul, gamma, noise_sigma)
# BGR order for cv2, so the multipliers here are (B, G, R) when we apply.
# Numbers come from a rough colour-constancy model tuned by eye; contributors
# wanting fancier physics can swap in a proper colour-temperature lookup.
_VARIANT_PARAMS: dict[str, tuple[float, float, float, float, float, float]] = {
    # (B_mul, G_mul, R_mul, brightness_mul, gamma, noise_sigma)
    "daytime": (1.00, 1.00, 1.00, 1.05, 0.95, 0.0),
    "evening": (0.78, 1.03, 1.22, 0.72, 1.20, 2.0),
    "night":   (1.32, 0.92, 0.72, 0.38, 1.35, 8.0),
}

SUPPORTED_VARIANTS: tuple[str, ...] = tuple(_VARIANT_PARAMS.keys())


def apply_time_of_day(
    bgr: np.ndarray,
    variant: str,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Return a copy of *bgr* transformed to the requested lighting variant.

    Parameters
    ----------
    bgr : np.ndarray of shape (H, W, 3), dtype uint8 (BGR order)
    variant : one of SUPPORTED_VARIANTS
    rng : optional random.Random for reproducible sensor-gain noise
    """
    if variant not in _VARIANT_PARAMS:
        raise ValueError(f"unknown lighting variant: {variant!r}")
    b_mul, g_mul, r_mul, brightness, gamma, noise_sigma = _VARIANT_PARAMS[variant]

    img = bgr.astype(np.float32)
    # Per-channel white balance
    img[..., 0] *= b_mul
    img[..., 1] *= g_mul
    img[..., 2] *= r_mul
    # Overall brightness
    img *= brightness
    # Clamp before gamma so dark-region crushing is predictable
    img = np.clip(img, 0.0, 255.0)
    # Gamma (applied on normalised [0,1] then scaled back)
    if abs(gamma - 1.0) > 1e-3:
        img = 255.0 * np.power(img / 255.0, gamma)
    # Sensor-gain noise — stronger at night
    if noise_sigma > 0:
        r = rng if rng is not None else random.Random()
        seed = int(r.random() * (2 ** 31 - 1))
        noise = np.random.default_rng(seed).normal(0.0, noise_sigma, size=img.shape)
        img = img + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def synthesize_lighting_variants(
    pairs: list[tuple[Path, Path]],
    n_samples: int,
    variants: Iterable[str] = ("daytime", "evening", "night"),
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Generate *n_samples* lighting-augmented samples from labeled pairs.

    Each output reuses its source image's mask verbatim. Variants are cycled
    round-robin so requesting 6 samples with ["evening","night"] yields 3 of
    each. Source images are sampled uniformly with replacement.
    """
    if not pairs:
        raise ValueError("synthesize_lighting_variants: empty pairs list")
    variants = tuple(v for v in variants if v in _VARIANT_PARAMS)
    if not variants:
        raise ValueError(
            f"no valid lighting variants selected (allowed: {SUPPORTED_VARIANTS})"
        )
    if n_samples <= 0:
        return []
    rng = random.Random(seed)
    out: list[tuple[np.ndarray, np.ndarray, dict]] = []
    for i in range(n_samples):
        variant = variants[i % len(variants)]
        img_p, mask_p = rng.choice(pairs)
        # cv2.imread can't handle non-ASCII paths on Windows; fall back to
        # fromfile+imdecode which reads bytes via python's path handling.
        try:
            img_bytes = np.fromfile(str(img_p), dtype=np.uint8)
            bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            mask_bytes = np.fromfile(str(mask_p), dtype=np.uint8)
            mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            _logger.warning("failed to read pair %s: %s", img_p, e)
            continue
        if bgr is None or mask is None:
            _logger.warning("decode failed for pair %s", img_p)
            continue
        synth = apply_time_of_day(bgr, variant, rng)
        meta = {
            "source_image": img_p.name,
            "variant": variant,
            "kind": "lighting",
        }
        out.append((synth, mask.astype(np.uint8), meta))
    return out
