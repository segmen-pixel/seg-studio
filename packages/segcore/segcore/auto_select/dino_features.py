# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Extract DINOv2 embeddings for project profile construction.

Uses ``dinov2_vitb14`` (768-d) via torch.hub.
Computes: global mean, foreground mean, background mean, foreground centroids.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DINO_DIM = 768  # vitb14 output dimension
_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_MAX_SAMPLES = 30
_NUM_CENTROIDS = 4

# Module-level cache for the model
_dino_model = None


def _get_dino_model(device: str = "cpu") -> torch.nn.Module:
    """Load DINOv2 model (cached)."""
    global _dino_model
    if _dino_model is not None:
        return _dino_model.to(device)

    logger.info("Loading %s via torch.hub...", _DINO_MODEL_NAME)
    model = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME)
    model.eval()
    model.to(device)
    _dino_model = model
    return model


def _list_paired_files(
    images_dir: Path, masks_dir: Path, max_samples: int = _MAX_SAMPLES,
) -> list[tuple[Path, Path]]:
    """Find image-mask pairs."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        for mext in (".png", ".bmp", ".tif"):
            mask_path = masks_dir / f"{img_path.stem}{mext}"
            if mask_path.exists():
                pairs.append((img_path, mask_path))
                break
    # Evenly sample
    if len(pairs) > max_samples:
        step = len(pairs) / max_samples
        pairs = [pairs[int(i * step)] for i in range(max_samples)]
    return pairs


def _preprocess_for_dino(img_np: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 image to DINOv2 input tensor.

    Resize so both dims are divisible by 14, normalize with ImageNet stats.
    """
    import cv2
    # BGR → RGB, float [0,1]
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    # Resize to nearest multiple of 14, max 518px
    new_h = min(518, max(14, (h // _DINO_PATCH_SIZE) * _DINO_PATCH_SIZE))
    new_w = min(518, max(14, (w // _DINO_PATCH_SIZE) * _DINO_PATCH_SIZE))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # HWC → CHW → BCHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _extract_patch_tokens(
    model: torch.nn.Module, img_tensor: torch.Tensor,
) -> np.ndarray:
    """Run DINOv2 and return patch tokens as (N, 768) numpy array."""
    with torch.no_grad():
        out = model.forward_features(img_tensor)
    # out["x_norm_patchtokens"] shape: (1, N, 768)
    tokens = out["x_norm_patchtokens"].squeeze(0).cpu().numpy()
    return tokens


def _simple_kmeans(vectors: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    """Simple k-means clustering, returns (k, d) centroids.

    No sklearn dependency needed.
    """
    n, d = vectors.shape
    if n <= k:
        # Not enough vectors: pad with zeros
        result = np.zeros((k, d), dtype=np.float32)
        result[:n] = vectors
        return result

    # Initialize with evenly-spaced samples
    indices = np.linspace(0, n - 1, k, dtype=int)
    centroids = vectors[indices].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = vectors[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


def extract_dino_embeddings(
    images_dir: str | Path,
    masks_dir: str | Path,
    device: str = "cpu",
    max_samples: int = _MAX_SAMPLES,
) -> dict[str, np.ndarray]:
    """Extract DINOv2 embeddings for a project.

    Returns
    -------
    dict with keys:
        dino_global_mean : (768,)
        dino_fg_mean : (768,)
        dino_bg_mean : (768,)
        dino_fg_centroids : (4, 768)
    """
    import cv2

    from segcore.image_io import imread as _imread

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    pairs = _list_paired_files(images_dir, masks_dir, max_samples)

    result = {
        "dino_global_mean": np.zeros(_DINO_DIM, dtype=np.float32),
        "dino_fg_mean": np.zeros(_DINO_DIM, dtype=np.float32),
        "dino_bg_mean": np.zeros(_DINO_DIM, dtype=np.float32),
        "dino_fg_centroids": np.zeros((_NUM_CENTROIDS, _DINO_DIM), dtype=np.float32),
    }

    if not pairs:
        logger.warning("No image-mask pairs found in %s / %s", images_dir, masks_dir)
        return result

    model = _get_dino_model(device)

    all_global = []
    all_fg = []
    all_bg = []

    for img_path, mask_path in pairs:
        try:
            img_bgr = _imread(str(img_path), cv2.IMREAD_COLOR)
            mask = _imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask is None:
                continue

            # Preprocess
            img_tensor = _preprocess_for_dino(img_bgr).to(device)
            tokens = _extract_patch_tokens(model, img_tensor)
            # tokens shape: (N_patches, 768)

            # Global embedding
            all_global.append(tokens.mean(axis=0))

            # Resize mask to patch grid
            h_patches = img_tensor.shape[2] // _DINO_PATCH_SIZE
            w_patches = img_tensor.shape[3] // _DINO_PATCH_SIZE
            mask_resized = cv2.resize(
                mask, (w_patches, h_patches), interpolation=cv2.INTER_NEAREST,
            )
            fg_flat = mask_resized.flatten() > 0

            if fg_flat.sum() > 0:
                all_fg.append(tokens[fg_flat].mean(axis=0))
            if (~fg_flat).sum() > 0:
                all_bg.append(tokens[~fg_flat].mean(axis=0))

        except Exception as e:
            logger.warning("DINOv2 extraction failed for %s: %s", img_path.name, e)
            continue

    if all_global:
        result["dino_global_mean"] = np.mean(all_global, axis=0).astype(np.float32)
    if all_fg:
        fg_stack = np.stack(all_fg)
        result["dino_fg_mean"] = np.mean(fg_stack, axis=0).astype(np.float32)
        result["dino_fg_centroids"] = _simple_kmeans(fg_stack, _NUM_CENTROIDS)
    if all_bg:
        result["dino_bg_mean"] = np.mean(all_bg, axis=0).astype(np.float32)

    logger.info(
        "DINOv2 extraction: %d images, %d fg, %d bg embeddings",
        len(all_global), len(all_fg), len(all_bg),
    )
    return result
