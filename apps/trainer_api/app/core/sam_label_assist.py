# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from segcore.image_io import imread as _imread

from .annotate_index import find_annotate_image, load_annotate_index
from .paths import annotate_masks_dir
from .sam_assist import _sam_load_predictor
from .torch_device import current_configured_torch_device, resolve_torch_device_or_cpu

# ---------------------------------------------------------------------------
# SAM Label Assist — learn per-class binary heads from SAM encoder features
# ---------------------------------------------------------------------------

_SLA_SUPPORTED_MODELS = {"mobile_sam", "sam2_tiny", "sam2_small", "tinysam"}

from .cache_utils import ThreadSafeLRUCache

_SLA_CACHE = ThreadSafeLRUCache(maxsize=20)
_SLA_FEAT_CACHE = ThreadSafeLRUCache(maxsize=50)


def _sla_get_feature_map(predictor: Any, model_name: str) -> Any:
    """Extract the (1, 256, 64, 64) encoder feature map from a SAM predictor
    after set_image has been called."""
    import torch  # noqa: F401
    if model_name in ("sam2_tiny", "sam2_small"):
        # SAM2ImagePredictor stores features in _features dict
        feat = predictor._features["image_embed"]  # (1, 256, 64, 64)
    else:
        # MobileSAM, TinySAM use SamPredictor.features
        feat = predictor.features  # (1, 256, 64, 64)
    return feat


def _sla_extract_features(predictor: Any, model_name: str,
                          image_rgb: np.ndarray, img_path: str) -> Any:
    """Encode image and return cached (256, 64, 64) feature tensor."""
    import torch  # noqa: F401
    cache_key = f"{model_name}:{img_path}"
    cached = _SLA_FEAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    predictor.set_image(image_rgb)
    feat = _sla_get_feature_map(predictor, model_name)  # (1, 256, 64, 64)
    feat = feat.squeeze(0).detach()  # (256, 64, 64)
    _SLA_FEAT_CACHE.put(cache_key, feat)
    return feat


def _sla_annotations_hash(project_id: str) -> str:
    """Quick hash of mask files for cache invalidation."""
    md = annotate_masks_dir(project_id)
    if not md.exists():
        return ""
    files = sorted(md.glob("*.png"))
    if not files:
        return ""
    total = sum(f.stat().st_size for f in files)
    mtime = max(f.stat().st_mtime for f in files)
    return f"{len(files)}_{total}_{mtime:.2f}"


def _sla_collect_data(project_id: str, model_name: str, device: str) -> dict[int, tuple[Any, Any]]:
    """Collect per-class (X, y) training data from all annotated images.
    Returns {class_id: (X_tensor[N,256], y_tensor[N])} on the given device."""
    import torch

    predictor = _sam_load_predictor(model_name, device)
    index = load_annotate_index(project_id)

    # class_id -> lists of feature vectors and labels
    class_data: dict[int, tuple[list, list]] = {}  # {cls: ([feat_vecs], [labels])}

    for item in index.get("items", []):
        ann = item.get("annotation", {})
        if not ann.get("hasMask"):
            continue
        item_id = item["id"]
        img_path = find_annotate_image(project_id, item_id)
        if not img_path:
            continue
        mask_path = annotate_masks_dir(project_id) / f"{item_id}.png"
        if not mask_path.exists():
            continue

        image_bgr = _imread(str(img_path))
        mask = _imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            continue
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Extract SAM encoder features (256, 64, 64)
        feat = _sla_extract_features(predictor, model_name, image_rgb, str(img_path))
        feat_h, feat_w = feat.shape[1], feat.shape[2]  # 64, 64

        # Resize mask to feature map resolution
        mask_small = cv2.resize(mask, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)

        # feat_flat: (256, H*W) -> (H*W, 256)
        feat_flat = feat.reshape(256, -1).permute(1, 0)  # (4096, 256)

        rng = np.random.default_rng(42)
        unique_classes = np.unique(mask_small)
        unique_classes = unique_classes[(unique_classes > 0) & (unique_classes != 255)]

        for cls_id in unique_classes:
            cls_id = int(cls_id)
            if cls_id not in class_data:
                class_data[cls_id] = ([], [])

            fg_mask = (mask_small == cls_id)
            fg_indices = np.where(fg_mask.ravel())[0]
            n_fg = len(fg_indices)
            if n_fg == 0:
                continue

            # Foreground samples (label=1)
            class_data[cls_id][0].append(feat_flat[fg_indices].cpu())
            class_data[cls_id][1].append(np.ones(n_fg, dtype=np.float32))

            # Hard negative: dilate fg by 5px, take surrounding bg pixels (label=0)
            fg_binary = fg_mask.astype(np.uint8)
            dilated = cv2.dilate(fg_binary, np.ones((11, 11), np.uint8), iterations=1)
            hard_neg_mask = (dilated > 0) & (~fg_mask) & (mask_small == 0)
            hard_neg_indices = np.where(hard_neg_mask.ravel())[0]
            max_hard = int(n_fg * 1.5)
            if len(hard_neg_indices) > max_hard:
                hard_neg_indices = rng.choice(hard_neg_indices, max_hard, replace=False)
            if len(hard_neg_indices) > 0:
                class_data[cls_id][0].append(feat_flat[hard_neg_indices].cpu())
                class_data[cls_id][1].append(np.zeros(len(hard_neg_indices), dtype=np.float32))

            # Random background (label=0)
            bg_mask = (mask_small == 0) & (~hard_neg_mask if len(hard_neg_indices) > 0 else np.ones_like(mask_small, dtype=bool))
            bg_indices = np.where(bg_mask.ravel())[0]
            max_rand = int(n_fg * 1.5)
            if len(bg_indices) > max_rand:
                bg_indices = rng.choice(bg_indices, max_rand, replace=False)
            if len(bg_indices) > 0:
                class_data[cls_id][0].append(feat_flat[bg_indices].cpu())
                class_data[cls_id][1].append(np.zeros(len(bg_indices), dtype=np.float32))

    if not class_data:
        raise HTTPException(status_code=400, detail="No annotated masks found for SAM Label Assist")

    # Concatenate and move to device
    result: dict[int, tuple[Any, Any]] = {}
    for cls_id, (feat_list, label_list) in class_data.items():
        X = torch.cat(feat_list, dim=0).to(device)     # (N, 256)
        y = torch.tensor(np.concatenate(label_list), dtype=torch.float32, device=device)  # (N,)
        result[cls_id] = (X, y)

    return result


def _sla_train(project_id: str, model_name: str) -> dict[str, Any]:
    """Train per-class binary MLP heads. Returns cache entry."""
    import torch
    import torch.nn as nn

    if model_name not in _SLA_SUPPORTED_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"SAM Label Assist does not support {model_name}. "
                                   f"Supported: {sorted(_SLA_SUPPORTED_MODELS)}")

    device = resolve_torch_device_or_cpu(current_configured_torch_device())
    ann_hash = _sla_annotations_hash(project_id)
    cache_key = f"{project_id}:{model_name}"

    cached = _SLA_CACHE.get(cache_key)
    if cached is not None and cached["hash"] == ann_hash and cached.get("device") == device:
        logging.getLogger(__name__).debug("Cache hit for %s", cache_key)
        return cached

    logging.getLogger(__name__).debug("Training heads for %s on %s", cache_key, device)

    # Collect per-class training data
    class_data = _sla_collect_data(project_id, model_name, device)

    # Train a binary MLP head per class
    heads: dict[int, nn.Module] = {}
    for cls_id, (X, y) in class_data.items():
        n_pos = y.sum().item()
        n_neg = len(y) - n_pos
        if n_pos < 1:
            continue

        head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        # pos_weight for imbalanced data
        pw = torch.tensor([n_neg / max(n_pos, 1)], device=device).clamp(max=10.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

        head.train()
        bs = 4096
        n = len(X)
        for _ep in range(50):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                logits = head(X[idx]).squeeze(-1)
                loss = loss_fn(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        head.eval()
        heads[cls_id] = head
        logging.getLogger(__name__).debug("  class %d: %d fg, %d bg, loss=%.4f", cls_id, int(n_pos), int(n_neg), loss.item())

    entry = {
        "hash": ann_hash,
        "heads": heads,
        "model_name": model_name,
        "device": device,
    }
    _SLA_CACHE.put(cache_key, entry)
    return entry


def _sla_predict(image_bgr: np.ndarray, entry: dict[str, Any],
                 img_path: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Predict mask + confidence for one image using trained heads.
    Returns (mask_uint8[H,W], confidence_uint8[H,W])."""
    import torch

    model_name = entry["model_name"]
    device = entry["device"]
    heads = entry["heads"]

    if not heads:
        raise HTTPException(status_code=400, detail="No trained heads available")

    predictor = _sam_load_predictor(model_name, device)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Extract features
    feat = _sla_extract_features(predictor, model_name, image_rgb, img_path)
    feat_h, feat_w = feat.shape[1], feat.shape[2]  # 64, 64

    # feat_flat: (H*W, 256)
    feat_flat = feat.reshape(256, -1).permute(1, 0)

    # Run all class heads
    prob_maps: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for cls_id, head in heads.items():
            logits = head(feat_flat).squeeze(-1)  # (H*W,)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(feat_h, feat_w)
            prob_maps[cls_id] = probs

    # Compose multi-class mask: argmax with threshold 0.5
    mask_small = np.zeros((feat_h, feat_w), dtype=np.uint8)
    conf_small = np.zeros((feat_h, feat_w), dtype=np.float32)

    if prob_maps:
        # Stack all class probabilities
        cls_ids = sorted(prob_maps.keys())
        prob_stack = np.stack([prob_maps[c] for c in cls_ids], axis=0)  # (C, H, W)

        # For each pixel, pick the class with highest prob if above 0.5
        max_prob = prob_stack.max(axis=0)  # (H, W)
        max_cls_idx = prob_stack.argmax(axis=0)  # (H, W)

        for i, cls_id in enumerate(cls_ids):
            where = (max_cls_idx == i) & (max_prob >= 0.5)
            mask_small[where] = cls_id
            conf_small[where] = prob_stack[i][where]

        # Background confidence: 1 - max_prob for bg pixels
        bg_mask = mask_small == 0
        conf_small[bg_mask] = 1.0 - max_prob[bg_mask]
        # FG pixels already have their prob assigned above

    # Upscale to original resolution
    orig_h, orig_w = image_bgr.shape[:2]
    mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    conf_float = cv2.resize(conf_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Morphological post-processing on each class
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_classes = np.unique(mask)
    fg_classes = fg_classes[fg_classes > 0]
    for cls_id in fg_classes:
        cls_mask = (mask == cls_id).astype(np.uint8)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, morph_kernel)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, morph_kernel)
        mask[mask == cls_id] = 0
        mask[cls_mask > 0] = cls_id

    # Superpixel boundary snap: majority vote per superpixel
    from .superpixel import compute_superpixels
    segments = compute_superpixels(image_bgr, n_segments=800, img_path=img_path)
    for seg_id in np.unique(segments):
        seg_pixels = segments == seg_id
        vals, counts = np.unique(mask[seg_pixels], return_counts=True)
        mask[seg_pixels] = vals[np.argmax(counts)]

    # Convert confidence to uint8 [0, 255]
    confidence = np.clip(conf_float * 255, 0, 255).astype(np.uint8)

    return mask, confidence


# ---------------------------------------------------------------------------
# Public aliases — routers should use the un-underscored names. The
# underscored variants remain as the canonical definitions so in-module
# references and ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
sla_train = _sla_train
sla_predict = _sla_predict

