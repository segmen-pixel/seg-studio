# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import base64
import hashlib
import io
import logging
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from segcore.image_io import imread as _imread

if TYPE_CHECKING:
    # torch is imported lazily at function scope at runtime; annotation-only here.
    import torch

# sklearn is imported lazily in _rf_train() (CPU fallback only)
_HAS_SKLEARN: bool | None = None  # None = not yet checked

# DINOv2: transformers is imported lazily in _try_load_dinov2() to avoid
# 10+ second startup penalty on cold boot (Windows DLL loading).
_HAS_TRANSFORMERS: bool | None = None  # None = not yet checked

from fastapi import HTTPException

from .annotate_index import find_annotate_image, load_annotate_index

# ---------------------------------------------------------------------------
# Random Forest / MLP pixel classifier
# ---------------------------------------------------------------------------
from .cache_utils import ThreadSafeLRUCache
from .paths import annotate_masks_dir
from .torch_device import current_configured_torch_device, resolve_torch_device_or_cpu

_RF_SIGMAS = (1.0, 2.0, 4.0)
_RF_FEATURE_VERSION = 4  # bump to invalidate feature cache when feature set changes
_N_HANDCRAFT = 26  # Number of handcraft feature dimensions (Lab + texture + edge + HSV + XY)
_RF_CACHE = ThreadSafeLRUCache(maxsize=20)
_RF_FEAT_CACHE = ThreadSafeLRUCache(maxsize=100)

# Sobel / Laplacian kernels (created once, moved to GPU on first use)
_RF_KERNELS = ThreadSafeLRUCache(maxsize=5)

_RF_MAX_PIXELS = 2_000_000  # Phase 5: higher resolution inference (was 750,000)

# ---------------------------------------------------------------------------
# DINOv2 model & feature caches
# ---------------------------------------------------------------------------
_DINO_MODEL_CACHE = ThreadSafeLRUCache(maxsize=1)
_DINO_FEAT_CACHE = ThreadSafeLRUCache(maxsize=10)
_DINO_MODEL_NAME = "facebook/dinov2-small"
_DINO_HIDDEN_DIM = 384
_DINO_MAX_SIZE = 448  # max input size (must be multiple of 14)
_DINO_PATCH_SIZE = 14
_DINO_BOUNDARY_SIGMA = 7.0
_DINO_BOUNDARY_WEIGHT = 3.0

_log = logging.getLogger(__name__)


def _try_load_dinov2(device: str) -> Any | None:
    """Load DINOv2-small model. Returns (model, processor) or None on failure."""
    global _HAS_TRANSFORMERS
    # Lazy check: only import transformers on first call
    if _HAS_TRANSFORMERS is None:
        try:
            import transformers  # noqa: F401
            _HAS_TRANSFORMERS = True
        except Exception:
            _HAS_TRANSFORMERS = False
    if not _HAS_TRANSFORMERS:
        _log.debug("transformers not installed, skipping DINOv2")
        return None

    import torch
    dev = torch.device(device)
    if dev.type == "cpu":
        _log.debug("CPU only, skipping DINOv2")
        return None

    cache_key = f"dinov2::{device}"
    cached = _DINO_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoImageProcessor, AutoModel
        _log.info("Loading DINOv2-small on %s ...", device)
        processor = AutoImageProcessor.from_pretrained(_DINO_MODEL_NAME)
        model = AutoModel.from_pretrained(_DINO_MODEL_NAME)
        model = model.to(dev).half().eval()
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False
        entry = (model, processor)
        _DINO_MODEL_CACHE.put(cache_key, entry)
        _log.info("DINOv2-small loaded successfully on %s", device)
        return entry
    except Exception as exc:
        _log.warning("Failed to load DINOv2: %s", exc)
        return None


def _dino_extract_features(image_bgr: np.ndarray, device: str) -> np.ndarray | None:
    """Extract DINOv2 patch token features at PATCH resolution (NOT pixel resolution).
    Returns (h_patches, w_patches, 384) float16 numpy array or None on failure.
    Caller maps pixel coords to patch indices via: pi = y * h_p // H, pj = x * w_p // W."""
    import torch

    H, W = image_bgr.shape[:2]

    img_hash = hashlib.md5(image_bgr.tobytes()[:4096]).hexdigest()[:12]
    cache_key = f"dino::{img_hash}::{H}x{W}"
    cached = _DINO_FEAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dino_entry = _try_load_dinov2(device)
    if dino_entry is None:
        return None

    model, processor = dino_entry
    dev = torch.device(device)

    try:
        scale = min(_DINO_MAX_SIZE / max(H, W), 1.0)
        new_h = max(_DINO_PATCH_SIZE, (int(H * scale) // _DINO_PATCH_SIZE) * _DINO_PATCH_SIZE)
        new_w = max(_DINO_PATCH_SIZE, (int(W * scale) // _DINO_PATCH_SIZE) * _DINO_PATCH_SIZE)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        pixel_values = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev).half()

        with torch.no_grad():
            outputs = model(pixel_values, output_hidden_states=False)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # skip CLS

        h_p = new_h // _DINO_PATCH_SIZE
        w_p = new_w // _DINO_PATCH_SIZE
        # Keep at patch resolution: (h_p, w_p, 384) — ~10MB for 68x96
        result = patch_tokens.squeeze(0).float().reshape(h_p, w_p, _DINO_HIDDEN_DIM).cpu().numpy().astype(np.float16)

        del pixel_values, outputs, patch_tokens
        torch.cuda.empty_cache()

        _DINO_FEAT_CACHE.put(cache_key, result)
        return result

    except Exception as exc:
        _log.warning("DINOv2 feature extraction failed: %s", exc)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Dual-Branch MLP for DINOv2 + handcraft features
# ---------------------------------------------------------------------------

class _DualBranchMLP:
    """Wrapper for dual-branch MLP (handcraft + DINOv2)."""

    @staticmethod
    def build(n_handcraft: int, n_dino: int, n_classes: int, device: Any) -> Any:
        """Build a dual-branch MLP module."""
        import torch.nn as nn

        class DualBranch(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hand_branch = nn.Sequential(
                    nn.Linear(n_handcraft, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                self.dino_branch = nn.Sequential(
                    nn.Linear(n_dino, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, n_classes),
                )
                self._n_handcraft = n_handcraft

            def forward(self, x: Any) -> Any:
                hand_feat = x[:, :self._n_handcraft]
                dino_feat = x[:, self._n_handcraft:]
                h = self.hand_branch(hand_feat)
                d = self.dino_branch(dino_feat)
                import torch
                combined = torch.cat([h, d], dim=1)
                return self.head(combined)

        return DualBranch().to(device)


def _dino_boundary_weights_from_feats(
    dino_feats: np.ndarray | None, img_shape: tuple[int, int],
) -> np.ndarray | None:
    """Compute boundary weight map from pre-extracted DINOv2 patch features.
    Avoids a redundant DINOv2 forward pass.
    Returns (H, W) float32 array with values in [1.0, _DINO_BOUNDARY_WEIGHT] or None."""
    if dino_feats is None:
        return None
    H, W = img_shape
    h_p, w_p, dim = dino_feats.shape
    tokens = dino_feats.astype(np.float32)  # (h_p, w_p, 384)

    # L2 normalize
    norms = np.linalg.norm(tokens, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    norm_t = tokens / norms

    # Cosine dissimilarity between adjacent patches
    h_sim = (norm_t[:, :-1, :] * norm_t[:, 1:, :]).sum(axis=2)  # (h_p, w_p-1)
    v_sim = (norm_t[:-1, :, :] * norm_t[1:, :, :]).sum(axis=2)  # (h_p-1, w_p)
    h_dissim = 1.0 - h_sim
    v_dissim = 1.0 - v_sim

    boundary = np.zeros((h_p, w_p), dtype=np.float32)
    boundary[:, :-1] = np.maximum(boundary[:, :-1], h_dissim)
    boundary[:, 1:] = np.maximum(boundary[:, 1:], h_dissim)
    boundary[:-1, :] = np.maximum(boundary[:-1, :], v_dissim)
    boundary[1:, :] = np.maximum(boundary[1:, :], v_dissim)

    # Upsample to pixel resolution (CPU)
    boundary_up = cv2.resize(boundary, (W, H), interpolation=cv2.INTER_LINEAR)

    # Gaussian blur
    ksize = int(_DINO_BOUNDARY_SIGMA * 6) | 1
    boundary_up = cv2.GaussianBlur(boundary_up, (ksize, ksize), _DINO_BOUNDARY_SIGMA)

    # Normalize to [1.0, BOUNDARY_WEIGHT]
    bmin, bmax = boundary_up.min(), boundary_up.max()
    if bmax > bmin:
        boundary_up = (boundary_up - bmin) / (bmax - bmin)
    else:
        boundary_up = np.zeros_like(boundary_up)
    return 1.0 + boundary_up * (_DINO_BOUNDARY_WEIGHT - 1.0)


def _rf_get_kernels(dev: torch.device) -> dict[str, Any]:
    """Lazily build and cache convolution kernels on the given device."""
    import torch
    key = str(dev)
    cached = _RF_KERNELS.get(key)
    if cached is not None:
        return cached
    # Sobel 3x3
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev).reshape(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev).reshape(1, 1, 3, 3)
    # Laplacian 3x3
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev).reshape(1, 1, 3, 3)
    # Gaussian kernels per sigma
    gauss = {}
    for sigma in _RF_SIGMAS:
        ksize = int(sigma * 6) | 1
        r = ksize // 2
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
        g1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g1d /= g1d.sum()
        g2d = (g1d[:, None] @ g1d[None, :]).reshape(1, 1, ksize, ksize)
        gauss[sigma] = g2d
    kernels = {"sx": sx, "sy": sy, "lap": lap, "gauss": gauss}
    _RF_KERNELS.put(key, kernels)
    return kernels


def _rf_compute_features_gpu(image_bgr: np.ndarray, dev: torch.device) -> torch.Tensor:
    """Per-pixel features on GPU: Lab + multi-scale texture + edge distance + local std + HSV HS + XY coords.
    Returns (H, W, D) fp16 tensor."""
    import torch
    import torch.nn.functional as F
    H, W = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Move to GPU as fp16: (1, C, H, W)
    lab_t = torch.from_numpy(lab).to(dev).permute(2, 0, 1).unsqueeze(0).half()
    gray_t = torch.from_numpy(gray).to(dev).unsqueeze(0).unsqueeze(0).half()
    k = _rf_get_kernels(dev)
    feats: list[torch.Tensor] = [lab_t]
    for sigma in _RF_SIGMAS:
        g2d = k["gauss"][sigma].half()
        pad_g = g2d.shape[-1] // 2
        blurred_lab = F.conv2d(F.pad(lab_t, [pad_g]*4, mode="reflect"),
                               g2d.expand(3, -1, -1, -1), groups=3)
        feats.append(blurred_lab)
        blurred_gray = F.conv2d(F.pad(gray_t, [pad_g]*4, mode="reflect"), g2d)
        sx = k["sx"].half()
        sy = k["sy"].half()
        gx = F.conv2d(F.pad(blurred_gray, [1]*4, mode="reflect"), sx)
        gy = F.conv2d(F.pad(blurred_gray, [1]*4, mode="reflect"), sy)
        feats.append(torch.sqrt(gx ** 2 + gy ** 2))
        feats.append(F.conv2d(F.pad(blurred_gray, [1]*4, mode="reflect"), k["lap"].half()).abs())
        # Phase 1: local standard deviation per scale (1 dim per scale)
        # Use float32 for variance computation to avoid fp16 precision issues (inf/nan)
        gray_f32 = gray_t.float()
        g2d_f32 = g2d.float()
        blurred_sq = F.conv2d(F.pad(gray_f32 ** 2, [pad_g]*4, mode="reflect"), g2d_f32)
        blurred_gray_f32 = F.conv2d(F.pad(gray_f32, [pad_g]*4, mode="reflect"), g2d_f32)
        local_var = (blurred_sq - blurred_gray_f32 ** 2).clamp(min=0)
        feats.append(torch.sqrt(local_var).half())

    # Phase 1: Canny edge distance (computed on CPU, transferred to GPU)
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge_dist = np.minimum(edge_dist, 50.0).astype(np.float32) / 50.0  # normalize to [0,1]
    edge_dist_t = torch.from_numpy(edge_dist).to(dev).unsqueeze(0).unsqueeze(0).half()
    feats.append(edge_dist_t)

    # Phase 1: HSV H and S channels
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hs = hsv[:, :, :2] / np.array([180.0, 255.0], dtype=np.float32)  # normalize H and S
    hs_t = torch.from_numpy(hs).to(dev).permute(2, 0, 1).unsqueeze(0).half()
    feats.append(hs_t)

    # Phase 1: normalized XY coordinates
    yy = torch.linspace(0.0, 1.0, H, device=dev, dtype=torch.float16).view(H, 1).expand(H, W)
    xx = torch.linspace(0.0, 1.0, W, device=dev, dtype=torch.float16).view(1, W).expand(H, W)
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, H, W)
    feats.append(coords)

    out = torch.cat(feats, dim=1).squeeze(0).permute(1, 2, 0)
    return out


def _rf_compute_features(image_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel features: Lab + multi-scale texture + edge distance + local std + HSV HS + XY coords. CPU path."""
    H, W = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    feats: list[np.ndarray] = [lab]
    for sigma in _RF_SIGMAS:
        ksize = int(sigma * 6) | 1
        blurred_lab = cv2.GaussianBlur(lab, (ksize, ksize), sigma)
        feats.append(blurred_lab)
        blurred_gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        gx = cv2.Sobel(blurred_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred_gray, cv2.CV_32F, 0, 1, ksize=3)
        feats.append(np.sqrt(gx ** 2 + gy ** 2)[:, :, np.newaxis])
        feats.append(np.abs(cv2.Laplacian(blurred_gray, cv2.CV_32F))[:, :, np.newaxis])
        # Phase 1: local standard deviation per scale
        blurred_sq = cv2.GaussianBlur(gray ** 2, (ksize, ksize), sigma)
        local_var = np.maximum(blurred_sq - blurred_gray ** 2, 0.0)
        feats.append(np.sqrt(local_var)[:, :, np.newaxis])

    # Phase 1: Canny edge distance (1 dim)
    gray_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge_dist = np.minimum(edge_dist, 50.0).astype(np.float32) / 50.0  # normalize to [0,1]
    feats.append(edge_dist[:, :, np.newaxis])

    # Phase 1: HSV H and S channels (2 dims)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hs = hsv[:, :, :2] / np.array([180.0, 255.0], dtype=np.float32)  # normalize H and S
    feats.append(hs)

    # Phase 1: normalized XY coordinates (2 dims)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx = xx / max(W - 1, 1)
    yy = yy / max(H - 1, 1)
    feats.append(np.stack([xx, yy], axis=2))

    return np.concatenate(feats, axis=2)


def _rf_compute_features_auto(image_bgr: np.ndarray, use_dino: bool = False,
                               device_id: str = "") -> np.ndarray:
    """Compute per-pixel handcraft features (26d). DINOv2 features are handled
    separately at patch resolution to avoid 1.76GB pixel-level arrays.
    Always returns a numpy ndarray (H, W, 26) suitable for caching."""
    if not device_id:
        device_id = resolve_torch_device_or_cpu(current_configured_torch_device())
    use_gpu = device_id.startswith("cuda") or device_id == "mps"
    if use_gpu:
        import torch
        dev = torch.device(device_id)
        features = _rf_compute_features_gpu(image_bgr, dev)
        if isinstance(features, torch.Tensor):
            features = features.cpu().float().numpy()
    else:
        features = _rf_compute_features(image_bgr)
    # DINOv2 features are NOT concatenated here — they are looked up per-pixel
    # at patch resolution during sampling in _rf_collect_data and _rf_predict

    return features


def _rf_annotations_hash(project_id: str) -> str:
    md = annotate_masks_dir(project_id)
    if not md.exists():
        return ""
    files = sorted(md.glob("*.png"))
    if not files:
        return ""
    total = sum(f.stat().st_size for f in files)
    mtime = max(f.stat().st_mtime for f in files)
    return f"{len(files)}_{total}_{mtime:.2f}"


def _rf_collect_data(project_id: str, use_dino: bool = False,
                     device_id: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect pixel features + labels + sample weights from all annotated masks.
    Returns (X, y, sample_weights) where sample_weights: FG=3.0, hard-neg=2.0, general-bg=1.0.
    When use_dino=True, also applies DINOv2 boundary weights to sample_weights."""
    index = load_annotate_index(project_id)
    import time as _time
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_w: list[np.ndarray] = []
    _t0 = _time.monotonic()
    _items_with_mask = [it for it in index.get("items", []) if it.get("annotation", {}).get("hasMask")]
    _n_items = len(_items_with_mask)
    _log.info("Collecting data from %d annotated images (use_dino=%s)", _n_items, use_dino)
    for _item_i, item in enumerate(_items_with_mask):
        item_id = item["id"]
        img_path = find_annotate_image(project_id, item_id)
        if not img_path:
            continue
        mask_path = annotate_masks_dir(project_id) / f"{item_id}.png"
        if not mask_path.exists():
            continue
        image = _imread(str(img_path))
        mask = _imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        # Downscale for faster feature computation
        n_pix = image.shape[0] * image.shape[1]
        if n_pix > _RF_MAX_PIXELS:
            scale = (float(_RF_MAX_PIXELS) / n_pix) ** 0.5
            new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Compute handcraft features ONCE for original image
        cache_key = f"{img_path}::v{_RF_FEATURE_VERSION}"
        orig_features = _RF_FEAT_CACHE.get(cache_key)
        if orig_features is not None and orig_features.shape[:2] != image.shape[:2]:
            orig_features = None
        if orig_features is None:
            orig_features = _rf_compute_features_auto(image, use_dino=False, device_id=device_id)
            orig_features = np.nan_to_num(orig_features, nan=0.0, posinf=0.0, neginf=0.0)
            _RF_FEAT_CACHE.put(cache_key, orig_features)

        # DINOv2 patch features (h_p, w_p, 384) — separate from handcraft
        dino_feats: np.ndarray | None = None
        if use_dino:
            dino_feats = _dino_extract_features(image, device_id)
            if dino_feats is None:
                _log.warning("DINOv2 extraction failed for %s, zero-padding", img_path)

        if (_item_i + 1) % 5 == 0 or _item_i == 0:
            _log.info("  image %d/%d  elapsed=%.1fs", _item_i + 1, _n_items, _time.monotonic() - _t0)

        # DINOv2 boundary weights (computed from extracted features, no extra forward pass)
        boundary_weight_map = None
        if use_dino:
            boundary_weight_map = _dino_boundary_weights_from_feats(dino_feats, image.shape[:2])

        # Augmentation: original + h-flip + v-flip
        # Instead of copying full feature arrays, we remap sampled indices
        # to original coordinates — saves ~128MB per flip copy
        h_img, w_img = mask.shape[:2]
        flat_f = orig_features.reshape(-1, orig_features.shape[2])  # (H*W, 26)

        aug_masks = [
            (0, mask),  # original
            (1, mask[:, ::-1].copy()),  # horizontal flip (only mask is copied, ~1MB)
            (2, mask[::-1, :].copy()),  # vertical flip
        ]

        for aug_i, aug_mask in aug_masks:
            flat_m = aug_mask.reshape(-1)
            rng = np.random.default_rng(42 + aug_i)
            aug_scale = 1.0 if aug_i == 0 else 0.5

            flat_bw = None
            if boundary_weight_map is not None:
                if aug_i == 0:
                    flat_bw = boundary_weight_map.reshape(-1)
                # boundary weights for flipped versions: flip the map
                elif aug_i == 1:
                    flat_bw = boundary_weight_map[:, ::-1].reshape(-1)
                elif aug_i == 2:
                    flat_bw = boundary_weight_map[::-1, :].reshape(-1)

            def _remap_indices(indices: np.ndarray) -> np.ndarray:
                """Map flat indices from augmented image back to original image coords."""
                if aug_i == 0:
                    return indices
                rows = indices // w_img
                cols = indices % w_img
                if aug_i == 1:  # h-flip
                    orig_cols = w_img - 1 - cols
                    return rows * w_img + orig_cols
                else:  # v-flip
                    orig_rows = h_img - 1 - rows
                    return orig_rows * w_img + cols

            def _append_samples(indices: np.ndarray, labels: np.ndarray, base_weight: float) -> None:
                """Append handcraft (+ DINOv2) features for sampled indices."""
                orig_idx = _remap_indices(indices)
                hand = flat_f[orig_idx].copy()  # (N, 26) — small copy
                # Fix XY coords for augmented samples
                if aug_i == 1:  # h-flip: invert x coord (index -2)
                    hand[:, -2] = 1.0 - hand[:, -2]
                elif aug_i == 2:  # v-flip: invert y coord (index -1)
                    hand[:, -1] = 1.0 - hand[:, -1]
                if use_dino:
                    if dino_feats is not None:
                        # Lookup DINOv2 with flipped patch coords
                        rows = indices // w_img
                        cols = indices % w_img
                        h_p, w_p = dino_feats.shape[:2]
                        patch_y = np.clip((rows * h_p) // h_img, 0, h_p - 1)
                        patch_x = np.clip((cols * w_p) // w_img, 0, w_p - 1)
                        if aug_i == 1:
                            patch_x = w_p - 1 - patch_x
                        elif aug_i == 2:
                            patch_y = h_p - 1 - patch_y
                        dino_sampled = dino_feats[patch_y, patch_x].astype(np.float32)
                    else:
                        dino_sampled = np.zeros((len(indices), _DINO_HIDDEN_DIM), dtype=np.float32)
                    combined = np.concatenate([hand, dino_sampled], axis=1)
                    all_X.append(combined)
                else:
                    all_X.append(hand)
                all_y.append(labels)
                w = np.full(len(indices), base_weight * aug_scale, dtype=np.float32)
                if flat_bw is not None:
                    w *= flat_bw[indices]
                all_w.append(w)

            # foreground: all labelled pixels (sparse -- keep them all), weight=3.0
            fg_idx = np.where((flat_m > 0) & (flat_m != 255))[0]
            if len(fg_idx) > 0:
                _append_samples(fg_idx, flat_m[fg_idx], 3.0)

            # hard negatives: background pixels near foreground, weight=2.0
            fg_binary = ((aug_mask > 0) & (aug_mask != 255)).astype(np.uint8)
            dilated = cv2.dilate(fg_binary, np.ones((31, 31), np.uint8), iterations=1)
            near_fg = ((dilated > 0) & (fg_binary == 0) & (aug_mask == 0)).reshape(-1)
            near_idx = np.where(near_fg)[0]
            max_near = max(3000, len(fg_idx) * 2)
            if len(near_idx) > max_near:
                near_idx = rng.choice(near_idx, max_near, replace=False)
            if len(near_idx) > 0:
                _append_samples(near_idx, np.zeros(len(near_idx), dtype=np.uint8), 2.0)

            # Phase 3: stratified 4x4 grid background sampling, weight=1.0
            bg_mask_2d = (aug_mask == 0)
            max_bg = max(5000, len(fg_idx) * 3)
            grid_rows, grid_cols = 4, 4
            per_cell = max(max_bg // (grid_rows * grid_cols), 1)
            cell_h, cell_w = h_img // grid_rows, w_img // grid_cols
            grid_bg_indices: list[np.ndarray] = []
            for gr in range(grid_rows):
                for gc in range(grid_cols):
                    r0, r1 = gr * cell_h, (gr + 1) * cell_h if gr < grid_rows - 1 else h_img
                    c0, c1 = gc * cell_w, (gc + 1) * cell_w if gc < grid_cols - 1 else w_img
                    cell_bg = bg_mask_2d[r0:r1, c0:c1]
                    cell_bg_ys, cell_bg_xs = np.where(cell_bg)
                    if len(cell_bg_ys) == 0:
                        continue
                    n_pick = min(per_cell, len(cell_bg_ys))
                    chosen = rng.choice(len(cell_bg_ys), n_pick, replace=False)
                    abs_rows = cell_bg_ys[chosen] + r0
                    abs_cols = cell_bg_xs[chosen] + c0
                    flat_indices = abs_rows * w_img + abs_cols
                    grid_bg_indices.append(flat_indices)
            if grid_bg_indices:
                bg_idx = np.concatenate(grid_bg_indices)
                if len(bg_idx) > max_bg:
                    bg_idx = rng.choice(bg_idx, max_bg, replace=False)
                _append_samples(bg_idx, np.zeros(len(bg_idx), dtype=np.uint8), 1.0)
    if not all_X:
        raise HTTPException(status_code=400, detail="No annotated masks found")
    X = np.vstack(all_X)
    _log.info("Data collection done: %d samples, %d features, %.1fs",
              X.shape[0], X.shape[1], _time.monotonic() - _t0)
    return X, np.concatenate(all_y), np.concatenate(all_w)


def _rf_train(project_id: str) -> dict[str, Any]:
    """Train pixel classifier (MLP on GPU or RF on CPU). Returns cache entry."""
    ann_hash = _rf_annotations_hash(project_id)
    device_id = resolve_torch_device_or_cpu(current_configured_torch_device())
    use_gpu = device_id.startswith("cuda") or device_id == "mps"
    _log.debug("device_id=%s  use_gpu=%s  configured=%s", device_id, use_gpu, current_configured_torch_device())

    # Determine if DINOv2 is available
    use_dino = False
    if use_gpu and _HAS_TRANSFORMERS:
        dino_entry = _try_load_dinov2(device_id)
        use_dino = dino_entry is not None

    cached = _RF_CACHE.get(project_id)
    if (cached is not None and cached["hash"] == ann_hash
            and cached.get("device_id") == device_id
            and cached.get("use_dino") == use_dino):
        return cached

    gpu_dev = None
    if use_gpu:
        import torch
        gpu_dev = torch.device(device_id)

    X, y, sample_weights = _rf_collect_data(project_id, use_dino=use_dino, device_id=device_id)
    n_features = X.shape[1]
    classes = np.unique(y)
    n_classes = int(classes.max()) + 1

    features_used = "handcraft+dinov2" if use_dino else "handcraft"

    if use_gpu:
        import torch
        import torch.nn as nn
        dev = gpu_dev

        if use_dino:
            # Dual-Branch MLP: handcraft 26d + DINOv2 384d = 410d
            n_handcraft = _N_HANDCRAFT
            expected = n_handcraft + _DINO_HIDDEN_DIM
            if n_features != expected:
                _log.error("Feature dim mismatch: got %d, expected %d (handcraft=%d + dino=%d)",
                           n_features, expected, n_handcraft, _DINO_HIDDEN_DIM)
                raise HTTPException(status_code=500, detail=f"Feature dimension mismatch: {n_features} != {expected}")
            model = _DualBranchMLP.build(n_handcraft, _DINO_HIDDEN_DIM, n_classes, dev)
        else:
            # Standard MLP for handcraft features only
            model = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes),
            ).to(dev)

        # Class weights
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        w = (1.0 / counts)
        w /= w.sum()
        w *= n_classes
        w_t = torch.tensor(w, device=dev)

        # Keep data on CPU, transfer mini-batches to GPU (3.3M × 410 × 4 = 5.4GB > 4GB VRAM)
        X_cpu = torch.tensor(X, dtype=torch.float32)  # CPU
        y_cpu = torch.tensor(y, dtype=torch.long)      # CPU
        sw_cpu = torch.tensor(sample_weights, dtype=torch.float32)  # CPU
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        n_epochs = 40
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=2e-4)
        loss_fn = nn.CrossEntropyLoss(weight=w_t, reduction='none')
        bs = 32768
        n = len(X_cpu)
        model.train()
        for _ep in range(n_epochs):
            perm = torch.randperm(n)  # CPU permutation
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:
                    continue
                xb = X_cpu[idx].to(dev)
                yb = y_cpu[idx].to(dev)
                wb = sw_cpu[idx].to(dev)
                logits = model(xb)
                per_sample_loss = loss_fn(logits, yb)
                loss = (per_sample_loss * wb).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()
        model.eval().half()  # fp16 for fast inference
        entry = {"hash": ann_hash, "type": "mlp", "model": model,
                 "device": dev, "device_id": device_id, "n_classes": n_classes,
                 "use_dino": use_dino, "features_used": features_used}
    else:
        global _HAS_SKLEARN
        if _HAS_SKLEARN is None:
            try:
                from sklearn.ensemble import RandomForestClassifier  # noqa: F401
                _HAS_SKLEARN = True
            except ImportError:
                _HAS_SKLEARN = False
        if not _HAS_SKLEARN:
            raise HTTPException(status_code=500, detail="No GPU and scikit-learn not installed")
        from sklearn.ensemble import RandomForestClassifier as _RFC
        clf = _RFC(n_estimators=100, max_depth=12, n_jobs=1,
                   class_weight="balanced", random_state=42)
        # Phase 3: pass sample_weights to RF fit
        clf.fit(X, y, sample_weight=sample_weights)
        entry = {"hash": ann_hash, "type": "rf", "model": clf, "device_id": device_id,
                 "n_classes": 0, "use_dino": False, "features_used": "handcraft"}

    _RF_CACHE.put(project_id, entry)
    return entry


def _rf_predict(image_bgr: np.ndarray, entry: dict[str, Any], img_path: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Predict mask + confidence."""
    orig_h, orig_w = image_bgr.shape[:2]
    use_dino = entry.get("use_dino", False)
    device_id = entry.get("device_id", "")

    # Downscale if too large
    n_pix = orig_h * orig_w
    if n_pix > _RF_MAX_PIXELS:
        scale = (float(_RF_MAX_PIXELS) / n_pix) ** 0.5
        small = cv2.resize(image_bgr, (int(orig_w * scale), int(orig_h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr

    # Handcraft features (always needed)
    cache_key = f"{img_path}::v{_RF_FEATURE_VERSION}" if img_path else ""
    features = _RF_FEAT_CACHE.get(cache_key) if cache_key else None
    if features is not None and features.shape[:2] != small.shape[:2]:
        features = None
    if features is None:
        features = _rf_compute_features_auto(small, use_dino=False, device_id=device_id)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if cache_key:
            _RF_FEAT_CACHE.put(cache_key, features)
    h, w = features.shape[:2]
    flat = features.reshape(-1, features.shape[2])  # (H*W, 26)

    # DINOv2 patch features for prediction
    dino_feats: np.ndarray | None = None
    if use_dino:
        dino_feats = _dino_extract_features(small, device_id)
        if dino_feats is None:
            _log.warning("DINOv2 extraction failed during prediction, zero-padding")

    if entry["type"] == "mlp":
        import torch
        dev = entry["device"]
        model = entry["model"]
        chunk = 100_000 if use_dino else 1_000_000
        preds, confs = [], []
        n_total = len(flat)
        with torch.no_grad():
            for i in range(0, n_total, chunk):
                hand_chunk = flat[i:i + chunk]  # (chunk_size, 26)
                if use_dino:
                    if dino_feats is not None:
                        h_p, w_p = dino_feats.shape[:2]
                        pixel_indices = np.arange(i, min(i + chunk, n_total))
                        rows = pixel_indices // w
                        cols = pixel_indices % w
                        patch_y = np.clip((rows * h_p) // h, 0, h_p - 1)
                        patch_x = np.clip((cols * w_p) // w, 0, w_p - 1)
                        dino_chunk = dino_feats[patch_y, patch_x].astype(np.float32)
                    else:
                        dino_chunk = np.zeros((len(hand_chunk), _DINO_HIDDEN_DIM), dtype=np.float32)
                    combined = np.concatenate([hand_chunk, dino_chunk], axis=1)
                    flat_t = torch.tensor(combined, dtype=torch.float16, device=dev)
                else:
                    flat_t = torch.tensor(hand_chunk, dtype=torch.float16, device=dev)
                logits = model(flat_t)
                preds.append(torch.argmax(logits, dim=1).cpu().to(torch.uint8).numpy())
                fg_p = 1.0 - torch.softmax(logits, dim=1)[:, 0]
                confs.append((fg_p * 255).clamp(0, 255).cpu().to(torch.uint8).numpy())
                del flat_t
        pred = np.concatenate(preds).reshape(h, w)
        conf = np.concatenate(confs).reshape(h, w)
    else:
        clf = entry["model"]
        pred = clf.predict(flat).reshape(h, w).astype(np.uint8)
        proba = clf.predict_proba(flat)
        classes = clf.classes_
        bg_col = np.where(classes == 0)[0]
        fg_prob = (1.0 - proba[:, bg_col[0]]) if len(bg_col) > 0 else np.ones(len(flat))
        conf = np.clip(fg_prob * 255, 0, 255).astype(np.uint8).reshape(h, w)

    # Phase 4: Edge-aware post-processing
    # Bilateral filter on confidence map for edge-preserving smoothing
    conf = cv2.bilateralFilter(conf, d=9, sigmaColor=75, sigmaSpace=75)
    # Morphological open + close on each foreground class mask with 3x3 elliptical kernel
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_classes = np.unique(pred)
    fg_classes = fg_classes[fg_classes > 0]  # skip background class 0
    for cls_id in fg_classes:
        cls_mask = (pred == cls_id).astype(np.uint8)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, morph_kernel)
        cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, morph_kernel)
        # Apply: set pixels that were this class but removed by morphology to 0
        pred[pred == cls_id] = 0
        pred[cls_mask > 0] = cls_id

    # Upscale back to original size if downscaled
    if pred.shape[0] != orig_h or pred.shape[1] != orig_w:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return pred, conf


def _encode_png_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Public aliases — routers and other callers should use the un-underscored
# names. The underscored variants remain as the canonical definitions so
# in-module references and ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
rf_collect_data = _rf_collect_data
rf_train = _rf_train
rf_predict = _rf_predict
encode_png_base64 = _encode_png_base64

