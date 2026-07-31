# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from segcore.image_io import imread as _imread

from .cache_utils import ThreadSafeLRUCache
from .torch_device import current_configured_torch_device, resolve_torch_device_or_cpu

# ---------------------------------------------------------------------------
# SAM click segmentation
# ---------------------------------------------------------------------------

_SAM_MODELS = ThreadSafeLRUCache(maxsize=10)
_SAM_EMB_CACHE = ThreadSafeLRUCache(maxsize=5)
_SAM_LOCK = threading.Lock()          # guards the check-then-set_image on _SAM_EMB_CACHE
_SAM_PREDICT_LOCK = threading.Lock()

_SAM_CHECKPOINTS = {
    "mobile_sam": "mobile_sam.pt",
    "sam2_tiny": "sam2.1_hiera_tiny.pt",
    "sam2_small": "sam2.1_hiera_small.pt",
    "tinysam": "tinysam.pth",
    "efficient_sam_ti": "efficient_sam_vitt.pt",
}

# Auto-download URLs.
# Primary: segmen-pixel HF mirror, where weights are redistributed under each
# upstream's Apache-2.0 license (see licenses/third_party/MODEL_WEIGHTS.md).
# Fallback: an authoritative upstream URL published by each model's authors.
# (2026-07-07 audit: the former `merve/*` community-mirror fallbacks were
# replaced — `merve/efficient-sam-vitt` had already rotted to a 401.)
_HF_REPO = "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints"
_SAM_DOWNLOAD_URLS: dict[str, list[str]] = {
    "mobile_sam.pt": [
        f"{_HF_REPO}/mobile_sam.pt",
        # Pinned to the same commit as the mobile-sam pip dependency and
        # scripts/build_installer.py — a moving branch would defeat the hash.
        "https://github.com/ChaoningZhang/MobileSAM/raw/b01a9ccef3b9e10b099b544efe004d0871802c3b/weights/mobile_sam.pt",
    ],
    "sam2.1_hiera_tiny.pt": [
        f"{_HF_REPO}/sam2.1_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ],
    "sam2.1_hiera_small.pt": [
        f"{_HF_REPO}/sam2.1_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ],
    "tinysam.pth": [
        f"{_HF_REPO}/tinysam.pth",
        # Authors' own HF repo (Apache-2.0, xinghaochen/TinySAM on GitHub).
        "https://huggingface.co/xinghaochen/tinysam/resolve/main/tinysam.pth",
    ],
    "efficient_sam_vitt.pt": [
        f"{_HF_REPO}/efficient_sam_vitt.pt",
        # Authors' own GitHub repo (Apache-2.0); raw link resolves the
        # LFS object via media.githubusercontent.
        "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt",
    ],
}

# SHA-256 of each checkpoint, kept identical to SAM_CHECKPOINTS in
# scripts/build_installer.py (recorded 2026-07-22 from the segmen-pixel
# HuggingFace mirror). Every download is checked against these regardless of
# which mirror served it, so a compromised or rotated mirror cannot slip a
# different file into a user's models directory.
_SAM_SHA256: dict[str, str] = {
    "mobile_sam.pt": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    "sam2.1_hiera_tiny.pt": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    "sam2.1_hiera_small.pt": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    "tinysam.pth": "4b8edcf93af46e2a658ae455574de62873778a5cc3fd8e8adf094dcdfa957cf2",
    "efficient_sam_vitt.pt": "dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a",
}

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "models" / "sam_checkpoints"
_log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    """Digest a checkpoint without holding it all in memory."""
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_checkpoint(model_name: str) -> Path:
    """Ensure checkpoint exists, downloading if missing. Returns checkpoint path.

    Tries the segmen-pixel HF mirror first, then the authors' own URL. A
    download whose digest does not match the expected value is discarded and
    the next source is tried; if none match, no file is left behind.
    """
    filename = _SAM_CHECKPOINTS[model_name]
    ckpt = _MODELS_DIR / filename
    if ckpt.exists():
        return ckpt
    urls = _SAM_DOWNLOAD_URLS.get(filename)
    if not urls:
        raise FileNotFoundError(f"No download URL for {filename}")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    tmp = ckpt.with_suffix(".download")
    last_err: Exception | None = None
    expected = _SAM_SHA256.get(filename)
    for url in urls:
        try:
            _log.info("Downloading SAM checkpoint: %s", url)
            urllib.request.urlretrieve(url, str(tmp))
            if expected:
                got = _sha256(tmp)
                if got != expected:
                    raise ValueError(
                        f"checksum mismatch for {filename}: expected {expected}, got {got}")
            tmp.rename(ckpt)
            _log.info("Downloaded %s (%.1f MB)", filename, ckpt.stat().st_size / 1024 / 1024)
            return ckpt
        except Exception as e:
            _log.warning("Download failed from %s: %s", url, e)
            last_err = e
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Failed to download {filename} from all sources") from last_err


class _EfficientSamPredictor:
    """Wrapper giving EfficientSAM a set_image/predict interface."""

    def __init__(self, model: Any, device: str):
        self.model = model
        self.device = device
        self._image_tensor: Any = None
        self._h = 0
        self._w = 0

    def set_image(self, image_rgb: np.ndarray) -> None:
        import torch
        self._h, self._w = image_rgb.shape[:2]
        self._image_tensor = (
            torch.from_numpy(image_rgb.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(self.device)

    def predict(self, point_coords: np.ndarray, point_labels: np.ndarray,
                multimask_output: bool = True) -> tuple:
        import torch
        pts = torch.from_numpy(point_coords).float().reshape(1, 1, -1, 2).to(self.device)
        labs = torch.from_numpy(point_labels).int().reshape(1, 1, -1).to(self.device)
        with torch.no_grad():
            logits, iou = self.model(self._image_tensor, pts, labs)
        masks_np = (logits[0, 0] > 0).cpu().numpy()  # (num_masks, H, W)
        scores_np = iou[0, 0].cpu().numpy()  # (num_masks,)
        h, w = self._h, self._w
        if masks_np.shape[-2:] != (h, w):
            resized = np.zeros((masks_np.shape[0], h, w), dtype=np.uint8)
            for i in range(masks_np.shape[0]):
                resized[i] = cv2.resize(
                    masks_np[i].astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks_np = resized
        return masks_np, scores_np, None


def _build_efficient_sam_predictor(model_name: str, ckpt: Path, device: str) -> _EfficientSamPredictor:
    """Build an EfficientSAM model and wrap it in a predictor."""
    import torch  # noqa: F401
    from efficient_sam.build_efficient_sam import build_efficient_sam
    model = build_efficient_sam(
        encoder_patch_embed_dim=192, encoder_num_heads=3,
        checkpoint=str(ckpt),
    )
    model.to(device)
    model.eval()
    return _EfficientSamPredictor(model, device)


def _sam_load_predictor(model_name: str, device: str) -> Any:
    """Load and cache a SAM predictor."""
    cache_key = f"{model_name}:{device}"
    cached = _SAM_MODELS.get(cache_key)
    if cached is not None:
        return cached

    import torch  # noqa: F401
    ckpt = _ensure_checkpoint(model_name)

    if model_name == "mobile_sam":
        from mobile_sam import SamPredictor as MobilePredictor
        from mobile_sam import sam_model_registry as mobile_registry
        sam = mobile_registry["vit_t"](checkpoint=str(ckpt))
        sam.to(device)
        sam.eval()
        predictor = MobilePredictor(sam)
    elif model_name in ("sam2_tiny", "sam2_small"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg = {
            "sam2_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        }[model_name]
        sam2 = build_sam2(cfg, str(ckpt), device=device)
        predictor = SAM2ImagePredictor(sam2)
    elif model_name == "tinysam":
        from tinysam import SamPredictor as TinyPredictor
        from tinysam import sam_model_registry as tiny_registry
        sam = tiny_registry["vit_t"](checkpoint=str(ckpt))
        sam.to(device)
        sam.eval()
        predictor = TinyPredictor(sam)
    elif model_name == "efficient_sam_ti":
        predictor = _build_efficient_sam_predictor(model_name, ckpt, device)
    else:
        raise ValueError(f"Unknown SAM model: {model_name}")

    _SAM_MODELS.put(cache_key, predictor)
    logging.getLogger(__name__).info("Loaded %s on %s", model_name, device)
    return predictor


def _sam_release_gpu() -> None:
    """Move all cached SAM models to CPU and free VRAM."""
    import torch
    for key in list(_SAM_MODELS._data.keys()):
        predictor = _SAM_MODELS.get(key)
        if predictor is None:
            continue
        model = getattr(predictor, "model", None)
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
    _SAM_EMB_CACHE._data.clear()
    torch.cuda.empty_cache()
    _log.info("SAM models moved to CPU, VRAM released")


def _sam_predict(project_id: str, item_id: str, img_path: str,
                 points: list | None, labels: list | None,
                 box: list | None, model_name: str) -> tuple[np.ndarray, float]:
    """Run SAM prediction with point and/or box prompts. Returns (mask_uint8, score).

    Serialized via _SAM_PREDICT_LOCK because the predictor holds internal
    embedding state — concurrent calls for different images would corrupt it.
    """
    import torch  # noqa: F401

    with _SAM_PREDICT_LOCK:
        return _sam_predict_inner(project_id, item_id, img_path, points, labels, box, model_name)


def _sam_predict_inner(project_id: str, item_id: str, img_path: str,
                       points: list | None, labels: list | None,
                       box: list | None, model_name: str) -> tuple[np.ndarray, float]:
    import torch  # noqa: F401

    device = resolve_torch_device_or_cpu(current_configured_torch_device())
    predictor = _sam_load_predictor(model_name, device)

    # Cache image embedding for multi-click efficiency.
    # The predictor holds only ONE image's embedding at a time, so we must
    # track which image is currently loaded and re-run set_image when it changes.
    # _SAM_LOCK serialises the check-then-set_image to prevent concurrent
    # requests from corrupting the predictor's internal embedding state.
    emb_key = f"{model_name}:{img_path}"
    with _SAM_LOCK:
        current_emb = _SAM_EMB_CACHE.get(f"_current:{model_name}")
        if current_emb != emb_key:
            image = _imread(img_path)
            if image is None:
                raise ValueError(f"Failed to read image: {img_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            _SAM_EMB_CACHE.put(f"_current:{model_name}", emb_key)

    # Build numpy arrays for points/labels
    point_coords = np.array(points, dtype=np.float32) if points else None
    point_labels = np.array(labels, dtype=np.int32) if labels else None
    box_np = np.array(box, dtype=np.float32) if box else None

    # EfficientSAM: convert box to point prompts (labels 2=top-left, 3=bottom-right)
    if model_name == "efficient_sam_ti" and box_np is not None:
        box_pts = np.array([[box_np[0], box_np[1]], [box_np[2], box_np[3]]], dtype=np.float32)
        box_labs = np.array([2, 3], dtype=np.int32)
        if point_coords is not None and point_labels is not None:
            point_coords = np.concatenate([point_coords, box_pts], axis=0)
            point_labels = np.concatenate([point_labels, box_labs], axis=0)
        else:
            point_coords = box_pts
            point_labels = box_labs
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    elif model_name == "efficient_sam_ti":
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    elif model_name == "tinysam":
        # TinySAM: no multimask_output parameter
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
        )
    else:
        # Standard SAM models (MobileSAM, SAM2)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
            multimask_output=True,
        )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8)  # 0/1
    best_score = float(scores[best_idx])

    return best_mask, best_score


# ---------------------------------------------------------------------------
# Public alias — routers should use the un-underscored name. The underscored
# variant remains as the canonical definition so in-module references and
# ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
sam_predict = _sam_predict

