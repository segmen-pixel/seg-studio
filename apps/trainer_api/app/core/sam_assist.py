# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable
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


class CheckpointDownloadError(RuntimeError):
    """A model file that is not on this machine could not be fetched from any source.

    Its own class, so the route can say that and what to do: on a machine
    with no way out, the first use of SAM read as "SAM inference failed".
    filename: the file, and where it goes when placed by hand.
    """

    def __init__(self, filename: str) -> None:
        super().__init__(f"Failed to download {filename} from all sources")
        self.filename = filename


def _ensure_checkpoint(model_name: str) -> Path:
    """Ensure checkpoint exists, downloading if missing. Returns checkpoint path.

    Tries the segmen-pixel HF mirror first, then the authors' own URL. A
    download whose digest does not match the expected value is discarded and
    the next source is tried; if none match, no file is left behind and
    CheckpointDownloadError is raised.
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
    raise CheckpointDownloadError(filename) from last_err


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


#: Names for SAM's nested candidates, finest first.
#:
#: One click is ambiguous -- a point on a bolt thread could mean the mark, the
#: thread, or the bolt -- so SAM answers with several nested masks and leaves
#: the choice to the caller. It does not label them, and the channel order is
#: not a documented contract, so they are named here by area instead.
SAM_LEVELS: tuple[str, ...] = ("subpart", "part", "whole")


def label_levels(masks: Any, scores: Any) -> list[tuple[str, np.ndarray, float]]:
    """Pair each candidate with a granularity name, smallest area first.

    Fewer than three candidates are named from the coarse end: a model that
    returns one mask (TinySAM) is returning the whole object, not a sub-part.
    More than three keeps the extremes and the middle, so a name goes on
    meaning the same thing whatever model is behind it.
    """
    if len(masks) == 0:
        return []
    order = sorted(range(len(masks)), key=lambda i: int(np.count_nonzero(masks[i])))
    if len(order) > len(SAM_LEVELS):
        order = [order[0], order[len(order) // 2], order[-1]]
    names = list(SAM_LEVELS)[-len(order):]
    return [
        (names[k], masks[i].astype(np.uint8), float(scores[i]))
        for k, i in enumerate(order)
    ]


def positive_point_labels(points: list | None, labels: list | None) -> np.ndarray | None:
    """The label array for *points*: all ones, or ValueError if a label is not 1.

    Interactive SAM takes positive points only -- each one marks the object.
    The route refuses anything else first; this holds the same line in front
    of the predictor, so no caller reaches it with a point that means
    something else.
    """
    if not points:
        return None
    if labels is not None:
        if len(labels) != len(points):
            raise ValueError("points and labels must be the same length")
        if any(isinstance(v, bool) or v != 1 for v in labels):
            raise ValueError("only positive points are accepted: every label must be 1")
    return np.ones(len(points), dtype=np.int32)


def region_of(region: list | None, w: int, h: int) -> list[int] | None:
    """A region [x0, y0, x1, y1] of a w x h picture, held to the picture; None for no region.

    Whole pixels, the far edges rounded out so the region never loses what it
    was asked to hold. A region that is the whole picture is no region: SAM is
    then asked on the picture as it always was, under the embedding it always
    had. ValueError when none of it is on the picture.
    """
    if region is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in region)
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(int(w), int(math.ceil(x1))), min(int(h), int(math.ceil(y1)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"region {list(region)} is not on the {w}x{h} picture")
    if [x0, y0, x1, y1] == [0, 0, int(w), int(h)]:
        return None
    return [x0, y0, x1, y1]


def shift_prompts(points: list | None, box: list | None,
                  region: list[int] | None) -> tuple[list | None, list | None]:
    """Points and a box in the picture's pixels, moved into the region SAM is shown.

    Held to the region too: SAM is shown nothing beyond it, and a box reaching
    past its edge asks about pixels it does not have.
    """
    if not region:
        return points, box
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0

    def held(v: float, top: int) -> float:
        return min(max(float(v), 0.0), float(top))
    pts = [[held(p[0] - x0, rw - 1), held(p[1] - y0, rh - 1)] for p in points] if points else points
    bx = ([held(box[0] - x0, rw), held(box[1] - y0, rh), held(box[2] - x0, rw), held(box[3] - y0, rh)]
          if box else box)
    return pts, bx


#: The last picture decoded for SAM, by path: one, because a walk asks SAM on
#: one picture's regions one after another, and a large photograph is
#: hundreds of MB as RGB. Without it every region of it was decoded again.
_DECODED: dict[str, tuple[Any, np.ndarray]] = {}
_DECODED_LOCK = threading.Lock()


def _decoded_rgb(img_path: str) -> np.ndarray:
    """The picture at img_path as RGB, decoded once while it is the one being worked on."""
    try:
        st = os.stat(img_path)
        stamp: Any = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    with _DECODED_LOCK:
        got = _DECODED.get(img_path)
        if got is not None and got[0] == stamp and stamp is not None:
            return got[1]
    image = _imread(img_path)
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with _DECODED_LOCK:
        _DECODED.clear()
        _DECODED[img_path] = (stamp, rgb)
    return rgb


def image_size(img_path: str) -> tuple[int, int]:
    """(width, height) of the picture SAM is shown, as it decodes it."""
    rgb = _decoded_rgb(img_path)
    return int(rgb.shape[1]), int(rgb.shape[0])


def _sam_predict_levels(project_id: str, item_id: str, img_path: str,
                        points: list | None, labels: list | None,
                        box: list | None, model_name: str, region: list | None = None):
    """Every candidate for one prompt, and the name of the one argmax picks.

    Returning them all costs one extra encode and no extra inference -- the
    image embedding is already cached and the decoder already produced every
    mask -- which is what lets the annotator change granularity without
    asking the server again.

    region [x0, y0, x1, y1], in the picture's pixels, shows SAM that part of
    the picture alone, as a person zoomed in on it sees it; the prompt is in
    the picture's pixels as always, and the masks come back the size of the
    region.
    """
    import torch  # noqa: F401

    with _SAM_PREDICT_LOCK:
        masks, scores = _sam_predict_inner(
            project_id, item_id, img_path, points, labels, box, model_name, region)
    levels = label_levels(masks, scores)
    if not levels:
        return [], ""
    best_area = int(np.count_nonzero(masks[int(np.argmax(scores))]))
    default_name = levels[-1][0]
    for name, mask, _score in levels:
        if int(np.count_nonzero(mask)) == best_area:
            default_name = name
            break
    return levels, default_name


def _sam_predict(project_id: str, item_id: str, img_path: str,
                 points: list | None, labels: list | None,
                 box: list | None, model_name: str,
                 region: list | None = None) -> tuple[np.ndarray, float]:
    """Run SAM prediction with point and/or box prompts. Returns (mask_uint8, score).

    Serialized via _SAM_PREDICT_LOCK because the predictor holds internal
    embedding state — concurrent calls for different images would corrupt it.
    """
    import torch  # noqa: F401

    with _SAM_PREDICT_LOCK:
        masks, scores = _sam_predict_inner(
            project_id, item_id, img_path, points, labels, box, model_name, region)
    best = int(np.argmax(scores))
    return masks[best].astype(np.uint8), float(scores[best])


def _sam_predict_inner(project_id: str, item_id: str, img_path: str,
                       points: list | None, labels: list | None,
                       box: list | None, model_name: str, region: list | None = None):
    """Raw candidates straight from the predictor: (masks, scores), unranked.

    With a region, SAM is shown that part of the picture: the masks are the
    region's size, and the prompt is moved into it here.
    """
    import torch  # noqa: F401

    device = resolve_torch_device_or_cpu(current_configured_torch_device())
    predictor = _sam_load_predictor(model_name, device)

    # Cache image embedding for multi-click efficiency.
    # The predictor holds only ONE image's embedding at a time, so we must
    # track which image is currently loaded and re-run set_image when it changes.
    # _SAM_LOCK serialises the check-then-set_image to prevent concurrent
    # requests from corrupting the predictor's internal embedding state.
    # A region is another picture to the predictor, so it is part of the key;
    # without one the key is what it always was.
    if region is not None:
        w, h = image_size(img_path)
        region = region_of(region, w, h)
    emb_key = f"{model_name}:{img_path}" + ("@{},{},{},{}".format(*region) if region else "")
    with _SAM_LOCK:
        current_emb = _SAM_EMB_CACHE.get(f"_current:{model_name}")
        if current_emb != emb_key:
            image_rgb = _decoded_rgb(img_path)
            if region:
                x0, y0, x1, y1 = region
                image_rgb = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
            predictor.set_image(image_rgb)
            _SAM_EMB_CACHE.put(f"_current:{model_name}", emb_key)
    points, box = shift_prompts(points, box, region)

    # Build numpy arrays for the prompt. Every point is a positive one.
    point_coords = np.array(points, dtype=np.float32) if points else None
    point_labels = positive_point_labels(points, labels)
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

    return masks, scores


def encode_on_shared_predictor(predictor: Any, model_name: str, image_rgb: np.ndarray,
                               read: Callable[[Any], Any]) -> Any:
    """Encode image_rgb on sam-segment's predictor and read what is needed off it.

    For SAM Label Assist, which encodes its pictures on the predictor
    sam-segment uses (one per model and device). A click sets its picture
    under _SAM_LOCK and asks the predictor under _SAM_PREDICT_LOCK, after the
    first lock is let go: an encode taken under _SAM_LOCK alone could land in
    that gap, and the click was answered from the other picture's embedding.
    So both locks, in the order a click takes them, and ``read`` runs before
    either is let go. The record of which picture is loaded is cleared, so
    the next click encodes its own picture again.
    """
    with _SAM_PREDICT_LOCK, _SAM_LOCK:
        predictor.set_image(image_rgb)
        _SAM_EMB_CACHE.pop(f"_current:{model_name}")
        return read(predictor)


# ---------------------------------------------------------------------------
# Public alias — routers should use the un-underscored name. The underscored
# variant remains as the canonical definition so in-module references and
# ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
sam_predict = _sam_predict
sam_predict_levels = _sam_predict_levels

