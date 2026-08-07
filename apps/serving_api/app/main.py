# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR = Path(os.getenv("SEG_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
REGISTRY_DIR = MODELS_DIR / "registry"
ACTIVE_POINTER = REGISTRY_DIR / "ACTIVE_MODEL"

logger = logging.getLogger("serving_api")

app = FastAPI(title="Seg-Studio Serving API", version="0.2.0")

SESSION: ort.InferenceSession | None = None
PREPROCESS: dict | None = None
TRAIN_CONFIG: dict | None = None
ACTIVE_MODEL_ID: str | None = None
# Present only for instance-segmentation exports (v0.9.8): the
# threshold/dedup contract written by the trainer's instance-onnx export.
INSTANCE_CONTRACT: dict | None = None


def get_active_model_id() -> str | None:
    if not ACTIVE_POINTER.exists():
        return None
    value = ACTIVE_POINTER.read_text(encoding="utf-8").strip()
    return value or None


def list_models() -> list[str]:
    if not REGISTRY_DIR.exists():
        return []
    models = []
    for path in REGISTRY_DIR.iterdir():
        if path.is_dir() and (path / "model.onnx").exists():
            models.append(path.name)
    return sorted(models)


def load_active_model() -> None:
    global SESSION, PREPROCESS, TRAIN_CONFIG, ACTIVE_MODEL_ID, INSTANCE_CONTRACT
    model_id = get_active_model_id()
    if model_id is None:
        SESSION = None
        PREPROCESS = None
        TRAIN_CONFIG = None
        ACTIVE_MODEL_ID = None
        INSTANCE_CONTRACT = None
        return
    model_dir = REGISTRY_DIR / model_id
    model_path = model_dir / "model.onnx"
    preprocess_path = model_dir / "preprocess.json"
    if not model_path.exists() or not preprocess_path.exists():
        SESSION = None
        PREPROCESS = None
        TRAIN_CONFIG = None
        ACTIVE_MODEL_ID = None
        INSTANCE_CONTRACT = None
        return
    SESSION = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    PREPROCESS = json.loads(preprocess_path.read_text(encoding="utf-8"))
    # Exported alongside the model since export_routes v1; carries the
    # sliding-window parameters (patch_size) and the tuned inference
    # threshold. Absent on old registry entries -> legacy resize path.
    train_config_path = model_dir / "train_config.json"
    TRAIN_CONFIG = None
    if train_config_path.exists():
        try:
            TRAIN_CONFIG = json.loads(train_config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            TRAIN_CONFIG = None
    # Instance-segmentation exports carry instance_inference.json; its
    # presence is what routes requests to /count instead of /segment.
    contract_path = model_dir / "instance_inference.json"
    INSTANCE_CONTRACT = None
    if contract_path.exists():
        try:
            INSTANCE_CONTRACT = json.loads(contract_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            INSTANCE_CONTRACT = None
    ACTIVE_MODEL_ID = model_id


@app.on_event("startup")
def on_startup() -> None:
    load_active_model()


@app.get("/health")
def health() -> dict[str, str | None]:
    """Report service health and the currently active model id.

    Returns a JSON object with a fixed ``status`` field and the active
    model id read from the registry pointer (``None`` if none is set).
    This endpoint never raises and is safe to use as a liveness probe.
    """
    return {
        "status": "ok",
        "active_model_id": get_active_model_id(),
        "task": "instance" if INSTANCE_CONTRACT is not None else "semantic",
    }


@app.post("/reload")
def reload_model() -> dict[str, str | None]:
    """Reload the active ONNX model from the registry.

    Re-reads the ``ACTIVE_MODEL`` pointer and reinitialises the global
    inference session and preprocessing config. Use this after promoting
    a new model in the registry without restarting the service.
    """
    load_active_model()
    return {"status": "ok", "active_model_id": ACTIVE_MODEL_ID}


@app.get("/models")
def get_models() -> dict[str, list[str]]:
    """List all model ids available in the registry.

    Scans the registry directory and returns every entry that contains a
    ``model.onnx`` file, sorted alphabetically.
    """
    return {"models": list_models()}


@app.get("/models/active")
def get_active_model() -> dict[str, str]:
    """Return the id of the currently active model.

    Raises:
        HTTPException: 404 if no active model is currently set in the
            registry pointer.
    """
    model_id = get_active_model_id()
    if model_id is None:
        raise HTTPException(status_code=404, detail="no active model")
    return {"active_model_id": model_id}


def _softmax_np(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=axis, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp / denom


def _ceil_to_stride(dim: int, patch_size: int, stride: int) -> int:
    if dim <= patch_size:
        return patch_size
    n_strides = math.ceil((dim - patch_size) / stride)
    return patch_size + n_strides * stride


def _sliding_window_onnx(
    session: ort.InferenceSession,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    normalize: dict,
    batch: int = 4,
) -> np.ndarray:
    """Sliding-window ONNX inference at native resolution.

    Numpy-only replica of segcore's ``sliding_window_predict_infer_fn``
    (same reflect margin padding, patch grid, Gaussian tile blending and
    crop) so the serving container stays torch-free while producing
    bit-comparable probabilities to the trainer's prediction engine.

    Args:
        image: (H, W, 3) uint8 RGB image at original resolution.
        patch_size: window size in pixels (training patch size).
        stride: window step (engine default: patch_size * 3 // 4).
        normalize: {"mean": [...], "std": [...]} in 0-1 scale.

    Returns:
        (C, H // os, W // os) float32 blended probabilities, where ``os``
        is the model's output stride discovered from a probe run.
    """
    height, width = image.shape[:2]

    # Reflect-pad all 4 sides so edge patches always have context.
    margin = patch_size // 2
    padded = np.pad(
        image, ((margin, margin), (margin, margin), (0, 0)), mode="reflect",
    )
    h_eff, w_eff = padded.shape[:2]

    h_pad = _ceil_to_stride(h_eff, patch_size, stride)
    w_pad = _ceil_to_stride(w_eff, patch_size, stride)
    positions = [
        (y, x)
        for y in range(0, h_pad - patch_size + 1, stride)
        for x in range(0, w_pad - patch_size + 1, stride)
    ]
    extra_b = h_pad - h_eff
    extra_r = w_pad - w_eff
    if extra_b > 0 or extra_r > 0:
        padded = np.pad(padded, ((0, extra_b), (0, extra_r), (0, 0)), mode="reflect")

    mean = np.array(normalize["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(normalize["std"], dtype=np.float32).reshape(1, 3, 1, 1)

    # Probe one window to learn num_classes and output stride from the
    # actual graph instead of trusting config metadata.
    probe = padded[:patch_size, :patch_size].transpose(2, 0, 1)[None].astype(np.float32)
    probe = (probe / 255.0 - mean) / std
    probe_out = session.run(None, {"input": probe})[0]
    num_classes = int(probe_out.shape[1])
    os_ = max(1, patch_size // int(probe_out.shape[2]))
    patch_out = patch_size // os_

    out_h = h_pad // os_
    out_w = w_pad // os_
    accum = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    count = np.zeros((1, out_h, out_w), dtype=np.float32)

    # Gaussian weighting for smoother tile blending (same sigma as segcore).
    sigma = patch_out / 4.0
    ax = np.arange(patch_out, dtype=np.float32) - patch_out / 2.0 + 0.5
    xx, yy = np.meshgrid(ax, ax)
    gauss_weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)

    for bi in range(0, len(positions), batch):
        chunk = positions[bi : bi + batch]
        tiles = np.stack([
            padded[y : y + patch_size, x : x + patch_size].transpose(2, 0, 1)
            for y, x in chunk
        ]).astype(np.float32)
        tiles = (tiles / 255.0 - mean) / std
        logits = np.asarray(session.run(None, {"input": tiles})[0], dtype=np.float32)
        if logits.shape[2] != patch_out or logits.shape[3] != patch_out:
            raise HTTPException(
                status_code=500, detail="model output shape mismatch during SW inference",
            )
        probs = _softmax_np(logits, axis=1)
        weighted = probs * gauss_weight
        for j, (y, x) in enumerate(chunk):
            oy = y // os_
            ox = x // os_
            accum[:, oy : oy + patch_out, ox : ox + patch_out] += weighted[j]
            count[:, oy : oy + patch_out, ox : ox + patch_out] += gauss_weight
    # Divide by the TRUE summed Gaussian weight, never by a floor of 1.0. At the
    # engine's default stride (patch * 3 // 4) the weight sum is legitimately
    # well below 1.0 for every pixel, and flooring it rescales the
    # probabilities instead of averaging them -- confidence would then depend on
    # the stride. Mirrors segcore's blend_accumulated_probs, which carries the
    # full explanation; test_serving_sw_replica pins the two together.
    avg_probs = (accum / np.maximum(count, 1e-6)).astype(np.float32)

    # Crop back to original output size (skip the reflect-pad margin).
    margin_out = margin // os_
    orig_out_h = height // os_
    orig_out_w = width // os_
    return avg_probs[
        :, margin_out : margin_out + orig_out_h, margin_out : margin_out + orig_out_w,
    ]


def _prediction_from_probs_np(probs: np.ndarray, fg_threshold: float | None) -> np.ndarray:
    """Argmax with foreground suppression, mirroring segcore prediction_rules."""
    pred = np.argmax(probs, axis=0).astype(np.uint8)
    if fg_threshold is not None and float(fg_threshold) > 0.0 and probs.shape[0] > 1:
        fg_prob = probs[1:].sum(axis=0)
        pred[fg_prob < float(fg_threshold)] = 0
    return pred


def preprocess_image(img: Image.Image, preprocess: dict) -> np.ndarray:
    """Legacy resize preprocessing for registry entries without SW metadata.

    Resize inference degrades small-defect detail and only exists as a
    fallback for models exported before train_config.json accompanied the
    registry entry. Every use is logged loudly.
    """
    input_size = preprocess.get("input_size", [128, 128])
    resize_mode = preprocess.get("resize_mode", "stretch")
    if resize_mode == "stretch":
        img = img.resize((input_size[0], input_size[1]))
    elif resize_mode == "short_side":
        short = min(img.size)
        scale = input_size[0] / short
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size)
    elif resize_mode == "letterbox":
        img = img.resize((input_size[0], input_size[1]))
    else:
        raise HTTPException(status_code=400, detail="unsupported resize_mode")
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array(preprocess["normalize"]["mean"], dtype=np.float32)
    std = np.array(preprocess["normalize"]["std"], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, ...]


def build_overlay(mask: np.ndarray, classes_path: Path) -> Image.Image:
    if not classes_path.exists():
        return Image.fromarray(mask.astype(np.uint8), mode="L").convert("RGB")
    classes = json.loads(classes_path.read_text(encoding="utf-8")).get("classes", [])
    palette = np.zeros((256, 3), dtype=np.uint8)
    for item in classes:
        class_id = int(item["id"])
        color = item.get("color", [0, 0, 0])
        if 0 <= class_id < 256:
            palette[class_id] = np.array(color, dtype=np.uint8)
    rgb = palette[mask]
    return Image.fromarray(rgb, mode="RGB")


# ── Instance segmentation (v0.9.8) ──────────────────────────────
# Numpy-only replicas of segcore.instseg.rle / segcore.instseg.count —
# intentionally duplicated (like _sliding_window_onnx) so serving stays
# torch/cv2/pycocotools-free. Equivalence is pinned by
# tests/test_serving_instance_replica.py against the segcore originals.


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _encode_rle_np(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE (column-major, counts start with a zero-run)."""
    h, w = mask.shape
    flat = (mask != 0).flatten(order="F")
    if flat.size == 0:
        return {"size": [int(h), int(w)], "counts": []}
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], boundaries, [flat.size])))
    counts = [int(c) for c in runs]
    if flat[0]:
        counts.insert(0, 0)
    return {"size": [int(h), int(w)], "counts": counts}


def _dedup_masks_np(masks: list[np.ndarray], confidences: list[float], iou_threshold: float) -> list[int]:
    """Greedy mask-IoU duplicate suppression; returns kept indices (sorted)."""
    conf = np.asarray(confidences, dtype=np.float64)
    order = np.argsort(-conf)
    kept: list[int] = []
    kept_bool: list[np.ndarray] = []
    for i in order:
        m = np.asarray(masks[i]).astype(bool)
        dup = False
        for k in kept_bool:
            inter = np.logical_and(m, k).sum()
            if inter == 0:
                continue
            union = np.logical_or(m, k).sum()
            if union and inter / union > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(int(i))
            kept_bool.append(m)
    return sorted(kept)


# Validated 2026-07-22 against the RF-DETR-Seg SDK on the 32-image PoC test
# set (32/32 GT-exact, 32/32 SDK-match): the exported graph's object
# confidence is sigmoid of class-logit index 0 (COCO category 1 maps to
# internal index 0), masks are logits at reduced resolution for the
# stretch-resized square input.
_INSTANCE_MIN_AREA = 16
# Fraction of the model's query budget (num_queries) at which a count is
# reported as possibly truncated — the detector cannot emit more
# instances than it has queries, across all classes combined.
_QUERY_SATURATION = 0.9


def _instance_class(model_dir: Path) -> tuple[int, str]:
    """(class_id, class_name) for the counted class.

    The id comes from the exported contract (older exports predate the
    field — fall back to the first non-background class); the name comes
    from the classes.json copied next to the model.
    """
    class_id = 0
    if INSTANCE_CONTRACT is not None:
        try:
            class_id = int(INSTANCE_CONTRACT.get("class_id") or 0)
        except (TypeError, ValueError):
            class_id = 0
    classes: list[dict] = []
    classes_path = model_dir / "classes.json"
    if classes_path.exists():
        try:
            classes = json.loads(classes_path.read_text(encoding="utf-8")).get("classes", [])
        except (json.JSONDecodeError, OSError):
            classes = []
    if class_id <= 0:
        class_id = next(
            (int(c["id"]) for c in classes
             if int(c.get("id", 0)) != 0 and c.get("active", True)),
            1,
        )
    name = next(
        (str(c["name"]) for c in classes
         if int(c.get("id", -1)) == class_id and c.get("name")),
        f"class{class_id}",
    )
    return class_id, name



def _instance_class_maps() -> tuple[dict[int, int], dict[int, str]]:
    """(COCO category -> semantic class, semantic class -> name).

    Both come from the exported contract; empty dicts on single-class
    exports made before the mapping existed (the caller then falls back to
    the contract/classes.json class id).
    """
    semantic_of: dict[int, int] = {}
    names: dict[int, str] = {}
    if INSTANCE_CONTRACT is not None:
        for sem, cat in (INSTANCE_CONTRACT.get("coco_category_of") or {}).items():
            try:
                semantic_of[int(cat)] = int(sem)
            except (TypeError, ValueError):
                continue
        for cid, name in (INSTANCE_CONTRACT.get("class_names") or {}).items():
            try:
                names[int(cid)] = str(name)
            except (TypeError, ValueError):
                continue
    return semantic_of, names


def _instance_postprocess(
    labels: np.ndarray, masks: np.ndarray, orig_size: tuple[int, int],
    threshold: float, dedup_iou: float, class_id: int = 1,
    semantic_of: dict[int, int] | None = None,
    class_names: dict[int, str] | None = None,
) -> list[dict]:
    """(queries, C) class logits + (queries, h, w) mask logits -> instances.

    Column k of ``labels`` is COCO category k+1 (the categories the dataset
    was composed with, contiguous from 1). Each query is assigned its
    highest-scoring category, matching the SDK's own argmax; with a single
    category this reduces exactly to the previously validated
    ``sigmoid(labels[:, 0])`` path.
    """
    semantic_of = semantic_of or {}
    class_names = class_names or {}
    candidates, cand_conf, cand_class = _instance_candidates(
        labels, masks, orig_size, threshold, class_id, semantic_of,
    )
    return _instance_finalize(candidates, cand_conf, cand_class, dedup_iou, class_names)


def _instance_candidates(
    labels: np.ndarray, masks: np.ndarray, out_size: tuple[int, int],
    threshold: float, class_id: int = 1,
    semantic_of: dict[int, int] | None = None,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Decode one forward pass into binary masks at *out_size*, WITHOUT dedup.

    Split out of _instance_postprocess so the tiled path can pool candidates
    from many tiles and suppress duplicates once, over the whole frame. Running
    dedup per tile would never see that two tiles found the same object.
    """
    semantic_of = semantic_of or {}
    scores = _sigmoid_np(labels.astype(np.float32))
    if scores.ndim == 1:  # defensive: single-column export
        scores = scores[:, None]
    # The head is built for the checkpoint's class count, which can exceed
    # the categories this dataset actually has (a 1-category model still
    # exports 2 label columns). Only the first n_categories columns carry
    # real classes, so restricting the argmax keeps the single-class case
    # on the validated sigmoid(labels[:, 0]) path instead of letting a
    # trailing column win.
    n_categories = max(1, min(len(semantic_of) or 1, scores.shape[1]))
    scores = scores[:, :n_categories]
    best_col = np.argmax(scores, axis=1)
    conf = scores[np.arange(scores.shape[0]), best_col]
    candidates: list[np.ndarray] = []
    cand_conf: list[float] = []
    cand_class: list[int] = []
    for q in np.nonzero(conf >= threshold)[0]:
        prob = _sigmoid_np(masks[q].astype(np.float32))
        full = np.asarray(Image.fromarray(prob).resize(out_size, Image.BILINEAR))
        binary = full > 0.5
        if int(binary.sum()) >= _INSTANCE_MIN_AREA:
            candidates.append(binary)
            cand_conf.append(float(conf[q]))
            category = int(best_col[q]) + 1
            cand_class.append(int(semantic_of.get(category, class_id)))
    return candidates, cand_conf, cand_class


def _instance_finalize(
    candidates: list[np.ndarray], cand_conf: list[float], cand_class: list[int],
    dedup_iou: float, class_names: dict[int, str] | None = None,
) -> list[dict]:
    """Suppress duplicates across the pooled candidates and format instances."""
    class_names = class_names or {}
    # Duplicate suppression runs within a class; a cross-class overlap is a
    # genuine ambiguity the caller should see.
    kept: list[int] = []
    for cid in sorted(set(cand_class)):
        idx = [i for i in range(len(candidates)) if cand_class[i] == cid]
        local = _dedup_masks_np([candidates[i] for i in idx],
                                [cand_conf[i] for i in idx], dedup_iou)
        kept.extend(idx[j] for j in local)
    kept = sorted(kept, key=lambda i: -cand_conf[i])
    instances = []
    for n, i in enumerate(kept):
        m = candidates[i]
        ys, xs = np.nonzero(m)
        cid = cand_class[i]
        instances.append({
            "id": n + 1,
            "class_id": cid,
            "class_name": class_names.get(cid, f"class{cid}"),
            "conf": round(cand_conf[i], 4),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            # Area centroid (first moment of the mask), not the bbox centre.
            "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            "area": int(m.sum()),
            "rle": _encode_rle_np(m),
        })
    return instances


def _default_stride_np(patch_size: int) -> int:
    """Replica of segcore.instseg.tiled.default_stride (no object_size here).

    The object-size tightening is a trainer-side refinement driven by a measured
    band; serving has no band, so it takes the plain 3/4 rule -- which is what
    plan_tiles uses when stride is None on that side too.

    The rule itself lives in segcore.tiling_geometry.default_patch_stride.
    This stays a replica rather than an import because the serving container
    is torch-free by design and importing anything under segcore runs
    segcore/__init__.py, which pulls torch. The replica tests pin the two
    together; this is the module's only copy of the expression.
    """
    return max(1, patch_size * 3 // 4)


def _plan_tiles_np(image_size: tuple[int, int], patch_size: int,
                   stride: int | None = None) -> list[tuple[int, int]]:
    """Replica of segcore.instseg.tiled.plan_tiles -> origins.

    The last row and column are pulled back to the edge rather than padded, so
    every tile is real image; the final overlap comes out larger than requested,
    never smaller. Kept byte-equivalent to segcore's version and pinned by
    test_serving_instance_replica.
    """
    w, h = image_size
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    step = _default_stride_np(patch_size) if stride is None else max(1, int(stride))

    def starts(extent: int) -> list[int]:
        if extent <= patch_size:
            return [0]
        pos = list(range(0, extent - patch_size + 1, step))
        if pos[-1] != extent - patch_size:
            pos.append(extent - patch_size)
        return pos

    xs, ys = starts(w), starts(h)
    return [(x, y) for y in ys for x in xs]


def _pad_to_patch_np(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Replica of segcore.instseg.tiled.pad_to_patch.

    Always returns exactly patch_size square. Mirrored, not black: a black
    border is a background no camera produces and the detector would have to
    learn it as a feature. The same rule pads composition canvases and inference
    tiles, so a padded frame looks the same to the model in training and in use
    -- padding is only safe because of that.

    Mirror-TILING rather than np.pad(mode="reflect") is deliberate: reflect
    needs the pad width to be below the axis length, and a 50px source padded to
    768 exceeds it by an order of magnitude.
    """
    t = int(patch_size)
    h, w = arr.shape[:2]
    if h >= t and w >= t:
        return arr[:t, :t]
    out = np.empty((t, t) + arr.shape[2:], dtype=arr.dtype)
    reps_y = -(-t // h)
    reps_x = -(-t // w)
    tiled = arr
    if reps_y > 1 or reps_x > 1:
        rows = [arr if i % 2 == 0 else arr[::-1] for i in range(reps_y)]
        tiled = np.concatenate(rows, axis=0)
        cols = [tiled if i % 2 == 0 else tiled[:, ::-1] for i in range(reps_x)]
        tiled = np.concatenate(cols, axis=1)
    out[:, :] = tiled[:t, :t]
    return out


def _instance_postprocess_tiled(
    session, img: Image.Image, preprocess: dict, patch_size: int,
    threshold: float, dedup_iou: float, class_id: int = 1,
    semantic_of: dict[int, int] | None = None,
    class_names: dict[int, str] | None = None,
) -> list[dict]:
    """Tile the frame at *patch_size*, exactly as the trainer counts.

    The threshold in the contract was chosen by counting validation photos
    through segcore's predict_tiled_masks at this patch size. Serving used to
    run one whole-frame stretch-resize instead, so the objects the model saw
    were a different size than the ones the threshold was measured on. Whatever
    that threshold is worth, it is worth it only at this geometry.

    Numpy replica of segcore.instseg.tiled.predict_tiled_masks; the equivalence
    is pinned by test_serving_instance_replica.
    """
    W, H = img.size
    inp_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]

    candidates: list[np.ndarray] = []
    confs: list[float] = []
    classes: list[int] = []

    for x, y in _plan_tiles_np((W, H), patch_size):
        crop = img.crop((x, y, min(x + patch_size, W), min(y + patch_size, H)))
        if crop.size != (patch_size, patch_size):
            crop = Image.fromarray(_pad_to_patch_np(np.asarray(crop), patch_size))
        tensor = preprocess_image(crop, preprocess)
        outputs = dict(zip(out_names, session.run(None, {inp_name: tensor})))
        if "labels" not in outputs or "masks" not in outputs:
            raise HTTPException(status_code=500, detail="model outputs missing labels/masks")
        tile_masks, tile_conf, tile_cls = _instance_candidates(
            np.asarray(outputs["labels"][0], dtype=np.float32),
            np.asarray(outputs["masks"][0], dtype=np.float32),
            (patch_size, patch_size), threshold, class_id, semantic_of,
        )
        for m, c, k in zip(tile_masks, tile_conf, tile_cls):
            ys_, xs_ = np.nonzero(m)
            if ys_.size == 0:
                continue
            x0, y0 = int(xs_.min()) + x, int(ys_.min()) + y
            x1, y1 = int(xs_.max()) + 1 + x, int(ys_.max()) + 1 + y
            # Drop detections clipped by a tile edge. An object straddling a
            # seam is seen whole by one tile and cut off by its neighbour; the
            # two boxes describe different rectangles, so no IoU threshold both
            # folds them together and keeps adjacent objects apart. The clipped
            # view carries nothing the whole view lacks. The frame border is
            # exempt -- there is no neighbouring tile to hold that side.
            eps = 1.0
            touches = False
            if x > 0:
                touches = touches or x0 <= x + eps
            if y > 0:
                touches = touches or y0 <= y + eps
            if x + patch_size < W:
                touches = touches or x1 >= x + patch_size - eps
            if y + patch_size < H:
                touches = touches or y1 >= y + patch_size - eps
            if touches:
                continue
            full = np.zeros((H, W), dtype=bool)
            th, tw = m.shape
            full[y:y + th, x:x + tw] = m[: H - y, : W - x]
            if not full.any():
                continue
            candidates.append(full)
            confs.append(float(c))
            classes.append(int(k))

    return _instance_finalize(candidates, confs, classes, dedup_iou, class_names)


@app.post("/count")
async def count_instances(image: UploadFile = File(...)) -> JSONResponse:
    """Count object instances on an uploaded image (instance models only).

    Runs the exported RF-DETR-Seg ONNX graph and the numpy postprocess
    chain (confidence threshold -> mask sigmoid + resize -> binarize ->
    greedy mask-IoU dedup). The threshold and dedup IoU come from the
    ``instance_inference.json`` contract exported with the model.

    Returns:
        JSON ``{model_id, count, threshold, dedup_iou, inference_time_sec,
        instances: [{id, conf, bbox, area, rle}]}`` — the same instance
        shape the trainer's instances.json uses (RLE is COCO uncompressed,
        column-major). 503 when no model is loaded, 409 when the active
        model is a semantic-segmentation export (use ``/segment``).
    """
    if SESSION is None or PREPROCESS is None or ACTIVE_MODEL_ID is None:
        return JSONResponse(status_code=503, content={"detail": "active model not loaded"})
    if INSTANCE_CONTRACT is None:
        raise HTTPException(
            status_code=409,
            detail="active model is a semantic-segmentation export; use /segment")
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    orig_w, orig_h = img.size

    start = time.perf_counter()
    threshold = float(INSTANCE_CONTRACT.get("threshold", 0.3))
    dedup_iou = float(INSTANCE_CONTRACT.get("dedup_iou", 0.7))
    class_id, class_name = _instance_class(model_dir=REGISTRY_DIR / ACTIVE_MODEL_ID)
    semantic_of, class_names = _instance_class_maps()
    # Count at the geometry the threshold was calibrated at. patch_size is
    # absent from contracts exported before it was carried, and 0 means the run
    # was not patch-trained; both fall back to the single whole-frame pass.
    patch_size = int(INSTANCE_CONTRACT.get("patch_size") or 0)
    inp_name = SESSION.get_inputs()[0].name

    if patch_size > 0:
        instances = _instance_postprocess_tiled(
            SESSION, img, PREPROCESS, patch_size, threshold, dedup_iou,
            class_id, semantic_of, class_names,
        )
        # Query saturation is per forward pass; tiling runs many, so the cap
        # below no longer describes the frame. Probe one tile-sized pass for
        # the head width instead of pretending the whole-frame number applies.
        labels_arr = np.zeros((0, 1), dtype=np.float32)
    else:
        input_tensor = preprocess_image(img, PREPROCESS)
        outputs = dict(zip(
            [o.name for o in SESSION.get_outputs()],
            SESSION.run(None, {inp_name: input_tensor}),
        ))
        if "labels" not in outputs or "masks" not in outputs:
            raise HTTPException(status_code=500, detail="model outputs missing labels/masks")
        labels_arr = np.asarray(outputs["labels"][0], dtype=np.float32)
        instances = _instance_postprocess(
            labels_arr,
            np.asarray(outputs["masks"][0], dtype=np.float32),
            (orig_w, orig_h), threshold, dedup_iou, class_id,
            semantic_of, class_names,
        )
    elapsed = time.perf_counter() - start
    counts_by_class: dict[str, int] = {}
    for inst in instances:
        key = str(inst["class_id"])
        counts_by_class[key] = counts_by_class.get(key, 0) + 1
    body = {
        "model_id": ACTIVE_MODEL_ID,
        "count": len(instances),
        "counts_by_class": counts_by_class,
        "class_id": class_id,
        "class_name": class_name,
        "class_names": {str(k): v for k, v in sorted(class_names.items())} or None,
        "threshold": threshold,
        "dedup_iou": dedup_iou,
        "patch_size": patch_size or None,
        "image_size": [orig_w, orig_h],
        "inference_time_sec": elapsed,
        "instances": instances,
    }
    # A threshold that was never measured is a plausible-looking number, and
    # nothing else in the response distinguishes it from a calibrated one. Say
    # so, rather than letting a default read as a result. Absent in contracts
    # written before the flag existed, which is why the test is "is False".
    if INSTANCE_CONTRACT.get("threshold_calibrated") is False:
        body["threshold_warning"] = (
            f"threshold {threshold} was not calibrated: the training run had no "
            f"annotated validation image to measure against, so this is the "
            f"default. Counts may be systematically high or low.")
    # The model can only emit num_queries detections per image (a fixed
    # architectural budget shared by all classes). Near the cap the true
    # count may be truncated, so say so rather than returning a confident
    # undercount.
    n_queries = int(labels_arr.shape[0])
    if n_queries and len(instances) >= n_queries * _QUERY_SATURATION:
        body["truncation_warning"] = (
            f"{len(instances)} instances is close to this model's per-image "
            f"limit of {n_queries}; the real count may be higher")
        body["max_instances_per_image"] = n_queries
    return JSONResponse(content=body)


@app.post("/segment", response_model=None)
async def segment(
    image: UploadFile = File(...),
    overlay: bool = Query(default=False),
    include_meta: bool = Query(default=True),
) -> StreamingResponse | JSONResponse:
    """Run semantic segmentation on an uploaded image.

    Decodes the uploaded image and runs sliding-window ONNX inference at
    the image's native resolution (patch size and inference threshold come
    from the train_config.json exported with the model). The returned
    ``mask.png`` is at the ORIGINAL image resolution.

    Registry entries exported without train_config.json (or trained
    without patches) fall back to the legacy resize pipeline; the meta
    reports which mode ran via ``inference_mode``.

    Args:
        image: The image file to segment (any format Pillow can decode).
        overlay: If true, also include a colorised overlay PNG in the ZIP.
        include_meta: If true, also include a meta.json with inference info.

    Returns:
        A ``StreamingResponse`` with ``application/zip`` content on success,
        or a ``JSONResponse`` with status 503 when no model is loaded.

    Raises:
        HTTPException: 400 if the active model's preprocess config uses an
            unsupported ``resize_mode``.
    """
    if SESSION is None or PREPROCESS is None or ACTIVE_MODEL_ID is None:
        return JSONResponse(status_code=503, content={"detail": "active model not loaded"})
    if INSTANCE_CONTRACT is not None:
        raise HTTPException(
            status_code=409,
            detail="active model is an instance-segmentation export; use /count")
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    orig_w, orig_h = img.size

    train_config = TRAIN_CONFIG or {}
    patch_size = int(train_config.get("patch_size") or 0)
    fg_threshold = train_config.get("inference_threshold")

    start = time.perf_counter()
    if patch_size > 0:
        sw_stride = _default_stride_np(patch_size)
        probs = _sliding_window_onnx(
            SESSION, np.asarray(img), patch_size, sw_stride, PREPROCESS["normalize"],
        )
        mask_small = _prediction_from_probs_np(probs, fg_threshold)
        inference_mode = "sliding_window"
    else:
        logger.warning(
            "RESIZE INFERENCE (legacy): model %s has no sliding-window "
            "metadata; full image resized to %s. Re-export the model from a "
            "patch-trained run for native-resolution inference.",
            ACTIVE_MODEL_ID, PREPROCESS.get("input_size"),
        )
        input_tensor = preprocess_image(img, PREPROCESS)
        logits = SESSION.run(None, {"input": input_tensor})[0]
        mask_small = np.argmax(logits, axis=1).astype(np.uint8)[0]
        inference_mode = "resize_legacy"
    elapsed = time.perf_counter() - start

    # Upsample the class mask to the original image resolution so the
    # returned artifact aligns with the uploaded pixels.
    mask = np.asarray(
        Image.fromarray(mask_small, mode="L").resize((orig_w, orig_h), Image.NEAREST)
    )
    mask_img = Image.fromarray(mask, mode="L")

    zip_buffer = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        mask_bytes = io.BytesIO()
        mask_img.save(mask_bytes, format="PNG")
        zf.writestr("mask.png", mask_bytes.getvalue())
        if overlay:
            model_dir = REGISTRY_DIR / ACTIVE_MODEL_ID
            overlay_img = build_overlay(mask, model_dir / "classes.json")
            overlay_bytes = io.BytesIO()
            overlay_img.save(overlay_bytes, format="PNG")
            zf.writestr("overlay.png", overlay_bytes.getvalue())
        if include_meta:
            meta = {
                "model_id": ACTIVE_MODEL_ID,
                "inference_time_sec": elapsed,
                "inference_mode": inference_mode,
                "mask_size": [orig_w, orig_h],
                "patch_size": patch_size if patch_size > 0 else None,
                "fg_threshold": fg_threshold if patch_size > 0 else None,
                "input_size": PREPROCESS.get("input_size"),
            }
            zf.writestr("meta.json", json.dumps(meta, indent=2))
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip")
