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
    global SESSION, PREPROCESS, TRAIN_CONFIG, ACTIVE_MODEL_ID
    model_id = get_active_model_id()
    if model_id is None:
        SESSION = None
        PREPROCESS = None
        TRAIN_CONFIG = None
        ACTIVE_MODEL_ID = None
        return
    model_dir = REGISTRY_DIR / model_id
    model_path = model_dir / "model.onnx"
    preprocess_path = model_dir / "preprocess.json"
    if not model_path.exists() or not preprocess_path.exists():
        SESSION = None
        PREPROCESS = None
        TRAIN_CONFIG = None
        ACTIVE_MODEL_ID = None
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
    count = np.maximum(count, 1.0)
    avg_probs = (accum / count).astype(np.float32)

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
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    orig_w, orig_h = img.size

    train_config = TRAIN_CONFIG or {}
    patch_size = int(train_config.get("patch_size") or 0)
    fg_threshold = train_config.get("inference_threshold")

    start = time.time()
    if patch_size > 0:
        sw_stride = max(1, patch_size * 3 // 4)
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
    elapsed = time.time() - start

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
