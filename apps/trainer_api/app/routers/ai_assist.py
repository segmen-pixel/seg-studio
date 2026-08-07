# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from segcore.image_io import imread as _imread

from ..core.annotate_index import find_annotate_image
from ..core.exceptions import (
    ImageNotFoundError,  # noqa: F401
    RFAssistError,
    SAMInferenceError,
    SAMLabelAssistError,
    SAMModelMissingError,
    SuperpixelError,
)
from ..core.paths import annotate_masks_dir
from ..core.recipe_engine import run_auto_label
from ..core.rf_assist import encode_png_base64, rf_predict, rf_train
from ..core.sam_assist import sam_predict
from ..core.sam_label_assist import sla_predict, sla_train
from ..core.superpixel import compute_superpixels, encode_boundaries_png, encode_segment_map_png

GRABCUT_MAX_SIDE = 512

_SAM_CHECKPOINTS = {
    "mobile_sam": "mobile_sam.pt",
    "sam2_tiny": "sam2.1_hiera_tiny.pt",
    "sam2_small": "sam2.1_hiera_small.pt",
    "tinysam": "tinysam.pth",
    "efficient_sam_ti": "efficient_sam_vitt.pt",
}
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "models" / "sam_checkpoints"

router = APIRouter()


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/auto_label")
async def auto_label(project_id: str, item_id: str, request: Request):
    body = await request.json()
    class_id = body.get("class_id", 1)
    if not isinstance(class_id, int) or class_id < 1:
        raise HTTPException(status_code=400, detail="class_id must be a positive integer")
    erode_pct = float(body.get("erode_pct", 5.0))
    iterations = int(body.get("iterations", 3))

    img_path = find_annotate_image(project_id, item_id)
    if img_path is None:
        raise HTTPException(status_code=404, detail="image not found")

    mask_file = annotate_masks_dir(project_id) / f"{item_id}.png"
    mask_path = str(mask_file) if mask_file.exists() else None

    loop = asyncio.get_running_loop()
    try:
        png_data = await loop.run_in_executor(
            None, run_auto_label, project_id, item_id, str(img_path),
            mask_path, class_id, erode_pct, iterations
        )
    except ValueError as exc:
        # The engine raises ValueError with an actionable explanation (too few
        # annotations, no matching region, ...) — pass it through instead of a
        # generic string the user cannot act on.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png_data, media_type="image/png")


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/rf-assist")
def rf_assist_endpoint(project_id: str, item_id: str):
    """Train RF pixel classifier from annotations and predict for one image."""
    try:
        t0 = time.perf_counter()
        entry = rf_train(project_id)
        t_train = time.perf_counter() - t0

        img_path = find_annotate_image(project_id, item_id)
        if not img_path:
            raise HTTPException(status_code=404, detail="image not found")
        image = _imread(str(img_path))
        if image is None:
            raise HTTPException(status_code=500, detail="failed to read image")

        t1 = time.perf_counter()
        mask, confidence = rf_predict(image, entry, img_path=str(img_path))
        t_pred = time.perf_counter() - t1
        logging.getLogger(__name__).debug("RF Assist train=%.0fms predict=%.0fms total=%.0fms img=%dx%d", t_train*1000, t_pred*1000, (t_train+t_pred)*1000, image.shape[1], image.shape[0])

        return {
            "mask": encode_png_base64(mask),
            "confidence": encode_png_base64(confidence),
            "train_time_ms": int(t_train * 1000),
            "predict_time_ms": int(t_pred * 1000),
            "features_used": entry.get("features_used", "handcraft"),
        }
    except (HTTPException, RFAssistError):
        raise
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logging.getLogger(__name__).error("RF Assist failed: %s\n%s", exc, tb)
        raise RFAssistError(
            detail=f"{exc}\n{tb[-500:]}",
            context={"project_id": project_id, "item_id": item_id},
        ) from exc


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/sam-label-assist")
async def sam_label_assist_endpoint(project_id: str, item_id: str, request: Request):
    """Train SAM encoder feature heads from annotations and predict for one image."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    model_name = body.get("model", "mobile_sam") if body else "mobile_sam"

    loop = asyncio.get_running_loop()
    try:
        t0 = time.perf_counter()
        entry = await loop.run_in_executor(None, sla_train, project_id, model_name)
        t_train = time.perf_counter() - t0

        img_path = find_annotate_image(project_id, item_id)
        if not img_path:
            raise HTTPException(status_code=404, detail="image not found")
        image = _imread(str(img_path))
        if image is None:
            raise HTTPException(status_code=500, detail="failed to read image")

        t1 = time.perf_counter()
        mask, confidence = await loop.run_in_executor(
            None, sla_predict, image, entry, str(img_path))
        t_pred = time.perf_counter() - t1
        logging.getLogger(__name__).debug("SLA train=%.0fms predict=%.0fms model=%s img=%dx%d", t_train*1000, t_pred*1000, model_name, image.shape[1], image.shape[0])

        return {
            "mask": encode_png_base64(mask),
            "confidence": encode_png_base64(confidence),
            "train_time_ms": int(t_train * 1000),
            "predict_time_ms": int(t_pred * 1000),
        }
    except (HTTPException, SAMLabelAssistError):
        raise
    except Exception as exc:
        raise SAMLabelAssistError(
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id},
        ) from exc


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/grabcut")
async def grabcut_segment(project_id: str, item_id: str, request: Request):
    return await auto_label(project_id, item_id, request)


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/color_assist")
async def color_assist_segment(project_id: str, item_id: str, request: Request):
    return await auto_label(project_id, item_id, request)


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/sam-segment")
async def sam_segment(project_id: str, item_id: str, request: Request):
    """Run SAM click segmentation. Returns mask as base64 PNG + score."""
    body = await request.json()
    points = body.get("points", None)
    labels = body.get("labels", None)
    box = body.get("box", None)
    model_name = body.get("model", "mobile_sam")

    has_points = points and labels and len(points) == len(labels)
    has_box = box and len(box) == 4
    if not has_points and not has_box:
        raise HTTPException(status_code=400, detail="points/labels or box required")
    if points and labels and len(points) != len(labels):
        raise HTTPException(status_code=400, detail="points and labels must be same length")
    if model_name not in _SAM_CHECKPOINTS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}. Options: {list(_SAM_CHECKPOINTS.keys())}")

    img_path = find_annotate_image(project_id, item_id)
    if not img_path:
        raise HTTPException(status_code=404, detail="image not found")

    loop = asyncio.get_running_loop()
    try:
        t0 = time.perf_counter()
        mask, score = await loop.run_in_executor(
            None, sam_predict, project_id, item_id, str(img_path),
            points if has_points else None,
            labels if has_points else None,
            box if has_box else None,
            model_name,
        )
        elapsed = time.perf_counter() - t0
        logging.getLogger(__name__).debug("SAM %s predict=%.0fms score=%.3f", model_name, elapsed*1000, score)
    except FileNotFoundError:
        raise SAMModelMissingError(
            detail=f"model={model_name}",
            context={"project_id": project_id, "item_id": item_id},
        )
    except Exception as exc:
        raise SAMInferenceError(
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id, "model": model_name},
        ) from exc

    return {
        "mask": encode_png_base64(mask * 255),
        "score": round(score, 4),
        "predict_time_ms": int((time.perf_counter() - t0) * 1000) if 't0' in dir() else 0,
    }


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/superpixel-map")
async def superpixel_map(project_id: str, item_id: str, request: Request):
    """Compute SLIC superpixel segmentation for an image."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    n_segments = int(body.get("n_segments", 500)) if body else 500

    img_path = find_annotate_image(project_id, item_id)
    if not img_path:
        raise HTTPException(status_code=404, detail="image not found")

    loop = asyncio.get_running_loop()
    try:
        t0 = time.perf_counter()
        image = _imread(str(img_path))
        if image is None:
            raise HTTPException(status_code=500, detail="failed to read image")
        segments = await loop.run_in_executor(
            None, compute_superpixels, image, n_segments, 20.0, str(img_path))
        segments_b64 = await loop.run_in_executor(None, encode_segment_map_png, segments)
        boundaries_b64 = await loop.run_in_executor(None, encode_boundaries_png, segments)
        elapsed = time.perf_counter() - t0
        actual_n = int(segments.max()) + 1
        logging.getLogger(__name__).debug("Superpixel %d segments, %.0fms, img=%dx%d", actual_n, elapsed*1000, image.shape[1], image.shape[0])
        return {
            "segments_b64": segments_b64,
            "boundaries_b64": boundaries_b64,
            "n_segments": actual_n,
            "time_ms": int(elapsed * 1000),
        }
    except (HTTPException, SuperpixelError):
        raise
    except Exception as exc:
        raise SuperpixelError(
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id},
        ) from exc



@router.post("/projects/{project_id}/datasets/annotate/{item_id}/crack-trace")
async def crack_trace(project_id: str, item_id: str, request: Request):
    """Compute crack trace map using Meijering neuriteness filter."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    sensitivity = max(1, min(100, int(body.get("sensitivity", 25)))) if body else 25
    width_px = max(0, min(20, int(body.get("width_px", 0)))) if body else 0

    img_path = find_annotate_image(project_id, item_id)
    if not img_path:
        raise HTTPException(status_code=404, detail="image not found")

    loop = asyncio.get_running_loop()
    try:
        from ..core.crack_trace import crack_trace_compute
        result = await loop.run_in_executor(
            None, crack_trace_compute, str(img_path), sensitivity, width_px)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        from ..core.exceptions import AppError
        raise AppError(
            "Crack trace computation failed.",
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id},
        ) from exc


@router.post("/projects/{project_id}/datasets/annotate/{item_id}/crack-trace/adaptive")
async def crack_trace_adaptive_endpoint(project_id: str, item_id: str, request: Request):
    """Adaptive crack detection seeded by a click point.

    Uses the Meijering response at the click location to derive a local
    threshold and returns the connected crack region passing through it.
    """
    body = await request.json()
    click_x = int(body.get("click_x", 0))
    click_y = int(body.get("click_y", 0))
    sensitivity = max(1, min(100, int(body.get("sensitivity", 25))))
    width_px = max(0, min(20, int(body.get("width_px", 0))))

    img_path = find_annotate_image(project_id, item_id)
    if not img_path:
        raise HTTPException(status_code=404, detail="image not found")

    loop = asyncio.get_running_loop()
    try:
        from ..core.crack_trace import crack_trace_adaptive
        result = await loop.run_in_executor(
            None, crack_trace_adaptive, str(img_path), click_x, click_y, sensitivity, width_px)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        from ..core.exceptions import AppError
        raise AppError(
            "Adaptive crack trace failed.",
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id},
        ) from exc


@router.get("/sam/models")
def sam_list_models():
    """List available SAM models and their status."""
    from ..core.sam_assist import _SAM_DOWNLOAD_URLS, _SAM_MODELS
    result = []
    for name, ckpt_file in _SAM_CHECKPOINTS.items():
        ckpt_path = _MODELS_DIR / ckpt_file
        exists = ckpt_path.exists()
        auto_dl = ckpt_file in _SAM_DOWNLOAD_URLS
        loaded_keys = [k for k in _SAM_MODELS.keys() if k.startswith(f"{name}:")]
        result.append({
            "id": name,
            "checkpoint_exists": exists or auto_dl,  # available if exists or can auto-download
            "downloaded": exists,
            "auto_download": auto_dl,
            "loaded": len(loaded_keys) > 0,
            "checkpoint_file": ckpt_file,
        })
    return result
