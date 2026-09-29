# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from segcore.image_io import imread as _imread

from ..core.agent_activity import note as note_agent
from ..core.annotate_index import find_annotate_image
from ..core.exceptions import (
    ImageNotFoundError,  # noqa: F401
    RFAssistError,
    SAMInferenceError,
    SAMLabelAssistError,
    SAMModelMissingError,
    SAMPackageMissingError,
    SuperpixelError,
)
from ..core.mask_write import load_mask_array
from ..core.paths import annotate_masks_dir
from ..core.recipe_engine import run_auto_label
from ..core.rf_assist import encode_png_base64, rf_predict, rf_train
from ..core.sam_assist import sam_predict_levels
from ..core.sam_label_assist import sla_predict, sla_train
from ..core.superpixel import compute_superpixels, encode_boundaries_png, encode_segment_map_png

GRABCUT_MAX_SIDE = 512

#: The range a spot-detect sensitivity sent back is held to. The detector
#: multiplies the spread of its scores by it, so a higher value keeps fewer
#: spots. Every answer carries this range, and the annotator's slider reads it
#: from there rather than keeping a copy of its own.
SPOT_SENSITIVITY_RANGE = (1, 60)

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

    # Positive points only: each one marks the object. Any other label is
    # refused, not dropped or changed, so no caller believes it was used.
    if labels is not None and (not isinstance(labels, list) or any(
            isinstance(v, bool) or v != 1 for v in labels)):
        raise HTTPException(status_code=422,
                            detail="only positive points are accepted: every label must be 1")

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
        levels, default_level = await loop.run_in_executor(
            None, sam_predict_levels, project_id, item_id, str(img_path),
            points if has_points else None,
            labels if has_points else None,
            box if has_box else None,
            model_name,
        )
        elapsed = time.perf_counter() - t0
        logging.getLogger(__name__).debug(
            "SAM %s predict=%.0fms levels=%s", model_name, elapsed*1000,
            ",".join(name for name, _m, _s in levels))
    except FileNotFoundError:
        raise SAMModelMissingError(
            detail=f"model={model_name}",
            context={"project_id": project_id, "item_id": item_id},
        )
    except ImportError as exc:
        # The optional package for this model is absent (e.g. TinySAM is a
        # copied package the installer may have skipped). Not an inference
        # failure: the client should pick another model or install it.
        raise SAMPackageMissingError(
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id, "model": model_name},
        ) from exc
    except Exception as exc:
        raise SAMInferenceError(
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id, "model": model_name},
        ) from exc

    if not levels:
        raise SAMInferenceError(
            detail="the model returned no mask",
            context={"project_id": project_id, "item_id": item_id, "model": model_name},
        )
    # An agent at work on an image keeps the activity feed alive and lets a
    # screen that follows the run move to that image. Where it pointed is not
    # kept: the screen shows results, not the intermediate prompts.
    note_agent(request.headers, action="sam_segment", project_id=project_id, item_id=item_id)
    # Every candidate goes back, not just the winner. Which granularity a click
    # meant is the caller's question, not the model's, and answering it here
    # would cost another round trip for a mask that has already been computed.
    # A caller deciding where something is does not need the mask at sixteen
    # megapixels: SAM upsamples its own 256-pixel logits to the full frame, and
    # three of those go back as PNGs for every click. On a 4608x3456 photograph
    # that is most of the 1.34 s a probe costs, against 12 ms at 512 px. Ask for
    # mask_side and they come back that size; what is finally written can be
    # asked for at full size, or scaled up from what the caller kept.
    mask_side = int(body.get("mask_side") or 0)
    if mask_side > 0:
        import cv2 as _cv2
        h0, w0 = levels[0][1].shape[:2]
        if max(h0, w0) > mask_side:
            f = mask_side / float(max(h0, w0))
            small = (max(1, int(round(w0 * f))), max(1, int(round(h0 * f))))
            levels = [(name, _cv2.resize(mask.astype("uint8"), small,
                                         interpolation=_cv2.INTER_NEAREST).astype(bool), score)
                      for name, mask, score in levels]
    encoded = [
        {
            "level": name,
            "mask": encode_png_base64(mask * 255),
            "score": round(score, 4),
            "area": int(np.count_nonzero(mask)),
        }
        for name, mask, score in levels
    ]
    chosen = next((e for e in encoded if e["level"] == default_level), encoded[-1])
    return {
        # Unchanged for existing callers: still the mask argmax would have picked.
        "mask": chosen["mask"],
        "score": chosen["score"],
        "levels": encoded,
        "default_level": chosen["level"],
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



@router.post("/projects/{project_id}/datasets/annotate/{item_id}/spot-detect")
async def spot_detect(project_id: str, item_id: str, request: Request):
    """Find the specks that look like the one the user painted.

    Body: ``mark_png_b64`` is the painted stroke at full image size, L-mode PNG,
    any non-zero pixel counts as painted; or ``point`` is an ``[x, y]`` to dab
    at, with an optional ``radius``, for callers that point rather than paint; or
    ``mark_points`` is the painted pixels as ``[[x, y], ...]``, which is what the
    annotator sends so the browser need not encode a full-frame PNG.
    ``sensitivity`` is optional: leave it
    out for the first call and the tightest threshold that still recovers the
    example is chosen; send it back for slider moves, which reuse the cached
    score map and so cost a fraction of the first call. It is rounded to a
    whole number and held to ``SPOT_SENSITIVITY_RANGE``, which the answer
    names as ``sensitivity_range``; a higher value keeps fewer spots.

    Or ``like_item_id``: an image the person drew, whose specks of ``class_id``
    are the example, measured against their own mask (detect_like_teacher).
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    mark_b64 = (body or {}).get("mark_png_b64")
    point = (body or {}).get("point")
    mark_points = (body or {}).get("mark_points")
    radius = int((body or {}).get("radius", 4))
    class_id = max(0, min(254, int((body or {}).get("class_id", 1))))
    like = str((body or {}).get("like_item_id") or "")
    if not mark_b64 and not point and not mark_points and not like:
        raise HTTPException(status_code=422, detail="give mark_png_b64, mark_points, point or like_item_id")
    sensitivity = (body or {}).get("sensitivity")
    if sensitivity is not None:
        # Rounded, not truncated: the answer reports the value it used and the
        # slider snaps to it, so what is used must be the nearest whole number.
        try:
            sensitivity = round(float(sensitivity))
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(status_code=422, detail="sensitivity must be a number") from None
        lo, hi = SPOT_SENSITIVITY_RANGE
        sensitivity = max(lo, min(hi, sensitivity))

    img_path = find_annotate_image(project_id, item_id)
    if not img_path:
        raise HTTPException(status_code=404, detail="image not found")

    loop = asyncio.get_running_loop()
    if like:
        # A bare id, as find_annotate_image's own fallback demands: the mask is
        # read from a path built out of it.
        if like in (".", "..") or any(c in like for c in ("/", chr(92), chr(0))):
            raise HTTPException(status_code=422, detail="like_item_id is an image id")
        like_path = find_annotate_image(project_id, like)
        # Read from where the mask lives: a tiled image's is an array with no
        # PNG, and a PNG that is there is an export that can be older than it.
        # Off the event loop: a tiled mask is decompressed whole, and every
        # other request to the API would wait while it was.
        like_arr = (await loop.run_in_executor(None, load_mask_array, project_id, like)
                    if like_path else None)
        if not like_path or like_arr is None:
            raise HTTPException(status_code=404, detail="like_item_id has no image, or no mask")
    try:
        if like:
            from ..core.spot_detect import detect_like_teacher
            result = await loop.run_in_executor(
                None,
                lambda: detect_like_teacher(str(img_path), str(like_path), like_arr,
                                            class_id, sensitivity))
        else:
            from ..core.spot_detect import detect_spots
            result = await loop.run_in_executor(
                None,
                lambda: detect_spots(str(img_path), mark_b64, sensitivity, point, radius,
                                     class_id, mark_points))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        from ..core.exceptions import AppError
        raise AppError(
            "Spot detection failed.",
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id},
        ) from exc
    if isinstance(result, dict):
        result["sensitivity_range"] = list(SPOT_SENSITIVITY_RANGE)
    return result


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
