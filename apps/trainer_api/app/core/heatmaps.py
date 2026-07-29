# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Heatmap PNG generation from cached prediction artifacts.

Extracted verbatim from prediction_engine.py during the pre-OSS refactor
(one added line: a function-scope import of _prediction_artifact_paths to
avoid a module-level cycle with prediction_engine). Never triggers
inference — renders confidence / class / error heatmaps from artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import HTTPException

from .annotate_index import find_annotate_image
from .paths import annotate_masks_dir
from .security import _sanitize_filename


def _heatmap_cache_path(
    pred_dir: Path,
    item_id: str,
    heatmap_type: str,
    class_id: int,
    threshold: float = 0.0,
    min_area: int = 0,
    max_area: int = 0,
) -> Path:
    # Encode threshold/min/max so each slider position has its own cache and
    # the legacy (no-filter) PNG keeps its original filename. The `v2` tag
    # invalidates cached files produced by the earlier buggy binarization
    # (1/255 cutoff, then `argmax > 0` for class heatmaps); bump again on
    # any change that affects pixel output for the same (threshold, min_area,
    # max_area) tuple.
    parts = []
    if threshold > 0.0:
        parts.append(f"t{int(round(threshold * 100)):03d}")
    if min_area > 0:
        parts.append(f"mn{min_area}")
    if max_area > 0:
        parts.append(f"mx{max_area}")
    if parts:
        parts.append("v2")
    suffix = ("_" + "_".join(parts)) if parts else ""
    if heatmap_type == "class":
        return pred_dir / f"{item_id}.heatmap_{heatmap_type}_{class_id}{suffix}.png"
    return pred_dir / f"{item_id}.heatmap_{heatmap_type}{suffix}.png"


def generate_heatmap(
    project_id: str,
    run_path: Path,
    model_path: Path,
    item_id: str,
    backend: str,
    tta: bool,
    heatmap_type: str,
    class_id: int = 0,
    threshold: float = 0.0,
    min_area: int = 0,
    max_area: int = 0,
) -> bytes:
    """Generate a colored heatmap PNG. Returns RGBA PNG bytes.

    heatmap_type: "confidence" | "class" | "error"

    For "confidence" and "class" heatmaps:
    - `threshold` (0..1) — pixels below the cutoff render as the colormap's
      cold end so the slider reveals only confident areas.
    - `min_area` / `max_area` (pixels in full resolution) — connected
      components of the above-threshold mask whose area falls outside the
      range are dropped from the heatmap.

    Default (all three at 0) preserves the legacy full-coverage heatmap.
    """
    import io

    import cv2
    from PIL import Image

    from .colormap import apply_colormap, make_error_map
    from .prediction_engine import _prediction_artifact_paths

    item_id = _sanitize_filename(item_id)  # neutralize ..\ / ../ path traversal in the route param

    # Only use existing artifacts — do NOT trigger inference
    pred_path, _conf_path, _score_path = _prediction_artifact_paths(run_path, backend, item_id, tta=tta)
    pred_dir = pred_path.parent

    # Check for cached heatmap PNG — invalidated when source artifact changes
    cache_path = _heatmap_cache_path(
        pred_dir, item_id, heatmap_type, class_id, threshold, min_area, max_area,
    )
    if cache_path.exists():
        # Support both new .npy and old .npz formats
        source_npy = pred_dir / f"{item_id}.probs.npy"
        source_npz = pred_dir / f"{item_id}.probs.npz"
        source_path = source_npy if source_npy.exists() else (source_npz if heatmap_type in ("confidence", "class") else pred_path)
        if source_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime:
            return cache_path.read_bytes()

    orig_img_path = find_annotate_image(project_id, item_id)
    if orig_img_path is None:
        raise HTTPException(status_code=404, detail="image not found")
    with Image.open(orig_img_path) as img:
        orig_w, orig_h = img.size

    if heatmap_type in ("confidence", "class"):
        # Support both new .npy and old .npz formats
        probs_path_npy = pred_dir / f"{item_id}.probs.npy"
        probs_path_npz = pred_dir / f"{item_id}.probs.npz"
        if probs_path_npy.exists():
            probs = np.load(probs_path_npy).astype("float32")  # (C, H, W)
        elif probs_path_npz.exists():
            probs = np.load(probs_path_npz)["probs"].astype("float32")  # (C, H, W)
        else:
            raise HTTPException(status_code=404, detail="prediction not found; run prediction first")

    # Pre-compute the model's argmax label map once at full resolution.
    # Used as the binarization basis for the area filter when no explicit
    # confidence threshold is given so the size cutoff lines up with what
    # the model actually predicts. NEAREST upsample preserves the discrete
    # class labels.
    argmax_full: np.ndarray | None = None
    if heatmap_type in ("confidence", "class") and (min_area > 0 or max_area > 0):
        argmax_small = np.argmax(probs, axis=0).astype("uint8")
        argmax_full = cv2.resize(argmax_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def _apply_area_filter(prob_full: np.ndarray, thr: float, target_class: int | None) -> np.ndarray:
        """Zero out connected components whose area is outside [min_area, max_area].

        For the "class" heatmap pass `target_class` so the binarization picks
        pixels where this class is the argmax — otherwise a small blob that's
        merely "the second-best class" would survive the filter even though
        the model didn't actually predict it as that class.
        """
        if min_area <= 0 and max_area <= 0:
            return prob_full
        if thr > 0.0:
            # User picked an explicit confidence cutoff — binarize there.
            binary = (prob_full >= thr).astype("uint8")
        else:
            assert argmax_full is not None  # set above when min/max area > 0
            if target_class is not None:
                binary = (argmax_full == target_class).astype("uint8")
            else:
                binary = (argmax_full > 0).astype("uint8")
        from segcore.postprocess import apply_min_size_filter
        kept = apply_min_size_filter(binary, min_area, max_area)
        return np.where(kept > 0, prob_full, 0.0)

    if heatmap_type == "confidence":
        max_conf = np.max(probs, axis=0)  # (H, W) float32 [0,1]
        max_conf_full = cv2.resize(max_conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Below-threshold pixels clamp to 0 (the cold end of the colormap, e.g.
        # the dark blue of turbo) so the user still sees the heatmap fill but
        # only above-threshold pixels rise out as warm colors.
        if threshold > 0.0:
            max_conf_full = np.where(max_conf_full < threshold, 0.0, max_conf_full)
        max_conf_full = _apply_area_filter(max_conf_full, threshold, target_class=None)
        gray_u8 = np.clip(max_conf_full * 255, 0, 255).astype("uint8")
        rgba = apply_colormap(gray_u8)

    elif heatmap_type == "class":
        if class_id < 0 or class_id >= probs.shape[0]:
            raise HTTPException(status_code=400, detail=f"class_id {class_id} out of range")
        class_prob = probs[class_id]  # (H, W) float32 [0,1]
        class_prob_full = cv2.resize(class_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if threshold > 0.0:
            class_prob_full = np.where(class_prob_full < threshold, 0.0, class_prob_full)
        class_prob_full = _apply_area_filter(class_prob_full, threshold, target_class=class_id)
        gray_u8 = np.clip(class_prob_full * 255, 0, 255).astype("uint8")
        rgba = apply_colormap(gray_u8)

    elif heatmap_type == "error":
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="prediction not found; run prediction first")
        pred_np = np.asarray(Image.open(pred_path).convert("L"))
        # Load GT mask
        gt_path = annotate_masks_dir(project_id) / f"{item_id}.png"
        if not gt_path.exists():
            raise HTTPException(status_code=404, detail="ground truth mask not found")
        gt_img = Image.open(gt_path).convert("L")
        gt_resized = gt_img.resize((pred_np.shape[1], pred_np.shape[0]), resample=Image.NEAREST)
        gt_np = np.asarray(gt_resized)
        rgba = make_error_map(pred_np, gt_np)

    else:
        raise HTTPException(status_code=400, detail=f"unknown heatmap type: {heatmap_type}")

    result_img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    try:
        cache_path.write_bytes(png_bytes)
    except OSError:
        pass

    return png_bytes
