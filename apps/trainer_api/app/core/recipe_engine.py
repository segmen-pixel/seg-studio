# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from segcore.image_io import imread as _imread

from .annotate_index import load_annotate_index
from .paths import annotate_images_dir, annotate_masks_dir

GRABCUT_MAX_SIDE = 512


def _filter_small_components(fg_mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    """Remove connected components smaller than min_area_ratio * largest area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )
    if num_labels <= 1:
        return fg_mask  # background only or single component
    # stats column CV2.CC_STAT_AREA; label 0 = background
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    min_area = max(max_area * min_area_ratio, 1)
    result = np.zeros_like(fg_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            result[labels == label_id] = 1
    return result


_RECIPE_STEP_TYPES = {"hsv_range", "lab_range", "morphology", "area_filter"}
_MORPH_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "dilate": cv2.MORPH_DILATE,
    "erode": cv2.MORPH_ERODE,
}


def _validate_recipe(data: dict) -> list[str]:
    """Return list of validation error strings (empty = ok)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["root must be object"]
    if data.get("version") != 1:
        errors.append("unsupported version (expected 1)")
    rules = data.get("rules")
    if not isinstance(rules, list) or len(rules) == 0:
        errors.append("rules must be non-empty array")
        return errors
    for ri, rule in enumerate(rules):
        cid = rule.get("class_id")
        if not isinstance(cid, int) or cid < 1:
            errors.append(f"rules[{ri}].class_id must be positive int")
        steps = rule.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            errors.append(f"rules[{ri}].steps must be non-empty array")
            continue
        for si, step in enumerate(steps):
            stype = step.get("type")
            if stype not in _RECIPE_STEP_TYPES:
                errors.append(f"rules[{ri}].steps[{si}].type unknown: {stype}")
                continue
            params = step.get("params", {})
            if stype in ("hsv_range", "lab_range"):
                channels = (
                    [("h", 0, 179), ("s", 0, 255), ("v", 0, 255)]
                    if stype == "hsv_range"
                    else [("l", 0, 255), ("a", 0, 255), ("b", 0, 255)]
                )
                for ch, lo, hi in channels:
                    mn = params.get(f"{ch}_min")
                    mx = params.get(f"{ch}_max")
                    if mn is None or mx is None:
                        errors.append(f"rules[{ri}].steps[{si}] missing {ch}_min/{ch}_max")
            elif stype == "morphology":
                op = params.get("operation")
                if op not in _MORPH_OPS:
                    errors.append(f"rules[{ri}].steps[{si}] invalid operation: {op}")
                ks = params.get("kernel_size", 3)
                if not isinstance(ks, int) or ks < 1 or ks % 2 == 0:
                    errors.append(f"rules[{ri}].steps[{si}] kernel_size must be odd positive int")
            elif stype == "area_filter":
                if "min_area_px" not in params and "min_area_ratio" not in params:
                    errors.append(f"rules[{ri}].steps[{si}] needs min_area_px or min_area_ratio")
    return errors


def _apply_recipe_to_image(img_bgr: np.ndarray, recipe: dict) -> np.ndarray:
    """Apply recipe rules to a BGR image. Returns class-index mask (H, W) uint8."""
    h, w = img_bgr.shape[:2]
    result = np.zeros((h, w), dtype=np.uint8)
    hsv = None
    lab = None

    for rule in recipe.get("rules", []):
        class_id = rule["class_id"]
        mask: np.ndarray | None = None

        for step in rule.get("steps", []):
            stype = step["type"]
            params = step.get("params", {})
            combine = step.get("combine", "or")

            if stype == "hsv_range":
                if hsv is None:
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                lower = np.array([params["h_min"], params["s_min"], params["v_min"]], dtype=np.uint8)
                upper = np.array([params["h_max"], params["s_max"], params["v_max"]], dtype=np.uint8)
                if params["h_min"] <= params["h_max"]:
                    step_mask = cv2.inRange(hsv, lower, upper)
                else:
                    # Wrap-around hue
                    m1 = cv2.inRange(hsv, np.array([params["h_min"], params["s_min"], params["v_min"]], dtype=np.uint8),
                                     np.array([179, params["s_max"], params["v_max"]], dtype=np.uint8))
                    m2 = cv2.inRange(hsv, np.array([0, params["s_min"], params["v_min"]], dtype=np.uint8),
                                     np.array([params["h_max"], params["s_max"], params["v_max"]], dtype=np.uint8))
                    step_mask = cv2.bitwise_or(m1, m2)
                if mask is None:
                    mask = step_mask
                elif combine == "and":
                    mask = cv2.bitwise_and(mask, step_mask)
                elif combine == "subtract":
                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(step_mask))
                else:
                    mask = cv2.bitwise_or(mask, step_mask)

            elif stype == "lab_range":
                if lab is None:
                    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
                lower = np.array([params["l_min"], params["a_min"], params["b_min"]], dtype=np.uint8)
                upper = np.array([params["l_max"], params["a_max"], params["b_max"]], dtype=np.uint8)
                step_mask = cv2.inRange(lab, lower, upper)
                if mask is None:
                    mask = step_mask
                elif combine == "and":
                    mask = cv2.bitwise_and(mask, step_mask)
                elif combine == "subtract":
                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(step_mask))
                else:
                    mask = cv2.bitwise_or(mask, step_mask)

            elif stype == "morphology":
                if mask is None:
                    continue
                op = _MORPH_OPS[params["operation"]]
                ks = params.get("kernel_size", 3)
                iters = params.get("iterations", 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
                mask = cv2.morphologyEx(mask, op, kernel, iterations=iters)

            elif stype == "area_filter":
                if mask is None:
                    continue
                binary = (mask > 0).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                min_px = params.get("min_area_px", 0)
                if min_px == 0 and "min_area_ratio" in params:
                    min_px = int(params["min_area_ratio"] * h * w)
                cleaned = np.zeros_like(binary)
                for label_id in range(1, num_labels):
                    if stats[label_id, cv2.CC_STAT_AREA] >= min_px:
                        cleaned[labels == label_id] = 255
                mask = cleaned

        if mask is not None:
            # First-wins: only write where result is still 0 (background)
            result[(mask > 0) & (result == 0)] = class_id

    return result


def _collect_shape_descriptors(project_id: str, class_id: int) -> list[dict]:
    """Collect shape descriptors (Hu moments, circularity, solidity) from annotated objects."""
    masks_dir = annotate_masks_dir(project_id)
    index = load_annotate_index(project_id)
    descriptors: list[dict] = []

    for item in index.get("items", []):
        iid = item.get("id", "")
        mask_path = masks_dir / f"{iid}.png"
        if not mask_path.exists():
            continue
        ann = _imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if ann is None or not np.any(ann == class_id):
            continue
        fg = (ann == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            perimeter = cv2.arcLength(cnt, True)
            hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
            circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 1
            descriptors.append({
                "hu": hu,
                "circularity": circularity,
                "solidity": solidity,
                "aspect": aspect,
                "area": area,
            })
    return descriptors


def _filter_by_shape(fg_mask: np.ndarray, ref_shapes: list[dict],
                     hu_threshold: float = 0.4) -> np.ndarray:
    """Remove connected components whose shape doesn't match any reference shape."""
    if not ref_shapes:
        return fg_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels <= 1:
        return fg_mask
    result = np.zeros_like(fg_mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        component = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = contours[0]
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        # Match against any reference shape
        matched = False
        for ref in ref_shapes:
            # Direct Hu moment comparison (log-scale, lower = more similar)
            hu_dist = 0.0
            for k in range(7):
                a = -np.sign(hu[k]) * np.log10(abs(hu[k]) + 1e-30)
                b = -np.sign(ref["hu"][k]) * np.log10(abs(ref["hu"][k]) + 1e-30)
                hu_dist += abs(a - b)
            hu_dist /= 7.0
            # Circularity and solidity similarity
            circ_diff = abs(circularity - ref["circularity"])
            solid_diff = abs(solidity - ref["solidity"])
            if hu_dist < hu_threshold and circ_diff < 0.35 and solid_diff < 0.35:
                matched = True
                break
        if matched:
            result[labels == label_id] = 1
    return result


def _collect_labeled_colors(project_id: str, target_item_id: str, class_id: int,
                            erode_pct: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect cleaned foreground and background HSV pixels from ALL annotated images."""
    masks_dir = annotate_masks_dir(project_id)
    images_dir = annotate_images_dir(project_id)
    index = load_annotate_index(project_id)
    fg_pixels_hsv: list[np.ndarray] = []
    bg_pixels_hsv: list[np.ndarray] = []

    for item in index.get("items", []):
        iid = item.get("id", "")
        mask_path = masks_dir / f"{iid}.png"
        if not mask_path.exists():
            continue
        ann = _imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if ann is None or not np.any(ann == class_id):
            continue
        src_path = None
        fname = item.get("filename")
        if fname:
            candidate = images_dir / fname
            if candidate.exists():
                src_path = candidate
        if src_path is None:
            continue
        src_bgr = _imread(str(src_path))
        if src_bgr is None:
            continue
        if src_bgr.shape[:2] != ann.shape[:2]:
            ann = cv2.resize(ann, (src_bgr.shape[1], src_bgr.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
        src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        fg_bin = (ann == class_id).astype(np.uint8)
        fg_lab = src_lab[fg_bin > 0]
        if len(fg_lab) >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, lbls, centers = cv2.kmeans(fg_lab, 2, None, criteria, 3,
                                          cv2.KMEANS_PP_CENTERS)
            lbls = lbls.flatten()
            fg_label = 0 if np.sum(lbls == 0) >= np.sum(lbls == 1) else 1
            dominant_center = centers[fg_label]
            dists = np.sqrt(((fg_lab[lbls == fg_label] - dominant_center) ** 2).sum(axis=1))
            if erode_pct > 0 and len(dists) > 0:
                cutoff_pct = min(erode_pct * 5, 95)
                thresh = np.percentile(dists, 100 - cutoff_pct)
            else:
                thresh = float("inf")
            keep = (lbls == fg_label)
            full_dists = np.sqrt(((fg_lab - dominant_center) ** 2).sum(axis=1))
            keep = keep & (full_dists <= thresh)
            fg_hsv = src_hsv[fg_bin > 0][keep]
            if len(fg_hsv) > 0:
                fg_pixels_hsv.append(fg_hsv)
        else:
            fg_pixels_hsv.append(src_hsv[fg_bin > 0])
        # Background pixels (class 0)
        bg_hsv = src_hsv[ann == 0]
        if len(bg_hsv) > 0:
            bg_pixels_hsv.append(bg_hsv)

    return fg_pixels_hsv, bg_pixels_hsv


def _run_auto_label(project_id: str, item_id: str, img_path: str,
                    mask_path: str | None, class_id: int,
                    erode_pct: float = 5.0, iterations: int = 3) -> bytes:
    """Unified auto-label: spatial seeds from current image + color seeds from all annotated images."""
    img_bgr = _imread(img_path)
    if img_bgr is None:
        raise ValueError("failed to read image")

    orig_h, orig_w = img_bgr.shape[:2]

    ann_mask: np.ndarray | None = None
    if mask_path and Path(mask_path).exists():
        raw = _imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw is not None and raw.shape[:2] == (orig_h, orig_w):
            ann_mask = raw
    has_spatial = ann_mask is not None and np.any(ann_mask == class_id)

    fg_pixels_hsv, bg_pixels_hsv = _collect_labeled_colors(
        project_id, item_id, class_id, erode_pct)
    has_color = len(fg_pixels_hsv) > 0
    ref_shapes = _collect_shape_descriptors(project_id, class_id)

    if not has_spatial and not has_color:
        raise ValueError("No annotations found. Paint foreground on at least one image first.")

    prob_map: np.ndarray | None = None
    if has_color:
        rng = np.random.default_rng(42)
        fg_all = np.concatenate(fg_pixels_hsv, axis=0)
        if len(fg_all) > 50000:
            fg_all = fg_all[rng.choice(len(fg_all), 50000, replace=False)]
        h_bins, s_bins = 32, 32
        fg_hist = cv2.calcHist([fg_all.reshape(-1, 1, 3)], [0, 1], None,
                               [h_bins, s_bins], [0, 180, 0, 256])
        if len(bg_pixels_hsv) > 0:
            bg_all = np.concatenate(bg_pixels_hsv, axis=0)
            if len(bg_all) > 50000:
                bg_all = bg_all[rng.choice(len(bg_all), 50000, replace=False)]
            bg_hist = cv2.calcHist([bg_all.reshape(-1, 1, 3)], [0, 1], None,
                                   [h_bins, s_bins], [0, 180, 0, 256])
            ratio_hist = fg_hist / (fg_hist + bg_hist + 1e-6)
            cv2.normalize(ratio_hist, ratio_hist, 0, 255, cv2.NORM_MINMAX)
            ratio_hist = ratio_hist.astype(np.float32)
        else:
            ratio_hist = fg_hist.copy()
            cv2.normalize(ratio_hist, ratio_hist, 0, 255, cv2.NORM_MINMAX)
        target_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        prob_map = cv2.calcBackProject([target_hsv], [0, 1], ratio_hist,
                                       [0, 180, 0, 256], 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prob_map = cv2.filter2D(prob_map, -1, kernel)
        prob_map = cv2.filter2D(prob_map, -1, kernel)

    long_side = max(orig_h, orig_w)
    if long_side > GRABCUT_MAX_SIDE:
        ratio = GRABCUT_MAX_SIDE / long_side
        work_w = max(2, int(orig_w * ratio))
        work_h = max(2, int(orig_h * ratio))
        work_img = cv2.resize(img_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        ann_small = cv2.resize(ann_mask, (work_w, work_h),
                               interpolation=cv2.INTER_NEAREST) if ann_mask is not None else None
        prob_small = cv2.resize(prob_map, (work_w, work_h),
                                interpolation=cv2.INTER_LINEAR) if prob_map is not None else None
    else:
        ratio = 1.0
        work_w, work_h = orig_w, orig_h
        work_img = img_bgr
        ann_small = ann_mask
        prob_small = prob_map

    grab_mask = np.full((work_h, work_w), cv2.GC_PR_BGD, dtype=np.uint8)

    if prob_small is not None and not has_spatial:
        grab_mask[prob_small > 80] = cv2.GC_PR_FGD
        grab_mask[prob_small < 20] = cv2.GC_BGD

    if ann_small is not None and has_spatial:
        fg_binary = (ann_small == class_id).astype(np.uint8)
        work_lab = cv2.cvtColor(work_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        fg_coords = np.argwhere(fg_binary > 0)
        fg_colors = work_lab[fg_binary > 0]

        if len(fg_colors) >= 2:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(
                fg_colors, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            labels = labels.flatten()
            fg_label = 0 if np.sum(labels == 0) >= np.sum(labels == 1) else 1
            dominant_center = centers[fg_label]
            dominant_dists = np.sqrt(
                ((fg_colors[labels == fg_label] - dominant_center) ** 2).sum(axis=1))
            if erode_pct > 0 and len(dominant_dists) > 0:
                cutoff_pct = min(erode_pct * 5, 95)
                dist_threshold = np.percentile(dominant_dists, 100 - cutoff_pct)
            else:
                dist_threshold = float("inf")
            fg_clean = np.zeros_like(fg_binary)
            for i, (coord, label, color) in enumerate(zip(fg_coords, labels, fg_colors)):
                if label == fg_label:
                    d = np.sqrt(((color - dominant_center) ** 2).sum())
                    if d <= dist_threshold:
                        fg_clean[coord[0], coord[1]] = 1
        else:
            fg_clean = fg_binary

        # Strong spatial seeds override everything
        grab_mask[fg_clean > 0] = cv2.GC_FGD
        fg_rejected = (fg_binary > 0) & (fg_clean == 0)
        grab_mask[fg_rejected] = cv2.GC_PR_FGD
        # Other classes → definite background
        other_cls = (ann_small != 0) & (ann_small != class_id)
        grab_mask[other_cls] = cv2.GC_BGD

        # 4c. Color seeds limited to neighborhood of spatial marks
        if prob_small is not None:
            # Dilate marked region to create search zone (~25% of image size)
            dilate_r = max(work_h, work_w) // 4
            dilate_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_r * 2 + 1, dilate_r * 2 + 1))
            search_zone = cv2.dilate(fg_binary, dilate_kern, iterations=1)
            # Only add color PR_FGD within search zone, and don't overwrite strong seeds
            color_candidate = (prob_small > 120) & (search_zone > 0) & (grab_mask == cv2.GC_PR_BGD)
            grab_mask[color_candidate] = cv2.GC_PR_FGD
    elif prob_small is not None:
        # Color-only mode: promote high-probability to definite FGD
        grab_mask[prob_small > 160] = cv2.GC_FGD

    # ---- 5. Edge barrier (spatial mode only) ----
    if has_spatial:
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_zone = (edge_dilated > 0) & (grab_mask == cv2.GC_PR_FGD)
        grab_mask[edge_zone] = cv2.GC_PR_BGD

    # Image border → definite background
    margin = max(1, min(work_h, work_w) // 40)
    for edge_strip in [grab_mask[:margin, :], grab_mask[-margin:, :],
                       grab_mask[:, :margin], grab_mask[:, -margin:]]:
        edge_strip[edge_strip == cv2.GC_PR_BGD] = cv2.GC_BGD

    # Fallback: need at least some FGD seeds
    if not np.any((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD)):
        if ann_small is not None and has_spatial:
            grab_mask[(ann_small == class_id)] = cv2.GC_FGD
        else:
            raise ValueError("Color matching found no similar regions in this image.")

    # ---- 6. GrabCut ----
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    iters = max(1, min(iterations, 10))
    cv2.grabCut(work_img, grab_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)

    # ---- 7. Post-process ----
    fg_small = ((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg_small = _filter_small_components(fg_small)
    if ref_shapes and has_spatial:
        # Shape filtering only reliable when spatial seeds provide context
        fg_small = _filter_by_shape(fg_small, ref_shapes)
    if ratio < 1.0:
        fg_full = cv2.resize(fg_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        fg_full = fg_small
    fg_full = fg_full * class_id
    _, png_bytes = cv2.imencode(".png", fg_full)
    return png_bytes.tobytes()


# ---------------------------------------------------------------------------
# Public aliases — routers and other callers should use the un-underscored
# names. The underscored variants remain as the canonical definitions so
# in-module references and ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
filter_small_components = _filter_small_components
validate_recipe = _validate_recipe
apply_recipe_to_image = _apply_recipe_to_image
run_auto_label = _run_auto_label

