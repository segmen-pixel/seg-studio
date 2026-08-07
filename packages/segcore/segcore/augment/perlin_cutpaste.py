# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Perlin-noise-based defect CutPaste synthesis.

Generates synthetic training samples by warping real defect crops with a
Perlin vector field and pasting them onto the background of another labeled
image. Useful for very small datasets where data diversity is the bottleneck.

This is a seg-studio-native adaptation of DRAEM-style anomaly synthesis
(Zavrtanik et al., ICCV 2021): we drop the reconstruction branch and the
DTD texture dependency and instead reuse the project's own defect textures
as the paste source — this keeps colors/textures domain-consistent and
avoids 600 MB of external data.

Pipeline for a single synthesized sample:
  1. Pick a random labeled (image, mask) pair as the background host.
  2. Pick K random defect crops from the full defect crop pool.
  3. For each crop: Perlin-warp it, color-jitter, scale, rotate.
  4. Find a paste location in the background that doesn't collide with
     existing defect or other pasted synths.
  5. Alpha-feather the crop onto the background.
  6. Merge the new defect mask into the host mask.
Returns (synth_image_bgr, synth_mask_uint8).
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Perlin noise (pure numpy, no external dep) ──────────────────────


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _gradient(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 8 possible 2D gradients (like classic Perlin)
    vectors = np.array(
        [[1, 1], [-1, 1], [1, -1], [-1, -1],
         [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=np.float32,
    )
    g = vectors[h % 8]  # (..., 2)
    return g[..., 0] * x + g[..., 1] * y


def perlin_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int] = (8, 8),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a 2D Perlin noise map in [-1, 1] with shape *shape*.

    *res* controls the frequency — higher values mean smaller features.
    """
    rng = np.random.RandomState(seed)
    H, W = shape
    res_y, res_x = res
    # Grid coordinates in the gradient space
    lin_y = np.linspace(0, res_y, H, endpoint=False)
    lin_x = np.linspace(0, res_x, W, endpoint=False)
    gx, gy = np.meshgrid(lin_x, lin_y)
    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    # Random permutation for gradient selection
    perm = rng.permutation(256).astype(np.int32)
    perm = np.concatenate([perm, perm])

    def _h(ix, iy):
        return perm[(perm[ix & 255] + iy) & 255]

    fx = gx - x0
    fy = gy - y0
    u = _fade(fx)
    v = _fade(fy)

    n00 = _gradient(_h(x0, y0), fx, fy)
    n10 = _gradient(_h(x1, y0), fx - 1, fy)
    n01 = _gradient(_h(x0, y1), fx, fy - 1)
    n11 = _gradient(_h(x1, y1), fx - 1, fy - 1)

    x_lerp_top = _lerp(n00, n10, u)
    x_lerp_bot = _lerp(n01, n11, u)
    result = _lerp(x_lerp_top, x_lerp_bot, v)
    # Normalize roughly to [-1, 1]
    max_abs = float(np.abs(result).max()) or 1.0
    return (result / max_abs).astype(np.float32)


# ── Perlin warp (non-uniform deformation of a crop) ─────────────────


def perlin_warp(
    image: np.ndarray,
    mask: np.ndarray,
    strength: float = 6.0,
    res: tuple[int, int] = (4, 4),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp *image* and *mask* by a Perlin vector field.

    *strength* is the max displacement in pixels.
    *image* is HxWxC (uint8), *mask* is HxW (uint8). Returns warped copies.
    """
    import cv2

    H, W = image.shape[:2]
    fx_map = perlin_noise_2d((H, W), res=res, seed=seed) * strength
    fy_map = perlin_noise_2d((H, W), res=res, seed=None if seed is None else seed + 1) * strength

    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    map_x = (xx + fx_map).astype(np.float32)
    map_y = (yy + fy_map).astype(np.float32)
    warped_img = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.remap(
        mask, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_img, warped_mask


# ── Defect crop extraction ──────────────────────────────────────────


@dataclass
class DefectCrop:
    """Single defect instance cut out of a source image."""
    image: np.ndarray  # HxWxC, BGR
    mask: np.ndarray   # HxW, uint8, nonzero = defect pixels
    class_id: int      # class label (same for all pixels of this crop)
    source_stem: str   # the source image stem (for traceability)


def extract_defect_crops(
    pairs: Iterable[tuple[Path, Path]],
    min_area: int = 16,
    max_area: int | None = None,
    margin: int = 4,
    max_crops: int = 200,
) -> list[DefectCrop]:
    """Scan (image, mask) pairs and cut out every connected defect region.

    Returns a flat list of DefectCrop objects. Each connected component of
    ``mask > 0`` becomes one crop, padded by *margin* pixels on all sides.
    """
    import cv2

    crops: list[DefectCrop] = []
    for img_p, mask_p in pairs:
        if len(crops) >= max_crops:
            break
        img = _imread_unicode(img_p, cv2.IMREAD_COLOR)
        mask = _imread_unicode(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        fg = ((mask > 0) & (mask != 255)).astype(np.uint8)
        if fg.sum() == 0:
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(img.shape[1], x + w + margin)
            y1 = min(img.shape[0], y + h + margin)
            img_crop = img[y0:y1, x0:x1].copy()
            comp_mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            # Use the actual class id at that component (first pixel)
            cls = int(mask[y + h // 2, x + w // 2]) or 1
            crops.append(DefectCrop(
                image=img_crop,
                mask=comp_mask * cls,
                class_id=cls,
                source_stem=img_p.stem,
            ))
            if len(crops) >= max_crops:
                break
    logger.info("Extracted %d defect crops from %d pairs", len(crops), sum(1 for _ in pairs))
    return crops


# ── Single-sample synthesis ─────────────────────────────────────────


def _imread_unicode(path: Path, flags: int):
    import cv2
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def _imwrite_unicode(path: Path, image: np.ndarray, ext: str = ".png") -> bool:
    import cv2
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


def _color_jitter(img: np.ndarray, strength: float = 0.15, rng: random.Random | None = None) -> np.ndarray:
    r = rng or random
    # Per-channel multiplicative jitter
    b = 1.0 + r.uniform(-strength, strength)
    g = 1.0 + r.uniform(-strength, strength)
    rr = 1.0 + r.uniform(-strength, strength)
    out = img.astype(np.float32) * np.array([b, g, rr], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _paste_with_feather(
    background: np.ndarray,
    crop_img: np.ndarray,
    crop_mask: np.ndarray,
    top: int,
    left: int,
    feather: int = 2,
) -> None:
    """In-place alpha-blend *crop_img* into *background* at (top, left).

    Only pixels where crop_mask>0 are blended. *feather* dilates the mask
    edge and fades the alpha so the seam isn't boxy.
    """
    import cv2

    H, W = background.shape[:2]
    ch, cw = crop_img.shape[:2]
    # Clip to background bounds
    y0 = max(0, top)
    x0 = max(0, left)
    y1 = min(H, top + ch)
    x1 = min(W, left + cw)
    if y1 <= y0 or x1 <= x0:
        return
    cy0 = y0 - top
    cx0 = x0 - left
    cy1 = cy0 + (y1 - y0)
    cx1 = cx0 + (x1 - x0)

    sub_bg = background[y0:y1, x0:x1]
    sub_crop = crop_img[cy0:cy1, cx0:cx1]
    sub_mask = crop_mask[cy0:cy1, cx0:cx1]

    if sub_mask.sum() == 0:
        return

    m = (sub_mask > 0).astype(np.float32)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=feather)
    m = np.clip(m, 0, 1)[..., None]
    blended = sub_bg.astype(np.float32) * (1 - m) + sub_crop.astype(np.float32) * m
    background[y0:y1, x0:x1] = blended.astype(np.uint8)


def _pick_paste_location(
    bg_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    forbidden_mask: np.ndarray,
    rng: random.Random,
    max_tries: int = 30,
) -> tuple[int, int] | None:
    """Try to find a (top, left) where the crop fits without overlapping
    forbidden regions (existing defects or previously pasted synths)."""
    H, W = bg_shape
    ch, cw = crop_shape
    if ch >= H or cw >= W:
        return None
    for _ in range(max_tries):
        top = rng.randint(0, H - ch - 1)
        left = rng.randint(0, W - cw - 1)
        if forbidden_mask[top:top + ch, left:left + cw].sum() == 0:
            return top, left
    return None


def synthesize_from_labeled(
    pairs: list[tuple[Path, Path]],
    n_samples: int,
    defects_per_image: tuple[int, int] = (1, 4),
    perlin_strength: float = 6.0,
    color_jitter: float = 0.15,
    seed: int | None = None,
    min_defect_area: int = 16,
    max_defect_area: int | None = None,
    host_pairs: list[tuple[Path, Path]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Generate *n_samples* synthesized (image, mask, meta) tuples.

    *pairs* are used as the source pool for defect crops (must contain FG
    pixels). *host_pairs*, if given, becomes the pool of background hosts
    onto which defects are pasted; pass clean (defect-free) images here in
    addition to the labeled ones to diversify the backgrounds. When
    *host_pairs* is ``None`` the labeled *pairs* double as hosts (legacy
    behaviour).

    Meta dict includes: source_image (host stem), defect_sources (list of
    stems the pasted crops came from), n_defects (int).
    """
    import cv2

    if not pairs:
        raise ValueError("no labeled pairs provided")
    if host_pairs is None:
        host_pairs = pairs
    elif not host_pairs:
        raise ValueError("host_pairs is empty")
    rng = random.Random(seed)
    np_rng_seed = seed if seed is not None else rng.randint(0, 2**31 - 1)

    crops = extract_defect_crops(
        pairs,
        min_area=min_defect_area,
        max_area=max_defect_area,
    )
    if not crops:
        raise ValueError("no defect crops could be extracted — dataset has no FG pixels")

    samples: list[tuple[np.ndarray, np.ndarray, dict]] = []
    lo, hi = defects_per_image
    for idx in range(n_samples):
        host_img_p, host_mask_p = rng.choice(host_pairs)
        host_img = _imread_unicode(host_img_p, cv2.IMREAD_COLOR)
        host_mask = _imread_unicode(host_mask_p, cv2.IMREAD_GRAYSCALE)
        if host_img is None or host_mask is None:
            continue
        bg = host_img.copy()
        new_mask = ((host_mask > 0) & (host_mask != 255)).astype(np.uint8) * host_mask
        forbidden = (new_mask > 0).astype(np.uint8)

        n_def = rng.randint(lo, hi)
        pasted_sources: list[str] = []
        for _ in range(n_def):
            crop = rng.choice(crops)
            # Random rotation 0/90/180/270 + optional hflip
            cimg = crop.image.copy()
            cmask = crop.mask.copy()
            k_rot = rng.randint(0, 3)
            if k_rot:
                cimg = np.rot90(cimg, k_rot).copy()
                cmask = np.rot90(cmask, k_rot).copy()
            if rng.random() < 0.5:
                cimg = np.ascontiguousarray(cimg[:, ::-1])
                cmask = np.ascontiguousarray(cmask[:, ::-1])
            # Scale 0.75x – 1.4x
            scale = rng.uniform(0.75, 1.4)
            new_h = max(4, int(cimg.shape[0] * scale))
            new_w = max(4, int(cimg.shape[1] * scale))
            cimg = cv2.resize(cimg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            cmask = cv2.resize(cmask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Perlin warp
            cimg, cmask = perlin_warp(
                cimg, cmask,
                strength=perlin_strength,
                res=(rng.randint(3, 6), rng.randint(3, 6)),
                seed=np_rng_seed + idx * 97 + rng.randint(0, 10_000),
            )
            # Color jitter
            cimg = _color_jitter(cimg, strength=color_jitter, rng=rng)
            # Place
            loc = _pick_paste_location(bg.shape[:2], cimg.shape[:2], forbidden, rng)
            if loc is None:
                continue
            top, left = loc
            _paste_with_feather(bg, cimg, cmask, top, left, feather=2)
            # Update masks
            sub = new_mask[top:top + cimg.shape[0], left:left + cimg.shape[1]]
            cm = cmask[: sub.shape[0], : sub.shape[1]]
            sub[cm > 0] = cm[cm > 0]
            forbidden[top:top + cimg.shape[0], left:left + cimg.shape[1]] |= (cm > 0).astype(np.uint8)
            pasted_sources.append(crop.source_stem)

        meta = {
            "source_image": host_img_p.stem,
            "defect_sources": pasted_sources,
            "n_defects": len(pasted_sources),
            "real_defects_kept": int((host_mask > 0).sum() > 0),
        }
        samples.append((bg, new_mask, meta))
    return samples
