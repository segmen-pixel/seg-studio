# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Copy-paste synthetic composer for instance segmentation.

Turns ordinary semantic masks into a COCO instance dataset with exact
per-instance visible masks:

  * cutouts        = connected components inside the single-object area band
  * real full-GT   = source images where every blob is single-object sized
  * background     = source images with foreground inpainted away
  * composition    = painter's algorithm (later paste occludes earlier), so
                     every instance's visible mask is known exactly
  * stack pairs    = two cutouts rotated onto a shared PCA axis and placed at
                     contact-to-slight-overlap along it — the configuration a
                     detector otherwise fuses into one instance

Inputs are in-memory ``(item_id, image_bgr, foreground_mask)`` triples; the
caller owns storage. All randomness flows from ``ComposeConfig.seed``.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

_MIN_BLOB_AREA = 300
_MIN_VISIBLE_FRACTION = 0.55   # of a cutout's own area when pasted
_KEEP_VISIBLE_FRACTION = 0.22  # after later pastes occlude it
_KEEP_VISIBLE_AREA = 400


@dataclass
class ComposeConfig:
    n_train: int = 500
    n_val: int = 80
    objects_min: int = 4
    objects_max: int = 8
    # P(2 stack pairs) = stack_pair_prob * pair2_share, P(1 pair) = the rest.
    stack_pair_prob: float = 0.55
    pair2_share: float = 0.36
    seed: int = 42
    area_band: tuple[int, int] | None = None  # None = estimate from blobs
    scale_jitter: tuple[float, float] = (0.92, 1.08)
    brightness_jitter: tuple[float, float] = (0.92, 1.08)
    stack_overlap: tuple[float, float] = (0.80, 0.97)
    bg_plate_count: int = 48
    bg_plate_stride: int = 8
    # Patch mode: build canvases by cropping the inpainted plate to this size
    # and pasting cutouts at their native scale, instead of using the whole
    # plate and normalising cutout size to it. Named to match the semantic
    # side, which trains on patches cropped at native resolution and evaluates
    # with a sliding window over the same size.
    #
    # It exists so training can match sliding-window inference. Whole-plate
    # canvases are the source resolution, and the detector resizes its input to
    # a fixed square, so a 2560x2048 photo arrives at 432 and a 110px screw
    # becomes 18px. Tiles are already the input size, so nothing is resized and
    # the object keeps the pixel size the camera gave it -- the same size the
    # tiled inference path will show the model.
    #
    # None keeps the whole-plate behaviour.
    patch_size: int | None = None


@dataclass
class Material:
    cutouts: list = field(default_factory=list)      # (rgb, alpha) patches
    real_full: list = field(default_factory=list)    # (item_id, image, [inst_mask...])
    bg_plates: list = field(default_factory=list)    # inpainted backgrounds
    area_band: tuple[int, int] = (0, 0)              # dominant-resolution band
    # {"WxH": [lo, hi]} — the band actually applied per source resolution
    area_bands_by_resolution: dict = field(default_factory=dict)
    # Parallel to cutouts / bg_plates: the source resolution each piece came
    # from, plus the single-object median area per (class, resolution). To put
    # a cutout on a plate from another resolution the composer compares that
    # cutout's OWN class median at both resolutions — a pure pixel-scale
    # conversion. Comparing medians across classes instead would rescale by
    # the class size ratio, which is not a scale difference at all.
    # Empty lists = uniform scale (legacy).
    cutout_res: list = field(default_factory=list)
    plate_res: list = field(default_factory=list)
    med_by_key: dict = field(default_factory=dict)
    # Semantic class id per cutout (parallel to cutouts) and the sorted set
    # of classes any source contributed — the composer and the COCO writer
    # carry these through so one model counts several classes at once.
    cutout_classes: list = field(default_factory=list)
    class_ids: list = field(default_factory=list)
    # Exclusion accounting — touching objects merge into one connected
    # component whose area falls outside the band; these counters make that
    # silent loss visible in the compose stats/log.
    n_blobs_excluded_band: int = 0    # blobs outside the area band
    n_blobs_excluded_border: int = 0  # in-band blobs clipped at the border
    n_real_excluded_band: int = 0     # images dropped from real_full
    n_blobs_split: int = 0            # merged blobs recovered by splitting
    n_objects_recovered: int = 0      # objects those splits added to the GT


def estimate_single_object_band(areas: list[int]) -> tuple[int, int]:
    """Estimate the single-object blob-area band from all foreground blobs.

    Assumes single objects are the most common blob population; merged blobs
    sit at rough integer multiples of the median. A relative band around the
    median separates them without any per-project tuning.
    """
    if not areas:
        raise ValueError("no foreground blobs to estimate the area band from")
    med = float(np.median(np.asarray(areas, dtype=np.float64)))
    return int(med * 0.5), int(med * 1.7)


def _blobs(mask: np.ndarray) -> tuple[np.ndarray, list]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= _MIN_BLOB_AREA:
            out.append((i, a, stats[i]))
    return lab, out


def estimate_bands_by_resolution(
    sources: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]]:
    """Estimate one single-object area band per image resolution.

    A single global band breaks on mixed-resolution projects: the same
    physical object fills ~5k px in a tight 512 crop but ~100k px in a full
    camera frame, so a global median lands between the two populations and
    excludes both. Within one resolution the objects share a pixel scale,
    so the median heuristic (estimate_single_object_band) holds.

    Returns ({(h, w): (lo, hi)}, {(h, w): n_blobs}); resolutions without
    foreground blobs are absent from both.
    """
    areas: dict[tuple[int, int], list[int]] = {}
    for _, _, fg in sources:
        _, bl = _blobs(fg)
        if bl:
            areas.setdefault(fg.shape[:2], []).extend(a for _, a, _ in bl)
    bands = {res: estimate_single_object_band(v) for res, v in areas.items()}
    counts = {res: len(v) for res, v in areas.items()}
    return bands, counts


def class_ids_in(mask: np.ndarray) -> list[int]:
    """Class ids present in a label mask (0 = background, 255 = ignore)."""
    return sorted(int(v) for v in np.unique(mask) if v not in (0, 255))


def estimate_bands_by_class_resolution(
    sources: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[tuple[int, tuple[int, int]], tuple[int, int]],
           dict[tuple[int, tuple[int, int]], int]]:
    """Estimate a single-object area band per (class, image resolution).

    Classes are sized independently for the same reason resolutions are:
    a nut and a screw in the same frame have different single-object areas,
    so one shared median would exclude both populations.

    Returns ({(class_id, (h, w)): (lo, hi)}, {key: n_blobs}).
    """
    areas: dict[tuple[int, tuple[int, int]], list[int]] = {}
    for _, _, label in sources:
        res = label.shape[:2]
        for cid in class_ids_in(label):
            _, bl = _blobs((label == cid).astype(np.uint8))
            if bl:
                areas.setdefault((cid, res), []).extend(a for _, a, _ in bl)
    bands = {k: estimate_single_object_band(v) for k, v in areas.items()}
    counts = {k: len(v) for k, v in areas.items()}
    return bands, counts


#: How far a blob's area may sit from an exact multiple of the single-object
#: median and still be read as that many merged objects. Measured on real
#: screw photos the four merged pairs came in at 1.98-2.06.
_MERGE_RATIO_TOL = 0.25


def split_merged_blob(mask: np.ndarray, n: int) -> list[np.ndarray] | None:
    """Split a blob of *n* touching objects, or return None if it cannot.

    Touching parts merge into one connected component whose area sits at a
    near-integer multiple of the single-object median, and until this existed
    the whole image was discarded as unusable ground truth -- on a real screw
    project that cost 3 of 4 annotated images, which left the validation split
    with no real GT at all and the count threshold silently uncalibrated.

    A distance transform separates them: the centre of each object is far from
    the background, the join between two is not. Seeds are taken at the highest
    threshold that yields exactly *n* of them, so the split is only accepted
    when the shape actually argues for *n* -- a blob that is one misshapen
    object never produces two peaks and is refused rather than invented.
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    if n < 2 or m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 0:
        return None
    for frac in (0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3):
        _, sure = cv2.threshold(dist, frac * peak, 255, 0)
        sure = sure.astype(np.uint8)
        k, markers = cv2.connectedComponents(sure)
        if k - 1 != n:
            continue
        markers = markers + 1
        markers[cv2.subtract(m, sure // 255) > 0] = 0
        markers = cv2.watershed(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR), markers)
        parts = [(markers == lbl).astype(np.uint8) for lbl in range(2, n + 2)]
        if all(p.sum() for p in parts):
            return parts
    return None


def collect_material(
    sources: list[tuple[str, np.ndarray, np.ndarray]],
    cfg: ComposeConfig,
    exclude_ids: set[str] | None = None,
) -> Material:
    """Extract cutouts, real full-GT images and background plates.

    ``sources`` yields (item_id, image_bgr, label_mask). The label mask
    carries semantic class ids (0 = background, 255 = ignore); a plain 0/1
    binary mask is the single-class case. Items in ``exclude_ids`` (e.g. an
    evaluation holdout) contribute nothing.

    The single-object area band is estimated per (class, resolution) — see
    estimate_bands_by_class_resolution; a manual ``cfg.area_band`` overrides
    it for every class. ``Material.area_band`` reports the band of the
    (class, resolution) with the most blobs.
    """
    exclude_ids = exclude_ids or set()
    manual_band = cfg.area_band
    bands: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
    if manual_band is None:
        bands, blob_counts = estimate_bands_by_class_resolution(sources)
        if not bands:
            raise ValueError("no foreground blobs to estimate the area band from")
        dominant = max(bands, key=lambda k: blob_counts.get(k, 0))
        default_band = bands[dominant]
    else:
        default_band = manual_band

    # Single-object median per (class, resolution) key (band = med*0.5 ..
    # med*1.7, so med ~= lo*2) — drives the composer's cross-resolution
    # rescaling. A cutout whose key is absent is simply left unscaled, so
    # there is no default: a fallback median would silently reintroduce a
    # cross-class ratio, which is the bug this table replaced.
    med_by_key = {k: max(float(b[0]) * 2.0, 1.0) for k, b in bands.items()}

    mat = Material(area_band=default_band)
    mat.med_by_key = dict(med_by_key)
    mat.area_bands_by_resolution = {
        f"class{cid}@{res[1]}x{res[0]}": list(b)
        for (cid, res), b in sorted(bands.items())}
    seen_classes: set[int] = set()
    for idx, (item_id, img, label) in enumerate(sources):
        if item_id in exclude_ids:
            continue
        h, w = label.shape[:2]
        res = label.shape[:2]
        fg_any = ((label != 0) & (label != 255)).astype(np.uint8)
        image_ok = True  # every blob of every class within its band
        real_instances: list[tuple[np.ndarray, int]] = []
        for cid in class_ids_in(label):
            seen_classes.add(cid)
            lo, hi = bands.get((cid, res), default_band)
            # Median single-object area for this key; the band is med*0.5..
            # med*1.7, so lo*2 recovers it. A manual band has no table.
            med = mat.med_by_key.get((cid, res), float(lo) * 2.0)
            lab, bl = _blobs((label == cid).astype(np.uint8))
            for i, a, st in bl:
                if a > hi:
                    # Above the band is normally several touching objects.
                    # Split them rather than discarding the image: the GT is
                    # recoverable, and dropping it is what left projects with
                    # no real validation images and an uncalibrated threshold.
                    ratio = a / med if med > 0 else 0.0
                    n = int(round(ratio))
                    parts = (split_merged_blob(lab == i, n)
                             if n >= 2 and abs(ratio - n) <= _MERGE_RATIO_TOL
                             else None)
                    # Only trust a split whose pieces look like single objects.
                    if parts and all(lo <= int(pt.sum()) <= hi for pt in parts):
                        mat.n_blobs_split += 1
                        mat.n_objects_recovered += len(parts) - 1
                        real_instances.extend((pt, cid) for pt in parts)
                        # Not offered as cutouts: a piece carries its
                        # neighbour's pixels along the join, and composition
                        # pastes the image content, not just the silhouette.
                        continue
                    mat.n_blobs_excluded_band += 1
                    image_ok = False
                    continue
                if a < lo:
                    mat.n_blobs_excluded_band += 1
                    image_ok = False
                    continue
                real_instances.append(((lab == i).astype(np.uint8), cid))
                x, y = st[cv2.CC_STAT_LEFT], st[cv2.CC_STAT_TOP]
                bw, bh = st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT]
                if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                    mat.n_blobs_excluded_border += 1
                    continue  # touches the border: probably clipped
                alpha = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
                mat.cutouts.append((img[y:y + bh, x:x + bw].copy(), alpha))
                mat.cutout_res.append(res)
                mat.cutout_classes.append(cid)
        if real_instances and image_ok:
            mat.real_full.append((item_id, img, real_instances))
        elif real_instances:
            # At least one out-of-band blob (typically touching objects merged
            # into one component): the whole image is unusable as real GT.
            mat.n_real_excluded_band += 1
        if (len(mat.bg_plates) < cfg.bg_plate_count
                and idx % cfg.bg_plate_stride == cfg.bg_plate_stride - 1):
            dil = cv2.dilate(fg_any, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            mat.bg_plates.append(cv2.inpaint(img, dil, 5, cv2.INPAINT_TELEA))
            # The plate's pixel scale is its resolution; which classes happen
            # to appear in this source is irrelevant to it.
            mat.plate_res.append(res)
    if not mat.bg_plates:
        # Fewer sources than the plate stride (the 4-7 image case): fall back
        # to one plate from the first usable source so composition can run.
        for item_id, img, label in sources:
            if item_id in exclude_ids:
                continue
            fg_any = ((label != 0) & (label != 255)).astype(np.uint8)
            dil = cv2.dilate(fg_any, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            mat.bg_plates.append(cv2.inpaint(img, dil, 5, cv2.INPAINT_TELEA))
            mat.plate_res.append(label.shape[:2])
            break
    mat.class_ids = sorted(seen_classes)
    return mat


# ── geometry ────────────────────────────────────────────────────

def axis_angle(alpha: np.ndarray) -> float:
    """Principal-axis angle of a mask in pixel coords (atan2(dy, dx), degrees)."""
    ys, xs = np.nonzero(alpha)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
    cov = pts @ pts.T / pts.shape[1]
    _, evecs = np.linalg.eigh(cov)
    v = evecs[:, -1]
    return float(np.degrees(np.arctan2(v[1], v[0])))


class _Composer:
    def __init__(self, mat: Material, cfg: ComposeConfig):
        self.mat = mat
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)

    def _warp(self, rgb: np.ndarray, alpha: np.ndarray, ang: float, sc: float):
        h, w = alpha.shape
        diag = int(np.hypot(h, w) * sc) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += (diag - w) / 2
        M[1, 2] += (diag - h) / 2
        rgb2 = cv2.warpAffine(rgb, M, (diag, diag), flags=cv2.INTER_LINEAR)
        a2 = cv2.warpAffine(alpha, M, (diag, diag), flags=cv2.INTER_NEAREST)
        lo, hi = self.cfg.brightness_jitter
        rgb2 = np.clip(rgb2.astype(np.float32) * self.rng.uniform(lo, hi), 0, 255).astype(np.uint8)
        return rgb2, a2

    def _rot_random(self, rgb, alpha, base_scale: float = 1.0):
        rgb2, a2 = self._warp(rgb, alpha, self.rng.uniform(0, 360),
                              base_scale * self.rng.uniform(*self.cfg.scale_jitter))
        if self.rng.random() < 0.5:
            rgb2, a2 = cv2.flip(rgb2, 1), cv2.flip(a2, 1)
        return rgb2, a2

    def _rot_to(self, rgb, alpha, target_deg, base_scale: float = 1.0):
        # cv2.getRotationMatrix2D(angle=a) maps a pixel-space direction phi to
        # phi - a, so a = intrinsic - target lands the axis on target. The
        # random extra 180 flips which end leads (head-head / head-tip / tip-tip).
        ang = axis_angle(alpha) - target_deg
        if self.rng.random() < 0.5:
            ang += 180.0
        return self._warp(rgb, alpha, ang,
                          base_scale * self.rng.uniform(*self.cfg.scale_jitter))

    def _fit_to_canvas(self, rgb, a, H, W):
        """Hard cap: a piece must fit the canvas (85% of each side).

        Cross-resolution rescaling normally keeps pieces canvas-sized, but a
        piece that still overflows (manual band override, extreme aspect)
        would make the placement randint range empty — shrink it instead.
        """
        ph, pw = a.shape
        f = min(0.85 * W / pw, 0.85 * H / ph, 1.0)
        if f < 1.0:
            nw, nh = max(1, int(pw * f)), max(1, int(ph * f))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return rgb, a

    def _paste(self, canvas, inst_masks, rgb, a, px, py, class_id: int = 1) -> bool:
        H, W = canvas.shape[:2]
        ph, pw = a.shape
        x0, y0 = max(px, 0), max(py, 0)
        x1, y1 = min(px + pw, W), min(py + ph, H)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return False
        sub_a = a[y0 - py:y1 - py, x0 - px:x1 - px]
        sub_rgb = rgb[y0 - py:y1 - py, x0 - px:x1 - px]
        if int(sub_a.sum()) / max(int(a.sum()), 1) < _MIN_VISIBLE_FRACTION:
            return False
        sh = cv2.GaussianBlur(sub_a.astype(np.float32), (9, 9), 0) * 0.25
        sy, sx = min(y0 + 4, H - 1), min(x0 + 3, W - 1)
        h_s, w_s = min(H - sy, sub_a.shape[0]), min(W - sx, sub_a.shape[1])
        region = canvas[sy:sy + h_s, sx:sx + w_s].astype(np.float32)
        canvas[sy:sy + h_s, sx:sx + w_s] = (region * (1 - sh[:h_s, :w_s, None])).astype(np.uint8)
        m3 = sub_a[:, :, None].astype(bool)
        canvas[y0:y1, x0:x1] = np.where(m3, sub_rgb, canvas[y0:y1, x0:x1])
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = sub_a
        for im in inst_masks:
            im["vis"][full > 0] = 0
        inst_masks.append({"vis": full.copy(), "orig_area": int(a.sum()),
                           "class_id": int(class_id)})
        return True

    def _pick(self, canvas_res=None):
        """Random cutout plus the base scale that matches it to the canvas.

        The cutout and the canvas plate may come from sources of different
        resolution. Comparing this cutout's own class median area at the two
        resolutions gives the pixel-scale ratio; areas scale with the square
        of length, so the linear scale is its square root. Same resolution —
        or an unknown one — leaves the cutout untouched.
        """
        i = self.rng.randrange(len(self.mat.cutouts))
        rgb, a = self.mat.cutouts[i]
        class_id = (int(self.mat.cutout_classes[i])
                    if len(self.mat.cutout_classes) == len(self.mat.cutouts) else 1)
        scale = 1.0
        if (canvas_res is not None and self.mat.med_by_key
                and len(self.mat.cutout_res) == len(self.mat.cutouts)):
            src_res = self.mat.cutout_res[i]
            src = self.mat.med_by_key.get((class_id, tuple(src_res)))
            dst = self.mat.med_by_key.get((class_id, tuple(canvas_res)))
            if src and dst and src > 0:
                scale = (float(dst) / float(src)) ** 0.5
        return rgb, a, scale, class_id

    def place_stack_pair(self, canvas, inst_masks, canvas_res=None) -> bool:
        """Two cutouts on one axis, touching or slightly overlapping along it.

        Transactional: if either paste fails, the canvas and every earlier
        instance mask are restored so a half-placed pair never leaks into
        the sample (a lone paste would silently miscount as 2 objects).
        """
        H, W = canvas.shape[:2]
        t = self.rng.uniform(0, 360)
        u = np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
        # Both cutouts are drawn independently, so a pair may mix classes —
        # touching unlike objects is the realistic case and teaches the model
        # to split them (owner decision 2026-07-23).
        rgbA0, aA0, sA, cidA = self._pick(canvas_res)
        rgbB0, aB0, sB, cidB = self._pick(canvas_res)
        rgbA, aA = self._rot_to(rgbA0, aA0, t, sA)
        rgbB, aB = self._rot_to(rgbB0, aB0, t, sB)
        rgbA, aA = self._fit_to_canvas(rgbA, aA, H, W)
        rgbB, aB = self._fit_to_canvas(rgbB, aB, H, W)

        def half_extents(a):
            ys, xs = np.nonzero(a)
            c = np.array([a.shape[1] / 2.0, a.shape[0] / 2.0])
            proj = (np.stack([xs, ys], 1) - c) @ u
            return -float(proj.min()), float(proj.max())  # (backward, forward)

        _, fwdA = half_extents(aA)
        backB, _ = half_extents(aB)
        cA = np.array([self.rng.uniform(0.27 * W, 0.73 * W),
                       self.rng.uniform(0.27 * H, 0.73 * H)])
        f = self.rng.uniform(*self.cfg.stack_overlap)
        perp = np.array([-u[1], u[0]]) * self.rng.uniform(-4, 4)
        cB = cA + u * (fwdA + backB) * f + perp
        snapshot_canvas = canvas.copy()
        snapshot_masks = [{"vis": im["vis"].copy(), "orig_area": im["orig_area"],
                           "class_id": im["class_id"]}
                          for im in inst_masks]
        okA = self._paste(canvas, inst_masks, rgbA, aA,
                          int(cA[0] - aA.shape[1] / 2), int(cA[1] - aA.shape[0] / 2), cidA)
        okB = okA and self._paste(canvas, inst_masks, rgbB, aB,
                                  int(cB[0] - aB.shape[1] / 2), int(cB[1] - aB.shape[0] / 2),
                                  cidB)
        if not (okA and okB):
            canvas[:] = snapshot_canvas
            inst_masks[:] = snapshot_masks
            return False
        return True

    def _crop_patch(self, plate):
        """A random patch_size square from *plate*, or the whole plate if smaller.

        A plate smaller than the patch is used whole rather than padded out to
        the patch size. Inference clamps its crops the same way, so a source
        smaller than the patch reaches the model as itself in both -- which is
        the only thing that keeps the two at one scale. Padding here would
        train the model on a frame that is mostly mirrored filler while
        inference showed it the plain image, and nothing would report the
        mismatch.

        A patch larger than every plate therefore composes exactly what
        whole-plate mode composes, which is the intended degradation: tiling
        turns itself off where there is nothing to tile.
        """
        from .tiled import pad_to_patch

        h, w = plate.shape[:2]
        # Both the shape and the padding come from the shared helpers inference
        # uses. Training and inference must show the model the same frame, and
        # a mismatch is silent -- the model runs and the count is simply wrong.
        # Do not reimplement either rule here.
        t = int(self.cfg.patch_size)
        ch, cw = min(h, t), min(w, t)
        y = self.rng.randrange(0, h - ch + 1)
        x = self.rng.randrange(0, w - cw + 1)
        return pad_to_patch(plate[y:y + ch, x:x + cw].copy(), t)

    def _synth_once(self):
        pi = self.rng.randrange(len(self.mat.bg_plates))
        if self.cfg.patch_size:
            canvas = self._crop_patch(self.mat.bg_plates[pi])
            # No canvas_res: cutouts must keep their native size, which is the
            # whole point of patch mode.
            canvas_res = None
        else:
            canvas = self.mat.bg_plates[pi].copy()
            canvas_res = None
            if len(self.mat.plate_res) == len(self.mat.bg_plates):
                canvas_res = self.mat.plate_res[pi]
        H, W = canvas.shape[:2]
        inst_masks: list[dict] = []
        n_objects = self.rng.randint(self.cfg.objects_min, self.cfg.objects_max)
        r = self.rng.random()
        p2 = self.cfg.stack_pair_prob * self.cfg.pair2_share
        n_pairs = 2 if r < p2 else (1 if r < self.cfg.stack_pair_prob else 0)
        placed = 0
        for _ in range(n_pairs):
            if placed + 2 > n_objects:
                break
            if self.place_stack_pair(canvas, inst_masks, canvas_res):
                placed += 2
        centers = [(self.rng.uniform(0.27 * W, 0.73 * W), self.rng.uniform(0.27 * H, 0.73 * H))
                   for _ in range(self.rng.randint(1, 2))]
        # Failed pastes don't consume the object budget; the attempt cap
        # bounds the loop when the material simply doesn't fit anywhere.
        single_attempts = 0
        while placed < n_objects and single_attempts < n_objects * 4:
            single_attempts += 1
            rgb0, a0, s, cid = self._pick(canvas_res)
            rgb, a = self._rot_random(rgb0, a0, s)
            rgb, a = self._fit_to_canvas(rgb, a, H, W)
            ph, pw = a.shape
            if self.rng.random() < 0.65:
                cx, cy = centers[self.rng.randrange(len(centers))]
                px = int(self.np_rng.normal(cx, 34)) - pw // 2
                py = int(self.np_rng.normal(cy, 34)) - ph // 2
            else:
                px = self.rng.randint(-20, W - pw + 20)
                py = self.rng.randint(-20, H - ph + 20)
            if self._paste(canvas, inst_masks, rgb, a, px, py, cid):
                placed += 1
        keep = [(im["vis"], im["class_id"]) for im in inst_masks
                if int(im["vis"].sum()) >= _KEEP_VISIBLE_AREA
                and int(im["vis"].sum()) / im["orig_area"] >= _KEEP_VISIBLE_FRACTION]
        return canvas, keep

    def synth_image(self, max_tries: int = 5):
        """Compose one sample, retrying until >= objects_min instances survive.

        Placement failures and the occlusion filter can drop the final
        instance count below objects_min; retry (bounded) and fall back to
        the attempt with the most surviving instances.
        """
        best = None
        for _ in range(max_tries):
            canvas, keep = self._synth_once()
            if len(keep) >= self.cfg.objects_min:
                return canvas, keep
            if best is None or len(keep) > len(best[1]):
                best = (canvas, keep)
        return best


# ── train/val source split (leakage guard) ──────────────────────

def split_source_ids(item_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Deterministic train/validation split of SOURCE image ids.

    Validation cutouts, backgrounds and real full-GT images must never feed
    the training composition — otherwise val mAP and the count-threshold
    calibration overstate real-world performance. Split at the original
    image id level, before any material extraction.
    """
    ids = sorted(set(item_ids))
    rng = random.Random(seed ^ 0x5EED)  # decoupled from the composition stream
    rng.shuffle(ids)
    n_val = max(1, len(ids) // 8) if len(ids) < 16 else max(2, len(ids) // 8)
    return sorted(ids[n_val:]), sorted(ids[:n_val])


# ── COCO output ─────────────────────────────────────────────────

def _polygons(mask: np.ndarray) -> list:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 60:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)
        if len(c) >= 3:
            polys.append([float(v) for v in c.reshape(-1)])
    return polys


def _new_coco(class_ids: list[int] | None = None,
              class_names: dict[int, str] | None = None) -> dict:
    """Fresh COCO skeleton.

    COCO category ids are 1..N and must be contiguous for the detector, so
    they are NOT the semantic class ids: ``coco_category_of`` maps between
    them and the mapping is persisted in the dataset stats so inference can
    translate predictions back to semantic classes.
    """
    ids = sorted(class_ids) if class_ids else [1]
    names = class_names or {}
    return {"images": [], "annotations": [],
            "categories": [{"id": n + 1, "name": names.get(cid, f"class{cid}"),
                            "supercategory": "none"}
                           for n, cid in enumerate(ids)]}


def coco_category_of(class_ids: list[int]) -> dict[int, int]:
    """Semantic class id -> contiguous COCO category id (1-based)."""
    return {cid: n + 1 for n, cid in enumerate(sorted(class_ids))}


def _add_sample(coco: dict, split_dir: Path, name: str, img: np.ndarray, inst_masks,
                cat_of: dict[int, int] | None = None) -> None:
    """Write one image + its instances. ``inst_masks`` is a list of
    ``(mask, class_id)`` pairs (a bare mask is treated as class 1)."""
    iid = len(coco["images"]) + 1
    cv2.imwrite(str(split_dir / name), img)
    h, w = img.shape[:2]
    coco["images"].append({"id": iid, "file_name": name, "width": w, "height": h})
    cat_of = cat_of or {}
    for entry in inst_masks:
        m, class_id = entry if isinstance(entry, tuple) else (entry, 1)
        polys = _polygons(m)
        if not polys:
            continue
        ys, xs = np.nonzero(m)
        coco["annotations"].append({
            "id": len(coco["annotations"]) + 1, "image_id": iid,
            "category_id": cat_of.get(int(class_id), 1),
            "segmentation": polys, "area": int(m.sum()),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "iscrowd": 0,
        })


def compose_dataset_split(
    sources: list[tuple[str, np.ndarray, np.ndarray]],
    out_dir: Path | str,
    cfg: ComposeConfig,
    progress_fn: Callable[[str], None] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict:
    """Leakage-safe dataset composition: split sources first, then compose.

    Source images are partitioned train/validation up front
    (``split_source_ids``); each side collects its own Material, so no
    validation cutout, background plate or real full-GT image ever reaches
    the training composition, and the count-threshold calibration runs on a
    true holdout. This is the production path — ``compose_dataset`` below
    composes both splits from one shared pool and remains only for
    PoC-parity experiments.
    """
    def _report(msg: str) -> None:
        if progress_fn is not None:
            progress_fn(msg)

    train_ids, val_ids = split_source_ids([s[0] for s in sources], cfg.seed)
    src_train = [s for s in sources if s[0] in set(train_ids)]
    src_val = [s for s in sources if s[0] in set(val_ids)]
    _report(f"collecting material from {len(src_train)}+{len(src_val)} sources")
    mat_train = collect_material(src_train, cfg)
    mat_val = collect_material(src_val, cfg)
    _report(f"material: {len(mat_train.cutouts)} cutouts, "
            f"{len(mat_train.real_full) + len(mat_val.real_full)} real-GT images, "
            f"{len(mat_train.bg_plates)} bg plates")
    if not mat_train.cutouts:
        raise ValueError("training split has no cutouts in the single-object area band; "
                         "check masks or override area_band")
    if not mat_train.bg_plates or not mat_val.bg_plates:
        raise ValueError("no background plates could be built from the sources")
    if not mat_val.cutouts and not mat_val.real_full:
        raise ValueError("validation split has no usable objects; "
                         "add more annotated images")

    out = Path(out_dir)
    train_dir, val_dir = out / "train", out / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    # One category list for both splits: the detector's head is sized from
    # it, so a class seen only in validation must still have a slot.
    all_class_ids = sorted(set(mat_train.class_ids) | set(mat_val.class_ids)) or [1]
    cat_of = coco_category_of(all_class_ids)
    coco_tr = _new_coco(all_class_ids, class_names)
    coco_va = _new_coco(all_class_ids, class_names)

    comp_tr = _Composer(mat_train, cfg)
    for n in range(cfg.n_train):
        img, masks = comp_tr.synth_image()
        _add_sample(coco_tr, train_dir, f"syn_{n:04d}.jpg", img, masks, cat_of)
        if (n + 1) % 100 == 0 or n + 1 == cfg.n_train:
            _report(f"composing train {n + 1}/{cfg.n_train}")
    if mat_val.cutouts:
        comp_va = _Composer(mat_val, replace(cfg, seed=cfg.seed + 1))
        for n in range(cfg.n_val):
            img, masks = comp_va.synth_image()
            _add_sample(coco_va, val_dir, f"synv_{n:04d}.jpg", img, masks, cat_of)
            if (n + 1) % 100 == 0 or n + 1 == cfg.n_val:
                _report(f"composing valid {n + 1}/{cfg.n_val}")

    for item_id, img, inst in mat_train.real_full:
        _add_sample(coco_tr, train_dir, f"real_{item_id}.jpg", img, inst, cat_of)
    for item_id, img, inst in mat_val.real_full:
        _add_sample(coco_va, val_dir, f"real_{item_id}.jpg", img, inst, cat_of)
    if not coco_va["images"]:
        raise ValueError("validation split produced no images; "
                         "add more annotated images")

    for coco, d in ((coco_tr, train_dir), (coco_va, val_dir)):
        with open(d / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump(coco, f)

    return {
        "n_train_images": len(coco_tr["images"]),
        "n_val_images": len(coco_va["images"]),
        "n_train_annotations": len(coco_tr["annotations"]),
        "n_val_annotations": len(coco_va["annotations"]),
        "n_cutouts": len(mat_train.cutouts),
        "n_real_full": len(mat_train.real_full) + len(mat_val.real_full),
        "n_bg_plates": len(mat_train.bg_plates),
        "area_band": list(mat_train.area_band),
        "area_bands_by_resolution": mat_train.area_bands_by_resolution,
        # Semantic class ids counted by this dataset, their contiguous COCO
        # category ids (inference maps predictions back through this), and
        # how much material each class contributed.
        "class_ids": all_class_ids,
        "coco_category_of": {str(k): v for k, v in cat_of.items()},
        "cutouts_by_class": {
            str(cid): int(sum(1 for c in mat_train.cutout_classes if c == cid))
            for cid in all_class_ids},
        "seed": cfg.seed,
        "n_train_sources": len(src_train),
        "n_val_sources": len(src_val),
        "val_source_ids": val_ids,
        # Exclusion accounting — non-zero band numbers usually mean touching
        # objects were annotated as one region (split them with a thin
        # background gap to recover the material). Merged regions whose area
        # is a clean multiple of the single-object median are split
        # automatically instead; n_blobs_split reports how often that saved an
        # image the old rule would have thrown away.
        "n_blobs_split": mat_train.n_blobs_split + mat_val.n_blobs_split,
        "n_objects_recovered":
            mat_train.n_objects_recovered + mat_val.n_objects_recovered,
        "n_blobs_excluded_band":
            mat_train.n_blobs_excluded_band + mat_val.n_blobs_excluded_band,
        "n_blobs_excluded_border":
            mat_train.n_blobs_excluded_border + mat_val.n_blobs_excluded_border,
        "n_real_excluded_band":
            mat_train.n_real_excluded_band + mat_val.n_real_excluded_band,
    }


def compose_dataset(mat: Material, out_dir: Path | str, cfg: ComposeConfig) -> dict:
    """Compose the synthetic dataset and write COCO train/valid splits.

    Real full-GT images are split ~1/8 to valid (min 4) and the rest to train,
    mirroring the composition the model will be evaluated against. Returns a
    stats dict for metrics.json / the UI.

    NOTE: both splits draw from ONE shared material pool, so validation
    metrics are optimistic. Production uses ``compose_dataset_split``.
    """
    if not mat.cutouts:
        raise ValueError("no cutouts in the single-object area band; "
                         "check masks or override area_band")
    if not mat.bg_plates:
        raise ValueError("no background plates could be built from the sources")
    out = Path(out_dir)
    comp = _Composer(mat, cfg)
    train_dir, val_dir = out / "train", out / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    class_ids = mat.class_ids or [1]
    cat_of = coco_category_of(class_ids)
    coco_tr, coco_va = _new_coco(class_ids), _new_coco(class_ids)

    for n in range(cfg.n_train):
        img, masks = comp.synth_image()
        _add_sample(coco_tr, train_dir, f"syn_{n:04d}.jpg", img, masks, cat_of)
    for n in range(cfg.n_val):
        img, masks = comp.synth_image()
        _add_sample(coco_va, val_dir, f"synv_{n:04d}.jpg", img, masks, cat_of)

    real = list(mat.real_full)
    comp.rng.shuffle(real)
    n_val_real = max(4, len(real) // 8) if real else 0
    for k, (item_id, img, inst) in enumerate(real):
        coco, d = (coco_va, val_dir) if k < n_val_real else (coco_tr, train_dir)
        _add_sample(coco, d, f"real_{item_id}.jpg", img, inst, cat_of)

    for coco, d in ((coco_tr, train_dir), (coco_va, val_dir)):
        with open(d / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump(coco, f)

    return {
        "n_train_images": len(coco_tr["images"]),
        "n_val_images": len(coco_va["images"]),
        "n_train_annotations": len(coco_tr["annotations"]),
        "n_val_annotations": len(coco_va["annotations"]),
        "n_cutouts": len(mat.cutouts),
        "n_real_full": len(mat.real_full),
        "n_bg_plates": len(mat.bg_plates),
        "area_band": list(mat.area_band),
        "seed": cfg.seed,
    }
