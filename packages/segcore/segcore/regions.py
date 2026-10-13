# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Canonical connected-region extraction for segmentation masks.

Single source of truth for turning a class-id mask into discrete regions.
Call sites historically re-implemented this with diverging choices of
connectivity, class granularity, ordering, resolution and mask source;
new code must call this module instead of running cv2.connectedComponents*
directly.

Contract (pinned by tests/test_region_contract.py):

- Connectivity is fixed at 8 and is not configurable.
- Canonical order of operations: upscale the mask to the original image
  size (nearest neighbour) -> connected component analysis -> area filter.
  ``area_px``, ``bbox`` and ``centroid`` are therefore expressed in
  original-image pixels whenever ``original_hw`` is given.
- Regions are returned in a deterministic order: ascending class id,
  then per-class cv2 label order (block-raster scan order in current
  OpenCV builds; deterministic for a given build, but not guaranteed to
  equal pixel raster order). They are deliberately never ranked by area
  or confidence, and ``max_regions`` truncation keeps the same order.
- ``min_area`` is applied after the upscale, i.e. it is measured in
  original-image pixels. ``min_area <= 0`` disables the filter and
  ``max_regions <= 0`` disables truncation.
- ``ignore_index`` pixels (default 255) never form a region. Pass
  ``ignore_index=None`` to treat every non-zero value as a class.
- ``extract_mask_regions_labeled`` returns the same regions plus an int32
  label map in which ``i + 1`` marks the pixels of ``regions[i]`` and 0
  marks every pixel no returned region claims. That map is the only
  supported way to recover which pixels a region owns: a caller that
  re-derives them (from the bbox, or with its own component pass) can
  disagree with the regions returned here about which pixels are whose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONNECTIVITY = 8


@dataclass
class MaskRegion:
    """One connected component of one class in a segmentation mask."""

    class_id: int
    area_px: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    centroid: tuple[float, float]  # (cx, cy), same coordinate space as bbox
    confidence: float | None = None  # mean of the confidence map over the region


def extract_mask_regions_labeled(
    mask: np.ndarray,
    *,
    original_hw: tuple[int, int] | None = None,
    confidence: np.ndarray | None = None,
    min_area: int = 0,
    max_regions: int = 0,
    ignore_index: int | None = 255,
) -> tuple[list[MaskRegion], np.ndarray]:
    """Extract connected regions from a class-id mask, with their pixels.

    Args:
        mask: (H, W) integer class-id mask (0 = background).
        original_hw: Optional (height, width) of the original image. When
            given and different from ``mask.shape``, the mask (and the
            confidence map) is upscaled with nearest-neighbour interpolation
            BEFORE component analysis, so all geometry is reported in
            original-image pixels.
        confidence: Optional (H, W) float confidence map aligned with
            ``mask``; each region reports the mean over its pixels.
        min_area: Drop regions smaller than this many pixels, measured
            after the upscale. ``<= 0`` disables the filter.
        max_regions: Keep at most this many regions, truncating in the
            canonical order (never by size). ``<= 0`` disables truncation.
        ignore_index: Class value that never forms a region (default 255).

    Returns:
        ``(regions, labels)``: the regions in the canonical order (see the
        module docstring), and an int32 map the size of the post-upscale
        mask where ``labels == i + 1`` is exactly the pixel set of
        ``regions[i]``. Area-filtered and truncated components leave 0
        behind, so the map never points at a region the caller was not
        given.
    """
    import cv2

    work = np.ascontiguousarray(mask)
    if work.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {work.shape}")
    conf = confidence
    if conf is not None and conf.shape != work.shape:
        raise ValueError(
            f"confidence shape {conf.shape} does not match mask shape {work.shape}"
        )

    if original_hw is not None:
        oh, ow = int(original_hw[0]), int(original_hw[1])
        if (oh, ow) != work.shape:
            if work.dtype != np.uint8:
                if work.size and (int(work.min()) < 0 or int(work.max()) > 255):
                    raise ValueError("mask values must fit uint8 to be upscaled")
                work = work.astype(np.uint8)
            work = cv2.resize(work, (ow, oh), interpolation=cv2.INTER_NEAREST)
            if conf is not None:
                conf = cv2.resize(
                    np.ascontiguousarray(conf), (ow, oh), interpolation=cv2.INTER_NEAREST
                )

    regions: list[MaskRegion] = []
    region_labels = np.zeros(work.shape, dtype=np.int32)
    for cid in np.unique(work):  # ascending class id
        cid = int(cid)
        if cid == 0 or (ignore_index is not None and cid == ignore_index):
            continue
        binary = (work == cid).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=CONNECTIVITY
        )
        for label_idx in range(1, num_labels):  # label 0 = background
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if min_area > 0 and area < min_area:
                continue
            rx = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ry = int(stats[label_idx, cv2.CC_STAT_TOP])
            rw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            rh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            # A component cannot leave its own bounding box, so both the
            # confidence mean and the label map are written through that
            # window instead of comparing the whole image once per region.
            ry2, rx2 = ry + rh, rx + rw
            window = labels[ry:ry2, rx:rx2] == label_idx
            conf_val: float | None = None
            if conf is not None:
                conf_val = float(np.mean(conf[ry:ry2, rx:rx2][window]))
            regions.append(
                MaskRegion(
                    class_id=cid,
                    area_px=area,
                    bbox=(rx, ry, rw, rh),
                    centroid=(
                        float(centroids[label_idx, 0]),
                        float(centroids[label_idx, 1]),
                    ),
                    confidence=conf_val,
                )
            )
            region_labels[ry:ry2, rx:rx2][window] = len(regions)

    if max_regions > 0 and len(regions) > max_regions:
        regions = regions[:max_regions]
        region_labels[region_labels > max_regions] = 0
    return regions, region_labels


def extract_mask_regions(
    mask: np.ndarray,
    *,
    original_hw: tuple[int, int] | None = None,
    confidence: np.ndarray | None = None,
    min_area: int = 0,
    max_regions: int = 0,
    ignore_index: int | None = 255,
) -> list[MaskRegion]:
    """Extract connected regions from a class-id mask.

    The regions alone; call :func:`extract_mask_regions_labeled` when the
    caller also has to know which pixels each region owns.
    """
    return extract_mask_regions_labeled(
        mask,
        original_hw=original_hw,
        confidence=confidence,
        min_area=min_area,
        max_regions=max_regions,
        ignore_index=ignore_index,
    )[0]
