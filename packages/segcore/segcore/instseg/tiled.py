# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Sliding-window inference for the instance detector.

The detector takes a fixed square input, so a whole photo is resized to reach
it: a 2560x2048 frame arrives at 432 and a 110px screw becomes 18px. Running
the model on native-resolution tiles instead keeps the object the size the
camera gave it -- several times the size the resize leaves it -- and
gives each tile its own query budget rather than spending one budget on the
whole frame.

Tiles overlap so an object is not only ever seen cut in half. An object smaller
than the overlap is guaranteed to appear whole in at least one tile; anything
larger may only ever be seen in pieces, which is why `plan_tiles` reports the
overlap and the caller is expected to check it against the object size it
expects (`max_whole_object_px`).

This module is deliberately model-agnostic: it plans the geometry, maps
detections back to full-frame coordinates and merges them. The caller supplies
a `predict` callable, so it works with the torch SDK and with an ONNX session
alike.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..tiling_geometry import default_patch_stride


@dataclass(frozen=True)
class TilePlan:
    """Where the patches sit, and what the geometry can and cannot see."""

    patch_size: int
    stride: int
    origins: list[tuple[int, int]]          # (x, y) top-left of each patch
    image_size: tuple[int, int]             # (width, height)

    @property
    def count(self) -> int:
        return len(self.origins)

    @property
    def overlap_px(self) -> int:
        return self.patch_size - self.stride

    @property
    def max_whole_object_px(self) -> int:
        """Largest object guaranteed to fall inside some tile complete.

        An object wider than the overlap can straddle a boundary with no tile
        containing all of it, so it may be counted twice or missed. Compare
        this against the object size you expect before trusting a tiled count.
        """
        return self.overlap_px


def patch_shape(image_size: tuple[int, int], patch_size: int) -> tuple[int, int]:
    """(width, height) the model is shown -- always the full patch.

    Every view is patch_size square, whatever the source measures. A source
    shorter than the patch is padded up to it (see :func:`pad_to_patch`) rather
    than sent at its own size, so the model's input geometry never varies.

    Kept as a function, and called by both composition and inference, because
    the whole scheme rests on the model seeing objects at the same size in both
    and two copies of the rule drift. They already did: an earlier version
    decided per plate rather than per axis, so a 900x700 source composed
    900x700 canvases while inference tiled it 784 wide.
    """
    return int(patch_size), int(patch_size)


def pad_to_patch(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Mirror-pad *arr* (HW or HWC) out to ``patch_size`` square.

    Mirrored, not black: a black border is a background no camera produces, and
    the detector would have to learn it as a feature. Mirroring extends the
    material that is already there.

    The same function pads composition canvases and inference tiles, so a
    padded frame looks the same to the model in training and in use. Padding is
    only safe because of that -- if only one side padded, the model would meet
    an input it had never seen and nothing would say so.
    """
    t = int(patch_size)
    h, w = arr.shape[:2]
    if h >= t and w >= t:
        return arr[:t, :t]
    out = np.empty((t, t) + arr.shape[2:], dtype=arr.dtype)
    # np.pad in reflect mode needs the pad width below the axis length; tile
    # the source instead so it works even when the pad exceeds the image.
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


def default_stride(patch_size: int, object_size: int | None = None) -> int:
    """Step between patches.

    default_patch_stride() by default, the same step the semantic sliding
    window takes.

    Given *object_size*, the step is tightened so the overlap clears the object
    with margin. That constraint is not cosmetic: predict_tiled discards views
    clipped by a tile edge, and when the overlap is narrower than the object
    almost every view is clipped by something. With the 3/4 rule at patch 384
    (96px overlap) and objects wider than that, most raw detections were
    discarded and the count came out low; widening the overlap past the object
    fixed it. Objects larger than the patch cannot be helped by any stride, so
    the step is floored rather than driven to zero.
    """
    base = default_patch_stride(patch_size)
    if not object_size or object_size <= 0:
        return base
    # A quarter of the object as margin, so a slightly larger-than-typical
    # instance still clears the seam.
    needed = patch_size - int(object_size * 1.25)
    # Never step less than half the patch. Tile count grows with the square of
    # the inverse step, so an object approaching the patch size would otherwise
    # ask for hundreds of tiles to buy an overlap it still cannot have: at
    # patch 384 on a 2560x2048 frame, a 300px object drives 456 tiles against
    # 63 for the plain rule, and it is still clipped. Past this point the patch
    # is simply too small for the object; max_whole_object_px reports that, and
    # a larger patch is the answer rather than more of a smaller one.
    floor = max(1, patch_size // 2)
    return max(floor, min(base, needed))


def plan_tiles(image_size: tuple[int, int], patch_size: int,
               stride: int | None = None, object_size: int | None = None) -> TilePlan:
    """Tile origins covering the image, clamped so no tile leaves the frame.

    The last row and column are pulled back to the edge rather than padded, so
    every tile is real image and the model never sees invented borders. That
    makes the final overlap larger than requested, never smaller.
    """
    w, h = image_size
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    stride = (default_stride(patch_size, object_size) if stride is None
              else max(1, int(stride)))

    def starts(extent: int) -> list[int]:
        if extent <= patch_size:
            return [0]
        pos = list(range(0, extent - patch_size + 1, stride))
        if pos[-1] != extent - patch_size:
            pos.append(extent - patch_size)
        return pos

    xs, ys = starts(w), starts(h)
    return TilePlan(patch_size=patch_size, stride=stride,
                    origins=[(x, y) for y in ys for x in xs],
                    image_size=(w, h))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two axis-aligned boxes given as (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def touches_tile_edge(
    boxes: np.ndarray, x: int, y: int, size: int, W: int, H: int,
    eps: float = 1.0,
) -> np.ndarray:
    """Which of *boxes* (full-frame coords) are cut by a seam of this tile.

    The frame border is exempt: there is no neighbouring tile to hold that
    side, so a box against it is not a clipped view of anything.

    One rule, two readers. It decides which views to discard when the overlap
    can hold the object whole, and which views are fragments to be stitched
    when it cannot -- and those two must be the same set or a fragment gets
    dropped by one rule and joined by the other.
    """
    touches = np.zeros(len(boxes), dtype=bool)
    if x > 0:
        touches |= boxes[:, 0] <= x + eps
    if y > 0:
        touches |= boxes[:, 1] <= y + eps
    if x + size < W:
        touches |= boxes[:, 2] >= x + size - eps
    if y + size < H:
        touches |= boxes[:, 3] >= y + size - eps
    return touches


#: Share of the smaller fragment that must be shared for two clipped views to
#: be one object. Two views of one object always overlap by the whole tile
#: overlap band -- a 767px object across a 192px band shares half its shorter
#: fragment -- while two genuinely adjacent objects share no pixels at all, so
#: anything above noise separates them. This is not a tuned number.
FRAGMENT_SHARE = 0.10


def merge_clipped_fragments(
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    clipped: Sequence[bool],
    min_share: float = FRAGMENT_SHARE,
) -> list[list[int]]:
    """Group views that are one object seen across a seam. Returns index groups.

    Needed only where no tile can hold the object whole. There, every view of
    it is clipped and dropping them loses the object outright: objects several
    times wider than the overlap came back as a small fraction of their number.

    The association is pixel overlap, not box IoU. Overlapping tiles mean two
    views of one object share the whole overlap band, while two objects lying
    side by side across a seam share nothing -- which is exactly the
    distinction box IoU cannot draw (0.19 to 0.70 for one screw, indexed to
    where the seam fell rather than to whether it was one object).

    Union-find, so an object crossing two seams joins into one group.
    """
    n = len(masks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Bounds first, pixels second. These masks are the whole frame -- 16 MB
    # each on a 4608x3456 photo -- and ANDing every pair of them outright is
    # tens of gigabytes of memory traffic for a question most pairs answer by
    # not overlapping at all. The box test settles those, and the pairs that
    # survive it are intersected only inside the rectangle they share.
    areas: list[int] = []
    boxes: list[tuple[int, int, int, int] | None] = []
    for m in masks:
        rows = np.any(m, axis=1)
        if not rows.any():
            areas.append(0)
            boxes.append(None)
            continue
        cols = np.any(m, axis=0)
        y0 = int(np.argmax(rows))
        y1 = len(rows) - int(np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = len(cols) - int(np.argmax(cols[::-1]))
        areas.append(int(np.count_nonzero(m[y0:y1, x0:x1])))
        boxes.append((x0, y0, x1, y1))

    for i in range(n):
        for j in range(i + 1, n):
            if int(class_ids[i]) != int(class_ids[j]):
                continue
            if not (clipped[i] or clipped[j]):
                continue
            smaller = min(areas[i], areas[j])
            if smaller <= 0:
                continue
            bi, bj = boxes[i], boxes[j]
            x0, y0 = max(bi[0], bj[0]), max(bi[1], bj[1])
            x1, y1 = min(bi[2], bj[2]), min(bi[3], bj[3])
            if x0 >= x1 or y0 >= y1:
                continue
            shared = int(np.count_nonzero(
                masks[i][y0:y1, x0:x1] & masks[j][y0:y1, x0:x1]))
            if shared / smaller >= min_share:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def merge_tile_detections(
    boxes: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int] | None = None,
    iou_threshold: float = 0.7,
) -> list[int]:
    """Greedy highest-confidence-first dedup of detections pooled from tiles.

    Returns the indices to keep. Duplicates arise because overlapping tiles, and
    different patch sizes, see the same object more than once; suppression is
    per class, so two different classes on the same spot both survive.

    Plain IoU, deliberately. An earlier version added a containment test to
    catch views clipped by a tile edge, which scored low IoU against the whole
    view. It made counting several times worse:
    an object with no unclipped view anywhere is seen as two halves that
    neither contain nor overlap each other, so both survive either way, while
    the extra rule started folding genuinely adjacent objects together. Clipped
    views are dropped at the source instead -- see predict_tiled.
    """
    if not len(boxes):
        return []
    arr = np.asarray(boxes, dtype=np.float64)
    conf = np.asarray(confidences, dtype=np.float64)
    cls = (np.asarray(class_ids, dtype=np.int64)
           if class_ids is not None else np.zeros(len(arr), dtype=np.int64))

    keep: list[int] = []
    for i in np.argsort(-conf):
        if any(cls[i] == cls[k] and _iou(arr[i], arr[k]) > iou_threshold for k in keep):
            continue
        keep.append(int(i))
    return sorted(keep)


# Tiles go through the model ONE AT A TIME, and that was measured, not assumed.
#
# A 4608x3456 photo at patch 768 is forty-eight forward passes, and issuing
# them singly leaves the GPU idle between each -- so handing the SDK a list
# looks free. It is not. On that photo, medium weights, threshold 0.35:
#
#     batch 1 twice   confidence identical to 0.0, mask areas identical
#     batch 1 vs 8    confidence differs by 0.065, mask area by 4.8%
#
# The model is perfectly deterministic at a fixed batch; changing the batch
# changes the answer. 0.065 is larger than one step of the calibration grid
# (0.05), so a detection can cross the shipped threshold on batch size alone.
# Serving runs the ONNX graph one tile at a time as well, so batching only the
# trainer would hand serving a threshold measured on numbers it cannot
# reproduce -- the divergence this file exists to prevent.
#
# The saving was 3.42s -> 2.58s. Not worth it.


def iter_tile_detections(
    image,
    predict: Callable[[object], tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    patch_size: int | Sequence[int],
    stride: int | None = None,
    keep_clipped: bool = False,
):
    """Yield one tile's surviving detections at a time, in full-frame coords.

    Yields ``(origin, boxes, confidences, class_ids, extras)`` in full-frame
    coordinates. *predict* may return a fourth element -- any sequence with one
    entry per detection, such as masks -- and the matching entries come back in
    ``extras``, already filtered to the views that survived. That keeps a caller
    carrying its own per-detection data from having to re-derive which rows were
    kept, and from depending on when the callback happened to be invoked.

    This is the single description of the tiling geometry and of which views are
    discarded. Both the box-only path and the mask-carrying inference path go
    through it: they used to each implement it, and drifted -- the inference
    copy never dropped clipped views, so on a large photo of many small objects
    it counted nearly twice what this one did.
    """
    sizes = [int(patch_size)] if isinstance(patch_size, int) else [int(p) for p in patch_size]
    W, H = image.size

    for size in sizes:
        # Stride scales with the patch, so every size keeps the same overlap
        # fraction and the same guarantee about what it can see whole.
        size_stride = (None if stride is None
                       else max(1, int(round(stride * size / sizes[0]))))
        sub = plan_tiles((W, H), size, size_stride)
        for x, y in sub.origins:
            # Clamped to the frame. An image smaller than the patch yields one
            # tile, and cropping past the edge would hand the model a
            # black-padded border it has never seen -- 57% of the frame for a
            # 512px image at patch 784. Clamping makes that case the whole
            # image, resized by the model exactly as it always was, so a patch
            # larger than the source degrades to the single-pass behaviour
            # rather than to garbage.
            crop = image.crop((x, y, min(x + size, W), min(y + size, H)))
            if crop.size != (size, size):
                # Source shorter than the patch: pad exactly as composition
                # does, so the model meets the same frame in both.
                from PIL import Image as _Image
                crop = _Image.fromarray(pad_to_patch(np.asarray(crop), size))
            yield from _tile_rows(predict(crop), x, y, size, W, H, keep_clipped)


def _tile_rows(out, x, y, size, W, H, keep_clipped):
    """One tile's detections, offset into full-frame coords.

    Yields at most once. Split out of the tiling loop so batching the forward
    passes did not have to reindent the rule about which views survive.
    """
    boxes, conf, cls = out[0], out[1], out[2]
    extras = out[3] if len(out) > 3 else None
    if boxes is None or len(boxes) == 0:
        return
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    boxes[:, [0, 2]] += x
    boxes[:, [1, 3]] += y
    conf_arr = np.asarray(conf, dtype=np.float64).ravel()
    cls_arr = (np.asarray(cls).ravel().astype(np.int64)
               if cls is not None else np.zeros(len(boxes), dtype=np.int64))

    # Drop detections clipped by a tile edge.
    #
    # An object straddling a seam is seen whole by the tile containing it and
    # cut off by its neighbour. The two boxes describe different rectangles, so
    # their IoU is low, and varies with where the seam falls, and no IoU
    # threshold both folds them together and keeps adjacent objects apart. The
    # clipped view carries nothing the whole view lacks, so it goes. The frame
    # border is exempt: there is no neighbouring tile to hold that side.
    #
    # That reasoning rests on "seen whole by the tile containing it". An object
    # longer than the overlap has no unclipped view anywhere, and dropping then
    # loses the object rather than a redundant view of it -- so the caller that
    # knows the object size can ask for the fragments instead and stitch them
    # (merge_clipped_fragments).
    touches = touches_tile_edge(boxes, x, y, size, W, H)
    local = (np.arange(len(boxes)) if keep_clipped
             else np.nonzero(~touches)[0])
    if local.size == 0:
        return
    kept_extras = None if extras is None else [extras[int(k)] for k in local]
    yield (x, y), boxes[local], conf_arr[local], cls_arr[local], kept_extras


def predict_tiled(
    image,
    predict: Callable[[object], tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    patch_size: int | Sequence[int],
    stride: int | None = None,
    iou_threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TilePlan]:
    """Run *predict* over tiles of *image* and merge the results.

    Args:
        image: a PIL image; cropped, never resized.
        predict: called with one tile, returns ``(boxes_xyxy, confidences,
            class_ids)`` in tile coordinates. ``class_ids`` may be None.
        patch_size: one size, or several to sweep. Several makes this a scale
            pyramid: each size is cropped and handed to the model, which fits
            it to its own input, so a 2x larger patch shows every object at
            half the size. Objects too large for the small patch and too small
            for the large one are both reached, which matters because a project
            does not announce its object size in advance -- the failure mode of
            a single size is that the model simply never sees the object at a
            size it can characterise.
        stride: step between patches; defaults to patch_size * 3 // 4, applied
            per size so a larger patch takes correspondingly larger steps.
        iou_threshold: box IoU above which two detections are the same object.

    Returns:
        ``(boxes_xyxy, confidences, class_ids, plan)`` in full-frame
        coordinates, deduplicated.
    """
    sizes = [int(patch_size)] if isinstance(patch_size, int) else [int(p) for p in patch_size]
    plan = plan_tiles(image.size, sizes[0], stride)
    all_boxes: list[list[float]] = []
    all_conf: list[float] = []
    all_cls: list[int] = []
    for _origin, boxes, conf, cls, _extras in iter_tile_detections(
            image, predict, patch_size, stride):
        all_boxes.extend(boxes.tolist())
        all_conf.extend(float(c) for c in conf)
        all_cls.extend(int(c) for c in cls)

    keep = merge_tile_detections(all_boxes, all_conf, all_cls, iou_threshold)
    if not keep:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64), plan)
    idx = np.asarray(keep, dtype=np.int64)
    return (np.asarray(all_boxes, dtype=np.float64)[idx],
            np.asarray(all_conf, dtype=np.float64)[idx],
            np.asarray(all_cls, dtype=np.int64)[idx],
            plan)


def predict_tiled_masks(
    image,
    predict: Callable[[object], tuple],
    patch_size: int,
    stride: int | None = None,
    iou_threshold: float = 0.7,
    object_span_px: int | None = None,
) -> tuple[list, list[float], list[int], TilePlan]:
    """Tiled prediction that keeps each detection's mask, in full-frame space.

    *predict* is called with one tile and returns
    ``(boxes_xyxy, confidences, class_ids, masks)`` in tile coordinates. Masks
    come back pasted at their true position, so callers that measure area,
    centroids or mask IoU work on the whole frame.

    Inference and the training-time threshold calibration both go through this.
    They must: the threshold is chosen by counting validation photos, and if
    calibration counted them a different way than inference will, the number it
    picks is right for a pipeline that never runs.

    *object_span_px* is the longest annotated object, in source pixels. Given
    it, and only when it exceeds what the tiling can hold whole, the clipped
    views are kept and stitched instead of dropped -- see
    merge_clipped_fragments. Left None the behaviour is exactly what it was.
    """
    W, H = image.size
    masks: list = []
    boxes: list = []
    confs: list[float] = []
    classes: list[int] = []
    clipped: list[bool] = []
    plan = plan_tiles((W, H), int(patch_size), stride)
    stitch = (object_span_px is not None
              and int(object_span_px) > plan.max_whole_object_px)
    for (x, y), b, conf, cls, tile_masks in iter_tile_detections(
            image, predict, int(patch_size), stride, keep_clipped=stitch):
        cut = touches_tile_edge(np.asarray(b, dtype=np.float64).reshape(-1, 4),
                                x, y, int(patch_size), W, H)
        for j, tile_mask in enumerate(tile_masks or []):
            full = np.zeros((H, W), dtype=bool)
            th, tw = tile_mask.shape
            full[y:y + th, x:x + tw] = tile_mask[: H - y, : W - x]
            ys, xs = np.nonzero(full)
            if ys.size == 0:
                continue
            masks.append(full)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            confs.append(float(conf[j]))
            classes.append(int(cls[j]))
            clipped.append(bool(cut[j]) if j < len(cut) else False)
    if stitch and masks:
        merged_masks: list = []
        merged_boxes: list = []
        merged_confs: list[float] = []
        merged_classes: list[int] = []
        for group in merge_clipped_fragments(masks, classes, clipped):
            union = masks[group[0]].copy()
            for i in group[1:]:
                union |= masks[i]
            ys, xs = np.nonzero(union)
            best = max(group, key=lambda i: confs[i])
            merged_masks.append(union)
            merged_boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            # The most confident view of the object speaks for it: a fragment
            # is a partial view, and averaging would penalise an object for
            # having been cut in more places.
            merged_confs.append(confs[best])
            merged_classes.append(classes[best])
        masks, boxes, confs, classes = (
            merged_masks, merged_boxes, merged_confs, merged_classes)
    keep = merge_tile_detections(boxes, confs, classes, iou_threshold)
    return ([masks[i] for i in keep], [confs[i] for i in keep],
            [classes[i] for i in keep], plan)


def sdk_tile_predict(model, threshold: float):
    """Adapt an RF-DETR-style model to the *predict* callback above."""
    def predict(crop):
        det = model.predict(crop, threshold=threshold)
        n = 0 if det.mask is None else len(det.mask)
        if not n:
            return np.zeros((0, 4)), np.zeros(0), None, []
        return (np.asarray(det.xyxy, dtype=np.float64).reshape(-1, 4),
                np.asarray(det.confidence, dtype=np.float64),
                np.asarray(det.class_id) if det.class_id is not None else None,
                [np.asarray(m).astype(bool) for m in det.mask])
    return predict
