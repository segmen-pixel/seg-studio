# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The arithmetic of the counting recipe, for the MCP bridge's recipe tools.

A language model can decide what to look at and whether a box is the
object; it cannot compute an acceptance band from teacher masks, judge
whether a SAM level fits it, or union masks. Those parts live here, as
plain functions over numpy arrays, so the bridge can offer them as tools
(teacher_band, accept_mask, write_kept) and any MCP client -- a local vision
model or a hosted one -- runs the same recipe the playbook describes.

Every threshold here was measured; the numbers and the reasons are in
scripts/mcp_playbook/counting_from_teacher.md. Keep them together.
"""
from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image
from scipy import ndimage

#: The teacher's observed range, widened by this on both sides.
SLACK = 1.8
#: Fewer teachers' objects than this, and they have not shown what the thing
#: can be -- only what these few were. teacher_band tells the model so; the
#: band leaves out its lower limit on size until there are this many.
FEW_TEACHERS = 20
#: Two masks are the same object when their overlap exceeds this share of
#: the smaller one.
OVERLAP = 0.55
#: Erosions tried when calibrating the shrink on the teachers. The numbers are
#: pixels of SAM's own working image, which is 1024 on its longest side: SAM
#: resizes whatever it is given to that before it looks, so the edge it draws
#: is out by about the same amount there whatever the picture's real size. On
#: a 512 px frame these are pixels as they stand; on a photograph several times
#: that size each one is several real pixels, and a table built in ones and twos could only ever
#: answer zero -- seconds of full-frame erosions to learn nothing.
ERODE_CANDIDATES = (0, 1, 2, 3, 4)
#: What SAM resizes its input to; the scale that turns the candidates above
#: into pixels of the picture in hand.
SAM_SIDE = 1024
#: How much better than not shrinking a candidate has to score before it is
#: taken. The objective always names a winner, including one that won inside
#: the rounding: with one teacher holding one object, the answer can move from
#: 0 to 1 px when the teacher is redrawn, with nothing in this file changed in
#: between. A shrink worth taking wins by little more than this, on the
#: teachers it was measured on and on ones held out alike, and even such a win
#: can change sides between segmenters, so the floor goes under it. Above
#: 0.088 the large-photograph case of the calibration test answers 0 where it
#: should answer 5.
SHRINK_MARGIN = 0.005


def erosion_candidates(long_side: int = 0) -> list[int]:
    """The candidates above, in pixels of a picture whose longest side is this."""
    scale = max(1.0, float(long_side) / SAM_SIDE) if long_side else 1.0
    # int(x + 0.5), not round(): round(4.5) is 4 in Python, and a candidate
    # table that depends on which side of even a half falls on is a puzzle.
    return sorted({int(e * scale + 0.5) for e in ERODE_CANDIDATES})


def decode_mask(png_b64: str) -> np.ndarray:
    """Class-id array from a base64 PNG; a palette or RGB(A) PNG reads channel 0."""
    a = np.array(Image.open(io.BytesIO(base64.b64decode(png_b64.split(",")[-1]))))
    return a[..., 0] if a.ndim == 3 else a


def encode_mask(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "L").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode_gray(image_b64: str) -> np.ndarray:
    im = Image.open(io.BytesIO(base64.b64decode(image_b64.split(",")[-1]))).convert("L")
    return np.array(im).astype(np.float32)


def foreground(ids: np.ndarray) -> np.ndarray:
    """Any painted class counts; 0 is background and 255 is ignore."""
    return (ids != 0) & (ids != 255)


#: A floor is kept only if it turns away most of the background it is shown.
#: Not "do the objects measure higher on average" -- they can, by a wide
#: margin, and the floor still admit most of the background once the teachers
#: cover both bright and dim frames, because the floor is set by the dimmest
#: object and that one is dimmer than most of the background.
#: The only question worth asking is the one the floor exists to answer.
MAX_BACKGROUND_ADMITTED = 0.3


def frame_view(gray: np.ndarray) -> dict:
    """Everything about a picture that judging a mask needs, computed once.

    The gradient of a large photograph is tens of MB and was being taken again
    for every proposal -- dozens of times on one image -- to answer a question
    about a few thousand pixels.
    """
    if isinstance(gray, dict):
        return gray
    g = gray.astype(np.float32)
    gy, gx = np.gradient(g)
    grad = np.hypot(gy, gx)
    return {"gray": gray, "grad": grad, "std": float(gray.std()), "edge": float(grad.mean())}


def appearance(view: np.ndarray | dict, m: np.ndarray) -> tuple[float, float]:
    """(contrast, texture) under a mask: grey std and mean gradient magnitude.

    Size alone cannot tell an object from an object-sized patch of the surface
    it lies on; the gradient can: a textured object's mean gradient can sit
    several times above that of a plain surface.
    """
    v = frame_view(view)
    return float(v["gray"][m].std()), float(v["grad"][m].mean())


def background_samples(view: np.ndarray | dict, fg: np.ndarray, shape: tuple,
                       count: int = 6) -> list[tuple[float, float]]:
    """Patches of this picture the person left unpainted, the size of an object.

    What the floors are measured against: a floor that admits these is not
    telling the object from the surface it sits on.
    """
    v = frame_view(view)
    h, w = v["gray"].shape[:2]
    bh, bw = max(4, int(shape[0])), max(4, int(shape[1]))
    if bh >= h or bw >= w:
        return []
    rng = np.random.default_rng(0)
    out = []
    for _ in range(count * 8):
        if len(out) >= count:
            break
        y, x = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
        if fg[y:y + bh, x:x + bw].mean() > 0.02:
            continue
        patch = np.zeros_like(fg, dtype=bool)
        patch[y:y + bh, x:x + bw] = True
        sd, ed = appearance(v, patch)
        out.append((sd / v["std"] if v["std"] else 0.0, ed / v["edge"] if v["edge"] else 0.0))
    return out


#: A speck is this share of the largest thing in the mask. The share is taken
#: of the largest and not of the middle one, because the middle one moves: a
#: mask of one object and two specks already has a speck for its median, and
#: the floor set from there is the 4 px minimum, which keeps every speck. That
#: is how one object in one blob came back as several -- one large object
#: beside a few crumbs, every crumb counted an object, in_pieces fired
#: on every write, and the redo it asked for could not help. Adding specks
#: cannot change the largest, so the count of the real objects holds still.
SPECK_SHARE = 0.02
#: The blobs kept have to be this much of what was written for the union to be
#: one object apiece. Blobs alone cannot say it: an object beside one crumb is
#: two blobs and is not a split, a head and a shaft of a size are two blobs and
#: are.
WHOLE_SHARE = 0.95


def _speck_floor(sizes: np.ndarray) -> float:
    """The area under which a blob is a speck a brush left, not an object.

    Relative to the objects, not the frame: a small object can be a tiny
    share of its frame, and a frame-relative floor threw all of them away.
    """
    big = sizes[sizes >= 4]
    return max(4.0, SPECK_SHARE * float(big.max())) if big.size else 4.0


def components(mask: np.ndarray, gray: np.ndarray | dict | None = None,
               want_masks: bool = False) -> list[dict]:
    """The objects in a teacher mask, ignoring the specks a brush leaves.

    The speck floor is relative to the objects, not the frame: a small object
    can be a tiny share of its frame, and a frame-relative floor threw all of them away.
    """
    lbl, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = np.asarray(ndimage.sum(mask, lbl, range(1, n + 1)))
    floor = _speck_floor(sizes)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        if sl is None:
            continue
        comp = lbl == i
        a = int(comp.sum())
        if a < floor:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        rec = {"area": a, "w": w, "h": h, "frac": a / mask.size,
               "wfrac": w / mask.shape[1], "hfrac": h / mask.shape[0],
               "bbox": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)]}
        if want_masks:
            rec["mask"] = comp
        dt = ndimage.distance_transform_edt(np.pad(comp[sl], 1))[1:-1, 1:-1]
        cy, cx = np.unravel_index(int(dt.argmax()), dt.shape)
        rec["deepest"] = [int(sl[1].start + cx), int(sl[0].start + cy)]
        if gray is not None:
            v = frame_view(gray)
            rec["std"], rec["edge"] = appearance(v, comp)
            # Also as a share of the picture's own contrast and texture, which
            # is what carries from a bright frame to a dim one: the same objects
            # in dim light measure half the contrast and are the same objects.
            rec["std_rel"] = rec["std"] / v["std"] if v["std"] else 0.0
            rec["edge_rel"] = rec["edge"] / v["edge"] if v["edge"] else 0.0
        out.append(rec)
    return out


def reference_band(objs: list[dict], slack: float = SLACK,
                   background: list[tuple[float, float]] | None = None) -> dict:
    """An accept/reject band around what the teachers actually drew.

    The full observed range widened by the slack, not a percentile band: a
    p10-p90 band called the one long part in each frame an outlier and
    threw away most of an image. The band rejects what is nothing like the
    object (the surface it lies on, the frame: many times the median), not
    the unusual one.

    The slack grows when there are few objects to have seen. The widest
    thing eight objects show is not the widest thing there is, and a fixed
    multiple treats it as though it were: from teachers that held only a
    few objects, the band's top came out well below the largest
    the same person drew elsewhere in the project -- several of their own
    objects outside a band drawn from their own hand. Everything the recipe
    then wrote came back about half the size the person drew. 1 + 1/sqrt(n) is the
    correction: half again as wide at eight objects, a tenth wider at a
    hundred, and the fixed multiple once there are enough to have seen the
    range.
    """
    if not objs:
        raise ValueError("no objects in the teacher masks after speck removal")
    fr = np.array([o["frac"] for o in objs])
    wf = np.array([o["wfrac"] for o in objs])
    hf = np.array([o["hfrac"] for o in objs])

    thin = 1.0 + 1.0 / math.sqrt(max(1, len(objs)))
    thin_slack = slack * thin
    # No lower limit on size until there are enough teachers to have shown
    # what small can mean. A few are a few examples of one kind of thing, and
    # something smaller than all of them may be a smaller kind rather than a
    # piece of one: with a single large object as the one teacher, a smaller
    # object of another kind, only a little under the teacher's lower limit,
    # was turned away by it. Which it is, is for the model to look at. The
    # upper limit stays: far larger than anything drawn is what a box that
    # took in what the object lies in looks like.
    few = len(objs) < FEW_TEACHERS

    def band(v):
        return (0.0 if few else float(v.min()) / thin_slack), float(v.max()) * thin_slack

    b = {"frac": band(fr), "wfrac": band(wf), "hfrac": band(hf),
         "median_frac": float(np.median(fr)), "n": len(objs),
         "slack": round(thin_slack, 2)}
    if all("edge_rel" in o for o in objs):
        # As a share of the picture's own texture and contrast, so a floor set
        # on brightly lit frames still means something on dim ones, where the
        # same objects measure half the contrast and are the same objects.
        # With the same allowance for having seen little that the size band
        # has. A floor is the dimmest object the teachers drew, and with one
        # teacher that is its one object on its one picture: the same object
        # photographed again from another angle measured below the floor set
        # on the first photograph, so every mask of the whole object was turned
        # away and only a part of it got over -- and was written. What a mask
        # is has to be
        # judged somewhere, and a threshold drawn from one example is the
        # wrong place; the review picture the model is shown after each write
        # is the right one.
        floors = {"edge_min_rel": float(min(o["edge_rel"] for o in objs)) / (1.5 * thin),
                  "std_min_rel": float(min(o["std_rel"] for o in objs)) / (1.3 * thin)}
        notes = {}
        if background:
            # Try each floor on the background of the very frames it was set
            # on. A floor that lets that through is not keeping anything out,
            # and all it can do from there is turn away real objects.
            for key, idx, name in (("edge_min_rel", 1, "texture"), ("std_min_rel", 0, "contrast")):
                floor = floors[key]
                admitted = sum(1 for bg in background if bg[idx] >= floor)
                share = admitted / len(background)
                if share > MAX_BACKGROUND_ADMITTED:
                    floors.pop(key)
                    notes[name] = (f"not used: it would admit {admitted} of {len(background)} "
                                   f"background patches from the teachers' own frames, so it turns "
                                   f"away real objects without keeping the background out")
                else:
                    notes[name] = (f"admits {admitted} of {len(background)} background patches from "
                                   f"the teachers' own frames")
        b.update(floors)
        if notes:
            b["appearance_notes"] = notes
    elif all("edge" in o for o in objs):        # a band from before the change
        b["edge_min"] = float(min(o["edge"] for o in objs)) / 1.5
        b["std_min"] = float(min(o["std"] for o in objs)) / 1.3
    return b


def accepts(m: np.ndarray, band: dict, gray: np.ndarray | dict | None = None) -> tuple[bool, str]:
    """Is this proposal the same kind of thing the teachers drew?"""
    a = int(m.sum())
    if a == 0:
        return False, "empty"
    frac = a / m.size
    lo, hi = band["frac"]
    if not (lo <= frac <= hi):
        return False, f"area {100 * frac:.2f}% outside {100 * lo:.2f}-{100 * hi:.2f}%"
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = np.asarray(ndimage.sum(m, lbl, range(1, n + 1)))
        if sizes.max() / a < 0.6:
            return False, f"fragmented into {n}"
    ys, xs = np.where(m)
    wf = (xs.max() - xs.min() + 1) / m.shape[1]
    hf = (ys.max() - ys.min() + 1) / m.shape[0]
    lo, hi = band["wfrac"]
    if not (lo <= wf <= hi):
        return False, f"width {100 * wf:.1f}% outside {100 * lo:.1f}-{100 * hi:.1f}%"
    lo, hi = band["hfrac"]
    if not (lo <= hf <= hi):
        return False, f"height {100 * hf:.1f}% outside {100 * lo:.1f}-{100 * hi:.1f}%"
    if gray is not None and ("edge_min_rel" in band or "std_min_rel" in band or "edge_min" in band):
        v = frame_view(gray)
        sd, ed = appearance(v, m)
        if "edge_min" in band:                  # a band from before the change
            if ed < band["edge_min"]:
                return False, f"texture {ed:.1f} below {band['edge_min']:.1f}"
            if sd < band["std_min"]:
                return False, f"contrast {sd:.1f} below {band['std_min']:.1f}"
        else:
            ed_rel = ed / v["edge"] if v["edge"] else 0.0
            sd_rel = sd / v["std"] if v["std"] else 0.0
            if "edge_min_rel" in band and ed_rel < band["edge_min_rel"]:
                return False, (f"texture {ed_rel:.2f} of this picture's own, below "
                               f"{band['edge_min_rel']:.2f}")
            if "std_min_rel" in band and sd_rel < band["std_min_rel"]:
                return False, (f"contrast {sd_rel:.2f} of this picture's own, below "
                               f"{band['std_min_rel']:.2f}")
    return True, "ok"


def where_to_add(mask: np.ndarray, box: list[int]) -> list[int] | None:
    """A point on the part of the object the mask has not taken yet.

    One point on a part made of a head and a shaft returns the head, or the shaft: they differ in
    brightness and finish, and SAM answers for the part the point is on. What
    fixes it is a second point on the part that is missing, which is what a
    person does by hand -- and what this finds, as the middle of the largest
    unclaimed piece of the box.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    free = ~mask[y0:y1, x0:x1]
    if not free.any():
        return None
    lbl, n = ndimage.label(free)
    if n == 0:
        return None
    sizes = np.asarray(ndimage.sum(free, lbl, range(1, n + 1)))
    which = int(sizes.argmax()) + 1
    if sizes[which - 1] < 0.05 * free.size:
        return None                      # nothing worth a second look
    piece = lbl == which
    # the deepest point of that piece, so the new point is not on its edge
    dt = ndimage.distance_transform_edt(np.pad(piece, 1))[1:-1, 1:-1]
    cy, cx = np.unravel_index(int(dt.argmax()), dt.shape)
    return [int(x0 + cx), int(y0 + cy)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """How much two masks agree: the shared area over the area either covers."""
    inter = int((a & b).sum())
    uni = int((a | b).sum())
    return inter / uni if uni else 0.0


def covers(m: np.ndarray, objs: list[dict], share: float = 0.3) -> int:
    """How many of these objects a mask has swallowed at least this much of.

    A mask that answers for two objects at once is a different failure from a
    mask that is the wrong size, and the band cannot see it: on parts lying
    against each other, the masks came back as far more blobs than the
    separate parts the person had drawn.
    """
    n = 0
    for o in objs:
        om = o.get("mask")
        if om is None:
            continue
        a = int(om.sum())
        if a and int((m & om).sum()) / a >= share:
            n += 1
    return n


def overlaps(m: np.ndarray, kept: list[np.ndarray], share: float = OVERLAP) -> bool:
    a = int(m.sum())
    return any(int((m & d).sum()) / max(1, min(a, int(d.sum()))) > share for d in kept)


def choose(cands: list[np.ndarray], kept: list[np.ndarray]) -> np.ndarray | None:
    """The largest passing level, or on overlap the next smaller one that
    does not overlap -- the largest level sometimes covers two touching
    objects."""
    for m in sorted(cands, key=lambda c: -int(c.sum())):
        if not overlaps(m, kept):
            return m
    return None


def shrink(m: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return m
    return ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=px, border_value=0)


def calibrate_erosion(pairs: list[tuple[np.ndarray, list[np.ndarray]]],
                      long_side: int = 0) -> tuple[int, dict[int, float]]:
    """How much wider than this person's brush does SAM paint?

    pairs: (hand mask, SAM masks from each hand object's own box) per teacher.
    Scores the union against the hand mask after each candidate erosion and
    returns the best; 0 is a valid answer. A
    candidate has to beat not shrinking by SHRINK_MARGIN to be taken: when the
    objects are large and smooth the differences live inside the rounding, and
    the winner then turns on how the brush was held rather than on SAM.

    long_side is the picture's longest side, which sets what the candidates
    mean: SAM's error is roughly constant in its own 1024 px working image, so
    on a bigger picture the same error is more pixels.
    """
    candidates = erosion_candidates(long_side)
    scores: dict[int, list[float]] = {e: [] for e in candidates}
    for hand, sams in pairs:
        if not sams:
            # The segmenter answered for none of this teacher's objects, so it
            # has nothing to say about how wide it paints. Averaging its zeros
            # in scales every candidate down alike: the winner does not move,
            # but the margin below is a difference and would shrink with them,
            # so a project the segmenter often fails would quietly be held to
            # a stricter floor than one it does not.
            continue
        for e in candidates:
            u = np.zeros_like(hand, dtype=bool)
            for s in sams:
                u |= shrink(s, e)
            inter = int((u & hand).sum())
            uni = int((u | hand).sum())
            scores[e].append(inter / uni if uni else 0.0)
    table = {e: float(np.mean(v)) for e, v in scores.items() if v}
    if not table:
        return 0, {}          # nothing was measured, which an empty table says
                              # and a measured 0 does not
    best = max(table, key=lambda e: (table[e], -e))
    # What the winner is worth against not shrinking at all. 0 is always among
    # the candidates -- int(0 * scale + 0.5) is 0 whatever the picture's size
    # -- and it is named here rather than taken as min(table) so that dropping
    # it from the candidates breaks this loudly instead of quietly changing
    # what the comparison is against.
    if best != 0 and 0 in table and table[best] - table[0] < SHRINK_MARGIN:
        return 0, table
    return int(best), table


def union(masks: list[np.ndarray], erode_px: int | list[int] = 0) -> np.ndarray:
    """The masks as one, each shrunk first -- by one amount, or by its own when given a list."""
    if not masks:
        raise ValueError("nothing to union")
    u = np.zeros_like(masks[0], dtype=bool)
    for i, m in enumerate(masks):
        u |= shrink(m, int(erode_px[i]) if isinstance(erode_px, (list, tuple)) else erode_px)
    return u


#: Matching works on a copy this wide at most. A large photograph costs more
#: to slide a template across than the answer is worth, and the objects are
#: still tens of pixels here.
MATCH_SIDE = 1600
#: Cut-outs are taken from the teachers, but not all of them: a hundred
#: near-identical objects cost a hundred sweeps and find the same things.
#: They are sorted by size and sampled across the range.
MAX_TEMPLATES = 8
#: A full turn, because a part lies whichever way it fell.
ANGLE_STEP = 20
#: Two proposals are the same find when their boxes overlap this much.
BOX_OVERLAP = 0.4


def _gradient(gray: np.ndarray) -> np.ndarray:
    """The edges of a picture, which is what a cut-out is recognisable by.

    Matching on brightness put the flat background above every object:
    masked template matching offers only TM_CCORR_NORMED, which does not
    subtract the mean, so a bright empty region beats a dark textured one. On
    the gradient the same templates found every object.
    """
    import cv2
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def cut_outs(pairs: list[tuple[np.ndarray, np.ndarray]], limit: int = MAX_TEMPLATES) -> list[dict]:
    """Teacher objects as cut-outs: the picture under each mask, and the mask.

    pairs is (gray, class-id mask) per teacher image. Objects are sampled
    across the size range rather than taken in order, so one sweep covers the
    long part and the short one.
    """
    objs: list[dict] = []
    for gray, ids in pairs:
        fg = foreground(ids)
        lbl, n = ndimage.label(fg)
        if n == 0:
            continue
        sizes = np.asarray(ndimage.sum(fg, lbl, range(1, n + 1)))
        floor = _speck_floor(sizes)
        for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
            if sl is None:
                continue
            comp = lbl[sl] == i
            if comp.sum() < floor or comp.shape[0] < 4 or comp.shape[1] < 4:
                continue
            objs.append({"gray": gray[sl], "mask": comp, "area": int(comp.sum())})
    if not objs:
        return []
    objs.sort(key=lambda o: o["area"])
    if len(objs) <= limit:
        return objs
    step = (len(objs) - 1) / float(limit - 1) if limit > 1 else 1
    return [objs[int(i * step + 0.5)] for i in range(limit)]


def propose_boxes(gray: np.ndarray, templates: list[dict], *, max_boxes: int = 60,
                  match_side: int = MATCH_SIDE, angle_step: int = ANGLE_STEP,
                  score_floor: float = 0.35) -> list[dict]:
    """Where the teachers' own objects turn up again in this picture.

    Returns boxes in the picture's pixels, best first, each with the score and
    the angle that found it. They are proposals, not answers: SAM and the band
    still judge every one of them.
    """
    import cv2
    if not templates or gray.size == 0:
        return []
    f = min(1.0, match_side / float(max(gray.shape)))
    small = cv2.resize(gray, (max(1, int(gray.shape[1] * f)), max(1, int(gray.shape[0] * f))),
                       interpolation=cv2.INTER_AREA) if f < 1.0 else gray
    target = _gradient(small)
    found: list[dict] = []
    for t in templates:
        th, tw = t["mask"].shape[:2]
        tw2, th2 = max(4, int(round(tw * f))), max(4, int(round(th * f)))
        if tw2 >= target.shape[1] or th2 >= target.shape[0]:
            continue
        patch = cv2.resize(t["gray"], (tw2, th2), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(t["mask"].astype(np.uint8), (tw2, th2), interpolation=cv2.INTER_NEAREST)
        for angle in range(0, 360, angle_step):
            rot = cv2.getRotationMatrix2D((tw2 / 2.0, th2 / 2.0), angle, 1.0)
            cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
            nw, nh = int(th2 * sin + tw2 * cos), int(th2 * cos + tw2 * sin)
            rot[0, 2] += nw / 2.0 - tw2 / 2.0
            rot[1, 2] += nh / 2.0 - th2 / 2.0
            rp = cv2.warpAffine(patch, rot, (nw, nh))
            rm = cv2.warpAffine(mask, rot, (nw, nh), flags=cv2.INTER_NEAREST)
            if nw >= target.shape[1] or nh >= target.shape[0] or rm.sum() < 8:
                continue
            res = cv2.matchTemplate(target, _gradient(rp), cv2.TM_CCORR_NORMED, mask=rm.astype(np.float32))
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            ys, xs = np.where(res >= score_floor)
            for y, x in zip(ys, xs):
                found.append({"score": float(res[y, x]), "angle": angle,
                              "box": [x / f, y / f, (x + nw) / f, (y + nh) / f]})
    found.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for cand in found:
        if len(kept) >= max_boxes:
            break
        if not any(_box_overlap(cand["box"], k["box"]) > BOX_OVERLAP for k in kept):
            kept.append(cand)
    for k in kept:
        k["box"] = [int(round(v)) for v in k["box"]]
        k["score"] = round(k["score"], 4)
    return kept


def _box_overlap(a: list[float], b: list[float]) -> float:
    """Intersection over the smaller box: two finds of one object overlap a lot."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _blob_sizes(mask: np.ndarray) -> np.ndarray:
    """The areas of a mask's connected parts, largest first."""
    lbl, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(0, dtype=float)
    return np.sort(np.asarray(ndimage.sum(mask, lbl, range(1, n + 1)), dtype=float))[::-1]


def count(mask: np.ndarray) -> int:
    """How many objects a mask holds, by the floor components() uses.

    Counting raw labels had write_kept tell the model that a few objects were
    many blobs, when most of those were specks a few pixels across.
    in_pieces fired on the noise, so the redo it asked for was the wrong
    instruction and none of the images it named were redone. Counted by the
    floor, the specks drop out, and most of those in_pieces flags with them.
    """
    sizes = _blob_sizes(mask)
    if not sizes.size:
        return 0
    return int((sizes >= _speck_floor(sizes)).sum())


def in_pieces(mask: np.ndarray, n_kept: int) -> tuple[bool, int]:
    """Whether a union holds more objects than were kept -- by area, not count.

    Returns (in_pieces, blobs), so that the caller reports the same number it
    judged on. The old test was blobs > 1.5 * n_kept, which on a project that
    keeps one object an image means blobs >= 2: a single crumb beside the
    object called the write a split, asked for a redo that could not help it,
    and kept the frame out of settled for good, because settled is only
    reached by a write that is not in pieces.
    """
    blobs = count(mask)
    if n_kept <= 0 or not blobs:
        return False, blobs
    sizes = _blob_sizes(mask)
    total = float(sizes.sum())
    if not total:
        return False, blobs
    return bool(float(sizes[:n_kept].sum()) / total < WHOLE_SHARE), blobs


def objects_touching(mask: np.ndarray, other: np.ndarray) -> tuple[int, int]:
    """How many blobs a mask holds, and how many of them touch another mask.

    Every blob counts, however small: on a picture of specks the objects are
    the size of what components() drops as a brush's crumbs. And one label
    image, not a full-frame mask per object, keeps a frame of hundreds of them
    to a single array.
    """
    lbl, n = ndimage.label(mask)
    touched = np.unique(lbl[np.logical_and(mask, other)])
    return int(n), int((touched > 0).sum())


# ---------------------------------------------------------------------------
# The picture of SAM's answers for a box
# ---------------------------------------------------------------------------
#: A model asked for a rung by name -- whole, again and again in a run, where whole
#: was what the object lay in -- because a name and a table of areas was all it
#: had. These draw the answers themselves, side by side, so it can choose by
#: looking.
#: The language is the review picture's (loop._render): what is not the answer
#: is dimmed, its edge drawn white just outside it. Brightness reads on any
#: picture and to any eye; a tint was tried there and read as the colour of
#: what the object lay in.
#:
#: The value is a copy: loop.REVIEW_DIM (apps/trainer_api/app/core/vlm_agent/
#: loop.py) is the one to change. The bridge and the loop are separate
#: processes and neither imports the other, so a test holds the two equal.
SHEET_DIM = 0.35
#: The white edge, in pixels of the panel as the model sees it.
SHEET_EDGE_PX = 3
#: The model's own box. Vermilion, as on every picture it is shown.
SHEET_INK = (213, 94, 0)
SHEET_BG = (24, 24, 24)
SHEET_GAP = 8
#: One box: panels this big. Several boxes in one picture: the smaller size.
SHEET_PANEL = 320
SHEET_PANEL_SMALL = 224
#: The close-up is this many box lengths across, and an answer no longer than
#: SHEET_NEAR box lengths is drawn in it. An answer with more than SHEET_SPILL
#: of itself outside the close-up is drawn on the whole frame instead, so a
#: rung that took in what the object lies in is shown whole, not as a bright
#: crop with no edge in it.
SHEET_CONTEXT = 1.6
SHEET_NEAR = 1.5
SHEET_SPILL = 0.05
#: Two rungs this alike are one answer, drawn once under both names: SAM often
#: answers part and whole with the same mask, and two identical panels side by
#: side are no choice at all.
SAME_ANSWER_IOU = 0.95


def distinct_answers(levels: list) -> list[dict]:
    """SAM's rungs as the different answers they are, smallest first.

    levels: [(name, bool mask), ...] as the bridge got them. Every name is
    kept -- "part = whole" -- so the name read off a panel is still one the
    segmenter takes as a level.
    """
    out: list[dict] = []
    for name, m in sorted(levels, key=lambda nm: int(np.count_nonzero(nm[1]))):
        m = np.asarray(m, bool)
        for a in out:
            if iou(a["mask"], m) >= SAME_ANSWER_IOU:
                a["names"].append(str(name))
                a["masks"].append(m)
                break
        else:
            out.append({"names": [str(name)], "mask": m, "masks": [m]})
    return out


def fill_outline(outline: list, shape: tuple) -> np.ndarray:
    """The region inside an outline [[x, y], ...] (frame pixels), on a frame of shape (h, w)."""
    from PIL import ImageDraw
    h, w = int(shape[0]), int(shape[1])
    im = Image.new("L", (max(1, w), max(1, h)), 0)
    pts = [(float(p[0]), float(p[1])) for p in outline]
    if len(pts) >= 3:
        ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.array(im) > 0


def outline_prompt(region: np.ndarray) -> tuple[list, list] | None:
    """What SAM is asked for a region: the point deepest inside it, with its box.

    The same function for an outline a model traced, filled, and for the
    person's own mask when calibrate_sam measures the way -- a perfect outline
    encloses exactly that mask, and this is the deepest point components()
    reports for it -- so the prompt that is scored is the prompt that labels.
    None when the region is empty.
    """
    ys, xs = np.nonzero(region)
    if not xs.size:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    # With its holes filled: an outline has none, and a person's washer or nut
    # is painted as a ring. Asked from the ring's metal in calibration and from
    # the middle of the hole in labelling, the two would not be the same question.
    dt = ndimage.distance_transform_edt(np.pad(ndimage.binary_fill_holes(region[y0:y1, x0:x1]), 1))[1:-1, 1:-1]
    cy, cx = np.unravel_index(int(dt.argmax()), dt.shape)
    return [[x0 + int(cx), y0 + int(cy)]], [x0, y0, x1, y1]


def object_region(fg: np.ndarray, obj: dict) -> np.ndarray:
    """One teacher object's own pixels, from the frame's foreground and its components() record.

    For callers that asked components() for no masks -- a mask per object is
    the frame's size, and a teacher can hold a hundred objects.
    """
    x0, y0, x1, y1 = obj["bbox"]
    lbl, _ = ndimage.label(fg[y0:y1, x0:x1])
    dx, dy = obj["deepest"]
    out = np.zeros(fg.shape, bool)
    out[y0:y1, x0:x1] = lbl == lbl[dy - y0, dx - x0]
    return out


def sheet_verdict(rung: dict) -> str:
    """What the band said of a rung, in the word the panel carries."""
    if rung.get("passes"):
        return "fits"
    why = str(rung.get("why") or "").split()
    return f"no: {why[0]}" if why else "no"


def _sheet_font(px: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=px)      # FreeType, Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()             # bitmap: enlarged below


def sheet_plate(img, xy: tuple[int, int], text: str, px: int,
                fg=(255, 255, 255), bg=(0, 0, 0), pad: int = 3) -> int:
    """Text on a plate of its own, legible on a black object or a white surface.
    ASCII only: the bitmap fallback has no other glyphs. Returns its width."""
    from PIL import ImageDraw, ImageFont
    f = _sheet_font(px)
    d = ImageDraw.Draw(img)
    x, y = xy
    left, top, right, bottom = d.textbbox((0, 0), text, font=f)
    if isinstance(f, ImageFont.FreeTypeFont):
        w, h = right - left, bottom - top
        d.rectangle([x, y, x + w + 2 * pad, y + h + 2 * pad], fill=bg)
        d.text((x + pad - left, y + pad - top), text, fill=fg, font=f)
        return w + 2 * pad
    # The bitmap face is 11 px tall; drawn as it is, a model reads it as noise.
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    k = max(1, math.ceil(px / max(1, h - 2 * pad)))
    small = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(small).text((pad - left, pad - top), text, fill=fg, font=f)
    img.paste(small.resize((w * k, h * k), Image.NEAREST), (x, y))
    return w * k


def _grow_mask(m: np.ndarray, px: int) -> np.ndarray:
    """Four-neighbour dilation, the same as loop._grow."""
    out = m.copy()
    for _ in range(max(0, int(px))):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def _extent(m: np.ndarray) -> list[int] | None:
    rows, cols = m.any(axis=1), m.any(axis=0)
    if not rows.any():
        return None
    return [int(cols.argmax()), int(rows.argmax()),
            m.shape[1] - int(cols[::-1].argmax()), m.shape[0] - int(rows[::-1].argmax())]


def sheet_row(answers: list[dict], box: list | None, points: list | None = None,
              object_px: list | tuple | None = None, panel: int = SHEET_PANEL,
              outline: list | None = None) -> dict:
    """One box's answers, laid out: where each panel looks, and its mask at panel size.

    answers: [{"label", "mask" (bool, the frame's pixels), "area_pct", "verdict"}],
    smallest first. box and points are in the frame's pixels. No photo pixels
    are held, so a batch can keep one of these per box and draw at the end.
    """
    if not answers:
        raise ValueError("no answers to lay out")
    H, W = answers[0]["mask"].shape
    if box:
        bx0, by0, bx1, by1 = (float(v) for v in box)
    elif outline:
        bx0, by0 = min(float(p[0]) for p in outline), min(float(p[1]) for p in outline)
        bx1, by1 = max(float(p[0]) for p in outline), max(float(p[1]) for p in outline)
    else:
        ow, oh = object_px or (max(W, H) / 10.0, max(W, H) / 10.0)
        xs = [float(p[0]) for p in points or [[W / 2.0, H / 2.0]]]
        ys = [float(p[1]) for p in points or [[W / 2.0, H / 2.0]]]
        bx0, by0 = min(xs) - ow / 2.0, min(ys) - oh / 2.0
        bx1, by1 = max(xs) + ow / 2.0, max(ys) + oh / 2.0
    long_side = max(bx1 - bx0, by1 - by0, 16.0)
    ux0, uy0, ux1, uy1 = bx0, by0, bx1, by1
    extents = [_extent(a["mask"]) for a in answers]
    for e in extents:
        if e and max(e[2] - e[0], e[3] - e[1]) <= SHEET_NEAR * long_side:
            ux0, uy0 = min(ux0, e[0]), min(uy0, e[1])
            ux1, uy1 = max(ux1, e[2]), max(uy1, e[3])
    side = max(SHEET_CONTEXT * long_side, 1.15 * max(ux1 - ux0, uy1 - uy0), 48.0)
    sw, sh = min(side, float(W)), min(side, float(H))
    cx, cy = (ux0 + ux1) / 2.0, (uy0 + uy1) / 2.0
    vx0 = int(round(min(max(cx - sw / 2.0, 0.0), W - sw)))
    vy0 = int(round(min(max(cy - sh / 2.0, 0.0), H - sh)))
    view = [vx0, vy0, min(W, vx0 + int(round(sw))), min(H, vy0 + int(round(sh)))]
    panels = []
    for a in answers:
        m = a["mask"]
        total = int(np.count_nonzero(m))
        inside = int(np.count_nonzero(m[view[1]:view[3], view[0]:view[2]]))
        whole = bool(total) and total - inside > SHEET_SPILL * total
        crop = [0, 0, W, H] if whole else list(view)
        panels.append({"label": str(a["label"]), "area_pct": float(a["area_pct"]),
                       "verdict": str(a["verdict"]), "crop": crop, "whole_frame": whole,
                       "mask": _fit_mask(m[crop[1]:crop[3], crop[0]:crop[2]], panel)})
    return {"frame": (W, H), "box": [bx0, by0, bx1, by1] if box else None,
            "points": [list(p) for p in (points or [])], "panels": panels,
            "outline": [list(p) for p in (outline or [])]}


def _fit(cw: int, ch: int, panel: int) -> tuple[int, int, float]:
    s = panel / float(max(cw, ch))
    return max(1, int(round(cw * s))), max(1, int(round(ch * s))), s


def _fit_mask(m: np.ndarray, panel: int) -> np.ndarray:
    fw, fh, _ = _fit(m.shape[1], m.shape[0], panel)
    return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((fw, fh), Image.NEAREST)) > 127


def _up32(v: float) -> int:
    """Up to whole 32-pixel patches, the unit the model's encoder reads in."""
    return -(-int(v) // 32) * 32


def candidate_sheet(rgb, rows: list[dict], captions: list[str] | None = None,
                    panel: int = SHEET_PANEL):
    """The rows as one picture, or None when a mask is not of this picture.

    rgb is the frame itself (not the ruled copy: its vermilion grid is the
    colour of the box). A mask of another size would land in the wrong place,
    so nothing is drawn rather than something wrong.
    """
    from PIL import ImageDraw
    if not rows or any(tuple(r["frame"]) != tuple(rgb.size) for r in rows):
        return None
    big = panel >= 300
    ncol = max(len(r["panels"]) for r in rows)
    head_h, label_h, cap_h = 24, (32 if big else 26), (22 if captions else 0)
    row_h = cap_h + label_h + panel + SHEET_GAP
    # Inside the white line, not "bright": a dark object in a coloured container
    # is darker than the dimmed container around it, and the line is what marks it.
    pointed = all(not r["box"] and r["points"] for r in rows)
    traced = len(rows) == 1 and bool(rows[0].get("outline"))
    head = ("SAM's answers for your outline, smallest first. Inside the white line: the answer. "
            "Orange line: your outline; orange dot: where SAM was asked from." if traced else
            "SAM's answers for your points, smallest first. Inside the white line: the answer. "
            "Orange dots: your points." if pointed and len(rows) == 1 else
            "SAM's answers for some of your objects, smallest first. Inside the white line: the "
            "answer. Orange dots: the points." if pointed else
            "SAM's answers for your box, smallest first. Inside the white line: the answer. "
            "Orange: your box." if len(rows) == 1 else
            "SAM's answers for some of your boxes, smallest first. Inside the white line: the answer. "
            "Orange: the box.")
    head_px = 15 if big else 13
    # SAM often gives one mask under two names, and a sheet of one or two
    # panels cut the heading off half way: it is as wide as the heading, up to
    # the width of three panels.
    head_w = sheet_plate(Image.new("RGB", (1, 1)), (0, 0), head, head_px, pad=2)   # measured, not kept
    sheet = Image.new("RGB", (max(_up32(ncol * panel + (ncol + 1) * SHEET_GAP),
                                  min(_up32(head_w + 2 * SHEET_GAP), _up32(3 * panel + 4 * SHEET_GAP))),
                              _up32(head_h + len(rows) * row_h)), SHEET_BG)
    sheet_plate(sheet, (SHEET_GAP, 4), head, head_px, bg=SHEET_BG, pad=2)
    y = head_h
    vpx = 16 if big else 13
    for r, row in enumerate(rows):
        if cap_h:
            sheet_plate(sheet, (SHEET_GAP, y + 2), (captions or [""])[r][:90], 14, bg=SHEET_BG, pad=2)
        y_lab, y_img = y + cap_h, y + cap_h + label_h
        for k, p in enumerate(row["panels"]):
            x = SHEET_GAP + k * (panel + SHEET_GAP)
            c0x, c0y, c1x, c1y = p["crop"]
            fw, fh, s = _fit(c1x - c0x, c1y - c0y, panel)
            ox, oy = (panel - fw) // 2, (panel - fh) // 2
            fg = p["mask"]
            if fg.shape != (fh, fw):
                fg = np.array(Image.fromarray(fg.astype(np.uint8) * 255).resize((fw, fh), Image.NEAREST)) > 127
            a = np.asarray(rgb.crop((c0x, c0y, c1x, c1y)).resize((fw, fh), Image.LANCZOS)).astype(np.float32)
            a[~fg] *= SHEET_DIM
            a[_grow_mask(fg, SHEET_EDGE_PX) & ~fg] = 255.0
            tile = Image.new("RGB", (panel, panel), SHEET_BG)
            tile.paste(Image.fromarray(a.clip(0, 255).astype(np.uint8)), (ox, oy))
            d = ImageDraw.Draw(tile)
            if row["box"]:
                bx0, by0, bx1, by1 = row["box"]
                d.rectangle([ox + (bx0 - c0x) * s, oy + (by0 - c0y) * s,
                             ox + (bx1 - c0x) * s, oy + (by1 - c0y) * s], outline=SHEET_INK, width=2)
            if row.get("outline"):
                ring = row["outline"] + row["outline"][:1]
                d.line([(ox + (qx_ - c0x) * s, oy + (qy_ - c0y) * s) for qx_, qy_ in ring],
                       fill=SHEET_INK, width=2)
            for px_, py_ in row["points"]:
                qx, qy = ox + (px_ - c0x) * s, oy + (py_ - c0y) * s
                d.ellipse([qx - 5, qy - 5, qx + 5, qy + 5], outline=(255, 255, 255), width=2)
                d.ellipse([qx - 3, qy - 3, qx + 3, qy + 3], fill=SHEET_INK)
            if p["verdict"] == "KEPT":                  # a shape, not a colour
                d.rectangle([0, 0, panel - 1, panel - 1], outline=(255, 255, 255), width=6)
                d.rectangle([6, 6, panel - 7, panel - 7], outline=(0, 0, 0), width=2)
                sheet_plate(tile, (9, 9), "KEPT", vpx, fg=(0, 0, 0), bg=(255, 255, 255))
            else:
                sheet_plate(tile, (4, 4), p["verdict"], vpx, fg=(235, 235, 235), bg=(0, 0, 0))
            if p["whole_frame"]:
                sheet_plate(tile, (4, panel - vpx - 14), "whole frame", vpx - 2, fg=(235, 235, 235), bg=(0, 0, 0))
            sheet.paste(tile, (x, y_img))
            sheet_plate(sheet, (x, y_lab + 5), f"{p['label']}  {p['area_pct']:.1f}%",
                        17 if big else 14, bg=SHEET_BG)
        y += row_h
    return sheet


# ---------------------------------------------------------------------------
# The picture of what spot_detect found
# ---------------------------------------------------------------------------
#: spot_detect answered with a count, and a count can only be held against the
#: teacher's. A model that wrote every image from that count alone flagged
#: the ones that came in under half the teacher's; on one of them
#: many of the finds sat off the surface -- its edge, holes in it, a
#: fixture, the stand below -- the very things it had
#: named "not the target" at its second step, and it was never shown that
#: those were what had been found. This draws every find where it is.
#:
#: The whole picture is shrunk to fit this box, which on a wide 4K frame makes
#: the sheet about 1024 across: about what one 1280 px copy costs the model, in
#: a 16k window where a picture there is no room for is said in words -- the
#: count again. A speck 10 px across is 2.5 px up there; the ring is what
#: finds it, and the close-ups are what show it.
SPOT_SIDE = 1024
SPOT_SIDE_H = 768
#: A close-up is this much of the picture at full size: the speck's own pixels,
#: not a resampling of them, and room around it to see what it sits on.
SPOT_TILE = 256
SPOT_TILES = 3
#: Close-ups of where it found least, in a row of their own below those.
SPOT_QUIET_TILES = 3
#: Close-ups only when the whole picture is shown at less than half size.
#: Nearer full size, the top of the sheet already shows each speck as it is.
SPOT_TILE_BELOW = 0.5
#: One line of the heading.
SPOT_HEAD_LINE = 22
#: What is not a find: the region the model gave, where it landed, and the
#: finds it put out of the mask. Yellow, which any eye tells from the
#: vermilion of the rings -- never purple. Not white: the specks are white,
#: and on the review picture a white edge on a speck and on nothing were the
#: same white square (loop.WROTE_UNPICTURED).
SPOT_SECOND = (240, 228, 66)


def region_ring(region) -> list[tuple[float, float]]:
    """A region -- [x0, y0, x1, y1] or an outline [[x, y], ...] -- as a closed line of its corners."""
    if len(region) == 4 and not isinstance(region[0], (list, tuple)):
        x0, y0, x1, y1 = (float(v) for v in region)
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        pts = [(float(p[0]), float(p[1])) for p in region]
    return pts + pts[:1]


def cut_to_region(found: np.ndarray, region) -> tuple[np.ndarray, np.ndarray, int, int]:
    """What was found inside a region and what was not: (kept, left out, how many of each).

    region is [x0, y0, x1, y1] or an outline [[x, y], ...], in the picture's
    pixels. A speck is kept or left out whole, by where its middle is: cut
    pixel by pixel, one the region's edge runs through would be written as a
    sliver of itself and still counted as one. The blobs here are the
    detector's own -- it keeps whole four-connected regions, and two it kept
    never touch, or they would have been one -- so the two counts add up to
    the count it gave.
    """
    lbl, n = ndimage.label(found)
    if n == 0:
        return found.copy(), np.zeros_like(found), 0, 0
    at = np.asarray(ndimage.center_of_mass(found.astype(np.uint8), lbl, np.arange(1, n + 1)),
                    dtype=float).reshape(n, 2)
    ys, xs = at[:, 0], at[:, 1]
    if len(region) == 4 and not isinstance(region[0], (list, tuple)):
        x0, x1 = sorted((float(region[0]), float(region[2])))
        y0, y1 = sorted((float(region[1]), float(region[3])))
        inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    else:
        h, w = found.shape
        area = fill_outline(region, (h, w))
        inside = area[np.clip(np.round(ys).astype(int), 0, h - 1), np.clip(np.round(xs).astype(int), 0, w - 1)]
    keep = np.zeros(n + 1, bool)
    keep[1:] = inside
    kept = keep[lbl]
    return kept, found & ~kept, int(inside.sum()), int(n - inside.sum())


def speck_centres(mask: np.ndarray) -> np.ndarray:
    """Every blob of a mask as a row (x, y, radius), in the mask's pixels.

    Every one, however small: on a picture of specks the objects are the size
    components() drops as a brush's crumbs. The radius is the blob's own, from
    its area, so a ring goes round a speck and not over it.
    """
    lbl, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 3))
    idx = np.arange(1, n + 1)
    at = np.asarray(ndimage.center_of_mass(mask.astype(np.uint8), lbl, idx), dtype=float).reshape(n, 2)
    area = np.asarray(ndimage.sum(mask, lbl, idx), dtype=float)
    return np.stack([at[:, 1], at[:, 0], np.sqrt(area / math.pi)], axis=1)


def spot_tiles(centres: np.ndarray, frame: tuple, at: list | None = None,
               tile: int = SPOT_TILE, count: int = SPOT_TILES) -> list[dict]:
    """Where the close-ups go: the point asked from, the most finds in one
    place, then each time the find farthest from everything shown so far.

    The most finds in one close-up is where the detector is busiest; on the
    person's own frame that is a patch of their specks, and it is as likely to
    be an edge that looks like a row of them. The rest are the finds farthest
    from the middle of the finds and from every close-up already chosen: finds
    unlike the others by where they are, which is what a find off the surface
    looks like from here. Nothing in this knows what the surface is; the model
    looking at the close-up does.

    centres: speck_centres() rows. frame: (width, height). Boxes come back as
    [x0, y0, x1, y1] in the picture's pixels, numbered in the order chosen.
    """
    from scipy.spatial import cKDTree
    W, H = int(frame[0]), int(frame[1])
    half = tile / 2.0
    xy = np.asarray(centres, dtype=float).reshape(-1, 3)[:, :2]
    shown = np.zeros(len(xy), bool)
    out: list[dict] = []
    seeds: list[np.ndarray] = []

    def take(x: float, y: float, why: str) -> None:
        x0 = int(round(min(max(x - half, 0.0), max(0.0, W - tile))))
        y0 = int(round(min(max(y - half, 0.0), max(0.0, H - tile))))
        box = [x0, y0, min(W, x0 + tile), min(H, y0 + tile)]
        got = (xy[:, 0] >= box[0]) & (xy[:, 0] < box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] < box[3])
        shown[:] |= got
        seeds.append(np.array([x, y], float))
        out.append({"n": len(out) + 1, "box": box, "finds": int(got.sum()), "why": why})

    if at is not None:
        take(float(at[0]), float(at[1]), "your point")
    if not len(xy):
        return out
    if len(out) < count and not shown.all():
        tree = cKDTree(xy)
        # A close-up is square, so its neighbours are counted in a square: the
        # ball of p=inf, half its side across.
        near = np.where(shown, -1, tree.query_ball_point(xy, r=half, p=np.inf, return_length=True))
        best = int(near.argmax())
        if near[best] >= 2:          # one find is not a place where most of them are
            cx, cy = xy[tree.query_ball_point(xy[best], r=half, p=np.inf)].mean(axis=0)
            take(float(cx), float(cy), "most finds in one place")
    middle = np.median(xy, axis=0)
    while len(out) < count and not shown.all():
        far = np.min([np.hypot(xy[:, 0] - s[0], xy[:, 1] - s[1]) for s in [middle] + seeds], axis=0)
        k = int(np.where(shown, -1.0, far).argmax())
        take(float(xy[k, 0]), float(xy[k, 1]), "farthest from the rest")
    return out


def quiet_tiles(centres: np.ndarray, frame: tuple, within=None, taken=(), tile: int = SPOT_TILE,
                count: int = SPOT_QUIET_TILES, first: int = 1) -> list[dict]:
    """Where the close-ups of what was not found go: the places with the fewest finds.

    A speck the detector missed is seen only at full size, and every close-up
    spot_tiles chooses is chosen by what was found. A frame written with far
    fewer specks than the person's showed had specks with no ring in two of its
    three close-ups, and nothing showed a stretch of the surface it had found
    nothing on.

    within is the region the model gave -- [x0, y0, x1, y1] or an outline, in
    the picture's pixels -- and each close-up lies wholly inside it; with none,
    inside the span of what was found, or anywhere if nothing was. Of the
    places with the fewest finds each is the one farthest from every close-up
    already chosen, taken included; the first, with none chosen, the one
    nearest the middle. Nothing in this knows what the surface is either.
    Boxes come back as spot_tiles gives them, numbered on from first.
    """
    W, H = int(frame[0]), int(frame[1])
    if count <= 0 or W < tile or H < tile:
        return []
    xy = np.asarray(centres, dtype=float).reshape(-1, 3)[:, :2]
    area = None
    if within is not None and len(within) == 4 and not isinstance(within[0], (list, tuple)):
        x0, x1 = sorted((float(within[0]), float(within[2])))
        y0, y1 = sorted((float(within[1]), float(within[3])))
    elif within is not None:
        area = fill_outline(within, (H, W))
        ys, xs = np.nonzero(area)
        if not len(xs):
            return []
        x0, x1, y0, y1 = float(xs.min()), float(xs.max()) + 1, float(ys.min()), float(ys.max()) + 1
    elif len(xy):
        # A tile past the finds on each side: a row of them, or a tight cluster,
        # spans less than a close-up, and it is the detector finding least that
        # most needs a stretch shown where it found nothing.
        (x0, y0), (x1, y1) = xy.min(axis=0) - tile, xy.max(axis=0) + 1 + tile
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)

    def starts(a: float, b: float) -> list[int]:
        lo, hi = int(math.ceil(a)), int(math.floor(b)) - tile
        return sorted(set(range(lo, hi + 1, tile // 2)) | {hi}) if hi >= lo else []

    cells = [(cx, cy) for cy in starts(y0, y1) for cx in starts(x0, x1)]
    if within is None and not cells:
        cells = [(cx, cy) for cy in starts(0.0, float(H)) for cx in starts(0.0, float(W))]
    if area is not None:
        # Wholly inside, every pixel: probing a few let a close-up straddle a
        # notch narrower than the gap between probes.
        ii = np.pad(area.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32), ((1, 0), (1, 0)))
        cells = [(cx, cy) for cx, cy in cells
                 if ii[cy + tile, cx + tile] - ii[cy, cx + tile] - ii[cy + tile, cx] + ii[cy, cx] == tile * tile]
    if not cells:
        return []
    boxes = np.array([[cx, cy, cx + tile, cy + tile] for cx, cy in cells], dtype=float)
    if len(xy):
        finds = ((xy[None, :, 0] >= boxes[:, None, 0]) & (xy[None, :, 0] < boxes[:, None, 2])
                 & (xy[None, :, 1] >= boxes[:, None, 1]) & (xy[None, :, 1] < boxes[:, None, 3])).sum(axis=1)
    else:
        finds = np.zeros(len(boxes), dtype=int)
    mids = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    chosen = [np.asarray(b, dtype=float) for b in taken]
    free = np.ones(len(boxes), bool)
    out: list[dict] = []
    while len(out) < count:
        for b in chosen:
            free &= ~((boxes[:, 0] < b[2]) & (boxes[:, 2] > b[0]) & (boxes[:, 1] < b[3]) & (boxes[:, 3] > b[1]))
        if not free.any():
            break
        pool = free & (finds == finds[free].min())
        if chosen:
            cm = np.array([((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in chosen])
            far = np.hypot(mids[:, None, 0] - cm[None, :, 0], mids[:, None, 1] - cm[None, :, 1]).min(axis=1)
            k = int(np.where(pool, far, -1.0).argmax())
        else:
            near = np.hypot(mids[:, 0] - (x0 + x1) / 2.0, mids[:, 1] - (y0 + y1) / 2.0)
            k = int(np.where(pool, near, np.inf).argmin())
        chosen.append(boxes[k])
        out.append({"n": first + len(out), "box": [int(v) for v in boxes[k]], "finds": int(finds[k]),
                    "why": "nothing found here" if finds[k] == 0 else "least found here"})
    return out


def spot_sheet(rgb, found: np.ndarray, *, point: list | None = None, region=None,
               dropped: np.ndarray | None = None, side: int = SPOT_SIDE, side_h: int = SPOT_SIDE_H):
    """What spot_detect found, as one picture: (picture, close-ups), or (None, []).

    rgb is the picture itself, not the ruled copy: the ruler is vermilion, the
    rings' colour. found is what would be written and dropped what a region
    put out of it, both masks in the picture's pixels; point and region are in
    the picture's pixels too, the region as it was put back, so one read in the
    wrong frame is drawn in the wrong place. A mask of another size would ring
    the wrong places, so nothing is drawn rather than something wrong.

    Top: the whole picture, each find ringed in vermilion. Below, when the top
    is less than half size: close-ups at full size, numbered, each from the
    square of its number above -- a row where it found most, then a row where
    it found least (quiet_tiles). No coordinates are written on it: the heading
    and the close-ups make it taller than any copy, and a picture taller than
    the size it is read in is how every y once came back scaled by the
    ratio between the two.
    """
    from PIL import ImageDraw
    W, H = rgb.size
    if found.shape != (H, W) or (dropped is not None and dropped.shape != (H, W)):
        return None, []
    s = min(1.0, side / float(W), side_h / float(H))
    ow, oh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    top = rgb.resize((ow, oh), Image.LANCZOS, reducing_gap=3.0) if s < 1.0 else rgb.copy()
    kept = speck_centres(found)
    gone = speck_centres(dropped) if dropped is not None else np.zeros((0, 3))
    tiles = spot_tiles(kept, (W, H), at=point) if s < SPOT_TILE_BELOW else []
    # Where it found least, below where it found most: a speck it missed is
    # seen only at full size, and each close-up above is chosen by a find.
    quiet = (quiet_tiles(kept, (W, H), within=region, taken=[t["box"] for t in tiles], first=len(tiles) + 1)
             if s < SPOT_TILE_BELOW else [])
    ring = region_ring(region) if region else None

    def marks(d, ox: float, oy: float, z: float, least: float) -> None:
        # Round the speck, not over it: its own radius and a gap.
        for x, y, r in kept:
            q = max(least, r * z + 4.0)
            d.ellipse([ox + x * z - q, oy + y * z - q, ox + x * z + q, oy + y * z + q],
                      outline=SHEET_INK, width=2)
        # Left out is a cross, kept is a ring: told apart by shape as well as colour.
        for x, y, _ in gone:
            gx, gy = ox + x * z, oy + y * z
            d.line([(gx - 4, gy - 4), (gx + 4, gy + 4)], fill=SPOT_SECOND, width=2)
            d.line([(gx - 4, gy + 4), (gx + 4, gy - 4)], fill=SPOT_SECOND, width=2)
        if ring:
            d.line([(ox + x * z, oy + y * z) for x, y in ring], fill=SPOT_SECOND, width=2)
        if point:
            # Ticks outside the rings, white on a dark edge: a dot on the point
            # painted over the ring of the speck pointed at, and in a close-up
            # the speck itself.
            qx, qy = ox + point[0] * z, oy + point[1] * z
            a, b = least + 5, least + 12
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                seg = [(qx + dx * a, qy + dy * a), (qx + dx * b, qy + dy * b)]
                d.line(seg, fill=(0, 0, 0), width=4)
                d.line(seg, fill=(255, 255, 255), width=2)

    d = ImageDraw.Draw(top)
    marks(d, 0.0, 0.0, s, 5.0)
    for t in tiles + quiet:
        # Where each close-up comes from, in white and black: neither is a
        # find's colour, and the pair reads on a white edge and a dark surface.
        bx0, by0, bx1, by1 = (v * s for v in t["box"])
        d.rectangle([bx0 - 1, by0 - 1, bx1 + 1, by1 + 1], outline=(0, 0, 0), width=1)
        d.rectangle([bx0, by0, bx1, by1], outline=(255, 255, 255), width=2)
        # The number outside its square, so it covers no find in it.
        sheet_plate(top, (int(bx0), int(by0 - 21 if by0 >= 21 else by1 + 2)), str(t["n"]), 14)
    say = f"spot_detect found {len(kept)}, each ringed orange."
    if tiles:
        say += (f" Squares 1-{len(tiles)}: the close-ups below, at full size." if len(tiles) > 1
                else " Square 1: the close-up below, at full size.")
    head = [say]
    if quiet:
        a, b = quiet[0]["n"], quiet[-1]["n"]
        where = " inside your region" if ring else ""
        head.append(f"Squares {a}-{b}, the lower row: where it found least{where}." if b > a else
                    f"Square {a}, the lower row: where it found least{where}.")
        # On the surface only: a close-up can take in an edge or a stand, whose
        # grains are not the specks being labelled.
        head.append("On the surface you label, a speck in a close-up with no ring is one it missed.")
    legend = ("Yellow line: your region. Yellow x: a find outside it, not written. " if ring or len(gone) else "")
    legend += "White ticks: your point." if point else ""
    if legend:
        head.append(legend.strip())
    head_h = 4 + SPOT_HEAD_LINE * len(head)
    # As wide as the heading on a picture narrower than it: a sheet cut off
    # half way through its first line says less than none.
    head_w = max(sheet_plate(Image.new("RGB", (1, 1)), (0, 0), line, 15, pad=2) for line in head)
    rows = [r for r in (tiles, quiet) if r]
    strip = max((len(r) * SPOT_TILE + (len(r) + 1) * SHEET_GAP for r in rows), default=0)
    width = _up32(max(ow, strip, min(head_w + 2 * SHEET_GAP, side)))
    band = SPOT_TILE + SHEET_GAP + SPOT_HEAD_LINE          # a row of close-ups, with their captions
    sheet = Image.new("RGB", (width, _up32(head_h + oh + (len(rows) * band + SHEET_GAP if rows else 0))),
                      SHEET_BG)
    for k, line in enumerate(head):
        sheet_plate(sheet, (SHEET_GAP, 4 + k * SPOT_HEAD_LINE), line, 15, bg=SHEET_BG, pad=2)
    sheet.paste(top, ((width - ow) // 2, head_h))
    y = head_h + oh + SHEET_GAP + SPOT_HEAD_LINE
    for row in rows:
        for k, t in enumerate(row):
            x0, y0, x1, y1 = t["box"]
            tile = rgb.crop((x0, y0, x1, y1))       # at full size: the speck's own pixels
            marks(ImageDraw.Draw(tile), -x0, -y0, 1.0, 6.0)
            t["at"] = [SHEET_GAP + k * (SPOT_TILE + SHEET_GAP), y]
            sheet.paste(tile, tuple(t["at"]))
            # Above the close-up, not on it: on it, a close-up pushed against the
            # frame's top edge had its find under the caption.
            sheet_plate(sheet, (t["at"][0], y - SPOT_HEAD_LINE + 2), f"{t['n']}  {t['why']} ({t['finds']})", 13,
                        bg=SHEET_BG)
        y += band
    return sheet, tiles + quiet
