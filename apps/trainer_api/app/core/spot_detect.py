# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Spot detection: find the other specks that look like the one that was painted.

Moved here from the browser (apps/trainer_ui/src/annotate/imageProcessing.ts and
spotDetectWorker.ts) so that there is one implementation instead of three. The
browser had two of them -- the sensitivity search flood-filled at one threshold
while the worker that produced the mask filled at half of it, so the search was
scoring an operator that never ran. On a large frame of many alike specks, that
mismatch was the difference between finding a handful of them and nearly all.

The algorithm, unchanged from the browser's:

  * Read the painted stroke for a channel, a colour, a size window and a colour
    tolerance. The stroke says WHERE an example is, not what it is: a brush wide
    enough to hit a 3px speck lays down ten times as much background as speck, so
    nothing here may treat the stroke's own extent as the object's extent.
  * Score every pixel by a Laplacian of Gaussian at two fixed scales and keep the
    stronger response. The kernels are fixed in PIXELS, so the frame is never
    resized before scoring.
  * Seed at the local maxima, grow each seed while the score stays above half the
    threshold, and keep the regions that pass the size window and the colour test.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from segcore.image_io import imread as _imread

from .cache_utils import ThreadSafeLRUCache
from .rf_assist import encode_png_base64

logger = logging.getLogger(__name__)

#: Bumped whenever a change here would invalidate a cached score map.
ALGO_VERSION = 1

#: The two scales the detector looks for, in pixels. A speck a few pixels across
#: sits on the first; anything much larger or smaller is outside both, which is
#: why the frame must reach this function at its own resolution.
_SIGMAS = (1.5, 3.0)

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, last axis 3) to CIE Lab, D65. Matches the browser's rgbToLab."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750)
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def _channel_image(rgb: np.ndarray, channel: str) -> np.ndarray:
    """The one plane the detector scores. 'a' and 'b' are the browser's cheap
    stand-ins for the Lab axes, kept so that both sides agree."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    if channel == "a":
        return r - g
    if channel == "b":
        return (r + g) * 0.5 - b
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _gaussian_blur(plane: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian with the browser's kernel size and edge clamping."""
    k_size = int(np.ceil(sigma * 6)) | 1
    half = k_size >> 1
    x = np.arange(k_size, dtype=np.float64) - half
    k = np.exp(-0.5 * (x * x) / (sigma * sigma))
    k /= k.sum()
    k = k.astype(np.float32)
    out = ndimage.correlate1d(plane, k, axis=1, mode="nearest")
    return ndimage.correlate1d(out, k, axis=0, mode="nearest")


def _laplacian_of_gaussian(plane: np.ndarray, sigma: float) -> np.ndarray:
    """Blur, then the 4-neighbour Laplacian. The border stays zero, as it does in
    the browser -- a speck on the outermost row is not a speck we can measure."""
    b = _gaussian_blur(plane, sigma)
    out = np.zeros_like(b)
    out[1:-1, 1:-1] = (b[1:-1, :-2] + b[1:-1, 2:] + b[:-2, 1:-1] + b[2:, 1:-1]
                       - 4.0 * b[1:-1, 1:-1])
    return out


@dataclass
class SpotScores:
    scores: np.ndarray      #: float32 (h, w)
    std: float              #: RMS of the whole score map, what thresholds are quoted in
    extrema: np.ndarray     #: bool (h, w), the 5x5 local maxima that seed the fill


def compute_spot_scores(rgb: np.ndarray, channel: str = "gray") -> SpotScores:
    plane = _channel_image(rgb, channel)
    log1 = _laplacian_of_gaussian(plane, _SIGMAS[0])
    log2 = _laplacian_of_gaussian(plane, _SIGMAS[1])
    scores = np.maximum(np.abs(log1), np.abs(log2)).astype(np.float32)
    std = float(np.sqrt(np.mean(scores.astype(np.float64) ** 2)))
    # A tie does not disqualify a maximum: a saturated speck has a flat top, and
    # requiring a strict maximum would throw away exactly the brightest ones.
    peak = ndimage.maximum_filter(scores, size=5, mode="nearest")
    extrema = scores >= peak
    extrema[:2, :] = False
    extrema[-2:, :] = False
    extrema[:, :2] = False
    extrema[:, -2:] = False
    return SpotScores(scores=scores, std=std, extrema=extrema)


# ---------------------------------------------------------------------------
# reading the painted stroke
# ---------------------------------------------------------------------------

def _js_round(x: float) -> int:
    """JavaScript's Math.round: halves go up, not to even."""
    return int(np.floor(x + 0.5))


@dataclass
class MarkAnalysis:
    channel: str
    target_lab: np.ndarray          #: (3,)
    color_tolerance: float
    size_range: tuple[int, int]
    best_snr: float


def analyze_mark(rgb: np.ndarray, mark: np.ndarray) -> MarkAnalysis | None:
    """What the painted stroke can and cannot tell us.

    It can say where an example is, roughly what colour it is, and roughly how
    big. It cannot say how big the OTHER instances are, and it is not itself the
    object: most of a brush dab over a 3px speck is the surface under it.
    """
    h, w = mark.shape
    painted = mark > 0
    fg_count = int(painted.sum())
    if fg_count < 5:
        return None

    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    ax = r - g
    bx = (r + g) * 0.5 - b

    fg_mean = (gray[painted].mean(), ax[painted].mean(), bx[painted].mean())
    fg_std = (gray[painted].std(), ax[painted].std(), bx[painted].std())

    # Background, measured beside what was painted rather than across the frame:
    # what a speck is distinguishable from is the surface it sits on, not the
    # average of a bright border and a dark field elsewhere in the frame.
    ys, xs = np.nonzero(painted)
    by0, by1, bx0, bx1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    reach = max(8, _js_round(max(bx1 - bx0, by1 - by0) * 2))
    ry0, ry1 = max(0, by0 - reach), min(h - 1, by1 + reach)
    rx0, rx1 = max(0, bx0 - reach), min(w - 1, bx1 + reach)
    band = np.zeros_like(painted)
    band[ry0:ry1 + 1, rx0:rx1 + 1] = True
    band &= ~painted
    if int(band.sum()) < 10:
        # A mark that fills its own neighbourhood leaves nothing to compare
        # against; that falls back to the frame, which is where this started.
        step = max(1, (h * w) // 3000)
        flat = np.zeros(h * w, dtype=bool)
        flat[::step] = True
        band = flat.reshape(h, w) & ~painted
    if int(band.sum()) < 10:
        return None

    bg_mean = (gray[band].mean(), ax[band].mean(), bx[band].mean())
    snr = [abs(fg_mean[i] - bg_mean[i]) / max(fg_std[i], 1.0) for i in range(3)]

    best_idx = 0
    if snr[1] > snr[best_idx]:
        best_idx = 1
    if snr[2] > snr[best_idx]:
        best_idx = 2
    channel = "gray" if snr[best_idx] < 1.5 else ("L", "a", "b")[best_idx]

    target_lab = rgb_to_lab(np.array([r[painted].mean(), g[painted].mean(), b[painted].mean()]))

    # The spread inside one stroke measures how uniform ONE example is, which
    # says nothing about how far the next instance of the same defect may sit: a
    # brighter speck of the same kind is tens of deltaE away, and a floor of 8
    # was rejecting exactly those.
    tolerance = max(16.0, min(40.0, 4.0 * float(np.sqrt(
        fg_std[0] * 0.3 + fg_std[1] * 0.5 + fg_std[2] * 0.3))))

    # Not the median plus or minus a half: that window admits only objects the
    # size of the one that was painted, and a brighter instance of the same
    # defect grows a wider region above the same threshold.
    lab, n = ndimage.label(painted, structure=_S4)
    if n > 0:
        sizes = np.bincount(lab.ravel())[1:]
        median = int(np.sort(sizes)[len(sizes) // 2])
        size_range = (max(1, _js_round(median / 4)), _js_round(median * 4))
    else:
        size_range = (1, 500)

    return MarkAnalysis(channel=channel, target_lab=target_lab, color_tolerance=tolerance,
                        size_range=size_range, best_snr=float(snr[best_idx]))


# ---------------------------------------------------------------------------
# thresholding
# ---------------------------------------------------------------------------

def threshold_spots(
    sc: SpotScores, rgb: np.ndarray, sensitivity: float,
    size_range: Sequence[int],
    target_lab: np.ndarray | None = None,
    color_tolerance: float | None = None,
) -> tuple[np.ndarray, int]:
    """Grow a region from every local maximum, keep the ones that look like the
    example.

    Flood-filling from each seed in turn, with one shared visited map, is exactly
    the connected components of the fill set that contain an admissible seed --
    the same answer, without walking 300,000 seeds one at a time.
    """
    threshold = sc.std * (sensitivity / 5.0)
    labels, n = ndimage.label(sc.scores > threshold * 0.5, structure=_S4)
    if n == 0:
        return np.zeros(sc.scores.shape, dtype=bool), 0

    seeded = np.zeros(n + 1, dtype=bool)
    admissible = sc.extrema & (sc.scores > threshold)
    seeded[labels[admissible]] = True
    seeded[0] = False

    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = seeded & (sizes >= size_range[0]) & (sizes <= size_range[1])

    if target_lab is not None and color_tolerance is not None and keep.any():
        idx = np.flatnonzero(keep)
        means = np.stack([ndimage.mean(rgb[..., c].astype(np.float64), labels, idx)
                          for c in range(3)], axis=-1)
        dist = np.linalg.norm(rgb_to_lab(means) - np.asarray(target_lab), axis=-1)
        keep[idx[dist > color_tolerance]] = False

    return keep[labels], int(keep.sum())


#: The sensitivities the search walks, as the browser did: 2, 6, 10 ... 38.
_SENSITIVITIES = tuple(range(2, 39, 4))


def _mark_core(sc: SpotScores, mark: np.ndarray) -> np.ndarray:
    """The third of the painted pixels that carry the strongest response.

    What "the example is recovered" is measured on -- by the search, and by the
    answer to a sensitivity sent back, alike. A brush dab is mostly the surface
    under the speck, and a threshold loose enough to swallow that surface
    answers with the whole frame.
    """
    painted = np.nonzero(mark > 0)
    core = np.zeros(mark.shape, dtype=bool)
    if len(painted[0]) == 0:
        return core
    vals = sc.scores[painted]
    keep_n = max(1, _js_round(len(vals) / 3))
    cut = np.sort(vals)[::-1][keep_n - 1]
    core[painted[0][vals >= cut], painted[1][vals >= cut]] = True
    return core


def auto_sensitivity(
    sc: SpotScores, rgb: np.ndarray, mark: np.ndarray,
    size_range: Sequence[int],
    target_lab: np.ndarray | None = None,
    color_tolerance: float | None = None,
) -> tuple[int, float]:
    """The tightest threshold that still recovers the example.

    Scored on the third of the painted pixels that carry the strongest response,
    not on all of them: a brush dab is mostly the surface under the speck, and a
    threshold loose enough to swallow that surface answers with the whole frame.

    It is scored with `threshold_spots`, the same function that produces the
    mask. The browser scored one flood-fill rule and shipped another, and the
    search's answer then found a small fraction of the specks that the rule it
    thought it was scoring would have found.
    """
    core = _mark_core(sc, mark)
    core_count = int(core.sum())
    if core_count == 0:
        return 15, 0.0

    best = (15, -1.0, np.inf)   # sensitivity, recall, extra pixels
    for sens in _SENSITIVITIES:
        mask, _ = threshold_spots(sc, rgb, sens, size_range, target_lab, color_tolerance)
        hit = int((mask & core).sum())
        recall = hit / core_count
        extra = int(mask.sum()) - hit
        # A hair of recall is worth more than any number of extras; among equals,
        # the one that offers least to sift through.
        if recall > best[1] + 0.01 or (abs(recall - best[1]) <= 0.01 and extra < best[2]):
            best = (sens, recall, extra)
    return best[0], best[1]


# ---------------------------------------------------------------------------
# the entry point the router calls
# ---------------------------------------------------------------------------

#: Scoring a 4K frame costs about 0.8s, so the map is kept for the slider.
#: Each entry is the score map plus its extrema, about 44 MB for a frame that size.
_SCORE_CACHE = ThreadSafeLRUCache(maxsize=3, ttl=600.0)
#: Reading the stroke is cheap but not free, and the slider re-sends the same one.
_ANALYSIS_CACHE = ThreadSafeLRUCache(maxsize=32, ttl=600.0)


def _decode_mark(mark_png_b64: str, shape: tuple[int, int]) -> np.ndarray:
    """The stroke the user painted, as sent by the browser: an L-mode PNG, any
    non-zero pixel counts as painted."""
    raw = base64.b64decode(mark_png_b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    mark = np.asarray(img, dtype=np.uint8)
    if mark.shape != shape:
        raise ValueError(
            f"the painted mark is {mark.shape[1]}x{mark.shape[0]} but the image is "
            f"{shape[1]}x{shape[0]}")
    return mark


def mark_from_point(shape: tuple[int, int], x: int, y: int, radius: int = 4) -> np.ndarray:
    """A round dab at a point, for callers that point rather than paint.

    An agent naming a speck's coordinates is in the same position as a person
    dabbing a brush on it, and the detector reads either the same way: as a
    place, not as the object.
    """
    h, w = shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"point ({x}, {y}) is outside the {w}x{h} image")
    radius = max(1, min(64, int(radius)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disc = (yy * yy + xx * xx) <= radius * radius
    mark = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    mark[y0:y1, x0:x1] = disc[y0 - (y - radius):y1 - (y - radius),
                              x0 - (x - radius):x1 - (x - radius)] * 255
    return mark


#: A brush stroke on a speck is a few hundred pixels. Well past any real stroke,
#: and far short of anything that would make the request itself the problem.
_MAX_MARK_POINTS = 200_000


def mark_from_points(shape: tuple[int, int], points: Sequence[Sequence[int]]) -> np.ndarray:
    """The painted pixels themselves, as the browser has them.

    Sending coordinates rather than a full-frame PNG keeps the browser out of
    the business of encoding an eight-megapixel image to say where it painted
    three hundred pixels.
    """
    if len(points) > _MAX_MARK_POINTS:
        raise ValueError(
            f"the mark has {len(points)} points, more than the {_MAX_MARK_POINTS} "
            "this takes; send it as mark_png_b64 instead")
    h, w = shape
    mark = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("mark_points must be a list of [x, y] pairs")
    xs, ys = pts[:, 0], pts[:, 1]
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not inside.any():
        raise ValueError(f"none of the {len(points)} marked points are inside the {w}x{h} image")
    mark[ys[inside], xs[inside]] = 255
    return mark


def detect_spots(
    img_path: str,
    mark_png_b64: str | None = None,
    sensitivity: int | None = None,
    point: Sequence[int] | None = None,
    radius: int = 4,
    class_id: int = 1,
    mark_points: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Find the specks that look like the painted one.

    Give the example any of three ways: `mark_png_b64` is a brush stroke at full
    image size (L-mode PNG, non-zero is painted), `mark_points` is the painted
    pixels as [x, y] pairs, `point` is a single (x, y) to dab at.

    `sensitivity` is what the slider sends back; leave it out and the tightest
    threshold that still recovers the example is chosen.

    The mask comes back as a single-channel PNG whose pixels are `class_id`, the
    format masks are stored in -- not 0/255, where 255 would mean "ignore".
    """
    t0 = time.perf_counter()
    image = _imread(img_path)
    if image is None:
        # The path goes to the log, not into the error: this message reaches
        # the browser and the agent, and an absolute path names the account.
        logger.warning("spot_detect: failed to read image %s", img_path)
        raise ValueError("failed to read the image")
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        # imread is OpenCV, so it hands back BGR, while the browser this moved
        # from worked on canvas pixels, which are RGB. Every colour decision
        # below -- which channel to score, the target colour, the deltaE gate --
        # would quietly differ by the swap.
        rgb = image[:, :, 2::-1]
    h, w = rgb.shape[:2]

    if mark_png_b64:
        mark = _decode_mark(mark_png_b64, (h, w))
    elif mark_points:
        mark = mark_from_points((h, w), mark_points)
    elif point is not None:
        mark = mark_from_point((h, w), int(point[0]), int(point[1]), radius)
    else:
        raise ValueError("give mark_png_b64, mark_points or point")

    mark_key = hashlib.sha1(mark.tobytes()).hexdigest()[:16]
    analysis = _ANALYSIS_CACHE.get(f"{img_path}::{mark_key}::v{ALGO_VERSION}")
    if analysis is None:
        analysis = analyze_mark(rgb, mark)
        if analysis is None:
            return {
                "mask": None, "count": 0, "why":
                "the painted mark is too small to read, or it fills its own "
                "neighbourhood so there is no surface to compare it against",
            }
        _ANALYSIS_CACHE.put(f"{img_path}::{mark_key}::v{ALGO_VERSION}", analysis)

    # The channel ranking answers a colour question and says nothing about
    # whether the mark is a local maximum, which is the only thing a blob
    # detector reads. Below the colour-mode bar, the blob detector looks at grey.
    color_mode = analysis.best_snr >= 2.5
    channel = analysis.channel if color_mode else "gray"

    cache_key = f"{img_path}::{channel}::v{ALGO_VERSION}"
    sc = _SCORE_CACHE.get(cache_key)
    cached = sc is not None
    if not cached:
        sc = compute_spot_scores(rgb, channel)
        _SCORE_CACHE.put(cache_key, sc)

    recall = None
    if sensitivity is None:
        sensitivity, recall = auto_sensitivity(
            sc, rgb, mark, analysis.size_range, analysis.target_lab, analysis.color_tolerance)

    mask, count = threshold_spots(
        sc, rgb, sensitivity, analysis.size_range, analysis.target_lab, analysis.color_tolerance)
    if recall is None:
        # A sensitivity sent back came with no recall at all, so a model moving
        # the threshold could not see whether the speck it pointed at was still
        # among what came back, and walked it back and forth with nothing to go
        # by but a count.
        core = _mark_core(sc, mark)
        recall = float((mask & core).sum()) / max(int(core.sum()), 1)

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("spot_detect: %d spots, sens=%s, %dms, scores_cached=%s",
                count, sensitivity, elapsed, cached)
    return {
        "class_id": int(class_id),
        "mask": encode_png_base64((mask * class_id).astype(np.uint8)),
        "count": count,
        "sensitivity": int(sensitivity),
        "mark_recall": None if recall is None else round(float(recall), 3),
        "mode": "color" if color_mode else "dog",
        "channel": channel,
        "best_snr": round(float(analysis.best_snr), 3),
        "size_range": list(analysis.size_range),
        "color_tolerance": round(float(analysis.color_tolerance), 2),
        "width": int(w),
        "height": int(h),
        "time_ms": elapsed,
        "scores_cached": cached,
    }


# ---------------------------------------------------------------------------
# the person's own specks as the example
# ---------------------------------------------------------------------------

#: The thresholds a teacher is measured at. Past the search's 38: a frame of
#: many alike specks can take a tighter threshold than one dab can say it needs.
_TEACHER_SENSITIVITIES = tuple(range(2, 61, 2))
#: Measuring a teacher walks thirty thresholds over its frame, twice over for
#: the held-out figure: about twenty seconds on a 4K frame, paid once.
_CALIBRATION_CACHE = ThreadSafeLRUCache(maxsize=8, ttl=1800.0)


@dataclass
class Calibration:
    analysis: MarkAnalysis
    channel: str
    sensitivity: int
    on_teacher: dict        #: their specks found, and found ones on none of theirs
    held_out: dict          #: the same, chosen on one side of the frame, scored on the other


def _read_rgb(img_path: str) -> np.ndarray:
    image = _imread(img_path)
    if image is None:
        # Logged, not raised with the path (see detect_spots).
        logger.warning("spot_detect: failed to read image %s", img_path)
        raise ValueError("failed to read the image")
    # OpenCV hands back BGR; everything here reads RGB (see detect_spots)
    return np.stack([image] * 3, axis=-1) if image.ndim == 2 else image[:, :, 2::-1]


def _scores_for(img_path: str, rgb: np.ndarray, channel: str) -> tuple[SpotScores, bool]:
    key = f"{img_path}::{channel}::v{ALGO_VERSION}"
    sc = _SCORE_CACHE.get(key)
    if sc is not None:
        return sc, True
    sc = compute_spot_scores(rgb, channel)
    _SCORE_CACHE.put(key, sc)
    return sc, False


def _objects(found: np.ndarray, truth: np.ndarray) -> dict:
    """Their specks found, and found ones on none of theirs: counted, not pixels.

    A speck is a few pixels across, and an edge one pixel off halves its
    overlap; whether it was found at all is what carries.
    """
    lt, nt = ndimage.label(truth)
    lf, nf = ndimage.label(found)
    both = found & truth
    hit_t = int((np.unique(lt[both]) > 0).sum())
    hit_f = int((np.unique(lf[both]) > 0).sum())
    p = hit_f / nf if nf else 0.0
    r = hit_t / nt if nt else 0.0
    return {"theirs": int(nt), "found": hit_t, "yours": int(nf), "on_nothing": int(nf - hit_f),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def _best_threshold(sc: SpotScores, rgb: np.ndarray, analysis: MarkAnalysis,
                    truth: np.ndarray, region: np.ndarray) -> tuple[int, dict]:
    best: tuple[int, dict] | None = None
    for sens in _TEACHER_SENSITIVITIES:
        mask, _ = threshold_spots(sc, rgb, sens, analysis.size_range,
                                  analysis.target_lab, analysis.color_tolerance)
        got = _objects(mask & region, truth & region)
        if best is None or got["f1"] > best[1]["f1"]:
            best = (sens, got)
    assert best is not None
    return best


def calibrate_on_teacher(teacher_path: str, truth: np.ndarray) -> Calibration:
    """The person's own specks as the example, and the threshold chosen against them.

    A point is one speck, and a model pointing at a large frame through a
    1280 px copy points at the one that stands out: a saturated blob twice the
    size of the rest, which the detector then found alone or not at all. The
    person's mask is every speck they meant. What the point decided -- channel,
    colour, size window, threshold -- is decided here against all of them,
    which finds more of the person's specks, and far fewer on nothing, than
    the model's best point did.

    How far that carries to a frame nobody drew is measured, not assumed: the
    frame is split at the middle of their specks, a threshold chosen on the
    left and scored on the right.
    """
    rgb = _read_rgb(teacher_path)
    if truth.shape != rgb.shape[:2]:
        raise ValueError("the person's mask is not the size of their image")
    if not truth.any():
        raise ValueError("the person's mask has nothing painted in it")
    analysis = analyze_mark(rgb, truth.astype(np.uint8) * 255)
    if analysis is None:
        raise ValueError("the person's specks are too few or too small to read")
    channel = analysis.channel if analysis.best_snr >= 2.5 else "gray"
    sc, _ = _scores_for(teacher_path, rgb, channel)
    sensitivity, on_teacher = _best_threshold(sc, rgb, analysis, truth, np.ones(truth.shape, dtype=bool))
    held_out: dict = {}
    cut = int(np.median(np.nonzero(truth)[1]))
    left = np.zeros(truth.shape, dtype=bool)
    left[:, :cut] = True
    part = analyze_mark(rgb, (truth & left).astype(np.uint8) * 255)
    if part is not None and (truth & ~left).any():
        part_channel = part.channel if part.best_snr >= 2.5 else "gray"
        part_sc = sc if part_channel == channel else _scores_for(teacher_path, rgb, part_channel)[0]
        part_sens, _ = _best_threshold(part_sc, rgb, part, truth, left)
        mask, _ = threshold_spots(part_sc, rgb, part_sens, part.size_range,
                                  part.target_lab, part.color_tolerance)
        held_out = {**_objects(mask & ~left, truth & ~left), "sensitivity": part_sens}
    return Calibration(analysis=analysis, channel=channel, sensitivity=sensitivity,
                       on_teacher=on_teacher, held_out=held_out)


def detect_like_teacher(img_path: str, teacher_path: str, teacher_mask_path: str | np.ndarray,
                        class_id: int = 1, sensitivity: int | None = None) -> dict:
    """Find the specks on one image that look like the person's on another.

    The answer has the shape detect_spots' has, with "like" saying how the
    example was measured on the teacher. mark_recall is None: there is no point.

    ``teacher_mask_path`` is the person's mask file, or the mask itself as a
    class-id array -- a tiled image's mask is an array with no file to name.
    """
    t0 = time.perf_counter()
    if isinstance(teacher_mask_path, np.ndarray):
        ids = teacher_mask_path
    else:
        try:
            with Image.open(teacher_mask_path) as img:
                ids = np.array(img)
        except (OSError, ValueError) as exc:
            logger.warning("spot_detect: failed to read mask %s: %s", teacher_mask_path, exc)
            raise ValueError("failed to read the person's mask") from exc
    ids = ids[..., 0] if ids.ndim == 3 else ids
    truth = ids == class_id
    if not truth.any():
        truth = (ids != 0) & (ids != 255)
    key = f"{teacher_path}::{hashlib.sha1(truth.tobytes()).hexdigest()[:16]}::v{ALGO_VERSION}"
    cal = _CALIBRATION_CACHE.get(key)
    if cal is None:
        cal = calibrate_on_teacher(teacher_path, truth)
        _CALIBRATION_CACHE.put(key, cal)
    rgb = _read_rgb(img_path)
    sc, cached = _scores_for(img_path, rgb, cal.channel)
    sens = int(sensitivity) if sensitivity else cal.sensitivity
    mask, count = threshold_spots(sc, rgb, sens, cal.analysis.size_range,
                                  cal.analysis.target_lab, cal.analysis.color_tolerance)
    h, w = rgb.shape[:2]
    return {
        "class_id": int(class_id),
        "mask": encode_png_base64((mask * class_id).astype(np.uint8)),
        "count": count,
        "sensitivity": sens,
        "mark_recall": None,
        "mode": "color" if cal.channel != "gray" else "dog",
        "channel": cal.channel,
        "size_range": list(cal.analysis.size_range),
        "color_tolerance": round(float(cal.analysis.color_tolerance), 2),
        "width": int(w),
        "height": int(h),
        "time_ms": int((time.perf_counter() - t0) * 1000),
        "scores_cached": cached,
        "like": {"sensitivity": cal.sensitivity, "on_teacher": cal.on_teacher,
                 "held_out": cal.held_out},
    }

