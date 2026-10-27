# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Crack Trace: oriented line filters, a centre-line test, hysteresis along centre lines.

Detects thin lines that are darker than the surface on both sides (cracks in concrete) or
lighter (scratches on film), from 1 px to about 25 px wide.  Returns a connected-component
label map so the frontend can select individual cracks by clicking, and a click-seeded tracer
for points where no candidate exists.

One map per image, from grey / 255 (no CLAHE: it lifts surface texture as much as cracks),
cached:
  responses: elongated second-derivative line filters (across-line sigma s, along-line 3 s,
    12 orientations), maximum over orientation, for dark lines and, negated, for light lines;
    s = 1, 2 on the image, 1.5, 2, 3 on a half-size copy and 2, 3 on a quarter-size copy, so
    the scales run 1, 2, 3, 4, 6, 8 and 12 px (a line about 2 s wide answers best at s)
  centre lines: a pixel is the centre of a dark (light) line at a scale when the image smoothed
    at that scale is a minimum (maximum) across the line -- the first derivative crosses zero
    there, along the normal the Hessian gives -- and the grey two sigmas out rises (falls) on
    both sides by comparable amounts (the shallower side at least 0.3 of the deeper).  The side
    lobes these filters give beside every line, the bright rim beside a crack and the foot of an
    edge are slopes or one-sided steps, not centres of a symmetric profile, so dark and light
    lines are kept on equal terms, with no ratio between the polarities.  (Keeping the dark
    response as it was and a light one only where it beat the nearby dark one let a wide light
    scratch's own lobes suppress it.)  A centre also needs a response of at least 1.5 grey
    levels, an answer from one orientation more than the others (the weakest under half the
    strongest: a dot answers all of them), and to be 2 px or more inside the frame (a frame
    edge mirrored by the border handling looks like a line along it).  A coarse scale's centre
    counts only where its response beats every finer scale's response of the same polarity
    nearby, so a thin line is never given a coarse scale's width; per pixel the scale with the
    strongest response wins
  strength: response / (Gauss_16(response) + mean response of the full-size image), the local
    term in each level's own pixels, in units of the full-size map's 95th percentile
global : hysteresis along the centre lines: high = 2.75 f, low = max(0.3 high, floor), with
         f = 2^((25 - sensitivity) / 35); the floor is the map's 90th percentile up to
         sensitivity 60, sliding to the 80th at 100.  Raising the slider adds fainter lines but
         the threshold never falls under the picture's ordinary texture (the earlier mapping put
         it under the median from about 45 and flooded the frame).  Candidates whose longer
         bbox side is under 40 px, or under 10 times their mean sigma, are dropped before any
         widening, so the width slider changes thickness only.  Where a line ends on another
         (a clean T: of three branches two leave in nearly opposite directions) the stem is its
         own candidate -- a scratch is not selected together with the frame line it touches;
         crossings, forks and tangles stay one candidate.  Each centre pixel then gets its
         width back, a disc of radius 0.75 x its scale, and the width slider dilates on top
click  : seed = the strongest centre pixel within 4 px of the click, or within the half-width
         of a wider line.  No centre there -> refused; a seed under 0.7 f -> refused (no stronger
         than the surface texture: a trace grown from it would spread over the surface).
         Otherwise selection = the centre lines >= max(1.0 f, 0.15 seed) connected to the seed
         (1 px gaps bridged), cut at a clean T as above, with their width.  If that is more
         than max(20,000 px, 2.5 % of the frame) the threshold is raised in 25 % steps up to the
         seed, and refused if still more; a selection whose centre line is shorter than 16 px is
         a spot, not a line, and refused
cost   : the filters run as one job per (pyramid level, horizontal strip, scale) in a pool of 4
         threads; filter2D and the numpy steps release the GIL
Only OpenCV (Apache-2.0), numpy (BSD) and Pillow (MIT-CMU).  No learned models.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from segcore.image_io import imread as _imread

from .cache_utils import ThreadSafeLRUCache

logger = logging.getLogger(__name__)

# Cache of the centre-line map (float32 strength + uint8 scale, ~4 MB per 1024x768 image, max 20)
_CRACK_MAP_CACHE = ThreadSafeLRUCache(maxsize=20)

# pyramid level -> across-line sigmas in that level's pixels (effective 1, 2, 3, 4, 6, 8, 12 px)
_LEVEL_SIGMAS = ((1.0, 2.0), (1.5, 2.0, 3.0), (2.0, 3.0))
_LINE_ELONG = 3.0                 # along-line sigma = 3 x across-line sigma
_LINE_ORIENTATIONS = 12
_DOT_RATIO = 0.5                  # a line answers one orientation: the weakest must stay under half the strongest
_SYM = 0.3                        # the shallower side at least this share of the deeper one
_SYM_DIST = 2.0                   # ... measured this many sigmas out
_BORDER_PX = 2                    # no centre lines this close to the frame border (each level)
_ABS_MIN = 1.5 / 255.0            # raw response floor: a line at least ~1.5 grey levels deep
_LOCALNORM = 16.0                 # local normalisation sigma, in each level's own pixels
_NORM_PCT = 95.0
_UNIT_MIN = 0.2                   # the 95th-percentile unit never below this (flat pictures)
_T_HIGH_AT_25 = 2.75              # candidate high threshold at sensitivity 25 (P95 units)
_T_LOW_RATIO = 0.3
_LOW_FLOOR_PCT = 90.0             # the low threshold never falls under this percentile of the response ...
_LOW_FLOOR_PCT_100 = 80.0         # ... up to sensitivity _FLOOR_S0, then down to this one at 100
_FLOOR_S0 = 60
_FLOOR_PCTS = tuple(float(x) for x in range(70, 96))
_SENS_STEP = 35.0                 # every _SENS_STEP sensitivity steps halve the high threshold
_MIN_EXTENT_PX = 40               # drop candidates whose longer bbox side is shorter (before widening)
_ASPECT = 10.0                    # ... or shorter than this many times their mean sigma
_JUNCTION_RING = 4                # junctions: three or more lines cross rings of this radius and twice it
_JOIN_COS = 0.8                   # ... branches leaving a junction within ~37 deg of opposite are one line
_JUNCTION_MAX_K = 2               # ... around centre pixels of the fine scales (sigma <= 3)
_WIDTH_GAIN = 0.75                # restored half-width = gain x winning sigma
_CLICK_RADIUS = 4
_CLICK_LINK_PX = 1
_CLICK_SEED_FRAC = 0.15
_CLICK_FLOOR_AT_25 = 1.0
_CLICK_SEED_MIN_AT_25 = 0.7
_CLICK_MAX_PX = 20000
_CLICK_MAX_FRAC = 0.025
_CLICK_MIN_EXTENT_PX = 16      # a click whose centre line is shorter than this is a dot, not a line
_THREADS = 4
_HALO = 40                        # strip overlap; >= the largest kernel radius (32)
_ALGO_VERSION = 6
_EPS = 1e-6

_KERNELS: dict = {}
_POOL = None
_POOL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Grayscale in [0, 1]."""
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    g = image_bgr.reshape(image_bgr.shape[:2]).astype(np.float32)
    return g / (65535.0 if image_bgr.dtype == np.uint16 else 255.0)


def _line_kernel(sigma_u: float, sigma_v: float, theta: float) -> np.ndarray:
    """Second derivative across the line (u) times a Gaussian along it (v); theta = normal angle.
    Positive on a dark line, zero DC, unit L1 mass per sign."""
    r = int(np.ceil(3.5 * max(sigma_u, sigma_v)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-u * u / (2 * sigma_u ** 2) - v * v / (2 * sigma_v ** 2))
    k = (u * u / sigma_u ** 4 - 1 / sigma_u ** 2) * g
    k -= k.mean()
    k /= np.abs(k).sum() / 2
    return k.astype(np.float32)


def _kernels(sigma: float):
    ks = _KERNELS.get(sigma)
    if ks is None:
        bank = [_line_kernel(sigma, _LINE_ELONG * sigma, np.pi * i / _LINE_ORIENTATIONS)
                for i in range(_LINE_ORIENTATIONS)]
        r = int(np.ceil(4 * sigma))
        x = np.arange(-r, r + 1, dtype=np.float64)
        g0 = np.exp(-x * x / (2 * sigma * sigma))
        g0 /= g0.sum()
        g1 = -x / sigma ** 2 * g0
        g2 = (x * x / sigma ** 4 - 1 / sigma ** 2) * g0
        g2 -= g2.mean()
        ks = (bank, g0.astype(np.float32), g1.astype(np.float32), g2.astype(np.float32))
        _KERNELS[sigma] = ks
    return ks


# neighbour offsets (dy, dx) of the four quantised normals: x, diagonal, y, anti-diagonal
_DIRS = ((0, 1), (1, 1), (1, 0), (-1, 1))
_LIGHT_BIT = 0x80


def _scale_maps(g: np.ndarray, s: float):
    """One scale on one strip: dense dark and light responses (max over orientation) and the
    same values kept only at centre-line pixels of each polarity."""
    h, w = g.shape
    bank, g0, g1, g2 = _kernels(s)
    hi = lo = None
    for k in bank:
        r = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
        if hi is None:
            hi, lo = r, r.copy()
        else:
            np.maximum(hi, r, out=hi)
            np.minimum(lo, r, out=lo)
    # geometry from the isotropic Gaussian at the same scale: the normal of a dark line is
    # the eigenvector of the larger Hessian eigenvalue, that of a light line the other one
    S = cv2.sepFilter2D(g, cv2.CV_32F, g0, g0, borderType=cv2.BORDER_REFLECT)
    a = cv2.sepFilter2D(g, cv2.CV_32F, g2, g0, borderType=cv2.BORDER_REFLECT)
    c = cv2.sepFilter2D(g, cv2.CV_32F, g0, g2, borderType=cv2.BORDER_REFLECT)
    b = cv2.sepFilter2D(g, cv2.CV_32F, g1, g1, borderType=cv2.BORDER_REFLECT)
    dd = a - c
    diag = np.abs(dd) < 2 * np.abs(b)
    qd = np.where(diag, np.where(b > 0, 1, 3), np.where(dd > 0, 0, 2)).astype(np.uint8)
    del a, c, b, dd, diag
    # symmetry: the grey 2 sigma out on either side must differ from the centre in the same
    # direction and by comparable amounts; the bright rim beside a crack, a side lobe or the
    # foot of an edge has one deep side and one shallow one
    dz = max(1, int(round(_SYM_DIST * s)))
    dg = max(1, int(round(_SYM_DIST * s / np.sqrt(2))))
    pz = max(dz, dg)
    Sp = cv2.copyMakeBorder(S, pz, pz, pz, pz, cv2.BORDER_REPLICATE)
    cd = np.zeros((h, w), bool)
    cl = np.zeros((h, w), bool)
    for q, (dy, dx) in enumerate(_DIRS):
        sa = Sp[pz + dy:pz + dy + h, pz + dx:pz + dx + w]
        sb = Sp[pz - dy:pz - dy + h, pz - dx:pz - dx + w]
        # centre of a dark line: the smoothed grey is a minimum across it (the first
        # derivative crosses zero here); a side lobe or an edge is a slope, not a minimum
        md = (qd == q) & (S <= sa) & (S <= sb)
        ml = (qd == (q + 2) % 4) & (S >= sa) & (S >= sb)
        if _SYM > 0:
            z = dg if dy and dx else dz
            ua = Sp[pz + z * dy:pz + z * dy + h, pz + z * dx:pz + z * dx + w] - S
            ub = Sp[pz - z * dy:pz - z * dy + h, pz - z * dx:pz - z * dx + w] - S
            mn = np.minimum(ua, ub)
            mx = np.maximum(ua, ub)
            md &= mn >= _SYM * mx
            ml &= mx <= _SYM * mn
        cd |= md
        cl |= ml
    del Sp, S, qd
    dark = hi
    light = np.negative(lo)
    cd &= (dark >= _ABS_MIN) & (lo < _DOT_RATIO * hi)
    cl &= (light >= _ABS_MIN) & (hi > _DOT_RATIO * lo)
    # one signed map: + dark centre, - light centre (the stronger where a pixel is both)
    cs = np.where(cd, dark, 0).astype(np.float32)
    vl = np.where(cl, light, 0)
    np.copyto(cs, -vl, where=vl > cs)
    return dark, light, cs


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=max(1, min(_THREADS, os.cpu_count() or 1)),
                                       thread_name_prefix="crack-trace")
        return _POOL


def _all_levels(levels):
    """Every (level, strip, scale) as one job in a small thread pool -- filter2D and the numpy
    steps release the GIL -- then per level: dense dark / light maxima over scale, and the
    centre-line strength C (max over scale and polarity) with K = scale index | _LIGHT_BIT."""
    jobs = []
    k0 = 0
    for li, (g, sig) in enumerate(zip(levels, _LEVEL_SIGMAS)):
        h = g.shape[0]
        n = max(1, min(2 * _THREADS, int(round(h / 400.0))))
        edges = np.linspace(0, h, n + 1).astype(int)
        for i in range(n):
            y0, y1 = int(edges[i]), int(edges[i + 1])
            p0, p1 = max(0, y0 - _HALO), min(h, y1 + _HALO)
            for si, s in enumerate(sig):
                jobs.append((li, y0, y1, p0, k0 + si, _pool().submit(_scale_maps, g[p0:p1], s)))
        k0 += len(sig)
    acc = []
    for g in levels:
        h, w = g.shape
        acc.append((np.zeros((h, w), np.float32), np.zeros((h, w), np.float32),
                    np.zeros((h, w), np.float32), np.zeros((h, w), np.uint8)))
    index = {fut: (li, y0, y1, p0, k) for li, y0, y1, p0, k, fut in jobs}
    del jobs
    for fut in as_completed(list(index)):
        li, y0, y1, p0, k = index.pop(fut)
        dark, light, cs = (x[y0 - p0:y1 - p0] for x in fut.result())
        del fut
        Dd, Dl, C, K = acc[li]
        sl = slice(y0, y1)
        np.maximum(Dd[sl], dark, out=Dd[sl])
        np.maximum(Dl[sl], light, out=Dl[sl])
        Cs, Ks = C[sl], K[sl]
        a = np.abs(cs)
        # ties go to the finer scale, whatever order the jobs finish in
        upd = (a > Cs) | ((a == Cs) & (a > 0) & ((Ks & 0x7F) > k))
        Cs[upd] = a[upd]
        Ks[upd] = np.where(cs[upd] < 0, k | _LIGHT_BIT, k)
        del dark, light, cs, a, upd
    return acc


def _crack_map(image_bgr: np.ndarray) -> dict:
    g = _preprocess(image_bgr)
    H, W = g.shape
    levels = [g]
    for _ in range(len(_LEVEL_SIGMAS) - 1):
        levels.append(cv2.pyrDown(levels[-1]))
    raw = _all_levels(levels)
    Dd0, Dl0 = raw[0][0], raw[0][1]
    mean0 = float(np.maximum(Dd0, Dl0).mean())
    C = K = R = None
    unit = floor = None
    for li, (Dd, Dl, Cr, Kl) in enumerate(raw):
        D = np.maximum(Dd, Dl)
        if _BORDER_PX:
            # a frame edge mirrored by the border handling looks like a line along the border
            b_ = _BORDER_PX
            Cr[:b_] = 0
            Cr[-b_:] = 0
            Cr[:, :b_] = 0
            Cr[:, -b_:] = 0
        N = cv2.GaussianBlur(D, (0, 0), _LOCALNORM) + (mean0 + _EPS)
        if li == 0:
            S = D / N
            st_ = max(1, int(np.sqrt(S.size / 1.0e6)))
            p = np.percentile(S[::st_, ::st_], (_NORM_PCT,) + _FLOOR_PCTS)
            unit = max(float(p[0]), _UNIT_MIN)
            floor = (p[1:] / unit).astype(np.float64)
            del S
        else:
            # a coarse centre counts only where its raw response beats every finer scale's
            # response of the same polarity nearby: beside a thin line the fine scales win,
            # so the thin line is not given a coarse scale's width
            hl, wl = Dd.shape
            fd = np.zeros((hl, wl), np.float32)
            fl = np.zeros((hl, wl), np.float32)
            for lj in range(li):
                kk = 2 ** (li - lj + 1) + 1
                ker = np.ones((kk, kk), np.uint8)
                for src, dst in ((raw[lj][0], fd), (raw[lj][1], fl)):
                    pooled = cv2.resize(cv2.dilate(src, ker), (wl, hl), interpolation=cv2.INTER_NEAREST)
                    np.maximum(dst, pooled, out=dst)
            light = (Kl & _LIGHT_BIT) > 0
            Cr = np.where(Cr > np.where(light, fl, fd), Cr, 0).astype(np.float32)
        Cn = Cr / (N * unit)
        if li:
            Cn = cv2.resize(Cn, (W, H), interpolation=cv2.INTER_NEAREST)
            Kl = cv2.resize(Kl, (W, H), interpolation=cv2.INTER_NEAREST)
            Cr = cv2.resize(Cr, (W, H), interpolation=cv2.INTER_NEAREST)
        if C is None:
            C, K, R = Cn.astype(np.float32), Kl.copy(), Cr.copy()
        else:
            upd = Cr > R
            C[upd] = Cn[upd]
            K[upd] = Kl[upd]
            R[upd] = Cr[upd]
    return {"C": C, "K": K & 0x7F, "floor": floor, "unit": unit}


def _sigmas_eff():
    out = []
    for li, sig in enumerate(_LEVEL_SIGMAS):
        out.extend(s * (2 ** li) for s in sig)
    return out


def _sens_factor(sensitivity: int) -> float:
    """sensitivity 1 (strict) .. 100 (permissive); 25 -> 1, every _SENS_STEP steps halve it."""
    s = int(min(100, max(1, sensitivity)))
    return 2.0 ** ((25 - s) / _SENS_STEP)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _link(b: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return b
    k = 2 * px + 1
    return cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def _floor(m: dict, sensitivity: int) -> float:
    """The low threshold's floor: a fixed percentile of this picture's response up to
    sensitivity _FLOOR_S0, a lower one beyond (so the slider keeps adding faint lines, not area)."""
    s = int(min(100, max(1, sensitivity)))
    q = _LOW_FLOOR_PCT - (_LOW_FLOOR_PCT - _LOW_FLOOR_PCT_100) * max(0, s - _FLOOR_S0) / (100.0 - _FLOOR_S0)
    return float(np.interp(q, _FLOOR_PCTS, m["floor"]))


def _centre_lines(m: dict, sensitivity: int) -> np.ndarray:
    """Centre-line pixels of the candidates: hysteresis along the centre lines, then the
    extent filter (before any widening)."""
    C = m["C"]
    f = _sens_factor(sensitivity)
    high = _T_HIGH_AT_25 * f
    low = max(_T_LOW_RATIO * high, _floor(m, sensitivity))
    high = max(high, low)
    weak = (C >= low).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[C >= high])] = True
    keep[0] = False
    if _MIN_EXTENT_PX:
        ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        need = np.full(n, float(_MIN_EXTENT_PX))
        if _ASPECT > 0:
            # a wide line must also be long: at least _ASPECT x its mean sigma
            on = weak > 0
            sig = np.asarray(_sigmas_eff(), np.float32)[m["K"][on]]
            cnt = np.bincount(lab[on], minlength=n)
            ssum = np.bincount(lab[on], weights=sig, minlength=n)
            need = np.maximum(need, _ASPECT * ssum / np.maximum(cnt, 1))
        keep &= ext >= need
    return (weak > 0) & keep[lab]


def _restore_width(centre: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Each centre-line pixel becomes a disc of radius gain x its winning sigma."""
    out = np.zeros(centre.shape, np.uint8)
    if not centre.any():
        return out
    for k, s in enumerate(_sigmas_eff()):
        mk = centre & (K == k)
        if not mk.any():
            continue
        r = int(round(_WIDTH_GAIN * s))
        mk = mk.astype(np.uint8)
        if r > 0:
            mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        out |= mk
    return out


def _ring(rho: int) -> np.ndarray:
    pts = []
    for t in np.linspace(0, 2 * np.pi, 8 * rho, endpoint=False):
        q = (int(round(rho * np.sin(t))), int(round(rho * np.cos(t))))
        if not pts or pts[-1] != q:
            pts.append(q)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return np.asarray(pts, np.int64)


def _branches(cen: np.ndarray, K: np.ndarray):
    """Labels 1..n of the centre lines.  Where three or more lines leave a small ring around a
    centre pixel (a scratch running into a frame line, a branch of a crack) the lines are cut,
    so each one is its own candidate; the cut pixels go back to the nearest branch."""
    cen8 = cen.astype(np.uint8)
    if not cen.any():
        n, lab = cv2.connectedComponents(cen8, connectivity=8)
        return lab, n - 1, set()
    rho = _JUNCTION_RING
    h, w = cen.shape
    pad = 2 * rho
    lp = cv2.copyMakeBorder(cv2.dilate(cen8, np.ones((3, 3), np.uint8)), pad, pad, pad, pad,
                            cv2.BORDER_CONSTANT, value=0)
    ys, xs = np.nonzero(cen & (K <= _JUNCTION_MAX_K))
    jn = np.ones(ys.size, bool)
    # three or more crossings on a ring of radius rho and on one of 2 rho: a short spur of
    # texture beside a line crosses the inner ring only
    for rr in (rho, 2 * rho):
        ring = _ring(rr)
        seq = lp[(ys + pad)[:, None] + ring[None, :, 0], (xs + pad)[:, None] + ring[None, :, 1]] > 0
        jn &= (seq & ~np.roll(seq, 1, axis=1)).sum(1) >= 3
    if not jn.any():
        n, lab = cv2.connectedComponents(cen8, connectivity=8)
        return lab, n - 1, set()
    J = np.zeros((h, w), np.uint8)
    J[ys[jn], xs[jn]] = 1
    cut = cv2.dilate(J, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rho + 1, 2 * rho + 1))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats((cen & ~cut).astype(np.uint8), connectivity=8)
    # pieces shorter than half the minimum extent (the forks at a line's ends, spurs) are not
    # branches of their own: they go back to the branch they touch, like the cut pixels
    small = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) < _MIN_EXTENT_PX // 2
    small[0] = False
    newid = np.zeros(n, np.int32)
    big = ~small
    big[0] = False
    newid[big] = np.arange(1, int(big.sum()) + 1, dtype=np.int32)
    lab = newid[lab]
    n = int(big.sum()) + 1
    sep = set()
    if n > 2:
        lab, n, sep = _join_through(lab, n, cut, rho)
    rest = cen & (lab == 0)
    if n > 1 and rest.any():
        # each left-over centre pixel takes the label of the nearest branch pixel
        _, near = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        lab[rest] = lab[lab > 0][near[rest] - 1]
    elif n == 1:
        n, lab = cv2.connectedComponents(cen8, connectivity=8)
    return lab, n - 1, sep


def _join_through(lab: np.ndarray, n: int, cut: np.ndarray, rho: int):
    """A line that runs on through a junction is one line: of the branches touching a cut
    zone, the two that leave it in the most nearly opposite directions are joined again (two
    such pairs at a crossing).  What is left -- the stem of a T, the side branch of a Y -- stays
    a candidate of its own.  A zone touched by two branches only is no junction: they are joined."""
    zn, zl, _, zc = cv2.connectedComponentsWithStats(cut.astype(np.uint8), connectivity=8)
    if zn <= 1:
        return lab, n, set()
    # nearest zone and the distance to it, for every branch pixel (one pass)
    dist, near_cc = cv2.distanceTransformWithLabels((~cut).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_CCOMP)
    cc2z = np.zeros(int(near_cc.max()) + 1, np.int64)
    cc2z[near_cc[cut]] = zl[cut]
    on = (lab > 0) & (dist <= 3 * rho)
    yy, xx = np.nonzero(on)
    zid = cc2z[near_cc[yy, xx]]
    key = zid * n + lab[yy, xx]
    tk = np.unique(key[dist[yy, xx] <= 2.5])
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mx = np.bincount(inv, weights=xx) / cnt
    my = np.bincount(inv, weights=yy) / cnt
    keep = np.isin(uk, tk)
    uk, mx, my = uk[keep], mx[keep], my[keep]
    kz, kb = uk // n, uk % n
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    apart = []
    order = np.argsort(kz, kind="stable")
    kz, kb, mx, my = kz[order], kb[order], mx[order], my[order]
    starts = np.flatnonzero(np.r_[True, kz[1:] != kz[:-1]])
    ends = np.r_[starts[1:], kz.size]
    for a0, a1 in zip(starts, ends):
        if a1 - a0 < 2:
            continue
        z = kz[a0]
        b = kb[a0:a1]
        v = np.stack([mx[a0:a1] - zc[z, 0], my[a0:a1] - zc[z, 1]], 1)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        d = v @ v.T
        allp = [(i, j) for i in range(b.size) for j in range(i + 1, b.size)]
        if b.size != 3:
            pairs = allp          # no junction, a crossing or a tangle: one candidate
        else:
            cand = sorted((float(d[i, j]), i, j) for i, j in allp if d[i, j] <= -_JOIN_COS)
            used, pairs = set(), []
            for _, i, j in cand:
                if i not in used and j not in used:
                    used.update((i, j))
                    pairs.append((i, j))
            # a clean T (one through pair + a stem) is cut; a crossing, a Y without a through
            # line, or anything else stays one candidate
            if not (b.size == 3 and len(pairs) == 1):
                pairs = allp
            else:
                paired = {i for pr in pairs for i in pr}
                grp = [list(pr) for pr in pairs] + [[i] for i in range(b.size) if i not in paired]
                for ga in range(len(grp)):
                    for gb in range(ga + 1, len(grp)):
                        apart.append((int(b[grp[ga][0]]), int(b[grp[gb][0]]), float(zc[z, 0]), float(zc[z, 1])))
        for i, j in pairs:
            ri, rj = find(int(b[i])), find(int(b[j]))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    ur, newid = np.unique(roots, return_inverse=True)
    # label 0 stays 0 (its root is 0, the smallest)
    sep = set()
    for a, b, cx, cy in apart:
        ra, rb = sorted((int(newid[find(a)]), int(newid[find(b)])))
        if ra != rb:
            sep.add((ra, rb, int(round(cx)), int(round(cy))))
    return newid[lab].astype(np.int32), int(ur.size), sep


def _restore_labels(lab: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Labelled centre lines -> labelled lines: each centre pixel becomes a disc of radius
    gain x its winning sigma, carrying its label (finer scales first)."""
    out = lab.astype(np.int32).copy()
    on = lab > 0
    if not on.any():
        return out
    h, w = lab.shape
    ys, xs = np.nonzero(on)
    ks = K[ys, xs]
    for k, s in enumerate(_sigmas_eff()):
        r = int(round(_WIDTH_GAIN * s))
        sel = ks == k
        if r <= 0 or not sel.any():
            continue
        y0, y1 = max(0, int(ys[sel].min()) - r), min(h, int(ys[sel].max()) + r + 1)
        x0, x1 = max(0, int(xs[sel].min()) - r), min(w, int(xs[sel].max()) + r + 1)
        src = np.zeros((y1 - y0, x1 - x0), np.float32)
        src[ys[sel] - y0, xs[sel] - x0] = lab[ys[sel], xs[sel]]
        d = cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        o = out[y0:y1, x0:x1]
        fill = (o == 0) & (d > 0)
        o[fill] = d[fill].astype(np.int32)
    return out


def _group(lab: np.ndarray, sep: set, reach: int = 0) -> tuple[np.ndarray, int]:
    """Candidates = connected parts of the labelled lines, as without cuts, except that two
    labels separated at a junction never touch: a one-pixel moat is cut between them, in a
    window around that junction."""
    on = lab > 0
    if sep:
        h, w = lab.shape
        R = 4 * _JUNCTION_RING + int(np.ceil(_WIDTH_GAIN * max(_sigmas_eff()))) + reach + 4
        for a_, b_, cx, cy in sep:
            y0, y1, x0, x1 = max(0, cy - R), min(h, cy + R + 1), max(0, cx - R), min(w, cx + R + 1)
            L = lab[y0:y1, x0:x1]
            A = L == a_
            Bm = L == b_
            if not A.any() or not Bm.any():
                continue
            # the moat: pixels of b that touch a pixel of a (8-neighbourhood)
            near_a = cv2.dilate(A.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            on[y0:y1, x0:x1] &= ~(Bm & near_a)
        lab2, n2 = _label(on)
        cnt = np.bincount(lab2.ravel())
        if (cnt[1:] < 5).any():
            # single pixels the moat cut off from their line
            small = cnt < 5
            small[0] = False
            on &= ~small[lab2]
            lab2, n2 = _label(on)
        # a line the moat happened to cut through is still one line: parts that carry the
        # same branch label are one candidate
        pr = np.unique(lab2[on].astype(np.int64) * (int(lab.max()) + 1) + lab[on])
        comp, br = pr // (int(lab.max()) + 1), pr % (int(lab.max()) + 1)
        parent = np.arange(n2 + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        first = {}
        for c_, b_ in zip(comp.tolist(), br.tolist()):
            if b_ in first:
                ra, rb = find(first[b_]), find(c_)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                first[b_] = c_
        roots = np.array([find(i) for i in range(n2 + 1)])
        ur, newid = np.unique(roots, return_inverse=True)
        return newid[lab2].astype(np.int32), int(ur.size - 1)
    return _label(on)


def _widen_labels(lab: np.ndarray, width_px: int) -> np.ndarray:
    """The width slider on a label map: the widened pixels take the nearest label."""
    if width_px <= 0:
        return lab
    binary = _widen((lab > 0).astype(np.uint8), width_px)
    reach = width_px + max(3, (width_px // 2) * 2 + 1)
    d = cv2.dilate(lab.astype(np.float32),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    out = lab.copy()
    fill = (binary > 0) & (lab == 0)
    out[fill] = d[fill].astype(np.int32)
    return out


def _label(binary: np.ndarray) -> tuple[np.ndarray, int]:
    n, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return lab.astype(np.int32), int(n - 1)


def _widen(binary: np.ndarray, width_px: int) -> np.ndarray:
    """The existing width slider: dilate by width_px, then close to smooth the edges."""
    if width_px <= 0:
        return binary
    ksize = width_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary = cv2.dilate(binary, kernel, iterations=1)
    smooth_k = max(3, (width_px // 2) * 2 + 1)
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_k, smooth_k))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel)


def _encode_label_map(label_map: np.ndarray) -> str:
    """Encode label map as RGBA PNG base64 (same format as superpixel)."""
    h, w = label_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (label_map & 0xFF).astype(np.uint8)
    rgba[:, :, 1] = ((label_map >> 8) & 0xFF).astype(np.uint8)
    rgba[:, :, 3] = 255
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _get_map(img_path: str) -> tuple[dict, bool]:
    cache_key = f"{img_path}::v{_ALGO_VERSION}"
    cached = _CRACK_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached, True
    image = _imread(img_path)
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")
    m = _crack_map(image)
    _CRACK_MAP_CACHE.put(cache_key, m)
    return m, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crack_trace_compute(
    img_path: str,
    sensitivity: int = 25,
    width_px: int = 0,
) -> dict:
    """Full pipeline: centre-line map -> hysteresis -> extent filter -> width -> label -> encode.

    Caches the centre-line map so slider changes and later clicks are fast.
    """
    t0 = time.perf_counter()
    m, map_cached = _get_map(img_path)
    lab, _, sep = _branches(_centre_lines(m, sensitivity), m["K"])
    reach = width_px + max(3, (width_px // 2) * 2 + 1) if width_px > 0 else 0
    label_map, n_cracks = _group(_widen_labels(_restore_labels(lab, m["K"]), width_px), sep, reach)
    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("crack_trace: %d cracks, %dms, cached=%s", n_cracks, elapsed, map_cached)
    return {
        "label_map_b64": _encode_label_map(label_map),
        "n_cracks": n_cracks,
        "time_ms": elapsed,
        "crack_map_cached": map_cached,
    }


def _refused(t0: float, map_cached: bool, why: str) -> dict:
    return {"label_map_b64": None, "n_cracks": 0, "time_ms": int((time.perf_counter() - t0) * 1000),
            "crack_map_cached": map_cached, "why": why}


def _seed_region(C: np.ndarray, K: np.ndarray, sy: int, sx: int, thr: float) -> np.ndarray:
    b = (C >= thr).astype(np.uint8)
    b[sy, sx] = 1
    _, cc = cv2.connectedComponents(_link(b, _CLICK_LINK_PX), connectivity=8)
    sel = (b > 0) & (cc == cc[sy, sx])
    # cut at a clean T like the global candidates, and keep the part holding the seed
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = max(0, ys.min() - 1), ys.max() + 2, max(0, xs.min() - 1), xs.max() + 2
    lab, _, sep = _branches(sel[y0:y1, x0:x1], K[y0:y1, x0:x1])
    if sep:
        g, _ = _group(_restore_labels(lab, K[y0:y1, x0:x1]), sep)
        v = g[sy - y0, sx - x0]
        if v > 0:
            keep = np.zeros_like(sel)
            keep[y0:y1, x0:x1] = (g == v) & sel[y0:y1, x0:x1]
            sel = keep
    return sel


def crack_trace_adaptive(
    img_path: str,
    click_x: int,
    click_y: int,
    sensitivity: int = 25,
    width_px: int = 0,
) -> dict:
    """Crack tracing seeded by a click where no candidate exists.

    Returns a *patch* label map (label 1) that the frontend merges into
    unlabelled pixels, or label_map_b64 None with the reason in "why".
    """
    t0 = time.perf_counter()
    m, map_cached = _get_map(img_path)
    C, K = m["C"], m["K"]
    h, w = C.shape
    cx = max(0, min(int(click_x), w - 1))
    cy = max(0, min(int(click_y), h - 1))

    # the seed: the strongest centre-line pixel whose line covers the click -- within
    # _CLICK_RADIUS, or within the restored half-width of a wider line
    radii = np.maximum(_CLICK_RADIUS, np.round(_WIDTH_GAIN * np.asarray(_sigmas_eff()))).astype(np.int32)
    r = int(radii.max())
    y0, y1, x0, x1 = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rk = radii[K[y0:y1, x0:x1]]
    cw = C[y0:y1, x0:x1]
    win = np.where(((yy - cy) ** 2 + (xx - cx) ** 2 <= rk * rk) & (cw > 0), cw, -1)
    k = int(np.argmax(win))
    sy, sx = y0 + k // win.shape[1], x0 + k % win.shape[1]
    seed = float(win.flat[k])
    if seed <= 0:
        return _refused(t0, map_cached, "No line here: click on the centre of the line. "
                        f"Nothing within {_CLICK_RADIUS} px of the click (or within the half-width of "
                        "a wider line) is the centre of a line darker or lighter than both its sides.")
    f = _sens_factor(sensitivity)
    if seed < _CLICK_SEED_MIN_AT_25 * f:
        return _refused(t0, map_cached, "Too faint to trace: click a clearer part, or raise the "
                        "sensitivity. "
                        f"Within {_CLICK_RADIUS} px of the click the line is no stronger than this "
                        "picture's ordinary surface texture, so a trace grown from it would spread over "
                        "the surface.")
    cap = max(_CLICK_MAX_PX, int(_CLICK_MAX_FRAC * h * w))
    thr = min(max(_CLICK_FLOOR_AT_25 * f, _CLICK_SEED_FRAC * seed), seed)
    sel = _seed_region(C, K, sy, sx, thr)
    region = _restore_width(sel, K)
    while int(region.sum()) > cap and thr < seed:
        thr = min(seed, thr * 1.25)
        sel = _seed_region(C, K, sy, sx, thr)
        region = _restore_width(sel, K)
    ys, xs = np.nonzero(sel)
    if max(int(np.ptp(ys)), int(np.ptp(xs))) + 1 < _CLICK_MIN_EXTENT_PX:
        return _refused(t0, map_cached, "Only a spot here, not a line: click on a line. "
                        f"What lies within {_CLICK_RADIUS} px of the click is a spot under "
                        f"{_CLICK_MIN_EXTENT_PX} px long, not a line.")
    if int(region.sum()) > cap:
        return _refused(t0, map_cached, "A broad area, not a line: use the brush or SAM here. "
                        "At that point the line does not stand out from the surface, and even the "
                        f"tightest threshold that keeps the click would take more than {cap:,} pixels.")
    label_map = _widen(region, width_px).astype(np.int32)
    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("crack_trace_adaptive: click=(%d,%d) seed=%.3f, %dpx, %dms",
                cx, cy, seed, int(np.count_nonzero(label_map)), elapsed)
    return {
        "label_map_b64": _encode_label_map(label_map),
        "n_cracks": 1,
        "time_ms": elapsed,
        "crack_map_cached": map_cached,
    }
