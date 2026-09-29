# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the masks of a project look like, hand ones and drafts apart.

A draft that a run wrote is only useful if a person can find the ones that
need them. Area and region count against the range of the hand masks is the
cheapest honest signal: a draft far outside the hand range is where to look.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def describe(arr: np.ndarray) -> dict:
    fg = (arr != 0) & (arr != 255)
    return {
        "area_frac": round(float(fg.mean()), 5),
        "regions": int(ndimage.label(fg)[1]),
        "class_ids": sorted(int(v) for v in np.unique(arr) if v not in (0, 255)),
    }


def _range(rows: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in rows]
    if not vals:
        return None
    return {"min": min(vals), "median": float(np.median(vals)), "max": max(vals)}


def summarise_masks(rows: list[tuple]) -> dict:
    """rows: (item_id, name, is_draft, class-id array[, made_by]).

    made_by is "hand", "agent" or "draft": who made the mask. Given, it decides
    the groups -- hand is a person's, everything else machine-made and under
    "drafts" -- and each image says it; is_draft is then the index's own draft
    flag, so an agent's write nobody flagged is not called a draft. Without it,
    is_draft alone decides, as it always did.
    """
    hand, drafts = [], []
    made: dict[str, list[str]] = {"hand": [], "agent": [], "draft": []}
    for row in rows:
        item_id, name, is_draft, arr = row[:4]
        who = row[4] if len(row) > 4 else ("draft" if is_draft else "hand")
        made.setdefault(who, []).append(item_id)
        d = {"item_id": item_id, "name": name, "draft": bool(is_draft), "made_by": who, **describe(arr)}
        (hand if who == "hand" else drafts).append(d)
    out = {
        "made_by": {k: len(v) for k, v in made.items()},
        "hand_ids": made["hand"][:20],
        "hand": {"n": len(hand), "area_frac": _range(hand, "area_frac"),
                 "regions": _range(hand, "regions")},
        "drafts": {"n": len(drafts), "area_frac": _range(drafts, "area_frac"),
                   "regions": _range(drafts, "regions")},
        "outliers": [],
        "per_image": hand + drafts,
    }
    if hand and drafts:
        # Region count first: it is what a person checks at a glance and it
        # survives a homogeneous teacher set. Area alone is too strict -- when
        # the hand masks all cover nearly the same share of the frame, most
        # drafts fall outside that narrow range while their counts are fine.
        # A draft is an outlier for its count, or for its area only when the
        # count also drifted or the area is off by more than half.
        lo_r, hi_r = out["hand"]["regions"]["min"], out["hand"]["regions"]["max"]
        lo_a, hi_a = out["hand"]["area_frac"]["min"], out["hand"]["area_frac"]["max"]
        for d in drafts:
            why = []
            if d["regions"] < lo_r:
                why.append(f"regions {d['regions']} < hand min {lo_r}")
            elif d["regions"] > hi_r:
                why.append(f"regions {d['regions']} > hand max {hi_r}")
            if d["area_frac"] < lo_a / 2:
                why.append(f"area {100*d['area_frac']:.2f}% < half the hand min {100*lo_a:.2f}%")
            elif d["area_frac"] > hi_a * 2:
                why.append(f"area {100*d['area_frac']:.2f}% > twice the hand max {100*hi_a:.2f}%")
            if why:
                out["outliers"].append({"item_id": d["item_id"], "name": d["name"], "why": why,
                                        "regions": d["regions"], "area_frac": d["area_frac"]})
        # the furthest from the hand range first
        def _dist(o):
            r = o["regions"]
            return max(lo_r - r, r - hi_r, 0) / max(hi_r, 1)
        out["outliers"].sort(key=lambda o: (-_dist(o), o["name"] or ""))
    out["n_outliers"] = len(out["outliers"])
    return out
