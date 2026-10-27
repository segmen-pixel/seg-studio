# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Mask statistics: hand masks and drafts apart, drafts outside the hand
range flagged."""
from __future__ import annotations

import numpy as np

from app.core.mask_stats import describe, summarise_masks


def _cells(n: int, size: int = 64) -> np.ndarray:
    arr = np.full((size, size), 255, np.uint8)
    for i in range(n):
        y, x = 2 + (i // 8) * 7, 2 + (i % 8) * 7
        arr[y:y + 3, x:x + 3] = 1
    return arr


def test_describe_ignores_background_and_ignore_values():
    d = describe(_cells(5))
    assert d["regions"] == 5 and d["class_ids"] == [1] and 0 < d["area_frac"] < 0.05


def test_drafts_outside_the_hand_range_are_flagged():
    rows = [("h1", "h1.png", False, _cells(10)), ("h2", "h2.png", False, _cells(12)),
            ("d1", "d1.png", True, _cells(11)), ("d2", "d2.png", True, _cells(2)),
            ("d3", "d3.png", True, _cells(40))]
    s = summarise_masks(rows)
    assert s["hand"]["n"] == 2 and s["drafts"]["n"] == 3
    assert s["hand"]["regions"] == {"min": 10, "median": 11.0, "max": 12}
    names = {o["name"]: o["why"] for o in s["outliers"]}
    assert set(names) == {"d2.png", "d3.png"}
    assert any("< hand min" in w for w in names["d2.png"])
    assert any("> hand max" in w for w in names["d3.png"])
    assert s["n_outliers"] == 2
    assert s["outliers"][0]["name"] == "d3.png", "the furthest from the hand range comes first"


def test_area_alone_flags_only_a_large_drift():
    """Hand masks that are all alike make a narrow area band; a draft with
    the right count and an area a little outside it is not an outlier."""
    hand = [("h1", "h1.png", False, _cells(10)), ("h2", "h2.png", False, _cells(10))]
    close = np.full((64, 64), 255, np.uint8)
    for i in range(10):                      # same count, slightly bigger cells
        y, x = 2 + (i // 8) * 7, 2 + (i % 8) * 7
        close[y:y + 4, x:x + 4] = 1
    s = summarise_masks(hand + [("d1", "d1.png", True, close)])
    assert s["outliers"] == []
    huge = np.full((64, 64), 255, np.uint8)
    for i in range(10):
        y, x = 2 + (i // 8) * 7, 2 + (i % 8) * 7
        huge[y:y + 6, x:x + 6] = 1      # 4x the area, same count
    s = summarise_masks(hand + [("d2", "d2.png", True, huge)])
    assert s["n_outliers"] == 1 and "twice the hand max" in s["outliers"][0]["why"][0]


def test_no_hand_masks_means_no_outliers():
    s = summarise_masks([("d1", "d1.png", True, _cells(3))])
    assert s["hand"]["n"] == 0 and s["hand"]["regions"] is None and s["outliers"] == []


def _png(arr):
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_an_agent_s_write_is_not_a_hand_mask(client, project_with_image):
    """Split on the draft flag alone, a run's own unflagged spot_write masks
    were counted as hand, and the range it was measured against was its own."""
    project_id, item_id = project_with_image
    url = f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png"
    stats = f"/api/v1/projects/{project_id}/datasets/annotate/mask-stats"
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 1
    assert client.put(url, files={"file": ("m.png", _png(mask), "image/png")}).status_code == 200
    assert client.get(stats).json()["hand"]["n"] == 1, "a person's save is theirs"
    mask[8:12, 8:12] = 1
    r = client.put(url + "?overwrite=1", files={"file": ("m.png", _png(mask), "image/png")},
                   headers={"X-Seg-Agent": "mcp/write"})
    assert r.status_code == 200, r.text
    s = client.get(stats).json()
    assert (s["hand"]["n"], s["drafts"]["n"]) == (0, 1), s
    assert s["made_by"] == {"hand": 0, "agent": 1, "draft": 0}, s["made_by"]


def test_a_blank_mask_file_is_nobody_s_mask(client, project_with_image):
    project_id, item_id = project_with_image
    url = f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png"
    assert client.put(url, files={"file": ("m.png", _png(np.zeros((16, 16), np.uint8)), "image/png")}).status_code == 200
    s = client.get(f"/api/v1/projects/{project_id}/datasets/annotate/mask-stats").json()
    assert (s["hand"]["n"], s["drafts"]["n"]) == (0, 0), s


def test_who_made_each_mask_decides_the_groups():
    """An agent's write nobody flagged is machine-made, and not a draft."""
    rows = [("h1", "h1.png", False, _cells(10), "hand"),
            ("a1", "a1.png", False, _cells(11), "agent"),
            ("d1", "d1.png", True, _cells(2), "draft")]
    s = summarise_masks(rows)
    assert s["made_by"] == {"hand": 1, "agent": 1, "draft": 1} and s["hand_ids"] == ["h1"], s
    assert s["hand"]["n"] == 1 and s["drafts"]["n"] == 2
    rows_by_id = {r["item_id"]: r for r in s["per_image"]}
    assert rows_by_id["a1"]["draft"] is False and rows_by_id["a1"]["made_by"] == "agent"
