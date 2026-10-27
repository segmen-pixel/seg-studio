# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A mask may not be cached blind, and may not be re-sent for nothing.

It changes under the browser -- a person paints, the bridge labels -- so a
minute of max-age could hand back the mask from before a write. Sending the
whole PNG on every switch between images is the other extreme, and clicking
through a project is exactly what a person does all day.
"""
from __future__ import annotations

import numpy as np

from .conftest import make_mask_png


def _painted(size=24):
    arr = np.zeros((size, size), np.uint8)
    arr[4:12, 4:12] = 1
    return arr


def test_an_unchanged_mask_is_a_304_with_no_body(client, project_with_image):
    project_id, item_id = project_with_image
    saved = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files=[("file", ("m.png", make_mask_png(_painted()), "image/png"))])
    assert saved.status_code == 200
    url = f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png"
    first = client.get(url)
    assert first.status_code == 200 and first.content
    etag = first.headers["etag"]
    again = client.get(url, headers={"If-None-Match": etag})
    assert again.status_code == 304 and not again.content
    assert again.headers["cache-control"] == "no-cache"


def test_a_mask_that_changed_comes_back_whole(client, project_with_image):
    project_id, item_id = project_with_image
    url = f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png"
    client.put(url, files=[("file", ("m.png", make_mask_png(_painted()), "image/png"))])
    etag = client.get(url).headers["etag"]
    other = _painted()
    other[16:20, 16:20] = 1
    client.put(url, files=[("file", ("m.png", make_mask_png(other), "image/png"))])
    got = client.get(url, headers={"If-None-Match": etag})
    assert got.status_code == 200 and got.content, "the mask the person is looking for"
    assert got.headers["etag"] != etag
