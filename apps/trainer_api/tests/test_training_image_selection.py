# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Which images a training run is allowed to learn from.

An image with no painted foreground can mean two different things, and the
difference decides whether the model is taught that its target is absent:

* the user looked and declared it good ("Mark Clean") -- a real negative
* nobody has painted it yet -- no statement about the image at all

A third state was left open longer: an image with no mask file at all. It is
the same "nobody has judged this" as an unpainted mask, stated more weakly
still, and it trained as pure background by default. Unannotated photos of the
same scene, outnumbering the annotated ones several to one, taught the model
that the object is background -- while the composite validation of the same
data scored high.

These used to be one pool, and that can cost the whole run: unpainted copies
of the annotated photo and unannotated defect photos outnumbering the
annotated ones several times over teach the model to answer background
everywhere. Its highest foreground probability stays under the shipped
threshold -- it cannot fire -- while the run still reports a high F1, because
"background" is nearly right when a tiny share of pixels are foreground.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.core.dataset_prep import prepare_annotate_dataset


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, pid, name):
    img = Image.new("RGB", (32, 32), color=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/upload",
        files=[("files", (name, buf.getvalue(), "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["items"][0]["id"]


def _put_mask(client, pid, item_id, arr):
    resp = client.put(
        f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png",
        files={"file": ("mask.png", _png(arr), "image/png")},
    )
    assert resp.status_code == 200, resp.text


def _splits(pid):
    from app.core.paths import prepared_dir
    out = {}
    for name in ("train", "val", "test"):
        f = prepared_dir(pid) / "splits" / f"{name}.txt"
        out[name] = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def _three_images(client, pid):
    """painted / declared-clean / empty-but-undeclared."""
    painted = _upload(client, pid, "painted.png")
    declared = _upload(client, pid, "declared.png")
    undeclared = _upload(client, pid, "undeclared.png")

    fg = np.zeros((32, 32), np.uint8)
    fg[8:24, 8:24] = 1
    _put_mask(client, pid, painted, fg)

    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/mark-clean",
                       json={"image_ids": [declared]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["marked"] == 1

    # An all-zero mask saved by painting and erasing: hasMask is set, nothing
    # was declared. This is the state that used to become a negative.
    _put_mask(client, pid, undeclared, np.zeros((32, 32), np.uint8))
    return painted, declared, undeclared


def test_an_undeclared_empty_mask_is_left_out_of_the_dataset(client, project_id):
    painted, declared, undeclared = _three_images(client, project_id)
    report = prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)

    everywhere = sum(_splits(project_id).values(), [])
    assert painted in everywhere, "a painted image must be trained on"
    assert declared in everywhere, "Mark Clean is a declaration; it is a negative"
    assert undeclared not in everywhere, (
        "an empty mask nobody declared clean is 'not annotated yet', and "
        "training it as background teaches the model its target is absent"
    )
    assert report["clean_declared"] == 1
    assert report["undeclared_excluded"] == 1


def test_the_excluded_image_is_not_even_copied(client, project_id):
    """Exclusion is from the dataset, not just from the split file."""
    from app.core.paths import prepared_dir
    _painted, _declared, undeclared = _three_images(client, project_id)
    prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)
    masks = prepared_dir(project_id) / "masks"
    assert not (masks / f"{undeclared}.png").exists()


def test_an_all_ignore_mask_is_not_mistaken_for_a_declaration(client, project_id):
    """255 everywhere is unpainted, not clean.

    The ignore value is rewritten to background further down, so a check made
    after that rewrite cannot tell this apart from a real declaration.
    """
    item = _upload(client, project_id, "all_ignore.png")
    _put_mask(client, project_id, item, np.full((32, 32), 255, np.uint8))
    # Something must survive, or the split has nothing to rank.
    keeper = _upload(client, project_id, "keeper.png")
    fg = np.zeros((32, 32), np.uint8)
    fg[4:12, 4:12] = 1
    _put_mask(client, project_id, keeper, fg)

    report = prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)
    everywhere = sum(_splits(project_id).values(), [])
    assert keeper in everywhere
    assert item not in everywhere
    assert report["undeclared_excluded"] == 1


def test_a_painted_image_is_kept_even_if_it_was_marked_clean_before(client, project_id):
    """The mask on disk decides, not the flag: the flag can lag behind."""
    item = _upload(client, project_id, "repainted.png")
    resp = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/mark-clean",
                       json={"image_ids": [item]})
    assert resp.status_code == 200
    fg = np.zeros((32, 32), np.uint8)
    fg[10:20, 10:20] = 1
    _put_mask(client, project_id, item, fg)

    report = prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)
    assert item in sum(_splits(project_id).values(), [])
    assert report["undeclared_excluded"] == 0


def test_an_image_with_no_mask_at_all_is_left_out(client, project_id):
    """No mask is not a declaration either -- it is a weaker one than an empty mask.

    This used to be an opt-out (include_unmasked, defaulted on by the request
    schema and the form), so the images trained as pure background unless
    somebody knew to turn them off.
    """
    painted, declared, _undeclared = _three_images(client, project_id)
    never_touched = _upload(client, project_id, "never_touched.png")

    report = prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)

    everywhere = sum(_splits(project_id).values(), [])
    assert painted in everywhere
    assert declared in everywhere
    assert never_touched not in everywhere, (
        "an image nobody has painted must not be trained as background"
    )
    assert report["without_mask"] == 1
    assert report["unmasked_excluded"] == 1


def test_the_unmasked_image_is_not_copied_either(client, project_id):
    from app.core.paths import prepared_dir
    _three_images(client, project_id)
    never_touched = _upload(client, project_id, "never_touched.png")
    prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)

    prepared = prepared_dir(project_id)
    assert not (prepared / "masks" / f"{never_touched}.png").exists(), (
        "an all-zero mask written for an unjudged image is the bug itself"
    )


def test_only_two_kinds_of_image_reach_a_split(client, project_id):
    """The whole rule, in one assertion: paint says NG, Mark Clean says OK."""
    painted, declared, undeclared = _three_images(client, project_id)
    never_touched = _upload(client, project_id, "never_touched.png")

    prepare_annotate_dataset(project_id, val_ratio=0.0, test_ratio=0.0)
    everywhere = set(sum(_splits(project_id).values(), []))

    assert everywhere == {painted, declared}, (
        f"only paint and Mark Clean may train; got {sorted(everywhere)}"
    )
    assert undeclared not in everywhere
    assert never_touched not in everywhere
