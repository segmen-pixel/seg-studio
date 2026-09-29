# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A project exported and imported again comes back as it left.

Two things were lost on the way.

* The import took every picture outside a ``masks/`` folder as an image, at
  any depth. An export carries its training runs under
  ``training/runs/<run>/``, and a run's reliability charts are PNGs, so the
  charts came back among the photographs.
* metadata.json named each image and its mask and nothing more. An OK mark is
  an all-background mask plus a person's word for it; the mask alone reads as
  an image nobody has labelled, so every OK was lost.

Two more things went missing on the way. A large image's mask is kept in
chunks and the export read only PNGs, so it went out as the blank placeholder
or as an older PNG copy. And an export extracted with Windows' Extract All and
zipped again sits a folder deeper than the import looked, so it was refused.

An export written before this change has no marks in its metadata and must
import as it always did, and so must a plain zip of pictures.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from app.core import annotate_index as idx
from app.core.paths import annotate_masks_dir, project_dir, runs_root_of
from app.routers.datasets import _plan_zip_import

SIZE = 16
TOP = "sample_20260101_0000"
CLASSES = [
    {"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
    {"id": 1, "name": "scratch", "color": [242, 36, 36], "active": True},
    {"id": 2, "name": "dent", "color": [36, 36, 242], "active": True},
]
# What the export writes now: img001/img002 painted, img003/img004 marked OK,
# img005 never labelled (masks/ holds the export's blank for it), img002
# flagged for review, two images placed in train / test by hand.
MARKED_ITEMS = [
    {"id": "img001", "filename": "img001.png", "hasMask": True, "markedClean": False,
     "name": "img001.jpg", "set": "train"},
    {"id": "img002", "filename": "img002.png", "hasMask": True, "markedClean": False,
     "draft": True, "draftRun": "review", "draftReason": "edge unclear"},
    {"id": "img003", "filename": "img003.png", "hasMask": True, "markedClean": True},
    {"id": "img004", "filename": "img004.png", "hasMask": True, "markedClean": True, "set": "test"},
    {"id": "img005", "filename": "img005.png", "hasMask": False, "markedClean": False},
]
# What every export before it wrote.
OLD_ITEMS = [{"id": it["id"], "filename": it["filename"]} for it in MARKED_ITEMS]
PAINT = {"img001": 1, "img002": 2}


def _picture(colour=(120, 90, 60), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (SIZE, SIZE), colour).save(buf, format=fmt)
    return buf.getvalue()


def _mask(class_id=0):
    arr = np.zeros((SIZE, SIZE), np.uint8)
    if class_id:
        arr[4:9, 4:9] = class_id
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _export_zip(items):
    """Laid out as the export writes it: one top folder holding images/,
    masks/ (a mask for every image), metadata.json, the splits and a training
    run whose charts are PNGs and which keeps a classes.json of its own."""
    members = []
    for n, it in enumerate(items):
        members.append((f"{TOP}/images/{it['filename']}", _picture((40 * n, 90, 60))))
        members.append((f"{TOP}/masks/{it['id']}.png", _mask(PAINT.get(it["id"], 0))))
    members += [
        (f"{TOP}/train.txt", "img001\nimg002\nimg003"),
        (f"{TOP}/val.txt", "img004"),
        (f"{TOP}/metadata.json", json.dumps({
            "project_id": "0" * 32, "project_name": "sample", "num_images": len(items),
            "classes": CLASSES, "ignore_index": 255, "items": items})),
        (f"{TOP}/training/runs/run01/train_config.json", "{}"),
        (f"{TOP}/training/runs/run01/classes.json", json.dumps({
            "version": 1, "classes": [{"id": 0, "name": "background", "color": [0, 0, 0]},
                                      {"id": 1, "name": "renamed-since", "color": [1, 2, 3]}]})),
    ]
    members += [(f"{TOP}/training/runs/run01/reliability_E{e:03d}.png", _picture((e, e, e)))
                for e in (1, 2, 5, 10)]
    return _zip(members)


def _import(client, pid, payload, status=200):
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/import_zip",
                       files=[("file", ("dataset.zip", payload, "application/zip"))])
    assert resp.status_code == status, resp.text
    return resp.json()


def _items(client, pid):
    idx._INDEX_CACHE.pop(pid)
    body = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    return {it["id"]: it for it in (body if isinstance(body, list) else body["items"])}


def _labelled(annotation):
    """What the project list and the image list count as labelled."""
    return bool(annotation.get("hasForeground") or annotation.get("markedClean"))


# -- which members are the dataset --------------------------------------------

def test_the_plan_reads_the_dataset_folders_and_nothing_below_them():
    plan = _plan_zip_import([
        f"{TOP}/images/img001.png", f"{TOP}/images/img002.jpg",
        f"{TOP}/masks/img001.png", f"{TOP}/masks/img002.png",
        f"{TOP}/metadata.json", f"{TOP}/train.txt",
        f"{TOP}/training/runs/run01/reliability_E001.png",
        f"{TOP}/training/runs/run01/classes.json",
        f"{TOP}/prepared/images/img001.jpg",
    ])
    assert [m for m, _ in plan.images] == [f"{TOP}/images/img001.png", f"{TOP}/images/img002.jpg"]
    assert [b for _, b in plan.masks] == ["img001.png", "img002.png"]
    assert plan.metadata == f"{TOP}/metadata.json"
    assert plan.classes is None
    assert plan.classes_elsewhere == f"{TOP}/training/runs/run01/classes.json"
    # The export's own run charts are left behind by design, not passed over.
    assert plan.passed_over == [f"{TOP}/prepared/images/img001.jpg"]


@pytest.mark.parametrize("names, images, masks", [
    # pictures at the root of the archive
    (["img001.png", "img002.jpg"], ["img001.png", "img002.jpg"], []),
    # a zipped folder of pictures, whichever separator the archiver used
    (["photos/img001.png", "photos/img002.png"], ["photos/img001.png", "photos/img002.png"], []),
    (["photos\\img001.png", "photos\\img002.png"], ["photos\\img001.png", "photos\\img002.png"], []),
    # pictures beside a lone masks/ folder: the root is the dataset
    (["img001.png", "img002.png", "masks/img001.png"], ["img001.png", "img002.png"], ["masks/img001.png"]),
    # images/ and masks/ at the root, with classes.json beside them
    (["images/img001.png", "masks/img001.png", "classes.json"], ["images/img001.png"], ["masks/img001.png"]),
    # a folder of pictures holding its masks
    (["photos/img001.png", "photos/masks/img001.png"], ["photos/img001.png"], ["photos/masks/img001.png"]),
    # what macOS and Windows leave behind is not a picture of anything
    (["photos/img001.png", "__MACOSX/photos/._img001.png", "photos/._img002.png", "photos/Thumbs.db"],
     ["photos/img001.png"], []),
])
def test_the_plan_keeps_the_flat_layouts(names, images, masks):
    plan = _plan_zip_import(names)
    assert [m for m, _ in plan.images] == images
    assert [m for m, _ in plan.masks] == masks
    assert plan.passed_over == []


def test_images_beside_an_images_folder_are_not_mixed_in():
    plan = _plan_zip_import(["images/img001.png", "stray.png", "classes.json"])
    assert [m for m, _ in plan.images] == ["images/img001.png"]
    assert plan.passed_over == ["stray.png"]
    assert plan.classes == "classes.json"


def test_the_plan_goes_down_the_folders_an_extracted_export_is_wrapped_in():
    # Windows' Extract All puts the export's folder inside a folder of the
    # same name, and zipping that folder again keeps both.
    plan = _plan_zip_import([
        f"{TOP}/{TOP}/images/img001.png", f"{TOP}/{TOP}/masks/img001.png",
        f"{TOP}/{TOP}/metadata.json", f"{TOP}/{TOP}/train.txt",
        f"{TOP}/{TOP}/training/runs/run01/reliability_E001.png",
        f"{TOP}/{TOP}/training/runs/run01/classes.json",
    ])
    assert [m for m, _ in plan.images] == [f"{TOP}/{TOP}/images/img001.png"]
    assert [m for m, _ in plan.masks] == [f"{TOP}/{TOP}/masks/img001.png"]
    assert plan.metadata == f"{TOP}/{TOP}/metadata.json"
    assert plan.classes_elsewhere == f"{TOP}/{TOP}/training/runs/run01/classes.json"
    assert plan.passed_over == []


@pytest.mark.parametrize("names, images", [
    # three folders deep, each alone in the one above it
    (["a/b/c/images/img001.png"], ["a/b/c/images/img001.png"]),
    # a fourth is too deep
    (["a/b/c/d/images/img001.png"], []),
    # a folder with pictures of its own is where the dataset is
    (["photos/img001.png", "photos/more/img002.png"], ["photos/img001.png"]),
    # a folder holding two folders is not gone through
    (["a/b/images/img001.png", "a/c/img002.png"], []),
    # below the top, never into a folder an export writes beside its dataset
    (["a/training/runs/run01/reliability_E001.png"], []),
    (["training/runs/run01/reliability_E001.png"], []),
    (["a/prepared/images/img001.png"], []),
])
def test_the_plan_goes_down_lone_folders_only_so_far(names, images):
    plan = _plan_zip_import(names)
    assert [m for m, _ in plan.images] == images
    assert sorted(plan.passed_over) == sorted(set(names) - set(images))


# -- through the route ---------------------------------------------------------

def test_an_export_shaped_zip_imports_its_images_masks_and_marks(client, project_id):
    body = _import(client, project_id, _export_zip(MARKED_ITEMS))
    assert body["image_count"] == 5, body  # and not the run's four charts as well
    assert body["mask_count"] == 4, body  # img005's blank was the export's, not a mask
    assert body["clean_count"] == 2, body
    assert body["passed_over"] == 0 and body["passed_over_sample"] == [], body  # the charts stay behind

    items = _items(client, project_id)
    assert sorted(items) == ["img001", "img002", "img003", "img004", "img005"]
    ann = {k: v["annotation"] for k, v in items.items()}
    assert ann["img001"]["classIds"] == [1] and not ann["img001"].get("markedClean")
    assert ann["img002"]["classIds"] == [2]
    for ok in ("img003", "img004"):
        assert ann[ok]["hasMask"] is True and ann[ok]["markedClean"] is True, ann[ok]
    assert not ann["img005"]["hasMask"] and not ann["img005"].get("markedClean")
    assert not (annotate_masks_dir(project_id) / "img005.png").exists()
    assert sorted(k for k, a in ann.items() if _labelled(a)) == ["img001", "img002", "img003", "img004"]

    assert ann["img002"]["draft"] is True and ann["img002"]["draftRun"] == "review"
    assert ann["img002"]["draftReason"] == "edge unclear"
    assert not any(a.get("draft") for k, a in ann.items() if k != "img002")
    assert {k: v["set"] for k, v in items.items()} == {
        "img001": "train", "img002": "none", "img003": "none", "img004": "test", "img005": "none"}
    assert items["img001"]["name"] == "img001.jpg"

    # The project's classes as exported, not the run's older copy.
    classes = client.get(f"/api/v1/projects/{project_id}/classes").json()["classes"]
    assert [c["name"] for c in classes] == ["background", "scratch", "dent"]


def test_a_clean_mark_is_not_restored_onto_paint(client, project_id):
    # A hand-edited metadata.json can say OK about a painted mask; the paint wins.
    items = [dict(it, markedClean=True) if it["id"] == "img001" else it for it in MARKED_ITEMS]
    _import(client, project_id, _export_zip(items))
    ann = _items(client, project_id)["img001"]["annotation"]
    assert ann["classIds"] == [1] and not ann.get("markedClean")


def test_an_older_export_imports_as_it_always_did(client, project_id):
    body = _import(client, project_id, _export_zip(OLD_ITEMS))
    assert body["image_count"] == 5, body  # the run's charts stay out all the same
    assert body["mask_count"] == 5, body  # no hasMask: every blank is taken as saved
    assert body["clean_count"] == 0, body
    items = _items(client, project_id)
    assert sorted(items) == ["img001", "img002", "img003", "img004", "img005"]
    for item_id, item in items.items():
        ann = item["annotation"]
        assert item["set"] == "none" and item["name"] == f"{item_id}.png", item
        assert ann["hasMask"] is True and not ann.get("markedClean") and not ann.get("draft"), ann
    assert items["img001"]["annotation"]["classIds"] == [1]


def test_a_flat_zip_of_pictures_still_imports(client, project_id):
    body = _import(client, project_id, _zip([
        ("img001.png", _picture((10, 20, 30))),
        ("img002.jpg", _picture((40, 50, 60), fmt="JPEG")),
    ]))
    assert body["image_count"] == 2 and body["mask_count"] == 0, body
    assert body["passed_over"] == 0 and body["passed_over_sample"] == [], body
    assert sorted(_items(client, project_id)) == ["img001", "img002"]


def test_a_zipped_folder_of_pictures_with_masks_imports(client, project_id):
    body = _import(client, project_id, _zip([
        ("photos/img001.png", _picture((10, 20, 30))),
        ("photos/img002.png", _picture((40, 50, 60))),
        ("photos/masks/img001.png", _mask(1)),
    ]))
    assert body["image_count"] == 2 and body["mask_count"] == 1, body
    assert _items(client, project_id)["img001"]["annotation"]["classIds"] == [1]


def test_pictures_only_in_deeper_folders_are_refused_with_the_reason(client, project_id):
    body = _import(client, project_id, _zip([
        ("datasets/prepared/images/img001.png", _picture()),
        ("classes.json", json.dumps({"classes": CLASSES})),
    ]), status=400)
    assert body["detail"].startswith("no images found in zip"), body
    assert "datasets/prepared/images/img001.png" in body["detail"], body
    assert _items(client, project_id) == {}


def _rezipped(payload, folder=TOP):
    """The archive after Windows' Extract All and zipping the folder it made:
    everything one folder deeper, inside a folder of the same name."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return _zip([(f"{folder}/{name}", zf.read(name)) for name in zf.namelist()])


def test_an_export_extracted_and_zipped_again_imports(client, project_id):
    body = _import(client, project_id, _rezipped(_export_zip(MARKED_ITEMS)))
    assert body["image_count"] == 5 and body["mask_count"] == 4 and body["clean_count"] == 2, body
    assert body["passed_over"] == 0, body
    ann = {k: v["annotation"] for k, v in _items(client, project_id).items()}
    assert ann["img001"]["classIds"] == [1] and ann["img003"]["markedClean"] is True
    classes = client.get(f"/api/v1/projects/{project_id}/classes").json()["classes"]
    assert [c["name"] for c in classes] == ["background", "scratch", "dent"]


def test_the_response_counts_and_names_the_pictures_passed_over(client, project_id):
    stray = [f"more/img{n:03d}.png" for n in range(10, 17)]
    body = _import(client, project_id, _zip(
        [("images/img001.png", _picture()), ("images/old/img002.png", _picture()),
         ("img003.png", _picture())] + [(name, _picture()) for name in stray]))
    assert body["image_count"] == 1, body
    assert body["passed_over"] == 9, body
    sample = body["passed_over_sample"]
    assert len(sample) == 5 and set(sample) <= {"images/old/img002.png", "img003.png", *stray}, body


# -- a mask kept in chunks -----------------------------------------------------

TALL, WIDE = 600, 300  # more rows than two bands of chunks hold


def _tiled_project(client, project_id, *, older_png, truth=None):
    """One image whose mask is kept in chunks, as a large image's is -- by
    default painted in two classes; with *older_png*, a PNG beside the chunks
    that says something else, as a view of the mask can leave."""
    pytest.importorskip("zarr")
    from app.core import zarr_mask as zm

    buf = io.BytesIO()
    Image.new("RGB", (WIDE, TALL), (90, 90, 90)).save(buf, format="PNG")
    resp = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                       files=[("files", ("img001.png", buf.getvalue(), "image/png"))])
    assert resp.status_code == 200, resp.text
    item_id = resp.json()["items"][0]["id"]
    if truth is None:
        truth = np.zeros((TALL, WIDE), np.uint8)
        truth[100:300, 50:150] = 1
        truth[400:598, 200:300] = 2
    z = zm.open_or_create_zarr_mask(project_id, item_id, TALL, WIDE)
    z[:] = truth
    zm.rebuild_class_tally(project_id, item_id, truth)
    if older_png:
        stale = io.BytesIO()
        Image.fromarray(np.zeros((TALL, WIDE), np.uint8), mode="L").save(stale, format="PNG")
        (annotate_masks_dir(project_id) / f"{item_id}.png").write_bytes(stale.getvalue())
    return item_id, truth


def _export(client, project_id, item_id, **params):
    """The export, the mask it wrote for *item_id* and that image's metadata entry."""
    exported = client.get(f"/api/v1/projects/{project_id}/datasets/export", params=params)
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as zf:
        names = zf.namelist()
        mask = zf.read(next(n for n in names if n.endswith(f"/masks/{item_id}.png")))
        meta = json.loads(zf.read(next(n for n in names if n.endswith("/metadata.json"))))
    with Image.open(io.BytesIO(mask)) as img:
        assert img.mode == "L", img.mode
        got = np.array(img)
    entry = next(it for it in meta["items"] if it["id"] == item_id)
    return exported.content, got, entry


@pytest.mark.parametrize("older_png", [False, True])
def test_a_tiled_mask_is_exported_from_its_chunks(client, project_id, older_png):
    item_id, truth = _tiled_project(client, project_id, older_png=older_png)
    payload, got, entry = _export(client, project_id, item_id)
    # Not the blank placeholder, and not the older PNG beside the chunks.
    assert got.shape == truth.shape and (got == truth).all()
    assert entry["hasMask"] is True

    target = client.post("/api/v1/projects", json={"name": "pytest-tiled"}).json()["id"]
    try:
        body = _import(client, target, payload)
        assert body["image_count"] == 1 and body["mask_count"] == 1, body
        (item,) = _items(client, target).values()
        assert item["annotation"]["classIds"] == [1, 2]
    finally:
        client.delete(f"/api/v1/projects/{target}")


@pytest.mark.parametrize("scale", [0.5, 0.37])
def test_a_tiled_mask_is_resized_as_a_png_mask_is(client, project_id, scale):
    # Every row and every column differs from its neighbours, so a pixel
    # sampled one off from where PIL samples it shows.
    stripes = (np.add.outer(np.arange(TALL), 2 * np.arange(WIDE)) % 7).astype(np.uint8)
    item_id, truth = _tiled_project(client, project_id, older_png=True, truth=stripes)
    _payload, got, _entry = _export(client, project_id, item_id, resize_scale=scale)
    size = (max(1, int(WIDE * scale)), max(1, int(TALL * scale)))
    expected = np.array(Image.fromarray(truth, mode="L").resize(size, Image.NEAREST))
    assert got.shape == expected.shape and (got == expected).all()


# -- the whole way round -------------------------------------------------------

def test_export_then_import_brings_the_project_back(client, project_id):
    ids = []
    for n, colour in enumerate([(200, 10, 10), (10, 200, 10), (10, 10, 200), (200, 200, 10)], start=1):
        resp = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                           files=[("files", (f"img00{n}.png", _picture(colour), "image/png"))])
        assert resp.status_code == 200, resp.text
        ids.append(resp.json()["items"][0]["id"])
    painted, ok_a, ok_b, bare = ids
    route = f"/api/v1/projects/{project_id}/datasets/annotate"
    resp = client.put(f"{route}/masks/{painted}.png", files={"file": ("m.png", _mask(1), "image/png")})
    assert resp.status_code == 200, resp.text
    assert client.post(f"{route}/mark-clean", json={"image_ids": [ok_a, ok_b]}).json()["marked"] == 2
    resp = client.post(f"{route}/review", json={"image_ids": [bare], "reason": "look again", "review": True})
    assert resp.status_code == 200, resp.text
    assert client.patch(f"{route}/{ok_b}", json={"set": "test"}).status_code == 200
    # A trained project has runs, and a run's charts are PNGs.
    run = runs_root_of(project_dir(project_id)) / "run01"
    run.mkdir(parents=True)
    (run / "reliability_E001.png").write_bytes(_picture((1, 2, 3)))
    (run / "metrics.json").write_text("{}", encoding="utf-8")

    exported = client.get(f"/api/v1/projects/{project_id}/datasets/export")
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as zf:
        names = zf.namelist()
        assert any(n.endswith("/training/runs/run01/reliability_E001.png") for n in names), names
        meta = json.loads(zf.read(next(n for n in names if n.endswith("/metadata.json"))))
    entries = {it["id"]: it for it in meta["items"]}
    assert entries[ok_a]["markedClean"] is True and entries[painted]["markedClean"] is False
    assert entries[bare]["hasMask"] is False and entries[bare]["draft"] is True
    assert entries[ok_b]["set"] == "test"

    target = client.post("/api/v1/projects", json={"name": "pytest-roundtrip"}).json()["id"]
    try:
        body = _import(client, target, exported.content)
        assert body["image_count"] == 4, body
        assert body["mask_count"] == 3 and body["clean_count"] == 2, body
        items = _items(client, target)
        assert sorted(items) == sorted(ids)
        ann = {k: v["annotation"] for k, v in items.items()}
        assert ann[painted]["classIds"] == [1]
        assert ann[ok_a]["markedClean"] is True and ann[ok_b]["markedClean"] is True
        assert not ann[bare]["hasMask"] and ann[bare]["draft"] is True
        assert ann[bare]["draftReason"] == "look again"
        assert items[ok_b]["set"] == "test"
        assert sorted(k for k, a in ann.items() if _labelled(a)) == sorted([painted, ok_a, ok_b])
    finally:
        client.delete(f"/api/v1/projects/{target}")
