# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The doctor reports what the bytes are, not what the names or settings say.

The population it exists for is files that are named .png and hold JPEG, most
of them 4:2:0 -- an installation can hold hundreds. Every assertion here is about
seeing that population -- a check that trusted a suffix or a setting would
pass on the same fixtures while missing all of it.
"""
from __future__ import annotations

import io
import json

from PIL import Image

from app.core.import_settings import (
    read_measured_census,
    save_image_store,
    save_measured_census,
)
from app.core.layout_doctor import chroma_census, doctor, measure, subsampled_count
from app.core.paths import annotate_images_dir, prepared_dir


def _bytes(fmt="PNG", **kw):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 90, 60)).save(buf, format=fmt, **kw)
    return buf.getvalue()


def _put(project_id, name, data):
    images = annotate_images_dir(project_id)
    images.mkdir(parents=True, exist_ok=True)
    (images / name).write_bytes(data)


def test_the_census_reads_the_bytes_not_the_name(client, project_id):
    _put(project_id, "honest.png", _bytes("PNG"))
    _put(project_id, "liar.png", _bytes("JPEG", quality=95, subsampling=2))
    census = chroma_census(annotate_images_dir(project_id))
    counts = census["counts"]
    assert (counts["total"], counts["png"], counts["jpeg420"]) == (2, 1, 1)
    assert counts["mislabelled"] == 1
    assert census["examples"]["mislabelled"] == ["liar.png"]
    assert census["examples"]["subsampled"] == ["liar.png"]


def test_444_is_not_counted_as_subsampled(client, project_id):
    _put(project_id, "a.jpg", _bytes("JPEG", quality=95, subsampling=0))
    counts = chroma_census(annotate_images_dir(project_id))["counts"]
    assert (counts["jpeg444"], counts["jpeg420"], counts["jpeg422"]) == (1, 0, 0)
    assert counts["mislabelled"] == 0


def test_greyscale_jpeg_is_its_own_answer(client, project_id):
    # PIL reports -1 for a single-component JPEG. Folding that into "not 0"
    # would report every greyscale file as 4:2:0.
    buf = io.BytesIO()
    Image.new("L", (16, 16), 100).save(buf, format="JPEG", quality=95)
    _put(project_id, "gray.jpg", buf.getvalue())
    counts = chroma_census(annotate_images_dir(project_id))["counts"]
    assert counts["jpeg_gray"] == 1
    assert subsampled_count({"counts": counts}) == 0


def test_a_png_is_never_read_as_subsampled(client, project_id):
    _put(project_id, "a.png", _bytes("PNG"))
    assert subsampled_count(chroma_census(annotate_images_dir(project_id))) == 0


def test_measure_stores_what_it_found(client, project_id):
    _put(project_id, "liar.png", _bytes("JPEG", quality=95, subsampling=2))
    measure(project_id)
    stored = read_measured_census(project_id)
    assert stored is not None
    assert subsampled_count(stored) == 1
    assert stored["measured_at"]


def test_the_census_survives_a_format_change(client, project_id):
    # The block used to be rewritten wholesale on every save, so a format
    # change dropped the census and the gate then read the project as never
    # measured -- which is the one answer that must not be inferred.
    save_measured_census(project_id, {"counts": {"jpeg420": 3}, "measured_at": "x"})
    save_image_store(project_id, "jpg")
    survived = read_measured_census(project_id)
    assert survived is not None and subsampled_count(survived) == 3


def test_the_census_survives_a_conversion(client, project_id):
    _put(project_id, "liar.png", _bytes("JPEG", quality=95, subsampling=2))
    measure(project_id)
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/convert-images",
        params={"format": "png"})
    assert resp.status_code == 200, resp.text
    assert read_measured_census(project_id) is not None


def test_the_doctor_counts_split_ids_that_do_not_resolve(client, project_id):
    _put(project_id, "a.png", _bytes("PNG"))
    prepared = prepared_dir(project_id)
    (prepared / "splits").mkdir(parents=True, exist_ok=True)
    (prepared / "images").mkdir(parents=True, exist_ok=True)
    (annotate_images_dir(project_id) / "a.png")  # keep the annotate copy
    (prepared / "images" / "a.png").write_bytes(_bytes("PNG"))
    (prepared / "splits" / "train.txt").write_text("a\nghost\n", encoding="utf-8")
    report = doctor(project_id)
    train = report["splits"]["train"]
    assert train["ids"] == 2
    assert train["unresolved"] == 1
    assert train["examples"] == ["ghost"]


def test_the_doctor_sees_an_id_that_a_split_can_never_match(client, project_id):
    # load_split_ids strips every line it reads, so an index id with
    # surrounding whitespace is unreachable from any split, and a project can
    # carry hundreds of them.
    _put(project_id, "a.png", _bytes("PNG"))
    index_path = annotate_images_dir(project_id).parent / "index.json"
    index_path.write_text(json.dumps({"items": [
        {"id": " a", "filename": "a.png", "set": "none"},
    ]}), encoding="utf-8")
    assert doctor(project_id)["index"]["whitespace_ids"] == 1


def test_a_file_no_item_names_is_reported_as_an_orphan(client, project_id):
    _put(project_id, "kept.png", _bytes("PNG"))
    _put(project_id, "leftover.png", _bytes("PNG"))
    index_path = annotate_images_dir(project_id).parent / "index.json"
    index_path.write_text(json.dumps({"items": [
        {"id": "kept", "filename": "kept.png", "set": "none"},
    ]}), encoding="utf-8")
    orphans = doctor(project_id)["orphans"]
    assert orphans["count"] == 1
    assert orphans["examples"] == ["leftover.png"]


def test_the_report_route_changes_nothing_and_measure_does(client, project_id):
    _put(project_id, "liar.png", _bytes("JPEG", quality=95, subsampling=2))
    resp = client.get(f"/api/v1/projects/{project_id}/layout/doctor")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["images"]["census"]["counts"]["jpeg420"] == 1
    assert body["images"]["stored_census_missing"] is True
    assert read_measured_census(project_id) is None

    resp = client.post(f"/api/v1/projects/{project_id}/layout/measure")
    assert resp.status_code == 200, resp.text
    assert subsampled_count(read_measured_census(project_id)) == 1


def test_an_unknown_project_is_a_404(client):
    assert client.get("/api/v1/projects/nosuchproject/layout/doctor").status_code == 404
