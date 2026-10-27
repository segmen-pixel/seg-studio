# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A prepared dataset describes itself, and the freshness gate believes it.

Two halves. The round-trip tests prove the writer in dataset_prep and the
reader in segcore.dataset_layout agree about every stem -- they are in
different packages and neither imports the other, which is the arrangement that
produced eight disagreeing extension lists in the first place.

The gate tests prove the descriptor is actually consulted. A gate that only
looks at split mtimes reports a stale cache as fresh, and the run then trains
on whatever the old copies happen to be.
"""
from __future__ import annotations

import io
import json

from PIL import Image

from app.core.paths import IMAGES_LAYOUT_KEY, prepared_dir, project_dir
from app.core.training_job_phases import prepare_run_dataset
from segcore.dataset_layout import (
    DESCRIPTOR_NAME,
    DESCRIPTOR_SCHEMA,
    LAYOUT_IMAGES,
    LAYOUT_PREPARED_IMAGES,
    STATS_SCHEMA,
    load_layout,
    read_descriptor,
)

SKIP_LINE = "skipping re-prepare"


def _upload(client, pid, n=8):
    files = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(10 * i % 250, 40, 60)).save(buf, format="PNG")
        files.append(("files", (f"img{i:02d}.png", buf.getvalue(), "image/png")))
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/upload", files=files)
    assert resp.status_code == 200, resp.text
    listing = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    ids = [it["id"] for it in listing["items"]]
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/mark-clean", json={"image_ids": ids})
    assert resp.status_code == 200, resp.text
    return ids


def _prepare(client, pid):
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/prepare")
    assert resp.status_code == 200, resp.text
    return prepared_dir(pid)


def _split_ids(prep):
    out = []
    for name in ("train", "val", "test"):
        path = prep / "splits" / f"{name}.txt"
        if path.exists():
            out += [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def _run_gate(pid):
    """Call phase 1 and return the log it produced."""
    lines: list[str] = []
    prepare_run_dataset(pid, "runid000000", {}, prepared_dir(pid), lines.append)
    return "".join(lines)


# ---------------------------------------------------------------------------
# The descriptor a prepare writes
# ---------------------------------------------------------------------------

def test_prepare_writes_a_descriptor(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    raw = json.loads((prep / DESCRIPTOR_NAME).read_text(encoding="utf-8"))
    assert raw["schema"] == DESCRIPTOR_SCHEMA
    assert raw["images_layout"] == LAYOUT_PREPARED_IMAGES
    assert raw["stats_schema"] == STATS_SCHEMA
    assert raw["source"] == "annotate"


def test_the_image_directory_is_recorded_relative(client, project_id):
    # The documented backup procedure copies a project folder back into
    # projects/, so an absolute path would name the machine that wrote it.
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    raw = json.loads((prep / DESCRIPTOR_NAME).read_text(encoding="utf-8"))
    assert raw["images_dir"] == "images"
    assert not raw["images_dir"].startswith(("/", "C:", "\\"))


def test_every_split_stem_resolves_through_the_reader(client, project_id):
    # The round trip that matters: dataset_prep writes it, segcore reads it,
    # and neither imports the other.
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    ids = _split_ids(prep)
    assert ids, "the fixture produced no splits"
    source = load_layout(prep)
    assert not source.is_legacy
    assert source.unresolved(ids) == []


def test_the_items_describe_the_files_that_are_actually_there(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    raw = json.loads((prep / DESCRIPTOR_NAME).read_text(encoding="utf-8"))
    for stem, entry in raw["items"].items():
        on_disk = prep / "images" / entry["file"]
        assert on_disk.exists(), f"{stem} -> {entry['file']}"
        assert entry["size"] == on_disk.stat().st_size


# ---------------------------------------------------------------------------
# The freshness gate
# ---------------------------------------------------------------------------

def test_a_freshly_prepared_cache_is_reused(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    assert SKIP_LINE in _run_gate(project_id)


def test_a_cache_without_a_descriptor_is_rebuilt(client, project_id):
    # Every project that predates this change is in exactly this state, and
    # each has to re-prepare once.
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    (prep / DESCRIPTOR_NAME).unlink()
    assert SKIP_LINE not in _run_gate(project_id)
    assert read_descriptor(prep) is not None, "the rebuild must write one back"


def test_a_descriptor_from_an_older_statistics_rule_is_rebuilt(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    raw = json.loads((prep / DESCRIPTOR_NAME).read_text(encoding="utf-8"))
    raw["stats_schema"] = STATS_SCHEMA - 1
    (prep / DESCRIPTOR_NAME).write_text(json.dumps(raw), encoding="utf-8")
    assert SKIP_LINE not in _run_gate(project_id)


def test_a_descriptor_disagreeing_with_the_project_is_rebuilt(client, project_id):
    # project.json owns the switch; the descriptor is a generated copy of it.
    # One written before a flip must not be treated as current.
    _upload(client, project_id)
    _prepare(client, project_id)
    pj = project_dir(project_id) / "project.json"
    data = json.loads(pj.read_text(encoding="utf-8"))
    data[IMAGES_LAYOUT_KEY] = LAYOUT_IMAGES
    pj.write_text(json.dumps(data), encoding="utf-8")
    assert SKIP_LINE not in _run_gate(project_id)


def test_a_cache_whose_images_went_missing_is_rebuilt(client, project_id):
    # The sample is bounded and deterministic, so emptying the directory is
    # the reliable way to make it notice.
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    for f in (prep / "images").iterdir():
        f.unlink()
    log = _run_gate(project_id)
    assert SKIP_LINE not in log
