# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What a project's pixels actually are, measured rather than declared.

Every gate in this application used to ask a setting or test for a directory.
Neither can see files whose names say PNG and whose bytes say JPEG -- an
installation can hold hundreds, most of them 4:2:0 -- and a colour measured over
4:2:0 pixels is the encoder's chroma averaging rather than the part. A
setting-based check cannot even be wrong about them: it is looking somewhere
else entirely.

So this module opens files. It reads headers, not pixels: a census over a
large project covers tens of thousands of files and tens of gigabytes, and
every answer it needs lives in the first few hundred bytes of each.

Everything here is read-only except :func:`measure`, which stores what it
measured so a launch-time gate does not have to re-scan tens of gigabytes to
ask one question.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from segcore.dataset_layout import (
    CANONICAL_IMAGE_EXTS,
    LAYOUT_IMAGES,
    LAYOUT_PREPARED_IMAGES,
    load_layout,
    read_descriptor,
)

from .annotate_index import load_annotate_index
from .image_encode import inspect_file
from .import_settings import (
    effective_image_store,
    read_measured_census,
    save_measured_census,
)
from .paths import IMAGES_LAYOUT_KEY, annotate_images_dir, prepared_dir, project_dir

logger = logging.getLogger(__name__)

#: How many offending names a report carries. The count is the finding; the
#: examples are so somebody can go and look at one.
_EXAMPLES = 10

#: Which suffixes are honest for a given container.
_HONEST_SUFFIXES = {
    "png": (".png",),
    "jpeg": (".jpg", ".jpeg"),
    "webp": (".webp",),
    "bmp": (".bmp",),
    "tiff": (".tif", ".tiff"),
    "gif": (".gif",),
}

_SAMPLING_NAMES = {0: "jpeg444", 1: "jpeg422", 2: "jpeg420", -1: "jpeg_gray"}


def empty_census() -> dict:
    return {
        "total": 0, "png": 0, "jpeg444": 0, "jpeg422": 0, "jpeg420": 0,
        "jpeg_gray": 0, "jpeg_unknown": 0, "webp": 0, "other": 0,
        "mislabelled": 0,
    }


def chroma_census(images_dir: Path) -> dict:
    """What the files in *images_dir* really are, by their bytes.

    ``mislabelled`` counts files whose suffix disagrees with their contents,
    which is precisely the population no declaration can see. ``subsampled``
    in the examples is the one that matters for colour work: 4:2:0 averages
    chroma over 2x2 blocks, so a defect whose only signal is a tint loses
    about two thirds of its contrast before anything looks at it.
    """
    counts = empty_census()
    examples: dict[str, list[str]] = {"subsampled": [], "mislabelled": []}
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return {"counts": counts, "examples": examples, "dir": str(images_dir)}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in CANONICAL_IMAGE_EXTS:
            continue
        counts["total"] += 1
        container, sampling = inspect_file(path)
        if container == "jpeg":
            key = _SAMPLING_NAMES.get(sampling, "jpeg_unknown")
            counts[key] += 1
            if key in ("jpeg422", "jpeg420") and len(examples["subsampled"]) < _EXAMPLES:
                examples["subsampled"].append(path.name)
        elif container in ("png", "webp"):
            counts[container] += 1
        else:
            counts["other"] += 1
        honest = _HONEST_SUFFIXES.get(container)
        if honest and path.suffix.lower() not in honest:
            counts["mislabelled"] += 1
            if len(examples["mislabelled"]) < _EXAMPLES:
                examples["mislabelled"].append(path.name)
    return {"counts": counts, "examples": examples, "dir": str(images_dir)}


def subsampled_count(census: dict | None) -> int:
    """How many files in *census* threw chroma away. None means unmeasured."""
    if not census:
        return 0
    counts = census.get("counts") if "counts" in census else census
    if not isinstance(counts, dict):
        return 0
    return int(counts.get("jpeg420", 0)) + int(counts.get("jpeg422", 0))


def measure(project_id: str) -> dict:
    """Census the project's images and store the result on the project.

    Stored because the alternative is re-scanning a 60 GB directory inside a
    synchronous request every time a gate wants to know. The timestamp is what
    lets a later reader decide the measurement is stale.
    """
    census = chroma_census(annotate_images_dir(project_id))
    census["measured_at"] = datetime.now(timezone.utc).isoformat()
    save_measured_census(project_id, census)
    return census


def census_is_stale(project_id: str, census: dict | None) -> bool:
    """Whether images/ changed after *census* was taken.

    Compares against the newest mtime in the directory rather than its own,
    because on Windows a directory mtime does not always move when a file
    inside it is rewritten in place.
    """
    if not census or not census.get("measured_at"):
        return True
    try:
        taken = datetime.fromisoformat(str(census["measured_at"]))
    except ValueError:
        return True
    images = annotate_images_dir(project_id)
    if not images.is_dir():
        return False
    newest = 0.0
    for path in images.iterdir():
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return newest > taken.timestamp()


def split_resolution(prepared: Path) -> dict:
    """How many split ids resolve to a file, per split.

    This is the number the freshness gate and the descriptor exist for: a
    stem that does not resolve is dropped from the split silently, so a run
    can complete on 40% of its training data and report a score for it.
    """
    from segcore.training.split_utils import load_split_ids

    source = load_layout(prepared)
    out: dict[str, dict] = {}
    for name in ("train", "val", "test"):
        ids = load_split_ids(prepared / "splits" / f"{name}.txt")
        unresolved = source.unresolved(ids) if ids else []
        out[name] = {
            "ids": len(ids),
            "unresolved": len(unresolved),
            "examples": unresolved[:_EXAMPLES],
        }
    return out


def stem_index(images_dir: Path) -> dict[str, str]:
    """stem -> filename for every canonical image in *images_dir*.

    One listing, not a probe per id. The legacy probe globs the directory
    twice per unresolved stem, and a large project holds tens of thousands
    of files; this is also exactly how the descriptor prepare will write is
    built, so it answers for the resolution that will actually be in force.
    """
    out: dict[str, str] = {}
    if not images_dir.is_dir():
        return out
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in CANONICAL_IMAGE_EXTS:
            out.setdefault(path.stem, path.name)
    return out


def _resolution_against(project_id: str, images_dir: Path, target: str) -> dict:
    """Would every split id resolve if this project read *images_dir*?

    Asked of a directory the project is not currently reading, which is what
    makes it different from ``split_resolution``: that one asks it of the
    layout in force, and every unflipped project is reading the copies.
    """
    from segcore.training.split_utils import load_split_ids

    prepared = prepared_dir(project_id)
    stems = stem_index(images_dir)
    splits: dict[str, dict] = {}
    total_unresolved = 0
    total_ids = 0
    for name in ("train", "val", "test"):
        ids = load_split_ids(prepared / "splits" / f"{name}.txt")
        unresolved = [i for i in ids if i.strip() not in stems]
        total_unresolved += len(unresolved)
        total_ids += len(ids)
        splits[name] = {
            "ids": len(ids),
            "unresolved": len(unresolved),
            "examples": unresolved[:_EXAMPLES],
        }
    return {
        "target": target,
        "dir": str(images_dir),
        "splits": splits,
        "ids": total_ids,
        "unresolved": total_unresolved,
        # Undecidable, not clean: a project whose splits are empty has not been
        # prepared since they were invalidated, and nothing has been asked.
        "would_resolve": total_unresolved == 0 and total_ids > 0,
    }


def flip_preflight(project_id: str) -> dict:
    """Would every split id still resolve if this project read images/ direct?

    The gate for a flip. Not ``split_resolution``, which asks it of the layout
    in force -- prepared_images for every project that has not been flipped.
    That directory is never wiped, so it holds copies of items the index has
    since lost and answers yes for ids images/ has nothing for. A project
    can be clean by the first question and not by this one, by as many ids
    as exist only as prepared copies -- hundreds, on a project used for
    long enough.

    Deliberately not a gate for anything else: a project that fails it is not
    broken today.
    """
    return _resolution_against(
        project_id, annotate_images_dir(project_id), LAYOUT_IMAGES)


def rollback_preflight(project_id: str) -> dict:
    """Would the splits still resolve if this project went back to the copies?

    The same question of the other directory, and the reason a flip is
    reversible at all. It stops being true the moment the copies are reclaimed,
    which is exactly when the answer has to be no rather than an empty
    directory silently resolving nothing.
    """
    return _resolution_against(
        project_id, prepared_dir(project_id) / "images", LAYOUT_PREPARED_IMAGES)


def index_report(project_id: str) -> dict:
    """Where the annotation index and the directory disagree.

    ``whitespace_ids`` is the one that already bit us: split files are read
    with every line stripped, so an index id carrying surrounding whitespace
    is unreachable from a split, and a project can hold hundreds of them.
    """
    index = load_annotate_index(project_id, sync=False)
    items = index.get("items", [])
    images = annotate_images_dir(project_id)
    missing: list[str] = []
    stem_mismatch: list[str] = []
    whitespace = 0
    named: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "")
        filename = str(item.get("filename") or "")
        if item_id != item_id.strip():
            whitespace += 1
        if not filename:
            missing.append(item_id)
            continue
        named.add(filename)
        if not (images / filename).exists():
            missing.append(item_id or filename)
        elif Path(filename).stem != item_id.strip():
            stem_mismatch.append(item_id)
    return {
        "items": len(items),
        "missing_file": len(missing),
        "missing_examples": missing[:_EXAMPLES],
        "stem_mismatch": len(stem_mismatch),
        "stem_mismatch_examples": stem_mismatch[:_EXAMPLES],
        "whitespace_ids": whitespace,
        "_named": named,
    }


def orphan_report(project_id: str, named: set[str]) -> dict:
    """Files in images/ that no index item claims.

    A file nothing names is not automatically rubbish -- it may be an import
    the index lost -- but the ones found so far were each the residue of a
    deleted item, so the report says how many and stops
    there. Deciding what to do with them is not a read-only operation.
    """
    images = annotate_images_dir(project_id)
    orphans: list[str] = []
    if images.is_dir():
        for path in sorted(images.iterdir()):
            if not path.is_file() or path.suffix.lower() not in CANONICAL_IMAGE_EXTS:
                continue
            if path.name not in named:
                orphans.append(path.name)
    return {"count": len(orphans), "examples": orphans[:_EXAMPLES]}


def doctor(project_id: str) -> dict:
    """The whole picture for one project, without changing anything.

    Deliberately not cached and deliberately not persisted: this is the call
    that tells you whether what is stored is still true.
    """
    base = project_dir(project_id)
    prepared = prepared_dir(project_id)
    store = effective_image_store(project_id)
    stored_census = read_measured_census(project_id)
    descriptor = read_descriptor(prepared)
    source = load_layout(prepared)

    project_json = base / "project.json"
    layout = None
    if project_json.exists():
        import json
        try:
            layout = json.loads(project_json.read_text(encoding="utf-8")).get(
                IMAGES_LAYOUT_KEY)
        except (OSError, ValueError):
            layout = None

    index = index_report(project_id)
    named = index.pop("_named")
    live_census = chroma_census(annotate_images_dir(project_id))

    return {
        "project_id": project_id,
        "images_layout": layout,
        "image_store": {k: v for k, v in store.items() if k != "measured"},
        "images": {
            "dir": str(annotate_images_dir(project_id)),
            "census": live_census,
            "stored_census_is_stale": census_is_stale(project_id, stored_census),
            "stored_census_missing": stored_census is None,
        },
        "prepared": {
            "exists": prepared.exists(),
            "descriptor": descriptor is not None,
            "descriptor_layout": (descriptor or {}).get("images_layout"),
            "stats_schema": source.stats_schema,
            "resolves_from": str(source.images_dir),
            "legacy_probe": source.is_legacy,
            "prepared_images": chroma_census(prepared / "images"),
        },
        "splits": split_resolution(prepared),
        "flip": flip_preflight(project_id),
        "index": index,
        "orphans": orphan_report(project_id, named),
    }
