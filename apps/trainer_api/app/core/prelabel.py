# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Adopt a finished run's predictions as draft annotations.

A prediction and an annotation mask share one on-disk encoding -- a single
channel PNG whose pixel value is the class id -- so adopting a prediction is
a copy.  Everything here is therefore about *which* items may be written,
not about how a mask is produced.

Existing annotations are never silently replaced: an item that already
carries a mask is skipped unless the caller opts in, and even then the
previous mask is copied aside before it is overwritten.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .annotate_index import (
    _scan_mask_info,
    load_annotate_index,
    refresh_annotation,
    save_annotate_index,
)
from .mask_write import save_mask_array
from .paths import annotate_masks_dir, get_project_lock, predictions_dir, project_dir, resolve_run_path
from .prediction_engine import ensure_prediction_artifacts, resolve_predict_context

logger = logging.getLogger("trainer_api")

# Backups live outside masks/ so that the scans which walk that directory
# (index sync, class presence) never see them.
BACKUP_DIRNAME = "masks_replaced"


def backup_dir(project_id: str) -> Path:
    return project_dir(project_id) / BACKUP_DIRNAME


def has_annotation(project_id: str, item_id: str) -> bool:
    """True when the item already carries an annotation of any kind.

    An all-background mask counts: it means "annotated as negative", which is
    a deliberate judgement a draft must not discard.
    """
    has_mask, _has_foreground, _class_ids = _scan_mask_info(
        annotate_masks_dir(project_id) / f"{item_id}.png")
    return has_mask


def candidate_item_ids(project_id: str, *, include_annotated: bool = False) -> list[str]:
    """Items a draft may be written to, in index order."""
    index = load_annotate_index(project_id)
    ids = [str(item["id"]) for item in index.get("items", []) if item.get("id")]
    if include_annotated:
        return ids
    return [item_id for item_id in ids if not has_annotation(project_id, item_id)]


def describe_error(exc: Exception) -> str:
    """A message worth showing a user, preferring FastAPI's own detail."""
    detail = getattr(exc, "detail", None)
    return str(detail) if detail else f"{type(exc).__name__}: {exc}"


def resolve_run(project_id: str, run_id: str, backend: str = "onnx"):
    """(run_path, model_path, backend) for a run that can predict.

    Whether a run has a usable model is a property of the run, not of any one
    image, so callers resolve it once instead of failing image by image.
    """
    return resolve_predict_context(project_id, run_id, backend)


def prediction_on_disk(project_id: str, run_id: str, item_id: str, *,
                       backend: str = "onnx", tta: bool = False) -> Path | None:
    """The prediction PNG a batch run already left for this item, or None.

    Adopting a prediction that exists needs no model. A run trained here
    keeps its torch checkpoint and may never have been exported to ONNX, yet
    predict_batch has already written every mask the draft needs; refusing
    to read them because ``model.onnx`` is missing sent the caller to export
    a model it was not going to run.
    """
    run_path = resolve_run_path(project_id, run_id)
    if run_path is None:
        return None
    pred = predictions_dir(run_path, backend=backend, tta=tta) / f"{item_id}.png"
    return pred if pred.exists() else None


def predicted_mask(project_id: str, run_id: str, item_id: str, *,
                   backend: str = "onnx", tta: bool = False,
                   context: tuple | None = None) -> np.ndarray:
    """The class-id array a finished run predicts for one item: what the
    batch run left on disk if it is there, else a fresh prediction."""
    pred_path = prediction_on_disk(project_id, run_id, item_id, backend=backend, tta=tta)
    if pred_path is None:
        if context is None:
            context = resolve_run(project_id, run_id, backend)
        run_path, model_path, backend_resolved = context
        pred_path, _confidence, _score = ensure_prediction_artifacts(
            project_id, run_path, model_path, item_id, backend_resolved, tta=tta)
    with Image.open(pred_path) as img:
        arr = np.array(img)
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def safe_item_id(item_id: str) -> bool:
    """An item id names a file inside masks/; it must not be able to name
    anything else. Ids keep the original stem, so anything but separators,
    NUL and the dot entries is allowed."""
    return bool(item_id) and item_id not in (".", "..") and not any(c in item_id for c in "/\\\x00")


def adopt(project_id: str, run_id: str, item_id: str, *,
          backend: str = "onnx", tta: bool = False,
          overwrite: bool = False, context: tuple | None = None) -> str:
    """Write one prediction into the annotation masks.

    Returns "written", "skipped" (already annotated and not overwriting) or
    "empty" (the run found no foreground -- adopting that as a confirmed
    negative would assert more than the model actually showed).
    """
    if not safe_item_id(item_id):
        raise ValueError(f"invalid item id: {item_id!r}")
    masks = annotate_masks_dir(project_id)
    dest = masks / f"{item_id}.png"
    if (masks / f"{item_id}.zarr").is_dir():
        # A tiled mask. A draft PNG written beside one is read by nothing --
        # exporting the mask rewrites it from the array -- so it would leave
        # the entry describing a file with no bearing on the annotation.
        return "skipped"
    if has_annotation(project_id, item_id) and not overwrite:
        return "skipped"
    arr = predicted_mask(project_id, run_id, item_id, backend=backend, tta=tta,
                         context=context)
    if not bool((arr > 0).any()):
        return "empty"
    # The write and the entry describing it go under one lock, and whether a
    # mask is already there is asked again inside it: a prediction takes
    # seconds, and a person can save in that time.
    with get_project_lock(project_id):
        if (masks / f"{item_id}.zarr").is_dir():
            return "skipped"
        if has_annotation(project_id, item_id) and not overwrite:
            return "skipped"
        if dest.exists():
            target = backup_dir(project_id)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, target / f"{item_id}.png")
        # Replaced whole, never rewritten in place: a write cut short leaves the
        # old mask, not a truncated file the index would read as no mask.
        class_ids, stamp = save_mask_array(project_id, item_id, arr)
        _mark_draft(project_id, item_id, run_id, class_ids, stamp)
    return "written"


def _mark_draft(project_id: str, item_id: str, run_id: str, class_ids: list[int],
                stamp: Path) -> None:
    """Record in the index that this mask is a draft from a run.

    A draft written as a bare PNG looked exactly like a hand mask to every
    reader of the index -- same hasMask, no revision, no author -- so nothing
    downstream could tell a person which images still needed their eyes. The
    entry now says what the mask route would say about a save, plus who made
    it. Reviewing the draft in the browser and saving bumps the revision and
    clears the flag through the normal route.

    Called with the project lock held, by the write it describes, so no save
    can land between the mask and its entry.
    """
    index = load_annotate_index(project_id)
    for item in index.get("items", []):
        if item.get("id") != item_id:
            continue
        annotation = dict(item.get("annotation") or {})
        # Stamped against the file just written, so an entry inherited from
        # whatever was there before cannot outlive the mask it described.
        refresh_annotation(annotation, stamp, ids=class_ids)
        annotation.update({
            "markedClean": False,
            "revision": int(annotation.get("revision", 0)) + 1,
            "lastSavedAt": datetime.now(timezone.utc).isoformat(),
            "draft": True,
            "draftRun": run_id,
        })
        item["annotation"] = annotation
        break
    save_annotate_index(project_id, index)


def adopt_stream(project_id: str, run_id: str, item_ids: list[str], *,
                 backend: str = "onnx", tta: bool = False,
                 overwrite: bool = False) -> Iterator[str]:
    """NDJSON progress: one line per item, then a summary line.

    A run that cannot predict at all ends the stream with a single error line
    rather than repeating the same failure once per image.
    """
    totals = {"written": 0, "skipped": 0, "empty": 0, "failed": 0}
    total = len(item_ids)
    context = None
    try:
        context = resolve_run(project_id, run_id, backend)
    except Exception as exc:
        # No usable checkpoint for this backend. That only matters for the
        # images predict_batch has not already covered; the rest adopt from
        # disk. Fail the stream as a whole only when nothing is on disk.
        if not any(prediction_on_disk(project_id, run_id, i, backend=backend, tta=tta)
                   for i in item_ids):
            logger.warning("prelabel: run %s cannot predict: %s", run_id, exc)
            yield json.dumps({"error": describe_error(exc),
                              "done": 0, "total": total}) + "\n"
            return
        logger.info("prelabel: run %s has no %s checkpoint; adopting the predictions "
                    "on disk (%s)", run_id, backend, exc)
    for done, item_id in enumerate(item_ids, 1):
        detail = None
        try:
            outcome = adopt(project_id, run_id, item_id, backend=backend,
                            tta=tta, overwrite=overwrite, context=context)
        except Exception as exc:  # one unreadable image must not end the batch
            outcome = "failed"
            detail = describe_error(exc)
            logger.warning("prelabel failed for %s: %s", item_id, exc)
        totals[outcome] += 1
        line = {"item_id": item_id, "outcome": outcome, "done": done, "total": total}
        if detail:
            line["detail"] = detail
        yield json.dumps(line) + "\n"
    yield json.dumps({"summary": totals, "done": total, "total": total}) + "\n"
