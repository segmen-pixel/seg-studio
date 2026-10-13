# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image

from segcore.image_io import imread as _imread

from ..core.agent_activity import agent_of
from ..core.agent_activity import note as note_agent
from ..core.annotate_index import (
    build_annotate_annotations,
    find_annotate_image,  # noqa: F401
    is_persons_work,
    load_annotate_index,
    refresh_annotation,
    save_annotate_index,
    sync_annotate_index,  # noqa: F401
)
from ..core.db_utils import touch_project
from ..core.image_encode import EncodeError, encode_for_store
from ..core.import_settings import effective_image_format
from ..core.mask_write import (
    is_tiled,
    load_mask_array,
    refuse_agent_overwrite,
    remove_mask,
    save_mask_array,
)
from ..core.paths import (
    annotate_annotations_path,
    annotate_images_dir,
    annotate_masks_dir,
    get_project_lock,
    project_dir,
    thumbnails_dir,
    write_bytes_atomic,
    write_json,
)
from ..core.security import safe_child, sanitize_filename

router = APIRouter()


@router.get("/projects/{project_id}/datasets/annotate")
def list_annotate_items(
    project_id: str,
    sync: bool = Query(default=True),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=0, ge=0),
):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    index = load_annotate_index(project_id, sync=sync)
    if limit > 0:
        items = index.get("items", [])
        total = len(items)
        index = {**index, "items": items[offset:offset + limit], "total": total, "offset": offset, "limit": limit}
    return index


@router.post("/projects/{project_id}/datasets/annotate/annotations/export")
def export_annotate_annotations(project_id: str):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    payload = build_annotate_annotations(project_id)
    out_path = annotate_annotations_path(project_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    return {
        "status": "ok",
        "path": str(out_path),
        "item_count": len(payload.get("items", [])),
        "annotation_count": payload.get("total_annotations", 0),
    }


@router.get("/projects/{project_id}/datasets/annotate/annotations.json")
def get_annotate_annotations(project_id: str):
    path = annotate_annotations_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="annotations.json not found")
    return FileResponse(path)


@router.patch("/projects/{project_id}/datasets/annotate/{item_id}")
def update_annotate_item(project_id: str, item_id: str, payload: dict):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        found = None
        for item in items:
            if item.get("id") == item_id:
                found = item
                break
        if found is None:
            raise HTTPException(status_code=404, detail="item not found")
        if "set" in payload:
            if payload["set"] not in ["none", "train", "test"]:
                raise HTTPException(status_code=400, detail="invalid set")
            found["set"] = payload["set"]
        if "name" in payload:
            found["name"] = payload["name"]
        index["items"] = items
        save_annotate_index(project_id, index)
    touch_project(project_id)
    return found


@router.post("/projects/{project_id}/datasets/annotate/batch_set")
def batch_update_annotate_set(project_id: str, payload: dict):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    items_payload = payload.get("items")
    if not isinstance(items_payload, list):
        raise HTTPException(status_code=400, detail="items list required")
    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        items_by_id = {item.get("id"): item for item in items}
        updated = 0
        for entry in items_payload:
            if not isinstance(entry, dict):
                continue
            item_id = entry.get("id")
            set_value = entry.get("set")
            if item_id not in items_by_id:
                continue
            if set_value not in ["none", "train", "test"]:
                continue
            items_by_id[item_id]["set"] = set_value
            updated += 1
        index["items"] = list(items_by_id.values())
        save_annotate_index(project_id, index)
    touch_project(project_id)
    return {"updated": updated}


@router.delete("/projects/{project_id}/datasets/annotate/{item_id}")
def delete_annotate_item(project_id: str, item_id: str):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        remaining = []
        target = None
        for item in items:
            if item.get("id") == item_id:
                target = item
            else:
                remaining.append(item)
        if target is None:
            raise HTTPException(status_code=404, detail="item not found")
        # Delete image and mask files
        if filename := target.get("filename"):
            image_path = annotate_images_dir(project_id) / filename
            image_path.unlink(missing_ok=True)
        remove_mask(project_id, item_id)
        # Update index
        index["items"] = remaining
        save_annotate_index(project_id, index)
    touch_project(project_id)
    return {"status": "ok", "remaining": len(remaining)}


@router.post("/projects/{project_id}/datasets/annotate/bulk-delete")
async def bulk_delete_annotate_items(project_id: str, request: Request):
    """Delete many items in one call.

    The single-item endpoint grabs a per-project lock, parses the entire
    index.json, writes it back, and touches the project for every ID —
    which is O(N²) on ``len(items)`` for bulk work. With 30k+ items the
    UI was sitting at a dead crawl.

    This endpoint takes the lock and parses ``index.json`` **once**,
    drops all requested items in a single filter pass, deletes the
    image / mask files in a thread pool, and rewrites the index once.
    """
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    body = await request.json()
    image_ids = body.get("image_ids") or []
    if not isinstance(image_ids, list):
        raise HTTPException(status_code=400, detail="image_ids must be a list")
    if not image_ids:
        return {"deleted": 0, "not_found": 0, "remaining": 0}

    id_set = {str(x) for x in image_ids}

    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        keep: list[dict] = []
        drop: list[dict] = []
        for item in items:
            if item.get("id") in id_set:
                drop.append(item)
            else:
                keep.append(item)

        images_dir = annotate_images_dir(project_id)

        def _remove_one(it: dict) -> None:
            if filename := it.get("filename"):
                try:
                    (images_dir / filename).unlink(missing_ok=True)
                except OSError:
                    pass
            # Every form the mask takes, tally included: whatever is left
            # behind is inherited by the next image given this id.
            try:
                remove_mask(project_id, it["id"])
            except OSError:
                pass

        if drop:
            # File I/O is the dominant cost — parallelize it.
            workers = min(16, max(2, (os.cpu_count() or 4)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_remove_one, drop))

        index["items"] = keep
        save_annotate_index(project_id, index)

    touch_project(project_id)
    return {
        "deleted": len(drop),
        "not_found": len(id_set) - len(drop),
        "remaining": len(keep),
    }


def _http_date(epoch_seconds: float) -> str:
    """RFC 7231 date for a Last-Modified header.

    utcfromtimestamp() is deprecated from 3.12 and returns a naive datetime;
    this renders byte-identically because the format string carries the zone
    itself. Kept as a function so the shape has one definition and a test.
    """
    stamp = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return stamp.strftime("%a, %d %b %Y %H:%M:%S GMT")


@router.get("/projects/{project_id}/datasets/annotate/images/{filename}")
def get_annotate_image(project_id: str, filename: str):
    path = safe_child(annotate_images_dir(project_id), sanitize_filename(filename))
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    mtime = path.stat().st_mtime
    last_modified = _http_date(mtime)
    return FileResponse(
        path,
        headers={
            "Cache-Control": "public, max-age=3600",
            "Last-Modified": last_modified,
        },
    )


_THUMB_SIZE = (256, 256)


@router.get("/projects/{project_id}/datasets/annotate/images/{filename}/thumbnail")
def get_annotate_image_thumbnail(project_id: str, filename: str):
    safe_name = sanitize_filename(filename)
    src = safe_child(annotate_images_dir(project_id), safe_name)
    if not src.exists():
        raise HTTPException(status_code=404, detail="image not found")
    thumb_dir = thumbnails_dir(project_id)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / (safe_name + ".thumb.jpg")
    src_mtime = src.stat().st_mtime
    if not thumb_path.exists() or thumb_path.stat().st_mtime < src_mtime:
        img = Image.open(src)
        img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
        img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=75, optimize=True)
    return Response(
        content=thumb_path.read_bytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/projects/{project_id}/datasets/annotate/masks/{item_id}.png")
def get_annotate_mask(project_id: str, item_id: str, request: Request):
    safe_id = sanitize_filename(item_id)
    png_path = safe_child(annotate_masks_dir(project_id), f"{safe_id}.png")

    # If Zarr mask exists and PNG is stale/missing, export from Zarr
    try:
        from ..core.zarr_mask import has_zarr_mask, zarr_mask_path, zarr_to_png
        if has_zarr_mask(project_id, safe_id):
            import zarr as _zarr
            zpath = zarr_mask_path(project_id, safe_id)
            zarr_arr = _zarr.open_array(str(zpath), mode="r")
            zarr_to_png(zarr_arr, png_path)
    except Exception:
        pass  # Zarr unavailable — fall through to PNG

    if not png_path.exists():
        raise HTTPException(status_code=404, detail="mask not found")
    mtime = png_path.stat().st_mtime
    etag = hashlib.md5(f"{png_path}:{mtime}".encode()).hexdigest()
    # A mask changes under the browser -- a person paints, the bridge labels --
    # so it may not be cached blind; but re-sending it on every switch between
    # images made clicking through a project heavy for nothing. Nothing here
    # answers a conditional request (starlette does not), so this route does:
    # unchanged is a 304 with no body, changed is the new mask.
    if (request.headers.get("if-none-match") or "").strip('"') == etag:
        return Response(status_code=304, headers={"Cache-Control": "no-cache",
                                                  "ETag": f'"{etag}"'})
    return FileResponse(
        png_path,
        headers={
            # A mask changes under the browser -- a person paints, the bridge
            # labels -- and a minute of max-age meant the reload that went
            # looking for a fresh one could be answered with the mask from
            # before the write. If-None-Match is not honoured here, so a
            # revalidation costs the whole PNG; a mask is tens of kilobytes
            # and being right is worth more than that.
            "Cache-Control": "no-cache",
            "ETag": f'"{etag}"',
        },
    )


@router.put("/projects/{project_id}/datasets/annotate/masks/{item_id}.png")
async def put_annotate_mask(project_id: str, item_id: str, request: Request, raw: int = Query(0),
                            w: int = Query(0), h: int = Query(0), overwrite: int = Query(0)):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    if raw and w > 0 and h > 0:
        # Raw Uint8Array from frontend — encode to PNG server-side
        raw_bytes = await request.body()
        mask_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w))
        unique = np.unique(mask_arr)
        class_ids = sorted(int(v) for v in unique if v != 0 and v != 255)
        has_foreground = len(class_ids) > 0
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(mask_arr, mode="L").save(buf, "PNG")
        mask_bytes = buf.getvalue()
    else:
        # Legacy: PNG file upload (multipart form)
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(status_code=400, detail="file or raw data required")
        mask_bytes = await file.read()  # type: ignore[union-attr]
        import io

        from PIL import Image as PILImage
        try:
            mask_arr = np.array(PILImage.open(io.BytesIO(mask_bytes)))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid mask PNG: {e}")
        if mask_arr.ndim >= 3:
            mask_arr = mask_arr[:, :, 0]
        if mask_arr.ndim != 2 or mask_arr.size == 0:
            raise HTTPException(status_code=400, detail="invalid mask PNG: expected a non-empty 2D image")
        unique = np.unique(mask_arr)
        class_ids = sorted(int(v) for v in unique if v != 0 and v != 255)
        has_foreground = len(class_ids) > 0

    # Atomic mask write + index update under project lock
    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        entry = next((item for item in items if item.get("id") == item_id), None)
        if entry is None:
            # Writing first and looking afterwards left a mask file that no
            # entry described, which the next image given this id inherited.
            raise HTTPException(status_code=404, detail="item not found")
        # An agent may not paint over what a person drew. A labelling run that
        # was asked for a few unlabelled images worked on past them and wrote
        # over teachers it had been learning from, one of them with a mask far
        # larger than the object. Nothing refused it, and the only copy of that
        # person's work was an old prepared dataset.
        was = entry.get("annotation") or {}
        # What is protected is a person's work, not the existence of a file. A
        # mask with nothing painted in it is not an annotation -- unless they
        # marked the image clean, which is a decision and is theirs.
        if agent_of(request.headers) and is_persons_work(was) and not overwrite:
            raise HTTPException(
                status_code=409,
                detail=(f"{item_id} already carries a mask a person saved; an agent does not paint "
                        f"over that. Label an image without one, or pass overwrite=1 if replacing it "
                        f"is really what was asked for."))
        class_ids, stamp_path = save_mask_array(
            project_id, sanitize_filename(item_id), mask_arr, png_bytes=mask_bytes)
        has_foreground = bool(class_ids)
        for item in items:
            if item.get("id") == item_id:
                annotation = item.get("annotation") or {}
                annotation["markedClean"] = False
                # Who saved it, so the next agent can tell a person's work from
                # its own. A person's save clears it.
                who = agent_of(request.headers)
                if who:
                    annotation["by"] = who[0]
                else:
                    annotation.pop("by", None)
                refresh_annotation(annotation, stamp_path, ids=class_ids)
                annotation["revision"] = int(annotation.get("revision", 0)) + 1
                annotation["lastSavedAt"] = datetime.now(timezone.utc).isoformat()
                # a save through here is a person's (or an agent's) decision,
                # no longer a run's draft
                annotation.pop("draft", None)
                annotation.pop("draftRun", None)
                annotation.pop("draftReason", None)
                item["annotation"] = annotation
                break
        index["items"] = items
        save_annotate_index(project_id, index)
    touch_project(project_id)
    note_agent(request.headers, action="mask_put", project_id=project_id, item_id=item_id)
    return {
        "status": "ok",
        "annotation": {
            "hasMask": True,
            "hasForeground": has_foreground,
            "markedClean": False,
            "classIds": class_ids,
            "revision": annotation.get("revision", 1),
            "lastSavedAt": annotation.get("lastSavedAt"),
        },
    }


@router.get("/projects/{project_id}/datasets/annotate/mask-stats")
def annotate_mask_stats(project_id: str):
    """Area and region count of every mask, a person's and machine-made apart,
    with the machine-made ones that fall outside the person's range. Machine-
    made is drafted (prelabel, mark_review) or written by an agent; the answer
    keeps the names hand and drafts.

    After a run drafts a project, this is what tells a person where to look
    first: a draft with far fewer or far more regions than any hand mask holds
    is the one to open. Reads the masks; no inference.
    """
    from ..core.mask_stats import summarise_masks
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    index = load_annotate_index(project_id)
    rows = []
    for item in index.get("items", []):
        ann = item.get("annotation") or {}
        if not ann.get("hasMask"):
            continue
        # A mask file with nothing painted in it is nobody's mask: every image
        # anyone opens gets one, and counted as hand they would pull the hand
        # range down to no regions at all. One a person marked clean is theirs.
        if not (ann.get("hasForeground") or ann.get("classIds") or ann.get("markedClean")):
            continue
        # Read from where the mask lives: a tiled image's is the array, and a
        # PNG beside one is an export that can be stale.
        arr = load_mask_array(project_id, str(item.get("id")))
        if arr is None:
            continue
        # Who made it: flagged or drafted, an agent (the trainer stamps "by" on
        # an agent's save), or a person. Split on the draft flag alone, a run's
        # own unflagged spot_write masks were counted as hand, and the model
        # was measured against -- and asked the person about -- a range of its
        # own. The bridge's _made_by reads the same fields.
        who = "draft" if ann.get("draft") else ("agent" if ann.get("by") else "hand")
        rows.append((item.get("id"), item.get("name"), bool(ann.get("draft")), arr, who))
    return summarise_masks(rows)


@router.post("/projects/{project_id}/datasets/annotate/mark-clean")
async def mark_images_clean(project_id: str, request: Request, overwrite: int = Query(0)):
    """Mark images as 'no defects' — write all-zero mask PNGs server-side."""
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    body = await request.json()
    image_ids: list[str] = body.get("image_ids", [])
    if not image_ids:
        return {"marked": 0}
    lock = get_project_lock(project_id)
    marked = 0
    with lock:
        id_set = set(image_ids)
        # Build a lookup: image_id -> (width, height)
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        # An agent may not blank a person's paint. Clean over their clean mark
        # writes what is already there, so only painted masks count here, and
        # that mark stays theirs (below).
        refuse_agent_overwrite(request.headers, items, id_set, overwrite=bool(overwrite),
                               clean_counts=False)
        who = agent_of(request.headers)
        dims: dict[str, tuple[int, int]] = {}
        for item in items:
            iid = item.get("id", "")
            if iid in id_set:
                dims[iid] = (item.get("width", 0), item.get("height", 0))
        # Write all-zero mask PNGs
        written: dict[str, object] = {}
        for iid in id_set:
            w, h = dims.get(iid, (0, 0))
            if w == 0 or h == 0:
                continue
            mask_arr = np.zeros((h, w), dtype=np.uint8)
            _, written[iid] = save_mask_array(project_id, sanitize_filename(iid), mask_arr)
        marked = len(written)
        # Update index
        for item in items:
            # Only what was actually written: an image the loop skipped for
            # want of its dimensions keeps whatever it was annotated with.
            if item.get("id") in written:
                was = item.get("annotation") or {}
                annotation = {
                    "markedClean": True,
                    "revision": int(was.get("revision", 0)) + 1,
                    "lastSavedAt": datetime.now(timezone.utc).isoformat(),
                }
                refresh_annotation(annotation, written[item["id"]], ids=[])
                # As on the mask route: the next agent tells its own work by
                # this. Clean over a person's clean mark replaced nothing, so
                # the mark stays theirs; stamped as the agent's, it no longer
                # stopped that agent's next paint or unmark.
                persons_mark = (bool(was.get("markedClean")) and not was.get("hasForeground")
                                and is_persons_work(was))
                if who and not persons_mark:
                    annotation["by"] = who[0]
                item["annotation"] = annotation
        index["items"] = items
        save_annotate_index(project_id, index)
    touch_project(project_id)
    note_agent(request.headers, action="mark_clean", project_id=project_id, count=marked)
    return {"marked": marked}


@router.post("/projects/{project_id}/datasets/annotate/review")
async def mark_review(project_id: str, request: Request):
    """Flag images as needing a person's eyes, or clear the flag.

    Body: ``{"image_ids": [...], "reason": "...", "review": true}``.

    The flag is the index's ``draft`` -- the same one prelabel sets on a
    machine-made mask -- so every reader that already shows drafts shows
    these, and a save from the browser clears it through the normal route.
    An agent uses it when its own check failed: it kept four objects where
    the teachers show five, and it would rather say so than paint on.
    """
    if not project_dir(project_id).exists():
        # Without this the index write below creates the directory it writes
        # into, and the startup sweep then takes that for a project.
        raise HTTPException(status_code=404, detail="project not found")
    body = await request.json()
    ids = body.get("image_ids") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="image_ids must be a non-empty list")
    review = bool(body.get("review", True))
    reason = str(body.get("reason") or "")[:200]
    updated = 0
    with get_project_lock(project_id):
        index = load_annotate_index(project_id)
        wanted = set(str(x) for x in ids)
        for item in index.get("items", []):
            if item.get("id") not in wanted:
                continue
            annotation = dict(item.get("annotation") or {})
            if review:
                annotation["draft"] = True
                annotation["draftRun"] = "review"
                annotation["draftReason"] = reason
            else:
                annotation.pop("draft", None)
                annotation.pop("draftRun", None)
                annotation.pop("draftReason", None)
            item["annotation"] = annotation
            updated += 1
        save_annotate_index(project_id, index)
    touch_project(project_id)
    note_agent(request.headers, action="mark_review" if review else "clear_review",
               project_id=project_id, count=updated)
    return {"status": "ok", "updated": updated, "review": review}


@router.post("/projects/{project_id}/datasets/annotate/unmark-clean")
async def unmark_images_clean(project_id: str, request: Request, overwrite: int = Query(0)):
    """Remove the OK (clean) label from images.

    The mask PNG is NOT deleted and the annotation state is NOT reset:
    each image's mask is overwritten with an all-255 ("ignore") mask of the
    image's dimensions, and the index entry keeps ``hasMask: true`` while
    ``markedClean`` is set to false (``hasForeground: false``,
    ``classIds: []``, revision bumped). Images whose dimensions are missing
    from the index are skipped and not counted in the returned ``unmarked``.
    """
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    body = await request.json()
    image_ids: list[str] = body.get("image_ids", [])
    if not image_ids:
        return {"unmarked": 0}
    lock = get_project_lock(project_id)
    unmarked = 0
    with lock:
        id_set = set(image_ids)
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        # An all-ignore mask wipes a person's paint and their clean mark alike.
        refuse_agent_overwrite(request.headers, items, id_set, overwrite=bool(overwrite))
        who = agent_of(request.headers)
        # Build dims lookup for zero→255 mask rewrite
        dims: dict[str, tuple[int, int]] = {}
        for item in items:
            iid = item.get("id", "")
            if iid in id_set:
                dims[iid] = (item.get("width", 0), item.get("height", 0))
        # Overwrite masks with all-255 (ignore) instead of deleting
        written: dict[str, object] = {}
        for iid in id_set:
            w, h = dims.get(iid, (0, 0))
            if w == 0 or h == 0:
                continue
            mask_arr = np.full((h, w), 255, dtype=np.uint8)
            _, written[iid] = save_mask_array(project_id, sanitize_filename(iid), mask_arr)
        unmarked = len(written)
        for item in items:
            # Only what was actually written: an image the loop skipped for
            # want of its dimensions keeps whatever it was annotated with.
            if item.get("id") in written:
                annotation = {
                    "markedClean": False,
                    "revision": int((item.get("annotation") or {}).get("revision", 0)) + 1,
                    "lastSavedAt": datetime.now(timezone.utc).isoformat(),
                }
                refresh_annotation(annotation, written[item["id"]], ids=[])
                if who:
                    annotation["by"] = who[0]
                item["annotation"] = annotation
        index["items"] = items
        save_annotate_index(project_id, index)
    touch_project(project_id)
    note_agent(request.headers, action="unmark_clean", project_id=project_id, count=unmarked)
    return {"unmarked": unmarked}


def _clear_class_in(project_id: str, image_ids: list[str], class_id: int) -> tuple[int, int]:
    """clear_class_from_images's read, edit and write; returns (updated, skipped).

    Every listed image's mask is read and written back under the project
    lock, and a tiled one is decompressed whole. The handler runs this in a
    worker thread: on the event loop it held every other request to the API
    until the last image was saved.
    """
    lock = get_project_lock(project_id)
    updated = 0
    skipped = 0
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        by_id = {item.get("id"): item for item in items}
        now = datetime.now(timezone.utc).isoformat()
        for iid in dict.fromkeys(image_ids):  # de-dupe, keep order
            item = by_id.get(iid)
            if item is None:
                skipped += 1
                continue
            safe_id = sanitize_filename(iid)
            # Read and write where the mask lives. A tiled image's mask is the
            # array: editing the PNG export beside it changed nothing, and the
            # class was back on the next listing.
            if is_tiled(project_id, safe_id):
                from ..core.zarr_mask import class_ids_from_tally, load_class_tally
                known = class_ids_from_tally(load_class_tally(project_id, safe_id))
                if known is not None and class_id not in known:
                    skipped += 1  # the tally says so; no need to read the array
                    continue
            arr = load_mask_array(project_id, safe_id)
            if arr is None or not (arr == class_id).any():
                skipped += 1
                continue
            arr = arr.copy()
            arr[arr == class_id] = 0
            class_ids, stamp = save_mask_array(project_id, safe_id, arr)
            annotation = item.get("annotation") or {}
            refresh_annotation(annotation, stamp, ids=class_ids)
            annotation.update({
                "revision": int(annotation.get("revision", 0)) + 1,
                "lastSavedAt": now,
            })
            # As on a class purge and on a save: clearing a class out of a mask
            # is a decision about that mask, so it is no longer a run's draft
            # waiting for someone to look at it.
            annotation.pop("draft", None)
            annotation.pop("draftRun", None)
            annotation.pop("draftReason", None)
            item["annotation"] = annotation
            updated += 1
        index["items"] = items
        save_annotate_index(project_id, index)
    return updated, skipped


@router.post("/projects/{project_id}/datasets/annotate/clear-class")
async def clear_class_from_images(project_id: str, request: Request):
    """Remove one class's strokes from the given images' saved masks.

    Pixels equal to ``class_id`` become 0 (explicit background) — the same
    semantics as the single-image class clear in the annotator. Images
    without a saved mask, or whose mask does not contain the class, are
    counted as ``skipped``. Index entries get recomputed classIds /
    hasForeground and a bumped revision.
    """
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    body = await request.json()
    image_ids: list[str] = body.get("image_ids", [])
    try:
        class_id = int(body.get("class_id", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="class_id must be an integer")
    if not (1 <= class_id <= 254):
        raise HTTPException(status_code=400, detail="class_id must be in 1..254")
    if not image_ids:
        return {"updated": 0, "skipped": 0}
    updated, skipped = await asyncio.to_thread(_clear_class_in, project_id, image_ids, class_id)
    touch_project(project_id)
    note_agent(request.headers, action="clear_class", project_id=project_id, count=updated)
    return {"updated": updated, "skipped": skipped}


@router.post("/projects/{project_id}/datasets/annotate/augment")
async def augment_annotate_items(project_id: str, request: Request):
    """Synthesize new labeled items via Perlin CutPaste and/or lighting variants.

    Body:
      { "count": 10,               // samples per enabled mode, 1..500 (default 10)
        "modes": {                 // optional mode toggles (default: perlin only)
          "perlin": true,          //   Perlin CutPaste synthesis (default true)
          "lighting": false        //   lighting-variant synthesis (default false)
        },                         // at least one mode must be enabled; `count`
                                   // applies per mode, so both => 2x output
        "perlin_strength": 6.0,    // max Perlin displacement px (default 6)
        "color_jitter": 0.15,      // per-channel jitter strength (default 0.15)
        "defects_per_image": [1,4],// random range of defects to paste (default [1,4])
        "seed": 42,                // optional — reproducible generation
        "class_id": 0,             // optional; 0 / null = all classes. A positive
                                   // id restricts synthesis to that class's pixels
                                   // (other classes are zeroed out before cut-paste)
        "lighting_variants":       // subset of ["daytime","evening","night"] used
          ["daytime","evening","night"],  // when modes.lighting is enabled (default all)
        "use_clean_hosts": false   // include "Mark Clean" images in the Perlin
                                   // CutPaste host (paste-target) pool
      }

    Reads every labeled (hasMask && hasForeground) image from the annotate
    directory, cuts out defect connected components, warps them with a
    Perlin vector field, and pastes them onto random host images. New
    samples are written as regular annotate items with ``synthetic=true``
    metadata so the annotate list can visually distinguish them and the
    user can delete any that look unnatural.
    """
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    body = await request.json()
    try:
        count = int(body.get("count", 10))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="count must be an integer")
    if count <= 0 or count > 500:
        raise HTTPException(status_code=400, detail="count must be in [1, 500]")
    perlin_strength = float(body.get("perlin_strength", 6.0))
    color_jitter = float(body.get("color_jitter", 0.15))
    dpi_raw = body.get("defects_per_image", [1, 4])
    try:
        defects_per_image = (int(dpi_raw[0]), int(dpi_raw[1]))
    except Exception:
        defects_per_image = (1, 4)
    seed = body.get("seed")

    # Optional class filter. 0 / missing / None means "all classes" and the
    # source masks go through untouched. A positive class_id restricts the
    # synthesis to just that class's pixels — other classes on the same mask
    # are zeroed out before cut-paste. Keeps multi-class projects from
    # accidentally blending defect types.
    cls_raw = body.get("class_id")
    try:
        class_filter = int(cls_raw) if cls_raw is not None else 0
    except (TypeError, ValueError):
        class_filter = 0
    if class_filter < 0:
        class_filter = 0

    # Mode toggles. Either or both may be enabled; `count` is honoured per
    # enabled mode, so enabling both doubles the output. Perlin defaults ON
    # so the legacy request shape (no modes field) keeps working.
    modes_raw = body.get("modes") or {}
    if not isinstance(modes_raw, dict):
        modes_raw = {}
    perlin_enabled = bool(modes_raw.get("perlin", True))
    lighting_enabled = bool(modes_raw.get("lighting", False))
    if not perlin_enabled and not lighting_enabled:
        raise HTTPException(
            status_code=400,
            detail="at least one of modes.perlin / modes.lighting must be enabled",
        )
    # Lighting variant selection (any subset of daytime/evening/night).
    lighting_variants_raw = body.get("lighting_variants") or ["daytime", "evening", "night"]
    if isinstance(lighting_variants_raw, str):
        lighting_variants_raw = [lighting_variants_raw]
    lighting_variants = [str(v) for v in lighting_variants_raw if isinstance(v, str)]

    # When True, "Mark Clean" (defect-free) images join the host pool for
    # Perlin CutPaste so synthesized defects can be pasted onto verified-OK
    # backgrounds too. The labeled pairs still drive defect-crop extraction.
    use_clean_hosts = bool(body.get("use_clean_hosts", False))

    images_dir = annotate_images_dir(project_id)
    masks_dir = annotate_masks_dir(project_id)
    if not images_dir.exists() or not masks_dir.exists():
        raise HTTPException(status_code=400, detail="no annotate data yet")

    # Build (image, mask) pairs from labeled items only. When the user
    # opted in via use_clean_hosts, also collect "Mark Clean" pairs to
    # extend the host (paste-target) pool — they never feed defect-crop
    # extraction since they have no foreground.
    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        pairs: list[tuple[Path, Path]] = []
        clean_pairs: list[tuple[Path, Path]] = []
        for item in items:
            ann = item.get("annotation") or {}
            if not ann.get("hasMask"):
                continue
            if ann.get("synthetic"):
                # Don't feed synths back into synthesis — avoid drift.
                continue
            filename = item.get("filename")
            if not filename:
                continue
            img_p = images_dir / filename
            mask_p = masks_dir / f"{item['id']}.png"
            if not (img_p.exists() and mask_p.exists()):
                continue
            if ann.get("markedClean"):
                clean_pairs.append((img_p, mask_p))
            elif ann.get("hasForeground"):
                pairs.append((img_p, mask_p))

        if len(pairs) < 1:
            raise HTTPException(
                status_code=400,
                detail="need at least 1 labeled image with foreground to synthesize from",
            )

        # If a specific class was requested, materialise per-pair filtered
        # mask PNGs in a temp directory so the synthesizer sees only that
        # class. Pairs whose filtered mask is empty are dropped.
        temp_mask_dir: Path | None = None
        if class_filter > 0:
            import tempfile as _tempfile

            import numpy as _np
            temp_mask_dir = Path(_tempfile.mkdtemp(prefix=f"augment_cls{class_filter}_", dir=str(base)))
            filtered_pairs: list[tuple[Path, Path]] = []
            for img_p, mask_p in pairs:
                try:
                    mask_arr = _np.array(Image.open(mask_p).convert("L"))
                except Exception:
                    continue
                keep = (mask_arr == class_filter)
                if not bool(keep.any()):
                    continue
                filt = _np.where(keep, mask_arr, 0).astype("uint8")
                out_p = temp_mask_dir / mask_p.name
                Image.fromarray(filt, mode="L").save(out_p)
                filtered_pairs.append((img_p, out_p))
            if len(filtered_pairs) < 1:
                import shutil as _shutil
                _shutil.rmtree(temp_mask_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"no labeled image has pixels for class_id={class_filter}",
                )
            pairs = filtered_pairs

        # Perlin CutPaste host pool: labeled pairs always; clean pairs added
        # only when the caller explicitly opted in. class_filter materialises
        # filtered masks tied to the labeled pairs only — clean pairs keep
        # their original (all-zero) masks since the filter is a no-op there.
        host_pairs: list[tuple[Path, Path]] | None = None
        if use_clean_hosts and clean_pairs:
            host_pairs = list(pairs) + list(clean_pairs)

        samples: list[tuple] = []
        try:
            if perlin_enabled:
                from segcore.augment import synthesize_from_labeled
                samples.extend(synthesize_from_labeled(
                    pairs,
                    n_samples=count,
                    defects_per_image=defects_per_image,
                    perlin_strength=perlin_strength,
                    color_jitter=color_jitter,
                    seed=seed,
                    host_pairs=host_pairs,
                ))
            if lighting_enabled:
                from segcore.augment import synthesize_lighting_variants
                samples.extend(synthesize_lighting_variants(
                    pairs,
                    n_samples=count,
                    variants=lighting_variants,
                    seed=seed,
                ))
        except ValueError as e:
            if temp_mask_dir is not None:
                import shutil as _shutil
                _shutil.rmtree(temp_mask_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e))
        if temp_mask_dir is not None:
            import shutil as _shutil
            _shutil.rmtree(temp_mask_dir, ignore_errors=True)

        import cv2 as _cv2
        # One read for the whole batch, as on the import routes.
        image_format = effective_image_format(project_id)
        new_items: list[dict] = []
        for img_bgr, mask_u8, meta in samples:
            new_id = f"synth_{uuid.uuid4().hex[:12]}"
            ok_img, buf_img = _cv2.imencode(".png", img_bgr)
            if not ok_img:
                continue
            try:
                suffix, img_data = encode_for_store(
                    buf_img.tobytes(), source_name=f"{new_id}.png",
                    image_format=image_format)
            except (EncodeError, OSError, ValueError):
                continue
            ok_mask, buf_mask = _cv2.imencode(".png", mask_u8)
            if not ok_mask:
                continue
            filename = f"{new_id}{suffix}"
            write_bytes_atomic(images_dir / filename, img_data)
            # The mask stays a lossless PNG whatever the project stores its
            # images as -- the class ids are read straight back out of it a
            # few lines below.
            write_bytes_atomic(masks_dir / f"{new_id}.png", buf_mask.tobytes())

            h, w = img_bgr.shape[:2]
            class_ids = sorted(int(v) for v in np.unique(mask_u8) if v != 0 and v != 255)
            new_items.append({
                "id": new_id,
                "name": filename,
                "filename": filename,
                "set": "none",
                "width": int(w),
                "height": int(h),
                "annotation": {
                    "hasMask": True,
                    "hasForeground": True,
                    "classIds": class_ids,
                    "revision": 1,
                    "lastSavedAt": datetime.now(timezone.utc).isoformat(),
                    "synthetic": True,
                    "synthSource": meta.get("source_image"),
                    "synthDefectSources": meta.get("defect_sources", []),
                    "synthKind": meta.get("kind", "perlin"),
                    "synthVariant": meta.get("variant"),
                },
            })

        items.extend(new_items)
        index["items"] = items
        save_annotate_index(project_id, index)

    touch_project(project_id)
    return {"generated": len(new_items), "items": new_items}


@router.get("/projects/{project_id}/datasets/annotate/class-presence")
def get_class_presence(project_id: str):
    """Return per-image unique class IDs from mask PNGs and Zarr masks."""
    masks_dir = annotate_masks_dir(project_id)
    if not masks_dir.exists():
        return {"items": {}}
    import cv2
    result: dict[str, list[int]] = {}
    seen_ids: set[str] = set()

    for entry in masks_dir.iterdir():
        # Handle Zarr directories
        if entry.is_dir() and entry.suffix == ".zarr":
            image_id = entry.stem
            seen_ids.add(image_id)
            try:
                from ..core.zarr_mask import zarr_to_numpy
                arr = zarr_to_numpy(project_id, image_id)
                if arr is not None:
                    unique = np.unique(arr)
                    class_ids = sorted(int(v) for v in unique if v != 0 and v != 255)
                    if class_ids:
                        result[image_id] = class_ids
            except Exception:
                pass
            continue

        # Handle PNG files
        if entry.suffix.lower() != ".png":
            continue
        if entry.stem in seen_ids:
            continue  # Zarr takes priority
        mask = _imread(str(entry), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim >= 3:
            mask = mask[:, :, 0]
        unique = np.unique(mask)
        class_ids = sorted(int(v) for v in unique if v != 0 and v != 255)
        if class_ids:
            result[entry.stem] = class_ids
    return {"items": result}
