# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image

from ..core import instance_predict
from ..core.paths import predictions_dir
from ..core.prediction_engine import (
    ensure_prediction_artifacts,
    generate_heatmap,
    predict_batch_stream,
    prediction_artifact_paths,
    resolve_predict_context,
)

router = APIRouter()


def _instance_run_context(project_id: str, run_id: str):
    """(run_path, contract) when the run is an instance run, else None.

    Instance runs have no ``model.pt``, so they must branch before
    ``resolve_predict_context`` (which 404s without one).
    """
    run_path, contract = instance_predict.resolve_instance_context(project_id, run_id)
    return (run_path, contract) if contract is not None else None


async def _sync_gen_to_async(sync_gen) -> AsyncIterator[str]:
    """Wrap a blocking sync generator as an async generator.

    Each ``next(sync_gen)`` is dispatched to a worker thread so the event
    loop stays free for other requests (e.g. UI polling endpoints).
    """
    loop = asyncio.get_running_loop()
    sentinel = object()

    def _next():
        try:
            return next(sync_gen)
        except StopIteration:
            return sentinel

    while True:
        value = await loop.run_in_executor(None, _next)
        if value is sentinel:
            break
        yield value



@router.get("/predict/batch-status")
def predict_batch_status(
    project_id: str = Query(None),
    run_id: str = Query(None),
):
    """Check if a batch inference is currently active (survives browser reload)."""
    try:
        from ..core.inference_runtime import get_inference_runtime
        runtime = get_inference_runtime()
        tracker = runtime.get_active_batch(project_id, run_id)
        if tracker is None:
            return {"active": False, "batch_id": None, "total": 0, "completed": 0, "started_at": None}
        return {
            "active": True,
            "batch_id": tracker.batch_id,
            "project_id": tracker.project_id,
            "run_id": tracker.run_id,
            "total": tracker.total,
            "completed": tracker.completed,
            "started_at": tracker.started_at,
            "client_connected": tracker.client_connected,
        }
    except Exception:
        return {"active": False, "batch_id": None, "total": 0, "completed": 0, "started_at": None}


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/status")
def predict_status(
    project_id: str, run_id: str,
    backend: str = Query("onnx"), tta: bool = Query(False),
):
    """Check which images have prediction artifacts on disk (no inference).

    Returns predicted item IDs and, for each, the detected foreground class IDs
    (extracted from score.json per_class_mean_confidence).
    """
    from ..core.paths import project_dir, resolve_run_path
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    run_path = resolve_run_path(project_id, run_id)
    if run_path is None:
        raise HTTPException(status_code=404, detail="run not found")
    pred_dir = predictions_dir(run_path, backend=backend, tta=tta)
    predicted_ids: list[str] = []
    per_image_classes: dict[str, list[int]] = {}
    if pred_dir.exists():
        for score_file in pred_dir.glob("*.score.json"):
            item_id = score_file.stem.replace(".score", "")
            has_artifact = (pred_dir / f"{item_id}.png").exists()
            if has_artifact:
                predicted_ids.append(item_id)
                try:
                    score_data = json.loads(score_file.read_text(encoding="utf-8"))
                    pcmc = score_data.get("per_class_mean_confidence")
                    if isinstance(pcmc, dict):
                        per_image_classes[item_id] = sorted(int(k) for k in pcmc if int(k) > 0)
                except (json.JSONDecodeError, OSError, ValueError):
                    pass
    return {"predicted": predicted_ids, "count": len(predicted_ids), "per_image_classes": per_image_classes}


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/verdicts")
def predict_verdicts(
    project_id: str, run_id: str,
    backend: str = Query("onnx"), tta: bool = Query(False),
    threshold: float = Query(0.0, ge=0.0, le=100.0),
    coverage: float = Query(0.5, gt=0.0, le=1.0),
    min_area: int = Query(0, ge=0),
    max_area: int = Query(0, ge=0),
):
    """Per-image detected / missed / over-detected at one slider position.

    The components were cut once when the prediction was written; this only
    sums histogram tails, so the whole run is re-judged in one request rather
    than one per image. That is what lets the confidence slider move the
    badges without a round trip per thumbnail.

    Images with no annotation get a null verdict: there is nothing to be right
    or wrong against.
    """
    from ..core.detection_verdict import (
        SUMMARY_SUFFIX,
        conf_bin_for_percent,
        evaluate_summary,
        load_prepared_summary,
    )
    from ..core.paths import project_dir, resolve_run_path
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    run_path = resolve_run_path(project_id, run_id)
    if run_path is None:
        raise HTTPException(status_code=404, detail="run not found")

    pred_dir = predictions_dir(run_path, backend=backend, tta=tta)
    # Predictions written before summaries existed have a mask and a
    # confidence map and no components, and the serving path hands them back
    # from cache without ever building one. Fill them in here, off this
    # thread, so opening the tab is what makes the list complete rather than
    # remembering to re-run inference.
    from ..core.prediction_engine import start_summary_backfill
    pending = start_summary_backfill(project_id, run_path, backend, tta)
    min_conf_bin = conf_bin_for_percent(threshold)
    verdicts: dict[str, dict] = {}
    counts = {"detected": 0, "missed": 0, "over": 0, "clean": 0, "unjudged": 0}
    if pred_dir.exists():
        for path in pred_dir.glob(f"*{SUMMARY_SUFFIX}"):
            item_id = path.name[: -len(SUMMARY_SUFFIX)]
            summary = load_prepared_summary(pred_dir, item_id)
            if summary is None:
                continue
            result = evaluate_summary(
                summary,
                min_conf_bin=min_conf_bin,
                coverage=coverage,
                min_area=min_area,
                max_area=max_area,
            )
            verdicts[item_id] = result
            counts[result["verdict"] or "unjudged"] += 1
    return {
        "verdicts": verdicts,
        "counts": counts,
        "threshold": threshold,
        "coverage": coverage,
        # How many predictions still lack components. Non-zero means a
        # backfill is running and asking again shortly will return more.
        "pending": pending,
    }


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}.png")
def predict_run_mask(project_id: str, run_id: str, item_id: str, backend: str = Query("onnx"), tta: bool = Query(False), force: bool = Query(False), readonly: bool = Query(False)):
    instance_ctx = _instance_run_context(project_id, run_id)
    if instance_ctx is not None:
        run_path_i, _contract = instance_ctx
        if readonly:
            *_head, mask_path, _conf, _score = instance_predict.instance_artifact_paths(run_path_i, item_id)
            if not mask_path.exists():
                raise HTTPException(status_code=404, detail="prediction artifact not found")
            return FileResponse(mask_path, media_type="image/png")
        _json, _overlay, mask_path, _conf, _score = instance_predict.ensure_instance_artifacts(
            project_id, run_path_i, item_id, force=force)
        return FileResponse(mask_path, media_type="image/png")
    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    if readonly:
        pred_path, _confidence_path, _score_path = prediction_artifact_paths(run_path, backend, item_id, tta=tta)
        if not pred_path.exists():
            raise HTTPException(status_code=404, detail="prediction artifact not found")
        return FileResponse(pred_path, media_type="image/png")
    pred_path, _confidence_path, _score = ensure_prediction_artifacts(project_id, run_path, model_path, item_id, backend, tta=tta, force=force)
    return FileResponse(pred_path, media_type="image/png")


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/confidence.png")
def predict_run_confidence(project_id: str, run_id: str, item_id: str, backend: str = Query("onnx"), tta: bool = Query(False), force: bool = Query(False), readonly: bool = Query(False)):
    instance_ctx = _instance_run_context(project_id, run_id)
    if instance_ctx is not None:
        run_path_i, _contract = instance_ctx
        if readonly:
            *_head, conf_path, _score = instance_predict.instance_artifact_paths(run_path_i, item_id)
            if not conf_path.exists():
                raise HTTPException(status_code=404, detail="prediction artifact not found")
            return FileResponse(conf_path, media_type="image/png")
        _json, _overlay, _mask, conf_path, _score = instance_predict.ensure_instance_artifacts(
            project_id, run_path_i, item_id, force=force)
        return FileResponse(conf_path, media_type="image/png")
    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    if readonly:
        _pred_path, confidence_path, _score_path = prediction_artifact_paths(run_path, backend, item_id, tta=tta)
        if not confidence_path.exists():
            raise HTTPException(status_code=404, detail="prediction artifact not found")
        return FileResponse(confidence_path, media_type="image/png")
    _pred_path, confidence_path, _score = ensure_prediction_artifacts(project_id, run_path, model_path, item_id, backend, tta=tta, force=force)
    return FileResponse(confidence_path, media_type="image/png")


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/score")
def predict_run_score(project_id: str, run_id: str, item_id: str, backend: str = Query("onnx"), tta: bool = Query(False), force: bool = Query(False), readonly: bool = Query(False)):
    instance_ctx = _instance_run_context(project_id, run_id)
    if instance_ctx is not None:
        run_path_i, _contract = instance_ctx
        if readonly:
            *_head, score_path = instance_predict.instance_artifact_paths(run_path_i, item_id)
            if not score_path.exists():
                raise HTTPException(status_code=404, detail="prediction artifact not found")
            return json.loads(score_path.read_text(encoding="utf-8"))
        *_paths, score = instance_predict.ensure_instance_artifacts(
            project_id, run_path_i, item_id, force=force)
        return score
    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    if readonly:
        _pred_path, _confidence_path, score_path = prediction_artifact_paths(run_path, backend, item_id, tta=tta)
        if not score_path.exists():
            raise HTTPException(status_code=404, detail="prediction artifact not found")
        return json.loads(score_path.read_text(encoding="utf-8"))
    _pred_path, _confidence_path, score = ensure_prediction_artifacts(project_id, run_path, model_path, item_id, backend, tta=tta, force=force)
    return score


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/pixel-histogram")
def pixel_confidence_histogram(
    project_id: str, run_id: str,
    backend: str = Query("onnx"), bins: int = Query(50),
):
    """Aggregate a foreground-confidence histogram over stored ``*.probs.npz`` artifacts.

    Only scans ``<run>/predictions[_coreml]/*.probs.npz`` (the legacy
    artifact format). Images whose probabilities were saved in the current
    per-image ``.probs.npy`` format are NOT included, so the histogram covers
    only the subset of images that have an ``.npz`` artifact — not all images.
    For each ``.npz`` found, every pixel contributes: FG confidence is the sum
    of all non-background class probabilities, clipped to [0, 1]. Unreadable
    files are skipped silently.

    Returns {bins: [bin edges, length bins+1], counts: [length bins], total_pixels: int}
    """
    import numpy as np
    run_path, _model_path, backend = resolve_predict_context(project_id, run_id, backend)
    # Deliberately not TTA-aware, matching what this endpoint has always
    # read. The histogram describes the plain pass; the TTA directory is a
    # separate artifact set and no caller asks this route for it.
    pred_dir = predictions_dir(run_path, backend=backend)
    if not pred_dir.exists():
        raise HTTPException(status_code=404, detail="No predictions found")

    counts = np.zeros(bins, dtype=np.int64)
    bin_edges = np.linspace(0, 1, bins + 1)
    total_pixels = 0

    for npz_path in sorted(pred_dir.glob("*.probs.npz")):
        try:
            data = np.load(npz_path)
            probs = data["probs"].astype(np.float32)  # (C, H, W)
            # FG confidence = sum of all non-background class probs
            fg_conf = probs[1:].sum(axis=0).clip(0, 1).ravel()
            h, _ = np.histogram(fg_conf, bins=bin_edges)
            counts += h
            total_pixels += len(fg_conf)
        except Exception:
            continue

    return {
        "bins": [round(float(b), 4) for b in bin_edges],
        "counts": [int(c) for c in counts],
        "total_pixels": int(total_pixels),
    }


@router.post("/projects/{project_id}/train/runs/{run_id}/predict/batch")
async def predict_batch(
    project_id: str,
    run_id: str,
    item_ids: list[str] = Body(..., embed=True),
    backend: str = Body("onnx"),
    tta: bool = Body(False),
    force: bool = Body(False),
):
    """Stream batch prediction results as NDJSON (one JSON line per image).

    Pipelines image loading and post-processing with GPU inference for
    higher throughput when processing many images.

    This endpoint is ``async def`` so the NDJSON generator does not hold an
    anyio threadpool token for the full duration of inference — which would
    starve concurrent UI polling requests and cause the browser to hang.
    """
    instance_ctx = _instance_run_context(project_id, run_id)
    if instance_ctx is not None:
        run_path_i, _contract = instance_ctx
        return StreamingResponse(
            _sync_gen_to_async(instance_predict.instance_batch_stream(
                project_id, run_path_i, item_ids, force=force,
            )),
            media_type="application/x-ndjson",
        )
    run_path, model_path, backend_resolved = resolve_predict_context(project_id, run_id, backend)
    # Sequential processing: one image at a time, reliable ordering, no skips.
    return StreamingResponse(
        _sync_gen_to_async(predict_batch_stream(
            project_id, run_path, model_path, item_ids,
            backend_resolved, tta=tta, force=force,
        )),
        media_type="application/x-ndjson",
    )


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/agreement")
def predict_agreement(
    project_id: str, run_id: str,
    backend: str = Query("onnx"), tta: bool = Query(False),
    limit: int = Query(0, ge=0, description="0 = every labelled image"),
):
    """How closely this run's existing predictions follow the project's own
    hand masks -- every labelled image, no inference.

    The number that decides whether to let the run draft the rest: a run can
    score well on its first few labelled images and far worse on all of them.
    An image without a prediction on disk, or whose own mask cannot be read,
    is counted, not scored; run the batch prediction first if that number is
    large.
    """
    from ..core.agreement import load_mask, score_pair, summarise
    from ..core.annotate_index import load_annotate_index
    from ..core.mask_write import load_mask_array
    from ..core.paths import project_dir, resolve_run_path
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    run_path = resolve_run_path(project_id, run_id)
    if run_path is None:
        raise HTTPException(status_code=404, detail="run not found")
    pred_dir = predictions_dir(run_path, backend=backend, tta=tta)
    items = [it for it in load_annotate_index(project_id).get("items", [])
             if (it.get("annotation") or {}).get("hasMask")]
    if limit:
        items = items[:limit]
    rows = []
    missing = 0
    for it in items:
        iid = it.get("id", "")
        pd_path = pred_dir / f"{iid}.png"
        if not pd_path.exists():
            missing += 1
            continue
        # From where the mask lives. Read as <id>.png, a tiled image -- an
        # array with no PNG -- was labelled but neither scored nor missing.
        gt = load_mask_array(project_id, iid)
        if gt is None:
            missing += 1
            continue
        pred = load_mask(pd_path)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
        rows.append(score_pair(iid, gt, pred))
    out = summarise(rows, n_labelled=len(items), n_missing=missing)
    out.update({"run_id": run_id, "backend": backend})
    return out


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/operating-points")
def predict_operating_points(
    project_id: str, run_id: str,
    backend: str = Query("onnx"), tta: bool = Query(False),
    coverage: float = Query(0.5, gt=0.0, le=1.0),
):
    """Named operating points swept over confidence AND min_area.

    metrics.json already carries operating points, but they are a
    threshold-only sweep taken on the val set during training: they cannot
    answer for the area filter, which is the other control on the panel, and
    they cannot be produced at all for a run that already finished.

    This sweeps both axes over the prediction summaries already on disk -- the
    components were cut once at zero confidence and carry a histogram, so area
    at any threshold is a suffix sum. Nothing is inferred and no model is
    loaded.

    The recall-first and balanced points are constrained to settings that cost
    no defects against the same threshold with the filter off; the
    precision-first point may spend some and reports how many.
    """
    from ..core.operating_sweep import load_run_summaries, sweep_operating_points
    from ..core.paths import project_dir, resolve_run_path

    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    run_path = resolve_run_path(project_id, run_id)
    if run_path is None:
        raise HTTPException(status_code=404, detail="run not found")

    pred_dir = predictions_dir(run_path, backend=backend, tta=tta)
    summaries = load_run_summaries(pred_dir)
    return sweep_operating_points(summaries, coverage=coverage)


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/postprocess.png")
def predict_run_postprocess(
    project_id: str,
    run_id: str,
    item_id: str,
    confidence_threshold: float = Query(0.0),
    min_area_px: int = Query(0),
    max_area_px: int = Query(0),
    morphology: str = Query("none"),
    morphology_kernel: int = Query(3),
    backend: str = Query("onnx"),
    tta: bool = Query(False),
    readonly: bool = Query(False),
):
    """Return post-processed prediction mask as PNG."""
    from segcore.postprocess import postprocess as _postprocess

    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    if readonly:
        pred_path, confidence_path, _score_path = prediction_artifact_paths(run_path, backend, item_id, tta=tta)
        if not pred_path.exists() or not confidence_path.exists():
            raise HTTPException(status_code=404, detail="prediction artifact not found")
    else:
        pred_path, confidence_path, _score = ensure_prediction_artifacts(
            project_id, run_path, model_path, item_id, backend, tta=tta,
        )
    pred_np = np.asarray(Image.open(pred_path).convert("L"))
    conf_np = np.asarray(Image.open(confidence_path).convert("L")).astype("float32") / 255.0
    result = _postprocess(
        pred_np,
        conf_np,
        confidence_threshold=confidence_threshold,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        morphology=morphology,
        morphology_kernel=morphology_kernel,
    )
    result_img = Image.fromarray(result.astype("uint8"), mode="L")
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/heatmap/confidence.png")
def predict_heatmap_confidence(
    project_id: str, run_id: str, item_id: str,
    backend: str = Query("onnx"), tta: bool = Query(False),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    min_area: int = Query(0, ge=0),
    max_area: int = Query(0, ge=0),
):
    """Confidence heatmap: max(softmax) per pixel, turbo colormap.

    `threshold`, `min_area`, and `max_area` mirror the UI Confidence slider
    and the Min/Max Area inputs so the heatmap reacts to those controls.
    """
    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    png_bytes = generate_heatmap(
        project_id, run_path, model_path, item_id, backend, tta, "confidence",
        threshold=threshold, min_area=min_area, max_area=max_area,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/heatmap/class/{class_id}.png")
def predict_heatmap_class(
    project_id: str, run_id: str, item_id: str, class_id: int,
    backend: str = Query("onnx"), tta: bool = Query(False),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    min_area: int = Query(0, ge=0),
    max_area: int = Query(0, ge=0),
):
    """Per-class probability heatmap: softmax[class_id] per pixel, turbo colormap.

    `threshold`, `min_area`, and `max_area` mirror the UI Confidence slider
    and the Min/Max Area inputs so the heatmap reacts to those controls.
    """
    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    png_bytes = generate_heatmap(
        project_id, run_path, model_path, item_id, backend, tta, "class", class_id,
        threshold=threshold, min_area=min_area, max_area=max_area,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/heatmap/error.png")
def predict_heatmap_error(
    project_id: str, run_id: str, item_id: str,
    backend: str = Query("onnx"), tta: bool = Query(False),
):
    """Error map: TP=green, FP=red, FN=yellow."""
    run_path, model_path, backend = resolve_predict_context(project_id, run_id, backend)
    png_bytes = generate_heatmap(project_id, run_path, model_path, item_id, backend, tta, "error")
    return Response(content=png_bytes, media_type="image/png")

