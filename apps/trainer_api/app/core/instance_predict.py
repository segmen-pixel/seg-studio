# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance-mode prediction engine (docs/design_instance_segmentation_v098.md §4.2).

Runs the fine-tuned RF-DETR-Seg checkpoint over annotate images and writes,
per item:

  instances/{item}.json         — RLE instances + count (the M3 contract)
  instances/{item}.overlay.png  — composite with colored fills + numbered badges
  predictions/{item}.png        — legacy semantic-style composite mask
  predictions/{item}.confidence.png — per-pixel instance confidence
  predictions/{item}.score.json — score in the semantic predict shape

The legacy artifacts land in ``predictions/`` under the exact filenames the
semantic pipeline uses, so ``/predict/status`` and existing mask/score
viewers keep working for instance runs without changes.

rfdetr is imported lazily (inside the model loader) so this module — and the
routers importing it — work in environments without the dependency; only an
actual inference request requires it.
"""
from __future__ import annotations

import gc
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from .annotate_index import find_annotate_image
from .paths import project_dir, resolve_run_path, run_dir
from .security import _sanitize_filename

logger = logging.getLogger(__name__)

# One resident model: instance predict is dev-box interactive, and rfdetr
# checkpoints are hundreds of MB — caching more than one invites OOM.
_model_lock = threading.Lock()
_model_cache: dict = {"key": None, "model": None}
# GPU inference is serialized: rfdetr models are not thread-safe and two
# concurrent predicts on one GPU invite OOM / internal-state races.
_predict_lock = threading.Lock()
_item_locks: dict[str, threading.Lock] = {}
_item_locks_guard = threading.Lock()


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clear_instance_model_cache() -> None:
    """Drop the resident rfdetr model and free its VRAM.

    Called from the training launcher's pre-spawn GPU release (alongside the
    SAM / torch caches) so a queued training run can claim the memory.
    """
    with _model_lock:
        _model_cache["key"] = None
        _model_cache["model"] = None
    _release_cuda()


def instance_contract(run_path: Path) -> dict | None:
    """Parse instance_inference.json; None when absent/corrupt (= not an instance run)."""
    contract_path = run_path / "instance_inference.json"
    if not contract_path.exists():
        return None
    try:
        data = json.loads(contract_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def resolve_instance_context(project_id: str, run_id: str) -> tuple[Path, dict | None]:
    """Resolve (run_path, contract). Contract is None for non-instance runs."""
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    run_path = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    if not run_path.is_dir():
        raise HTTPException(status_code=404, detail="run not found")
    return run_path, instance_contract(run_path)


def instance_artifact_paths(
    run_path: Path, item_id: str, *, ensure_dir: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """(instances.json, overlay.png, legacy mask, confidence, score) for one item."""
    item_id = _sanitize_filename(item_id)  # neutralize ..\ / ../ path traversal
    inst_dir = run_path / "instances"
    pred_dir = run_path / "predictions"
    if ensure_dir:
        inst_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
    return (
        inst_dir / f"{item_id}.json",
        inst_dir / f"{item_id}.overlay.png",
        pred_dir / f"{item_id}.png",
        pred_dir / f"{item_id}.confidence.png",
        pred_dir / f"{item_id}.score.json",
    )


def _resolve_checkpoint(run_path: Path, contract: dict) -> Path:
    name = contract.get("checkpoint")
    if name:
        candidate = run_path / "rfdetr" / _sanitize_filename(str(name))
        if candidate.exists():
            return candidate
    fallback = sorted((run_path / "rfdetr").glob("checkpoint_best*.pth"))
    if fallback:
        return fallback[-1]
    raise HTTPException(status_code=404, detail="instance checkpoint not found")


def _get_model(model_size: str, checkpoint: Path):
    key = (model_size, str(checkpoint))
    with _model_lock:
        if _model_cache["key"] == key and _model_cache["model"] is not None:
            return _model_cache["model"]
        if _model_cache["model"] is not None:
            # Switching checkpoints: drop the old model and free its VRAM
            # before loading the new one, never hold both.
            _model_cache["key"] = None
            _model_cache["model"] = None
            _release_cuda()
        from segcore.instseg.train_rfdetr import build_model

        model = build_model(model_size, pretrain_weights=str(checkpoint))
        _model_cache["key"] = key
        _model_cache["model"] = model
        return model


def _resolve_class_id(run_path: Path) -> int:
    """The semantic class the run trained on (for the legacy composite mask)."""
    from .instance_training import _resolve_class_id as resolve

    config: dict = {}
    config_path = run_path / "train_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
    return resolve(run_path, config)


def _resolve_class_name(run_path: Path, class_id: int) -> str:
    """Display name of the counted class; falls back to ``classN``."""
    classes_file = run_path / "classes.json"
    if classes_file.exists():
        try:
            data = json.loads(classes_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        for item in data.get("classes", []):
            if int(item.get("id", -1)) == class_id and item.get("name"):
                return str(item["name"])
    return f"class{class_id}"



def _semantic_class_of(contract: dict, run_path: Path) -> dict[int, int]:
    """COCO category id -> semantic class id (identity when unmapped)."""
    mapping = contract.get("coco_category_of") or {}
    out: dict[int, int] = {}
    for sem, cat in mapping.items():
        try:
            out[int(cat)] = int(sem)
        except (TypeError, ValueError):
            continue
    if out:
        return out
    # Pre-multi-class exports: one category, one class.
    ids = contract.get("class_ids") or [_resolve_class_id(run_path)]
    return {1: int(ids[0])}


def _class_names(contract: dict, run_path: Path) -> dict[int, str]:
    """{semantic class id: display name} from the contract, else classes.json."""
    names: dict[int, str] = {}
    for k, v in (contract.get("class_names") or {}).items():
        try:
            names[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    if names:
        return names
    from .instance_training import class_name_map

    return class_name_map(run_path)


def _item_lock(run_path: Path, item_id: str) -> threading.Lock:
    key = f"{run_path}:{item_id}"
    with _item_locks_guard:
        return _item_locks.setdefault(key, threading.Lock())


def instance_highlight_overlay_path(run_path: Path, item_id: str) -> Path:
    """Path of the per-instance ("detection highlight") overlay variant."""
    item_id = _sanitize_filename(item_id)
    return run_path / "instances" / f"{item_id}.overlay_inst.png"


def ensure_instance_highlight_overlay(run_path: Path, item_id: str) -> Path:
    """Render (and cache) the per-instance overlay from the stored
    instances.json — one vivid colour per detected object, on the blue
    background wash. Rendered lazily so runs no one toggles into this view
    never pay for it; regenerated when the base instances.json is newer.

    Requires the base artifacts to exist already (the caller runs prediction
    first); raises 404 otherwise so the UI can prompt a predict.
    """
    import cv2

    from segcore.instseg.overlay import draw_instance_overlay
    from segcore.instseg.rle import decode_rle

    json_path, *_rest = instance_artifact_paths(run_path, item_id)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="instance artifact not found")
    out_path = instance_highlight_overlay_path(run_path, item_id)
    if out_path.exists() and out_path.stat().st_mtime >= json_path.stat().st_mtime:
        return out_path

    with _item_lock(run_path, item_id):
        if out_path.exists() and out_path.stat().st_mtime >= json_path.stat().st_mtime:
            return out_path
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        insts = payload.get("instances", [])
        # The source image lives under the project's annotate dir; the run
        # path is <projects>/<project_id>/training/runs/<run_id>.
        project_id = run_path.parents[2].name
        image_path = find_annotate_image(project_id, item_id)
        if image_path is None:
            raise HTTPException(status_code=404, detail="image not found")
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=500, detail="failed to read image")
        masks = [decode_rle(i["rle"]).astype("uint8") for i in insts]
        confs = [float(i.get("conf", 0.0)) for i in insts]
        cids = [int(i.get("class_id", 1)) for i in insts]
        names = {int(k): v for k, v in (payload.get("class_names") or {}).items()}
        overlay = draw_instance_overlay(
            image_bgr, masks, confs, class_ids=cids, class_names=names,
            color_mode="instance")
        import os
        tmp = out_path.with_name(out_path.stem + ".tmp.png")
        if not cv2.imwrite(str(tmp), overlay):
            raise HTTPException(status_code=500, detail="failed to write overlay")
        os.replace(tmp, out_path)
    return out_path


def _resolve_patch_size(contract: dict, run_path: Path) -> int | None:
    """Patch size this run must be inferred at, or None for the single pass.

    Mirrors the semantic side's _should_use_sliding_window / RESIZE INFERENCE
    warning. A contract carrying patch_size means the model was trained on
    patch-sized composites at native scale, so inference has to tile at the
    same size -- a whole resized frame would show it objects several times
    smaller than anything it saw.

    A run without it predates tiling, and falls back to the resize path with a
    loud warning rather than silently. The silence is the danger: the model
    runs either way, and only the count is wrong.
    """
    raw = contract.get("patch_size")
    try:
        size = int(raw) if raw else 0
    except (TypeError, ValueError):
        size = 0
    if size > 0:
        return size
    logger.warning(
        "RESIZE INFERENCE: instance run %s has no patch_size in its contract; "
        "the full image is resized to the model input, which shrinks small "
        "objects (a 110px object in a 2560px frame reaches the model at 18px). "
        "Retrain with instance_patch_size set for native-resolution tiled "
        "inference.",
        run_path.name,
    )
    return None


class _TiledDetections:
    """The subset of the SDK's detection object the caller downstream uses."""

    def __init__(self, mask, confidence, class_id):
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id


def _predict_over_patches(model, image, patch_size: int, threshold: float, dedup_iou: float):
    """Run the detector over overlapping patches and merge, in full-frame space.

    Patch-trained models saw objects at capture resolution; a whole resized
    frame would show them several times smaller. Each patch is cropped, never
    resized, and its masks are pasted back at their true position so everything
    downstream (RLE, overlay, centroid, count) keeps working unchanged.

    All of that lives in segcore.instseg.tiled, which the training-time
    threshold calibration also calls. This function walked the tiles itself
    once, and never dropped views clipped by a tile edge: on a 2560x2048 photo
    of 40 screws it returned 75 where the shared implementation returned 40.
    """
    import numpy as np

    from segcore.instseg.tiled import predict_tiled_masks, sdk_tile_predict

    masks, confs, classes, _plan = predict_tiled_masks(
        image, sdk_tile_predict(model, threshold), patch_size,
        iou_threshold=dedup_iou)
    return _TiledDetections(
        mask=np.asarray(masks) if masks else None,
        confidence=np.asarray(confs),
        class_id=np.asarray(classes),
    )


def ensure_instance_artifacts(
    project_id: str, run_path: Path, item_id: str, *, force: bool = False,
) -> tuple[Path, Path, Path, Path, dict]:
    """Run inference (unless cached) and return artifact paths + score dict."""
    json_path, overlay_path, mask_path, conf_path, score_path = instance_artifact_paths(
        run_path, item_id, ensure_dir=True)
    with _item_lock(run_path, item_id):
        if not force and all(p.exists() for p in
                             (json_path, overlay_path, mask_path, conf_path, score_path)):
            score = json.loads(score_path.read_text(encoding="utf-8"))
            return json_path, overlay_path, mask_path, conf_path, score

        contract = instance_contract(run_path)
        if contract is None:
            raise HTTPException(status_code=404, detail="not an instance run")
        image_path = find_annotate_image(project_id, item_id)
        if image_path is None:
            raise HTTPException(status_code=404, detail="image not found")

        import cv2
        from PIL import Image

        from segcore.instseg.count import dedup_masks_by_class
        from segcore.instseg.overlay import draw_instance_overlay
        from segcore.instseg.rle import encode_rle

        threshold = float(contract.get("threshold", 0.3))
        dedup_iou = float(contract.get("dedup_iou", 0.7))
        checkpoint = _resolve_checkpoint(run_path, contract)

        t0 = time.perf_counter()
        patch_size = _resolve_patch_size(contract, run_path)
        with _predict_lock:  # rfdetr predict is not thread-safe; one GPU pass at a time
            model = _get_model(str(contract.get("model_size", "nano")), checkpoint)
            image = Image.open(image_path).convert("RGB")
            if patch_size:
                det = _predict_over_patches(model, image, patch_size, threshold, dedup_iou)
            else:
                det = model.predict(image, threshold=threshold)
        masks = list(det.mask) if det.mask is not None else []
        confs = [float(c) for c in det.confidence] if masks else []
        # The SDK reports the model's own class index, which is 0-based,
        # while the composed dataset numbers COCO categories from 1
        # (verified on the dev box 2026-07-24: a single-category checkpoint
        # returns class_id 0 for category 1). Shift, then map back to the
        # project's semantic class ids through the exported mapping; an
        # unmapped category falls back to the run's primary class rather
        # than to 0, which would silently read as background.
        semantic_of = _semantic_class_of(contract, run_path)
        primary_class = _resolve_class_id(run_path)
        raw_cats = ([int(c) + 1 for c in det.class_id]
                    if masks and getattr(det, "class_id", None) is not None
                    else [1] * len(masks))
        cats = [semantic_of.get(c) or primary_class for c in raw_cats]
        # Dedup within a class: an overlap across classes is a genuine
        # ambiguity, not the duplicate-mask artifact.
        kept = dedup_masks_by_class(masks, confs, cats, dedup_iou) if masks else []
        # Number instances by descending confidence: badge #1 = most confident.
        kept = sorted(kept, key=lambda i: -confs[i])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=500, detail="failed to read image")
        h, w = image_bgr.shape[:2]

        kept_masks = [(np.asarray(masks[i]) != 0).astype(np.uint8) for i in kept]
        kept_confs = [confs[i] for i in kept]
        kept_classes = [cats[i] for i in kept]

        names = _class_names(contract, run_path)
        class_id = kept_classes[0] if kept_classes else _resolve_class_id(run_path)
        class_name = names.get(class_id) or _resolve_class_name(run_path, class_id)

        instances = []
        for n, (m, c, cid) in enumerate(zip(kept_masks, kept_confs, kept_classes)):
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            instances.append({
                "id": n + 1,
                "class_id": cid,
                "class_name": names.get(cid, f"class{cid}"),
                "conf": round(c, 4),
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                # Area centroid (first moment of the mask), not the bbox
                # centre: for elongated or L-shaped objects the two differ.
                "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                "area": int(m.sum()),
                "rle": encode_rle(m),
            })
        counts_by_class = {}
        for inst in instances:
            counts_by_class[inst["class_id"]] = counts_by_class.get(inst["class_id"], 0) + 1
        payload = {
            "instances": instances,
            "count": len(instances),
            "counts_by_class": {str(k): v for k, v in sorted(counts_by_class.items())},
            "class_names": {str(k): names.get(k, f"class{k}")
                            for k in sorted(set(kept_classes) | set(names))},
            "class_id": class_id,
            "class_name": class_name,
            "threshold": threshold,
            "dedup_iou": dedup_iou,
        }

        # Default overlay keeps the original image as background (no wash):
        # class-coloured masks over the real pixels. The blue/grey background
        # wash lives only in the "detection highlight" toggle, rendered
        # separately in ensure_instance_highlight_overlay.
        overlay = draw_instance_overlay(image_bgr, kept_masks, kept_confs,
                                        class_ids=kept_classes, class_names=names,
                                        style="tint")
        legacy = np.zeros((h, w), dtype=np.uint8)
        confidence = np.zeros((h, w), dtype=np.uint8)
        # Ascending confidence so overlapping pixels keep the higher value.
        # The legacy composite is a semantic mask, so each instance paints
        # its own class id — multi-class runs stay readable in the old viewers.
        for m, c, cid in sorted(zip(kept_masks, kept_confs, kept_classes),
                                key=lambda mc: mc[1]):
            legacy[m.astype(bool)] = cid
            confidence[m.astype(bool)] = int(round(min(max(c, 0.0), 1.0) * 255))

        mean_conf = float(np.mean(kept_confs)) if kept_confs else 0.0
        score = {
            "backend": "rfdetr",
            "item_id": item_id,
            "instance_count": len(instances),
            "threshold": threshold,
            "mean_confidence": mean_conf,
            "foreground_mean_confidence": mean_conf,
            "background_mean_confidence": 0.0,
            "foreground_ratio": float((legacy != 0).sum()) / float(h * w),
            "max_confidence": float(max(kept_confs)) if kept_confs else 0.0,
            "min_confidence": float(min(kept_confs)) if kept_confs else 0.0,
            "per_class_mean_confidence": {str(class_id): mean_conf} if instances else {},
            "inference_ms": round(inference_ms, 1),
        }

        # Stage every artifact next to its final path, then commit with
        # os.replace so a crash mid-write never leaves a partial file that
        # the cache-hit check above would treat as complete. instances.json
        # commits last as the completion marker.
        import os

        staged: list[tuple[Path, Path]] = []

        def _stage(final: Path, suffix: str) -> Path:
            tmp = final.with_name(final.stem + ".tmp" + suffix)
            staged.append((tmp, final))
            return tmp

        try:
            if not cv2.imwrite(str(_stage(overlay_path, ".png")), overlay):
                raise HTTPException(status_code=500, detail="failed to write overlay artifact")
            Image.fromarray(legacy, mode="L").save(_stage(mask_path, ".png"))
            Image.fromarray(confidence, mode="L").save(_stage(conf_path, ".png"))
            _stage(score_path, ".json").write_text(json.dumps(score), encoding="utf-8")
            _stage(json_path, ".json").write_text(json.dumps(payload), encoding="utf-8")
            for tmp, final in staged:
                os.replace(tmp, final)
        except BaseException:
            for tmp, _final in staged:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return json_path, overlay_path, mask_path, conf_path, score


def instance_batch_stream(
    project_id: str, run_path: Path, item_ids: list[str], *, force: bool = False,
):
    """NDJSON generator mirroring the semantic predict_batch_stream contract."""
    for item_id in item_ids:
        t0 = time.perf_counter()
        try:
            *_paths, score = ensure_instance_artifacts(
                project_id, run_path, item_id, force=force)
            yield json.dumps({
                "item_id": item_id,
                "status": "ok",
                "score": score,
                "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001 — per-item errors must not kill the stream
            detail = getattr(exc, "detail", None) or str(exc)
            yield json.dumps({
                "item_id": item_id,
                "status": "error",
                "detail": str(detail),
                "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }, ensure_ascii=False) + "\n"


# ── ONNX export (v0.9.8 M4) ─────────────────────────────────────


def export_instance_onnx(run_path: Path, contract: dict, *, force: bool = False) -> Path:
    """Export the run's rfdetr checkpoint to ONNX (cached under the run dir).

    Uses the rfdetr SDK exporter (R1 spike contract: opset 17, input
    1x3x312x312 stretch-normalized, outputs dets/labels/masks). fp32 only —
    fp16/int8 are out of scope for v0.9.8.
    """
    export_dir = run_path / "export_onnx"
    if not force:
        existing = sorted(export_dir.rglob("*.onnx")) if export_dir.exists() else []
        if existing:
            return existing[-1]
    checkpoint = _resolve_checkpoint(run_path, contract)
    from segcore.instseg.train_rfdetr import build_model

    model_size = str(contract.get("model_size", "nano"))
    export_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(model_size, pretrain_weights=str(checkpoint))
    model.export(output_dir=str(export_dir), format="onnx", verbose=False)
    exported = sorted(export_dir.rglob("*.onnx"))
    if not exported:
        raise HTTPException(status_code=500, detail="rfdetr export produced no onnx file")
    return exported[-1]


def read_onnx_input_size(onnx_path: Path) -> tuple[int, int]:
    """(width, height) of the graph's image input, via a CPU ORT session."""
    import onnxruntime as ort_rt

    sess = ort_rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    try:
        return int(shape[3]), int(shape[2])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"exported onnx has dynamic input dims: {shape}") from exc


# ── Synthesis preview (training form) ───────────────────────────



def compose_preview_samples(project_id: str, params: dict) -> dict:
    """Compose a few synthetic samples in memory for the training form.

    Uses the same ComposeConfig defaults/plumbing as the real run so what
    the user previews is what training will compose.
    """
    import base64

    import cv2

    from segcore.instseg.compose import ComposeConfig, _Composer, collect_material
    from segcore.instseg.overlay import draw_instance_overlay

    from .instance_training import (
        _MIN_SOURCE_IMAGES,
        _load_sources,
        class_name_map,
        resolve_class_ids,
    )

    project_root = project_dir(project_id)
    class_ids = resolve_class_ids(project_root, params)
    preview_class_names = class_name_map(project_root)
    band_min = params.get("instance_area_band_min")
    band_max = params.get("instance_area_band_max")
    cfg = ComposeConfig(
        objects_min=int(params.get("instance_objects_min", 4)),
        objects_max=int(params.get("instance_objects_max", 8)),
        stack_pair_prob=float(params.get("instance_stack_pair_prob", 0.55)),
        seed=int(params.get("instance_seed", 42)),
        area_band=(int(band_min), int(band_max)) if band_min and band_max else None,
    )
    sources = _load_sources(project_id, class_ids, lambda _msg: None)
    if len(sources) < _MIN_SOURCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"instance mode needs at least {_MIN_SOURCE_IMAGES} annotated images "
                   f"containing one of classes {class_ids}, found {len(sources)}")
    try:
        material = collect_material(sources, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not material.cutouts:
        raise HTTPException(
            status_code=400,
            detail="no cutouts in the single-object area band; "
                   "check masks or override the area band")
    if not material.bg_plates:
        raise HTTPException(
            status_code=400,
            detail="no background plates could be built from the sources "
                   "(more annotated images needed)")

    composer = _Composer(material, cfg)
    samples = []
    for _ in range(int(params.get("n_samples", 3))):
        canvas, keep = composer.synth_image()
        keep_masks = [m for m, _ in keep]
        keep_classes = [c for _, c in keep]
        overlay = draw_instance_overlay(canvas, keep_masks, None, alpha=0.35,
                                        class_ids=keep_classes,
                                        class_names=preview_class_names)
        h, w = overlay.shape[:2]
        scale = 512.0 / max(h, w)
        if scale < 1.0:
            overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(status_code=500, detail="preview encode failed")
        samples.append({
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
            "n_instances": len(keep),
        })
    return {
        "samples": samples,
        "class_ids": class_ids,
        "class_names": {str(k): v for k, v in preview_class_names.items()},
        "class_id": class_ids[0],
        "n_sources": len(sources),
        "n_cutouts": len(material.cutouts),
        "n_bg_plates": len(material.bg_plates),
        "area_band": list(material.area_band),
    }
