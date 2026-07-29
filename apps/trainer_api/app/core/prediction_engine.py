# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import HTTPException

# Per-item lock to prevent concurrent GPU inference for the same image.
# Bounded LRU so lock objects don't accumulate indefinitely.
from .cache_utils import ThreadSafeLRUCache

_inference_locks: ThreadSafeLRUCache = ThreadSafeLRUCache(maxsize=256)

from segcore.training.prediction_rules import prediction_from_probs
from segcore.training.sliding_window import normalize_logits_batch, sliding_window_predict_infer_fn

from .annotate_index import find_annotate_image
from .classes import find_coreml_model_path, resolve_active_class_ids
from .config import NORMALIZE
from .coreml_backend import load_coreml_model
from .exceptions import AppError, PredictModelMissingError
from .paths import classes_path, project_dir, resolve_run_path
from .run_config import (
    _load_run_arch,
    _load_run_base_channels,
    _load_run_inference_threshold,
    _load_run_input_size,
    _load_run_num_classes,
    _load_run_output_stride,
    _load_run_patch_size,
    _load_run_sw_stride,
    _load_run_train_size,
)
from .security import _sanitize_filename

# ---------------------------------------------------------------------------
# Named constants — no magic numbers below
# ---------------------------------------------------------------------------
# Timeout (seconds) when joining a background image-preload thread.
_PRELOAD_JOIN_TIMEOUT_S = 10.0


# ONNX export / ORT session / model-cache infrastructure was extracted
# verbatim to ort_infra. Re-imported here so in-module references, router
# imports, and app.main.__getattr__ lookups keep working; the caches stay
# process-wide singletons through this import.
from .inference_math import (  # noqa: F401 — backward-compat re-exports
    _SUPPRESS_LOGIT,
    _ort_run_logits,
    _resize_probs_chw_np,
    _should_use_sliding_window,
    _softmax_np,
    _tta_predict_ort,
    _tta_predict_torch,
)
from .ort_infra import (  # noqa: F401 — backward-compat re-exports
    _MODEL_CACHE_SIZE,
    _auto_model_cache_size,
    _build_ort_session_options,
    _cuda_provider_options,
    _ensure_onnx_model,
    _export_onnx_model,
    _load_ort_session,
    _load_torch_model,
    _ort_session_cache,
    _ort_session_cache_guard,
    _resolve_ort_device,
    _torch_model_cache,
    _torch_model_cache_guard,
    clear_torch_model_cache,
)

PREDICTION_ARTIFACT_VERSION = 4  # v4: sliding-window inference artifacts


class _SkipOrt(Exception):
    """Raised to leave the ORT branch without having run anything."""


def _predict_with_ort(
    run_path: Path,
    model_path: Path,
    requested_device_id: str,
    *,
    img,
    img_np: np.ndarray,
    infer_w: int,
    infer_h: int,
    num_classes: int,
    run_output_stride: int,
    run_base_channels: int,
    run_arch: str,
    patch_size: int,
    sw_stride: int,
    use_sw: bool,
    tta: bool,
    inactive_class_ids: list[int],
    active_class_ids: list[int],
    _preloaded_session=None,
) -> tuple[np.ndarray, float, str]:
    def run_once(session: object, input_name: str, output_name: str) -> tuple[np.ndarray, float]:
        _t_infer_start = time.perf_counter()
        if use_sw:
            if tta:
                def ort_tta_fn(batch_np: np.ndarray) -> np.ndarray:
                    return _tta_predict_ort(
                        session,
                        input_name,
                        output_name,
                        batch_np,
                        num_classes=num_classes,
                        inactive_class_ids=inactive_class_ids,
                    )

                _pred_argmax, probs = sliding_window_predict_infer_fn(
                    ort_tta_fn,
                    img_np,
                    patch_size,
                    sw_stride,
                    num_classes,
                    run_output_stride,
                    NORMALIZE,
                    active_class_ids=active_class_ids,
                    infer_fn_returns_probs=True,
                )
            else:
                def ort_infer_fn(batch_np: np.ndarray) -> np.ndarray:
                    return _ort_run_logits(session, input_name, output_name, batch_np, num_classes)

                _pred_argmax, probs = sliding_window_predict_infer_fn(
                    ort_infer_fn,
                    img_np,
                    patch_size,
                    sw_stride,
                    num_classes,
                    run_output_stride,
                    NORMALIZE,
                    active_class_ids=active_class_ids,
                )
        else:
            from PIL import Image

            resized = img.resize((infer_w, infer_h), resample=Image.BILINEAR)
            arr = np.asarray(resized).astype("float32") / 255.0
            mean = np.array(NORMALIZE["mean"], dtype="float32")
            std = np.array(NORMALIZE["std"], dtype="float32")
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2, 0, 1))[None, ...].astype("float32", copy=False)
            if tta:
                probs = _tta_predict_ort(
                    session,
                    input_name,
                    output_name,
                    arr,
                    num_classes=num_classes,
                    inactive_class_ids=inactive_class_ids,
                )[0].astype("float32", copy=False)
            else:
                logits = _ort_run_logits(session, input_name, output_name, arr, num_classes)
                if inactive_class_ids:
                    logits[:, inactive_class_ids, :, :] = _SUPPRESS_LOGIT
                probs = _softmax_np(logits, axis=1)[0].astype("float32", copy=False)
            probs = _resize_probs_chw_np(probs, infer_w, infer_h)
        return probs.astype("float32", copy=False), (time.perf_counter() - _t_infer_start) * 1000.0

    import logging as _log_ort
    _ort_logger = _log_ort.getLogger(__name__)
    if _preloaded_session is not None:
        session, input_name, output_name, provider_name = _preloaded_session
        _ort_logger.debug("Using preloaded ORT session (provider=%s)", provider_name)
    else:
        _t_load = time.perf_counter()
        session, input_name, output_name, provider_name = _load_ort_session(
            run_path,
            model_path,
            requested_device_id,
            num_classes=num_classes,
            run_output_stride=run_output_stride,
            run_base_channels=run_base_channels,
            run_arch=run_arch,
        )
        _ort_logger.info(
            "ORT session loaded in %.0fms (provider=%s, session_providers=%s)",
            (time.perf_counter() - _t_load) * 1000, provider_name, session.get_providers(),
        )
    try:
        probs_np, infer_ms = run_once(session, input_name, output_name)
        _ort_logger.info("ORT inference %.0fms (provider=%s)", infer_ms, provider_name)
        return probs_np, infer_ms, provider_name
    except Exception as e:
        if provider_name == "cpu":
            raise
        _ort_logger.warning(
            "ORT inference failed on provider %s — falling back to CPU: %s", provider_name, e,
        )
        session, input_name, output_name, provider_name = _load_ort_session(
            run_path,
            model_path,
            "cpu",
            num_classes=num_classes,
            run_output_stride=run_output_stride,
            run_base_channels=run_base_channels,
            run_arch=run_arch,
        )
        probs_np, infer_ms = run_once(session, input_name, output_name)
        return probs_np, infer_ms, provider_name


def _prediction_artifact_paths(run_path: Path, backend: str, item_id: str, tta: bool = False, *, ensure_dir: bool = False) -> tuple[Path, Path, Path]:
    item_id = _sanitize_filename(item_id)  # neutralize ..\ / ../ path traversal in the route param
    suffix = "_tta" if tta else ""
    pred_dir = run_path / (("predictions_coreml" if backend == "coreml" else "predictions") + suffix)
    if ensure_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{item_id}.png"
    confidence_path = pred_dir / f"{item_id}.confidence.png"
    score_path = pred_dir / f"{item_id}.score.json"
    return pred_path, confidence_path, score_path


def _resolve_predict_context(project_id: str, run_id: str, backend: str) -> tuple[Path, Path, str]:
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    run_path = resolve_run_path(project_id, run_id)
    if run_path is None:
        raise HTTPException(status_code=404, detail="run not found")
    backend = backend.lower().strip()
    if backend not in {"onnx", "coreml"}:
        raise HTTPException(status_code=400, detail="backend must be 'onnx' or 'coreml'")
    model_path = run_path / "model.pt"
    if backend == "onnx":
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="model checkpoint not found")
    else:
        if not find_coreml_model_path(run_path) and not model_path.exists():
            raise HTTPException(status_code=404, detail="model checkpoint not found for Core ML export")
    return run_path, model_path, backend


def _ensure_prediction_artifacts(project_id: str, run_path: Path, model_path: Path, item_id: str, backend: str, tta: bool = False, force: bool = False, _preloaded_session=None) -> tuple[Path, Path, dict]:
    item_id = _sanitize_filename(item_id)  # neutralize ..\ / ../ path traversal before it reaches probs/artifact paths
    # Acquire per-item lock to prevent duplicate GPU inference for the same image
    lock_key = (str(run_path), item_id, backend, tta)
    existing = _inference_locks.get(lock_key)
    if existing is None:
        new_lock = threading.Lock()
        _inference_locks.put(lock_key, new_lock)
        lock = new_lock
    else:
        lock = existing
    with lock:
        return _ensure_prediction_artifacts_inner(project_id, run_path, model_path, item_id, backend, tta=tta, force=force, _preloaded_session=_preloaded_session)


def _ensure_prediction_artifacts_inner(project_id: str, run_path: Path, model_path: Path, item_id: str, backend: str, tta: bool = False, force: bool = False, _preloaded_session=None) -> tuple[Path, Path, dict]:
    if not model_path.exists():
        raise PredictModelMissingError(detail=f"path={model_path}")
    image_path = find_annotate_image(project_id, item_id)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    if tta and backend == "coreml":
        raise HTTPException(status_code=400, detail="TTA is not supported with CoreML backend. Use ONNX or torch backend for TTA.")
    classes_file = run_path / "classes.json"
    if classes_file.exists():
        classes = json.loads(classes_file.read_text(encoding="utf-8"))
    else:
        classes = json.loads(classes_path(project_id).read_text(encoding="utf-8"))
    class_ids = [int(item.get("id", 0)) for item in classes.get("classes", [])]
    if not class_ids:
        raise HTTPException(status_code=400, detail="classes.json has no classes defined")
    num_classes = _load_run_num_classes(run_path)
    # Prefer active_class_ids from train_config.json (may differ from classes.json,
    # e.g. quick learning bumps num_classes for mask IDs not in classes.json)
    active_class_ids = None
    _tc_path = run_path / "train_config.json"
    if _tc_path.exists():
        try:
            _tc = json.loads(_tc_path.read_text(encoding="utf-8"))
            _tc_active = _tc.get("active_class_ids")
            if isinstance(_tc_active, list) and _tc_active:
                active_class_ids = [int(x) for x in _tc_active]
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    if active_class_ids is None:
        active_class_ids = resolve_active_class_ids(classes)
    inactive_class_ids = [cls_id for cls_id in range(num_classes) if cls_id not in active_class_ids]
    infer_w, infer_h = _load_run_input_size(run_path)
    run_output_stride = _load_run_output_stride(run_path)
    patch_size = _load_run_patch_size(run_path)
    sw_stride = _load_run_sw_stride(run_path)
    use_sw = _should_use_sliding_window(patch_size, sw_stride, run_output_stride)
    if not use_sw:
        import logging as _log_sw
        _log_sw.getLogger(__name__).warning(
            "RESIZE INFERENCE: run %s has no sliding-window params "
            "(patch_size=%s, sw_stride=%s); the full image is resized to "
            "input_size, which degrades small-defect detail. Retrain with "
            "patch training for native-resolution SW inference.",
            run_path.name, patch_size, sw_stride,
        )
    inference_threshold = _load_run_inference_threshold(run_path)

    pred_path, confidence_path, score_path = _prediction_artifact_paths(run_path, backend, item_id, tta=tta, ensure_dir=True)
    if force:
        pred_path.unlink(missing_ok=True)
        confidence_path.unlink(missing_ok=True)
        score_path.unlink(missing_ok=True)
        probs_path_npz = pred_path.parent / f"{item_id}.probs.npz"
        probs_path_npy = pred_path.parent / f"{item_id}.probs.npy"
        probs_path_npz.unlink(missing_ok=True)
        probs_path_npy.unlink(missing_ok=True)
    if pred_path.exists() and confidence_path.exists() and score_path.exists():
        invalidate_cache = False
        score: dict = {}
        try:
            score = json.loads(score_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            invalidate_cache = True
        if not invalidate_cache:
            version = int(score.get("artifact_version", 0))
            score_size = score.get("inference_input_size")
            if version != PREDICTION_ARTIFACT_VERSION or score_size != [infer_w, infer_h]:
                invalidate_cache = True
        if inactive_class_ids:
            try:
                from PIL import Image
                with Image.open(pred_path) as pred_img_check:
                    pred_np_check = np.asarray(pred_img_check.convert("L"))
                present = set(np.unique(pred_np_check).tolist())
                if any(cls_id in present for cls_id in inactive_class_ids):
                    invalidate_cache = True
            except (OSError, ValueError):
                invalidate_cache = True
        # Also invalidate if probs file is missing (needed for heatmaps)
        # Support both old .npz and new .npy formats
        probs_path_npy = pred_path.parent / f"{item_id}.probs.npy"
        probs_path_npz = pred_path.parent / f"{item_id}.probs.npz"
        if not probs_path_npy.exists() and not probs_path_npz.exists():
            invalidate_cache = True
        if invalidate_cache:
            pred_path.unlink(missing_ok=True)
            confidence_path.unlink(missing_ok=True)
            score_path.unlink(missing_ok=True)
            probs_path_npy.unlink(missing_ok=True)
            probs_path_npz.unlink(missing_ok=True)
        if pred_path.exists() and confidence_path.exists() and score_path.exists():
            return pred_path, confidence_path, score

    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Pre-resize for resized models (camera-independent: uses absolute train_size)
        train_size = _load_run_train_size(run_path)
        if train_size is not None:
            img = img.resize((train_size[0], train_size[1]), Image.LANCZOS)

        img_np = np.asarray(img)
        probs_np: np.ndarray
        pred_small: np.ndarray
        confidence_small: np.ndarray
        predict_device_id = "coreml"

        if backend == "onnx":
            from .torch_device import current_configured_torch_device, resolve_torch_device_or_cpu
            run_base_channels = _load_run_base_channels(run_path)
            run_arch = _load_run_arch(run_path)
            # Use smart scheduling: pick a free GPU instead of the static global setting
            raw_device = current_configured_torch_device()
            requested_predict_device = resolve_torch_device_or_cpu(raw_device)

            # ORT path: use ORT when it actually runs on GPU, otherwise prefer
            # torch GPU inference (ORT CPU is 50-100x slower than torch GPU).
            #
            # Decide before spending, not after. This used to run the whole CPU
            # inference and only then look at the provider it came back with, so
            # a machine whose CUDA EP fails to load paid for every prediction
            # twice -- the slow way first, then again on torch. Loading the
            # session is what reveals the provider, and sessions are cached, so
            # asking first costs nothing. The batch path already worked this way
            # (it drops the session and lets each item fall through to torch);
            # this is the same decision, one image at a time.
            _ort_session = _preloaded_session
            _ort_available = True
            if _ort_session is None:
                try:
                    _sess, _in_name, _out_name, _provider = _load_ort_session(
                        run_path,
                        model_path,
                        requested_predict_device,
                        num_classes=num_classes,
                        run_output_stride=run_output_stride,
                        run_base_channels=run_base_channels,
                        run_arch=run_arch,
                    )
                except Exception:
                    import logging as _log
                    _log.getLogger(__name__).exception(
                        "ORT session load failed; falling back to torch eager")
                    _ort_available = False
                else:
                    if requested_predict_device.startswith("cuda") and _provider == "cpu":
                        import logging as _log
                        _log.getLogger(__name__).info(
                            "ORT fell back to CPU despite CUDA request; "
                            "using torch GPU instead (no CPU pass run)"
                        )
                        _ort_available = False
                    else:
                        _ort_session = (_sess, _in_name, _out_name, _provider)

            _ort_succeeded = False
            try:
                if not _ort_available:
                    raise _SkipOrt
                probs_np, _t_infer_ms, ort_provider_name = _predict_with_ort(
                    run_path,
                    model_path,
                    requested_predict_device,
                    img=img,
                    img_np=img_np,
                    infer_w=infer_w,
                    infer_h=infer_h,
                    num_classes=num_classes,
                    run_output_stride=run_output_stride,
                    run_base_channels=run_base_channels,
                    run_arch=run_arch,
                    patch_size=patch_size,
                    sw_stride=sw_stride,
                    use_sw=use_sw,
                    tta=tta,
                    inactive_class_ids=inactive_class_ids,
                    active_class_ids=active_class_ids,
                    _preloaded_session=_ort_session,
                )
                # A session that started on CUDA can still fall back mid-flight
                # (_predict_with_ort retries on CPU when a run throws). That
                # work is already paid for, so keep the result rather than
                # repeating it on torch.
                predict_device_id = f"ort:{ort_provider_name}"
                _ort_succeeded = True
            except _SkipOrt:
                pass
            except Exception:
                import logging as _log
                _log.getLogger(__name__).exception("ORT inference failed; falling back to torch eager")

            if not _ort_succeeded:
                # Torch fallback: only import torch here, after ORT has failed
                import torch
                import torch.nn.functional as F

                from .torch_device import acquired_torch_device

                with acquired_torch_device(
                    requested_predict_device,
                    owner_kind="inference",
                    owner_id=f"infer:{run_path.name}:{item_id}:{uuid.uuid4().hex}",
                    project_id=project_id,
                ) as predict_device_id:
                    predict_device = torch.device(predict_device_id)
                    model = _load_torch_model(
                        run_path,
                        model_path,
                        predict_device_id,
                        num_classes=num_classes,
                        run_output_stride=run_output_stride,
                        run_base_channels=run_base_channels,
                        run_arch=run_arch,
                    )
                    _t_infer_start = time.perf_counter()
                    if use_sw:
                        if tta:
                            def torch_tta_fn(batch_np: np.ndarray) -> np.ndarray:
                                batch_t = torch.from_numpy(batch_np).to(predict_device)
                                probs_tta = _tta_predict_torch(model, batch_t, inactive_class_ids, predict_device)
                                return probs_tta.cpu().numpy()

                            _pred_argmax, probs_np = sliding_window_predict_infer_fn(
                                torch_tta_fn, img_np, patch_size, sw_stride,
                                num_classes, run_output_stride, NORMALIZE,
                                active_class_ids=active_class_ids,
                                infer_fn_returns_probs=True,
                            )
                        else:
                            def torch_infer_fn(batch_np: np.ndarray) -> np.ndarray:
                                with torch.inference_mode():
                                    batch_t = torch.from_numpy(batch_np).to(predict_device)
                                    logits = model(batch_t)
                                    return logits.detach().cpu().numpy()

                            _pred_argmax, probs_np = sliding_window_predict_infer_fn(
                                torch_infer_fn, img_np, patch_size, sw_stride,
                                num_classes, run_output_stride, NORMALIZE,
                                active_class_ids=active_class_ids,
                            )
                    else:
                        resized = img.resize((infer_w, infer_h), resample=Image.BILINEAR)
                        arr = np.asarray(resized).astype("float32") / 255.0
                        mean = np.array(NORMALIZE["mean"], dtype="float32")
                        std = np.array(NORMALIZE["std"], dtype="float32")
                        arr = (arr - mean) / std
                        arr = np.transpose(arr, (2, 0, 1))
                        input_t = torch.tensor(arr[None, ...], device=predict_device)
                        with torch.inference_mode():
                            if tta:
                                probs = _tta_predict_torch(model, input_t, inactive_class_ids, predict_device)
                            else:
                                logits = model(input_t)
                                if inactive_class_ids:
                                    logits[:, inactive_class_ids, :, :] = _SUPPRESS_LOGIT
                                probs = torch.softmax(logits, dim=1)
                            probs_up = F.interpolate(probs, size=(infer_h, infer_w), mode="bilinear", align_corners=False)
                            probs_np = probs_up[0].cpu().numpy().astype("float32")
                    if predict_device.type == "cuda":
                        torch.cuda.synchronize(predict_device)
                    _t_infer_ms = (time.perf_counter() - _t_infer_start) * 1000.0
                    predict_device_id = f"torch:{predict_device_id}"
            pred_small = prediction_from_probs(probs_np, fg_threshold=inference_threshold).astype("uint8")
            confidence_small = probs_np[1:, :, :].sum(axis=0).astype("float32")
        else:
            mlmodel = load_coreml_model(run_path)
            def coreml_infer_fn(batch_np: np.ndarray) -> np.ndarray:
                outputs: list[np.ndarray] = []
                for idx in range(batch_np.shape[0]):
                    try:
                        prediction = mlmodel.predict({"input": batch_np[idx:idx + 1]}, useCPUOnly=False)
                    except Exception:
                        import logging as _log
                        _log.getLogger(__name__).exception("CoreML predict failed")
                        raise HTTPException(status_code=500, detail="coreml predict failed")
                    logits = np.asarray(next(iter(prediction.values())))
                    outputs.append(normalize_logits_batch(logits, num_classes))
                return np.concatenate(outputs, axis=0)

            _t_infer_start = time.perf_counter()
            if use_sw:
                _pred_argmax, probs_np = sliding_window_predict_infer_fn(
                    coreml_infer_fn,
                    img_np,
                    patch_size,
                    sw_stride,
                    num_classes,
                    run_output_stride,
                    NORMALIZE,
                    active_class_ids=active_class_ids,
                )
            else:
                resized = img.resize((infer_w, infer_h), resample=Image.BILINEAR)
                arr = np.asarray(resized).astype("float32") / 255.0
                mean = np.array(NORMALIZE["mean"], dtype="float32")
                std = np.array(NORMALIZE["std"], dtype="float32")
                arr = (arr - mean) / std
                arr = np.transpose(arr, (2, 0, 1))
                logits = coreml_infer_fn(arr[None, ...])
                if inactive_class_ids:
                    logits = logits.copy()
                    logits[:, inactive_class_ids, :, :] = _SUPPRESS_LOGIT
                probs_np = _softmax_np(logits, axis=1)[0].astype("float32")
            _t_infer_ms = (time.perf_counter() - _t_infer_start) * 1000.0
            pred_small = prediction_from_probs(probs_np, fg_threshold=inference_threshold).astype("uint8")
            confidence_small = probs_np[1:, :, :].sum(axis=0).astype("float32")

        import cv2
        _t_post_start = time.perf_counter()

        # Resize to original resolution
        pred_full = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        confidence_full = cv2.resize(confidence_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        confidence_full = np.clip(confidence_full, 0.0, 1.0)
        confidence_u8 = np.clip(confidence_full * 255.0, 0, 255).astype("uint8")

        # Save PNGs with cv2.imencode (C++ encoder, much faster than PIL)
        _png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        ok, pred_buf = cv2.imencode('.png', pred_full, _png_params)
        pred_path.write_bytes(pred_buf.tobytes())
        ok, conf_buf = cv2.imencode('.png', confidence_u8, _png_params)
        confidence_path.write_bytes(conf_buf.tobytes())

        # Score computation from numpy arrays directly (no PIL round-trip)
        fg_mask = pred_full > 0
        per_class_mean_conf: dict[str, float] = {}
        for class_id in class_ids:
            class_mask = pred_full == class_id
            if class_mask.any():
                per_class_mean_conf[str(class_id)] = float(np.mean(confidence_full[class_mask]))
        score = {
            "artifact_version": PREDICTION_ARTIFACT_VERSION,
            "backend": backend,
            "item_id": item_id,
            "inference_input_size": [infer_w, infer_h],
            "inference_ms": round(_t_infer_ms, 1),
            "inference_device": predict_device_id,
            "mean_confidence": float(np.mean(confidence_full)),
            "foreground_mean_confidence": float(np.mean(confidence_full[fg_mask])) if fg_mask.any() else 0.0,
            "background_mean_confidence": float(np.mean(confidence_full[~fg_mask])) if (~fg_mask).any() else 0.0,
            "foreground_ratio": float(np.mean(fg_mask)),
            "max_confidence": float(np.max(confidence_full)),
            "min_confidence": float(np.min(confidence_full)),
            "per_class_mean_confidence": per_class_mean_conf,
        }
        # Save score JSON and probs in background (not on critical path)
        probs_save = probs_np.astype("float16")
        probs_path = pred_path.parent / f"{item_id}.probs.npy"
        _score_json = json.dumps(score, ensure_ascii=False, indent=2)
        def _save_deferred_bg():
            try:
                score_path.write_text(_score_json, encoding="utf-8")
                np.save(probs_path, probs_save)
            except Exception:
                import logging as _log
                _log.getLogger(__name__).warning("Deferred save failed for %s", item_id)
        threading.Thread(target=_save_deferred_bg, daemon=True).start()

        _t_post_ms = (time.perf_counter() - _t_post_start) * 1000.0
        import logging as _log_post
        _log_post.getLogger(__name__).info(
            "Post-processing %.0fms (resize+PNG+score+probs_bg) for %s", _t_post_ms, item_id)

        return pred_path, confidence_path, score
    except (HTTPException, AppError):
        raise
    except Exception as exc:
        from .exceptions import PredictError
        raise PredictError(
            detail=str(exc),
            context={"project_id": project_id, "item_id": item_id},
        ) from exc


def predict_batch_stream(
    project_id: str,
    run_path: Path,
    model_path: Path,
    item_ids: list[str],
    backend: str,
    tta: bool = False,
    force: bool = False,
):
    """Generator yielding NDJSON lines for batch prediction.

    Pre-loads the ORT session once for the entire batch so that per-image
    overhead (device resolution, nvidia-smi, cache lookup) is paid only once
    instead of on every image.

    Yields one JSON line per item: {"item_id": ..., "status": "ok"|"error", ...}
    """
    import logging as _log_batch
    _logger = _log_batch.getLogger(__name__)

    # --- Pre-warm ORT session once for the batch ---
    _batch_session = None
    _batch_input_name = ""
    _batch_output_name = ""
    _batch_provider = "cpu"
    if backend == "onnx":
        try:
            num_classes = _load_run_num_classes(run_path)
            run_output_stride = _load_run_output_stride(run_path)
            run_base_channels = _load_run_base_channels(run_path)
            run_arch = _load_run_arch(run_path)
            t_session = time.perf_counter()
            from .torch_device import current_configured_torch_device, resolve_torch_device_or_cpu
            raw_device = current_configured_torch_device()
            device_id = resolve_torch_device_or_cpu(raw_device)
            _batch_session, _batch_input_name, _batch_output_name, _batch_provider = _load_ort_session(
                run_path, model_path, device_id,
                num_classes=num_classes,
                run_output_stride=run_output_stride,
                run_base_channels=run_base_channels,
                run_arch=run_arch,
            )
            _logger.info("Batch ORT session loaded in %.0fms (provider=%s)", (time.perf_counter() - t_session) * 1000, _batch_provider)
            # If CUDA was requested but ORT fell back to CPU, discard session
            # so each item falls through to torch GPU in _ensure_prediction_artifacts_inner
            if device_id.startswith("cuda") and _batch_provider == "cpu":
                _logger.info("ORT fell back to CPU despite CUDA request; batch will use torch GPU")
                _batch_session = None
        except Exception:
            _logger.exception("Failed to pre-load ORT session for batch")
            _batch_session = None

    # Warm OS page cache for the next image while current one is being inferred.
    _preload_future: threading.Thread | None = None

    def _preload_image(next_iid: str):
        try:
            img_path = find_annotate_image(project_id, next_iid)
            if img_path and img_path.exists():
                _ = img_path.read_bytes()
        except Exception:
            pass

    # Preload the first image before the loop starts
    if item_ids:
        _preload_image(item_ids[0])

    for idx, iid in enumerate(item_ids):
        # Kick off preload of next image in background while this one is processed
        if idx + 1 < len(item_ids):
            _preload_future = threading.Thread(
                target=_preload_image, args=(item_ids[idx + 1],), daemon=True,
            )
            _preload_future.start()

        t0 = time.perf_counter()
        try:
            _pred_path, _conf_path, score = _ensure_prediction_artifacts(
                project_id, run_path, model_path, iid, backend, tta=tta, force=force,
                _preloaded_session=(_batch_session, _batch_input_name, _batch_output_name, _batch_provider) if _batch_session else None,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            yield json.dumps({
                "item_id": iid,
                "status": "ok",
                "score": score,
                "total_ms": round(elapsed_ms, 1),
            }, ensure_ascii=False) + "\n"
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            detail = str(exc)
            if hasattr(exc, "detail"):
                detail = exc.detail
            _logger.warning("Batch predict failed for %s: %s", iid, detail, exc_info=True)
            yield json.dumps({
                "item_id": iid,
                "status": "error",
                "detail": detail,
                "total_ms": round(elapsed_ms, 1),
            }, ensure_ascii=False) + "\n"

        # Wait for preload to finish before next iteration
        if _preload_future is not None:
            _preload_future.join(timeout=_PRELOAD_JOIN_TIMEOUT_S)
            _preload_future = None


# Heatmap generation was extracted verbatim to heatmaps (which lazy-imports
# _prediction_artifact_paths from here to avoid a module-level cycle).
from .heatmaps import _heatmap_cache_path, generate_heatmap  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Public aliases — routers and other callers should use the un-underscored
# names. The underscored variants remain as the canonical definitions so
# in-module references and ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
export_onnx_model = _export_onnx_model
prediction_artifact_paths = _prediction_artifact_paths
resolve_predict_context = _resolve_predict_context
ensure_prediction_artifacts = _ensure_prediction_artifacts

