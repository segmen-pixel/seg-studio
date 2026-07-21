# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Pipelined inference runtime with persistent GPU workers.

Architecture:
  Stage A  (PrepWorkers, CPU threads)  — image load, patch extract, normalize
  Stage B  (GpuWorkers, 1 per GPU)     — ORT inference, cross-image batching
  Stage C  (Collector, 1 thread)       — accumulate probs into SW accumulators
  Stage D  (PostWorkers, CPU threads)  — argmax, PNG save, score calculation

Supports two input modes:
  - batch:  process all images, save artifacts, yield NDJSON results
  - stream: latest-frame-wins, callback/WebSocket delivery (future)
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

# Worker/type definitions were extracted verbatim to inference_workers.
# Re-imported here so in-module references and any external users keep
# working; _SHUTDOWN identity is shared through this import.
from .inference_workers import (  # noqa: F401 — backward-compat re-exports
    _DEFAULT_CHUNK_SIZE,
    _MIN_BATCH_SIZE,
    _PROFILE_SAFETY_MARGIN,
    _SHUTDOWN,
    _SW_GAUSS_SIGMA_DIV,
    _auto_cpu_workers,
    _BatchTracker,
    _Chunk,
    _ChunkResult,
    _estimate_vram_mb,
    _GpuWorker,
    _Job,
    _profile_search_upper_bound,
)


# ---------------------------------------------------------------------------
# Inference Runtime
# ---------------------------------------------------------------------------
class InferenceRuntime:
    """Persistent pipelined inference engine.

    Usage::

        runtime = InferenceRuntime(devices=["cuda:0", "cuda:1"])
        runtime.start()
        # ... use submit_batch() ...
        runtime.stop()
    """

    def __init__(
        self,
        devices: list[str] | None = None,
        prep_workers: int | None = None,
        post_workers: int | None = None,
        target_batch: int | None = None,
    ):
        if devices is None:
            devices = self._detect_devices()
        self.devices = devices

        # Auto-tune CPU worker counts from available cores
        auto_workers = _auto_cpu_workers()
        self.prep_workers = prep_workers if prep_workers is not None else auto_workers
        self.post_workers = post_workers if post_workers is not None else auto_workers

        # target_batch is a queue-drain hint; the real per-call limit is
        # profiled by each _GpuWorker at session load time.
        self.target_batch = target_batch if target_batch is not None else 4

        _logger.info(
            "InferenceRuntime config: devices=%s, prep_workers=%d, post_workers=%d, target_batch=%d",
            self.devices, self.prep_workers, self.post_workers, self.target_batch,
        )

        # Queues — sized relative to target_batch
        self._gpu_queue: queue.Queue = queue.Queue(maxsize=self.target_batch * 4)
        self._result_queue: queue.Queue = queue.Queue(maxsize=self.target_batch * 4)

        # Workers
        self._gpu_workers: list[_GpuWorker] = []
        self._collector_thread: threading.Thread | None = None
        self._prep_pool: ThreadPoolExecutor | None = None
        self._post_pool: ThreadPoolExecutor | None = None

        # Job registry
        self._jobs: dict[str, _Job] = {}
        self._jobs_lock = threading.Lock()

        # Batch-level tracker (for status recovery after browser reload)
        self._batch_trackers: dict[str, _BatchTracker] = {}
        self._batch_trackers_lock = threading.Lock()

        # Stream / predict_one: dedicated ORT session (not competing with batch queue)
        self._stream_session = None
        self._stream_input_name: str = ""
        self._stream_output_name: str = ""
        self._stream_session_key: str = ""
        self._stream_num_classes: int = 0
        self._stream_lock = threading.Lock()

        self._started = False

    @staticmethod
    def _detect_devices() -> list[str]:
        """Auto-detect available GPU devices, falling back to CPU.

        Prefers idle CUDA devices.  If no CUDA is available at all (no GPU,
        driver issue, all busy with training), falls back to CPU so inference
        still works — just slower.
        """
        try:
            from .torch_device import active_torch_jobs, list_torch_devices
            devices = list_torch_devices()
            busy = set(active_torch_jobs().keys())
            cuda = [d["id"] for d in devices if d.get("kind") == "cuda" and d["id"] not in busy]
            if cuda:
                return cuda
            # All GPUs busy — try to verify at least one CUDA device exists
            all_cuda = [d["id"] for d in devices if d.get("kind") == "cuda"]
            if all_cuda:
                _logger.warning("All CUDA devices busy with training, using first: %s", all_cuda[0])
                return [all_cuda[0]]
        except Exception:
            pass

        # Verify CUDA actually works before defaulting to it
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return ["cuda"]
        except Exception:
            pass

        _logger.warning("No CUDA devices available — inference will use CPU")
        return ["cpu"]

    def start(self):
        if self._started:
            return
        _logger.info("InferenceRuntime starting with devices=%s", self.devices)
        self._prep_pool = ThreadPoolExecutor(max_workers=self.prep_workers, thread_name_prefix="InfPrep")
        self._post_pool = ThreadPoolExecutor(max_workers=self.post_workers, thread_name_prefix="InfPost")

        # Start GPU workers
        for dev in self.devices:
            w = _GpuWorker(dev, self._gpu_queue, self._result_queue, self.target_batch)
            w.start()
            self._gpu_workers.append(w)

        # Start collector
        self._collector_thread = threading.Thread(target=self._collector_loop, daemon=True, name="InfCollector")
        self._collector_thread.start()

        self._started = True
        _logger.info("InferenceRuntime started")

    def stop(self):
        if not self._started:
            return
        # Signal GPU workers to stop
        for _ in self._gpu_workers:
            self._gpu_queue.put(_SHUTDOWN)
        for w in self._gpu_workers:
            w.join(timeout=10)
        self._gpu_workers.clear()

        # Signal collector to stop
        self._result_queue.put(_SHUTDOWN)
        if self._collector_thread:
            self._collector_thread.join(timeout=10)

        if self._prep_pool:
            self._prep_pool.shutdown(wait=False)
        if self._post_pool:
            self._post_pool.shutdown(wait=False)

        self._started = False
        _logger.info("InferenceRuntime stopped")

    # ------------------------------------------------------------------
    # Batch submission
    # ------------------------------------------------------------------
    def predict_batch_stream(
        self,
        project_id: str,
        run_path: Path,
        model_path: Path,
        item_ids: list[str],
        backend: str,
        tta: bool = False,
        force: bool = False,
    ):
        """Generator yielding NDJSON lines, drop-in replacement for old predict_batch_stream."""
        # Register batch tracker for status recovery after browser reload
        batch_id = uuid.uuid4().hex[:12]
        run_id = run_path.name
        tracker = _BatchTracker(
            batch_id=batch_id, project_id=project_id, run_id=run_id,
            item_ids=list(item_ids), total=len(item_ids),
        )
        with self._batch_trackers_lock:
            self._batch_trackers[batch_id] = tracker

        import json
        import json as _json

        from .annotate_index import find_annotate_image
        from .classes import resolve_active_class_ids
        from .config import NORMALIZE
        from .prediction_engine import (
            _ensure_onnx_model,
            _ensure_prediction_artifacts,
            _prediction_artifact_paths,
            _resolve_ort_device,
            _should_use_sliding_window,
        )
        from .run_config import (
            _load_run_arch,
            _load_run_base_channels,
            _load_run_inference_threshold,
            _load_run_input_size,
            _load_run_output_stride,
            _load_run_patch_size,
            _load_run_sw_stride,
            _load_run_train_size,
        )

        # Load run config once
        infer_w, infer_h = _load_run_input_size(run_path)
        run_output_stride = _load_run_output_stride(run_path)
        patch_size = _load_run_patch_size(run_path)
        sw_stride = _load_run_sw_stride(run_path)
        use_sw = _should_use_sliding_window(patch_size, sw_stride, run_output_stride)
        train_size = _load_run_train_size(run_path)

        if not use_sw:
            # Non-SW path: fall back to sequential (rare, legacy small models)
            try:
                for iid in item_ids:
                    t0 = time.perf_counter()
                    try:
                        _pred_path, _conf_path, score = _ensure_prediction_artifacts(
                            project_id, run_path, model_path, iid, backend, tta=tta, force=force,
                        )
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        tracker.completed += 1
                        yield _json.dumps({"item_id": iid, "status": "ok", "score": score, "total_ms": round(elapsed_ms, 1)}, ensure_ascii=False) + "\n"
                    except Exception as exc:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        tracker.completed += 1
                        yield _json.dumps({"item_id": iid, "status": "error", "detail": str(exc), "total_ms": round(elapsed_ms, 1)}, ensure_ascii=False) + "\n"
            finally:
                tracker.client_connected = False
                self._schedule_batch_cleanup(batch_id)
            return

        # Resolve device and ONNX model
        from .torch_device import current_configured_torch_device, resolve_torch_device_or_cpu
        raw_device = current_configured_torch_device()
        device_id = resolve_torch_device_or_cpu(raw_device)
        device_id = _resolve_ort_device(device_id)

        from .run_config import _load_run_num_classes
        num_classes = _load_run_num_classes(run_path)

        run_base_channels = _load_run_base_channels(run_path)
        run_arch = _load_run_arch(run_path)
        onnx_path = _ensure_onnx_model(
            run_path, model_path,
            num_classes=num_classes,
            run_output_stride=run_output_stride,
            run_base_channels=run_base_channels,
            run_arch=run_arch,
        )

        # Read classes
        classes_file = run_path / "classes.json"
        from .paths import classes_path
        if classes_file.exists():
            classes = json.loads(classes_file.read_text(encoding="utf-8"))
        else:
            classes = json.loads(classes_path(project_id).read_text(encoding="utf-8"))

        active_class_ids = None
        _tc_path = run_path / "train_config.json"
        if _tc_path.exists():
            try:
                _tc = json.loads(_tc_path.read_text(encoding="utf-8"))
                _tc_active = _tc.get("active_class_ids")
                if isinstance(_tc_active, list) and _tc_active:
                    active_class_ids = [int(x) for x in _tc_active]
            except Exception:
                pass
        if active_class_ids is None:
            active_class_ids = resolve_active_class_ids(classes)

        inference_threshold = _load_run_inference_threshold(run_path)

        suppress_mask = None
        if active_class_ids is not None:
            m = np.ones(num_classes, dtype=bool)
            for cid in active_class_ids:
                if 0 <= cid < num_classes:
                    m[cid] = False
            if np.any(m):
                suppress_mask = m

        # Submit all images to prep pipeline
        from PIL import Image

        from segcore.training.sliding_window import compute_patch_grid

        job_order: list[str] = []  # preserve item order for output
        results_ready: dict[str, threading.Event] = {}

        def _prep_one(iid: str):
            """Prep worker: load image, build SW job, feed chunks to GPU queue."""
            _logger.info("Prep start: %s", iid)
            try:
                # Check cache first
                pred_path, confidence_path, score_path = _prediction_artifact_paths(
                    run_path, backend, iid, tta=tta, ensure_dir=True
                )
                if not force and pred_path.exists() and confidence_path.exists() and score_path.exists():
                    # Already cached — skip GPU entirely
                    try:
                        score = json.loads(score_path.read_text(encoding="utf-8"))
                    except Exception:
                        score = {}
                    with self._jobs_lock:
                        job_id = f"{iid}:{uuid.uuid4().hex[:8]}"
                        job = _Job(
                            job_id=job_id, item_id=iid, project_id=project_id,
                            run_path=run_path, model_path=model_path, backend=backend,
                            tta=tta, force=force, orig_hw=(0, 0), positions=[], padded=np.empty(0),
                            accum=np.empty(0), count=np.empty(0), gauss_weight=np.empty(0),
                            patch_out=0, margin=0, output_stride=run_output_stride,
                            num_classes=num_classes, patch_size=patch_size, sw_stride=sw_stride,
                            normalize=NORMALIZE, active_class_ids=active_class_ids,
                            suppress_mask=suppress_mask, inference_threshold=inference_threshold,
                            total_tiles=0, accumulated_tiles=0,
                        )
                        job.result = {"score": score, "cached": True}
                        job.result_event.set()
                        self._jobs[job_id] = job
                        results_ready[iid] = job.result_event
                    return

                # Load image
                img_path = find_annotate_image(project_id, iid)
                if not img_path or not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {iid}")
                img = Image.open(img_path).convert("RGB")
                orig_H_full, orig_W_full = img.height, img.width

                # Pre-resize for resized models (camera-independent: absolute train_size)
                if train_size is not None:
                    img = img.resize((train_size[0], train_size[1]), Image.LANCZOS)

                img_np = np.asarray(img)
                H, W = img_np.shape[:2]

                # Build SW job
                margin = patch_size // 2
                padded = np.pad(img_np, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
                H_eff, W_eff = padded.shape[:2]
                H_pad, W_pad, positions = compute_patch_grid(H_eff, W_eff, patch_size, sw_stride)

                extra_b = H_pad - H_eff
                extra_r = W_pad - W_eff
                if extra_b > 0 or extra_r > 0:
                    padded = np.pad(padded, ((0, extra_b), (0, extra_r), (0, 0)), mode="reflect")

                out_h = H_pad // run_output_stride
                out_w = W_pad // run_output_stride
                accum = np.zeros((num_classes, out_h, out_w), dtype="float32")
                count = np.zeros((1, out_h, out_w), dtype="float32")
                patch_out = patch_size // run_output_stride

                sigma = patch_out / _SW_GAUSS_SIGMA_DIV
                ax = np.arange(patch_out, dtype="float32") - patch_out / 2.0 + 0.5
                xx, yy = np.meshgrid(ax, ax)
                gauss_weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype("float32")

                job_id = f"{iid}:{uuid.uuid4().hex[:8]}"
                job = _Job(
                    job_id=job_id, item_id=iid, project_id=project_id,
                    run_path=run_path, model_path=model_path, backend=backend,
                    tta=tta, force=force, orig_hw=(orig_H_full, orig_W_full), positions=positions,
                    padded=padded, accum=accum, count=count, gauss_weight=gauss_weight,
                    patch_out=patch_out, margin=margin, output_stride=run_output_stride,
                    num_classes=num_classes, patch_size=patch_size, sw_stride=sw_stride,
                    normalize=NORMALIZE, active_class_ids=active_class_ids,
                    suppress_mask=suppress_mask, inference_threshold=inference_threshold,
                    total_tiles=len(positions),
                )
                # Load frequency map if available
                try:
                    from segcore.training.frequency_map import load_frequency_map
                    _fmap = load_frequency_map(run_path)
                    if _fmap is not None:
                        job._freq_map = _fmap  # type: ignore[attr-defined]
                except Exception:
                    pass

                with self._jobs_lock:
                    self._jobs[job_id] = job
                    results_ready[iid] = job.result_event

                # Extract ALL patches at once, then normalize in one shot
                mean = np.array(NORMALIZE["mean"], dtype="float32").reshape(1, 3, 1, 1)
                std = np.array(NORMALIZE["std"], dtype="float32").reshape(1, 3, 1, 1)
                n_tiles = len(positions)
                all_patches = np.empty((n_tiles, 3, patch_size, patch_size), dtype="float32")
                for j, (y, x) in enumerate(positions):
                    all_patches[j] = padded[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1)
                # Single vectorized normalize for all patches
                all_patches *= (1.0 / 255.0)
                all_patches = (all_patches - mean) / std

                # Feed chunks to GPU queue
                chunk_size = min(_DEFAULT_CHUNK_SIZE, n_tiles)
                for ci in range(0, n_tiles, chunk_size):
                    chunk_positions = positions[ci:ci + chunk_size]
                    chunk = _Chunk(
                        job_id=job_id,
                        chunk_index=ci // chunk_size,
                        positions=chunk_positions,
                        batch_np=all_patches[ci:ci + chunk_size],
                    )
                    self._gpu_queue.put((chunk, onnx_path, device_id, num_classes))
                _logger.info("Prep done: %s (%d tiles, %d chunks queued)", iid, len(positions), (len(positions) + chunk_size - 1) // chunk_size)

            except Exception as e:
                _logger.exception("Prep failed for %s", iid)
                job_id = f"{iid}:{uuid.uuid4().hex[:8]}"
                job = _Job(
                    job_id=job_id, item_id=iid, project_id=project_id,
                    run_path=run_path, model_path=model_path, backend=backend,
                    tta=tta, force=force, orig_hw=(0, 0), positions=[], padded=np.empty(0),
                    accum=np.empty(0), count=np.empty(0), gauss_weight=np.empty(0),
                    patch_out=0, margin=0, output_stride=run_output_stride,
                    num_classes=num_classes, patch_size=patch_size, sw_stride=sw_stride,
                    normalize=NORMALIZE, active_class_ids=active_class_ids,
                    suppress_mask=suppress_mask, inference_threshold=inference_threshold,
                    total_tiles=0,
                )
                with self._jobs_lock:
                    job.error = str(e)
                    job.result_event.set()
                    self._jobs[job_id] = job
                    results_ready[iid] = job.result_event

        # Submit prep work
        for iid in item_ids:
            job_order.append(iid)
            self._prep_pool.submit(_prep_one, iid)

        # Yield results in order (wrapped in try/finally for browser disconnect)
        try:
            for iid in job_order:
                # Wait for this image's job to complete
                event = None
                while event is None:
                    with self._jobs_lock:
                        event = results_ready.get(iid)
                    if event is None:
                        time.sleep(0.01)

                event.wait()

                # Find the job
                job = None
                with self._jobs_lock:
                    for j in self._jobs.values():
                        if j.item_id == iid and j.result_event.is_set():
                            job = j
                            break

                if job is None:
                    tracker.completed += 1
                    yield _json.dumps({"item_id": iid, "status": "error", "detail": "job not found"}, ensure_ascii=False) + "\n"
                    continue

                tracker.completed += 1

                if job.error:
                    yield _json.dumps({"item_id": iid, "status": "error", "detail": job.error, "total_ms": 0}, ensure_ascii=False) + "\n"
                elif job.result and job.result.get("cached"):
                    yield _json.dumps({"item_id": iid, "status": "ok", "score": job.result.get("score", {}), "total_ms": 0}, ensure_ascii=False) + "\n"
                elif job.result:
                    yield _json.dumps({"item_id": iid, "status": "ok", "score": job.result.get("score", {}), "total_ms": job.result.get("total_ms", 0)}, ensure_ascii=False) + "\n"
                else:
                    yield _json.dumps({"item_id": iid, "status": "error", "detail": "unknown error"}, ensure_ascii=False) + "\n"

                # Cleanup job
                with self._jobs_lock:
                    self._jobs.pop(job.job_id, None)
        except GeneratorExit:
            _logger.info("Batch %s: client disconnected (browser reload?), pipeline continues", batch_id)
            tracker.client_connected = False
            raise
        finally:
            tracker.client_connected = False
            self._schedule_batch_cleanup(batch_id)

    # ------------------------------------------------------------------
    # Batch status helpers
    # ------------------------------------------------------------------
    def get_active_batch(self, project_id: str | None = None, run_id: str | None = None) -> _BatchTracker | None:
        """Return the most recent active batch tracker, optionally filtered."""
        with self._batch_trackers_lock:
            for t in reversed(list(self._batch_trackers.values())):
                if t.completed >= t.total:
                    continue  # already finished
                if project_id and t.project_id != project_id:
                    continue
                if run_id and t.run_id != run_id:
                    continue
                return t
        return None

    def _schedule_batch_cleanup(self, batch_id: str, delay_s: float = 120):
        """Schedule removal of a batch tracker after a delay."""
        def _cleanup():
            with self._batch_trackers_lock:
                self._batch_trackers.pop(batch_id, None)
        timer = threading.Timer(delay_s, _cleanup)
        timer.daemon = True
        timer.start()

    # ------------------------------------------------------------------
    # Collector loop
    # ------------------------------------------------------------------
    def _collector_loop(self):
        """Receive chunk results from GPU workers, accumulate, trigger post-processing."""
        _logger.info("Collector loop started")
        while True:
            try:
                item = self._result_queue.get()
                if item is _SHUTDOWN:
                    return

                status, payload, error_msg = item
                if status == "error":
                    chunk = payload
                    _logger.warning("Collector: chunk error for job %s: %s", chunk.job_id, error_msg)
                    with self._jobs_lock:
                        job = self._jobs.get(chunk.job_id)
                    if job:
                        with self._jobs_lock:
                            job.error = error_msg or "GPU inference failed"
                            job.result_event.set()
                    continue

                result: _ChunkResult = payload
                with self._jobs_lock:
                    job = self._jobs.get(result.job_id)
                if not job:
                    _logger.warning("Collector: no job found for %s", result.job_id)
                    continue

                # Apply suppress mask
                probs = result.probs_np
                if job.suppress_mask is not None:
                    probs[:, job.suppress_mask, :, :] = 0.0

                # Accumulate with gaussian weight
                weighted = probs * job.gauss_weight
                for j, (y, x) in enumerate(result.positions):
                    oy = y // job.output_stride
                    ox = x // job.output_stride
                    job.accum[:, oy:oy + job.patch_out, ox:ox + job.patch_out] += weighted[j]
                    job.count[:, oy:oy + job.patch_out, ox:ox + job.patch_out] += job.gauss_weight

                job.accumulated_tiles += len(result.positions)
                _logger.debug("Collector: job %s tiles %d/%d", job.item_id, job.accumulated_tiles, job.total_tiles)

                # Check if job is complete
                if job.accumulated_tiles >= job.total_tiles:
                    _logger.info("Collector: job %s complete, submitting finalize", job.item_id)
                    self._post_pool.submit(self._finalize_job, job)
            except Exception:
                _logger.exception("Collector: unexpected error")

    def _finalize_job(self, job: _Job):
        """Post-process completed job: argmax, save artifacts."""
        try:
            t0 = time.perf_counter()
            H, W = job.orig_hw
            margin_out = job.margin // job.output_stride
            orig_out_h = H // job.output_stride
            orig_out_w = W // job.output_stride

            count = np.maximum(job.count, 1.0)
            avg_probs = (job.accum / count).astype("float32")
            avg_probs = avg_probs[:, margin_out:margin_out + orig_out_h, margin_out:margin_out + orig_out_w]
            pred = np.argmax(avg_probs, axis=0).astype("int64")

            # Post-processing: remove small connected components
            if getattr(job, "min_area", 0) > 0:
                from segcore.training.postprocess import filter_small_components
                pred = filter_small_components(pred, job.min_area)

            # Post-processing: frequency map penalty
            _freq_map = getattr(job, "_freq_map", None)
            if _freq_map is not None:
                from segcore.training.frequency_map import apply_frequency_map
                _conf_for_freq = np.max(avg_probs, axis=0)
                pred, _conf_for_freq = apply_frequency_map(
                    pred, _conf_for_freq, _freq_map, alpha=0.3,
                )

            import json

            import cv2

            from segcore.image_io import imwrite as _imwrite

            from .prediction_engine import _prediction_artifact_paths

            pred_path, confidence_path, score_path = _prediction_artifact_paths(
                job.run_path, job.backend, job.item_id, tta=job.tta, ensure_dir=True
            )

            # Resize pred and probs to original resolution
            pred_full = cv2.resize(pred.astype("uint8"), (W, H), interpolation=cv2.INTER_NEAREST)
            confidence_full = np.max(avg_probs, axis=0)
            confidence_full = cv2.resize(confidence_full, (W, H), interpolation=cv2.INTER_LINEAR)

            # Save prediction mask and confidence (Unicode-safe)
            _imwrite(str(pred_path), pred_full, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            conf_uint8 = np.clip(confidence_full * 255, 0, 255).astype("uint8")
            _imwrite(str(confidence_path), conf_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # Compute score
            fg_mask = pred_full > 0
            score = {
                "version": 4,
                "inference_device": "ort:pipelined",
                "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
                "num_classes": job.num_classes,
                "foreground_pixels": int(np.sum(fg_mask)),
                "total_pixels": int(fg_mask.size),
                "foreground_ratio": float(np.mean(fg_mask)),
                "max_confidence": float(np.max(confidence_full)),
                "min_confidence": float(np.min(confidence_full)),
                "foreground_mean_confidence": float(np.mean(confidence_full[fg_mask])) if fg_mask.any() else 0.0,
                "background_mean_confidence": float(np.mean(confidence_full[~fg_mask])) if (~fg_mask).any() else 0.0,
            }
            if job.inference_threshold is not None:
                score["inference_threshold"] = job.inference_threshold

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._jobs_lock:
                job.result = {"score": score, "total_ms": round(elapsed_ms, 1)}
                job.result_event.set()

            # Save score JSON and probs in background (not on critical path)
            def _save_deferred():
                try:
                    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
                    probs_path = pred_path.parent / f"{job.item_id}.probs.npy"
                    np.save(probs_path, avg_probs.astype("float16"))
                except Exception:
                    _logger.warning("Deferred save failed for %s", job.item_id)
            threading.Thread(target=_save_deferred, daemon=True).start()

            # Free large arrays
            job.padded = np.empty(0)
            job.accum = np.empty(0)
            job.count = np.empty(0)

        except Exception as e:
            _logger.exception("Finalize failed for %s", job.item_id)
            with self._jobs_lock:
                job.error = str(e)
                job.result_event.set()


    # ------------------------------------------------------------------
    # Stream session management (for predict_one / WebSocket)
    # ------------------------------------------------------------------
    def _ensure_stream_session(self, onnx_path: str, device_id: str, num_classes: int):
        """Load or reuse a dedicated ORT session for single-frame inference."""
        key = f"{onnx_path}:{device_id}"
        if self._stream_session_key == key and self._stream_session is not None:
            return
        import onnxruntime as ort

        from .prediction_engine import _build_ort_session_options, _cuda_provider_options

        # DLL setup (same as _GpuWorker._ensure_session)
        if device_id.startswith("cuda"):
            from .ort_infra import _preload_cuda_dlls
            _preload_cuda_dlls()

        use_cuda = device_id.startswith("cuda")
        opts = _build_ort_session_options(use_cuda=use_cuda)
        if use_cuda:
            providers = [
                ("CUDAExecutionProvider", _cuda_provider_options(device_id)),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        except Exception:
            if use_cuda:
                session = ort.InferenceSession(
                    onnx_path,
                    sess_options=_build_ort_session_options(use_cuda=False),
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

        self._stream_session = session
        self._stream_input_name = session.get_inputs()[0].name
        self._stream_output_name = session.get_outputs()[0].name
        self._stream_session_key = key
        self._stream_num_classes = num_classes

        # Warmup with actual model dimensions
        inp_shape = session.get_inputs()[0].shape
        try:
            wh = int(inp_shape[2]), int(inp_shape[3])
        except (TypeError, IndexError, ValueError):
            # Dynamic axes: shape contains strings like "height", "width"
            wh = (256, 256)
        try:
            dummy = np.zeros((1, 3, wh[0], wh[1]), dtype="float32")
            session.run([self._stream_output_name], {self._stream_input_name: dummy})
        except Exception:
            pass
        _logger.info("Stream session loaded for %s on %s (input=%s)", onnx_path, device_id, inp_shape)

    def warm_up_session(self, onnx_path: str, device_id: str, num_classes: int) -> dict:
        """Eagerly load the stream ORT session. Returns timing info."""
        t0 = time.perf_counter()
        with self._stream_lock:
            self._ensure_stream_session(onnx_path, device_id, num_classes)
        elapsed = (time.perf_counter() - t0) * 1000
        return {"status": "ready", "warmup_ms": round(elapsed, 1), "device": device_id}

    def release_stream_session(self):
        """Release the stream ORT session."""
        with self._stream_lock:
            self._stream_session = None
            self._stream_session_key = ""
            self._stream_num_classes = 0
        _logger.info("Stream session released")

    def stream_session_status(self) -> dict:
        return {
            "loaded": self._stream_session is not None,
            "session_key": self._stream_session_key,
        }

    # ------------------------------------------------------------------
    # Mask overlay generation
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_mask_overlay(pred: np.ndarray, orig_w: int, orig_h: int) -> str:
        """Encode segmentation mask as a semi-transparent RGBA PNG (base64).

        Uses a fixed palette for class IDs. Background (0) is fully transparent.
        Returns base64-encoded PNG string, or empty string if no foreground.
        """
        if not np.any(pred > 0):
            return ""
        import base64
        import io  # noqa: F401

        import cv2

        # Fixed palette: class_id -> (R, G, B)
        _PALETTE = [
            (0, 0, 0),        # 0: background (transparent)
            (255, 60, 60),     # 1: red
            (60, 120, 255),    # 2: blue
            (60, 220, 60),     # 3: green
            (255, 180, 0),     # 4: orange
            (213, 94, 0),      # 5: vermilion
            (0, 220, 220),     # 6: cyan
            (255, 255, 60),    # 7: yellow
            (255, 100, 180),   # 8: pink
        ]

        h, w = pred.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for cid in range(1, int(pred.max()) + 1):
            mask = pred == cid
            if not np.any(mask):
                continue
            color = _PALETTE[cid % len(_PALETTE)]
            rgba[mask, 0] = color[0]
            rgba[mask, 1] = color[1]
            rgba[mask, 2] = color[2]
            rgba[mask, 3] = 140  # semi-transparent

        # Resize to original image dimensions
        if (w, h) != (orig_w, orig_h):
            rgba = cv2.resize(rgba, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Encode as PNG (fast, no optimization)
        ok, png_buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not ok:
            return ""
        return base64.b64encode(png_buf.tobytes()).decode("ascii")

    # ------------------------------------------------------------------
    # Single-frame immediate inference (bypasses batch queue)
    # ------------------------------------------------------------------
    def predict_one(
        self,
        image_bytes: bytes,
        onnx_path: str,
        device_id: str,
        num_classes: int,
        normalize: dict,
        patch_size: int,
        frame_id: str = "",
        classes: list[dict] | None = None,
        sw_stride: int = 0,
        output_stride: int = 2,
    ):
        """Run inference on a single image immediately. Returns StreamInferenceResult.

        Uses sliding-window inference when sw_stride > 0 (matches Results tab behavior).
        This does NOT use the batch GPU queue. It uses a dedicated ORT session
        so streaming inference doesn't contend with batch processing.
        """
        import cv2

        from segcore.training.sliding_window import (
            normalize_logits_batch,  # noqa: F401
            sliding_window_predict_infer_fn,
        )

        from .inference_types import StreamInferenceResult
        from .region_extract import extract_regions

        timings = {}
        t_total = time.perf_counter()

        # 1. Decode
        t0 = time.perf_counter()
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        timings["decode"] = round((time.perf_counter() - t0) * 1000, 2)
        orig_h, orig_w = img.shape[:2]

        # 2. Ensure ORT session
        t0 = time.perf_counter()
        with self._stream_lock:
            self._ensure_stream_session(onnx_path, device_id, num_classes)
            session = self._stream_session
            out_name = self._stream_output_name
            inp_name = self._stream_input_name
        timings["_lock_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        def _ort_infer(batch_np: np.ndarray) -> np.ndarray:
            return session.run([out_name], {inp_name: batch_np})[0]

        # Always use sliding-window inference (matches Results tab / training evaluation)
        t0 = time.perf_counter()
        effective_stride = sw_stride if sw_stride > 0 else (patch_size * 3 // 4)
        pred, probs_full = sliding_window_predict_infer_fn(
            _ort_infer, img, patch_size, effective_stride,
            num_classes, output_stride, normalize,
        )
        timings["inference"] = round((time.perf_counter() - t0) * 1000, 2)
        pred = pred.astype(np.uint8)
        confidence = np.max(probs_full, axis=0).astype(np.float32)

        # 4. Postprocess: region extraction
        t0 = time.perf_counter()

        # Post-processing: remove small connected components
        _min_area = getattr(self, "_min_area", 0)
        if _min_area > 0:
            from segcore.training.postprocess import filter_small_components
            pred = filter_small_components(pred, _min_area)

        # Post-processing: frequency map penalty
        _freq_map_stream = getattr(self, "_freq_map", None)
        if _freq_map_stream is not None:
            from segcore.training.frequency_map import apply_frequency_map
            pred, confidence = apply_frequency_map(pred, confidence, _freq_map_stream, alpha=0.3)

        # Extract regions at model resolution (fast), then scale bboxes to original
        regions = extract_regions(pred, confidence, classes)
        scale_x = orig_w / pred.shape[1]
        scale_y = orig_h / pred.shape[0]
        for r in regions:
            bx, by, bw, bh = r.bbox
            r.bbox = (
                int(bx * scale_x), int(by * scale_y),
                int(bw * scale_x), int(bh * scale_y),
            )
            cx, cy = r.centroid
            r.centroid = (int(cx * scale_x), int(cy * scale_y))
            r.area_px = int(r.area_px * scale_x * scale_y)

        pred_full = pred
        conf_full = confidence
        fg_mask = pred_full > 0
        fg_ratio = float(np.mean(fg_mask)) if fg_mask.size > 0 else 0.0
        max_conf = float(np.max(conf_full)) if conf_full.size > 0 else 0.0
        defect_found = bool(np.any(fg_mask))
        try:
            mask_b64 = self._encode_mask_overlay(pred, orig_w, orig_h)
        except Exception as e:
            _logger.warning("Mask overlay encoding failed: %s", e)
            mask_b64 = ""
        timings["postprocess"] = round((time.perf_counter() - t0) * 1000, 2)
        timings["total"] = round((time.perf_counter() - t_total) * 1000, 2)

        result_id = f"r-{uuid.uuid4().hex[:12]}"
        return StreamInferenceResult(
            frame_id=frame_id,
            judgement="NG" if defect_found else "OK",
            defect_found=defect_found,
            regions=regions,
            summary={
                "fg_ratio": round(fg_ratio, 6),
                "max_confidence": round(max_conf, 4),
                "num_defects": len(regions),
            },
            latency_ms=timings,
            result_id=result_id,
            mask_png_b64=mask_b64,
        )

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        onnx_path: str,
        device_id: str,
        num_classes: int,
        normalize: dict,
        patch_size: int,
        frame_id: str = "",
        classes: list[dict] | None = None,
        sw_stride: int = 0,
        output_stride: int = 2,
    ):
        """Run inference on a BGR numpy array directly (no decode step).

        Used by CameraManager to avoid encode/decode overhead.
        Always uses sliding-window inference.
        """
        import cv2

        from segcore.training.sliding_window import sliding_window_predict_infer_fn

        from .inference_types import StreamInferenceResult
        from .region_extract import extract_regions

        timings = {}
        t_total = time.perf_counter()

        # 1. BGR→RGB
        t0 = time.perf_counter()
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timings["decode"] = round((time.perf_counter() - t0) * 1000, 2)
        orig_h, orig_w = img.shape[:2]

        # 2. Ensure ORT session
        t0 = time.perf_counter()
        with self._stream_lock:
            self._ensure_stream_session(onnx_path, device_id, num_classes)
            session = self._stream_session
            out_name = self._stream_output_name
            inp_name = self._stream_input_name
        timings["_lock_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        def _ort_infer(batch_np: np.ndarray) -> np.ndarray:
            return session.run([out_name], {inp_name: batch_np})[0]

        # 3. Sliding-window inference
        t0 = time.perf_counter()
        effective_stride = sw_stride if sw_stride > 0 else (patch_size * 3 // 4)
        pred, probs_full = sliding_window_predict_infer_fn(
            _ort_infer, img, patch_size, effective_stride,
            num_classes, output_stride, normalize,
        )
        timings["inference"] = round((time.perf_counter() - t0) * 1000, 2)
        pred = pred.astype(np.uint8)
        confidence = np.max(probs_full, axis=0).astype(np.float32)

        # 4. Postprocess
        t0 = time.perf_counter()
        regions = extract_regions(pred, confidence, classes)
        scale_x = orig_w / pred.shape[1]
        scale_y = orig_h / pred.shape[0]
        for r in regions:
            bx, by, bw, bh = r.bbox
            r.bbox = (
                int(bx * scale_x), int(by * scale_y),
                int(bw * scale_x), int(bh * scale_y),
            )
            cx, cy = r.centroid
            r.centroid = (int(cx * scale_x), int(cy * scale_y))
            r.area_px = int(r.area_px * scale_x * scale_y)

        fg_mask = pred > 0
        fg_ratio = float(np.mean(fg_mask)) if fg_mask.size > 0 else 0.0
        max_conf = float(np.max(confidence)) if confidence.size > 0 else 0.0
        defect_found = bool(np.any(fg_mask))
        try:
            mask_b64 = self._encode_mask_overlay(pred, orig_w, orig_h)
        except Exception as e:
            _logger.warning("Mask overlay encoding failed: %s", e)
            mask_b64 = ""
        timings["postprocess"] = round((time.perf_counter() - t0) * 1000, 2)
        timings["total"] = round((time.perf_counter() - t_total) * 1000, 2)

        result_id = f"r-{uuid.uuid4().hex[:12]}"
        return StreamInferenceResult(
            frame_id=frame_id,
            judgement="NG" if defect_found else "OK",
            defect_found=defect_found,
            regions=regions,
            summary={
                "fg_ratio": round(fg_ratio, 6),
                "max_confidence": round(max_conf, 4),
                "num_defects": len(regions),
            },
            latency_ms=timings,
            result_id=result_id,
            mask_png_b64=mask_b64,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_runtime: InferenceRuntime | None = None
_runtime_lock = threading.Lock()


def get_inference_runtime() -> InferenceRuntime:
    """Get or create the global InferenceRuntime singleton."""
    global _runtime
    if _runtime is not None and _runtime._started:
        return _runtime
    with _runtime_lock:
        if _runtime is None or not _runtime._started:
            _runtime = InferenceRuntime()
            _runtime.start()
        return _runtime
