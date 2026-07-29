# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""GPU worker and job types for the pipelined inference runtime.

Extracted verbatim from inference_runtime.py during the pre-OSS refactor:
auto-tuning constants and helpers, the pipeline dataclasses (_Chunk,
_ChunkResult, _BatchTracker, _Job), and the persistent _GpuWorker thread
(ORT session load, CUDA DLL preload, VRAM-aware batch-size profiling,
cross-image batching). The _SHUTDOWN sentinel is defined here; its identity
is shared with InferenceRuntime via import.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

# Sentinel value to signal worker shutdown
_SHUTDOWN = object()

# ---------------------------------------------------------------------------
# Auto-tuning constants — no magic numbers in worker code
# ---------------------------------------------------------------------------
# Safety margin applied to profiled max batch size (keeps ~20% headroom for
# OS/driver allocations, other GPU consumers, etc.)
_PROFILE_SAFETY_MARGIN = 0.80

# Minimum sub-batch size (even on the weakest GPU we try at least 1 patch)
_MIN_BATCH_SIZE = 1

# Default chunk size for prep→GPU queue.  Overridden per-session once the
# GPU worker profiles the actual max batch.  16 is a reasonable default that
# keeps queue items small without excessive Python-loop overhead.
_DEFAULT_CHUNK_SIZE = 16

# Gaussian sigma denominator for sliding-window weighting.
# sigma = patch_out / _SW_GAUSS_SIGMA_DIV  → standard in literature.
_SW_GAUSS_SIGMA_DIV = 4.0


def _auto_cpu_workers() -> int:
    """Choose CPU worker count based on available cores.

    Returns a sensible default for prep/post thread-pool sizes that works on
    both a 4-core laptop and a 64-core workstation.
    """
    import os
    cores = os.cpu_count() or 4
    # Use ~half the cores (capped 2..8) — the other half is for the main
    # thread, collector, GPU workers, and the OS.
    return max(2, min(8, cores // 2))


def _estimate_vram_mb(device_id: str) -> int:
    """Return total GPU VRAM in MB, or 0 if unknown."""
    if not device_id.startswith("cuda"):
        return 0
    try:
        import torch
        idx = int(device_id.split(":")[-1]) if ":" in device_id else 0
        total = torch.cuda.get_device_properties(idx).total_mem
        return int(total / (1024 * 1024))
    except Exception:
        return 0


def _profile_search_upper_bound(vram_mb: int, patch_h: int, patch_w: int) -> int:
    """Estimate the upper bound for batch-size binary search.

    A single float32 patch occupies  3 * H * W * 4 bytes of input, and the
    output + intermediate buffers roughly 3-5x that.  We assume 5x and leave
    50% of VRAM for the model itself, then cap at 256 (diminishing returns
    beyond that).
    """
    if vram_mb <= 0:
        return 64  # unknown VRAM — use conservative default
    per_patch_mb = 3 * patch_h * patch_w * 4 * 5 / (1024 * 1024)
    if per_patch_mb <= 0:
        return 64
    available_mb = vram_mb * 0.50  # half for model weights, half for data
    upper = int(available_mb / per_patch_mb)
    return max(4, min(256, upper))


# ---------------------------------------------------------------------------
# Lightweight internal types (avoid importing inference_types for simplicity)
# ---------------------------------------------------------------------------
@dataclass
class _Chunk:
    """Batch of normalized patches ready for GPU."""
    job_id: str
    chunk_index: int
    positions: list[tuple[int, int]]
    batch_np: np.ndarray  # [B, 3, patch, patch] float32


@dataclass
class _ChunkResult:
    """GPU output for one chunk."""
    job_id: str
    chunk_index: int
    positions: list[tuple[int, int]]
    probs_np: np.ndarray  # [B, C, patch_out, patch_out] float32


@dataclass
class _BatchTracker:
    """Tracks a batch prediction session for status recovery after browser reload."""
    batch_id: str
    project_id: str
    run_id: str
    item_ids: list[str]
    total: int
    completed: int = 0
    started_at: float = field(default_factory=time.time)
    client_connected: bool = True


@dataclass
class _Job:
    """Per-image sliding-window job tracker."""
    job_id: str
    item_id: str
    project_id: str
    run_path: Path
    model_path: Path
    backend: str
    tta: bool
    force: bool
    orig_hw: tuple[int, int]
    positions: list[tuple[int, int]]
    padded: np.ndarray
    accum: np.ndarray
    count: np.ndarray
    gauss_weight: np.ndarray
    patch_out: int
    margin: int
    output_stride: int
    num_classes: int
    patch_size: int
    sw_stride: int
    normalize: dict
    active_class_ids: list[int] | None
    suppress_mask: np.ndarray | None
    inference_threshold: float | None
    total_tiles: int
    accumulated_tiles: int = 0
    result_event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# GPU Worker
# ---------------------------------------------------------------------------
class _GpuWorker(threading.Thread):
    """Persistent GPU worker: drains chunk queue, runs ORT inference."""

    def __init__(
        self,
        device_id: str,
        in_q: queue.Queue,
        out_q: queue.Queue,
        target_batch: int | None = None,
        flush_timeout_s: float = 0.005,
    ):
        super().__init__(daemon=True, name=f"GpuWorker-{device_id}")
        self.device_id = device_id
        self.in_q = in_q
        self.out_q = out_q
        # target_batch is set after profiling; this is just the queue-drain
        # hint (how many chunks to collect before flushing).
        self.target_batch = target_batch if target_batch is not None else 4
        self.flush_timeout_s = flush_timeout_s
        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._session_key: str = ""
        self._num_classes: int = 0

    def _ensure_session(self, onnx_path: Path, device_id: str, num_classes: int):
        """Load or reuse ORT session."""
        key = f"{onnx_path}:{device_id}"
        if self._session_key == key and self._session is not None:
            return
        import onnxruntime as ort

        from .prediction_engine import (  # noqa: F401
            _build_ort_session_options,
            _cuda_provider_options,
            _ensure_onnx_model,
        )

        # Preload PyTorch + cuDNN 8 DLLs for Windows (ORT 1.18.x needs cuDNN 8)
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
            session = ort.InferenceSession(
                onnx_path.as_posix(), sess_options=opts, providers=providers
            )
        except Exception:
            if use_cuda:
                session = ort.InferenceSession(
                    onnx_path.as_posix(),
                    sess_options=_build_ort_session_options(use_cuda=False),
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        self._session_key = key
        self._num_classes = num_classes

        # Resolve actual model input dimensions for warmup & profiling
        inp_shape = session.get_inputs()[0].shape  # e.g. ['batch', 3, 256, 256]
        try:
            self._model_h = int(inp_shape[2])
            self._model_w = int(inp_shape[3])
        except (TypeError, IndexError, ValueError):
            self._model_h, self._model_w = 256, 256
            _logger.warning(
                "GpuWorker %s: dynamic input shape %s, assuming 256x256 for profiling",
                self.device_id, inp_shape,
            )

        # Warmup with actual model dimensions (not hardcoded 256x256)
        try:
            dummy = np.zeros((1, 3, self._model_h, self._model_w), dtype="float32")
            session.run([self._output_name], {self._input_name: dummy})
        except Exception:
            pass

        self._max_patches_per_call = self._profile_max_infer_batch()
        # Update target_batch to match profiled capability
        self.target_batch = max(self.target_batch, self._max_patches_per_call)
        _logger.info("GpuWorker %s: session loaded for %s (max_batch=%d, target_batch=%d, model=%dx%d)",
                      self.device_id, onnx_path.name, self._max_patches_per_call,
                      self.target_batch, self._model_h, self._model_w)

    def run(self):
        pending: list[_Chunk] = []
        pending_meta: list[tuple[Path, str, int]] = []  # (onnx_path, device, num_classes)

        while True:
            try:
                # Drain queue up to target_batch or flush_timeout
                deadline = time.monotonic() + self.flush_timeout_s
                while len(pending) < self.target_batch:
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        item = self.in_q.get(timeout=remaining if pending else 2.0)
                    except queue.Empty:
                        break
                    if item is _SHUTDOWN:
                        # Flush remaining
                        if pending:
                            self._run_batch(pending)
                        return
                    chunk, onnx_path, device, num_classes = item
                    pending.append(chunk)
                    pending_meta.append((onnx_path, device, num_classes))

                if not pending:
                    continue

                self._run_batch_with_meta(pending, pending_meta)
                pending.clear()
                pending_meta.clear()
            except Exception:
                _logger.exception("GpuWorker %s: unexpected error in run loop", self.device_id)
                # Signal errors for any pending chunks so jobs don't hang
                for chunk in pending:
                    self.out_q.put(("error", chunk, "worker loop error"))
                pending.clear()
                pending_meta.clear()

    def _run_batch_with_meta(self, chunks: list[_Chunk], meta: list[tuple[Path, str, int]]):
        """Run a batch of chunks, ensuring session is loaded."""
        # All chunks in one batch should use the same model
        onnx_path, device, num_classes = meta[0]
        try:
            self._ensure_session(onnx_path, device, num_classes)
        except Exception:
            _logger.exception("GpuWorker %s: failed to load session", self.device_id)
            for chunk in chunks:
                self.out_q.put(("error", chunk, "session load failed"))
            return
        self._run_batch(chunks)

    # Profiled at session load; updated by _profile_max_infer_batch().
    _max_patches_per_call: int = _MIN_BATCH_SIZE  # conservative default until profiled

    def _profile_max_infer_batch(self) -> int:
        """Binary-search the largest batch that fits in GPU for ORT forward.

        Runs actual inference with increasing batch sizes and backs off on
        OOM.  The search range upper bound is estimated from available VRAM so
        that a 4 GB GPU doesn't waste time probing batch=256 while an 80 GB
        GPU can find its real limit.

        Result is cached on the worker for the lifetime of the session.
        """
        if not self._session:
            return _MIN_BATCH_SIZE

        h, w = self._model_h, self._model_w
        vram_mb = _estimate_vram_mb(self.device_id)
        search_hi = _profile_search_upper_bound(vram_mb, h, w)

        _logger.info(
            "GpuWorker %s: profiling batch size (VRAM=%dMB, search 1..%d, patch=%dx%d)",
            self.device_id, vram_mb, search_hi, h, w,
        )

        lo, hi, best = 1, search_hi, 1
        while lo <= hi:
            mid = (lo + hi) // 2
            dummy = np.zeros((mid, 3, h, w), dtype="float32")
            try:
                self._session.run([self._output_name], {self._input_name: dummy})
                best = mid
                lo = mid + 1
            except Exception:
                hi = mid - 1
                # Clear ORT allocator after OOM
                try:
                    self._session.run(
                        [self._output_name],
                        {self._input_name: np.zeros((1, 3, h, w), dtype="float32")},
                    )
                except Exception:
                    pass

        safe = max(_MIN_BATCH_SIZE, int(best * _PROFILE_SAFETY_MARGIN))
        _logger.info(
            "GpuWorker %s: profiled max_infer_batch=%d (safe=%d, VRAM=%dMB) for %dx%d patches",
            self.device_id, best, safe, vram_mb, h, w,
        )
        return safe

    def _run_batch(self, chunks: list[_Chunk]):
        """Concatenate chunk batches and run ORT inference.

        To avoid GPU OOM the combined tensor is split into sub-batches of at
        most ``_max_patches_per_call`` patches (auto-profiled at session load).
        If OOM still occurs, the batch size is halved and the call retried.
        """
        if not self._session:
            return
        try:
            from segcore.training.sliding_window import normalize_logits_batch, softmax_np

            all_batches = [c.batch_np for c in chunks]
            sizes = [b.shape[0] for b in all_batches]
            combined = np.concatenate(all_batches, axis=0)
            combined = np.ascontiguousarray(combined, dtype="float32")

            total = combined.shape[0]
            max_b = self._max_patches_per_call

            # Run in sub-batches to stay within VRAM budget.
            # On OOM, halve the sub-batch size and retry (adapts at runtime).
            logits_parts: list[np.ndarray] = []
            pos = 0
            while pos < total:
                sub = combined[pos:pos + max_b]
                try:
                    outputs = self._session.run(
                        [self._output_name], {self._input_name: sub}
                    )
                    logits_parts.append(normalize_logits_batch(outputs[0], self._num_classes))
                    pos += max_b
                except Exception as oom_exc:
                    if "allocate" in str(oom_exc).lower() and max_b > _MIN_BATCH_SIZE:
                        max_b = max(_MIN_BATCH_SIZE, max_b // 2)
                        self._max_patches_per_call = max_b
                        _logger.warning(
                            "GpuWorker %s: OOM during inference, reducing batch to %d",
                            self.device_id, max_b,
                        )
                        # Retry same position with smaller batch
                        continue
                    raise

            logits = np.concatenate(logits_parts, axis=0) if len(logits_parts) > 1 else logits_parts[0]

            offset = 0
            for chunk, sz in zip(chunks, sizes):
                chunk_logits = logits[offset:offset + sz]
                probs = softmax_np(chunk_logits, axis=1)
                result = _ChunkResult(
                    job_id=chunk.job_id,
                    chunk_index=chunk.chunk_index,
                    positions=chunk.positions,
                    probs_np=probs,
                )
                self.out_q.put(("ok", result, None))
                offset += sz
        except Exception as e:
            _logger.exception("GpuWorker %s: batch failed", self.device_id)
            for chunk in chunks:
                self.out_q.put(("error", chunk, str(e)))
