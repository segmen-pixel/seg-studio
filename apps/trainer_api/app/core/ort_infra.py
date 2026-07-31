# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""ONNX export and ORT/torch model-loading infrastructure.

Extracted verbatim from prediction_engine.py during the pre-OSS refactor:
ONNX (re-)export with dynamic axes, ORT session options / CUDA provider
options / device resolution, the VRAM-scaled model caches (torch + ORT
sessions) with their guards, and the cached loaders. prediction_engine
re-imports everything, so the caches remain process-wide singletons.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException

from .cache_utils import ThreadSafeLRUCache
from .exceptions import PredictModelMissingError
from .run_config import _load_run_input_size


def _auto_model_cache_size() -> int:
    """Choose model cache capacity based on GPU VRAM.

    Each cached model + ORT session may hold 50-200 MB of GPU memory.
    Scale the cache so it doesn't consume more than ~40% of the smallest GPU.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            if vram_mb >= 16_000:   # 16 GB+
                return 8
            if vram_mb >= 8_000:    # 8 GB+
                return 4
            if vram_mb >= 4_000:    # 4 GB+
                return 2
            return 1                # < 4 GB — keep only 1
    except Exception:
        pass
    return 2  # CPU fallback — memory is usually abundant but be conservative


_MODEL_CACHE_SIZE = _auto_model_cache_size()

# Cached torch models for repeated inference on the same run/device.
# Capacity auto-scaled by available VRAM.
_torch_model_cache: ThreadSafeLRUCache = ThreadSafeLRUCache(maxsize=_MODEL_CACHE_SIZE)
_torch_model_cache_guard = threading.Lock()

# Cached ORT sessions for repeated inference on the same run/provider.
_ort_session_cache: ThreadSafeLRUCache = ThreadSafeLRUCache(maxsize=_MODEL_CACHE_SIZE)
_ort_session_cache_guard = threading.Lock()

# When the cached sessions were last put to work, and how many calls are using
# one right now. The LRU can answer neither question: ThreadSafeLRUCache stamps
# a key on put() and get() does not refresh it, so its clock measures age
# rather than idleness -- and a TTL would not be enough on its own either,
# because expiry is only ever noticed on the next access and a cache nobody is
# accessing is exactly the one whose VRAM has to come back.
#: None until a session is handed out. Not 0.0: time.monotonic() has no
#: defined origin, so on a machine that booted a moment ago a legitimate
#: reading can be smaller than any interval measured against it, and
#: subtracting one from the other goes negative. A sentinel a clock value
#: can reach is not a sentinel.
_ort_session_last_used: float | None = None
_ort_sessions_in_flight: int = 0
_ort_session_use_guard = threading.Lock()

_ort_idle_thread_guard = threading.Lock()
_ort_idle_thread_started = False

#: Owners of sessions the cache above cannot see. See register_idle_release_hook().
_idle_release_hooks: list[Callable[[], bool]] = []
_idle_release_hooks_guard = threading.Lock()

#: How often the idle check runs. Well under the release threshold so the
#: sessions are handed back close to when they become eligible.
ORT_IDLE_POLL_SECONDS = 30.0


def _idle_release_seconds() -> float:
    """How long the sessions may sit unused before the VRAM is handed back.

    SEG_ORT_IDLE_RELEASE_SECONDS overrides it; 0 switches the release off. A
    value that is not a number falls back to the default rather than raising:
    this is read at import time, so a typo in the environment must not be able
    to stop the server from starting.
    """
    raw = os.environ.get("SEG_ORT_IDLE_RELEASE_SECONDS", "")
    if not raw.strip():
        return 300.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logging.getLogger(__name__).warning(
            "SEG_ORT_IDLE_RELEASE_SECONDS=%r is not a number; using the 300s default", raw,
        )
        return 300.0


ORT_IDLE_RELEASE_SECONDS = _idle_release_seconds()


def reclaim_released_vram() -> None:
    """Make the memory behind a just-dropped session actually come back.

    Dropping the last reference is not the same as freeing the card: an ORT
    CUDA session only releases its arena when the object is collected, and on
    CPython a session caught in a reference cycle waits for the collector. The
    torch call is separate -- it returns torch's own cache, not ORT's, and is
    kept because the same runs put weights there too.

    Every place that lets go of a session ends here, so there is one answer to
    "and now how does the VRAM come back" rather than a copy per caller.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clear_ort_session_cache() -> None:
    """Drop cached ONNX Runtime sessions and the VRAM they hold.

    The counterpart to clear_torch_model_cache(), which existed while this did
    not. An ORT CUDA session's arena belongs to ONNX Runtime rather than to
    torch, so torch.cuda.empty_cache() never returns it, and the LRU only
    evicted once a fifth different model was loaded. Measured on a 4 GB card:
    a session loaded for one batch of predictions was still holding 3.9 GB of
    4 GB an hour and a half after the training run that followed it had
    finished, with the GPU otherwise idle.

    This covers the sessions in _ort_session_cache and nothing else. The
    pipelined runtime's GPU workers and its stream path each hold a session of
    their own that never enters this cache; those are released through the
    hooks registered with register_idle_release_hook().
    """
    with _ort_session_cache_guard:
        _ort_session_cache.clear()
    reclaim_released_vram()


def clear_torch_model_cache() -> None:
    with _torch_model_cache_guard:
        _torch_model_cache.clear()
    reclaim_released_vram()


def note_ort_session_use() -> None:
    """Record that a cached ORT session was just handed out or used."""
    global _ort_session_last_used
    with _ort_session_use_guard:
        _ort_session_last_used = time.monotonic()


@contextmanager
def ort_session_in_use():
    """Hold the idle release off for the duration of one ORT call.

    A single session.run() over a large image can outlast the idle threshold.
    Releasing the cache underneath it would free nothing -- the caller still
    holds the session -- while making the next request build a second one
    beside the first, which on a 4 GB card is the failure this whole mechanism
    exists to avoid. Counting the calls in flight keeps that from happening.
    """
    global _ort_session_last_used, _ort_sessions_in_flight
    with _ort_session_use_guard:
        _ort_sessions_in_flight += 1
        _ort_session_last_used = time.monotonic()
    try:
        yield
    finally:
        with _ort_session_use_guard:
            _ort_sessions_in_flight -= 1
            _ort_session_last_used = time.monotonic()


def ort_session_idle_seconds() -> float:
    """Seconds since the cached sessions were last used.

    0.0 means "not idle": either a call is in flight, or nothing has been handed
    out since the process started. In the second case there is no evidence of
    use, so there is no evidence of idleness either, and the release stays out
    of it rather than guessing at a cache it never saw filled.
    """
    with _ort_session_use_guard:
        if _ort_sessions_in_flight > 0:
            return 0.0
        last = _ort_session_last_used
    if last is None:
        return 0.0
    return max(0.0, time.monotonic() - last)


def release_ort_sessions_if_idle(idle_seconds: float | None = None) -> bool:
    """Drop the cached ORT sessions once they have been idle long enough.

    The only other caller of clear_ort_session_cache() is _release_gpu_caches(),
    which runs when a training run starts -- so a machine used for predictions
    and then left alone kept the card until the next run or a restart. Measured
    on a 4 GB GTX 1650: 3,915 of 4,096 MiB still held an hour and a half after
    the GPU had gone idle at 2.34 W. Returns True when the cache was released.
    """
    threshold = ORT_IDLE_RELEASE_SECONDS if idle_seconds is None else idle_seconds
    if threshold <= 0:
        return False
    count = len(_ort_session_cache)
    if count == 0:
        return False
    idle = ort_session_idle_seconds()
    if idle < threshold:
        return False
    clear_ort_session_cache()
    logging.getLogger(__name__).info(
        "Released %d cached ORT session(s) after %.0fs idle", count, idle,
    )
    return True


def register_idle_release_hook(hook: Callable[[], bool]) -> None:
    """Have the idle poller also ask `hook` to hand its VRAM back.

    Not everything holding a session is in _ort_session_cache: the pipelined
    runtime's GPU workers and its stream path each load one of their own, so
    clearing the cache says nothing about them. Neither can be released from
    here, either -- only the code that owns a session knows whether it is
    running one. So the poller asks, and the owner answers.

    A hook rather than an import also keeps the dependency pointing one way:
    inference_runtime imports ort_infra, and ort_infra knows nothing about it.

    Registering the same hook twice registers it once; the poller runs
    for the life of the process and a duplicate would only double the log
    lines.
    """
    with _idle_release_hooks_guard:
        if hook not in _idle_release_hooks:
            _idle_release_hooks.append(hook)


def unregister_idle_release_hook(hook: Callable[[], bool]) -> None:
    """Stop asking `hook`. Unknown hooks are ignored.

    A runtime that has been stopped no longer owns anything to release, and a
    poller still holding its bound method would keep the instance -- and the
    session it was told to drop -- alive for the life of the process.
    """
    with _idle_release_hooks_guard:
        if hook in _idle_release_hooks:
            _idle_release_hooks.remove(hook)


def run_idle_release_hooks() -> int:
    """Ask every registered owner to release, and report how many did.

    One hook raising must not cost the others their turn: they hold separate
    sessions on the same card, and the poller only comes round again after
    ORT_IDLE_POLL_SECONDS.
    """
    with _idle_release_hooks_guard:
        hooks = list(_idle_release_hooks)
    released = 0
    for hook in hooks:
        try:
            if hook():
                released += 1
        except Exception:
            logging.getLogger(__name__).warning(
                "an idle release hook failed", exc_info=True,
            )
    return released


def start_ort_idle_release_thread() -> bool:
    """Start the poller behind release_ort_sessions_if_idle(). Idempotent.

    Returns False when a poller is already running or the release is switched
    off; two pollers on one cache would only double the log lines.
    """
    global _ort_idle_thread_started
    logger = logging.getLogger(__name__)
    with _ort_idle_thread_guard:
        if _ort_idle_thread_started:
            return False
        if ORT_IDLE_RELEASE_SECONDS <= 0:
            logger.info("ORT idle release disabled (SEG_ORT_IDLE_RELEASE_SECONDS=0)")
            return False
        _ort_idle_thread_started = True

    def _loop() -> None:
        while True:
            time.sleep(ORT_IDLE_POLL_SECONDS)
            try:
                release_ort_sessions_if_idle()
            except Exception:
                logger.warning("ORT idle release failed", exc_info=True)
            run_idle_release_hooks()

    threading.Thread(target=_loop, daemon=True, name="ort-idle-release").start()
    logger.info(
        "ORT idle release armed: %.0fs idle, checked every %.0fs",
        ORT_IDLE_RELEASE_SECONDS, ORT_IDLE_POLL_SECONDS,
    )
    return True


def _export_onnx_model(
    run_path: Path,
    model_path: Path,
    onnx_path: Path,
    *,
    num_classes: int,
    run_output_stride: int,
    run_base_channels: int,
    run_arch: str,
    infer_w: int,
    infer_h: int,
) -> Path:
    import torch

    from segcore.training.model import build_model

    model = build_model(
        run_arch,
        num_classes=num_classes,
        output_stride=run_output_stride,
        base_channels=run_base_channels,
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=False)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=f"Model checkpoint incompatible with current architecture. Please retrain. ({e})")
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, infer_h, infer_w)
    # dynamo=False pins the legacy tracer-based exporter. torch>=2.11's
    # default dynamo path logs emoji ('✅') through stdlib logging, which
    # blows up under Windows' cp932 console codec (UnicodeEncodeError).
    # The legacy path also avoids the onnxscript dependency.
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def _ensure_onnx_model(
    run_path: Path,
    model_path: Path,
    *,
    num_classes: int,
    run_output_stride: int,
    run_base_channels: int,
    run_arch: str,
) -> Path:
    infer_w, infer_h = _load_run_input_size(run_path)
    onnx_path = model_path.with_suffix(".onnx")
    model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    onnx_mtime = onnx_path.stat().st_mtime if onnx_path.exists() else -1.0
    if onnx_path.exists() and onnx_mtime >= model_mtime:
        # Re-export if ONNX has fixed H/W axes (pre-dynamic-axes version)
        try:
            import onnx as _onnx
            onnx_model = _onnx.load(str(onnx_path))
            inp = onnx_model.graph.input[0]
            h_dim = inp.type.tensor_type.shape.dim[2]
            if h_dim.dim_param == "":  # fixed, not dynamic
                import logging as _log
                _log.getLogger(__name__).info("Re-exporting ONNX with dynamic H/W axes")
            else:
                return onnx_path
        except Exception:
            return onnx_path
    return _export_onnx_model(
        run_path,
        model_path,
        onnx_path,
        num_classes=num_classes,
        run_output_stride=run_output_stride,
        run_base_channels=run_base_channels,
        run_arch=run_arch,
        infer_w=infer_w,
        infer_h=infer_h,
    )


def _build_ort_session_options(use_cuda: bool):
    import os

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.log_severity_level = 3
    if use_cuda:
        # GPU: parallel graph execution to overlap CPU pre/post with GPU compute.
        # inter_op threads handle graph-level parallelism (CPU pre/post alongside
        # GPU kernels).  Scale modestly with cores but cap at 4 — more threads
        # contend on the GIL and ORT internal locks.
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        opts.inter_op_num_threads = min(4, max(2, (os.cpu_count() or 4) // 4))
        opts.intra_op_num_threads = 1
    else:
        # CPU: use available cores for intra-op parallelism
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        cpu_threads = max(1, (os.cpu_count() or 4) // 2)
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = cpu_threads
    return opts


def _cuda_provider_options(device_id: str, gpu_mem_limit: int | None = None) -> dict[str, str]:
    device_index = 0
    if ":" in device_id:
        try:
            device_index = int(device_id.split(":", 1)[1])
        except (TypeError, ValueError):
            device_index = 0
    opts: dict[str, str] = {
        "device_id": str(device_index),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": "1",
    }
    if gpu_mem_limit is not None and gpu_mem_limit > 0:
        opts["gpu_mem_limit"] = str(gpu_mem_limit)
    return opts


def _resolve_ort_device(requested_device_id: str) -> str:
    """Resolve device ID for ORT, preferring a free GPU when available.

    When ``auto`` or bare ``cuda`` is requested, pick the least-loaded GPU
    that is not currently claimed by a training job so inference doesn't
    collide and silently fall back to CPU.
    """
    value = (requested_device_id or "").strip().lower()
    if value.startswith("cuda") and ":" in value:
        return value  # explicit cuda:N — honour it
    # auto / cuda (no index) — pick the best free GPU
    if value in ("auto", "cuda"):
        try:
            from .torch_device import _device_score, _query_nvidia_smi, active_torch_jobs, list_torch_devices
            devices = list_torch_devices()
            cuda_devices = [d for d in devices if str(d["id"]).startswith("cuda")]
            if cuda_devices:
                busy = set(active_torch_jobs().keys())
                smi = _query_nvidia_smi()
                for d in cuda_devices:
                    smi_info = smi.get(str(d["id"]), {})
                    if smi_info:
                        d["free_mb"] = smi_info.get("free_mb")
                        d["utilization"] = smi_info.get("utilization")
                        d["temperature_c"] = smi_info.get("temperature_c")
                free_devices = [d for d in cuda_devices if str(d["id"]) not in busy]
                pool = free_devices if free_devices else cuda_devices
                ranked = sorted(pool, key=lambda d: _device_score(d, busy, smi), reverse=True)
                if ranked:
                    return str(ranked[0]["id"])
        except Exception:
            pass
        # Fallback: check ORT CUDA availability without torch
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"
    return value


def _preload_cuda_dlls(*, include_ort_capi: bool = True) -> None:
    """Best-effort: put the torch and cuDNN (and optionally ORT capi) DLL
    directories on the Windows DLL search path before creating a CUDA ORT
    session. Safe to call repeatedly; every step is individually guarded."""
    try:
        import torch
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_torch_lib)
                except OSError:
                    pass
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    # cuDNN 8 from nvidia-cudnn-cu11 (ORT 1.18.x needs cuDNN 8)
    try:
        import nvidia.cudnn as _cudnn_pkg
        _cudnn_bin = os.path.join(os.path.dirname(os.path.dirname(_cudnn_pkg.__file__)), "cudnn", "bin")
        if os.path.isdir(_cudnn_bin):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_cudnn_bin)
            os.environ["PATH"] = _cudnn_bin + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass
    if include_ort_capi:
        # Also try the ORT capi dir (DLLs may have been copied there)
        try:
            import onnxruntime as _ort_mod
            _capi_dir = os.path.join(os.path.dirname(_ort_mod.__file__), "capi")
            if os.path.isdir(_capi_dir):
                os.environ["PATH"] = _capi_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass



def _load_ort_session(
    run_path: Path,
    model_path: Path,
    requested_device_id: str,
    *,
    num_classes: int,
    run_output_stride: int,
    run_base_channels: int,
    run_arch: str,
) -> tuple[object, str, str, str]:
    requested_device_id = _resolve_ort_device(requested_device_id)
    if requested_device_id.startswith("cuda"):
        _preload_cuda_dlls()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed") from exc

    onnx_path = _ensure_onnx_model(
        run_path,
        model_path,
        num_classes=num_classes,
        run_output_stride=run_output_stride,
        run_base_channels=run_base_channels,
        run_arch=run_arch,
    )
    onnx_mtime = onnx_path.stat().st_mtime if onnx_path.exists() else 0.0

    provider_key = "cpu"
    if requested_device_id.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        provider_key = requested_device_id

    cache_key = (str(onnx_path), provider_key)
    with _ort_session_cache_guard:
        cached = _ort_session_cache.get(cache_key)
        if cached is not None:
            cached_path, cached_session, cached_mtime, cached_provider = cached
            if cached_path.exists() and cached_mtime == onnx_mtime:
                input_name = cached_session.get_inputs()[0].name
                output_name = cached_session.get_outputs()[0].name
                note_ort_session_use()
                return cached_session, input_name, output_name, cached_provider

    # Only clear torch GPU memory if there are cached torch models to free.
    # Avoid calling torch.cuda.empty_cache() unconditionally — it initializes
    # PyTorch's CUDA context which makes ORT CUDA EP 14x slower on low-VRAM GPUs.
    if requested_device_id.startswith("cuda"):
        with _torch_model_cache_guard:
            has_cached = len(_torch_model_cache) > 0
        if has_cached:
            clear_torch_model_cache()

    use_cuda = provider_key != "cpu"
    providers: list[object]
    if use_cuda:
        providers = [
            ("CUDAExecutionProvider", _cuda_provider_options(provider_key)),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(
            onnx_path.as_posix(),
            sess_options=_build_ort_session_options(use_cuda=use_cuda),
            providers=providers,
        )
    except Exception:
        if use_cuda:
            # Say why. This fallback was silent, and the CPU session it installs
            # is indistinguishable from a machine that never had CUDA: the only
            # trace was `provider=cpu` in a later line. A box whose CUDA EP
            # would not load could only be diagnosed by rebuilding the session
            # by hand from outside the app, because the reason died here.
            logging.getLogger(__name__).warning(
                "ORT CUDA session failed for %s (%s); falling back to CPU. "
                "Inference will be much slower until this is resolved.",
                onnx_path.name, requested_device_id, exc_info=True,
            )
            session = ort.InferenceSession(
                onnx_path.as_posix(),
                sess_options=_build_ort_session_options(use_cuda=False),
                providers=["CPUExecutionProvider"],
            )
            provider_key = "cpu"
        else:
            raise

    actual_provider = "cpu"
    for provider_name in session.get_providers():
        if provider_name == "CUDAExecutionProvider":
            actual_provider = provider_key
            break
        if provider_name == "CPUExecutionProvider":
            actual_provider = "cpu"

    with _ort_session_cache_guard:
        if len(_ort_session_cache) >= 4:
            oldest_keys = _ort_session_cache.keys()
            if oldest_keys:
                evicted = _ort_session_cache.pop(oldest_keys[0])
                if evicted is not None:
                    del evicted
        _ort_session_cache.put(cache_key, (onnx_path, session, onnx_mtime, actual_provider))
    note_ort_session_use()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name, actual_provider


def _load_torch_model(
    run_path: Path,
    model_path: Path,
    device_id: str,
    *,
    num_classes: int,
    run_output_stride: int,
    run_base_channels: int,
    run_arch: str,
):
    import torch

    from segcore.training.model import build_model

    if not model_path.exists():
        raise PredictModelMissingError(detail=f"path={model_path}")
    cache_key = (str(model_path), device_id)
    current_mtime = model_path.stat().st_mtime
    with _torch_model_cache_guard:
        cached = _torch_model_cache.get(cache_key)
        if cached is not None:
            cached_path, cached_model, cached_mtime = cached
            if cached_path.exists() and cached_mtime == current_mtime:
                return cached_model

    device = torch.device(device_id)
    model = build_model(
        run_arch,
        num_classes=num_classes,
        output_stride=run_output_stride,
        base_channels=run_base_channels,
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=f"Model checkpoint incompatible with current architecture. Please retrain. ({e})")
    model.to(device)
    model.eval()
    # torch.compile for faster inference (PyTorch 2.x)
    if device.type == "cuda":
        # torch.compile: skip on Windows (Triton unavailable)
        import sys
        if sys.platform != "win32":
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass
        # CUDA warmup: triggers kernel compilation, subsequent calls are faster
        try:
            with torch.inference_mode():
                _dummy = torch.zeros(1, 3, 256, 256, device=device)
                model(_dummy)
                del _dummy
                torch.cuda.synchronize(device)
        except Exception:
            pass
    with _torch_model_cache_guard:
        # Evict oldest entry if at capacity, freeing GPU memory
        if len(_torch_model_cache) >= 4:
            oldest_keys = _torch_model_cache.keys()
            if oldest_keys:
                evicted = _torch_model_cache.pop(oldest_keys[0])
                if evicted is not None:
                    del evicted
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
        _torch_model_cache.put(cache_key, (model_path, model, current_mtime))
    return model
