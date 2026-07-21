# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Training subprocess worker.

Extracted verbatim from training_runner.py during the pre-OSS refactor.
_train_subprocess_worker runs in a child process (multiprocessing spawn
pickles it by module+name, so it must stay importable at this path).
"""
from __future__ import annotations

_TRAIN_EXIT_OK = 0
_TRAIN_EXIT_ERROR = 1
_TRAIN_EXIT_OOM = 2


def _train_subprocess_worker(
    prepared_dir_str: str,
    run_dir_str: str,
    num_classes: int,
    config_kwargs: dict,
    logs_path_str: str,
    stop_file_str: str,
) -> None:
    """Training worker running in a child process.
    CUDA crashes here do NOT kill the API process.
    Exit codes: 0=success, 1=error, 2=OOM (retryable)."""
    import os as _os
    import signal as _signal
    import sys as _sys
    from pathlib import Path as _Path

    # Ignore Ctrl+C in training subprocess — parent manages lifecycle via stop_file.
    # Windows: prevents console event propagation; macOS/Linux: prevents terminal SIGINT.
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

    # Ensure packages/ is on sys.path for embedded Python (multiprocessing.spawn
    # doesn't inherit ._pth entries). Find project root from this file's location.
    _this_file = _Path(__file__).resolve()
    _project_root = _this_file.parent.parent.parent.parent.parent  # core -> app -> trainer_api -> apps -> root
    _packages_dir = _project_root / "packages" / "segcore"
    for _p in [str(_project_root), str(_packages_dir)]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    # Debug: dump sys.path and env to stderr for diagnostics
    _sys.stderr.write(f"[train-subprocess] __file__={__file__}\n")
    _sys.stderr.write(f"[train-subprocess] _project_root={_project_root}\n")
    _sys.stderr.write(f"[train-subprocess] sys.path={_sys.path[:5]}\n")
    _sys.stderr.write(f"[train-subprocess] PYTHONPATH={_os.environ.get('PYTHONPATH', '(unset)')}\n")
    _sys.stderr.flush()

    def log_fn(line: str) -> None:
        with open(logs_path_str, "a", encoding="utf-8") as fh:
            fh.write(line)
        # Also echo to stderr for debugging
        _sys.stderr.write(f"[train-log] {line}")
        _sys.stderr.flush()

    def is_stopped() -> bool:
        return _Path(stop_file_str).exists()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as _e:
        _sys.stderr.write(f"[train-subprocess] torch init warning: {_e}\n")
        _sys.stderr.flush()

    try:
        _sys.stderr.write("[train-subprocess] Importing segcore.training.train...\n")
        _sys.stderr.flush()
        from segcore.training.train import TrainConfig
        from segcore.training.train import train as train_pipeline
        # Extract non-constructor kwargs before creating TrainConfig
        teacher_model_dir = config_kwargs.pop("distill_teacher_model_dir", None)
        tc = TrainConfig(**config_kwargs)
        if teacher_model_dir:
            tc.distill_teacher_model_dir = teacher_model_dir
        train_pipeline(
            prepared_dir=_Path(prepared_dir_str),
            run_dir=_Path(run_dir_str),
            num_classes=num_classes,
            config=tc,
            log_fn=log_fn,
            stop_flag=is_stopped,
        )
    except RuntimeError as exc:
        import traceback as _tb
        msg = str(exc).lower()
        is_oom = "out of memory" in msg or ("cuda" in msg and "alloc" in msg)
        tb_str = _tb.format_exc()
        # Write error + traceback in single call to avoid partial output
        log_fn(
            f"Training {'OOM' if is_oom else 'error'}: {exc}\n"
            f"--- Traceback ---\n{tb_str}\n--- End Traceback ---\n"
        )
        _sys.exit(_TRAIN_EXIT_OOM if is_oom else _TRAIN_EXIT_ERROR)
    except BaseException as exc:
        # Catch ALL exceptions including SystemExit, KeyboardInterrupt,
        # and any other BaseException that would otherwise kill the
        # subprocess silently without writing to the log.
        import traceback as _tb
        try:
            log_fn(f"Training failed (subprocess): {type(exc).__name__}: {exc}\n")
            log_fn(_tb.format_exc() + "\n")
        except Exception:
            # Last resort: write to stderr so it's not completely silent
            _sys.stderr.write(f"Training failed (log_fn also failed): {exc}\n")
            _tb.print_exc(file=_sys.stderr)
        _sys.exit(_TRAIN_EXIT_ERROR)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _sys.exit(_TRAIN_EXIT_OK)


