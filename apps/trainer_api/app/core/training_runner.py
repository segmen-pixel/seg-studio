# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from sqlmodel import Session, select

from ..db import get_engine
from ..models import TrainingRun
from . import state as _state
from .auto_select_utils import (
    build_query_profile,  # noqa: F401 — re-exported; routers import it from here
)
from .auto_select_utils import (  # noqa: F401 — re-export for backward compat
    compute_basic_stats_fallback as _compute_basic_stats_fallback,
)
from .config import (
    AUTO_VAL_MIN_COUNT,  # noqa: F401
    AUTO_VAL_TARGET_RATIO,  # noqa: F401
    FIXED_INPUT_SIZE,  # noqa: F401
    IGNORE_INDEX,  # noqa: F401
    NORMALIZE,  # noqa: F401
    OUTPUT_STRIDE,  # noqa: F401
    PKG_DIR,  # noqa: F401
    ROOT_DIR,  # noqa: F401
    TRAINER_BUILD_ID,
    read_class_ids,  # noqa: F401
    read_num_classes,  # noqa: F401
)
from .dataset_prep import (
    estimate_foreground_ratio,  # noqa: F401 — re-export for backward compat
    prepare_annotate_dataset,  # noqa: F401 — re-export for backward compat
    prepare_dataset,  # noqa: F401
    rebalance_train_val_ids,  # noqa: F401
)
from .db_utils import log_action, touch_project
from .iter_chain import (  # noqa: F401 — re-export
    _maybe_launch_next_iteration,
    _summarize_iter_chain,
)
from .paths import run_dir, write_json
from .torch_device import (
    _batch_limit_for_input,  # noqa: F401
    _build_oom_retry_config,  # noqa: F401
    _clear_cuda_cache,  # noqa: F401
    _cuda_free_memory_mb,  # noqa: F401
    _cuda_total_memory_mb,  # noqa: F401
    _is_cuda_oom_error,
    _max_input_for_memory,  # noqa: F401
    _patches_limit_for_memory,  # noqa: F401
    _profile_max_batch_size,  # noqa: F401
    claim_torch_device,  # noqa: F401
    current_configured_torch_device,
    release_torch_device,
    resolve_torch_device_or_cpu,  # noqa: F401
    touch_torch_device_claim,  # noqa: F401
)
from .training_job_phases import (
    apply_auto_select_and_config,
    cap_batch_for_vram,
    resolve_device_and_distill,
    run_training_subprocess,
)
from .training_launcher import (  # noqa: F401 — re-export (routers/startup import these)
    _launch_single_run,
    _launch_training_run,
    _release_gpu_caches,
    _start_next_reserved,
    get_queue_positions,
)
from .training_modes import dataset_dir_for, prepare_for, resolve, run_post_recipe
from .training_workers import (  # noqa: F401 — re-export
    _TRAIN_EXIT_ERROR,
    _TRAIN_EXIT_OK,
    _TRAIN_EXIT_OOM,
    _train_subprocess_worker,
)


def run_training_job(project_id: str, run_id: str, config: dict, stop_event: threading.Event) -> None:
    import logging as _logging_job
    _job_logger = _logging_job.getLogger(__name__)
    _job_logger.info("[run_training_job] ENTER project=%s run=%s", project_id[:8], run_id[:8])

    run_path = run_dir(project_id, run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    logs_path = run_path / "train.log"
    # Which dataset a run reads, and which phase builds it, are facts about
    # the mode; the registry holds them so the two cannot disagree.
    profile = resolve(config)
    prepared_dir = dataset_dir_for(profile, project_id)

    # Truncate log at start of each run to prevent duplication when a run_id
    # is re-entered (retry, restart, etc.). Without truncation, stale content
    # from a prior run attempt would be streamed to the UI alongside the new
    # run's output.
    try:
        logs_path.open("w", encoding="utf-8").close()
    except OSError:
        pass

    _log_unwritable = False

    def log_fn(line: str) -> None:
        nonlocal _log_unwritable
        try:
            with logs_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            # The run directory can be removed while the run is still going
            # (delete_run, delete_project). A lost log line is not a training
            # failure, and letting it raise took the supervision loop down
            # with it. Narrow to OSError, and say it once: a real disk or
            # permission problem must not silently turn a failed run into a
            # completed one.
            if not _log_unwritable:
                _log_unwritable = True
                _job_logger.warning(
                    "run %s: run log %s is no longer writable: %s",
                    run_id[:8], logs_path, exc,
                )

    try:
        # Release SAM GPU memory before training to avoid VRAM conflicts
        _job_logger.info("[run_training_job] Releasing SAM GPU...")
        try:
            from .sam_assist import _sam_release_gpu
            _sam_release_gpu()
            _job_logger.info("[run_training_job] SAM GPU released")
        except Exception as sam_err:
            _job_logger.warning("[run_training_job] SAM release failed: %s", sam_err)

        log_fn(f"Trainer build: {TRAINER_BUILD_ID}\n")
        log_fn("Training started.\n")
        _job_logger.info("[run_training_job] Training started, entering pipeline...")

        if profile.pipeline == "instance":
            # Instance segmentation: synthesis + rfdetr; no semantic prepare,
            # auto-select, or VRAM cap phases (design_instance_segmentation_v098).
            from .instance_training import run_instance_phases
            run_instance_phases(
                project_id, run_id, run_path, logs_path, config, stop_event, log_fn,
            )
        else:
            prep = prepare_for(profile)(
                project_id, run_id, config, prepared_dir, log_fn,
            )
            pretrained_checkpoint = apply_auto_select_and_config(
                project_id, config, prepared_dir, run_path,
                prep.pretrained_checkpoint, log_fn,
            )
            # Automatic configuration has had its say and has already
            # persisted train_config.json. A mode whose identity lives in
            # those same values gets the last word here -- and the file has
            # to be rewritten afterwards, because train_config.json is what
            # inference reads back to rebuild the network. Left unwritten,
            # a position run would train with the coordinate channels and
            # then be served as the architecture the recipe had picked.
            run_post_recipe(profile, config, log_fn)
            if profile.post_recipe:
                try:
                    write_json(run_path / "train_config.json", config)
                except OSError as err:
                    log_fn(f"[{profile.id}] could not persist the corrected "
                           f"train_config.json: {err}\n")
            train_output_stride, resolved_device, memory_mb = resolve_device_and_distill(
                config, log_fn,
            )
            attempt_config = cap_batch_for_vram(
                config, resolved_device, train_output_stride, memory_mb,
                prep.num_classes, log_fn,
            )
            run_training_subprocess(
                run_id, run_path, logs_path, prepared_dir, prep, attempt_config,
                train_output_stride, resolved_device, pretrained_checkpoint,
                stop_event, log_fn,
            )
    except Exception as exc:
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        from . import error_codes as _EC
        error_code = _EC.TRAIN_OOM if _is_cuda_oom_error(exc) else _EC.TRAIN_SUBPROCESS_CRASH
        detail = str(exc)
        if _is_cuda_oom_error(exc):
            detail += (
                " | hint: reduce Batch, Input size, or Patches/Image"
            )
        try:
            log_fn(f"[{error_code}] Training failed: {detail}\n")
        except Exception as log_exc:
            _logger.error("[%s] Failed to write training error to log file: %s (original error: %s)", error_code, log_exc, detail)
        _logger.error("[%s] Training failed for run %s: %s", error_code, run_id, detail, exc_info=True)
        engine = get_engine()
        with Session(engine) as session:
            record = session.exec(
                select(TrainingRun).where(TrainingRun.project_id == project_id, TrainingRun.run_id == run_id)
            ).first()
            if record and record.status == "running":
                record.status = "failed"
                record.updated_at = datetime.now(timezone.utc)
                session.add(record)
                log_action(session, "train_failed", "run", run_id)
                session.commit()
        # NOTE: _start_next_reserved moved to finally block (after device release)
        return
    finally:
        try:
            release_torch_device(
                str(config.get("resolved_torch_device") or config.get("torch_device", current_configured_torch_device())),
                owner_id=run_id,
            )
        except Exception:
            pass
        # Always clean up RUN_FLAGS to prevent memory leak
        _state.RUN_FLAGS.pop(run_id, None)
        # Wait for GPU memory to be fully released before launching next run.
        # On Windows, CreateProcess can fail with [WinError 8] if CUDA memory
        # from the previous subprocess hasn't been reclaimed by the driver yet.
        try:
            _clear_cuda_cache()
            import gc
            gc.collect()
            time.sleep(3)
        except Exception:
            pass
        # Launch next reserved run AFTER device release
        try:
            _start_next_reserved(project_id)
        except Exception:
            _job_logger.exception("Failed to launch next reserved run after device release")

    log_fn("[POST] Updating DB status...\n")
    _job_logger.info("[run_training_job] POST: updating DB status")
    if stop_event.is_set():
        status = "stopped"
        action = "train_stop"
    else:
        status = "completed"
        action = "train_complete"
    for attempt in range(3):
        try:
            engine = get_engine()
            with Session(engine) as session:
                record = session.exec(
                    select(TrainingRun).where(TrainingRun.project_id == project_id, TrainingRun.run_id == run_id)
                ).first()
                if record and record.status == "running":
                    record.status = status
                    record.updated_at = datetime.now(timezone.utc)
                    session.add(record)
                    log_action(session, action, "run", run_id)
                    session.commit()
            touch_project(project_id)
            break
        except Exception as exc:
            if attempt < 2:
                _job_logger.debug("Status update attempt %d for run %s failed: %s", attempt + 1, run_id, exc)
                time.sleep(0.5)
            else:
                _job_logger.error(
                    "Failed to update training status to %s for run %s after 3 attempts",
                    status,
                    run_id,
                    exc_info=True,
                )
    # NOTE: _start_next_reserved is called in the finally block after device release
    # Iterative hard-mining fanout: if this run was in iterative mode and
    # train.py flagged iterative_hard_ids.json, chain the next iteration.
    if not stop_event.is_set() and status == "completed":
        try:
            _maybe_launch_next_iteration(project_id, run_id, run_path, config, log_fn)
        except Exception as _iter_err:
            _job_logger.warning(
                "[run_training_job] iterative next-launch skipped: %s", _iter_err
            )
    log_fn("[POST] All done.\n")
    _job_logger.info("[run_training_job] POST: all done")
