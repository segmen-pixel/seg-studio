# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Startup maintenance tasks: project/DB cleanup, dependency and health
checks, UI auto-build and the deferred post-ready scans.

Extracted verbatim from main.py during the pre-OSS refactor; main
re-exports the names for backward compatibility (tests import them).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .config import PROJECTS_DIR, ROOT_DIR

logger = logging.getLogger("trainer_api")


def _cleanup_orphan_project_dirs() -> None:
    """Reconcile PROJECTS_DIR with the DB.

    For project-named dirs not present in the DB:
    - If a `.deleted` tombstone exists, the dir is the remains of a partial
      delete (Windows file lock etc.) -- remove it, don't resurrect.
    - If neither project.json nor any actual content (images/masks/index/training)
      exists, the dir is a stub from a failed create -- remove it too.
    - Otherwise, adopt the dir into the DB so externally imported projects
      aren't lost.
    """
    import json
    import shutil as _shutil
    from datetime import datetime, timezone

    from sqlmodel import Session, select

    from ..db import get_engine
    from ..models import Project
    from .paths import RUNS_DIRNAME, is_project_dir_name
    engine = get_engine()
    with Session(engine) as session:
        db_ids = {row.id for row in session.exec(select(Project)).all()}
    adopted: list[str] = []
    purged: list[str] = []
    for entry in PROJECTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        # An allow-list, because the rmtree below is what happens to anything
        # that gets past here without a database row. .library, .gpu_locks and
        # hand-made directories must never reach it.
        if not is_project_dir_name(entry.name):
            continue
        if entry.name in db_ids:
            continue
        pj_path = entry / "project.json"
        tombstone = entry / ".deleted"
        has_content = any(
            (entry / sub).exists()
            for sub in ("images", "masks", "index.json", "training", RUNS_DIRNAME)
        )
        if tombstone.exists() or (not pj_path.exists() and not has_content):
            # Remnant from a partial delete or a failed create -- drop it.
            _shutil.rmtree(entry, ignore_errors=True)
            purged.append(entry.name[:8])
            continue
        # Genuine externally-imported project -- adopt it.
        name = entry.name[:8]
        description = ""
        now = datetime.now(timezone.utc)
        if pj_path.exists():
            try:
                pj = json.loads(pj_path.read_text(encoding="utf-8"))
                name = pj.get("name", name)
                description = pj.get("description", "")
            except Exception as e:
                logger.debug("Failed to read project.json for %s: %s", entry.name[:8], e)
        with Session(engine) as session:
            project = Project(
                id=entry.name, name=name, description=description,
                created_at=now, updated_at=now,
            )
            session.add(project)
            session.commit()
        adopted.append(f"{entry.name[:8]}({name})")
    if adopted:
        logger.info("Adopted %d orphan project dirs into DB: %s", len(adopted), adopted)
    if purged:
        logger.info("Purged %d remnant project dir(s): %s", len(purged), purged)


def _cleanup_false_ok_masks() -> None:
    """Startup data integrity checks for all projects.

    Fixes:
    1. False-OK masks: all-zero mask PNG from cache bug → delete + reset index
    2. Ghost index entries: index references non-existent image file → remove entry
    3. Orphan masks: mask file with no matching index entry → delete
    4. Orphan thumbnails: thumbnail with no matching image → delete
    5. Index hasMask mismatch: index says hasMask but file missing → fix flag
    """
    from .annotate_index import load_annotate_index, save_annotate_index
    total_fixes = 0
    try:
        for proj_path in PROJECTS_DIR.iterdir():
            if not proj_path.is_dir():
                continue
            images_dir = proj_path / "images"
            masks_dir = proj_path / "masks"
            thumbs_dir = proj_path / ".cache" / "thumbnails"
            idx_path = proj_path / "index.json"
            if not idx_path.exists():
                continue
            pid = proj_path.name
            index = load_annotate_index(pid)
            items = index.get("items", [])
            dirty = False

            # Build lookup sets
            indexed_ids = {item["id"] for item in items if "id" in item}
            indexed_filenames = {item.get("filename", "") for item in items}

            # 1. False-OK masks (all-zero mask files)
            # 2. Index hasMask but mask file missing
            for item in items:
                ann = item.get("annotation") or {}
                iid = item.get("id", "")
                mask_file = masks_dir / f"{iid}.png" if masks_dir.exists() else None

                if ann.get("hasMask") and mask_file and not mask_file.exists():
                    # Fix: index says hasMask but file is gone
                    item["annotation"] = {**ann, "hasMask": False, "hasForeground": False, "classIds": []}
                    dirty = True
                    total_fixes += 1

                # NOTE: Previously deleted all-zero masks assuming they were
                # cache artefacts ("false OK").  However, mark-clean intentionally
                # creates all-zero masks to represent "verified no defects".
                # Deleting them on startup erased legitimate OK labels.
                # Removed: the false-OK cleanup.  All-zero masks are now valid.

            # 3. Ghost index entries (image file missing)
            if images_dir.exists():
                valid_items = []
                for item in items:
                    img_file = images_dir / item.get("filename", "MISSING")
                    if img_file.exists():
                        valid_items.append(item)
                    else:
                        total_fixes += 1
                        dirty = True
                if len(valid_items) != len(items):
                    items = valid_items

            # 4. Orphan masks (no index entry)
            if masks_dir.exists():
                for mask_file in masks_dir.iterdir():
                    if mask_file.suffix.lower() != ".png":
                        continue
                    mask_id = mask_file.stem
                    if mask_id not in indexed_ids:
                        mask_file.unlink()
                        total_fixes += 1

            # 5. Orphan thumbnails
            if thumbs_dir.exists():
                for thumb_file in thumbs_dir.iterdir():
                    thumb_stem = thumb_file.stem
                    if not any(thumb_stem in fn for fn in indexed_filenames) and thumb_stem not in indexed_ids:
                        thumb_file.unlink()
                        total_fixes += 1

            if dirty:
                index["items"] = items
                save_annotate_index(pid, index)

        if total_fixes > 0:
            logger.info("Startup cleanup: %d data issues fixed across all projects", total_fixes)
    except Exception as exc:
        logger.warning("Startup cleanup failed: %s", exc)


def _cleanup_stale_runs_on_startup() -> None:
    """DB integrity check on startup: fix stale status, remove orphans."""
    import shutil

    from sqlmodel import Session, select

    from ..db import get_engine
    from ..models import ModelRecord, Project, TrainingRun
    from .paths import runs_root

    engine = get_engine()
    with Session(engine) as session:
        # 1. Mark stale 'running' records as 'failed' (no threads survive restart)
        stale = session.exec(
            select(TrainingRun).where(TrainingRun.status == "running")
        ).all()
        if stale:
            for record in stale:
                # updated_at is deliberately left alone. The run stopped when
                # the process died, not when the next one started, and the run
                # list sorts on this column -- stamping it here floated a run
                # that failed days ago above today's finished ones every time
                # the application restarted.
                record.status = "failed"
                session.add(record)
            session.commit()
            logger.info("Cleaned up %d stale 'running' records", len(stale))

        # 2. Remove DB records whose run directory no longer exists
        all_runs = session.exec(select(TrainingRun)).all()
        orphan_db = []
        for record in all_runs:
            # runs_root() resolves through project_dir(), which migrates the
            # layout first. Spelling the path here instead is what made this
            # loop delete every run row the moment the directory moved.
            proj_path = runs_root(record.project_id) / record.run_id
            if not proj_path.is_dir():
                orphan_db.append(record)
        if orphan_db:
            for record in orphan_db:
                # Also remove related ModelRecords
                related = session.exec(
                    select(ModelRecord).where(ModelRecord.run_id == record.run_id)
                ).all()
                for m in related:
                    session.delete(m)
                session.delete(record)
            session.commit()
            logger.info(
                "Removed %d orphan DB records (no run directory): %s",
                len(orphan_db),
                ", ".join(r.run_id[:8] for r in orphan_db),
            )

        # 3. Remove orphan run directories with no DB record and no model file
        #    (directories with model.pt are kept — discoverable via
        #    _discover_fs_runs and usable from the UI. anomaly_model.pkl is a
        #    legacy artifact of the removed anomaly mode: such dirs are kept
        #    on disk as user data but are no longer surfaced in the UI.)
        known_run_ids = {r.run_id for r in session.exec(select(TrainingRun)).all()}
        known_project_ids = {p.id for p in session.exec(select(Project)).all()}
        orphan_dirs_removed = 0
        for pid in known_project_ids:
            runs_path = runs_root(pid)
            if not runs_path.is_dir():
                continue
            for entry in runs_path.iterdir():
                if not entry.is_dir() or entry.name in known_run_ids:
                    continue
                has_model = (entry / "model.pt").exists() or (entry / "anomaly_model.pkl").exists()
                if has_model:
                    continue  # keep — usable from UI
                shutil.rmtree(entry, ignore_errors=True)
                orphan_dirs_removed += 1
                logger.info("Removed orphan run directory (no model): %s/%s", pid[:8], entry.name[:8])
        if orphan_dirs_removed:
            logger.info("Removed %d orphan run directories total", orphan_dirs_removed)

        # 4. Check for reserved runs that were queued before restart
        has_reserved = bool(session.exec(
            select(TrainingRun).where(TrainingRun.status == "reserved")
        ).first())
    if has_reserved:
        logger.info("Found reserved runs after startup, attempting to launch...")
        try:
            from .training_runner import _start_next_reserved
            _start_next_reserved()
        except Exception:
            logger.exception("Failed to launch reserved runs on startup")


def _run_health_check() -> None:
    """Check libraries, SAM checkpoints, and GPU.

    Uses sys.modules to avoid redundant imports (routers already loaded them).
    """
    logger.info("-- Health Check --")

    # -- Libraries (check sys.modules first, only import if not yet loaded) --
    libs = [
        ("torch",        "torch",        "Training / inference"),
        ("torchvision",  "torchvision",  "Data augmentation"),
        ("cv2",          "cv2",          "Image processing"),
        ("sklearn",      "sklearn",      "MLP Assist CPU fallback"),
        ("skimage",      "skimage",      "Superpixel (SLIC)"),
        ("mobile_sam",   "mobile_sam",   "MobileSAM"),
        ("sam2",         "sam2",         "SAM2"),
        ("coremltools",  "coremltools",  "CoreML export"),
        ("transformers", "transformers", "Knowledge distillation"),
    ]
    for display_name, pkg, purpose in libs:
        mod = sys.modules.get(pkg)
        if mod is not None:
            ver = getattr(mod, "__version__", "-")
            logger.info("  lib %-16s %-20s OK", display_name, ver)
        else:
            # Not yet imported --check if importable without loading
            import importlib.util
            if importlib.util.find_spec(pkg) is not None:
                logger.info("  lib %-16s %-20s available", display_name, "(not loaded)")
            else:
                logger.warning("  lib %-16s %-20s MISSING (%s disabled)", display_name, "--", purpose)

    # -- SAM Checkpoints --
    sam_dir = ROOT_DIR / "models" / "sam_checkpoints"
    checkpoints = [
        ("mobile_sam.pt",          "MobileSAM"),
        ("sam2.1_hiera_tiny.pt",   "SAM2 Tiny"),
        ("sam2.1_hiera_small.pt",  "SAM2 Small"),
    ]
    for fname, label in checkpoints:
        fpath = sam_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            logger.info("  ckpt %-28s %7.1f MB OK", fname, size_mb)
        else:
            logger.warning("  ckpt %-28s MISSING (%s)", fname, label)

    # -- Device (torch already imported by routers) --
    _torch = sys.modules.get("torch")
    if _torch is not None:
        if _torch.cuda.is_available():
            name = _torch.cuda.get_device_name(0)
            vram = _torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            logger.info("  device CUDA: %s (%.0f MB)", name, vram)
        elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            logger.info("  device MPS: available")
        else:
            logger.info("  device CPU only")
    else:
        logger.info("  device CPU only (torch not loaded)")

    logger.info("-- Health Check Complete --")


def _is_packaged_build() -> bool:
    """Detect if running from an installed (packaged) build."""
    return (ROOT_DIR / "python" / "python.exe").exists()


def _auto_check_deps() -> None:
    """Check critical Python packages and node_modules; install if missing.

    In packaged builds, pip/setuptools are stripped, so skip self-heal and
    only report missing packages as errors.
    """
    import importlib.util
    import subprocess

    packaged = _is_packaged_build()

    # --- Python packages ---
    critical = ["torch", "cv2", "numpy", "PIL", "sqlmodel", "pydantic", "sklearn"]
    missing = [pkg for pkg in critical if importlib.util.find_spec(pkg) is None]

    if missing:
        if packaged:
            logger.error("Missing Python packages in packaged build: %s "
                         "--reinstall the application to fix this", missing)
        else:
            logger.warning("Missing Python packages: %s --running pip install", missing)
            req_file = ROOT_DIR / "apps" / "trainer_api" / "requirements.txt"
            if req_file.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                    capture_output=True, text=True, timeout=600,
                    encoding="utf-8", errors="replace",
                )
                if result.returncode == 0:
                    logger.info("pip install OK")
                else:
                    logger.error("pip install failed (rc=%d): %s", result.returncode,
                                 result.stderr[-500:] if result.stderr else "")
            else:
                logger.warning("requirements.txt not found at %s", req_file)
    else:
        logger.info("All critical Python packages present")

    # --- Node modules (for UI build, dev only) ---
    if packaged:
        return  # Packaged builds ship pre-built UI, no npm needed

    ui_dir = ROOT_DIR / "apps" / "trainer_ui"
    node_modules = ui_dir / "node_modules"
    pkg_json = ui_dir / "package.json"

    if pkg_json.exists() and not node_modules.exists():
        logger.info("node_modules missing --running npm install")
        import shutil
        npm_cmd = shutil.which("npm") or "npm"
        result = subprocess.run(
            [npm_cmd, "install"],
            cwd=str(ui_dir),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=300, shell=False,
        )
        if result.returncode == 0:
            logger.info("npm install OK")
        else:
            logger.warning("npm install failed (rc=%d): %s", result.returncode,
                           result.stderr[-500:] if result.stderr else "")


def _auto_build_ui() -> None:
    """Rebuild trainer_ui if dist/ is missing or stale (source newer than build)."""
    import subprocess
    ui_dir = ROOT_DIR / "apps" / "trainer_ui"
    dist_dir = ui_dir / "dist"
    marker = dist_dir / "index.html"
    src_dir = ui_dir / "src"

    needs_build = False
    if not marker.exists():
        needs_build = True
        logger.info("UI dist/ not found --will build")
    elif src_dir.exists():
        # Check if any source file is newer than the built index.html
        build_mtime = marker.stat().st_mtime
        for f in src_dir.rglob("*"):
            if f.is_file() and f.stat().st_mtime > build_mtime:
                needs_build = True
                logger.info("UI source newer than dist/ --will rebuild")
                break

    if not needs_build:
        return

    pkg_json = ui_dir / "package.json"
    if not pkg_json.exists():
        logger.warning("UI package.json not found, skipping build")
        return

    logger.info("Building UI (npm run build)...")
    try:
        import shutil
        npm_cmd = shutil.which("npm") or "npm"
        # encoding is explicit because text=True otherwise decodes with the
        # console code page. On a Japanese Windows install that is cp932, and
        # vite alone is enough to break it: the check mark in "vite built in
        # 1.53s" is UTF-8 E2 9C 93, and the 0x93 raises UnicodeDecodeError
        # inside subprocess's reader thread. The exception never reaches this
        # frame -- the output simply arrives empty, so a failing UI build was
        # logged with an empty reason.
        result = subprocess.run(
            [npm_cmd, "run", "build"],
            cwd=str(ui_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            shell=False,
        )
        if result.returncode == 0:
            logger.info("UI build OK")
        else:
            logger.warning("UI build failed (rc=%d): %s", result.returncode, result.stderr[-500:] if result.stderr else "")
    except Exception as exc:
        logger.warning("UI build error: %s", exc)


def _check_path_budget(startup_state: dict) -> None:
    """Warn when this install sits too deep for the files it will write.

    Windows refuses to create a path past MAX_PATH, and the refusal surfaces
    tens of minutes into a training run as a FileNotFoundError naming a
    prediction the user never asked about. The arithmetic is knowable at boot,
    so say it at boot instead.

    Reported as an error rather than a warning on purpose: once the warning
    panel has been dismissed the UI shows only errors for the rest of the
    session, and a machine deep enough to trip this is exactly the machine
    already showing other warnings.
    """
    from .paths import (
        WINDOWS_MAX_PATH,
        artifact_path_length,
        is_project_dir_name,
        runs_root_of,
    )

    over: list[tuple[str, int]] = []
    fresh = artifact_path_length()
    if PROJECTS_DIR.is_dir():
        for entry in PROJECTS_DIR.iterdir():
            if not entry.is_dir() or not is_project_dir_name(entry.name):
                continue
            longest_stem = 0
            images = entry / "images"
            if images.is_dir():
                try:
                    with os.scandir(images) as it:
                        for f in it:
                            stem_len = len(f.name.rsplit(".", 1)[0])
                            longest_stem = max(longest_stem, stem_len)
                except OSError:
                    continue
            runs = runs_root_of(entry)
            longest_run = 0
            if runs.is_dir():
                try:
                    with os.scandir(runs) as it:
                        longest_run = max((len(r.name) for r in it if r.is_dir()),
                                          default=0)
                except OSError:
                    longest_run = 0
            worst = artifact_path_length(
                longest_stem or None,
                project_id_len=len(entry.name),
                run_id_len=longest_run or None,
            )
            if worst > WINDOWS_MAX_PATH:
                over.append((entry.name, worst))

    if not over and fresh <= WINDOWS_MAX_PATH:
        return

    over.sort(key=lambda pair: -pair[1])
    root_is_too_deep = fresh > WINDOWS_MAX_PATH
    lines: list[str] = []
    if root_is_too_deep:
        title = "Installation path is too deep"
        lines += [
            f"seg-studio is installed at {ROOT_DIR}, which does not leave "
            f"enough room for the files it writes underneath.",
            "",
            f"Windows rejects paths longer than {WINDOWS_MAX_PATH} characters, "
            f"and a newly created project here already needs up to {fresh}. "
            "Saving predictions and heatmaps will fail.",
        ]
    else:
        title = "Some projects exceed the Windows path limit"
        lines += [
            f"Windows rejects paths longer than {WINDOWS_MAX_PATH} characters. "
            "New projects fit here, but some existing ones do not, because "
            "they hold images with very long filenames.",
        ]
    if over:
        shown = ", ".join(f"{name[:8]} ({length})" for name, length in over[:5])
        more = f", and {len(over) - 5} more" if len(over) > 5 else ""
        lines += [
            "",
            f"Over the limit: {shown}{more}. Predictions and heatmaps for "
            "their longest-named images cannot be saved. Shortening those "
            "filenames, or moving the installation, fixes it.",
        ]
    lines += [
        "",
        "Fix: move the installation closer to the drive root, for example "
        "C:\\seg-studio.",
    ]
    startup_state["warnings"].append({
        "level": "error",
        "title": title,
        "message": "\n".join(lines),
    })
    logger.error(
        "Path budget exceeded: root=%s fresh=%d limit=%d over=%d project(s)",
        ROOT_DIR, fresh, WINDOWS_MAX_PATH, len(over),
    )


def _check_inference_deps(resolved_device: str, startup_state: dict) -> None:
    """Check inference dependencies and add warnings if anything is missing."""

    # --- SAM packages ---
    sam_status: list[tuple[str, bool]] = []
    for pkg_name, import_name in [
        ("mobile_sam", "mobile_sam"),
        ("sam2", "sam2"),
        ("timm", "timm"),
        ("efficient_sam", "efficient_sam"),
    ]:
        try:
            __import__(import_name)
            sam_status.append((pkg_name, True))
        except ImportError:
            sam_status.append((pkg_name, False))
    missing_sam = [name for name, ok in sam_status if not ok]
    if missing_sam:
        startup_state["warnings"].append({
            "level": "warning",
            "title": "SAM packages missing",
            "message": (
                f"The following SAM packages are not installed: {', '.join(missing_sam)}\n"
                "SAM assist (click-to-segment) will not work.\n\n"
                "Fix: run `python scripts/install.py`."
            ),
        })
        logger.warning("Missing SAM packages: %s", ", ".join(missing_sam))
    else:
        logger.info("SAM packages OK: %s", ", ".join(n for n, _ in sam_status))

    # --- SAM checkpoints ---
    from .sam_assist import _MODELS_DIR, _SAM_CHECKPOINTS
    missing_ckpt = [
        name for name, fname in _SAM_CHECKPOINTS.items()
        if not (_MODELS_DIR / fname).exists()
    ]
    if missing_ckpt:
        logger.info("SAM checkpoints not yet downloaded: %s (will auto-download on first use)", ", ".join(missing_ckpt))
    else:
        logger.info("SAM checkpoints OK: %d models ready", len(_SAM_CHECKPOINTS))

    is_cuda = resolved_device.startswith("cuda")
    if not is_cuda:
        return  # CPU-only setup --nothing more to warn about

    # Check ORT CUDA EP --must actually load the CUDA provider DLL,
    # not just check get_available_providers() (which may list CUDA
    # even when the DLL dependencies like cuDNN are missing).
    ort_cuda = False
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            # Try to actually create a tiny session with CUDA to verify DLLs load
            try:
                _test_opts = ort.SessionOptions()
                _test_opts.log_severity_level = 4  # suppress noise
                # If InferenceSession init with CUDA doesn't throw, CUDA works
                # We can't test without a model, so check provider library directly
                # torch registers its bundled CUDA DLL directory via
                # os.add_dll_directory() at import time; the provider DLL cannot
                # resolve cublas/cudnn without it, so probe only after torch loads.
                try:
                    import torch as _torch  # noqa: F401
                except ImportError:
                    pass
                import ctypes
                import os as _os
                _cuda_dll = _os.path.join(_os.path.dirname(ort.__file__), "capi", "onnxruntime_providers_cuda.dll")
                if _os.path.exists(_cuda_dll):
                    try:
                        ctypes.CDLL(_cuda_dll)
                        ort_cuda = True
                    except OSError as exc:
                        # Name the missing dependency: a CUDA major mismatch
                        # between onnxruntime-gpu and the torch wheels reports
                        # the exact DLL that could not be resolved.
                        logger.warning("ORT CUDA DLL exists but failed to load: %s", exc)
                else:
                    ort_cuda = True  # non-Windows or different layout, trust get_available_providers
            except Exception:
                pass
    except ImportError:
        pass
    except Exception as exc:
        # A half-removed onnxruntime leaves the package importable but empty,
        # so get_available_providers() raises AttributeError. That must not take
        # startup down with it; report it and carry on with torch.
        logger.warning("ORT probe failed (%s): %s", type(exc).__name__, exc)
        startup_state["warnings"].append({
            "level": "error",
            "title": "ONNX Runtime is not usable",
            "message": (
                "The onnxruntime package is installed but not importable: "
                f"{type(exc).__name__}: {exc}\n\n"
                "This usually means the CPU and GPU wheels were installed over "
                "each other and one was removed, taking shared files with it. "
                "Recreate the venv and reinstall."
            ),
        })

    # Check torch CUDA
    torch_cuda = False
    try:
        import torch
        torch_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    if not ort_cuda and not torch_cuda:
        startup_state["warnings"].append({
            "level": "error",
            "title": "GPU inference unavailable",
            "message": (
                f"Device {resolved_device} is selected, but "
                "neither ONNX Runtime CUDA nor PyTorch CUDA is available.\n"
                "Inference will run on CPU and will be very slow (100+ seconds per image).\n\n"
                "Fix: install onnxruntime-gpu, or verify the PyTorch CUDA build."
            ),
        })
        logger.warning("Neither ORT CUDA EP nor torch CUDA available --inference will be CPU-only")
    elif not ort_cuda and torch_cuda:
        # torch GPU covers training, but ONNX inference still runs on the CPU
        # provider at 50-100 s per image. Staying silent here is what made a
        # broken install look like ordinary slowness.
        startup_state["warnings"].append({
            "level": "warning",
            "title": "ONNX Runtime GPU unavailable",
            "message": (
                "PyTorch CUDA is working, but the ONNX Runtime CUDA provider "
                "failed to load, so ONNX inference runs on CPU "
                "(50-100 seconds per image).\n\n"
                "Fix: install an onnxruntime-gpu build whose CUDA major version "
                "matches the installed PyTorch wheels. onnxruntime-gpu 1.27+ on "
                "PyPI is built for CUDA 13, while the CUDA 12 PyTorch wheels "
                "ship cublasLt64_12.dll; pin onnxruntime-gpu to the 1.25 line."
            ),
        })
        logger.warning("ORT CUDA EP failed to load; ONNX inference will run on CPU")
    else:
        logger.info("Inference deps OK: ORT CUDA=%s, torch CUDA=%s", ort_cuda, torch_cuda)


def _scan_all_projects_integrity() -> None:
    """Background scan: auto-reconcile orphan classes and sync annotation indexes."""
    try:
        from .annotate_index import load_annotate_index
        from .classes import auto_reconcile_if_needed
        projects_root = Path(PROJECTS_DIR)
        if not projects_root.exists():
            return
        count = 0
        for d in projects_root.iterdir():
            if not d.is_dir() or d.name.startswith(".") or d.name == "app.db":
                continue
            pid = d.name
            try:
                result = auto_reconcile_if_needed(pid)
                if result:
                    count += len(result["added"])
                    logger.info("Startup reconcile: project %s --added %d class(es)", pid, len(result["added"]))
            except Exception:
                logger.warning("integrity check failed for project %s", pid, exc_info=True)
            # Sync annotation index to fix stale hasForeground flags
            try:
                load_annotate_index(pid, sync=True)
            except Exception:
                logger.warning("integrity check failed for project %s", pid, exc_info=True)
        if count > 0:
            logger.info("Startup integrity scan: reconciled %d orphan class(es) total", count)
    except Exception:
        logger.debug("Integrity scan skipped", exc_info=True)


def _deferred_post_startup() -> None:
    """Non-critical tasks that run after the server is already ready."""
    try:
        # Armed before the checks below, which walk every project on disk: a
        # failure in one of them takes the rest of that try block with it, and
        # the card would then stay occupied for the life of the process.
        from .ort_infra import start_ort_idle_release_thread
        start_ort_idle_release_thread()
    except Exception:
        logger.warning("ORT idle release could not be armed", exc_info=True)
    try:
        _run_health_check()
        _auto_check_deps()
        _scan_all_projects_integrity()
    except Exception:
        logger.warning("Post-startup check error", exc_info=True)
