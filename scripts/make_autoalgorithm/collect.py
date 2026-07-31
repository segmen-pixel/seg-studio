# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Collect sweep results from filesystem (no API needed).

Supports new unified layout (projects/) and legacy layouts
(state/ + storage/, data/ + database/).
Reads metrics.json from each training run directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# HP combo key: the 3 sweep-variable parameters (everything else is BASE_CONFIG)
COMBO_KEYS = ("loss_type", "dice_weight", "lr")


def _find_projects_dir(root: Path) -> Path | None:
    """Detect projects directory (new: projects/, legacy: state/projects, storage/projects)."""
    # New unified layout
    d = root / "projects"
    if d.is_dir():
        return d
    # Legacy fallback
    for name in ("storage", "database", "state", "data"):
        d = root / name / "projects"
        if d.is_dir():
            return d
    return None


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, return None on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_config(metrics: dict) -> dict[str, Any]:
    """Extract HP config from metrics.json.

    Config is embedded in dataset_stats (from train.py compute_dataset_stats).
    Fall back to auto_tuned or top-level fields.
    """
    ds = metrics.get("dataset_stats") or {}
    auto = metrics.get("auto_tuned") or {}

    # Priority: dataset_stats (has the requested config) > auto_tuned > top-level
    return {
        "loss_type": auto.get("loss_type") or ds.get("loss_type", "ce"),
        "output_stride": ds.get("output_stride", 2),
        "dice_weight": _coalesce(auto.get("dice_weight"), ds.get("dice_weight")),
        "lr": _coalesce(ds.get("lr"), auto.get("tuned_lr"), 3e-4),
        "batch_size": ds.get("batch_size", 4),
        "input_size": ds.get("input_size", [512, 512]),
        "patch_size": ds.get("patch_size", 256),
        "patches_per_image": ds.get("patches_per_image", 6),
        "fg_patch_prob": auto.get("fg_patch_prob") or ds.get("fg_patch_prob", 0.7),
        "crop_foreground": ds.get("crop_foreground", False),
        "crop_scale": ds.get("crop_scale", 0.7),
        "epochs": ds.get("epochs", 50),
        "augment_enabled": ds.get("augment_enabled", False),
        "use_class_weights": ds.get("use_class_weights", True),
        "early_stopping_patience": ds.get("early_stopping_patience", 12),
        # Extended keys for transfer learning profile
        "arch": ds.get("arch", "simpleunet"),
        "base_channels": ds.get("base_channels", 64),
        "distill_mode": ds.get("distill_mode", "off"),
    }


def _coalesce(*values: Any) -> Any:
    """Return the first non-None value."""
    for v in values:
        if v is not None:
            return v
    return None


def combo_key(config: dict) -> tuple:
    """Create a hashable HP combo key from config."""
    parts = []
    for k in COMBO_KEYS:
        v = config.get(k)
        # Normalize floats for consistent grouping
        if isinstance(v, float):
            v = round(v, 6)
        parts.append(v)
    return tuple(parts)


def combo_key_str(config: dict) -> str:
    """Human-readable combo key string."""
    loss = config.get("loss_type", "?")
    dw = config.get("dice_weight")
    lr = config.get("lr")
    dw_str = f"{dw:.1f}" if isinstance(dw, (int, float)) else "none"
    lr_str = f"{lr:.0e}" if isinstance(lr, float) else str(lr)
    return f"{loss}/dice_w={dw_str}/lr={lr_str}"


def collect_runs(
    root: str | Path,
    min_f1: float = 0.01,
    verbose: bool = False,
) -> list[dict]:
    """Scan filesystem for all training runs with metrics.

    Returns list of records with structure:
      {project_id, run_id, best_F1_val, best_mIoU_val, best_epoch,
       config, dataset_stats, auto_tuned, class_weights, combo_key}
    """
    root = Path(root)
    projects_dir = _find_projects_dir(root)
    if projects_dir is None:
        print(f"ERROR: No projects directory found in {root}")
        return []

    def _runs_dir_of(proj: Path) -> Path:
        """Runs directory, in whichever layout this project is in.

        This walks the projects directory itself, so nothing has migrated it.
        """
        current = proj / "runs"
        return current if current.is_dir() else proj / "training" / "runs"

    records = []
    project_ids = sorted(
        d.name for d in projects_dir.iterdir()
        # Ids are opaque and now short, so length says nothing. A project is
        # a directory that holds runs; dot-prefixed entries (.library,
        # .gpu_locks) are not projects at all.
        if d.is_dir() and not d.name.startswith(".") and (d / "training").is_dir()
    )

    if verbose:
        print(f"  Projects dir: {projects_dir}")
        print(f"  Found {len(project_ids)} project(s)")

    for pid in project_ids:
        runs_dir = _runs_dir_of(projects_dir / pid)
        if not runs_dir.is_dir():
            continue

        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        project_count = 0

        for run_dir in run_dirs:
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            metrics = _read_json(metrics_path)
            if metrics is None:
                continue

            best_f1 = metrics.get("best_F1_val", -1)
            if not isinstance(best_f1, (int, float)) or best_f1 < min_f1:
                continue

            config = _extract_config(metrics)

            # Filter: exclude stride=1 (known inferior from sweep)
            if config.get("output_stride") == 1:
                continue

            record = {
                "project_id": pid,
                "run_id": run_dir.name,
                "best_F1_val": float(best_f1),
                "best_mIoU_val": float(metrics.get("best_mIoU_val", 0)),
                "best_epoch": int(metrics.get("best_epoch", 0)),
                "config": config,
                "dataset_stats": metrics.get("dataset_stats"),
                "auto_tuned": metrics.get("auto_tuned"),
                "class_weights": metrics.get("class_weights"),
                "combo_key": combo_key(config),
            }
            records.append(record)
            project_count += 1

        if verbose and project_count > 0:
            print(f"  Project {pid[:8]}... {project_count} valid run(s)")

    # Summary
    n_projects = len(set(r["project_id"] for r in records))
    n_combos = len(set(r["combo_key"] for r in records))
    if verbose:
        print(f"  Total: {len(records)} runs, {n_projects} project(s), {n_combos} HP combo(s)")

    return records


def get_project_images_dir(root: str | Path, project_id: str) -> Path | None:
    """Find the images directory for a project (for feature extraction).

    Checks multiple locations:
    1. projects/{id}/prepared/images/         (v2 flat layout)
    2. projects/{id}/images/                  (v2 flat layout)
    3. projects/{id}/datasets/prepared/images/ (v1 legacy)
    4. projects/{id}/datasets/annotate/images/ (v1 legacy)
    5. Legacy: storage/projects, state/projects, etc.
    """
    root = Path(root)
    projects_dir = _find_projects_dir(root)

    candidates: list[Path] = []
    if projects_dir:
        # v2 flat layout
        candidates.append(projects_dir / project_id / "prepared" / "images")
        candidates.append(projects_dir / project_id / "images")
        # v1 legacy layout
        candidates.append(projects_dir / project_id / "datasets" / "prepared" / "images")
        candidates.append(projects_dir / project_id / "datasets" / "annotate" / "images")

    # Legacy fallback: check all possible layout dirs
    for name in ("storage", "database", "state", "data"):
        d = root / name / "projects"
        if d.is_dir() and d != projects_dir:
            candidates.append(d / project_id / "prepared" / "images")
            candidates.append(d / project_id / "images")
            candidates.append(d / project_id / "datasets" / "prepared" / "images")
            candidates.append(d / project_id / "datasets" / "annotate" / "images")

    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def get_project_masks_dir(root: str | Path, project_id: str) -> Path | None:
    """Find the masks directory for a project.

    Checks multiple locations with v2 flat layout first, v1 legacy fallback.
    """
    root = Path(root)
    projects_dir = _find_projects_dir(root)

    candidates: list[Path] = []
    if projects_dir:
        # v2 flat layout
        candidates.append(projects_dir / project_id / "prepared" / "masks")
        candidates.append(projects_dir / project_id / "masks")
        # v1 legacy layout
        candidates.append(projects_dir / project_id / "datasets" / "prepared" / "masks")
        candidates.append(projects_dir / project_id / "datasets" / "annotate" / "masks")

    # Legacy fallback: check all possible layout dirs
    for name in ("storage", "database", "state", "data"):
        d = root / name / "projects"
        if d.is_dir() and d != projects_dir:
            candidates.append(d / project_id / "prepared" / "masks")
            candidates.append(d / project_id / "masks")
            candidates.append(d / project_id / "datasets" / "prepared" / "masks")
            candidates.append(d / project_id / "datasets" / "annotate" / "masks")

    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None
