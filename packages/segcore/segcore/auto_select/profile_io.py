# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Save / load ProjectProfile as compressed .npz files."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .schema import ProjectProfile

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "feature_profile.npz"

# Scalar fields stored as JSON in the npz
_SCALAR_KEYS = [
    "project_id", "run_id", "arch", "base_channels", "output_stride",
    "patch_size", "patches_per_image", "fg_patch_prob", "loss_type",
    "distill_mode", "best_f1", "best_miou", "best_epoch", "total_epochs",
    "checkpoint_path",
]


def save_profile(profile: ProjectProfile, directory: str | Path) -> Path:
    """Save a ProjectProfile to ``directory/feature_profile.npz``.

    Returns the path to the saved file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / PROFILE_FILENAME

    scalars = {k: getattr(profile, k) for k in _SCALAR_KEYS}
    scalars_json = json.dumps(scalars, ensure_ascii=False)

    meta_json = json.dumps(profile.meta, ensure_ascii=False, default=str)

    np.savez_compressed(
        str(out_path),
        # Arrays
        dino_global_mean=np.asarray(profile.dino_global_mean, dtype=np.float32),
        dino_fg_mean=np.asarray(profile.dino_fg_mean, dtype=np.float32),
        dino_bg_mean=np.asarray(profile.dino_bg_mean, dtype=np.float32),
        dino_fg_centroids=np.asarray(profile.dino_fg_centroids, dtype=np.float32),
        handcrafted=np.asarray(profile.handcrafted, dtype=np.float32),
        # JSON-encoded metadata
        scalars_json=np.array(scalars_json),
        meta_json=np.array(meta_json),
        handcrafted_names=np.array(profile.handcrafted_names),
    )
    logger.info("Saved profile to %s", out_path)
    return out_path


def load_profile(path: str | Path) -> ProjectProfile:
    """Load a ProjectProfile from a ``.npz`` file."""
    path = Path(path)
    data = np.load(str(path), allow_pickle=False)

    scalars = json.loads(str(data["scalars_json"]))
    meta = json.loads(str(data["meta_json"]))
    handcrafted_names = list(data["handcrafted_names"])

    return ProjectProfile(
        project_id=scalars["project_id"],
        run_id=scalars["run_id"],
        arch=scalars.get("arch", "simpleunet"),
        base_channels=int(scalars.get("base_channels", 64)),
        output_stride=int(scalars.get("output_stride", 2)),
        patch_size=int(scalars.get("patch_size", 256)),
        patches_per_image=int(scalars.get("patches_per_image", 4)),
        fg_patch_prob=float(scalars.get("fg_patch_prob", 0.5)),
        loss_type=scalars.get("loss_type", "focal"),
        distill_mode=scalars.get("distill_mode", "off"),
        best_f1=float(scalars.get("best_f1", 0.0)),
        best_miou=float(scalars.get("best_miou", 0.0)),
        best_epoch=int(scalars.get("best_epoch", 0)),
        total_epochs=int(scalars.get("total_epochs", 0)),
        dino_global_mean=np.array(data["dino_global_mean"], dtype=np.float32),
        dino_fg_mean=np.array(data["dino_fg_mean"], dtype=np.float32),
        dino_bg_mean=np.array(data["dino_bg_mean"], dtype=np.float32),
        dino_fg_centroids=np.array(data["dino_fg_centroids"], dtype=np.float32),
        handcrafted=np.array(data["handcrafted"], dtype=np.float32),
        handcrafted_names=handcrafted_names,
        meta=meta,
        checkpoint_path=scalars.get("checkpoint_path", ""),
    )


def load_library(projects_dir: str | Path, min_f1: float = 0.5) -> list[ProjectProfile]:
    """Scan a projects directory and load all available profiles.

    Expects layout: ``projects/{project_id}/runs/{run_id}/feature_profile.npz``

    Parameters
    ----------
    min_f1 : float
        Minimum best_f1 to include in the library. Default 0.5.
        Set to 0 to include all profiles.
    """
    projects_dir = Path(projects_dir)
    profiles: list[ProjectProfile] = []

    for npz_path in sorted(projects_dir.rglob(PROFILE_FILENAME)):
        try:
            profile = load_profile(npz_path)
            # Resolve checkpoint path relative to the npz location.  The
            # stored path is absolute, so it goes stale whenever the
            # projects dir is relocated (another machine, another drive,
            # SEG_PROJECTS_DIR) — re-resolve when it is empty or gone.
            if not profile.checkpoint_path or not Path(profile.checkpoint_path).exists():
                model_pt = npz_path.parent / "model.pt"
                if model_pt.exists():
                    profile.checkpoint_path = str(model_pt)
            # Skip profiles below quality threshold
            if profile.best_f1 < min_f1:
                logger.debug("Skipping profile %s: best_f1=%.3f < min_f1=%.2f", npz_path, profile.best_f1, min_f1)
                continue
            profiles.append(profile)
        except Exception as e:
            logger.warning("Failed to load profile %s: %s", npz_path, e)

    logger.info("Loaded %d profiles from %s", len(profiles), projects_dir)
    return profiles
