# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""One-pass feature aggregation for Auto-select + Auto-config.

Both recommenders currently re-derive most image features independently
(see ADR-005 for the full context). This module offers a single pass that
fills every signal they need, so a caller can hand the same FeatureBundle
to both without re-reading images. Neither recommender is modified here;
consumers keep using the existing entry points until Phase B.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .auto_select_utils import (
    build_query_profile,
    compute_basic_stats_fallback,
)

if TYPE_CHECKING:
    from segcore.auto_select.schema import ProjectProfile

logger = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    """All image-feature signals collected in one pass.

    Fields
    ------
    project_id, images_dir, masks_dir
        Provenance / where the features came from.
    basic_stats
        The dataset_stats.json schema (num_train, mean_width, min_width,
        fg_ratio, log_img_pixels, ...). Consumed by build_query_profile
        (Auto-select side) and by recommend_combo (Auto-config side).
    query_profile
        ProjectProfile built from basic_stats + image features. What
        recommend() currently consumes as its query.
    runtime_features
        Geometric + edge + fourier + bg-variance stats — the scalar half
        of extract_runtime_features. Currently recommend_combo recomputes
        these; Phase B lets it accept them via the bundle instead.
    dino_global_768
        768-d global DINO vector produced by extract_runtime_features
        during the same pass, or None if DINO was disabled/failed.
    min_width
        Cached from basic_stats["min_width"] for the wave6 epoch rule.
        None when unavailable — callers fall back to _DEFAULT_SCRATCH_EPOCHS.
    notes
        Human-readable trace of which fallbacks fired during construction.
        Callers can surface these via log_fn for transparency.
    """

    project_id: str
    images_dir: Path
    masks_dir: Path
    basic_stats: dict[str, float]
    query_profile: ProjectProfile
    runtime_features: dict[str, float] = field(default_factory=dict)
    dino_global_768: np.ndarray | None = None
    min_width: float | None = None
    notes: list[str] = field(default_factory=list)


def _load_dataset_stats(prepared_dir: Path) -> dict[str, float]:
    """Read prepared_dir/dataset_stats.json when present; keep numeric fields only."""
    ds_path = prepared_dir / "dataset_stats.json"
    if not ds_path.exists():
        return {}
    try:
        raw = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("dataset_stats.json read failed: %s", exc)
        return {}
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def build_feature_bundle(
    project_id: str,
    prepared_dir: Path,
    *,
    arch: str = "simpleunet",
    base_channels: int = 64,
    device: str = "cpu",
    compute_dino_runtime: bool = True,
    runtime_max_samples: int | None = None,
) -> FeatureBundle:
    """Extract every image feature the recommenders need in a single pass.

    Pure aggregation: does not mutate prepared_dir and does not call
    recommend() / recommend_combo(). See ADR-005 (Phase A) for rationale.

    Parameters
    ----------
    project_id
        Project identifier; tags the resulting ProjectProfile only.
    prepared_dir
        Directory containing images/, masks/, and optionally
        dataset_stats.json / report.json.
    arch, base_channels
        Fed into the ProjectProfile used by the transfer-library query.
        These are the query candidates; if Auto-config later picks a
        different arch, Phase C's Orchestrator will rebuild the profile.
    device
        "cuda:0" / "cpu" — used by runtime feature extraction for DINO.
    compute_dino_runtime
        When True (default) let extract_runtime_features also fetch the
        768-d global DINO vector. Set False for cheap fallback paths.
    runtime_max_samples
        Override extract_runtime_features(max_samples=...). None uses the
        extractor's default.

    Returns
    -------
    FeatureBundle
    """
    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"

    notes: list[str] = []

    basic_stats = _load_dataset_stats(prepared_dir)
    if not basic_stats:
        fallback = compute_basic_stats_fallback(
            images_dir, masks_dir, prepared_dir / "report.json",
        )
        if fallback:
            basic_stats = fallback
            notes.append(
                f"basic_stats: dataset_stats.json missing, computed fallback "
                f"({len(fallback)} keys)"
            )
        else:
            notes.append("basic_stats: no signals available (empty dirs)")

    query_profile = build_query_profile(
        project_id, images_dir, masks_dir,
        arch=arch,
        base_channels=base_channels,
        dataset_stats=basic_stats,
    )

    runtime_features: dict[str, float] = {}
    dino_global_768: np.ndarray | None = None
    if images_dir.exists() and masks_dir.exists():
        try:
            from segcore.auto_select.feature_extractor import extract_runtime_features
            kwargs: dict[str, Any] = {
                "device": device,
                "compute_dino": compute_dino_runtime,
            }
            if runtime_max_samples is not None:
                kwargs["max_samples"] = runtime_max_samples
            runtime_features, dino_global_768 = extract_runtime_features(
                images_dir, masks_dir, **kwargs,
            )
        except Exception as exc:
            notes.append(f"runtime_features: extraction failed ({exc})")

    min_width: float | None = None
    raw_mw = basic_stats.get("min_width")
    if raw_mw is not None:
        try:
            mw = float(raw_mw)
            if mw > 0:
                min_width = mw
        except (TypeError, ValueError):
            pass

    return FeatureBundle(
        project_id=project_id,
        images_dir=images_dir,
        masks_dir=masks_dir,
        basic_stats=basic_stats,
        query_profile=query_profile,
        runtime_features=runtime_features,
        dino_global_768=dino_global_768,
        min_width=min_width,
        notes=notes,
    )
