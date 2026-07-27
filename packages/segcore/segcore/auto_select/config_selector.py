# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Recommend optimal (arch, patch_size, base_channels) for a new project.

Primary algorithm: XGBoost ensemble combo predictor v6 — a
``reg:squarederror`` regressor (F1 magnitude) mixed with a ``rank:ndcg``
LambdaRank ranker (ordering), with per-project min-max normalisation
and a tuned ensemble weight (``weight_reg=0.8``).  Returns ranked combos
with predicted F1 and a confidence band derived from regressor / ranker
disagreement.

Legacy fallback: similarity-weighted z-score portfolio used when the ML
model or runtime features are unavailable.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .combo_predictor import combo_is_buildable
from .similarity import _standardized_euclidean_sim

logger = logging.getLogger(__name__)


@dataclass
class ConfigRecommendation:
    """Result of recommend_combo()."""
    arch: str
    base_channels: int
    patch_size: int
    score: float
    confidence: str  # "high" / "medium" / "low" / "none"
    top_combos: list[tuple[str, float]]
    reasoning: str
    # ML predictor extras (optional; None for legacy z-score path)
    pred_f1: float | None = None
    pred_std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    top_combos_detail: list[dict] = field(default_factory=list)
    source: str = "zscore"  # "ml" or "zscore"
    distill_on: bool | None = None  # None for z-score fallback, bool for ML
    # v6 time predictor extras (only set when the time bundle is loaded).
    # `pred_elapsed_sec` is the warmup-calibrated estimate of training
    # time for `(arch, base_channels, patch_size, distill_on, …)` on this
    # project; populated only on the ML path.
    pred_elapsed_sec: float | None = None
    pred_elapsed_min: float | None = None
    # Anchor combo recommended for warmup calibration on this project.
    # Callers that want absolute-magnitude time accuracy should run this
    # combo first and pass its actual elapsed_sec back via
    # `recommend_combo(..., anchor_elapsed_sec=...)`.
    time_anchor_combo: str | None = None
    time_calibrated: bool = False  # True when anchor_elapsed_sec was supplied
    # Why this recommendation did NOT come from the ML path (z-score
    # fallback only; None on the ML path).  Surfaced in train.log so a
    # broken ML predictor (e.g. missing xgboost) is visible to the user
    # instead of silently degrading to the legacy portfolio.
    ml_fallback_reason: str | None = None


# ── Patch-size prior rules ──────────────────────────────────────────

def _patch_prior(combo_key: str, features: dict[str, float]) -> float:
    """Rule-based patch/bc/arch bias from prior empirical studies.

    Returns an additive score bonus/penalty.
    Kept intentionally mild so similarity-based scores dominate.
    """
    fg_area_frac = features.get("fg_area_frac", 0)
    fg_ratio = features.get("fg_ratio", 0)
    mean_w = features.get("mean_width", features.get("mean_img_size", 0))
    mean_h = features.get("mean_height", mean_w)
    img_px = mean_w * mean_h if mean_w and mean_h else 0
    jig_score = features.get("jig_score", -1)  # -1 = unavailable
    inter_var = features.get("inter_image_variance", -1)
    fg_bg_contrast = features.get("fg_bg_contrast", -1)

    is_p128 = "_p128" in combo_key
    is_p256 = "_p256" in combo_key
    is_p512 = "_p512" in combo_key

    bonus = 0.0

    # ── Patch size rules ──

    # Tiny images (< 150k px) → p512
    if img_px > 0 and img_px < 150_000:
        bonus += 0.15 if is_p512 else -0.05 if is_p128 else 0.0

    # Medium-small images + high FG fraction → p128
    elif img_px < 800_000 and fg_area_frac > 0.004:
        bonus += 0.08 if is_p128 else 0.0

    # Default: p256 mild preference
    else:
        bonus += 0.05 if is_p256 else -0.03 if is_p512 else 0.0

    # ── BC rules ──

    is_bc32 = "_bc32" in combo_key
    is_bc64 = "_bc64" in combo_key
    is_bc128 = "_bc128" in combo_key

    # bc128 for complex scenes: high inter-image variance or high contrast
    if inter_var > 1000 or (fg_bg_contrast > 0.2 and fg_ratio > 0.003):
        bonus += 0.03 if is_bc128 else 0.0
    # bc64 mild default preference
    else:
        bonus += 0.02 if is_bc64 else 0.0

    # bc32 penalty except for simple tasks (high data + low FG)
    if is_bc32 and not (features.get("num_train", 0) > 80 and fg_ratio < 0.002):
        bonus -= 0.02

    # ── Architecture rules ──

    is_stdc = "stdc_" in combo_key
    is_su = "simpleunet_" in combo_key

    # High variance scenes → STDC mild preference. (The former low-variance /
    # high-contrast bonus favoured deeplabv3plus, retired in 0.9.7.)
    if jig_score >= 0 and inter_var >= 0:
        if inter_var > 1500:
            bonus += 0.02 if is_stdc else 0.0

    # SimpleUNet for very sparse/tiny FG
    if fg_ratio < 0.001 and fg_area_frac < 0.001:
        bonus += 0.03 if is_su else 0.0

    return bonus


# ── Feature-based similarity for config selection ───────────────────

_CONFIG_FEATURE_KEYS = [
    "fg_ratio", "mean_fg_area_px", "num_active_classes",
    "log_num_train", "log_img_pixels", "fg_area_frac",
    # "mean_img_size" was here and is deliberately gone: _enrich_features always
    # derives it for the QUERY, but it appears in 0 of the 21 projects in the
    # shipped model_combo_library.json, so its library std was exactly 0 while
    # the query carried a raw 500-2000. Image scale is already represented by
    # log_img_pixels. See _standardized_euclidean_sim for the guard that now
    # keeps a gap like this from annihilating the score.
    "num_train",
    # Background & inter-image variance features
    "bg_mean_intensity", "bg_std_intensity", "fg_bg_contrast",
    "inter_image_variance", "inter_image_variance_bg", "jig_score",
]


def _project_feature_vec(features: dict[str, float]) -> np.ndarray:
    """Extract a feature vector for similarity computation."""
    vec = np.zeros(len(_CONFIG_FEATURE_KEYS), dtype=np.float32)
    for i, key in enumerate(_CONFIG_FEATURE_KEYS):
        val = features.get(key)
        if val is not None:
            vec[i] = float(val)
    return vec


def _compute_similarities(
    query_features: dict[str, float],
    library_projects: dict[str, dict],
) -> list[tuple[str, float]]:
    """Compute similarity between query and each library project.

    Returns list of (project_id, similarity) sorted descending.
    """
    q_vec = _project_feature_vec(query_features)

    # Compute std across library for standardization
    if len(library_projects) >= 3:
        all_vecs = np.stack([
            _project_feature_vec(pdata["features"])
            for pdata in library_projects.values()
        ])
        lib_std = np.std(all_vecs, axis=0)
    else:
        lib_std = None

    sims = []
    for pid, pdata in library_projects.items():
        p_vec = _project_feature_vec(pdata["features"])
        sim = _standardized_euclidean_sim(q_vec, p_vec, lib_std)
        sims.append((pid, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


# ── Main recommendation function ───────────────────────────────────

def recommend_combo(
    query_features: dict[str, float],
    library: dict | None = None,
    top_k: int = 7,
    *,
    images_dir: str | Path | None = None,
    masks_dir: str | Path | None = None,
    device: str = "cpu",
    anchor_elapsed_sec: float | None = None,
    runtime_features_override: dict[str, float] | None = None,
    dino_global_768_override: np.ndarray | None = None,
) -> ConfigRecommendation:
    """Recommend the best (arch, patch_size, base_channels) combo.

    When *images_dir* and *masks_dir* are provided, uses the ML dual-model
    predictor (LambdaRank + Regression ensemble) which requires runtime
    image/mask features (geometric, edge, fourier, bg variance, DINOv2).

    Otherwise falls back to the legacy z-score portfolio over *library*.

    Parameters
    ----------
    query_features : dict
        Dataset features (from dataset_stats + derived). For the ML path
        these are merged with runtime-extracted features.
    library : dict, optional
        Legacy model_combo_library.json (only used by fallback).
    top_k : int
        Similar projects to consider in the z-score fallback.
    images_dir, masks_dir : optional paths
        If supplied, enables the ML path.
    device : str
        Device for DINOv2 extraction ("cpu" or "cuda").
    anchor_elapsed_sec : float, optional
        Actual measured runtime (seconds) of the v6 warmup anchor combo on
        this project.  When supplied the ML path returns calibrated
        training-time estimates (LOPO R²(log) ≈ +0.958 / MAPE ≈ 14 %); when
        omitted only the weak physical-only prediction is returned.  The
        recommended anchor combo for this bundle is exposed as
        ``ConfigRecommendation.time_anchor_combo``.
    runtime_features_override : dict, optional
        Pre-computed scalar runtime features (edge / fourier / bg-variance
        / geometric). When supplied, the ML path skips its own
        ``extract_runtime_features`` call. Used by the AutoOrchestrator
        (ADR-005) so the same features are not extracted twice per run.
    dino_global_768_override : ndarray, optional
        Pre-computed 768-d global DINO vector. Paired with
        ``runtime_features_override``. May be ``None`` even when overriding
        (equivalent to the extractor returning no DINO).
    """
    # Enrich query features with derived values (both paths use these)
    _enrich_features(query_features)

    # ── ML path ──────────────────────────────────────────────────
    if images_dir is not None and masks_dir is not None:
        ml_rec, ml_fallback_reason = _recommend_via_ml(
            query_features, images_dir, masks_dir,
            device=device, anchor_elapsed_sec=anchor_elapsed_sec,
            runtime_features_override=runtime_features_override,
            dino_global_768_override=dino_global_768_override,
        )
        if ml_rec is not None:
            return ml_rec
        logger.info(
            "ML predictor unavailable (%s), falling back to z-score",
            ml_fallback_reason,
        )
    else:
        ml_fallback_reason = "prepared images/masks not available (ML path skipped)"

    # ── Legacy z-score path ─────────────────────────────────────
    library = library or {}
    projects = library.get("projects", {})
    global_combos = library.get("global_combos", {})

    if not global_combos:
        return ConfigRecommendation(
            arch="simpleunet", base_channels=64, patch_size=256,
            score=0.0, confidence="none",
            top_combos=[], reasoning="Empty library — using defaults",
            ml_fallback_reason=ml_fallback_reason,
        )

    # 1. Compute similarities to library projects
    similarities = _compute_similarities(query_features, projects)
    top_projects = similarities[:top_k]

    max_sim = top_projects[0][1] if top_projects else 0.0

    # 2. Compute α (shrinkage toward global prior)
    # High similarity → trust local, low → trust global
    # Use sqrt to push alpha lower (more local trust) when similar projects exist
    alpha = max(0.0, 1.0 - max_sim ** 0.5)

    # 3. Score each combo
    combo_scores: dict[str, float] = {}
    for combo_key, gdata in global_combos.items():
        # Skip architectures retired from the trainer (e.g. deeplabv3plus,
        # removed in 0.9.7). The bundled library still lists them; recommending
        # one would fail the subsequent build. See combo_predictor.combo_is_buildable.
        if not combo_is_buildable(combo_key):
            continue
        global_z = gdata.get("mean_z", 0.0)

        # Similarity-weighted local z-score
        local_z_num = 0.0
        weight_sum = 0.0
        for pid, sim in top_projects:
            pdata = projects.get(pid, {})
            combo_data = pdata.get("combos", {}).get(combo_key)
            if combo_data is not None:
                local_z_num += sim * combo_data["z"]
                weight_sum += sim

        local_z = local_z_num / weight_sum if weight_sum > 1e-8 else 0.0

        # Combined score
        base_score = alpha * global_z + (1 - alpha) * local_z

        # Patch-size prior
        prior = _patch_prior(combo_key, query_features)

        combo_scores[combo_key] = base_score + prior

    # 4. Rank and select
    ranked = sorted(combo_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        # Every library combo named a retired architecture (should not happen
        # while simpleunet/stdc remain in the library, but fail safe).
        return ConfigRecommendation(
            arch="simpleunet", base_channels=64, patch_size=256,
            score=0.0, confidence="none",
            top_combos=[], reasoning="No buildable combo in library — using defaults",
            ml_fallback_reason=ml_fallback_reason,
        )
    best_key, best_score = ranked[0]
    best_data = global_combos[best_key]

    # 5. Confidence assessment
    gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
    n_supporting = sum(1 for pid, sim in top_projects if sim > 0.5)
    if max_sim > 0.8 and gap > 0.15 and n_supporting >= 3:
        confidence = "high"
    elif max_sim > 0.5 and gap > 0.05 and n_supporting >= 2:
        confidence = "medium"
    elif max_sim > 0.3:
        confidence = "low"
    else:
        confidence = "none"

    # 6. Build reasoning string
    top_similar = ", ".join(f"{pid}({sim:.2f})" for pid, sim in top_projects[:3])
    reasoning = (
        f"alpha={alpha:.2f} (max_sim={max_sim:.2f}), "
        f"top_similar=[{top_similar}], "
        f"gap={gap:.3f}"
    )

    return ConfigRecommendation(
        arch=best_data.get("arch", "simpleunet"),
        base_channels=best_data.get("base_channels", 64),
        patch_size=best_data.get("patch_size", 256),
        score=best_score,
        confidence=confidence,
        top_combos=ranked[:5],
        reasoning=reasoning,
        source="zscore",
        ml_fallback_reason=ml_fallback_reason,
    )


# ── ML predictor path ──────────────────────────────────────────────

def _recommend_via_ml(
    query_features: dict[str, float],
    images_dir: str | Path,
    masks_dir: str | Path,
    *,
    device: str = "cpu",
    anchor_elapsed_sec: float | None = None,
    runtime_features_override: dict[str, float] | None = None,
    dino_global_768_override: np.ndarray | None = None,
) -> tuple[ConfigRecommendation | None, str | None]:
    """Runtime feature extraction -> XGBoost ensemble combo predictor (v6).

    Returns ``(recommendation, None)`` on success, or ``(None, reason)``
    if the bundled model or dependencies are missing, so the caller can
    fall back to the legacy z-score portfolio path and surface *reason*
    to the user (a missing dependency once degraded every run silently).

    When ``runtime_features_override`` is provided the internal
    ``extract_runtime_features`` call is skipped and the supplied values
    are used verbatim (see AutoOrchestrator / ADR-005).
    """
    try:
        from .combo_predictor import (
            get_default_predictor,
            get_default_predictor_load_error,
        )
        from .feature_extractor import extract_runtime_features
    except Exception as e:
        logger.warning("ML combo predictor imports failed: %s", e)
        return None, f"combo predictor imports failed: {e}"

    predictor = get_default_predictor()
    if predictor is None:
        load_err = get_default_predictor_load_error() or "unknown load failure"
        return None, f"combo predictor bundle failed to load: {load_err}"

    if runtime_features_override is not None:
        runtime_feats = runtime_features_override
        dino_vec = dino_global_768_override
    else:
        try:
            runtime_feats, dino_vec = extract_runtime_features(
                images_dir, masks_dir, device=device,
            )
        except Exception as e:
            logger.warning("Runtime feature extraction failed: %s", e)
            return None, f"runtime feature extraction failed: {e}"

    # Merge: query_features (dataset_stats + bg) ∪ runtime_feats.
    # Runtime wins for overlapping keys (e.g. bg_inter_image_variance).
    merged = dict(query_features)
    merged.update(runtime_feats)

    try:
        ranked = predictor.rank(
            merged, dino_vec_768=dino_vec,
            anchor_elapsed_sec=anchor_elapsed_sec,
        )
    except Exception as e:
        logger.warning("ComboPredictor.rank failed: %s", e)
        return None, f"ComboPredictor.rank failed: {e}"

    if not ranked:
        return None, "combo predictor returned no candidates"
    best = ranked[0]

    # Confidence from ranker/regressor agreement + top-1 margin.
    # v6 `rank_score` is the ensemble score min-max normalised over the
    # candidate combos (range [0, 1]), so the absolute gap between top-1
    # and top-2 is on the same scale as ``pred_std`` (the regressor/
    # ranker disagreement); thresholds are calibrated to that scale.
    gap = float(ranked[0]["rank_score"] - ranked[1]["rank_score"]) if len(ranked) > 1 else 0.0
    std = float(best.get("pred_std", 0.0))
    if std < 0.01 and gap > 0.05:
        conf = "high"
    elif std < 0.03 and gap > 0.02:
        conf = "medium"
    elif std < 0.06:
        conf = "low"
    else:
        conf = "none"

    pred_elapsed_sec = best.get("pred_elapsed_sec")
    pred_elapsed_min = best.get("pred_elapsed_min")
    time_anchor = predictor.anchor_combo  # may be None if bundle missing phys_time
    time_calibrated = bool(anchor_elapsed_sec is not None and anchor_elapsed_sec > 0)

    time_blurb = ""
    if pred_elapsed_min is not None:
        tag = "calibrated" if time_calibrated else "physical-only"
        time_blurb = f", ~{pred_elapsed_min:.1f} min ({tag})"

    reasoning = (
        f"ML ensemble v6: pred_f1={best['pred_f1']:.3f} "
        f"(disagreement={std:.3f}), top1-top2 gap={gap:.3f}, "
        f"CI=[{best['ci_low']:.2f},{best['ci_high']:.2f}]{time_blurb}"
    )

    rec = ConfigRecommendation(
        arch=best["arch"],
        base_channels=int(best["base_channels"]),
        patch_size=int(best["patch_size"]),
        score=float(best["rank_score"]),
        confidence=conf,
        top_combos=[(r["combo"], float(r["rank_score"])) for r in ranked[:5]],
        reasoning=reasoning,
        pred_f1=float(best["pred_f1"]),
        pred_std=std,
        ci_low=float(best["ci_low"]),
        ci_high=float(best["ci_high"]),
        top_combos_detail=ranked[:5],
        source="ml",
        distill_on=bool(best.get("distill_on", False)),
        pred_elapsed_sec=float(pred_elapsed_sec) if pred_elapsed_sec is not None else None,
        pred_elapsed_min=float(pred_elapsed_min) if pred_elapsed_min is not None else None,
        time_anchor_combo=time_anchor,
        time_calibrated=time_calibrated,
    )
    return rec, None


def _enrich_features(features: dict[str, float]) -> None:
    """Add derived features in-place if not already present."""
    num_train = features.get("num_train", 0)
    mean_w = features.get("mean_width", 0)
    mean_h = features.get("mean_height", 0)
    fg_area = features.get("mean_fg_area_px", 0)
    img_px = mean_w * mean_h if mean_w and mean_h else 0

    if "log_num_train" not in features and num_train:
        features["log_num_train"] = math.log1p(float(num_train))
    if "log_img_pixels" not in features and img_px > 0:
        features["log_img_pixels"] = math.log(img_px)
    if "fg_area_frac" not in features and img_px > 0 and fg_area:
        features["fg_area_frac"] = float(fg_area) / img_px
    if "mean_img_size" not in features and mean_w and mean_h:
        features["mean_img_size"] = (float(mean_w) + float(mean_h)) / 2.0


# ── Library loading ─────────────────────────────────────────────────

def _default_combo_library_path() -> Path:
    """Resolve combo library bundled with the package."""
    return Path(__file__).resolve().parent / "model_combo_library.json"


def load_combo_library(path: str | Path | None = None) -> dict:
    """Load model_combo_library.json.

    If *path* is None, uses the bundled copy in segcore.auto_select.
    """
    if path is None:
        path = _default_combo_library_path()
    path = Path(path)
    if not path.exists():
        logger.warning("Combo library not found: %s", path)
        return {"projects": {}, "global_combos": {}, "meta": {}}
    return json.loads(path.read_text(encoding="utf-8"))
