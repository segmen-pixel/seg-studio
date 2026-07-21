# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""AutoOrchestrator — one-pass Auto-select + Auto-config for training jobs.

Consolidates FeatureBundle construction (Phase A), Recipe selection
selection (Phase B), and the Phase-3 config mutation / VRAM warning into
a two-step API:

  decide(...)          -> AutoDecision   # no config mutation
  apply_decision(...)  -> str | None     # mutates config, writes JSON,
                                          # returns pretrained_checkpoint

Config mutation happens exclusively in apply_decision; decide only reads
config and packages the recommenders' output. Log ordering matches the
pre-refactor stream: bundle notes fire immediately, Auto-select messages
are buffered inside decide() and flushed inside the [PHASE 2/6] block,
Recipe messages are emitted inside the [PHASE 3/6] block.

See ADR-005 (Phase C) for the design context.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .auto_feature_bundle import FeatureBundle, build_feature_bundle
from .auto_select_utils import (
    _recommend_scratch_epochs,
)
from .auto_select_utils import (
    compute_basic_stats_fallback as _compute_basic_stats_fallback,
)
from .paths import write_json
from .torch_device import (
    current_configured_torch_device,
    resolve_torch_device_or_cpu,
)

if TYPE_CHECKING:
    from segcore.auto_select.config_selector import ConfigRecommendation

logger = logging.getLogger(__name__)

# Modern user-facing knob (ADR-005 Phase D). The legacy field
# `auto_config` remains supported for callers that set it explicitly
# (backward-compat window until v1.0.0). The former "full" mode
# (automatic donor warm-start) was retired as a product decision
# (ADR-005 addendum, 2026-07): it now coerces to "recipe_only", and a
# legacy `auto_select` field is accepted but ignored.
_AUTO_MODES = ("recipe_only", "off")


def _resolve_phase_toggles(config: dict) -> tuple[bool, str]:
    """Derive (auto_config_on, mode_used) from config.

    Resolution rules:
      1. If the legacy ``auto_config`` field is present in ``config``
         (i.e. the caller set it, not just relied on a getter default),
         it wins. This preserves pre-Phase-D behaviour for existing API
         callers that pass it in the request body.
      2. Otherwise the field is derived from ``config["auto_mode"]``:
             "recipe_only" -> config on   (default)
             "off"         -> config off
      3. Any other ``auto_mode`` value — including the retired "full" —
         coerces to "recipe_only" silently. The returned ``mode_used``
         reflects the coercion so callers can surface it in the log.
    """
    has_legacy_config = "auto_config" in config
    mode = str(config.get("auto_mode", "recipe_only")).lower()
    if mode not in _AUTO_MODES:
        mode = "recipe_only"

    default_config = mode != "off"
    auto_config = bool(config["auto_config"]) if has_legacy_config else default_config
    return auto_config, mode


@dataclass
class AutoDecision:
    """Aggregated recommendation from decide() — no config mutation.

    ``apply_decision`` consumes this to update ``config`` in place.

    Fields
    ------
    project_id, pretrained_checkpoint
        Echoed from ``decide`` inputs so ``apply_decision`` does not need
        to be handed the raw config twice.
    resolved_torch_device, device_kind
        Resolved once for the whole pipeline (used by the VRAM predictor).
    feature_bundle
        The bundle shared across the recipe phase; kept for observability
        and epoch budgeting.
    bundle_notes
        Human-readable trace of which fallbacks fired while building the
        bundle. Already logged in ``decide``; kept for observability.
    recipe
        ``ConfigRecommendation`` from ``recommend_combo``. None if
        Auto-config was off or precompute failed.
    recipe_error
        Stringified precompute exception (None when the call succeeded or
        Auto-config was off).
    apply_recipe
        Gate result (source == "ml" OR confidence in {"high", "medium"}).
        When True the recipe's arch/bc/patch/distill will replace the
        user's config in apply_decision.
    runtime_features_from_bundle
        True when Auto-config's ML path consumed the bundle's cached
        runtime features (Phase A optimisation).
    query_features
        The features dict that fed ``recommend_combo`` — kept because the
        VRAM predictor still needs ``num_train`` from it in apply_decision.
    recommended_epochs
        From-scratch epoch budget (wave6 min_width rule) computed when the
        caller did not request an explicit epoch count. None when the user
        set ``epochs`` explicitly. Written back to config by
        ``apply_decision``.
    pinned_target_arch
        The arch apply_decision will write into config when the recipe is
        applied.
    """

    project_id: str
    pretrained_checkpoint: str | None

    resolved_torch_device: str = "cpu"
    device_kind: str = "cpu"

    feature_bundle: FeatureBundle | None = None
    bundle_notes: list[str] = field(default_factory=list)

    recipe: ConfigRecommendation | None = None
    recipe_error: str | None = None
    apply_recipe: bool = False
    runtime_features_from_bundle: bool = False
    query_features: dict = field(default_factory=dict)

    recommended_epochs: int | None = None

    pinned_target_arch: str = "simpleunet"

    # Phase D: preserve the resolved phase gates so apply_decision can
    # honour them (previously apply_decision re-read config directly,
    # which conflicted with auto_mode support).
    auto_config_on: bool = False


def _resolve_device(config: dict) -> tuple[str, str]:
    """Resolve the training device once. Returns (resolved, kind)."""
    try:
        resolved = str(config.get("resolved_torch_device") or resolve_torch_device_or_cpu(
            str(config.get("torch_device", current_configured_torch_device()))
        ))
    except Exception:
        resolved = "cpu"
    kind = "cuda" if resolved.startswith("cuda") else "cpu"
    return resolved, kind


def _load_query_features(prepared_dir: Path) -> dict:
    """Legacy path: read dataset_stats.json + optional fallback stats."""
    ds_path = prepared_dir / "dataset_stats.json"
    features: dict[str, float] = {}
    if ds_path.exists():
        try:
            raw = json.loads(ds_path.read_text(encoding="utf-8"))
            features = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
        except Exception:
            features = {}
    if not features:
        try:
            fb = _compute_basic_stats_fallback(
                prepared_dir / "images",
                prepared_dir / "masks",
                prepared_dir / "report.json",
            )
            if fb:
                features.update(fb)
        except Exception:
            pass
    return features


def decide(
    project_id: str,
    prepared_dir: Path,
    config: dict,
    pretrained_checkpoint: str | None,
    log_fn: Callable[[str], None],
) -> AutoDecision:
    """Precompute the Recipe recommendation. Does not mutate config.

    Order
    -----
    1. Build the shared FeatureBundle (Phase A, one pass).
    2. Precompute the Recipe (``recommend_combo``) silently.
    3. Decide ``apply_recipe`` + ``pinned_target_arch`` from the recipe.
    4. Pick the from-scratch epoch budget (wave6 rule) when the caller
       did not request an explicit epoch count.

    ``log_fn`` receives bundle notes and bundle-construction failures
    immediately (matching the pre-refactor ordering); Recipe log lines
    are not emitted at all in this step.
    """
    auto_config_on, _mode = _resolve_phase_toggles(config)
    need_auto_config = auto_config_on

    decision = AutoDecision(
        project_id=project_id,
        pretrained_checkpoint=pretrained_checkpoint,
        auto_config_on=auto_config_on,
    )

    if need_auto_config:
        decision.resolved_torch_device, decision.device_kind = _resolve_device(config)
        try:
            decision.feature_bundle = build_feature_bundle(
                project_id, prepared_dir,
                arch=str(config.get("arch", "simpleunet")),
                base_channels=int(config.get("base_channels", 64)),
                device=decision.device_kind,
                compute_dino_runtime=need_auto_config,
            )
            decision.bundle_notes = list(decision.feature_bundle.notes)
            for note in decision.feature_bundle.notes:
                log_fn(f"Auto: {note}\n")
        except Exception as err:
            log_fn(f"Auto: feature bundle failed (non-fatal, falling back per-phase): {err}\n")
            decision.feature_bundle = None

    if need_auto_config:
        try:
            from segcore.auto_select.config_selector import load_combo_library, recommend_combo
            combo_lib = load_combo_library()

            if decision.feature_bundle is not None:
                decision.query_features = dict(decision.feature_bundle.basic_stats)
                runtime_override = decision.feature_bundle.runtime_features
                dino_override = decision.feature_bundle.dino_global_768
                decision.runtime_features_from_bundle = True
            else:
                decision.query_features = _load_query_features(prepared_dir)
                runtime_override = None
                dino_override = None

            images = prepared_dir / "images"
            masks = prepared_dir / "masks"
            decision.recipe = recommend_combo(
                decision.query_features, combo_lib,
                images_dir=images if images.exists() else None,
                masks_dir=masks if masks.exists() else None,
                device=decision.device_kind,
                runtime_features_override=runtime_override,
                dino_global_768_override=dino_override,
            )
        except Exception as err:
            decision.recipe_error = str(err)

    if decision.recipe is not None:
        decision.apply_recipe = (
            decision.recipe.source == "ml"
            or decision.recipe.confidence in ("high", "medium")
        )
    decision.pinned_target_arch = (
        decision.recipe.arch if (decision.recipe is not None and decision.apply_recipe)
        else str(config.get("arch", "simpleunet"))
    )

    # From-scratch epoch budget (wave6 min_width rule). Only when the
    # caller did not pin epochs explicitly; mirrors the retired
    # auto-select path's scratch-epochs behaviour.
    requested_epochs = config.get("epochs")
    if auto_config_on and (requested_epochs is None or int(requested_epochs) <= 0):
        min_width = None
        if decision.feature_bundle is not None:
            min_width = decision.feature_bundle.min_width
        if min_width is None:
            raw = decision.query_features.get("min_width") if decision.query_features else None
            min_width = raw
        decision.recommended_epochs = _recommend_scratch_epochs(min_width)

    return decision


def _apply_recipe(
    config: dict,
    run_path: Path,
    decision: AutoDecision,
    log_fn: Callable[[str], None],
) -> None:
    """Emit Phase-3 recipe logs and mutate config with the applied pick."""
    if decision.recipe_error is not None:
        log_fn(f"Auto-config: failed (non-fatal): {decision.recipe_error}\n")
        return
    if decision.recipe is None:
        log_fn("Auto-config: no recommendation available, keeping user settings\n")
        return

    if decision.runtime_features_from_bundle:
        log_fn(
            f"Auto-config: using cached runtime features from bundle "
            f"(device={decision.device_kind})\n"
        )

    rec = decision.recipe
    if rec.source == "ml":
        ds_suffix = f" distill={'ON' if rec.distill_on else 'OFF'}" if rec.distill_on is not None else ""
        log_fn(
            f"Auto-config [ML]: recommended {rec.arch} bc={rec.base_channels} "
            f"p={rec.patch_size}{ds_suffix} "
            f"pred_f1={rec.pred_f1:.3f}±{rec.pred_std:.3f} "
            f"(confidence={rec.confidence})\n"
        )
        if rec.pred_elapsed_min is not None:
            tag = "calibrated" if rec.time_calibrated else "physical-only"
            log_fn(
                f"Auto-config [ML]: estimated training time "
                f"~{rec.pred_elapsed_min:.1f} min ({tag})\n"
            )
            if not rec.time_calibrated and rec.time_anchor_combo:
                log_fn(
                    f"  (tip) for calibrated ETAs run the anchor combo first: "
                    f"{rec.time_anchor_combo}\n"
                )
        for k, s in rec.top_combos[:5]:
            log_fn(f"  top: {k}  score={s:.3f}\n")
    else:
        # A z-score recommendation means the ML predictor did not run —
        # say why, or a broken predictor degrades every run invisibly
        # (e.g. xgboost missing from the serving venv, 2026-07-07).
        fallback_reason = getattr(rec, "ml_fallback_reason", None)
        if fallback_reason:
            log_fn(
                f"Auto-config: ML predictor unavailable "
                f"({fallback_reason}); using z-score fallback\n"
            )
        log_fn(
            f"Auto-config [zscore]: recommended {rec.arch} bc={rec.base_channels} "
            f"p={rec.patch_size} (score={rec.score:.3f}, "
            f"confidence={rec.confidence})\n"
        )

    if not decision.apply_recipe:
        log_fn("Auto-config: low confidence, keeping user settings\n")
        return

    config["arch"] = rec.arch
    config["base_channels"] = rec.base_channels
    config["patch_size"] = rec.patch_size
    # NOTE: intentionally do NOT reset the surrounding recipe (loss, HNM,
    # deep_supervision, frequency_map, ...) back to a library baseline.
    # The library was captured with an older recipe but the modern
    # seg-studio defaults typically match or beat those library-era
    # numbers when val split / epochs are matched.
    if rec.distill_on is not None:
        old_distill = str(config.get("distill_mode", "off"))
        if rec.distill_on:
            if old_distill in ("off", "", "none"):
                config["distill_mode"] = "feature"
                if not config.get("distill_teacher_model_dir"):
                    config["distill_teacher_model_dir"] = "dinov2_vitb14"
        else:
            config["distill_mode"] = "off"
        if old_distill != config["distill_mode"]:
            log_fn(f"Auto-config: distill_mode {old_distill} -> {config['distill_mode']}\n")

    try:
        write_json(run_path / "train_config.json", config)
    except Exception as err:
        log_fn(f"Auto-config: failed to persist train_config.json: {err}\n")
    log_fn("Auto-config: applied recommendation\n")

    _vram_check(config, decision, log_fn)


def _vram_check(
    config: dict,
    decision: AutoDecision,
    log_fn: Callable[[str], None],
) -> None:
    """Predict peak VRAM for the picked combo and warn on OOM risk."""
    rec = decision.recipe
    if rec is None:
        return
    try:
        if not (rec.source == "ml" and rec.top_combos_detail and decision.resolved_torch_device.startswith("cuda")):
            return
        from segcore.auto_select import get_default_vram_predictor
        vp = get_default_vram_predictor()
        if vp is None:
            return
        import torch as _torch
        gpu_idx = (
            int(decision.resolved_torch_device.split(":")[1])
            if ":" in decision.resolved_torch_device else 0
        )
        gpu_total_mb = _torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 2)
        is_wddm = os.name == "nt"
        combo_str = rec.top_combos_detail[0]["combo"]
        num_train = float(decision.query_features.get("num_train", 0) or 0)
        verdict = vp.verdict(combo_str, gpu_total_mb, is_wddm, num_train)
        log_fn(
            f"Auto-config [VRAM]: {gpu_total_mb:.0f}MB GPU "
            f"({'WDDM' if is_wddm else 'Linux'}); "
            f"predicted peak ~{verdict['pred_vram_mb']:.0f}MB "
            f"(safe ~{verdict['vram_safe_mb']:.0f}MB / "
            f"budget {verdict['budget_mb']:.0f}MB) -> "
            f"{verdict['verdict']}\n"
        )
        if verdict["verdict"] == "oom_risk":
            log_fn(
                "Auto-config [VRAM]: WARNING - this combo may OOM on this "
                "GPU. Consider a smaller base_channels or ce/focal loss "
                "(lovasz has the highest VRAM overhead).\n"
            )
    except Exception as err:
        log_fn(f"Auto-config [VRAM]: check skipped ({err})\n")


# Post-ML sanity rules — 2026-07-07 per-axis EDA on 37 projects.
# Each rule fires only for a value shown by per-project best-F1 analysis to be
# dominated by a wide margin (<= 15% best-hit rate vs 35-51% for the target).
# The ML model's picks on other axes are left untouched.
_SANITY_RULES: list[tuple[str, Any, Any, str]] = [
    (
        "arch",
        "simpleunet",
        "stdc",
        "arch simpleunet -> stdc (EDA: 5/37 vs 15/37 per-project best)",
    ),
    (
        "fg_patch_prob",
        0.5,
        0.7,
        "fg_patch_prob 0.5 -> 0.7 (EDA: 1/37 vs 13/37 per-project best)",
    ),
    (
        "class_weight_strength",
        0.8,
        0.5,
        "class_weight_strength 0.8 -> 0.5 (EDA: 3/37 vs 19/37 per-project best)",
    ),
]


def _matches(current: Any, target: Any) -> bool:
    """Value-equality with a small numeric tolerance for float rules."""
    if isinstance(target, str):
        return current == target
    if current is None:
        return False
    try:
        return abs(float(current) - float(target)) < 0.05
    except (TypeError, ValueError):
        return False


def _apply_evidence_based_sanity_rules(config: dict) -> list[str]:
    """Conservative post-ML sanity rules from per-axis EDA (2026-07-07).

    Returns a list of one-line notes describing any rules that fired.
    Does not persist the config; the caller is responsible for that.
    """
    notes: list[str] = []
    for key, dominated_value, target_value, note in _SANITY_RULES:
        if _matches(config.get(key), dominated_value):
            config[key] = target_value
            notes.append(f"Auto-config [sanity]: {note}")
    return notes


def apply_decision(
    config: dict,
    run_path: Path,
    decision: AutoDecision,
    log_fn: Callable[[str], None],
) -> str | None:
    """Emit the [PHASE 2/6] + [PHASE 3/6] log stream and mutate config.

    Returns the possibly-updated pretrained checkpoint path.
    """
    pretrained_checkpoint = decision.pretrained_checkpoint

    log_fn("[PHASE 2/6] 転移学習選択 (Transfer learning selection)\n")
    if pretrained_checkpoint:
        log_fn(f"Transfer: using user-specified checkpoint {pretrained_checkpoint}\n")
    else:
        log_fn("Transfer: none requested — training from scratch\n")
    if decision.recommended_epochs and int(decision.recommended_epochs) > 0:
        prev_epochs = config.get("epochs")
        new_epochs = int(decision.recommended_epochs)
        if prev_epochs != new_epochs:
            log_fn(
                f"Auto-config: epochs {prev_epochs} -> {new_epochs} "
                f"(from-scratch budget, wave6 min_width rule)\n"
            )
        config["epochs"] = new_epochs

    log_fn("[PHASE 3/6] モデル設定 (Model configuration)\n")
    if decision.auto_config_on:
        _apply_recipe(config, run_path, decision, log_fn)

        # Post-ML sanity rules are part of the Auto-config recommendation
        # surface: auto_mode="off" means "use the request body verbatim"
        # (docs/auto-config-rationale.md), so they must not fire there.
        sanity_notes = _apply_evidence_based_sanity_rules(config)
        if sanity_notes:
            for note in sanity_notes:
                log_fn(note + "\n")
            try:
                write_json(run_path / "train_config.json", config)
            except Exception as err:
                log_fn(
                    f"Auto-config [sanity]: failed to persist train_config.json: {err}\n"
                )

    return pretrained_checkpoint
