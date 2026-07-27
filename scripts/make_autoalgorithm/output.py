# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""JSON output and human-readable report generation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .collect import COMBO_KEYS, combo_key_str


def _base_config() -> dict[str, Any]:
    """Sweep BASE_CONFIG defaults (shared between config builders)."""
    return {
        "preset": "fast",
        "epochs": 50,
        "batch_size": 4,
        "input_size": [512, 512],
        "crop_foreground": True,
        "crop_scale": 0.7,
        "patch_size": 256,
        "patches_per_image": 6,
        "fg_patch_prob": 0.7,
        "augment_enabled": True,
        "augment_hflip_prob": 0.5,
        "augment_vflip_prob": 0.0,
        "augment_rotate90_prob": 0.25,
        "augment_brightness": 0.15,
        "augment_contrast": 0.15,
        "augment_noise_std": 0.02,
        "output_stride": 2,
        "use_class_weights": True,
        "early_stopping_patience": 15,
        "min_epochs": 10,
    }


def build_recommended_config(portfolio: list[dict]) -> dict[str, Any]:
    """Build a TrainRequest-compatible config from the #1 ranked combo.

    Merges sweep BASE_CONFIG defaults with the recommended HP values.
    """
    if not portfolio:
        return {}

    top = portfolio[0]

    config = _base_config()

    # Override with recommended HP values
    config["loss_type"] = top.get("loss_type", "focal")
    if top.get("dice_weight") is not None:
        config["dice_weight"] = top["dice_weight"]
    if top.get("lr") is not None:
        config["lr"] = top["lr"]

    return config


def build_recommended_config_with_autotune(
    portfolio: list[dict],
    records: list[dict],
    metric: str = "best_F1_val",
) -> dict[str, Any]:
    """Build auto_tune-friendly config from portfolio #1.

    Unlike build_recommended_config (which outputs explicit sweep values for
    reproducibility), this variant delegates dataset-adaptive parameters to
    auto_tune:

    - loss_type, lr: from portfolio #1 (sweep-tuned)
    - dice_weight: None (let auto_tune decide based on fg_ratio)
    - fg_patch_prob: from portfolio #1 best run's auto_tuned value
      (reflects the actual value used during training)
    """
    if not portfolio:
        return {}

    top = portfolio[0]

    config = _base_config()

    # Override loss_type and lr from portfolio #1
    config["loss_type"] = top.get("loss_type", "focal")
    if top.get("lr") is not None:
        config["lr"] = top["lr"]

    # dice_weight = None -> auto_tune will set it based on fg_ratio
    config["dice_weight"] = None

    # Retrieve fg_patch_prob from the best run's auto_tuned values
    auto_tuned = get_top_combo_auto_tuned(top, records, metric=metric)
    if auto_tuned and auto_tuned.get("fg_patch_prob") is not None:
        config["fg_patch_prob"] = auto_tuned["fg_patch_prob"]

    return config


def get_top_combo_auto_tuned(
    top_combo: dict,
    records: list[dict],
    metric: str = "best_F1_val",
) -> dict | None:
    """Find auto_tuned dict from the best run of the top combo.

    Looks up per_project to find the best run_id, then searches records
    for the matching auto_tuned data.

    Parameters
    ----------
    top_combo : dict
        Portfolio combo entry (typically portfolio[0]).
    records : list[dict]
        Full run records from collect_runs().
    metric : str
        Metric key used in per_project info dicts (e.g. "best_F1_val").
    """
    per_project = top_combo.get("per_project", {})
    if not per_project:
        return None

    # Find the best run across all projects for this combo
    best_run_id = None
    best_metric = -1.0
    for _pid, info in per_project.items():
        metric_val = info.get(metric, 0.0)
        if info.get("run_id") and metric_val > best_metric:
            best_metric = metric_val
            best_run_id = info["run_id"]

    if not best_run_id:
        return None

    # Find the matching record
    for r in records:
        if r["run_id"] == best_run_id:
            return r.get("auto_tuned")

    return None


def _build_auto_tune_note(auto_tuned: dict | None) -> dict[str, Any] | None:
    """Build auto_tune_note from the best run's auto_tuned values."""
    if not auto_tuned:
        return None

    note = {}
    for key in ("tuned_lr", "fg_patch_prob", "dice_weight"):
        val = auto_tuned.get(key)
        if val is not None:
            note[key] = val

    if not note:
        return None

    note["note"] = "Actual values used during training after auto_tune adjustment"
    return note


def build_output(
    portfolio_result: dict,
    project_features: dict[str, dict] | None,
    metric: str,
    records: list[dict] | None = None,
    top_auto_tuned: dict | None = None,
) -> dict[str, Any]:
    """Build the final autoalgorithm.json structure.

    Parameters
    ----------
    portfolio_result : dict
        Output from build_portfolio().
    project_features : dict or None
        Per-project feature dicts.
    metric : str
        Metric name used for ranking (e.g. "best_F1_val").
    records : list[dict] or None
        Full run records (needed for recommended_config_with_autotune).
    top_auto_tuned : dict or None
        auto_tuned dict from portfolio #1's best run
        (for auto_tune_note metadata).
    """
    portfolio = portfolio_result["portfolio"]

    # Clean up combo_key tuples (not JSON-serializable)
    clean_portfolio = []
    for combo in portfolio:
        c = {**combo}
        c["combo_key"] = combo_key_str(combo)
        # Clean per_project run_id to short form
        pp = {}
        for pid, info in c.get("per_project", {}).items():
            pp[pid[:8]] = info
        c["per_project"] = pp
        c["projects"] = [p[:8] for p in c.get("projects", [])]
        clean_portfolio.append(c)

    # Clean all_combos similarly
    clean_all = []
    for combo in portfolio_result.get("all_combos", []):
        c = {**combo}
        c["combo_key"] = combo_key_str(combo)
        c["projects"] = [p[:8] for p in c.get("projects", [])]
        c.pop("per_project", None)  # too verbose for all_combos
        clean_all.append(c)

    # Clean project summary
    clean_summary = {}
    for pid, stats in portfolio_result.get("project_summary", {}).items():
        clean_summary[pid[:8]] = stats

    output = {
        "generated_at": datetime.now().isoformat(),
        "metric": metric,
        "combo_keys": list(COMBO_KEYS),
        "portfolio": clean_portfolio,
        "recommended_config": build_recommended_config(portfolio),
        "recommended_config_with_autotune": build_recommended_config_with_autotune(
            portfolio, records or [], metric=metric,
        ),
        "confidence": portfolio_result.get("confidence", {}),
        "project_summary": clean_summary,
        "all_combos": clean_all,
    }

    # auto_tune_note: actual auto_tuned values from portfolio #1's best run
    auto_tune_note = _build_auto_tune_note(top_auto_tuned)
    if auto_tune_note:
        output["auto_tune_note"] = auto_tune_note

    if project_features:
        clean_features = {}
        for pid, feat in project_features.items():
            clean_features[pid[:8]] = feat
        output["project_features"] = clean_features

    return output


def write_json(output: dict, path: str | Path) -> None:
    """Write output to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON serializer for types not serializable by default."""
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return str(obj)


def print_report(
    portfolio_result: dict,
    metric: str,
    verbose: bool = False,
) -> None:
    """Print human-readable report to stdout."""
    portfolio = portfolio_result["portfolio"]
    confidence = portfolio_result.get("confidence", {})
    project_summary = portfolio_result.get("project_summary", {})

    sep = "=" * 70

    # Header
    print(f"\n{sep}")
    print("AUTOALGORITHM REPORT")
    print(f"{sep}")
    print(f"  Metric: {metric}")
    print(f"  Projects: {confidence.get('n_projects_total', '?')}")
    print(f"  Total runs: {confidence.get('n_runs_total', '?')}")
    print(f"  Eligible combos: {confidence.get('n_eligible_combos', '?')}")
    print(f"  Confidence: {confidence.get('level', '?').upper()}")

    # Portfolio
    print(f"\n{sep}")
    print("PORTFOLIO (ranked by cross-project mean z-score)")
    print(f"{sep}")
    for combo in portfolio:
        label = combo_key_str(combo)
        z = combo["mean_z_score"]
        raw = combo["mean_raw"]
        n_proj = combo["n_projects"]
        n_runs = combo["n_runs"]
        marker = " <<<" if combo.get("rank") == 1 else ""
        print(
            f"  #{combo.get('rank', '?'):>2}  {label:<30}  "
            f"z={z:+.4f}  mean_{metric}={raw:.4f}  "
            f"({n_runs} runs / {n_proj} projects){marker}"
        )

    # Recommended config
    if portfolio:
        print(f"\n{sep}")
        print("RECOMMENDED CONFIG (TrainRequest-compatible)")
        print(f"{sep}")
        config = build_recommended_config(portfolio)
        # Highlight the tuned parameters
        for key in COMBO_KEYS:
            val = config.get(key)
            if isinstance(val, float) and val < 0.01:
                print(f"  {key}: {val:.0e}  <-- tuned")
            else:
                print(f"  {key}: {val}  <-- tuned")
        print("  (other params: sweep BASE_CONFIG defaults)")

    # Confidence details
    print(f"\n{sep}")
    print("CONFIDENCE")
    print(f"{sep}")
    print(f"  Level:       {confidence.get('level', '?').upper()}")
    print(f"  Coverage:    {confidence.get('coverage', 0):.1%} of projects")
    print(f"  Z-gap (#1-#2): {confidence.get('z_gap', 0):.4f}")
    print(f"  Strength:    {confidence.get('strength', 0):.4f}")
    print(f"  Consistency: {confidence.get('consistency', 0):.4f}")

    # Per-project summary
    if verbose and project_summary:
        print(f"\n{sep}")
        print("PER-PROJECT SUMMARY")
        print(f"{sep}")
        for pid, stats in sorted(project_summary.items()):
            print(
                f"  {pid[:8]}...  "
                f"runs={stats['n_runs']:>3}  "
                f"best={stats['best_metric']:.4f}  "
                f"mean={stats['mean_metric']:.4f}  "
                f"std={stats['std_metric']:.4f}"
            )

    # Verbose: all combos
    if verbose:
        all_combos = portfolio_result.get("all_combos", [])
        print(f"\n{sep}")
        print(f"ALL COMBOS ({len(all_combos)} total)")
        print(f"{sep}")
        for i, combo in enumerate(all_combos[:20], 1):
            label = combo_key_str(combo)
            z = combo["mean_z_score"]
            n_proj = combo["n_projects"]
            print(f"  {i:>3}. {label:<30}  z={z:+.4f}  ({n_proj} proj)")
        if len(all_combos) > 20:
            print(f"  ... and {len(all_combos) - 20} more")
