# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Core algorithm: z-score normalization + portfolio ranking.

1. Per-project z-score normalization (removes task difficulty bias)
2. Group by HP combo, compute cross-project mean z-score
3. Filter by min_projects_tested (avoids n=1 overfit)
4. Rank by mean z-score -> top-K portfolio
5. Confidence metrics
"""
from __future__ import annotations

import math
from typing import Any


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def compute_project_zscores(
    records: list[dict],
    metric: str = "best_F1_val",
) -> list[dict]:
    """Add per-project z-score to each record.

    Z-score = (metric - project_mean) / project_std
    If a project has std=0 (all runs identical), z-score = 0.
    """
    # Group by project
    by_project: dict[str, list[dict]] = {}
    for r in records:
        by_project.setdefault(r["project_id"], []).append(r)

    result = []
    for pid, runs in by_project.items():
        vals = [r[metric] for r in runs]
        mu = _mean(vals)
        sigma = _std(vals)

        for r in runs:
            z = (r[metric] - mu) / sigma if sigma > 1e-12 else 0.0
            enriched = {**r, "z_score": z, "project_mean": mu, "project_std": sigma}
            result.append(enriched)

    return result


def build_portfolio(
    records: list[dict],
    metric: str = "best_F1_val",
    top_k: int = 5,
    min_projects: int = 2,
) -> dict[str, Any]:
    """Build HP portfolio from z-score normalized sweep results.

    Returns:
      {
        "portfolio": [...],         # top-K combos ranked by mean z-score
        "all_combos": [...],        # all combos (unfiltered for debugging)
        "project_summary": {...},   # per-project stats
        "confidence": {...},        # confidence metrics
      }
    """
    # Step 1: z-score normalization
    enriched = compute_project_zscores(records, metric)

    # Step 2: group by combo key
    by_combo: dict[tuple, list[dict]] = {}
    for r in enriched:
        by_combo.setdefault(r["combo_key"], []).append(r)

    # Step 3: compute per-combo stats
    all_combos = []
    for ck, runs in by_combo.items():
        z_scores = [r["z_score"] for r in runs]
        raw_metrics = [r[metric] for r in runs]
        projects = sorted(set(r["project_id"] for r in runs))

        # Per-project breakdown
        per_project = {}
        for r in runs:
            pid = r["project_id"]
            if pid not in per_project or r[metric] > per_project[pid][metric]:
                per_project[pid] = {
                    metric: r[metric],
                    "z_score": r["z_score"],
                    "run_id": r["run_id"],
                }

        combo_config = runs[0]["config"]
        combo = {
            "combo_key": ck,
            "loss_type": combo_config.get("loss_type"),
            "dice_weight": combo_config.get("dice_weight"),
            "lr": combo_config.get("lr"),
            "mean_z_score": _mean(z_scores),
            "std_z_score": _std(z_scores),
            "mean_raw": _mean(raw_metrics),
            "max_raw": max(raw_metrics),
            "min_raw": min(raw_metrics),
            "n_runs": len(runs),
            "n_projects": len(projects),
            "projects": projects,
            "per_project": per_project,
        }
        all_combos.append(combo)

    # Step 4: filter and rank
    eligible = [c for c in all_combos if c["n_projects"] >= min_projects]
    eligible.sort(key=lambda c: c["mean_z_score"], reverse=True)

    # If not enough eligible combos, relax the filter
    if len(eligible) < top_k and min_projects > 1:
        eligible = sorted(all_combos, key=lambda c: c["mean_z_score"], reverse=True)

    portfolio = eligible[:top_k]

    # Assign ranks
    for i, combo in enumerate(portfolio):
        combo["rank"] = i + 1

    # Step 5: project summary
    by_project_stats: dict[str, dict] = {}
    for r in enriched:
        pid = r["project_id"]
        if pid not in by_project_stats:
            by_project_stats[pid] = {
                "n_runs": 0,
                "mean_metric": 0.0,
                "best_metric": 0.0,
                "std_metric": 0.0,
                "metrics": [],
            }
        by_project_stats[pid]["n_runs"] += 1
        by_project_stats[pid]["metrics"].append(r[metric])

    for pid, stats in by_project_stats.items():
        vals = stats.pop("metrics")
        stats["mean_metric"] = _mean(vals)
        stats["best_metric"] = max(vals)
        stats["std_metric"] = _std(vals)

    # Step 6: confidence metrics
    confidence = _compute_confidence(portfolio, all_combos, enriched)

    return {
        "portfolio": portfolio,
        "all_combos": sorted(
            all_combos, key=lambda c: c["mean_z_score"], reverse=True
        ),
        "project_summary": by_project_stats,
        "confidence": confidence,
    }


def _compute_confidence(
    portfolio: list[dict],
    all_combos: list[dict],
    records: list[dict],
) -> dict[str, Any]:
    """Compute confidence indicators for the recommendation."""
    if not portfolio:
        return {"level": "none", "coverage": 0, "z_gap": 0, "strength": 0}

    top = portfolio[0]
    n_projects_total = len(set(r["project_id"] for r in records))
    n_runs_total = len(records)

    # Coverage: fraction of projects tested by top combo
    coverage = top["n_projects"] / n_projects_total if n_projects_total > 0 else 0

    # Z-score gap: difference between #1 and #2
    z_gap = 0.0
    if len(portfolio) >= 2:
        z_gap = portfolio[0]["mean_z_score"] - portfolio[1]["mean_z_score"]

    # Strength: how far above zero the top z-score is (>0.5 is strong)
    strength = top["mean_z_score"]

    # Consistency: low std means reliable across projects
    consistency = 1.0 / (1.0 + top["std_z_score"]) if top["std_z_score"] >= 0 else 0

    # Overall level
    if coverage >= 0.5 and strength > 0.3 and consistency > 0.5:
        level = "high"
    elif coverage >= 0.3 and strength > 0.1:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "coverage": round(coverage, 3),
        "z_gap": round(z_gap, 4),
        "strength": round(strength, 4),
        "consistency": round(consistency, 4),
        "n_projects_total": n_projects_total,
        "n_runs_total": n_runs_total,
        "n_eligible_combos": len([c for c in all_combos if c["n_projects"] >= 2]),
    }
