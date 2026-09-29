# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Main orchestration: collect -> features -> portfolio -> output."""
from __future__ import annotations

from pathlib import Path

from .collect import collect_runs, get_project_images_dir, get_project_masks_dir
from .features import compute_project_features
from .output import build_output, get_top_combo_auto_tuned, print_report, write_json
from .portfolio import build_portfolio


def run(
    results_dir: str,
    output_path: str = "autoalgorithm.json",
    top_k: int = 5,
    min_projects: int = 2,
    min_f1: float = 0.01,
    metric: str = "best_F1_val",
    skip_features: bool = False,
    verbose: bool = False,
) -> int:
    """Run the full autoalgorithm pipeline. Returns exit code."""
    root = Path(results_dir).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 1

    print(f"make_autoalgorithm: scanning {root}")
    print()

    # 1. Collect
    records = collect_runs(root, min_f1=min_f1, verbose=verbose)
    if not records:
        print("No valid training runs found.")
        return 1

    n_projects = len(set(r["project_id"] for r in records))
    print(f"\nCollected {len(records)} runs from {n_projects} project(s)")

    # Auto-adjust min_projects if not enough projects
    if n_projects < min_projects:
        print(f"\nWARNING: Only {n_projects} project(s) available but min_projects={min_projects}.")
        min_projects = max(1, n_projects)
        print(f"  -> Auto-adjusted min_projects to {min_projects}")

    # 2. Features (per-project)
    project_features = {}
    project_ids = sorted(set(r["project_id"] for r in records))

    if not skip_features:
        print("\nComputing project features...")
        for pid in project_ids:
            # Get dataset_stats from the best run of this project
            project_runs = [r for r in records if r["project_id"] == pid]
            best_run = max(project_runs, key=lambda r: r[metric])
            ds_stats = best_run.get("dataset_stats")

            images_dir = get_project_images_dir(root, pid)
            masks_dir = get_project_masks_dir(root, pid)

            features = compute_project_features(
                root, pid, ds_stats,
                images_dir=images_dir,
                masks_dir=masks_dir,
                skip_image_features=(images_dir is None or masks_dir is None),
            )
            if features:
                project_features[pid] = features
                if verbose:
                    n_feat = len(features)
                    has_img = "color_divergence" in features
                    print(f"  {pid[:8]}... {n_feat} features"
                          f" ({'+ image' if has_img else 'basic only'})")
    else:
        # Basic features only from dataset_stats
        for pid in project_ids:
            project_runs = [r for r in records if r["project_id"] == pid]
            best_run = max(project_runs, key=lambda r: r[metric])
            ds_stats = best_run.get("dataset_stats")
            features = compute_project_features(
                root, pid, ds_stats, skip_image_features=True
            )
            if features:
                project_features[pid] = features

    # 3. Portfolio
    print("\nBuilding portfolio...")
    portfolio_result = build_portfolio(
        records,
        metric=metric,
        top_k=top_k,
        min_projects=min_projects,
    )

    # 4. Output
    # Resolve auto_tuned from portfolio #1's best run for metadata
    top_auto_tuned = None
    portfolio = portfolio_result.get("portfolio", [])
    if portfolio:
        top_auto_tuned = get_top_combo_auto_tuned(portfolio[0], records, metric=metric)

    output = build_output(
        portfolio_result,
        project_features,
        metric,
        records=records,
        top_auto_tuned=top_auto_tuned,
    )
    write_json(output, output_path)

    # 5. Report
    print_report(portfolio_result, metric, verbose=verbose)
    print(f"\nOutput saved to: {output_path}")

    return 0
