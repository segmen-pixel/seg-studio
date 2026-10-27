# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Build model_combo_library.json from ablation experiment results.

Scans ablation experiment artifacts, extracts:
- project features (dataset_stats + image features)
- config (arch, base_channels, patch_size, distill_mode)
- performance (best_F1_val, best_mIoU_val)

Computes per-project z-score normalization and outputs a JSON library
for use by config_selector.py at runtime.

Usage:
    python -m scripts.make_autoalgorithm.build_combo_library \
        --artifacts-dir /path/to/ablation_artifacts \
        --output model_combo_library.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Config combo key: the 3 architecture-level parameters
ARCH_COMBO_KEYS = ("arch", "base_channels", "patch_size")


def _infer_arch_from_dirname(name: str) -> str:
    """Infer architecture from run directory name like 'ablation_stdc_dinov2_bc128_...'."""
    low = name.lower()
    if "deeplabv3plus" in low or "deeplabv3" in low:
        return "deeplabv3plus"
    if "stdc" in low:
        return "stdc"
    return "simpleunet"


def _infer_bc_from_dirname(run_name: str, group_name: str) -> int:
    """Infer base_channels from directory names."""
    import re
    for name in (run_name, group_name):
        m = re.search(r"bc(\d+)", name)
        if m:
            return int(m.group(1))
    return 64


def _infer_patch_from_dirname(group_name: str) -> int:
    """Infer patch_size from group directory name like 'ablation_arch_bc32_p128'."""
    import re
    m = re.search(r"p(\d+)", group_name)
    if m:
        return int(m.group(1))
    return 256


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_features(ds: dict) -> dict[str, float]:
    """Extract project features from dataset_stats."""
    features: dict[str, float] = {}
    for key in (
        "fg_ratio", "num_train", "num_total", "num_val",
        "mean_width", "mean_height", "mean_fg_area_px", "std_fg_area_px",
        "num_active_classes", "mean_fg_ratio_per_image",
    ):
        val = ds.get(key)
        if val is not None and isinstance(val, (int, float)):
            features[key] = float(val)

    # Derived features
    w = features.get("mean_width", 0)
    h = features.get("mean_height", 0)
    n = features.get("num_train", 0)
    fg_area = features.get("mean_fg_area_px", 0)
    img_px = w * h

    if n > 0:
        features["log_num_train"] = math.log1p(n)
    if img_px > 0:
        features["log_img_pixels"] = math.log(img_px)
        features["fg_area_frac"] = fg_area / img_px if fg_area > 0 else 0.0

    # Per-class pixel ratios → class imbalance
    ratios = ds.get("per_class_pixel_ratios", [])
    nonzero = [r for r in ratios if isinstance(r, (int, float)) and r > 1e-8]
    if len(nonzero) >= 2:
        features["class_imbalance_ratio"] = max(nonzero) / min(nonzero)

    return features


def collect_ablation_runs(artifacts_dir: Path) -> list[dict]:
    """Scan all ablation_*/**/metrics.json and return run records."""
    records = []
    for metrics_path in sorted(artifacts_dir.glob("ablation_*/**/metrics.json")):
        m = _read_json(metrics_path)
        if m is None:
            continue

        f1 = m.get("best_F1_val")
        if not isinstance(f1, (int, float)) or f1 < 0.01:
            continue

        ds = m.get("dataset_stats") or {}
        cfg = m.get("config") or {}

        # Resolve config fields: config > dataset_stats > directory name
        run_dir_name = metrics_path.parent.name  # e.g. ablation_stdc_dinov2_bc128_20260317_010106
        group_dir_name = metrics_path.parent.parent.parent.name  # e.g. ablation_arch_bc128

        arch = cfg.get("arch") or ds.get("arch") or _infer_arch_from_dirname(run_dir_name)
        bc = cfg.get("base_channels") or ds.get("base_channels") or _infer_bc_from_dirname(run_dir_name, group_dir_name)
        ps = cfg.get("patch_size") or ds.get("patch_size") or _infer_patch_from_dirname(group_dir_name)
        distill = cfg.get("distill_mode") or ds.get("distill_mode") or ("feature" if "dinov2" in run_dir_name else "off")

        # Project name from directory structure: ablation_xxx/{project}/{run}/metrics.json
        project = metrics_path.parent.parent.name

        record = {
            "project": project,
            "run_dir": str(metrics_path.parent),
            "ablation_group": metrics_path.parent.parent.parent.name,
            "best_F1_val": float(f1),
            "best_mIoU_val": float(m.get("best_mIoU_val", 0)),
            "best_epoch": int(m.get("best_epoch", 0)),
            "arch": arch,
            "base_channels": int(bc),
            "patch_size": int(ps),
            "distill_mode": distill,
            "features": _extract_features(ds),
        }
        records.append(record)

    return records


def compute_zscore_library(records: list[dict]) -> dict:
    """Compute per-project z-scores and build the combo library.

    Returns a dict with:
    - projects: {project_id: {features, combos: {combo_key: z_score, f1, ...}}}
    - global_combos: {combo_key: {mean_z, n_projects, ...}}
    - meta: {n_runs, n_projects, n_combos}
    """
    # Group by project
    by_project: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_project[r["project"]].append(r)

    projects_data: dict[str, Any] = {}
    all_combo_zscores: dict[str, list[float]] = defaultdict(list)
    all_combo_f1s: dict[str, list[float]] = defaultdict(list)

    for project, runs in sorted(by_project.items()):
        f1s = [r["best_F1_val"] for r in runs]
        mu = sum(f1s) / len(f1s)
        if len(f1s) > 1:
            sd = (sum((x - mu) ** 2 for x in f1s) / (len(f1s) - 1)) ** 0.5
        else:
            sd = 0.0

        # Use first run's features as project features (should be same across configs)
        project_features = runs[0]["features"]

        combos: dict[str, dict] = {}
        for r in runs:
            combo_key = f"{r['arch']}_bc{r['base_channels']}_p{r['patch_size']}"
            z = (r["best_F1_val"] - mu) / sd if sd > 1e-8 else 0.0

            # Keep best run per combo per project
            if combo_key not in combos or r["best_F1_val"] > combos[combo_key]["f1"]:
                combos[combo_key] = {
                    "f1": r["best_F1_val"],
                    "miou": r["best_mIoU_val"],
                    "z": round(z, 4),
                    "arch": r["arch"],
                    "base_channels": r["base_channels"],
                    "patch_size": r["patch_size"],
                    "distill_mode": r["distill_mode"],
                }

            all_combo_zscores[combo_key].append(z)
            all_combo_f1s[combo_key].append(r["best_F1_val"])

        projects_data[project] = {
            "features": project_features,
            "n_runs": len(runs),
            "mean_f1": round(mu, 4),
            "std_f1": round(sd, 4),
            "combos": combos,
        }

    # Global combo rankings
    global_combos: dict[str, dict] = {}
    for combo_key, zs in sorted(all_combo_zscores.items()):
        f1s = all_combo_f1s[combo_key]
        n_projects = len(set(
            p for p, pdata in projects_data.items()
            if combo_key in pdata["combos"]
        ))
        global_combos[combo_key] = {
            "mean_z": round(sum(zs) / len(zs), 4),
            "std_z": round((sum((x - sum(zs) / len(zs)) ** 2 for x in zs) / max(1, len(zs) - 1)) ** 0.5, 4) if len(zs) > 1 else 0.0,
            "mean_f1": round(sum(f1s) / len(f1s), 4),
            "n_projects": n_projects,
            "n_runs": len(zs),
            # Parse combo key back to components
            **_parse_combo_key(combo_key),
        }

    # Sort global combos by mean_z descending
    global_ranked = sorted(global_combos.items(), key=lambda x: x[1]["mean_z"], reverse=True)

    return {
        "projects": projects_data,
        "global_combos": dict(global_ranked),
        "meta": {
            "n_runs": len(records),
            "n_projects": len(projects_data),
            "n_combos": len(global_combos),
            "top_combo": global_ranked[0][0] if global_ranked else None,
        },
    }


def _parse_combo_key(key: str) -> dict:
    """Parse 'simpleunet_bc64_p256' → {arch, base_channels, patch_size}."""
    parts = key.split("_")
    result: dict[str, Any] = {"arch": parts[0]}
    for p in parts[1:]:
        if p.startswith("bc"):
            result["base_channels"] = int(p[2:])
        elif p.startswith("p"):
            result["patch_size"] = int(p[1:])
    return result


def main():
    parser = argparse.ArgumentParser(description="Build model combo library from ablation results")
    parser.add_argument("--artifacts-dir", type=str,
                        default="./ablation_artifacts",
                        help="Path to ablation artifacts directory")
    parser.add_argument("--output", type=str, default="model_combo_library.json",
                        help="Output JSON path")
    parser.add_argument("--min-runs", type=int, default=10,
                        help="Minimum runs per project to include (default: 10)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        print(f"ERROR: Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {artifacts_dir} ...")
    records = collect_ablation_runs(artifacts_dir)
    print(f"Found {len(records)} valid runs across {len(set(r['project'] for r in records))} projects")

    # Filter out projects with too few runs
    if args.min_runs > 1:
        from collections import Counter
        proj_counts = Counter(r["project"] for r in records)
        dropped = {p for p, c in proj_counts.items() if c < args.min_runs}
        if dropped:
            records = [r for r in records if r["project"] not in dropped]
            print(f"Dropped {len(dropped)} projects with < {args.min_runs} runs: {sorted(dropped)}")
            print(f"Remaining: {len(records)} runs across {len(set(r['project'] for r in records))} projects")

    library = compute_zscore_library(records)
    meta = library["meta"]
    print(f"Library: {meta['n_runs']} runs, {meta['n_projects']} projects, {meta['n_combos']} combos")
    print(f"Top combo: {meta['top_combo']}")

    if args.verbose:
        print("\nTop 10 global combos:")
        for i, (key, data) in enumerate(list(library["global_combos"].items())[:10]):
            print(f"  {i+1}. {key}: mean_z={data['mean_z']:.3f}, mean_f1={data['mean_f1']:.3f}, n_proj={data['n_projects']}")

        print("\nPer-project winners:")
        for proj, pdata in sorted(library["projects"].items()):
            best_combo = max(pdata["combos"].items(), key=lambda x: x[1]["f1"])
            print(f"  {proj}: {best_combo[0]} F1={best_combo[1]['f1']:.4f} (n_runs={pdata['n_runs']})")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
