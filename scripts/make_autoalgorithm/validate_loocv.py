# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Leave-One-Project-Out Cross Validation for config_selector.

For each project P:
  1. Remove P from the library
  2. Run recommend_combo() with P's features
  3. Compare recommended config to P's actual best config
  4. Report: exact hit, top-3 hit, F1 gap to oracle

Usage:
    python -m scripts.make_autoalgorithm.validate_loocv \
        --library scripts/make_autoalgorithm/model_combo_library.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _oracle_combo(project_data: dict) -> tuple[str, float]:
    """Find the best combo for a project (oracle)."""
    combos = project_data.get("combos", {})
    if not combos:
        return ("unknown", 0.0)
    best_key = max(combos, key=lambda k: combos[k]["f1"])
    return (best_key, combos[best_key]["f1"])


def run_loocv(library: dict) -> list[dict]:
    """Run leave-one-project-out cross validation."""
    # Lazy import to allow running standalone
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
    from segcore.auto_select.config_selector import recommend_combo

    projects = library["projects"]
    global_combos = library["global_combos"]
    results = []

    for held_out_id in sorted(projects.keys()):
        held_out = projects[held_out_id]
        oracle_key, oracle_f1 = _oracle_combo(held_out)

        # Build leave-one-out library
        loo_projects = {k: v for k, v in projects.items() if k != held_out_id}

        # Recompute global combos without held-out project
        loo_global = {}
        for combo_key in global_combos:
            zs = []
            for pid, pdata in loo_projects.items():
                cd = pdata.get("combos", {}).get(combo_key)
                if cd is not None:
                    zs.append(cd["z"])
            if zs:
                loo_global[combo_key] = {
                    **global_combos[combo_key],
                    "mean_z": sum(zs) / len(zs),
                    "n_projects": len(zs),
                }

        loo_lib = {
            "projects": loo_projects,
            "global_combos": loo_global,
            "meta": library.get("meta", {}),
        }

        # Recommend
        rec = recommend_combo(held_out["features"], loo_lib)
        rec_key = f"{rec.arch}_bc{rec.base_channels}_p{rec.patch_size}"

        # Check if oracle combo was in recommendations
        top_keys = [k for k, _ in rec.top_combos]
        exact_hit = (rec_key == oracle_key)
        top3_hit = oracle_key in top_keys[:3]
        top5_hit = oracle_key in top_keys[:5]

        # F1 gap: what F1 would we get with recommended combo vs oracle
        # If recommended combo has no data for this project, use best
        # available combo from top recommendations as practical fallback
        held_combos = held_out.get("combos", {})
        rec_combo_data = held_combos.get(rec_key)
        if rec_combo_data:
            rec_f1 = rec_combo_data["f1"]
        else:
            # Find best available combo from ranked recommendations
            rec_f1 = 0.0
            for tk, _ in rec.top_combos:
                if tk in held_combos:
                    rec_f1 = held_combos[tk]["f1"]
                    rec_key = tk  # update for reporting
                    break
        f1_gap = oracle_f1 - rec_f1

        results.append({
            "project": held_out_id,
            "oracle": oracle_key,
            "oracle_f1": round(oracle_f1, 4),
            "recommended": rec_key,
            "rec_f1": round(rec_f1, 4),
            "f1_gap": round(f1_gap, 4),
            "exact_hit": exact_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "confidence": rec.confidence,
            "score": round(rec.score, 4),
            "reasoning": rec.reasoning,
        })

    return results


def print_report(results: list[dict]) -> None:
    """Print human-readable LOOCV report."""
    n = len(results)
    exact_hits = sum(1 for r in results if r["exact_hit"])
    top3_hits = sum(1 for r in results if r["top3_hit"])
    top5_hits = sum(1 for r in results if r["top5_hit"])
    mean_gap = sum(r["f1_gap"] for r in results) / n if n else 0
    max_gap = max(r["f1_gap"] for r in results) if results else 0

    print("=" * 70)
    print(f"LOOCV Results: {n} projects")
    print("=" * 70)
    print(f"  Exact hit rate:  {exact_hits}/{n} ({exact_hits/n*100:.1f}%)")
    print(f"  Top-3 hit rate:  {top3_hits}/{n} ({top3_hits/n*100:.1f}%)")
    print(f"  Top-5 hit rate:  {top5_hits}/{n} ({top5_hits/n*100:.1f}%)")
    print(f"  Mean F1 gap:     {mean_gap:.4f}")
    print(f"  Max F1 gap:      {max_gap:.4f}")
    print()

    print(f"{'Project':<25} {'Oracle':<25} {'Recommended':<25} {'Gap':>6} {'Hit':>4} {'Conf':>6}")
    print("-" * 95)
    for r in results:
        hit_mark = "Y" if r["exact_hit"] else ("~3" if r["top3_hit"] else "N")
        print(
            f"{r['project']:<25} "
            f"{r['oracle']:<25} "
            f"{r['recommended']:<25} "
            f"{r['f1_gap']:>6.4f} "
            f"{hit_mark:>4} "
            f"{r['confidence']:>6}"
        )

    print()
    # Confidence calibration
    for conf in ("high", "medium", "low", "none"):
        subset = [r for r in results if r["confidence"] == conf]
        if subset:
            hits = sum(1 for r in subset if r["exact_hit"])
            gap = sum(r["f1_gap"] for r in subset) / len(subset)
            print(f"  Confidence={conf:>6}: {hits}/{len(subset)} exact, mean_gap={gap:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LOOCV validation for config_selector")
    parser.add_argument("--library", type=str,
                        default="scripts/make_autoalgorithm/model_combo_library.json")
    args = parser.parse_args()

    lib_path = Path(args.library)
    if not lib_path.exists():
        print(f"ERROR: Library not found: {lib_path}", file=sys.stderr)
        return 1

    library = json.loads(lib_path.read_text(encoding="utf-8"))
    results = run_loocv(library)
    print_report(results)

    # Save results
    out_path = lib_path.parent / "loocv_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
