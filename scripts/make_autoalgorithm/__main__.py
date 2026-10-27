#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""CLI entry point: python -m make_autoalgorithm <results_dir> [options]"""
from __future__ import annotations

import argparse
import sys

from .run import run


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="make_autoalgorithm",
        description=(
            "Build HP auto-recommendation algorithm from sweep results.\n"
            "Reads metrics.json from filesystem (no API needed)."
        ),
    )
    parser.add_argument(
        "results_dir",
        help="Root directory containing sweep results (projects/ or legacy state/ + storage/)",
    )
    parser.add_argument(
        "-o", "--output",
        default="autoalgorithm.json",
        help="Output JSON path (default: autoalgorithm.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top HP combos in portfolio (default: 5)",
    )
    parser.add_argument(
        "--min-projects",
        type=int,
        default=2,
        help="Min projects a combo must appear in (default: 2)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.01,
        help="Exclude runs with best_F1_val below this (default: 0.01)",
    )
    parser.add_argument(
        "--metric",
        choices=["best_F1_val", "best_mIoU_val"],
        default="best_F1_val",
        help="Metric to optimize (default: best_F1_val)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip image feature extraction (works without images)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    return run(
        results_dir=args.results_dir,
        output_path=args.output,
        top_k=args.top_k,
        min_projects=args.min_projects,
        min_f1=args.min_f1,
        metric=args.metric,
        skip_features=args.skip_features,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
