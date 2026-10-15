#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Example: batch-inference every image in a folder and write the results to CSV.

Usage:
    python batch_inspect.py
    python batch_inspect.py --image-dir ./images --output results.csv

The CSV is written with utf-8-sig encoding so it opens cleanly in Excel.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests
from seg_sdk import SegClient

# ---- Settings ----
BASE_URL = "http://localhost:8002"
PROJECT_ID = "your-project-id"
RUN_ID = "your-run-id"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference over a folder, with CSV output")
    parser.add_argument("--image-dir", type=str, default="./images", help="Path to the image folder")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV filename")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Error: folder not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect the list of image files
    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"Error: no image files found in: {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Detected {len(image_files)} image(s)")

    # Create the client and start a session
    client = SegClient(BASE_URL, timeout=30)
    try:
        client.start_session(project_id=PROJECT_ID, run_id=RUN_ID, backend="onnx")
    except requests.exceptions.ConnectionError:
        print(f"Error: cannot connect to server: {BASE_URL}", file=sys.stderr)
        sys.exit(1)

    # Batch inference
    results = []
    ok_count = 0
    ng_count = 0
    start_time = time.time()

    for i, img_path in enumerate(image_files, 1):
        image_bytes = img_path.read_bytes()
        result = client.predict(image_bytes, frame_id=img_path.name)

        # Flatten the representative region (largest area) into the first columns of the CSV.
        # result.regions is sorted by area descending, so the first entry is the largest.
        top = result.regions[0] if result.regions else None
        top_cx, top_cy = (top.centroid if top else (0, 0))
        # Record every NG centroid pipe-separated (useful for downstream coordinate visualization)
        all_centroids = "|".join(
            f"{r.class_name}:{r.centroid[0]},{r.centroid[1]}"
            for r in result.regions
        )
        summary = result.summary
        row = {
            "file": img_path.name,
            "judgement": result.judgement,
            "defect_found": result.defect_found,
            "num_regions": len(result.regions),
            "classes": ", ".join(r.class_name for r in result.regions),
            "fg_ratio": f"{summary.get('fg_ratio', 0.0):.6f}",
            "max_confidence": f"{summary.get('max_confidence', 0.0):.4f}",
            "top_class": (top.class_name if top else ""),
            "top_area_px": (top.area_px if top else 0),
            "top_centroid_x": top_cx,
            "top_centroid_y": top_cy,
            "all_centroids": all_centroids,
            "result_id": result.result_id,
        }
        results.append(row)

        if result.judgement == "OK":
            ok_count += 1
        else:
            ng_count += 1

        print(f"[{i}/{len(image_files)}] {img_path.name}: {result.judgement}")

    elapsed = time.time() - start_time

    # Write CSV (utf-8-sig for Excel compatibility)
    fieldnames = [
        "file", "judgement", "defect_found", "num_regions", "classes",
        "fg_ratio", "max_confidence",
        "top_class", "top_area_px", "top_centroid_x", "top_centroid_y",
        "all_centroids",
        "result_id",
    ]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print("\n--- Done ---")
    print(f"Total: {len(results)} image(s)")
    print(f"OK: {ok_count} / NG: {ng_count}")
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / len(results):.2f} s/image)")
    print(f"CSV written to: {args.output}")

    # End session
    client.stop_session()


if __name__ == "__main__":
    main()
