#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Minimal single-image inference example.

Usage:
    python quick_start.py
"""
from __future__ import annotations

import sys

import requests
from seg_sdk import SegClient

# ---- Settings ----
BASE_URL = "http://localhost:8002"
PROJECT_ID = "your-project-id"
RUN_ID = "your-run-id"
IMAGE_PATH = "test.jpg"


def main() -> None:
    # Create the client
    client = SegClient(BASE_URL, timeout=30)

    try:
        # Start a session (model load + warm-up)
        print("Starting session...")
        client.start_session(project_id=PROJECT_ID, run_id=RUN_ID, backend="onnx")
        print("Session started")

        # Read the image and run inference
        image_bytes = open(IMAGE_PATH, "rb").read()
        result = client.predict(image_bytes)

        # Print the result
        print(f"judgement    : {result.judgement}")
        print(f"defect_found : {result.defect_found}")
        print(f"regions      : {len(result.regions)}")

        # Summary (dict) — whole-image statistics
        #   fg_ratio       : ratio of defect pixels to the image (0.0-1.0)
        #   max_confidence : highest confidence in the image (0.0-1.0)
        #   num_defects    : number of detected regions (= len(result.regions))
        s = result.summary
        print("Summary:")
        print(f"  fg_ratio       : {s.get('fg_ratio', 0.0):.4%}  (defect pixel ratio)")
        print(f"  max_confidence : {s.get('max_confidence', 0.0):.3f}  (max confidence in image)")
        print(f"  num_defects    : {s.get('num_defects', 0)}")

        # Latency (dict) — processing time breakdown [ms]
        #   decode       : image decode
        #   inference    : model inference (sliding-window)
        #   postprocess  : CCA + region extraction
        #   total        : end to end
        lat = result.latency_ms
        print("Latency [ms]:")
        print(f"  decode={lat.get('decode', 0)}  "
              f"inference={lat.get('inference', 0)}  "
              f"postprocess={lat.get('postprocess', 0)}  "
              f"total={lat.get('total', 0)}")

        # Each NG region — sorted by area, largest first
        if result.regions:
            print("Detected defect regions:")
            for i, region in enumerate(result.regions, 1):
                x, y, w, h = region.bbox
                cx, cy = region.centroid
                print(
                    f"  [{i}] {region.class_name} (id={region.class_id}) "
                    f"area={region.area_px}px  "
                    f"bbox=(x={x}, y={y}, w={w}, h={h})  "
                    f"centroid=({cx}, {cy})  "
                    f"conf={region.confidence:.3f}"
                )

    except FileNotFoundError:
        print(f"Error: image file not found: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(
            f"Error: cannot connect to server: {BASE_URL}\n"
            "Check that the inference server is running.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code} - {e.response.text}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Stop the session
        try:
            client.stop_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()
