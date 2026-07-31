#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio MCP Server

Bridges the Trainer API so that MCP clients can inspect images, review annotations,
run predictions, check training results, and manage projects programmatically.

Security model:
  - Tools are classified as READ / WRITE / DESTRUCTIVE
  - --policy flag controls which tiers are enabled (default: read-only)
  - DESTRUCTIVE tools require explicit --policy=full
  - All tool calls are logged to stderr with timestamps
  - API responses are sanitized to strip potential prompt injection

Install:  pip install fastmcp httpx
Run:      python scripts/mcp_server.py [--api http://localhost:8002] [--policy read|write|full]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any

import httpx

try:
    from fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "fastmcp is required. Install with: pip install fastmcp httpx\n"
        f"import error: {exc}"
    )


API_BASE = "http://localhost:8002/api/v1"
POLICY = "read"  # "read" | "write" | "full"
mcp = FastMCP("seg-studio")


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _audit(tool_name: str, tier: str, **kwargs: Any) -> None:
    """Log every tool call to stderr for audit trail."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    print(f"[MCP AUDIT] {ts} [{tier}] {tool_name}({params})", file=sys.stderr)


def _check_policy(tier: str, tool_name: str) -> None:
    """Raise if current policy doesn't allow the requested tier."""
    allowed = {"read": {"READ"}, "write": {"READ", "WRITE"}, "full": {"READ", "WRITE", "DESTRUCTIVE"}}
    if tier not in allowed.get(POLICY, {"READ"}):
        raise PermissionError(
            f"Tool '{tool_name}' requires '{tier}' permission, "
            f"but current policy is '{POLICY}'. "
            f"Restart with --policy={'write' if tier == 'WRITE' else 'full'} to enable."
        )


_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|all|above)\s+instructions|"
    r"you\s+are\s+now|system\s*:\s*|<\s*/?\s*system|"
    r"IMPORTANT\s*:\s*(delete|drop|remove|execute|run|ignore))",
    re.IGNORECASE,
)


def _sanitize(data: Any) -> Any:
    """Strip potential prompt-injection patterns from string fields in API responses."""
    if isinstance(data, str):
        if _INJECTION_PATTERNS.search(data):
            return f"[SANITIZED: suspicious content removed] (original length: {len(data)})"
        return data
    if isinstance(data, dict):
        return {k: _sanitize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize(v) for v in data]
    return data


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = f"{API_BASE}{path}"
    with httpx.Client(timeout=120.0) as client:
        resp = client.request(method, url, json=payload)
    resp.raise_for_status()
    if not resp.content:
        return {"status": "ok"}
    return _sanitize(resp.json())


def _request_bytes(method: str, path: str) -> bytes:
    url = f"{API_BASE}{path}"
    with httpx.Client(timeout=120.0) as client:
        resp = client.request(method, url)
    resp.raise_for_status()
    return resp.content


# ===================================================================
# 1. Projects  [READ]
# ===================================================================

@mcp.tool()
def projects_list() -> Any:
    """[READ] List all projects (id, name, description)."""
    _check_policy("READ", "projects_list")
    _audit("projects_list", "READ")
    return _request("GET", "/projects")


@mcp.tool()
def projects_summary() -> Any:
    """[READ] List projects with image_count, mask_count, first_filename."""
    _check_policy("READ", "projects_summary")
    _audit("projects_summary", "READ")
    return _request("GET", "/projects/summary")


@mcp.tool()
def project_get(project_id: str) -> Any:
    """[READ] Get detailed info for one project."""
    _check_policy("READ", "project_get")
    _audit("project_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}")


# ===================================================================
# 2. Dataset / Image Inspection  [READ]
# ===================================================================

@mcp.tool()
def dataset_images(project_id: str) -> Any:
    """[READ] List all images in a project with annotation status.

    Returns per-image: id, name, filename, set (train/test/none),
    width, height, annotation.hasMask, annotation.revision.
    Use this to find which images have masks and which don't.
    """
    _check_policy("READ", "dataset_images")
    _audit("dataset_images", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/datasets/annotate?sync=false")


@mcp.tool()
def dataset_export_zip(project_id: str) -> Any:
    """[READ] Get the dataset export URL for reference."""
    _check_policy("READ", "dataset_export_zip")
    _audit("dataset_export_zip", "READ", project_id=project_id)
    return {"export_url": f"{API_BASE}/projects/{project_id}/datasets/export"}


# ===================================================================
# 3. Classes  [READ / WRITE]
# ===================================================================

@mcp.tool()
def classes_get(project_id: str) -> Any:
    """[READ] Get class definitions (id, name, color, active) for a project.

    Essential for understanding what each mask value means.
    Class 0 is always background.
    """
    _check_policy("READ", "classes_get")
    _audit("classes_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/classes")


@mcp.tool()
def classes_set(project_id: str, classes_json: str) -> Any:
    """[WRITE] Update class definitions. classes_json is a JSON string of the full
    ClassesPayload: {version, ignore_index, classes: [{id, name, color, active}]}."""
    _check_policy("WRITE", "classes_set")
    _audit("classes_set", "WRITE", project_id=project_id)
    payload = json.loads(classes_json)
    return _request("PUT", f"/projects/{project_id}/classes", payload)


# ===================================================================
# 4. Training  [READ / WRITE / DESTRUCTIVE]
# ===================================================================

@mcp.tool()
def train_runs_list(project_id: str) -> Any:
    """[READ] List training runs with status, best_f1, best_miou, has_model."""
    _check_policy("READ", "train_runs_list")
    _audit("train_runs_list", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/train/runs")


@mcp.tool()
def train_run_get(project_id: str, run_id: str) -> Any:
    """[READ] Get details for a specific training run."""
    _check_policy("READ", "train_run_get")
    _audit("train_run_get", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}")


@mcp.tool()
def train_start(project_id: str, config_json: str = "{}") -> Any:
    """[WRITE] Start training. config_json is a JSON string with optional overrides:
    epochs, batch_size, lr, input_size, patch_size, loss_type, dice_weight, etc.
    Empty string or {} uses auto-tuned defaults.

    NOTE: This starts a GPU-intensive process. Use responsibly.
    """
    _check_policy("WRITE", "train_start")
    _audit("train_start", "WRITE", project_id=project_id, config=config_json[:100])
    payload = json.loads(config_json) if config_json.strip() else {}
    return _request("POST", f"/projects/{project_id}/train", payload)


@mcp.tool()
def train_stop(project_id: str, run_id: str) -> Any:
    """[WRITE] Stop a running training job."""
    _check_policy("WRITE", "train_stop")
    _audit("train_stop", "WRITE", project_id=project_id, run_id=run_id)
    return _request("POST", f"/projects/{project_id}/train/runs/{run_id}/stop")


@mcp.tool()
def train_run_delete(project_id: str, run_id: str) -> Any:
    """[DESTRUCTIVE] Delete a training run and ALL its artifacts (model, logs, metrics).
    This action is IRREVERSIBLE."""
    _check_policy("DESTRUCTIVE", "train_run_delete")
    _audit("train_run_delete", "DESTRUCTIVE", project_id=project_id, run_id=run_id)
    return _request("DELETE", f"/projects/{project_id}/train/runs/{run_id}")


@mcp.tool()
def run_metrics_get(project_id: str, run_id: str) -> Any:
    """[READ] Get training metrics (loss curves, F1, mIoU per epoch) and config.

    Use this to evaluate how well training went.
    Key fields: best_val_f1, best_val_miou, epoch_metrics[].
    """
    _check_policy("READ", "run_metrics_get")
    _audit("run_metrics_get", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/metrics")


@mcp.tool()
def run_logs_get(project_id: str, run_id: str, offset: int = 0) -> Any:
    """[READ] Get training log text (incremental from offset).
    Returns {log: str, total: int}."""
    _check_policy("READ", "run_logs_get")
    _audit("run_logs_get", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/logs?offset={offset}")


# ===================================================================
# 5. Prediction / Score  [READ]
# ===================================================================

@mcp.tool()
def predict_score_get(project_id: str, run_id: str, item_id: str, backend: str = "onnx") -> Any:
    """[READ] Get prediction score for one image.

    Returns: mean_confidence, foreground_mean_confidence, foreground_ratio,
    per_class_mean_confidence (dict of class_id -> confidence).

    Use to check how confident the model is about each image.
    Low confidence = potential NG or difficult area.
    """
    _check_policy("READ", "predict_score_get")
    _audit("predict_score_get", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    b = backend.strip().lower() or "onnx"
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/score?backend={b}")


@mcp.tool()
def predict_score_all(project_id: str, run_id: str, backend: str = "onnx") -> Any:
    """[READ] Run prediction on ALL images and return scores.

    Iterates through every image in the dataset, runs inference,
    and returns a summary with per-image scores sorted by confidence.
    Low-confidence images are likely NG or poorly annotated.

    Returns: {summary: {total, mean_confidence, ...}, images: [{id, name, score}]}
    """
    _check_policy("READ", "predict_score_all")
    _audit("predict_score_all", "READ", project_id=project_id, run_id=run_id)
    b = backend.strip().lower() or "onnx"
    items_data = _request("GET", f"/projects/{project_id}/datasets/annotate?sync=false")
    items = items_data.get("items", [])

    results = []
    errors = []
    for item in items:
        iid = item["id"]
        try:
            score = _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{iid}/score?backend={b}")
            results.append({
                "id": iid,
                "name": item.get("name", ""),
                "set": item.get("set", "none"),
                "has_mask": item.get("annotation", {}).get("hasMask", False),
                **score,
            })
        except Exception as e:
            errors.append({"id": iid, "name": item.get("name", ""), "error": str(e)})

    results.sort(key=lambda r: r.get("mean_confidence", 0))

    if results:
        avg_conf = sum(r.get("mean_confidence", 0) for r in results) / len(results)
        avg_fg = sum(r.get("foreground_mean_confidence", 0) for r in results) / len(results)
        avg_ratio = sum(r.get("foreground_ratio", 0) for r in results) / len(results)
    else:
        avg_conf = avg_fg = avg_ratio = 0

    return {
        "summary": {
            "total_images": len(results),
            "errors": len(errors),
            "avg_mean_confidence": round(avg_conf, 4),
            "avg_foreground_confidence": round(avg_fg, 4),
            "avg_foreground_ratio": round(avg_ratio, 4),
        },
        "images": results,
        "errors": errors if errors else None,
    }


@mcp.tool()
def predict_mask_b64(project_id: str, run_id: str, item_id: str, backend: str = "onnx") -> Any:
    """[READ] Get predicted segmentation mask as base64 PNG.

    The mask is a single-channel image where pixel values = class IDs.
    Combine with classes_get() to understand what each value means.
    """
    _check_policy("READ", "predict_mask_b64")
    _audit("predict_mask_b64", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    b = backend.strip().lower() or "onnx"
    raw = _request_bytes("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}.png?backend={b}")
    return {"item_id": item_id, "mask_png_base64": base64.b64encode(raw).decode()}


@mcp.tool()
def predict_confidence_b64(project_id: str, run_id: str, item_id: str, backend: str = "onnx") -> Any:
    """[READ] Get prediction confidence map as base64 PNG (grayscale 0-255).

    Bright = high confidence, Dark = low confidence (potential NG areas).
    """
    _check_policy("READ", "predict_confidence_b64")
    _audit("predict_confidence_b64", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    b = backend.strip().lower() or "onnx"
    raw = _request_bytes("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/confidence.png?backend={b}")
    return {"item_id": item_id, "confidence_png_base64": base64.b64encode(raw).decode()}


# ===================================================================
# 6. Annotation Inspection  [READ]
# ===================================================================

@mcp.tool()
def annotation_status(project_id: str) -> Any:
    """[READ] Get annotation overview: how many images are annotated, train/test split.

    Returns counts and lists of unannotated images.
    Use this to quickly assess dataset readiness.
    """
    _check_policy("READ", "annotation_status")
    _audit("annotation_status", "READ", project_id=project_id)
    items_data = _request("GET", f"/projects/{project_id}/datasets/annotate?sync=false")
    items = items_data.get("items", [])

    total = len(items)
    with_mask = sum(1 for i in items if i.get("annotation", {}).get("hasMask", False))
    no_mask = [{"id": i["id"], "name": i.get("name", "")} for i in items if not i.get("annotation", {}).get("hasMask", False)]
    train = sum(1 for i in items if i.get("set") == "train")
    test = sum(1 for i in items if i.get("set") == "test")
    none_set = sum(1 for i in items if i.get("set") in (None, "none", ""))

    return {
        "total": total,
        "with_mask": with_mask,
        "without_mask": total - with_mask,
        "train": train,
        "test": test,
        "unassigned": none_set,
        "unannotated_images": no_mask if len(no_mask) <= 50 else no_mask[:50],
        "ready_for_training": with_mask >= 2 and train >= 1,
    }


@mcp.tool()
def mask_get_b64(project_id: str, item_id: str) -> Any:
    """[READ] Get the annotation mask for an image as base64 PNG.

    Pixel values = class IDs (0 = background).
    Returns null mask_png_base64 if no mask exists.
    """
    _check_policy("READ", "mask_get_b64")
    _audit("mask_get_b64", "READ", project_id=project_id, item_id=item_id)
    try:
        raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{item_id}.png")
        return {"item_id": item_id, "mask_png_base64": base64.b64encode(raw).decode()}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"item_id": item_id, "mask_png_base64": None, "note": "No mask exists for this image"}
        raise


@mcp.tool()
def image_get_b64(project_id: str, filename: str) -> Any:
    """[READ] Get an original image as base64 (PNG/JPEG).

    Use dataset_images() first to get filenames.
    """
    _check_policy("READ", "image_get_b64")
    _audit("image_get_b64", "READ", project_id=project_id, filename=filename)
    raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/images/{filename}")
    return {"filename": filename, "image_base64": base64.b64encode(raw).decode()}


# ===================================================================
# 7. Export  [WRITE]
# ===================================================================

@mcp.tool()
def export_onnx(project_id: str, run_id: str) -> Any:
    """[WRITE] Export a trained model to ONNX format."""
    _check_policy("WRITE", "export_onnx")
    _audit("export_onnx", "WRITE", project_id=project_id, run_id=run_id)
    return _request("POST", f"/projects/{project_id}/export/onnx?run_id={run_id}")


@mcp.tool()
def export_coreml(project_id: str, run_id: str) -> Any:
    """[WRITE] Export a trained model to CoreML format (for iOS deployment)."""
    _check_policy("WRITE", "export_coreml")
    _audit("export_coreml", "WRITE", project_id=project_id, run_id=run_id)
    return _request("POST", f"/projects/{project_id}/train/runs/{run_id}/export/coreml")


@mcp.tool()
def models_list() -> Any:
    """[READ] List all exported models across all projects."""
    _check_policy("READ", "models_list")
    _audit("models_list", "READ")
    return _request("GET", "/models")


# ===================================================================
# 8. Recipes  [READ / WRITE]
# ===================================================================

@mcp.tool()
def recipes_list(project_id: str) -> Any:
    """[READ] List annotation recipes for a project."""
    _check_policy("READ", "recipes_list")
    _audit("recipes_list", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/recipes")


@mcp.tool()
def recipe_active_get(project_id: str) -> Any:
    """[READ] Get the currently active recipe."""
    _check_policy("READ", "recipe_active_get")
    _audit("recipe_active_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/recipes/active")


@mcp.tool()
def recipe_preview(project_id: str, item_id: str) -> Any:
    """[READ] Preview the active recipe on one image.
    Returns mask_base64, fg_pixels, fg_ratio."""
    _check_policy("READ", "recipe_preview")
    _audit("recipe_preview", "READ", project_id=project_id, item_id=item_id)
    return _request("POST", f"/projects/{project_id}/recipes/preview/{item_id}")


@mcp.tool()
def recipe_apply(project_id: str, item_ids_json: str = "[]") -> Any:
    """[WRITE] Apply active recipe to specified images (or all un-masked if empty).
    item_ids_json: JSON array of item IDs, e.g. '["abc","def"]'.

    This OVERWRITES existing masks for the specified images.
    """
    _check_policy("WRITE", "recipe_apply")
    _audit("recipe_apply", "WRITE", project_id=project_id, item_ids=item_ids_json[:100])
    ids = json.loads(item_ids_json) if item_ids_json.strip() and item_ids_json.strip() != "[]" else None
    payload = {"item_ids": ids} if ids else {}
    return _request("POST", f"/projects/{project_id}/recipes/apply", payload)


# ===================================================================
# 9. Dataset Prepare  [WRITE]
# ===================================================================

@mcp.tool()
def dataset_prepare_annotate(project_id: str) -> Any:
    """[WRITE] Prepare annotate dataset into train/val splits.

    This reassigns images to train/val/test sets.
    """
    _check_policy("WRITE", "dataset_prepare_annotate")
    _audit("dataset_prepare_annotate", "WRITE", project_id=project_id)
    return _request("POST", f"/projects/{project_id}/datasets/annotate/prepare")


# ===================================================================
# 10. Assistant  [READ / WRITE]
# ===================================================================

@mcp.tool()
def assistant_context_get(project_id: str) -> Any:
    """[READ] Get assistant markdown context for a project."""
    _check_policy("READ", "assistant_context_get")
    _audit("assistant_context_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/assistant/context")


@mcp.tool()
def assistant_context_set(project_id: str, markdown: str) -> Any:
    """[WRITE] Save assistant markdown context for a project."""
    _check_policy("WRITE", "assistant_context_set")
    _audit("assistant_context_set", "WRITE", project_id=project_id)
    return _request("PUT", f"/projects/{project_id}/assistant/context", {"markdown": markdown})


@mcp.tool()
def assistant_thread_get(project_id: str, limit: int = 200) -> Any:
    """[READ] Get assistant thread messages for a project."""
    _check_policy("READ", "assistant_thread_get")
    _audit("assistant_thread_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/assistant/thread?limit={int(limit)}")


@mcp.tool()
def assistant_command(project_id: str, command: str) -> Any:
    """[WRITE] Run project assistant command (/help, /prepare, /runs, /stop <run_id>).

    Commands can trigger training, data preparation, and other side effects.
    """
    _check_policy("WRITE", "assistant_command")
    _audit("assistant_command", "WRITE", project_id=project_id, command=command[:100])
    return _request("POST", f"/projects/{project_id}/assistant/command", {"command": command})


# ===================================================================
# 11. Hardware  [READ / WRITE]
# ===================================================================

@mcp.tool()
def hardware_devices() -> Any:
    """[READ] Get available torch devices and current selection."""
    _check_policy("READ", "hardware_devices")
    _audit("hardware_devices", "READ")
    return _request("GET", "/hardware/torch/devices")


@mcp.tool()
def hardware_set_device(device: str) -> Any:
    """[WRITE] Set the active torch device (e.g. 'cuda', 'cpu')."""
    _check_policy("WRITE", "hardware_set_device")
    _audit("hardware_set_device", "WRITE", device=device)
    return _request("PUT", "/hardware/torch/device", {"device": device})


# ===================================================================
# 12. System  [READ]
# ===================================================================

@mcp.tool()
def server_version() -> Any:
    """[READ] Get server version info."""
    _check_policy("READ", "server_version")
    _audit("server_version", "READ")
    return _request("GET", "/version")


@mcp.tool()
def startup_status() -> Any:
    """[READ] Check if the API server is ready."""
    _check_policy("READ", "startup_status")
    _audit("startup_status", "READ")
    return _request("GET", "/startup-status")


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Seg-Studio MCP server")
    parser.add_argument("--api", default="http://localhost:8002", help="Trainer API base URL")
    parser.add_argument(
        "--policy", default="read", choices=["read", "write", "full"],
        help="Security policy: read (default, safe), write (allows modifications), full (allows deletions)"
    )
    args = parser.parse_args()
    global API_BASE, POLICY
    # Accept both forms: bare host (`http://host:8002`) and pre-prefixed
    # (`http://host:8002/api/v1`). The Trainer API only mounts routers under
    # /api/v1 (the SEG_API_TOKEN middleware guards that prefix), so anything
    # else would 404.
    base = args.api.rstrip("/")
    API_BASE = base if base.endswith("/api/v1") else f"{base}/api/v1"
    POLICY = args.policy

    tier_counts = {"READ": 0, "WRITE": 0, "DESTRUCTIVE": 0}
    for tool in mcp._tool_manager._tools.values():
        desc = tool.description or ""
        if "[DESTRUCTIVE]" in desc:
            tier_counts["DESTRUCTIVE"] += 1
        elif "[WRITE]" in desc:
            tier_counts["WRITE"] += 1
        else:
            tier_counts["READ"] += 1

    allowed = {"read": "READ only", "write": "READ + WRITE", "full": "READ + WRITE + DESTRUCTIVE"}
    print(f"[MCP] Policy: {POLICY} ({allowed[POLICY]})", file=sys.stderr)
    print(f"[MCP] Tools: {tier_counts['READ']}R / {tier_counts['WRITE']}W / {tier_counts['DESTRUCTIVE']}D", file=sys.stderr)
    print(f"[MCP] API: {API_BASE}", file=sys.stderr)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
