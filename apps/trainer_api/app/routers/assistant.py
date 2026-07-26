# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import shlex
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select

from segcore.image_io import imread as _imread
from segcore.image_io import imwrite as _imwrite

from ..core.annotate_index import load_annotate_index, save_annotate_index
from ..core.config import OUTPUT_STRIDE
from ..core.dataset_prep import prepare_annotate_dataset
from ..core.paths import (
    annotate_images_dir,
    annotate_masks_dir,
    assistant_context_path,
    assistant_dir,
    assistant_thread_path,
    project_dir,
    recipes_dir,
    run_dir,  # noqa: F401
    )
from ..core.prediction_engine import ensure_prediction_artifacts, resolve_predict_context
from ..core.recipe_engine import apply_recipe_to_image
from ..core.state import RUN_FLAGS
from ..core.training_runner import _launch_training_run
from ..db import get_engine
from ..models import TrainingRun
from ..schemas import TrainRequest

router = APIRouter()


def ensure_assistant_files(project_id: str) -> None:
    a_dir = assistant_dir(project_id)
    a_dir.mkdir(parents=True, exist_ok=True)
    context = assistant_context_path(project_id)
    if not context.exists():
        context.write_text(
            "# Project Context\n\n"
            "- Goals:\n"
            "- Defect definitions:\n"
            "- Constraints:\n"
            "- Notes:\n",
            encoding="utf-8",
        )
    thread = assistant_thread_path(project_id)
    if not thread.exists():
        thread.write_text("", encoding="utf-8")


def read_assistant_context(project_id: str) -> str:
    ensure_assistant_files(project_id)
    path = assistant_context_path(project_id)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_assistant_context(project_id: str, markdown: str) -> None:
    ensure_assistant_files(project_id)
    assistant_context_path(project_id).write_text(markdown, encoding="utf-8")


def read_assistant_thread(project_id: str, limit: int = 200) -> list[dict[str, Any]]:
    ensure_assistant_files(project_id)
    path = assistant_thread_path(project_id)
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []
    if limit > 0 and len(lines) > limit:
        lines = lines[-limit:]
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
            if isinstance(item, dict):
                out.append(item)
        except Exception:
            continue
    return out


def append_assistant_message(
    project_id: str,
    role: str,
    content: str,
    kind: str = "message",
) -> dict[str, Any]:
    ensure_assistant_files(project_id)
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "kind": kind,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = assistant_thread_path(project_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(message, ensure_ascii=False))
        fh.write("\n")
    return message


def _latest_project_run(project_id: str) -> TrainingRun | None:
    engine = get_engine()
    with Session(engine) as session:
        rows = session.exec(select(TrainingRun).where(TrainingRun.project_id == project_id)).all()
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda r: r.updated_at, reverse=True)
    return rows_sorted[0]


def _resolve_assistant_run_id(project_id: str, token: str) -> str | None:
    raw = (token or "").strip().lower()
    if raw in {"latest", "last"}:
        latest = _latest_project_run(project_id)
        return latest.run_id if latest else None
    return token.strip() if token else None


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_train_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not tokens:
        return overrides
    if len(tokens) == 1 and tokens[0].strip().startswith("{"):
        raw = json.loads(tokens[0])
        if isinstance(raw, dict):
            return raw
        raise ValueError("train config json must be an object")
    for tok in tokens:
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        k = key.strip().lower()
        v = val.strip()
        if not k:
            continue
        if k in {"epochs", "batch_size", "patch_size", "patches_per_image", "output_stride", "early_stopping_patience", "min_epochs"}:
            overrides[k] = int(v)
            continue
        if k in {"lr", "crop_scale", "fg_patch_prob", "augment_hflip_prob", "augment_vflip_prob", "augment_rotate90_prob", "augment_brightness", "augment_contrast", "augment_noise_std", "class_weight_strength", "background_weight_boost", "dice_weight"}:
            overrides[k] = float(v)
            continue
        if k in {"crop_foreground", "augment_enabled", "use_class_weights"}:
            overrides[k] = _coerce_bool(v)
            continue
        if k in {"model_name", "loss_type", "notes"}:
            overrides[k] = v
            continue
        if k in {"input_size", "input"}:
            if "x" in v.lower():
                w_raw, h_raw = v.lower().split("x", 1)
                overrides["input_size"] = [int(w_raw), int(h_raw)]
            else:
                size = int(v)
                overrides["input_size"] = [size, size]
            continue
    return overrides


def run_assistant_command(project_id: str, command: str) -> dict[str, Any]:
    text = command.strip()
    if not text:
        return {"ok": False, "message": "empty command"}
    if not text.startswith("/"):
        return {
            "ok": True,
            "mode": "note",
            "persisted": False,
            "message": "non-command text is not persisted here; post via /assistant/thread to save",
        }
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    head = parts[0].lower()
    if head == "/help":
        msg = ("commands: /help, /prepare, /runs, /train start [json|key=value...], "
               "/metrics <run_id|latest>, /predict <run_id|latest> <item_id> [backend], "
               "/stop <run_id|latest>, /recipe list|active|apply")
        return {"ok": True, "mode": "help", "message": msg}
    if head == "/prepare":
        report = prepare_annotate_dataset(project_id)
        msg = (f"prepared with_mask={report.get('with_mask', 0)}, "
               f"train={report.get('train_count', 0)}, val={report.get('val_count', 0)}")
        return {"ok": True, "mode": "prepare", "message": msg, "report": report}
    if head == "/runs":
        engine = get_engine()
        with Session(engine) as session:
            rows = session.exec(select(TrainingRun).where(TrainingRun.project_id == project_id)).all()
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        latest = sorted(rows, key=lambda r: r.updated_at, reverse=True)[0] if rows else None
        return {
            "ok": True,
            "mode": "runs",
            "message": (
                f"runs total={len(rows)} "
                f"(completed={counts.get('completed', 0)}, running={counts.get('running', 0)}, "
                f"failed={counts.get('failed', 0)}, stopped={counts.get('stopped', 0)})"
            ),
            "counts": counts,
            "latest_run_id": latest.run_id if latest else None,
            "latest_status": latest.status if latest else None,
        }
    if head == "/train":
        if len(parts) < 2:
            return {"ok": False, "message": "usage: /train start [json|key=value ...]"}
        action = parts[1].strip().lower()
        if action not in {"start"}:
            return {"ok": False, "message": "usage: /train start [json|key=value ...]"}
        try:
            base = TrainRequest().model_dump()
            overrides = _parse_train_overrides(parts[2:])
            merged = dict(base)
            merged.update(overrides)
            if merged.get("output_stride") not in (1, 2, 4):
                merged["output_stride"] = OUTPUT_STRIDE
            payload = TrainRequest(**merged)
            record = _launch_training_run(project_id, payload.model_dump())
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
            return {"ok": False, "message": f"train start failed: {detail}"}
        except Exception as exc:
            return {"ok": False, "message": f"train start failed: {exc}"}
        return {
            "ok": True,
            "mode": "train_start",
            "run_id": record.run_id,
            "message": f"train started: run_id={record.run_id}",
            "config": payload.model_dump(),
        }
    if head == "/metrics":
        if len(parts) < 2:
            return {"ok": False, "message": "usage: /metrics <run_id|latest>"}
        resolved_run_id = _resolve_assistant_run_id(project_id, parts[1])
        if not resolved_run_id:
            return {"ok": False, "message": "run not found"}
        metrics_path = run_dir(project_id, resolved_run_id) / "metrics.json"
        if not metrics_path.exists():
            return {"ok": False, "message": f"metrics not found: {resolved_run_id}"}
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return {"ok": False, "message": f"metrics parse failed: {resolved_run_id}"}
        best_f1 = metrics.get("best_F1_val")
        best_miou = metrics.get("best_mIoU_val")
        best_epoch = metrics.get("best_epoch")
        return {
            "ok": True,
            "mode": "metrics",
            "run_id": resolved_run_id,
            "message": (
                f"run={resolved_run_id} "
                f"best_F1={best_f1 if best_f1 is not None else '-'} "
                f"best_mIoU={best_miou if best_miou is not None else '-'} "
                f"best_epoch={best_epoch if best_epoch is not None else '-'}"
            ),
            "summary": {
                "best_F1_val": best_f1,
                "best_mIoU_val": best_miou,
                "best_epoch": best_epoch,
            },
        }
    if head == "/predict":
        if len(parts) < 3:
            return {"ok": False, "message": "usage: /predict <run_id|latest> <item_id> [backend]"}
        resolved_run_id = _resolve_assistant_run_id(project_id, parts[1])
        if not resolved_run_id:
            return {"ok": False, "message": "run not found"}
        item_id = parts[2]
        backend = parts[3] if len(parts) >= 4 else "onnx"
        try:
            run_path, model_path, resolved_backend = resolve_predict_context(project_id, resolved_run_id, backend)
            pred_path, conf_path, score = ensure_prediction_artifacts(
                project_id,
                run_path,
                model_path,
                item_id,
                resolved_backend,
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
            return {"ok": False, "message": f"predict failed: {detail}"}
        return {
            "ok": True,
            "mode": "predict",
            "run_id": resolved_run_id,
            "item_id": item_id,
            "backend": resolved_backend,
            "message": (
                f"predict ok: run={resolved_run_id} item={item_id} backend={resolved_backend} "
                f"fg_ratio={score.get('foreground_ratio', 0):.4f} "
                f"mean_conf={score.get('mean_confidence', 0):.4f}"
            ),
            "score": score,
            "artifacts": {
                "mask_path": str(pred_path),
                "confidence_path": str(conf_path),
            },
        }
    if head == "/stop":
        if len(parts) < 2:
            return {"ok": False, "message": "usage: /stop <run_id|latest>"}
        run_id = _resolve_assistant_run_id(project_id, parts[1])
        if not run_id:
            return {"ok": False, "message": "run not found"}
        stop_event = RUN_FLAGS.get(run_id)
        if stop_event is None:
            return {"ok": False, "message": f"run not active: {run_id}"}
        stop_event.set()
        engine = get_engine()
        with Session(engine) as session:
            record = session.exec(
                select(TrainingRun).where(TrainingRun.project_id == project_id, TrainingRun.run_id == run_id)
            ).first()
            if record:
                record.status = "stopped"
                record.updated_at = datetime.now(timezone.utc)
                session.add(record)
                session.commit()
        return {"ok": True, "mode": "stop", "message": f"stop requested: {run_id}", "run_id": run_id}
    if head == "/recipe":
        action = parts[1].lower() if len(parts) >= 2 else "help"
        if action == "list":
            rdir = recipes_dir(project_id)
            if not rdir.exists():
                return {"ok": True, "mode": "recipe_list", "message": "no recipes", "recipes": []}
            recipes = []
            for p in sorted(rdir.glob("*.json")):
                if p.name == "active.json":
                    continue
                try:
                    r = json.loads(p.read_text(encoding="utf-8"))
                    recipes.append({"id": r.get("id"), "name": r.get("name", "untitled"), "rules": len(r.get("rules", []))})
                except Exception:
                    pass
            return {
                "ok": True,
                "mode": "recipe_list",
                "message": f"{len(recipes)} recipe(s): " + ", ".join(r["name"] for r in recipes) if recipes else "no recipes",
                "recipes": recipes,
            }
        if action == "active":
            rdir = recipes_dir(project_id)
            active_path = rdir / "active.json"
            if not active_path.exists():
                return {"ok": True, "mode": "recipe_active", "message": "no active recipe", "recipe": None}
            try:
                active = json.loads(active_path.read_text(encoding="utf-8"))
                rid = active.get("recipe_id")
                rp = rdir / f"{rid}.json"
                if rp.exists():
                    r = json.loads(rp.read_text(encoding="utf-8"))
                    return {
                        "ok": True,
                        "mode": "recipe_active",
                        "message": f"active: {r.get('name', 'untitled')} ({len(r.get('rules', []))} rules)",
                        "recipe": r,
                    }
            except Exception:
                pass
            return {"ok": True, "mode": "recipe_active", "message": "no active recipe", "recipe": None}
        if action == "apply":
            rdir = recipes_dir(project_id)
            active_path = rdir / "active.json"
            if not active_path.exists():
                return {"ok": False, "message": "no active recipe — import one first"}
            try:
                active = json.loads(active_path.read_text(encoding="utf-8"))
                rid = active.get("recipe_id")
                rp = rdir / f"{rid}.json"
                if not rp.exists():
                    return {"ok": False, "message": "active recipe file missing"}
                recipe = json.loads(rp.read_text(encoding="utf-8"))
                index = load_annotate_index(project_id)
                items = index.get("items", [])
                masks_dp = annotate_masks_dir(project_id)
                masks_dp.mkdir(parents=True, exist_ok=True)
                images_dp = annotate_images_dir(project_id)
                applied = 0
                for item in items:
                    iid = item.get("id", "")
                    existing = masks_dp / f"{iid}.png"
                    if existing.exists():
                        continue
                    img_path = images_dp / item["filename"]
                    if not img_path.exists():
                        continue
                    img_bgr = _imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    mask = apply_recipe_to_image(img_bgr, recipe)
                    _imwrite(str(existing), mask)
                    ann = item.get("annotation", {})
                    ann["hasMask"] = True
                    ann["revision"] = ann.get("revision", 0) + 1
                    ann["lastSavedAt"] = datetime.now(timezone.utc).isoformat()
                    item["annotation"] = ann
                    applied += 1
                save_annotate_index(project_id, index)
                return {
                    "ok": True,
                    "mode": "recipe_apply",
                    "message": f"recipe applied to {applied} image(s)",
                    "applied": applied,
                }
            except Exception as exc:
                return {"ok": False, "message": f"recipe apply failed: {exc}"}
        return {"ok": False, "message": "usage: /recipe list|active|apply"}

    return {"ok": False, "message": f"unknown command: {head}"}


@router.get("/projects/{project_id}/assistant/context")
def get_assistant_context(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    markdown = read_assistant_context(project_id)
    return {
        "project_id": project_id,
        "markdown": markdown,
    }


@router.put("/projects/{project_id}/assistant/context")
def put_assistant_context(project_id: str, payload: dict[str, Any]):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    markdown = payload.get("markdown")
    if not isinstance(markdown, str):
        raise HTTPException(status_code=400, detail="markdown is required")
    write_assistant_context(project_id, markdown)
    return {
        "project_id": project_id,
        "markdown": markdown,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/projects/{project_id}/assistant/thread")
def get_assistant_thread(project_id: str, limit: int = Query(200, ge=1, le=1000)):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    messages = read_assistant_thread(project_id, limit=limit)
    return {
        "project_id": project_id,
        "messages": messages,
    }


@router.post("/projects/{project_id}/assistant/thread/messages")
def post_assistant_thread_message(project_id: str, payload: dict[str, Any]):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    content = payload.get("content")
    role = str(payload.get("role") or "user").strip().lower()
    if role not in {"user", "assistant", "system"}:
        raise HTTPException(status_code=400, detail="role must be one of: user, assistant, system")
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    kind = str(payload.get("kind") or "message").strip().lower()
    message = append_assistant_message(project_id, role=role, content=content.strip(), kind=kind or "message")
    return {"status": "ok", "message": message}


@router.post("/projects/{project_id}/assistant/command")
def post_assistant_command(project_id: str, payload: dict[str, Any]):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise HTTPException(status_code=400, detail="command is required")
    user_message = append_assistant_message(project_id, role="user", content=command.strip(), kind="command")
    result = run_assistant_command(project_id, command)
    assistant_message = append_assistant_message(
        project_id,
        role="assistant",
        content=str(result.get("message", "")),
        kind="command_result",
    )
    return {
        "status": "ok",
        "user_message": user_message,
        "assistant_message": assistant_message,
        "result": result,
    }
