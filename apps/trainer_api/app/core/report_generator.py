# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Report generator for model evaluation and batch inspection reports.

Produces HTML (canonical), PDF, and Excel outputs from training metrics,
prediction artifacts, and training logs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..core.paths import project_dir, resolve_run_path, run_dir
from ..schemas import (
    ReportFileInfo,
    ReportGenerateResponse,
    ReportOptions,
)

logger = logging.getLogger(__name__)

# Seg-Studio app version shown in report footer / traceability.
# Single source within the backend; keep in sync with
# apps/trainer_ui/package.json and packages/*/pyproject.toml.
APP_VERSION = "0.9.6"

# Report building blocks were extracted verbatim to report_builders.
# Re-imported here so the entry points and any external users keep working.
from .report_builders import (  # noqa: F401 — backward-compat re-exports
    _EPOCH_RE,
    _build_confusion_matrix_chart,
    _build_hard_case_images,
    _build_learning_curves,
    _build_per_class_metrics,
    _build_score_distribution,
    _build_threshold_chart,
    _compute_instance_recall,
    _compute_model_hash,
    _encode_image_base64,
    _fig_to_base64,
    _generate_excel,
    _html_to_pdf,
    _load_project_info,
    _parse_epoch_history,
    _render_html,
    _report_labels,
    _select_hard_cases,
)


def generate_model_eval_report(
    project_id: str,
    run_id: str,
    formats: list[str],
    options: ReportOptions,
    lang: str = "en",
) -> ReportGenerateResponse:
    """Generate a model evaluation report."""
    L = _report_labels(lang)
    rdir = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    pdir = project_dir(project_id)

    # Load data sources
    metrics = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
    config_path = rdir / "train_config.json"
    train_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    project_info = _load_project_info(project_id)
    class_map = project_info["class_map"]

    # Parse epoch history from train.log
    epoch_history = _parse_epoch_history(rdir / "train.log") if options.include_learning_curves else []

    # Build per-class metrics
    per_class = _build_per_class_metrics(metrics, class_map)

    # Build class names list (including background)
    class_names_all = [
        class_map.get(str(c["id"]), f"class_{c['id']}")
        for c in project_info["classes"]
        if c.get("active", True)
    ]
    if not class_names_all or class_names_all[0] != "background":
        class_names_all = ["background"] + class_names_all

    # Charts
    learning_curves_img = _build_learning_curves(epoch_history) if options.include_learning_curves else ""
    confusion_matrix_img = ""
    if options.include_confusion_matrix and metrics.get("confusion_matrix_val"):
        confusion_matrix_img = _build_confusion_matrix_chart(
            metrics["confusion_matrix_val"], class_names_all
        )
    threshold_chart_img = _build_threshold_chart(epoch_history) if options.include_threshold_analysis else ""

    # Score distribution
    preds_dir = rdir / "predictions"
    score_dist_img = _build_score_distribution(preds_dir) if preds_dir.exists() else ""

    # Hard cases
    hard_cases: dict = {}
    hard_case_images: list[dict] = []
    if options.include_hard_cases and preds_dir.exists():
        hard_cases = _select_hard_cases(preds_dir, options.hard_case_top_n)
        images_dir = pdir / "prepared" / "images"
        hard_case_images = _build_hard_case_images(
            hard_cases, images_dir, preds_dir, max_items=min(5, options.hard_case_top_n),
        )

    # Instance recall
    instance_recall_data: dict = {}
    if options.include_instance_recall and preds_dir.exists():
        gt_masks_dir = pdir / "prepared" / "masks"
        if gt_masks_dir.exists():
            active_ids = [c["id"] for c in project_info["classes"] if c.get("active", True)]
            instance_recall_data = _compute_instance_recall(
                gt_masks_dir, preds_dir, active_ids,
            )

    # Model hash (traceability)
    model_hash = _compute_model_hash(rdir / "model.pt")

    # Dataset stats
    dataset_stats = metrics.get("dataset_stats", {})

    # KPIs (localized label + value)
    best_f1 = round(metrics.get("best_F1_val", 0), 4)
    kpis = [
        {"label": L["kpi_best_f1"], "value": best_f1},
        {"label": L["kpi_best_miou"], "value": round(metrics.get("best_mIoU_val", 0), 4)},
        {"label": L["kpi_best_epoch"], "value": metrics.get("best_epoch", "N/A")},
        {"label": L["kpi_total_epochs"], "value": epoch_history[-1]["total_epochs"] if epoch_history else train_config.get("epochs", "N/A")},
        {"label": L["kpi_final_loss"], "value": round(metrics.get("loss", 0), 4)},
        {"label": L["kpi_optimal_threshold"], "value": metrics.get("optimal_threshold", "N/A")},
        {"label": L["kpi_ece"], "value": round(metrics.get("ece", 0), 4) if metrics.get("ece") is not None else "N/A"},
        {"label": L["kpi_train_images"], "value": dataset_stats.get("num_train", "N/A")},
        {"label": L["kpi_val_images"], "value": dataset_stats.get("num_val", "N/A")},
        {"label": L["kpi_input_size"], "value": str(dataset_stats.get("input_size", "N/A"))},
        {"label": L["kpi_architecture"], "value": train_config.get("arch", "N/A")},
    ]

    # Prepare report output directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_id = f"{timestamp}_{run_id[:8]}"
    report_dir = pdir / "reports" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build template context
    context = {
        "title": f"{L['report_title']} - {project_info['project_name']}",
        "lang": lang,
        "L": L,
        "best_f1": best_f1,
        "project_name": project_info["project_name"],
        "project_id": project_id,
        "run_id": run_id,
        "report_type": "model_eval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kpis": kpis,
        "per_class_metrics": per_class,
        "learning_curves_img": learning_curves_img,
        "confusion_matrix_img": confusion_matrix_img,
        "threshold_chart_img": threshold_chart_img,
        "score_dist_img": score_dist_img,
        "hard_cases": hard_cases,
        "hard_case_images": hard_case_images,
        "instance_recall": instance_recall_data,
        "model_hash": model_hash,
        "train_config": train_config,
        "dataset_stats": dataset_stats,
        "epoch_history": epoch_history,
        "app_version": APP_VERSION,
    }

    # Generate outputs
    files: list[ReportFileInfo] = []

    if "html" in formats:
        html_content = _render_html("report_model_eval.html", context)
        html_path = report_dir / "report.html"
        html_path.write_text(html_content, encoding="utf-8")
        files.append(ReportFileInfo(
            filename="report.html", format="html",
            size_bytes=html_path.stat().st_size,
        ))

    if "pdf" in formats:
        # Need HTML content for PDF conversion
        if "html" not in formats:
            html_content = _render_html("report_model_eval.html", context)
        pdf_path = _html_to_pdf(html_content, report_dir / "report.pdf")
        if pdf_path:
            files.append(ReportFileInfo(
                filename="report.pdf", format="pdf",
                size_bytes=pdf_path.stat().st_size,
            ))

    if "xlsx" in formats:
        excel_data = {
            "title": context["title"],
            "kpis": kpis,
            "per_class_metrics": per_class,
            "hard_cases": hard_cases,
            "epoch_history": epoch_history,
        }
        xlsx_path = _generate_excel(excel_data, report_dir / "report.xlsx")
        files.append(ReportFileInfo(
            filename="report.xlsx", format="xlsx",
            size_bytes=xlsx_path.stat().st_size,
        ))

    # Save meta.json
    meta = {
        "report_id": report_id,
        "report_type": "model_eval",
        "run_id": run_id,
        "files": [f.model_dump() for f in files],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ReportGenerateResponse(
        report_id=report_id,
        report_type="model_eval",
        files=files,
        status="completed",
        created_at=datetime.now(timezone.utc),
    )


def generate_batch_report(
    project_id: str,
    run_id: str,
    formats: list[str],
    options: ReportOptions,
) -> ReportGenerateResponse:
    """Generate a batch inspection report."""
    rdir = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    pdir = project_dir(project_id)

    metrics = json.loads((rdir / "metrics.json").read_text(encoding="utf-8"))
    project_info = _load_project_info(project_id)
    class_map = project_info["class_map"]

    preds_dir = rdir / "predictions"
    if not preds_dir.exists():
        raise ValueError("No predictions found — run inference first")

    # Collect all prediction scores
    all_scores: list[dict] = []
    for score_path in sorted(preds_dir.glob("*.score.json")):
        try:
            data = json.loads(score_path.read_text(encoding="utf-8"))
            data["_stem"] = score_path.stem.replace(".score", "")
            all_scores.append(data)
        except Exception:
            continue

    # Determine threshold
    threshold = options.confidence_threshold or metrics.get("optimal_threshold", 0.5)

    # Classify OK/NG
    ok_count = sum(1 for s in all_scores if s.get("foreground_ratio", 0) < 0.001)
    ng_count = len(all_scores) - ok_count
    ng_rate = ng_count / max(len(all_scores), 1)

    # Per-class defect counts
    class_defect_counts: dict[str, int] = {}
    for s in all_scores:
        for cid, conf in s.get("per_class_mean_confidence", {}).items():
            if conf > threshold:
                class_defect_counts[cid] = class_defect_counts.get(cid, 0) + 1

    # Score distribution
    score_dist_img = _build_score_distribution(preds_dir)

    # Detail table
    detail_rows = []
    for s in all_scores:
        detail_rows.append({
            "image": s["_stem"],
            "verdict": "NG" if s.get("foreground_ratio", 0) >= 0.001 else "OK",
            "mean_confidence": round(s.get("mean_confidence", 0), 4),
            "fg_ratio": round(s.get("foreground_ratio", 0), 6),
            "inference_ms": round(s.get("inference_ms", 0), 1),
        })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_id = f"{timestamp}_{run_id[:8]}"
    report_dir = pdir / "reports" / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "title": f"Batch Inspection Report - {project_info['project_name']}",
        "project_name": project_info["project_name"],
        "project_id": project_id,
        "run_id": run_id,
        "report_type": "batch",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": len(all_scores),
        "ok_count": ok_count,
        "ng_count": ng_count,
        "ng_rate": round(ng_rate, 4),
        "threshold": threshold,
        "class_defect_counts": {class_map.get(k, f"class_{k}"): v for k, v in class_defect_counts.items()},
        "score_dist_img": score_dist_img,
        "detail_rows": detail_rows,
        "app_version": APP_VERSION,
    }

    files: list[ReportFileInfo] = []

    if "html" in formats:
        html_content = _render_html("report_batch.html", context)
        html_path = report_dir / "report.html"
        html_path.write_text(html_content, encoding="utf-8")
        files.append(ReportFileInfo(
            filename="report.html", format="html",
            size_bytes=html_path.stat().st_size,
        ))

    if "pdf" in formats:
        if "html" not in formats:
            html_content = _render_html("report_batch.html", context)
        pdf_path = _html_to_pdf(html_content, report_dir / "report.pdf")
        if pdf_path:
            files.append(ReportFileInfo(
                filename="report.pdf", format="pdf",
                size_bytes=pdf_path.stat().st_size,
            ))

    if "xlsx" in formats:
        excel_data = {
            "title": context["title"],
            "kpis": {
                "Total Images": len(all_scores),
                "OK": ok_count,
                "NG": ng_count,
                "NG Rate": f"{ng_rate:.2%}",
                "Threshold": threshold,
            },
            "per_class_metrics": [
                {"name": name, "count": count}
                for name, count in context["class_defect_counts"].items()
            ],
            "hard_cases": {},
            "epoch_history": [],
        }
        xlsx_path = _generate_excel(excel_data, report_dir / "report.xlsx")
        files.append(ReportFileInfo(
            filename="report.xlsx", format="xlsx",
            size_bytes=xlsx_path.stat().st_size,
        ))

    meta = {
        "report_id": report_id,
        "report_type": "batch",
        "run_id": run_id,
        "files": [f.model_dump() for f in files],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (report_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ReportGenerateResponse(
        report_id=report_id,
        report_type="batch",
        files=files,
        status="completed",
        created_at=datetime.now(timezone.utc),
    )
