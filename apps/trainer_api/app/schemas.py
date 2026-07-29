# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    memo: str | None = None
    tags: list[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    memo: str | None = None
    tags: list[str] | None = None


class ProjectRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    memo: str | None = None
    sort_order: int = 0
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
    }

    @field_validator("tags", mode="before")
    @classmethod
    def _decode_tags(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v) if v else []
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        if v is None:
            return []
        return v


class ClassItem(BaseModel):
    id: int
    name: str
    color: list[int]
    active: bool


class ClassesPayload(BaseModel):
    version: int
    ignore_index: int
    classes: list[ClassItem]


class TrainRequest(BaseModel):
    preset: str = Field(default="fast")
    epochs: int = Field(default=80, ge=1)
    batch_size: int = Field(default=8, ge=1)
    lr: float = Field(default=5e-4, gt=0.0)
    input_size: list[int] = Field(default_factory=lambda: [256, 256])
    crop_foreground: bool = Field(default=False)
    crop_scale: float = Field(default=0.7, ge=0.2, le=1.0)
    patch_size: int = Field(default=256, ge=0)
    patches_per_image: int = Field(default=8, ge=1)
    fg_patch_prob: float = Field(default=0.7, ge=0.0, le=1.0)
    augment_enabled: bool = Field(default=True)
    augment_hflip_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    augment_vflip_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    augment_rotate90_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    augment_brightness: float = Field(default=0.15, ge=0.0, le=1.0)
    augment_contrast: float = Field(default=0.15, ge=0.0, le=1.0)
    augment_noise_std: float = Field(default=0.02, ge=0.0, le=0.5)
    output_stride: int = Field(default=2, ge=1)
    use_class_weights: bool = Field(default=True)
    class_weight_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    background_weight_boost: float | None = Field(default=None, ge=1.0, le=3.0)
    loss_type: str | None = Field(default=None)
    dice_weight: float | None = Field(default=None, ge=0.0, le=10.0)
    early_stopping_patience: int = Field(default=15, ge=0)
    min_epochs: int = Field(default=5, ge=1)
    distill_mode: str = Field(default="off")
    distill_feature_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    distill_feature_loss: str = Field(default="smooth_l1")
    distill_teacher_model_dir: str | None = Field(default=None)
    base_channels: int = Field(default=128, ge=8, le=128)
    annotation_patches_only: bool = Field(default=True)
    context_expand: float = Field(default=3.0, ge=0.0, le=10.0)
    arch: str = Field(default="simpleunet")
    ohem_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    tversky_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    tversky_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    tversky_beta: float = Field(default=0.7, ge=0.0, le=1.0)
    tversky_gamma: float = Field(default=1.5, ge=0.0, le=3.0)
    distill_ensemble: bool = Field(default=False)
    distill_ensemble_temperature: float = Field(default=2.0, ge=0.1, le=20.0)
    distill_ensemble_weight: float = Field(default=0.1, ge=0.0, le=10.0)
    postprocess_min_area: int = Field(default=0, ge=0, le=10000)
    deep_supervision: bool = Field(default=False)
    frequency_map: bool = Field(default=False)
    auto_select: bool = Field(default=True)
    auto_config: bool = Field(default=True)
    # ADR-005 Phase D: single-knob replacement for the two flags above.
    # "full" = both on; "recipe_only" = auto_config only; "off" = both off.
    # Legacy flags still win when explicitly present in the request body,
    # so this remains a no-op passthrough until the UI switches to the
    # single toggle in Phase D step 2b.
    auto_mode: str = Field(default="recipe_only")  # "recipe_only" | "off" (legacy "full" coerces to recipe_only)
    model_name: str | None = None
    include_unmasked: bool = Field(default=True)
    val_ratio: float = Field(default=0.15, ge=0.0, le=0.5)
    k_folds: int = Field(default=1, ge=1, le=20)
    split_method: str = Field(default="hash")  # "hash" | "embedding_stratified"
    iterative_mode: bool = Field(default=False)
    auto_epochs: bool = Field(default=True)
    target_recall: float = Field(default=0.90, ge=0.0, le=1.0)
    target_precision: float = Field(default=0.80, ge=0.0, le=1.0)
    iter_max: int = Field(default=3, ge=1, le=10)
    hard_weight_boost: float = Field(default=3.0, ge=1.0, le=10.0)
    target_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.10, ge=0.0, le=0.5)
    # Literal so a typo ("instnace") is a 422, not a silent expensive standard
    # run. "anomaly" stays accepted at schema level only so the launcher can
    # return its explicit ANOMALY_MODE_REMOVED 400 instead of a generic 422.
    training_mode: Literal["standard", "quick", "transfer", "instance", "anomaly"] = (
        Field(default="standard"))
    # Instance mode (docs/design_instance_segmentation_v098.md): synthesis-first
    # instance segmentation — instance GT is composed from semantic masks.
    instance_class_id: int | None = Field(default=None, ge=1, le=254)
    instance_n_train: int = Field(default=500, ge=8, le=5000)
    instance_n_val: int = Field(default=80, ge=2, le=1000)
    instance_objects_min: int = Field(default=4, ge=1, le=64)
    instance_objects_max: int = Field(default=8, ge=1, le=64)
    instance_stack_pair_prob: float = Field(default=0.55, ge=0.0, le=1.0)
    instance_seed: int = Field(default=42, ge=0)
    instance_model_size: str = Field(default="small")  # small | medium | large
    # Tile size the model is shown, for BOTH composition and inference -- the
    # value travels in the export contract so the two cannot drift apart. A
    # mismatch raises nothing; the count simply comes out wrong. 0 disables
    # tiling and composes whole plates. Default mirrors DEFAULT_PATCH_SIZE in
    # core.instance_training; 4096 is a sanity ceiling, not a capability claim.
    instance_patch_size: int = Field(default=768, ge=0, le=4096)
    instance_lr: float = Field(default=1e-4, gt=0.0)
    instance_grad_accum: int = Field(default=2, ge=1, le=16)
    # Optional manual override for the single-object blob-area band (px^2);
    # both must be set, otherwise the composer auto-estimates from the data.
    instance_area_band_min: int | None = Field(default=None, ge=1)
    instance_area_band_max: int | None = Field(default=None, ge=1)


class TrainRunRead(BaseModel):
    run_id: str
    status: str
    model_name: str | None = None
    has_model: bool = False
    best_f1: float | None = None
    best_miou: float | None = None
    queue_position: int | None = None
    optimized_from: str | None = None
    fp16: bool = False
    active_class_ids: list[int] | None = None
    inference_threshold: float | None = None
    training_mode: str | None = None
    fold_index: int | None = None
    total_folds: int | None = None
    cv_group_id: str | None = None
    iter_index: int | None = None
    iter_max_iters: int | None = None
    iter_group_id: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
    }


class ModelRead(BaseModel):
    model_id: str
    project_id: str
    run_id: str | None = None
    created_at: datetime

    model_config = {
        "from_attributes": True,
        "protected_namespaces": (),
    }


# ---------------------------------------------------------------------------
# Report schemas
# ---------------------------------------------------------------------------

class ReportOptions(BaseModel):
    """Optional settings for report generation."""
    hard_case_top_n: int = Field(default=10, ge=1, le=50)
    include_instance_recall: bool = Field(default=True)
    include_hard_cases: bool = Field(default=True)
    include_learning_curves: bool = Field(default=True)
    include_confusion_matrix: bool = Field(default=True)
    include_threshold_analysis: bool = Field(default=True)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ReportGenerateRequest(BaseModel):
    run_id: str
    report_type: str = Field(default="model_eval")  # model_eval | batch
    formats: list[str] = Field(default_factory=lambda: ["html", "pdf", "xlsx"])
    lang: str = Field(default="en")
    options: ReportOptions = Field(default_factory=ReportOptions)


class ReportFileInfo(BaseModel):
    filename: str
    format: str
    size_bytes: int


class ReportGenerateResponse(BaseModel):
    report_id: str
    report_type: str
    files: list[ReportFileInfo]
    status: str = "completed"
    created_at: datetime


class ReportListItem(BaseModel):
    report_id: str
    report_type: str
    run_id: str
    files: list[ReportFileInfo]
    created_at: datetime
