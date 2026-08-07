# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Custom exception classes for the training API.

All domain exceptions inherit from :class:`AppError`, which carries a
structured ``code`` (NSS-XXXX) used in API responses and logs.
Existing exception names are preserved for backward compatibility.

Security: ``detail`` is logged server-side only — never sent to the client.
"""
from __future__ import annotations

from typing import Any

from . import error_codes as E


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Base exception for all application errors.

    Attributes:
        code:  NSS error code string (e.g. ``"NSS-3004"``).
        user_message:  Safe message returned to the client.
        detail:  Internal-only context (logged, **never** in API response).
        context:  Structured metadata (project_id, run_id, …).
    """

    code: str = E.SYSTEM_INTERNAL

    def __init__(
        self,
        user_message: str = "",
        *,
        detail: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        if not user_message:
            user_message = E.get_meta(self.code).message_en
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail
        self.context = context or {}

    @property
    def http_status(self) -> int:
        return E.get_meta(self.code).http_status

    @property
    def log_level(self) -> str:
        return E.get_meta(self.code).log_level


# ---------------------------------------------------------------------------
# Project / Resource (NSS-2xxx)
# ---------------------------------------------------------------------------
class ProjectNotFoundError(AppError):
    """The requested project does not exist on disk."""
    code = E.NOT_FOUND_PROJECT


class ImageNotFoundError(AppError):
    """The requested image does not exist."""
    code = E.NOT_FOUND_IMAGE


class MaskNotFoundError(AppError):
    """The requested mask does not exist."""
    code = E.NOT_FOUND_MASK


class CheckpointNotFoundError(AppError):
    """Model checkpoint file not found."""
    code = E.NOT_FOUND_CHECKPOINT


class RunNotFoundError(AppError):
    """Training run not found."""
    code = E.NOT_FOUND_RUN


class PredictionNotFoundError(AppError):
    """Prediction artifacts not found."""
    code = E.NOT_FOUND_PREDICTION


class DatasetItemNotFoundError(AppError):
    """Dataset item not found."""
    code = E.NOT_FOUND_ITEM


# ---------------------------------------------------------------------------
# Validation (NSS-1xxx)
# ---------------------------------------------------------------------------
class ValidationError(AppError):
    """Generic validation error."""
    code = E.VALIDATION_REQUIRED_PARAM


class ConfigurationError(AppError):
    """Invalid training configuration.

    Covers validation failures such as invalid output_stride values,
    non-positive input_size dimensions, input_size not divisible by
    output_stride, or class IDs outside the 0..254 range.
    """
    code = E.TRAIN_CONFIG_INVALID


# ---------------------------------------------------------------------------
# Training (NSS-3xxx)
# ---------------------------------------------------------------------------
class TrainingError(AppError):
    """Base exception for all training-related errors."""
    code = E.TRAIN_SUBPROCESS_CRASH


class TrainingOOMError(TrainingError):
    """CUDA out-of-memory during training.

    Raised when the training subprocess exits with exit code 2 (OOM)
    or when a CUDA OOM RuntimeError is detected.
    """
    code = E.TRAIN_OOM


class TrainingSubprocessError(TrainingError):
    """Training subprocess crashed or exited with a non-zero code.

    Attributes:
        exit_code: The process exit code (e.g. 1 for generic error,
                   negative values for signals on Unix).
    """
    code = E.TRAIN_SUBPROCESS_CRASH

    def __init__(self, message: str, exit_code: int, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.exit_code = exit_code


class TrainingBusyError(TrainingError):
    """Training is already in progress on this device."""
    code = E.TRAIN_BUSY


class TrainingNoMasksError(TrainingError):
    """No annotated masks found for training."""
    code = E.TRAIN_NO_MASKS


# ---------------------------------------------------------------------------
# Inference / Prediction (NSS-4xxx)
# ---------------------------------------------------------------------------
class PredictError(AppError):
    """Generic prediction failure."""
    code = E.INFER_FAILED


class PredictModelMissingError(AppError):
    """Model file not found (may have been deleted)."""
    code = E.INFER_MODEL_MISSING


class CheckpointIncompatibleError(AppError):
    """Model checkpoint is incompatible with the current architecture."""
    code = E.INFER_CKPT_INCOMPATIBLE


class CoreMLPredictError(AppError):
    """CoreML prediction failed."""
    code = E.INFER_COREML_FAILED


# ---------------------------------------------------------------------------
# AI Assist (NSS-5xxx)
# ---------------------------------------------------------------------------
class SAMModelMissingError(AppError):
    """SAM checkpoint not found."""
    code = E.AI_SAM_CKPT_MISSING


class SAMInferenceError(AppError):
    """SAM inference failed."""
    code = E.AI_SAM_INFERENCE_FAILED


class RFAssistError(AppError):
    """RF/MLP assist failed."""
    code = E.AI_RF_FAILED


class SuperpixelError(AppError):
    """Superpixel computation failed."""
    code = E.AI_SUPERPIXEL_FAILED


class SAMLabelAssistError(AppError):
    """SAM Label Assist failed."""
    code = E.AI_SAM_LABEL_FAILED


# ---------------------------------------------------------------------------
# Dataset (NSS-6xxx)
# ---------------------------------------------------------------------------
class DatasetZipError(AppError):
    """ZIP import error."""
    code = E.DATASET_INVALID_ZIP


class DatasetImageReadError(AppError):
    """Failed to read an image file."""
    code = E.DATASET_IMAGE_READ


# ---------------------------------------------------------------------------
# System (NSS-7xxx)
# ---------------------------------------------------------------------------
class GPUDeviceError(AppError):
    """GPU device configuration failed."""
    code = E.SYSTEM_GPU_DEVICE


class DistillError(AppError):
    """Distillation precompute failed."""
    code = E.SYSTEM_DISTILL_FAILED


class ExportError(AppError):
    """Model export failed."""
    code = E.SYSTEM_EXPORT_FAILED


class FileIOError(AppError):
    """File I/O error."""
    code = E.SYSTEM_FILE_IO


# ---------------------------------------------------------------------------
# Security (NSS-8xxx)
# ---------------------------------------------------------------------------
class PathTraversalError(AppError):
    """Path traversal detected."""
    code = E.SECURITY_PATH_TRAVERSAL
