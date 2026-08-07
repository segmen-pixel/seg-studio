# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Structured error code registry for Seg-Studio.

Every user-facing error MUST have a code from this module.
Internal details (paths, stack traces) are NEVER included in responses.

Categories:
    NSS-1xxx  Validation        400  Request parameter issues
    NSS-2xxx  Not Found         404  Resource lookup failures
    NSS-3xxx  Training          4/5  Training config / runtime
    NSS-4xxx  Inference         4/5  Prediction / export
    NSS-5xxx  AI Assist         4/5  SAM, MLP, superpixel
    NSS-6xxx  Dataset           4xx  Import / export
    NSS-7xxx  System            5xx  Internal / hardware
    NSS-8xxx  Security          4xx  Path traversal, access
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorMeta:
    """Immutable metadata for a single error code."""

    http_status: int
    message_en: str
    message_ja: str
    hint_en: str | None = None
    hint_ja: str | None = None
    log_level: str = "WARNING"


# ---------------------------------------------------------------------------
# Registry — code string → ErrorMeta
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, ErrorMeta] = {}


def _reg(code: str, meta: ErrorMeta) -> str:
    """Register a code and return the code string for assignment."""
    _REGISTRY[code] = meta
    return code


def get_meta(code: str) -> ErrorMeta:
    """Look up metadata for an error code. Falls back to NSS-7004."""
    return _REGISTRY.get(code, _REGISTRY["NSS-7004"])


# ── NSS-1xxx: Validation ─────────────────────────────────────────────────
VALIDATION_CLASS_ID_RANGE = _reg("NSS-1001", ErrorMeta(
    400, "Class ID must be in 0..254.",
    "クラスIDは0〜254の範囲で指定してください。"))
VALIDATION_IGNORE_INDEX = _reg("NSS-1002", ErrorMeta(
    400, "ignore_index must be 255.",
    "ignore_indexは255でなければなりません。"))
VALIDATION_DUPLICATE_IDS = _reg("NSS-1003", ErrorMeta(
    400, "Duplicate class IDs are not allowed.",
    "クラスIDが重複しています。"))
VALIDATION_BG_DELETE = _reg("NSS-1005", ErrorMeta(
    400, "Cannot delete the background class.",
    "背景クラスは削除できません。"))
VALIDATION_REQUIRED_PARAM = _reg("NSS-1006", ErrorMeta(
    400, "A required parameter is missing.",
    "必須パラメータが不足しています。"))
VALIDATION_JSON_PARSE = _reg("NSS-1007", ErrorMeta(
    400, "Invalid JSON in request body.",
    "リクエストのJSONが不正です。"))
VALIDATION_FILE_FORMAT = _reg("NSS-1008", ErrorMeta(
    400, "Unsupported file format.",
    "対応していないファイル形式です。"))
VALIDATION_EMPTY_FILE = _reg("NSS-1009", ErrorMeta(
    400, "Uploaded file is empty.",
    "アップロードされたファイルが空です。"))
VALIDATION_INVALID_SET = _reg("NSS-1010", ErrorMeta(
    400, "Set must be 'train' or 'val'.",
    "セットは'train'または'val'を指定してください。"))

# ── NSS-2xxx: Not Found ──────────────────────────────────────────────────
NOT_FOUND_PROJECT = _reg("NSS-2001", ErrorMeta(
    404, "Project not found.",
    "プロジェクトが見つかりません。"))
NOT_FOUND_IMAGE = _reg("NSS-2002", ErrorMeta(
    404, "Image not found.",
    "画像が見つかりません。"))
NOT_FOUND_MASK = _reg("NSS-2003", ErrorMeta(
    404, "Mask not found.",
    "マスクが見つかりません。"))
NOT_FOUND_CHECKPOINT = _reg("NSS-2004", ErrorMeta(
    404, "Model checkpoint not found.",
    "モデルチェックポイントが見つかりません。"))
NOT_FOUND_RUN = _reg("NSS-2005", ErrorMeta(
    404, "Training run not found.",
    "学習ランが見つかりません。"))
NOT_FOUND_PREDICTION = _reg("NSS-2006", ErrorMeta(
    404, "Prediction not found. Run inference first.",
    "推論結果がありません。先に推論を実行してください。"))
NOT_FOUND_ANNOTATIONS = _reg("NSS-2007", ErrorMeta(
    404, "annotations.json not found.",
    "annotations.jsonが見つかりません。"))
NOT_FOUND_CLASSES = _reg("NSS-2008", ErrorMeta(
    404, "classes.json not found or has no classes defined.",
    "classes.jsonが見つからないか、クラスが未定義です。"))
NOT_FOUND_ITEM = _reg("NSS-2009", ErrorMeta(
    404, "Dataset item not found.",
    "データセットアイテムが見つかりません。"))

# ── NSS-3xxx: Training ───────────────────────────────────────────────────
TRAIN_INVALID_STRIDE = _reg("NSS-3001", ErrorMeta(
    400, "output_stride must be one of {1, 2, 4}.",
    "output_strideは{1, 2, 4}のいずれかを指定してください。"))
TRAIN_INVALID_INPUT_SIZE = _reg("NSS-3002", ErrorMeta(
    400, "input_size must be [width, height] with positive values.",
    "input_sizeは正の値の[幅, 高さ]で指定してください。"))
TRAIN_SIZE_STRIDE_MISMATCH = _reg("NSS-3003", ErrorMeta(
    400, "input_size must be divisible by output_stride.",
    "input_sizeはoutput_strideで割り切れる必要があります。"))
TRAIN_OOM = _reg("NSS-3004", ErrorMeta(
    500, "GPU memory insufficient for current configuration.",
    "現在の設定ではGPUメモリが不足しています。",
    hint_en="Reduce batch_size or input_size and retry.",
    hint_ja="バッチサイズまたは入力サイズを下げて再試行してください。",
    log_level="ERROR"))
TRAIN_SUBPROCESS_CRASH = _reg("NSS-3005", ErrorMeta(
    500, "Training process terminated unexpectedly.",
    "学習プロセスが異常終了しました。",
    log_level="ERROR"))
TRAIN_RUN_NOT_FOUND = _reg("NSS-3006", ErrorMeta(
    404, "Training run not found or already finished.",
    "学習ランが見つからないか、既に完了しています。"))
TRAIN_NO_MASKS = _reg("NSS-3007", ErrorMeta(
    400, "No annotated masks found for training.",
    "学習用のアノテーション済みマスクがありません。",
    hint_en="Annotate at least one image before starting training.",
    hint_ja="学習を開始する前に、少なくとも1枚の画像にアノテーションしてください。"))
TRAIN_BUSY = _reg("NSS-3008", ErrorMeta(
    409, "Training is already in progress.",
    "学習が既に実行中です。"))
TRAIN_CONFIG_INVALID = _reg("NSS-3009", ErrorMeta(
    400, "Invalid training configuration.",
    "学習設定が不正です。"))

# ── NSS-4xxx: Inference ──────────────────────────────────────────────────
INFER_CKPT_INCOMPATIBLE = _reg("NSS-4001", ErrorMeta(
    400, "Model checkpoint is incompatible. Please retrain.",
    "モデルチェックポイントに互換性がありません。再学習してください。"))
INFER_FAILED = _reg("NSS-4002", ErrorMeta(
    500, "Prediction failed.",
    "推論に失敗しました。",
    log_level="ERROR"))
INFER_COREML_FAILED = _reg("NSS-4003", ErrorMeta(
    500, "CoreML prediction failed.",
    "CoreML推論に失敗しました。",
    log_level="ERROR"))
INFER_TTA_COREML = _reg("NSS-4004", ErrorMeta(
    400, "TTA is not supported with the CoreML backend.",
    "CoreMLバックエンドではTTAは使用できません。"))
INFER_UNKNOWN_HEATMAP = _reg("NSS-4005", ErrorMeta(
    400, "Unknown heatmap type.",
    "不明なヒートマップタイプです。"))
INFER_CLASS_RANGE = _reg("NSS-4006", ErrorMeta(
    400, "class_id is out of range for this model.",
    "class_idがモデルの範囲外です。"))
INFER_INVALID_BACKEND = _reg("NSS-4007", ErrorMeta(
    400, "Backend must be 'onnx' or 'coreml'.",
    "バックエンドは'onnx'または'coreml'を指定してください。"))
INFER_MODEL_MISSING = _reg("NSS-4008", ErrorMeta(
    404, "Model not found — it may have been deleted. Please retrain.",
    "モデルが見つかりません — 削除された可能性があります。再学習してください。"))

# ── NSS-5xxx: AI Assist ──────────────────────────────────────────────────
AI_SAM_UNKNOWN_MODEL = _reg("NSS-5001", ErrorMeta(
    400, "Unknown SAM model name.",
    "不明なSAMモデル名です。"))
AI_SAM_POINTS_MISMATCH = _reg("NSS-5002", ErrorMeta(
    400, "Points and labels must have the same length.",
    "ポイントとラベルの長さが一致しません。"))
AI_SAM_CKPT_MISSING = _reg("NSS-5003", ErrorMeta(
    404, "SAM model checkpoint not found.",
    "SAMモデルのチェックポイントが見つかりません。",
    hint_en="Download the checkpoint first via the SAM model selector.",
    hint_ja="SAMモデルセレクターからチェックポイントをダウンロードしてください。"))
AI_SAM_INFERENCE_FAILED = _reg("NSS-5004", ErrorMeta(
    500, "SAM inference failed.",
    "SAM推論に失敗しました。",
    log_level="ERROR"))
AI_RF_NO_ANNOTATIONS = _reg("NSS-5005", ErrorMeta(
    400, "No annotated masks found for AI assist.",
    "AIアシスト用のアノテーション済みマスクがありません。"))
AI_RF_FAILED = _reg("NSS-5006", ErrorMeta(
    500, "AI assist (RF/MLP) failed.",
    "AIアシスト（RF/MLP）に失敗しました。",
    log_level="ERROR"))
AI_SUPERPIXEL_FAILED = _reg("NSS-5007", ErrorMeta(
    500, "Superpixel computation failed.",
    "スーパーピクセル計算に失敗しました。",
    log_level="ERROR"))
AI_SAM_LABEL_FAILED = _reg("NSS-5008", ErrorMeta(
    500, "SAM Label Assist failed.",
    "SAMラベルアシストに失敗しました。",
    log_level="ERROR"))
AI_AUTO_LABEL_INVALID = _reg("NSS-5009", ErrorMeta(
    400, "Auto-label parameters are invalid.",
    "自動ラベルパラメータが不正です。"))

# ── NSS-6xxx: Dataset ────────────────────────────────────────────────────
DATASET_ZIP_REQUIRED = _reg("NSS-6001", ErrorMeta(
    400, "A ZIP file is required.",
    "ZIPファイルが必要です。"))
DATASET_INVALID_ZIP = _reg("NSS-6002", ErrorMeta(
    400, "The uploaded ZIP file is invalid or corrupted.",
    "アップロードされたZIPファイルが不正または破損しています。"))
DATASET_NO_IMAGES = _reg("NSS-6003", ErrorMeta(
    400, "No images found in the uploaded ZIP.",
    "ZIPファイル内に画像がありません。"))
DATASET_UNSAFE_ENTRY = _reg("NSS-6004", ErrorMeta(
    400, "ZIP contains an unsafe entry (path traversal detected).",
    "ZIPに安全でないエントリが含まれています（パストラバーサル検出）。"))
DATASET_NO_EXPORT_MASKS = _reg("NSS-6005", ErrorMeta(
    400, "No images with masks found to export.",
    "エクスポート対象のマスク付き画像がありません。"))
DATASET_IMAGE_READ = _reg("NSS-6006", ErrorMeta(
    500, "Failed to read image file.",
    "画像ファイルの読み取りに失敗しました。",
    log_level="ERROR"))

# ── NSS-7xxx: System ─────────────────────────────────────────────────────
SYSTEM_GPU_DEVICE = _reg("NSS-7001", ErrorMeta(
    500, "GPU device configuration failed.",
    "GPUデバイスの設定に失敗しました。",
    log_level="ERROR"))
SYSTEM_DISTILL_FAILED = _reg("NSS-7002", ErrorMeta(
    500, "Distillation precompute failed.",
    "蒸留の前処理に失敗しました。",
    log_level="ERROR"))
SYSTEM_EXPORT_FAILED = _reg("NSS-7003", ErrorMeta(
    500, "Model export failed.",
    "モデルのエクスポートに失敗しました。",
    log_level="ERROR"))
SYSTEM_INTERNAL = _reg("NSS-7004", ErrorMeta(
    500, "An internal error occurred.",
    "内部エラーが発生しました。",
    hint_en="Check server logs or contact support with the error code.",
    hint_ja="サーバーログを確認するか、エラーコードを添えてサポートに連絡してください。",
    log_level="ERROR"))
SYSTEM_FILE_IO = _reg("NSS-7005", ErrorMeta(
    500, "File I/O error.",
    "ファイルI/Oエラーが発生しました。",
    log_level="ERROR"))

# ── NSS-8xxx: Security ───────────────────────────────────────────────────
SECURITY_PATH_TRAVERSAL = _reg("NSS-8001", ErrorMeta(
    400, "Invalid path detected.",
    "不正なパスが検出されました。"))
SECURITY_INVALID_PROJECT_ID = _reg("NSS-8002", ErrorMeta(
    400, "Invalid project ID.",
    "不正なプロジェクトIDです。"))
SECURITY_INVALID_RUN_ID = _reg("NSS-8003", ErrorMeta(
    400, "Invalid run ID.",
    "不正なランIDです。"))
