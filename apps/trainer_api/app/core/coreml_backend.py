# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""CoreML model export, updatable model generation, and cached loading.

Separated from ``prediction_engine`` to keep platform-specific (macOS/iOS)
logic isolated.  All three public functions are re-exported from
``prediction_engine`` for backward compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

from . import state as _state
from .classes import find_coreml_model_path
from .config import TRAINER_BUILD_ID
from .run_config import (
    _load_run_arch,
    _load_run_base_channels,
    _load_run_input_size,
    _load_run_num_classes,
    _load_run_output_stride,
)


def export_coreml_model(run_path: Path, model_path: Path) -> Path:
    try:
        import coremltools as ct
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="coremltools is required for Core ML export but is not installed",
        )

    import torch

    from segcore.training.model import build_model

    infer_w, infer_h = _load_run_input_size(run_path)
    run_output_stride = _load_run_output_stride(run_path)
    run_base_channels = _load_run_base_channels(run_path)
    run_arch = _load_run_arch(run_path)
    num_classes = _load_run_num_classes(run_path)
    model = build_model(run_arch, num_classes=num_classes, output_stride=run_output_stride, base_channels=run_base_channels)
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=False)
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Model checkpoint incompatible with current architecture. Please retrain.")
    model.eval()

    example = torch.zeros(1, 3, infer_h, infer_w)
    traced = torch.jit.trace(model, example)
    output_path = run_path / "model.mlmodel"
    convert_kwargs = {
        "inputs": [ct.TensorType(name="input", shape=example.shape)],
    }
    try:
        mlmodel = ct.convert(traced, convert_to="neuralnetwork", **convert_kwargs)
    except (TypeError, RuntimeError):
        mlmodel = ct.convert(traced, **convert_kwargs)
    mlmodel.short_description = "Seg-Studio segmentation model"
    mlmodel.author = "Seg-Studio"
    mlmodel.version = TRAINER_BUILD_ID

    # --- Updatable version (MUST be generated BEFORE FP16 quantization) ---
    # CoreML updatable layers require FP32 weights; FP16 models cannot be
    # marked updatable.
    try:
        updatable_path = _make_updatable(mlmodel, num_classes, run_path)
        if updatable_path:
            import logging
            logging.getLogger(__name__).info("Updatable model saved: %s", updatable_path.name)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Updatable model generation failed: %s", exc)

    # Quantize weights to FP16 (halves model size, negligible accuracy loss)
    try:
        from coremltools.models.neural_network import quantization_utils
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)
    except Exception:
        pass
    try:
        mlmodel.user_defined_metadata["num_classes"] = str(num_classes)
        mlmodel.user_defined_metadata["input_size"] = f"{infer_w}x{infer_h}"
        mlmodel.user_defined_metadata["output_stride"] = str(run_output_stride)
        # image_size: actual source image dimensions (always present)
        # train_size: model input dimensions (= image_size if no resize-clone)
        _cfg = json.loads((run_path / "train_config.json").read_text(encoding="utf-8"))
        _img_sz = _cfg.get("image_size")
        _trn_sz = _cfg.get("train_size")
        if _img_sz:
            mlmodel.user_defined_metadata["image_size"] = f"{_img_sz[0]}x{_img_sz[1]}"
        if _trn_sz:
            mlmodel.user_defined_metadata["train_size"] = f"{_trn_sz[0]}x{_trn_sz[1]}"
        elif _img_sz:
            # No resize-clone: train_size = image_size
            mlmodel.user_defined_metadata["train_size"] = f"{_img_sz[0]}x{_img_sz[1]}"
    except (AttributeError, KeyError, json.JSONDecodeError, OSError):
        pass
    mlmodel.save(str(output_path))

    return output_path


def _make_updatable(mlmodel: object, num_classes: int, run_path: Path) -> Path | None:
    """Convert a neuralnetwork CoreML model to an updatable version.

    The updatable model allows on-device fine-tuning of the final
    classification Conv layer on iOS via MLUpdateTask.

    Steps:
      1. Find the last Conv2D layer (classification head).
      2. Set model output to conv's direct output (no softmax -- CoreML
         cannot backprop through softmax for updatable layers).
      3. Mark the final Conv layer as updatable.
      4. Set MSE loss + SGD optimizer (auto adds trainingInput).
      5. Save as model_updatable.mlmodel.

    Returns the output path, or None if the model is not a neuralnetwork.
    """
    import coremltools as ct
    import coremltools.models.datatypes as datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

    spec = mlmodel.get_spec()

    # Only neuralnetwork supports updatable (mlprogram does not)
    nn = getattr(spec, "neuralNetwork", None)
    if nn is None:
        return None

    # --- Step 1: Find the last Conv layer (classification head) ---
    last_conv_name = None
    last_conv_output = None
    for layer in nn.layers:
        if layer.WhichOneof("layer") == "convolution":
            last_conv_name = layer.name
            last_conv_output = layer.output[0] if layer.output else None

    if last_conv_name is None or last_conv_output is None:
        return None

    # Derive output shape from model input shape: (num_classes, H, W)
    input_desc = spec.description.input[0]
    if input_desc.type.HasField("multiArrayType"):
        in_shape = list(input_desc.type.multiArrayType.shape)  # [1, C, H, W]
        out_h, out_w = in_shape[2], in_shape[3]
    else:
        out_h, out_w = 128, 128  # fallback
    output_shape = (num_classes, out_h, out_w)

    # --- Step 2: Set model output to conv's direct output ---
    # No softmax -- CoreML cannot backprop through softmax for updatable layers.
    spec.description.output[0].name = last_conv_output

    builder = NeuralNetworkBuilder(spec=spec)

    # --- Step 3: Mark final Conv as updatable ---
    builder.make_updatable([last_conv_name])

    # --- Step 4: Loss function + optimizer ---
    # MSE loss for segmentation: target is a soft mask (same shape as conv output).
    # set_mean_squared_error_loss automatically adds trainingInput entries
    # (copies model inputs + adds "{input}_true" as target).
    builder.set_mean_squared_error_loss(
        name="loss",
        input_feature=(last_conv_output, datatypes.Array(*output_shape)),
    )

    builder.set_sgd_optimizer(
        SgdParams(lr=0.001, batch=1, momentum=0.9)
    )
    builder.set_epochs(10, allowed_set=range(1, 201))

    target_name = last_conv_output + "_true"

    # --- Step 6: Metadata ---
    updatable_model = ct.models.MLModel(spec)
    updatable_model.short_description = "Seg-Studio updatable segmentation model"
    updatable_model.author = "Seg-Studio"
    try:
        updatable_model.user_defined_metadata["updatable"] = "true"
        updatable_model.user_defined_metadata["num_classes"] = str(num_classes)
        updatable_model.user_defined_metadata["target_name"] = target_name
        # Copy metadata from original
        for key in ["input_size", "output_stride", "image_size", "train_size"]:
            val = mlmodel.user_defined_metadata.get(key)
            if val:
                updatable_model.user_defined_metadata[key] = val
    except (AttributeError, KeyError):
        pass

    output_path = run_path / "model_updatable.mlmodel"
    updatable_model.save(str(output_path))
    return output_path


def load_coreml_model(run_path: Path) -> object:
    model_path = find_coreml_model_path(run_path)
    if model_path is None:
        model_path = export_coreml_model(run_path, run_path / "model.pt")
    cache_key = str(model_path)
    cached = _state.COREML_CACHE.get(cache_key)
    if cached is not None:
        cached_path, cached_model, cached_mtime = cached
        if cached_path.exists():
            current_mtime = cached_path.stat().st_mtime
            if current_mtime == cached_mtime:
                return cached_model
    try:
        import coremltools as ct
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="coremltools is required for Core ML inference but is not installed",
        )
    compute_units = None
    try:
        compute_units = ct.ComputeUnit.ALL
    except (AttributeError, ValueError):
        compute_units = None
    if compute_units is not None:
        mlmodel = ct.models.MLModel(str(model_path), compute_units=compute_units)
    else:
        mlmodel = ct.models.MLModel(str(model_path))
    mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    _state.COREML_CACHE.put(cache_key, (model_path, mlmodel, mtime))
    return mlmodel
