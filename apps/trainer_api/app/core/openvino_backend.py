# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""OpenVINO IR (.xml/.bin) model export for Intel edge deployment.

Mirrors ``coreml_backend`` but targets OpenVINO Runtime (Intel CPU/iGPU/NPU).
Supports three precisions:

* ``fp32`` — baseline; largest file, full accuracy.
* ``fp16`` — half-size, near-zero accuracy loss, faster on iGPU/NPU.
* ``int8`` — NNCF post-training quantization using the project's val
  patches; ~4x smaller / faster on CPU but requires calibration data.

The ONNX file is the conversion source. If ``model.onnx`` is missing we
generate it on the fly via the existing ``export_onnx_model`` helper, so
this endpoint works even before the user clicks "Export ONNX".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import HTTPException

from .run_config import (
    _load_run_arch,
    _load_run_base_channels,
    _load_run_input_size,
    _load_run_num_classes,
    _load_run_output_stride,
    _load_run_patch_size,
)

logger = logging.getLogger(__name__)

Precision = Literal["fp32", "fp16", "int8"]
_VALID_PRECISIONS: tuple[Precision, ...] = ("fp32", "fp16", "int8")

# Cap calibration to keep INT8 quantization quick (~30s) and memory bounded.
_INT8_CALIBRATION_SAMPLES = 100


def _ensure_onnx(run_path: Path, model_path: Path) -> Path:
    """Return ``run_path/model.onnx``, exporting it from ``model.pt`` if absent."""
    onnx_path = run_path / "model.onnx"
    if onnx_path.exists():
        return onnx_path
    # Lazy import: prediction_engine pulls torch and would slow startup; this
    # function-local import also breaks any future circular dependency since
    # prediction_engine may grow to reference openvino_backend.
    from .prediction_engine import export_onnx_model

    infer_w, infer_h = _load_run_input_size(run_path)
    return export_onnx_model(
        run_path,
        model_path,
        onnx_path,
        num_classes=_load_run_num_classes(run_path),
        run_output_stride=_load_run_output_stride(run_path),
        run_base_channels=_load_run_base_channels(run_path),
        run_arch=_load_run_arch(run_path),
        infer_w=infer_w,
        infer_h=infer_h,
    )


def _iter_calibration_patches(run_path: Path, max_samples: int):
    """Yield normalized NCHW float32 calibration tensors from the val split.

    Matches the runtime input contract: mean/std normalization on top of
    the 0-1 scale, and — for patch-trained runs — native-resolution
    ``patch_size`` tiles on the sliding-window grid instead of resized
    full images (resize would calibrate activation ranges on a distribution
    the deployed model never sees). Full-image runs keep the resize, which
    IS their runtime input. Yields up to ``max_samples`` items and raises
    ``HTTPException`` when no calibration data is available — silently
    degrading to FP16 would surprise the user.
    """
    import numpy as np
    from PIL import Image

    from .paths import project_dir_of

    # prepared/ lives in the project directory. Resolved by walking up to
    # PROJECTS_DIR: a fixed hop count cannot be right for both live runs and
    # archived ones, which sit at different depths.
    project_root = project_dir_of(run_path)
    prepared = project_root / "prepared"
    val_split = prepared / "splits" / "val.txt"
    images_dir = prepared / "images"
    if not val_split.exists() or not images_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "INT8 quantization requires a prepared val split, but "
                f"{val_split} or {images_dir} is missing. Re-run "
                "dataset preparation, or pick FP32/FP16 export instead."
            ),
        )
    ids = [line.strip() for line in val_split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="INT8 quantization needs a non-empty val split.",
        )

    from .config import NORMALIZE

    mean = np.array(NORMALIZE["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(NORMALIZE["std"], dtype=np.float32).reshape(3, 1, 1)
    infer_w, infer_h = _load_run_input_size(run_path)
    patch_size = _load_run_patch_size(run_path)
    yielded = 0
    for stem in ids:
        if yielded >= max_samples:
            break
        # prepared/images stores extensions verbatim; try a few common suffixes.
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            path = images_dir / f"{stem}{suffix}"
            if path.exists():
                break
        else:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.debug("OpenVINO calibration: skip %s (%s)", path.name, exc)
            continue
        arr_hwc = np.asarray(img, dtype=np.float32)
        img_h, img_w = arr_hwc.shape[:2]
        if patch_size > 0 and img_h >= patch_size and img_w >= patch_size:
            # Patch-trained models run sliding-window inference at native
            # resolution, so calibration must see the same distribution:
            # native-res patch tiles, normalized like the runtime input.
            stride = max(1, patch_size * 3 // 4)
            ys = list(range(0, img_h - patch_size + 1, stride))
            if ys[-1] != img_h - patch_size:
                ys.append(img_h - patch_size)
            xs = list(range(0, img_w - patch_size + 1, stride))
            if xs[-1] != img_w - patch_size:
                xs.append(img_w - patch_size)
            for y in ys:
                if yielded >= max_samples:
                    break
                for x in xs:
                    if yielded >= max_samples:
                        break
                    tile = arr_hwc[y : y + patch_size, x : x + patch_size].transpose(2, 0, 1)
                    tile = (tile / 255.0 - mean) / std
                    yield tile[None, ...].astype(np.float32)
                    yielded += 1
        else:
            # Full-image-trained models (patch_size=0) resize at inference
            # too, so calibrating on the resized image matches the runtime.
            resized = img.resize((infer_w, infer_h), Image.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
            arr = (arr / 255.0 - mean) / std
            yield arr[None, ...].astype(np.float32)
            yielded += 1
    if yielded == 0:
        raise HTTPException(
            status_code=400,
            detail="INT8 quantization found no usable val images.",
        )


def export_openvino_model(
    run_path: Path, model_path: Path, precision: Precision = "fp32",
) -> Path:
    """Export ``model.pt`` (via ONNX) to OpenVINO IR.

    Returns the path to the ``.xml`` file; the matching ``.bin`` is written
    alongside it. Output layout::

        run_path/openvino/{fp32,fp16,int8}/model.xml
        run_path/openvino/{fp32,fp16,int8}/model.bin
    """
    if precision not in _VALID_PRECISIONS:
        raise HTTPException(
            status_code=400,
            detail=f"precision must be one of {_VALID_PRECISIONS}, got {precision!r}",
        )
    try:
        import openvino as ov  # type: ignore[import-not-found]
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail=(
                "openvino is required for OpenVINO export but is not installed. "
                "Install it via `pip install openvino` or rerun the installer "
                "with `install_windows.bat --with-openvino`."
            ),
        )

    onnx_path = _ensure_onnx(run_path, model_path)

    out_dir = run_path / "openvino" / precision
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "model.xml"

    ov_model = ov.convert_model(str(onnx_path))

    if precision == "int8":
        try:
            import nncf  # type: ignore[import-not-found]
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail=(
                    "nncf is required for INT8 quantization but is not installed. "
                    "Install it via `pip install nncf` or rerun the installer "
                    "with `install_windows.bat --with-openvino`."
                ),
            )
        patches = list(_iter_calibration_patches(run_path, _INT8_CALIBRATION_SAMPLES))
        logger.info("OpenVINO INT8 calibration: %d patches", len(patches))
        # NNCF expects an iterable producing single-input arrays; wrap with
        # nncf.Dataset to satisfy the API.
        calibration_dataset = nncf.Dataset(patches, lambda x: x)
        ov_model = nncf.quantize(ov_model, calibration_dataset)
        ov.save_model(ov_model, str(xml_path))
    else:
        # fp16 collapses weights to half precision in the .bin payload;
        # fp32 keeps them full.
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=(precision == "fp16"))

    logger.info("OpenVINO export complete: %s (%s)", xml_path, precision)
    return xml_path
