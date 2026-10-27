# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""ONNX export must honour the checkpoint's use_se flag.

A checkpoint trained with --no-se has no se1/se2/se3 keys; before the fix
the export built an SE-enabled model and strict=False silently left the SE
blocks randomly initialised, so the shipped ONNX diverged from the trained
model. The export now infers use_se from the checkpoint and a guarded load
rejects any remaining key-set mismatch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core.exceptions import CheckpointIncompatibleError
from app.core.ort_infra import _export_onnx_model
from segcore.training.model import build_model


def test_onnx_export_matches_no_se_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model(
        "simpleunet", num_classes=2, output_stride=4, base_channels=8, use_se=False
    )
    model.eval()
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    onnx_path = tmp_path / "model.onnx"
    _export_onnx_model(
        tmp_path,
        ckpt,
        onnx_path,
        num_classes=2,
        run_output_stride=4,
        run_base_channels=8,
        run_arch="simpleunet",
        infer_w=32,
        infer_h=32,
    )

    import onnxruntime as ort

    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        want = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)


def test_export_rejects_partial_se_checkpoint(tmp_path):
    # One SE sub-key stripped: use_se inference still says True (se2/se3
    # remain), so the guarded load must refuse instead of silently
    # shipping a randomly initialised se1.
    model = build_model(
        "simpleunet", num_classes=2, output_stride=4, base_channels=8, use_se=True
    )
    sd = model.state_dict()
    sd.pop("se1.fc1.weight")
    ckpt = tmp_path / "model.pt"
    torch.save(sd, ckpt)

    with pytest.raises(CheckpointIncompatibleError):
        _export_onnx_model(
            tmp_path,
            ckpt,
            tmp_path / "model.onnx",
            num_classes=2,
            run_output_stride=4,
            run_base_channels=8,
            run_arch="simpleunet",
            infer_w=32,
            infer_h=32,
        )
