# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio Inference SDK — Python client for real-time segmentation inference."""
from .async_client import AsyncSegClient, AsyncSegStream
from .client import SegClient, SegStream
from .models import InferenceResult, Region

__all__ = [
    "SegClient",
    "SegStream",
    "AsyncSegClient",
    "AsyncSegStream",
    "InferenceResult",
    "Region",
]
# Keep in sync with packages/seg-sdk/pyproject.toml [project] version.
__version__ = "0.9.6"
