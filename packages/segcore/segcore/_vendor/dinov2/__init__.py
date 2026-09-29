# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The DINOv2 vision transformer, vendored from facebookresearch/dinov2.

Upstream: https://github.com/facebookresearch/dinov2
Revision: 7764ea0f912e53c92e82eb78a2a1631e92725fc8
Licence:  Apache-2.0 (``LICENSE`` here is the upstream root LICENSE)

Only the model definition is here: ``vision_transformer.py`` and the
``layers`` package it is built from. Every one of those files carries the
upstream Apache-2.0 header and is the upstream file unchanged, except for one
line of ``vision_transformer.py``: ``from dinov2.layers import`` became a
relative import, and the file says so under its header.

Why a copy rather than ``torch.hub.load``: see :mod:`segcore.dinov2_loader`,
which builds the backbones from these files. ``tests/test_dinov2_loader.py``
checks each file against its upstream git blob hash.
"""
