# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Model servers, as the product defines them.

Moved to ``apps/trainer_api/app/core/vlm_agent/backends.py`` when the labelling
loop became part of the product; this re-export keeps the old import working.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.trainer_api.app.core.vlm_agent.backends import (  # noqa: E402,F401
    ALIASES,
    DEFAULT_NUM_CTX,
    Backend,
    OllamaBackend,
    OpenAIBackend,
    Reply,
    describe,
    make_backend,
)
