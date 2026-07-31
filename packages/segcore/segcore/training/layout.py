# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Names of the directories a run writes into.

They live here, with no imports, because the writer and the readers are in
different packages: segcore fills the predictions directory at the end of a
run, and every consumer of it is in the Trainer API. Renaming it on one side
only is silent -- new runs write one name, the API looks for the other, and the
Results tab shows no predictions with nothing logged anywhere.
"""
from __future__ import annotations

#: Where a run keeps its per-image prediction artifacts.
#:
#: Shortened from "predictions" in v0.9.8.post2. Every artifact path is
#: projects/<project id>/runs/<run id>/<this>/<image>.<suffix>, and Windows
#: refuses the whole thing past 260 characters.
PRED_DIRNAME = "pred"

#: What PRED_DIRNAME was called before v0.9.8.post2. The layout migration
#: renames it; nothing else should still be reading it.
LEGACY_PRED_DIRNAME = "predictions"
