# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The two defect-level recalls in this product must mean the same thing.

The evaluation report counts a defect found when the prediction covers at least
half of it (``_compute_instance_recall``). The threshold sweep now reports the
same quantity at every candidate threshold, so the Confidence presets can be
chosen on defects rather than on pixels -- and it does so with its own constant
in segcore, which cannot import the report module.

Two numbers both presented to a customer as "defects found", computed with
different cut-offs, is exactly the kind of drift nobody notices until the two
disagree in a report. This pins them together mechanically instead of by
comment. If one is deliberately changed, change both and update this test.
"""
from __future__ import annotations

import inspect

from app.core.report_builders import _compute_instance_recall
from segcore.training.metrics_pro import INSTANCE_DETECT_COVERAGE


def test_report_and_sweep_use_the_same_found_cut_off():
    report_default = inspect.signature(_compute_instance_recall).parameters["threshold"].default
    assert report_default == INSTANCE_DETECT_COVERAGE, (
        "report_builders._compute_instance_recall and segcore's "
        "INSTANCE_DETECT_COVERAGE define the same 'defect found' rule and have "
        f"drifted: report={report_default!r} sweep={INSTANCE_DETECT_COVERAGE!r}"
    )


def test_both_measure_coverage_of_the_instance_not_of_the_union():
    """The shared rule is intersection / instance area, in both places.

    region_overlaps divides the covered pixel count by ``areas`` (the region's
    own size) and _compute_instance_recall divides by ``inst_mask.sum()``. The
    report's denominator was once the union, which made an instance's score
    fall as the model found MORE defects; the docstring there records the fix.
    A same-shaped case through both must land on the same fraction.
    """
    import numpy as np

    from segcore.training.metrics_pro import region_overlaps

    labels = np.zeros((1, 10), dtype=np.int32)
    labels[0, 0:10] = 1
    areas = np.array([10], dtype=np.int64)

    predicted = np.zeros((1, 10), dtype=bool)
    predicted[0, 0:6] = True
    covered, n, found = region_overlaps(labels, areas, predicted)

    inst_mask = labels == 1
    intersection = int((inst_mask & predicted).sum())
    assert covered == intersection / int(inst_mask.sum())
    assert (n, found) == (1, 1)
