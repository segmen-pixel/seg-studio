# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A patch too small for the objects must be said out loud, before the GPU spends an hour.

An object longer than the tile overlap has no unclipped view anywhere. The
fragments are stitched back together rather than dropped, so the count
survives -- from a small fraction of the annotated objects to about all of
them, on the same weights -- but a stitched reading depends on the
fragments overlapping in the band, and it joins objects that genuinely touch.
Seeing the object whole in one tile is the stronger answer, so the run says
which patch would give it.

The area band cannot catch this. It is a length problem and a long thin bar
has the area of a much smaller square, which is why compose records the span
separately.
"""
from __future__ import annotations

from app.core.instance_training import (
    _warn_if_patch_too_small,
    suggested_patch_for_span,
)


def _warn(span, patch):
    lines = []
    fired = _warn_if_patch_too_small({"object_span_px": span}, patch, lines.append)
    return fired, "".join(lines)


class TestPatchVsObject:
    def test_an_object_wider_than_the_patch_warns(self):
        fired, msg = _warn(767, 768)
        assert fired
        assert "767" in msg and "192" in msg
        # The message has to name a way out, not just a complaint.
        assert "3840" in msg

    def test_the_message_does_not_claim_the_count_is_lost(self):
        """It was true before the fragments were stitched, and is not now."""
        _fired, msg = _warn(767, 768)
        assert "discarded" not in msg
        assert "stitch" in msg

    def test_an_object_the_overlap_holds_is_silent(self):
        # A 110px object at patch 768: overlap 192, comfortably clear. This is
        # the case DEFAULT_PATCH_SIZE is for, and it must stay quiet.
        assert _warn(110, 768)[0] is False

    def test_the_same_objects_at_the_suggested_patch_are_silent(self):
        # The suggestion has to actually satisfy the test that produced it,
        # or the warning sends the user in a circle.
        assert _warn(767, suggested_patch_for_span(767))[0] is False

    def test_whole_plate_mode_is_not_warned_about(self):
        # No tiling, no seams, nothing to clip.
        assert _warn_if_patch_too_small({"object_span_px": 767}, None, print) is False

    def test_a_dataset_composed_before_the_span_existed_is_not_guessed_at(self):
        # Absent is absent: warning on a missing key would fire on every old
        # run in the library.
        assert _warn_if_patch_too_small({}, 768, print) is False
        assert _warn_if_patch_too_small({"object_span_px": 0}, 768, print) is False

    def test_the_suggestion_clears_the_object_with_the_documented_margin(self):
        from segcore.instseg.tiled import default_stride

        for span in (40, 110, 300, 767, 1200):
            patch = suggested_patch_for_span(span)
            overlap = patch - default_stride(patch)
            assert overlap >= span * 1.25, (
                f"span {span}: suggested {patch} leaves {overlap}px of overlap"
            )
            assert patch % 256 == 0
