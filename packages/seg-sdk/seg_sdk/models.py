# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Data models for Seg-Studio Inference SDK."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Region:
    """One connected defect region.

    Coordinates (``bbox`` and ``centroid``) are in the original input
    image's pixel space — no scaling is needed on the client side.
    """
    class_name: str
    class_id: int
    area_px: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    centroid: tuple[int, int] = (0, 0)  # (cx, cy)


@dataclass
class InferenceResult:
    """Result from a single-frame inference."""
    frame_id: str
    judgement: str  # "OK" or "NG"
    defect_found: bool
    regions: list[Region] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    result_id: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> InferenceResult:
        def _parse_centroid(raw) -> tuple[int, int]:
            if not raw:
                return (0, 0)
            try:
                cx, cy = raw
                return (int(cx), int(cy))
            except Exception:
                return (0, 0)

        regions = [
            Region(
                class_name=r.get("class", ""),
                class_id=r.get("class_id", 0),
                area_px=r.get("area_px", 0),
                bbox=tuple(r.get("bbox", [0, 0, 0, 0])),
                confidence=r.get("confidence", 0.0),
                centroid=_parse_centroid(r.get("centroid")),
            )
            for r in d.get("regions", [])
        ]
        return cls(
            frame_id=d.get("frame_id", ""),
            judgement=d.get("judgement", ""),
            defect_found=d.get("defect_found", False),
            regions=regions,
            summary=d.get("summary", {}),
            latency_ms=d.get("latency_ms", {}),
            result_id=d.get("result_id", ""),
        )
