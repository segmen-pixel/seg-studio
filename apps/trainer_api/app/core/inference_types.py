# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Streaming / v2 API types
# ---------------------------------------------------------------------------
@dataclass
class Region:
    """One connected defect region in the segmentation output."""
    class_name: str
    class_id: int
    area_px: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    centroid: tuple[int, int] = (0, 0)  # (cx, cy) in original image coordinates


@dataclass
class StreamInferenceResult:
    """Full result from predict_one() / WebSocket streaming inference."""
    frame_id: str
    judgement: str  # "OK" or "NG"
    defect_found: bool
    regions: list[Region] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    result_id: str = ""
    mask_png_b64: str = ""  # base64-encoded RGBA PNG of defect overlay

    def to_dict(self) -> dict:
        d = {
            "type": "result",
            "frame_id": self.frame_id,
            "judgement": self.judgement,
            "defect_found": self.defect_found,
            "regions": [
                {
                    "class": r.class_name,
                    "class_id": r.class_id,
                    "area_px": r.area_px,
                    "bbox": list(r.bbox),
                    "centroid": list(r.centroid),
                    "confidence": round(r.confidence, 4),
                }
                for r in self.regions
            ],
            "summary": self.summary,
            "latency_ms": self.latency_ms,
            "result_id": self.result_id,
        }
        if self.mask_png_b64:
            d["mask_png_b64"] = self.mask_png_b64
        return d
