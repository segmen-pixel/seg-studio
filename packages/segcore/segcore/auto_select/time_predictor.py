# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Warmup-calibrated training-time predictor (v6).

Companion to ``combo_predictor.py``.  When the caller runs one
"anchor combo" on a new project as a warm-up, this module turns its
actual elapsed_sec into a per-project scale factor and uses it to
calibrate physical-model predictions for every other candidate combo.

The shipped bundle is a pure JSON file (``phys_time.json``) containing
the 10 fitted coefficients of a log-linear regression on
``(num_train, num_total, img_pixels, base_channels, patch_size,
arch_one_hot, distill_on, fg_patch_prob)``.  Inference uses only
``numpy.dot`` so the runtime needs no sklearn at scoring time.

LOPO benchmark (37 projects, research_artifacts/combo_predictor_v4):

  * physical-only baseline:   R²(log) = -0.005, MAPE ≈ 78 %
  * warmup-calibrated (v4):   R²(log) = +0.958, MAPE ≈ 14 %

See ``docs/auto_select_v6_combo_predictor.md`` for the integration
recipe (recommended anchor combo, UI flow).

Dependencies: numpy (BSD-3).  No LightGBM, no MIT-licensed runtime dep.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "best_model_v6"
_DEFAULT_TIME_PATH = _DEFAULT_MODEL_DIR / "phys_time.json"


# ---------------------------------------------------------------------------
# Combo parsing — accepts both the v3 4-axis form ("simpleunet_bc64_p256
# _distillOn") and the v6 8-axis form ("…_fp0.5_dw1.0_focal_cws0.0").
# Mirrors ComboPredictor.parse_combo to keep the two predictors in sync.
# ---------------------------------------------------------------------------

def _parse_combo(key: str) -> dict[str, Any]:
    parts = key.split("_")
    out: dict[str, Any] = {
        "arch": parts[0],
        "base_channels": 0,
        "patch_size": 0,
        "distill_on": 0,
        "fg_patch_prob": 0.5,
    }
    for p in parts[1:]:
        if p.startswith("bc") and p[2:].isdigit():
            out["base_channels"] = int(p[2:])
        elif p.startswith("p") and p[1:].isdigit():
            out["patch_size"] = int(p[1:])
        elif p.startswith("distill"):
            out["distill_on"] = 1 if p[len("distill"):].lower() == "on" else 0
        elif p.startswith("fp"):
            try:
                out["fg_patch_prob"] = float(p[2:])
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Physical feature vector — must match the layout used at fit time
# (package.py::PHYS_TIME_FEATURE_NAMES).
# ---------------------------------------------------------------------------

# NOTE: "deeplabv3plus" is retired from the trainer (0.9.7) but MUST stay in
# this tuple — it fixes the arch one-hot dimension order the bundled v6 model
# was trained on. Removing it shifts every downstream feature and corrupts the
# prediction. Retired combos simply never reach the predictor (they are
# filtered out in combo_predictor.combo_is_buildable), so the deeplabv3plus
# one-hot slot is always 0 at inference.
_ARCHS_DEFAULT = ("simpleunet", "stdc", "deeplabv3plus")


def _physical_feature_vec(
    combo: dict[str, Any],
    scalar: dict[str, float],
    archs: tuple[str, ...] = _ARCHS_DEFAULT,
) -> np.ndarray:
    def _f(key: str, default: float = 0.0) -> float:
        v = scalar.get(key)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
        return float(default)

    num_train = max(_f("num_train", 1.0), 1.0)
    log_num_train = _f("log_num_train", math.log(num_train))
    num_total = max(_f("num_total", num_train), 1.0)
    log_num_total = math.log(num_total)
    log_img_pixels = _f("log_img_pixels", 0.0)
    bc = int(combo.get("base_channels") or 0)
    ps = int(combo.get("patch_size") or 0)
    log_bc = math.log(bc) if bc > 0 else 0.0
    log_ps = math.log(ps) if ps > 0 else 0.0
    arch_oh = [1.0 if combo.get("arch") == a else 0.0 for a in archs]
    distill = float(combo.get("distill_on", 0))
    fp = float(combo.get("fg_patch_prob", 0.5))
    return np.asarray(
        [log_num_train, log_num_total, log_img_pixels, log_bc, log_ps,
         *arch_oh, distill, fp],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

@dataclass
class TimePredictor:
    coefs: np.ndarray
    intercept: float
    feature_names: list[str]
    anchor_combo: str
    archs: tuple[str, ...] = _ARCHS_DEFAULT
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> TimePredictor:
        path = Path(path) if path else _DEFAULT_TIME_PATH
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"phys_time.json is not in 'ok' status: {payload.get('reason')}"
            )
        return cls(
            coefs=np.asarray(payload["coefs"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            feature_names=list(payload["feature_names"]),
            anchor_combo=str(payload["anchor_combo"]),
            metadata=payload,
        )

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def _predict_log(self, combos: list[str], scalar: dict[str, float]) -> np.ndarray:
        rows = [
            _physical_feature_vec(_parse_combo(k), scalar, archs=self.archs)
            for k in combos
        ]
        X = np.stack(rows) if rows else np.zeros((0, len(self.coefs)), dtype=np.float64)
        return X @ self.coefs + self.intercept

    def predict_seconds(
        self,
        combos: list[str],
        scalar: dict[str, float],
        anchor_elapsed_sec: float | None = None,
    ) -> np.ndarray:
        """Return predicted elapsed_sec for each combo.

        When ``anchor_elapsed_sec`` is given (the user's actual measured
        runtime for ``self.anchor_combo`` on this project), apply the
        warmup calibration: ``log_pred += log(anchor_elapsed_sec) -
        log_pred_anchor``.  Without an anchor we return the
        physical-only prediction (R² ≈ -0.005 — fine for *relative*
        ordering, poor for absolute magnitude).
        """
        if not combos:
            return np.zeros(0, dtype=np.float64)
        log_pred = self._predict_log(combos, scalar)
        if anchor_elapsed_sec is not None and anchor_elapsed_sec > 0:
            log_pred_anchor = self._predict_log([self.anchor_combo], scalar)[0]
            scale_log = math.log(anchor_elapsed_sec) - float(log_pred_anchor)
            log_pred = log_pred + scale_log
        return np.exp(log_pred)


_cached_time_predictor: TimePredictor | None = None


def get_default_time_predictor() -> TimePredictor | None:
    """Load the bundled v6 time predictor (cached).  Returns None on failure."""
    global _cached_time_predictor
    if _cached_time_predictor is not None:
        return _cached_time_predictor
    if not _DEFAULT_TIME_PATH.exists():
        return None
    try:
        _cached_time_predictor = TimePredictor.load(_DEFAULT_TIME_PATH)
    except Exception as e:
        logger.warning("TimePredictor load failed: %s", e)
        return None
    return _cached_time_predictor
