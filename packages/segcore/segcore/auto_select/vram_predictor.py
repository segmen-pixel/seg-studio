# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""VRAM Predictor v6 — OOM-avoidance for segcore.auto_select.

Estimates the peak GPU memory of an **auto-fit** training run for a
candidate `(combo, project, GPU)` and issues a **WDDM-aware OOM
verdict** with an explicit safety margin.  Companion of
``combo_predictor.py`` — both load from ``models/best_model_v6/``.

Bundle:
  * vram_regressor.json — XGBoost reg:squarederror, emits log(vram_mb)
  * oom_classifier.json — XGBoost binary:logistic, emits P(OOM)
  * vram_metadata.json  — feature names, safety policy, LOPO metrics

Trained on the wave5 cross-device probe (9,669 rows / 37 projects).
LOPO: regressor MAPE ≈ 3.6 % / R²(log) ≈ 0.99; classifier AUC ≈ 0.99.

Why no batch_size input
-----------------------
wave5 measured each cell at its **auto-fit** batch size — the trainer's
VRAM dry-run picks the largest batch that fits the GPU.  Heavier
settings get a *smaller* auto-fit batch yet still peak *higher*, so
``batch_size`` and ``vram_peak`` are negatively correlated
(corr = -0.24).  Feeding batch_size to the model teaches it the
inverted causality.  The predictor therefore estimates the peak of an
*auto-fit* run directly; ``gpu_total_mb`` (which determines the auto-fit
batch) is the hardware-budget proxy.  This means the predictor answers
"will training this combo on this GPU OOM?" — not "what batch fits?".

Why a safety margin (the WDDM problem)
--------------------------------------
Windows **WDDM** reserves and recycles VRAM behind PyTorch's back;
cuDNN warms up to heavier algorithms and the compositor reclaims
surfaces.  In wave5 all 542 OOM events hit the Linux 3080 Ti running
with no headroom; the WDDM GPUs (2 GB headroom) never OOM'd.  The
regressor also under-predicts on ~67 % of rows, so the verdict layer
inflates the estimate by the LOPO 95th-percentile under-prediction
band and subtracts a driver-specific headroom before the budget check.

See ``docs/auto_select_v6_combo_predictor.md`` §7.

Dependencies: xgboost (Apache-2.0), numpy (BSD-3).  No LightGBM.
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
_DEFAULT_VRAM_META = _DEFAULT_MODEL_DIR / "vram_metadata.json"

# NOTE: "deeplabv3plus" is retired from the trainer (0.9.7) but MUST stay here —
# it fixes the arch one-hot dimension order the bundled v6 VRAM model expects.
# Removing it shifts every downstream feature and corrupts the prediction.
# Retired combos never reach the predictor (combo_predictor.combo_is_buildable),
# so this one-hot slot is always 0 at inference.
ARCHS: tuple[str, ...] = ("simpleunet", "stdc", "deeplabv3plus")
LOSSES: tuple[str, ...] = ("focal", "lovasz", "ce")


# ---------------------------------------------------------------------------
# Combo parsing — accepts the 8-axis condition string and the legacy
# 4-axis form (recipe knobs fall back to neutral defaults).
# ---------------------------------------------------------------------------

def parse_combo(key: str) -> dict[str, Any]:
    """Parse a combo condition string into a feature dict.

    Accepts the 8-axis condition string and the legacy 4-axis form.
    Recipe knobs that are missing from the key fall back to neutral
    defaults so the result is always safe to feed to the predictor.

    Key format:
        ``<arch>_<token>_<token>_...`` where the first underscore-separated
        segment is the architecture name (one of ``ARCHS``) and each later
        token is parsed by prefix:

        ===========  ==================================================
        Prefix       Meaning
        ===========  ==================================================
        ``bc<int>``  ``base_channels`` (e.g. ``bc32``)
        ``p<int>``   ``patch_size`` (e.g. ``p512``)
        ``distill``  ``distill_on``: ``distillon`` -> 1, else 0
        ``fp<num>``  ``fg_patch_prob`` (e.g. ``fp0.7``)
        ``dw<num>``  ``dice_weight`` (e.g. ``dw2.0``)
        ``cws<num>`` ``class_weight_strength`` (e.g. ``cws0.3``)
        in LOSSES    ``loss_type``: literal ``focal``/``lovasz``/``ce``
        ===========  ==================================================

        Numeric tokens that fail to parse are silently ignored (the
        default value stays). Unknown tokens are skipped.

    Args:
        key: Combo condition string, e.g.
            ``"simpleunet_bc32_p512_distillon_fp0.7_dw2.0_cws0.3_ce"``.

    Returns:
        Dict with keys (all always present):
            - ``arch`` (``str``): first segment of the key, unvalidated.
            - ``base_channels`` (``int``, default ``0``)
            - ``patch_size`` (``int``, default ``0``)
            - ``distill_on`` (``int`` 0/1, default ``0``)
            - ``fg_patch_prob`` (``float``, default ``0.5``)
            - ``dice_weight`` (``float``, default ``1.0``)
            - ``loss_type`` (``str``, default ``"ce"``)
            - ``class_weight_strength`` (``float``, default ``0.0``)
    """
    parts = key.split("_")
    out: dict[str, Any] = {
        "arch": parts[0],
        "base_channels": 0,
        "patch_size": 0,
        "distill_on": 0,
        "fg_patch_prob": 0.5,
        "dice_weight": 1.0,
        "loss_type": "ce",
        "class_weight_strength": 0.0,
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
        elif p.startswith("dw"):
            try:
                out["dice_weight"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("cws"):
            try:
                out["class_weight_strength"] = float(p[3:])
            except ValueError:
                pass
        elif p in LOSSES:
            out["loss_type"] = p
    return out


# ---------------------------------------------------------------------------
# Safety policy
# ---------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    """WDDM-aware safety policy.  Loaded from vram_metadata.json::safety."""
    underpred_log_p95: float = 0.171
    wddm_headroom_mb: float = 2048.0
    wddm_usable_fraction: float = 0.92
    linux_headroom_mb: float = 512.0
    linux_usable_fraction: float = 0.94
    oom_prob_veto: float = 0.5

    @property
    def safety_multiplier(self) -> float:
        return math.exp(self.underpred_log_p95)

    def budget_mb(self, gpu_total_mb: float, is_wddm: bool) -> float:
        if is_wddm:
            b = gpu_total_mb * self.wddm_usable_fraction - self.wddm_headroom_mb
        else:
            b = gpu_total_mb * self.linux_usable_fraction - self.linux_headroom_mb
        return max(b, 0.0)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

@dataclass
class VramPredictor:
    regressor: Any                # xgboost.Booster (log-VRAM)
    classifier: Any | None        # xgboost.Booster (P(OOM)) or None
    feature_names: list[str]
    safety: SafetyConfig
    metadata: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> VramPredictor:
        import xgboost as xgb

        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        meta = json.loads((model_dir / "vram_metadata.json").read_text(encoding="utf-8"))
        regressor = xgb.Booster()
        regressor.load_model(str(model_dir / meta.get("regressor_path", "vram_regressor.json")))
        classifier = None
        clf_rel = meta.get("classifier_path")
        if clf_rel and (model_dir / clf_rel).exists():
            try:
                classifier = xgb.Booster()
                classifier.load_model(str(model_dir / clf_rel))
            except Exception as e:
                logger.warning("OOM classifier load failed: %s", e)
                classifier = None
        s = meta.get("safety", {})
        safety = SafetyConfig(
            underpred_log_p95=float(s.get("underpred_log_p95", 0.171)),
            wddm_headroom_mb=float(s.get("wddm_headroom_mb", 2048.0)),
            wddm_usable_fraction=float(s.get("wddm_usable_fraction", 0.92)),
            linux_headroom_mb=float(s.get("linux_headroom_mb", 512.0)),
            linux_usable_fraction=float(s.get("linux_usable_fraction", 0.94)),
            oom_prob_veto=float(s.get("oom_prob_veto", 0.5)),
        )
        return cls(
            regressor=regressor,
            classifier=classifier,
            feature_names=list(meta["feature_names"]),
            safety=safety,
            metadata=meta,
        )

    # -- feature assembly ---------------------------------------------------

    def _build_row(
        self,
        combo: dict[str, Any],
        is_wddm: bool,
        gpu_total_mb: float,
        num_train: float,
    ) -> list[float]:
        """Feature row.  Order must match vram_metadata.json::feature_names.

        No ``batch_size`` — see the module docstring.
        """
        bc = float(combo["base_channels"])
        ps = float(combo["patch_size"])
        ps2 = ps * ps
        loss = combo["loss_type"]
        row: list[float] = []
        row.extend(1.0 if combo["arch"] == a else 0.0 for a in ARCHS)
        row.append(bc)
        row.append(math.log2(bc) if bc > 0 else 0.0)
        row.append(ps)
        row.append(math.log2(ps) if ps > 0 else 0.0)
        row.append(float(combo["distill_on"]))
        row.append(float(combo["fg_patch_prob"]))
        row.append(float(combo["dice_weight"]))
        row.append(float(combo["class_weight_strength"]))
        row.extend(1.0 if loss == loss_name else 0.0 for loss_name in LOSSES)
        row.append(1.0 if is_wddm else 0.0)
        row.append(float(gpu_total_mb))
        row.append(float(num_train))
        row.append(math.log1p(max(num_train, 0.0)))
        is_lovasz = 1.0 if loss == "lovasz" else 0.0
        row.append(bc * ps2 / 1e6)
        row.append(is_lovasz * bc)
        return row

    # -- raw model heads ----------------------------------------------------

    def predict_vram_mb(
        self,
        combo: str | dict[str, Any],
        gpu_total_mb: float,
        is_wddm: bool,
        num_train: float,
    ) -> float:
        """Expected peak VRAM (MB) of an auto-fit run — raw, no safety margin."""
        import xgboost as xgb

        c = parse_combo(combo) if isinstance(combo, str) else combo
        row = self._build_row(c, is_wddm, gpu_total_mb, num_train)
        dmat = xgb.DMatrix(np.asarray([row], dtype=np.float64),
                           feature_names=self.feature_names)
        log_pred = float(self.regressor.predict(dmat)[0])
        return math.exp(log_pred)

    def predict_oom_prob(
        self,
        combo: str | dict[str, Any],
        gpu_total_mb: float,
        is_wddm: bool,
        num_train: float,
    ) -> float | None:
        """Direct P(OOM) from the classifier head, or None if not bundled."""
        if self.classifier is None:
            return None
        import xgboost as xgb

        c = parse_combo(combo) if isinstance(combo, str) else combo
        row = self._build_row(c, is_wddm, gpu_total_mb, num_train)
        dmat = xgb.DMatrix(np.asarray([row], dtype=np.float64),
                           feature_names=self.feature_names)
        return float(self.classifier.predict(dmat)[0])

    # -- safety-aware verdict ----------------------------------------------

    def verdict(
        self,
        combo: str | dict[str, Any],
        gpu_total_mb: float,
        is_wddm: bool,
        num_train: float,
    ) -> dict:
        """Decide whether auto-fit training of this combo will OOM.

        Returns ``{verdict, reason, pred_vram_mb, vram_safe_mb, budget_mb,
        headroom_mb, utilization, oom_prob, driver}``.  ``verdict`` is
        ``"ok"`` or ``"oom_risk"``.

        The estimate is the peak VRAM of an *auto-fit* run; the trainer
        will pick its own batch size.  If the verdict is ``oom_risk`` the
        operator should drop ``base_channels`` or switch ``lovasz`` →
        ``ce``/``focal`` (the wave5-measured VRAM-cheapest losses).
        """
        pred = self.predict_vram_mb(combo, gpu_total_mb, is_wddm, num_train)
        oom_prob = self.predict_oom_prob(combo, gpu_total_mb, is_wddm, num_train)
        vram_safe = pred * self.safety.safety_multiplier
        budget = self.safety.budget_mb(gpu_total_mb, is_wddm)
        over_budget = vram_safe > budget
        veto = oom_prob is not None and oom_prob >= self.safety.oom_prob_veto
        if over_budget and veto:
            reason = "budget+classifier"
        elif over_budget:
            reason = "budget"
        elif veto:
            reason = "classifier"
        else:
            reason = "ok"
        return {
            "verdict": "oom_risk" if (over_budget or veto) else "ok",
            "reason": reason,
            "pred_vram_mb": float(pred),
            "vram_safe_mb": float(vram_safe),
            "budget_mb": float(budget),
            "headroom_mb": float(budget - vram_safe),
            "utilization": float(vram_safe / budget) if budget > 0 else float("inf"),
            "oom_prob": oom_prob,
            "driver": "wddm" if is_wddm else "linux",
        }


_cached_vram_predictor: VramPredictor | None = None


def get_default_vram_predictor() -> VramPredictor | None:
    """Load the bundled v6 VRAM predictor (cached).  None on failure."""
    global _cached_vram_predictor
    if _cached_vram_predictor is not None:
        return _cached_vram_predictor
    if not _DEFAULT_VRAM_META.exists():
        return None
    try:
        _cached_vram_predictor = VramPredictor.load(_DEFAULT_MODEL_DIR)
    except Exception as e:
        logger.warning("VramPredictor load failed: %s", e)
        return None
    return _cached_vram_predictor
