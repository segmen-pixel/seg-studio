# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""XGBoost ensemble combo predictor (v6, OSS-clean stack).

See ``docs/auto_select_v6_combo_predictor.md`` for the full method
note (architecture, training, LOPO benchmark, operational notes).

Loads the v6 production bundle written by
``scripts/research/combo_predictor_v6/package.py``:

  * regressor.json  — XGBoost ``reg:squarederror`` booster (F1 magnitude)
  * ranker.json     — XGBoost ``rank:ndcg`` LambdaRank booster (ordering)
  * dino_pca.pkl    — scikit-learn PCA over the DINOv2 embedding
  * metadata.json   — schema, feature names, archs, dino_dims, ensemble
                      weight, scalar z-score mean/std, combo list

The ranker emits unitless scores; the regressor emits calibrated F1 in
[0, 1].  We min-max normalise each model's scores over the candidate
combo set and mix them as ``score = w * reg_norm + (1 - w) * rank_norm``
(``w = ensemble_weight_reg``, default 0.8 — the LOPO-best choice in the
2026-05-21 sweep).

Dependencies: ``xgboost`` (Apache-2.0), ``scikit-learn`` (BSD-3),
``numpy`` (BSD-3).  No LightGBM, no MIT-licensed runtime deps.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "best_model_v6"

# Architecture prefixes the recommender must never emit. The bundled v6 model
# was trained on a combo library that still contained "deeplabv3plus_*" rows,
# but that architecture was removed from the trainer in 0.9.7. We do NOT
# retrain the model — instead every candidate list is filtered to the
# buildable architectures at load time, so a retired combo can never be
# recommended for a build that would then fail with "Unknown architecture".
_RETIRED_ARCH_PREFIXES: tuple[str, ...] = ("deeplabv3plus_",)


def combo_is_buildable(combo_key: str) -> bool:
    """True unless ``combo_key`` names an architecture retired from the trainer."""
    return not combo_key.startswith(_RETIRED_ARCH_PREFIXES)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass
class ComboPredictor:
    regressor: Any                  # xgboost.Booster
    ranker: Any                     # xgboost.Booster
    feature_columns: list[str]
    scalar_feature_names: list[str]
    archs: list[str]
    losses: list[str]
    dino_dims: int
    all_combos: list[str]
    pca: Any | None = None
    scalar_zscore_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scalar_zscore_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    ensemble_weight_reg: float = 0.8
    time_predictor: Any | None = None       # segcore.auto_select.time_predictor.TimePredictor
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> ComboPredictor:
        import xgboost as xgb

        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        regressor = xgb.Booster()
        regressor.load_model(str(model_dir / meta.get("regressor_path", "regressor.json")))
        ranker = xgb.Booster()
        ranker.load_model(str(model_dir / meta.get("ranker_path", "ranker.json")))
        pca = None
        pca_path = model_dir / meta.get("pca_path", "dino_pca.pkl")
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
        mean = np.asarray(meta.get("scalar_zscore_mean", []), dtype=np.float64)
        std = np.asarray(meta.get("scalar_zscore_std", []), dtype=np.float64)
        if std.size:
            std = np.where(std < 1e-8, 1.0, std)
        time_predictor = None
        phys_time_rel = meta.get("phys_time_path")
        if phys_time_rel:
            phys_time_path = model_dir / phys_time_rel
            if phys_time_path.exists():
                try:
                    from .time_predictor import TimePredictor
                    time_predictor = TimePredictor.load(phys_time_path)
                except Exception as e:
                    logger.warning("TimePredictor load failed: %s", e)
        return cls(
            regressor=regressor,
            ranker=ranker,
            feature_columns=meta["feature_columns"],
            scalar_feature_names=meta["scalar_feature_names"],
            archs=meta["archs"],
            losses=meta.get("losses", ["focal", "lovasz", "ce"]),
            dino_dims=int(meta.get("dino_dims", 0)),
            all_combos=[c for c in meta["all_combos"] if combo_is_buildable(c)],
            pca=pca,
            scalar_zscore_mean=mean,
            scalar_zscore_std=std,
            ensemble_weight_reg=float(meta.get("ensemble_weight_reg", 0.8)),
            time_predictor=time_predictor,
            metadata=meta,
        )

    @property
    def anchor_combo(self) -> str | None:
        """Recommended anchor combo for warmup time calibration, if bundled."""
        return self.time_predictor.anchor_combo if self.time_predictor is not None else None

    # -------------------------------------------------------------------
    # Combo identifier parsing.  Accepts both the legacy 4-axis form
    # ("simpleunet_bc64_p256_distillOn") and the v6 8-axis form
    # ("simpleunet_bc64_p256_distillOn_fp0.5_dw1.0_focal_cws0.0").
    # -------------------------------------------------------------------

    @staticmethod
    def parse_combo(key: str) -> dict[str, Any]:
        """Parse a combo identifier into the recipe knobs the model expects.

        Missing recipe axes (legacy 4-axis combos) fall back to neutral
        defaults: ``fg_patch_prob=0.5``, ``dice_weight=1.0``,
        ``loss_type='ce'``, ``class_weight_strength=0.0``.
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
            elif p in ("focal", "lovasz", "ce"):
                out["loss_type"] = p
        return out

    # -------------------------------------------------------------------
    # Feature assembly — mirrors `feature_extractor.combo_feature_vec`
    # and `project_feature_vec` from the research script verbatim.
    # -------------------------------------------------------------------

    def _project_dino_vec(self, dino_vec_768: np.ndarray | None) -> np.ndarray:
        if self.dino_dims == 0:
            return np.zeros(0, dtype=np.float64)
        if dino_vec_768 is None:
            return np.zeros(self.dino_dims, dtype=np.float64)
        vec = np.asarray(dino_vec_768, dtype=np.float64).reshape(1, -1)
        if self.pca is not None:
            reduced = self.pca.transform(vec)[0]
        else:
            reduced = vec[0][: self.dino_dims]
        out = np.zeros(self.dino_dims, dtype=np.float64)
        out[: len(reduced)] = reduced[: self.dino_dims]
        return out

    def _build_row(
        self,
        scalar_features: dict[str, float],
        dino_reduced: np.ndarray,
        combo: dict[str, Any],
    ) -> list[float]:
        row: list[float] = []
        # 26 scalar features
        for name in self.scalar_feature_names:
            val = scalar_features.get(name)
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                row.append(float(val))
            else:
                row.append(float("nan"))
        # DINOv2 PCA components
        row.extend(float(v) for v in dino_reduced.tolist())
        # arch one-hot
        for a in self.archs:
            row.append(1.0 if combo["arch"] == a else 0.0)
        # base_channels (raw + log2), patch_size (raw + log2), distill
        bc = combo["base_channels"]
        ps = combo["patch_size"]
        row.append(float(bc))
        row.append(math.log2(bc) if bc > 0 else 0.0)
        row.append(float(ps))
        row.append(math.log2(ps) if ps > 0 else 0.0)
        row.append(float(combo["distill_on"]))
        # recipe axes
        row.append(float(combo["fg_patch_prob"]))
        row.append(float(combo["dice_weight"]))
        row.append(float(combo["class_weight_strength"]))
        # loss_type one-hot
        for loss_name in self.losses:
            row.append(1.0 if combo["loss_type"] == loss_name else 0.0)
        # cross-interactions (these are appended only if the trained model
        # has them; the metadata column list is the source of truth).
        n_remaining = len(self.feature_columns) - len(row)
        if n_remaining > 0:
            fg_ratio = float(scalar_features.get("fg_ratio", 0.0) or 0.0)
            cir = float(scalar_features.get("class_imbalance_ratio", 0.0) or 0.0)
            d = float(combo["distill_on"])
            interactions = [
                d * float(bc),
                d * float(ps),
                float(combo["fg_patch_prob"]) * fg_ratio,
                (1.0 if combo["loss_type"] == "focal" else 0.0) * cir,
                (1.0 if combo["loss_type"] == "lovasz" else 0.0) * float(bc),
            ]
            row.extend(interactions[:n_remaining])
        return row

    def _apply_zscore(self, X: np.ndarray) -> np.ndarray:
        n_scalar = len(self.scalar_feature_names)
        if n_scalar == 0 or self.scalar_zscore_mean.size != n_scalar:
            return X
        out = X.copy()
        out[:, :n_scalar] = (out[:, :n_scalar] - self.scalar_zscore_mean) / self.scalar_zscore_std
        return out

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def rank(
        self,
        scalar_features: dict[str, float],
        dino_vec_768: np.ndarray | None = None,
        candidate_combos: list[str] | None = None,
        anchor_elapsed_sec: float | None = None,
    ) -> list[dict]:
        """Score candidate combos and return them sorted by ensemble score.

        Each row: ``{combo, arch, base_channels, patch_size, distill_on,
        rank_score, pred_f1, pred_std, ci_low, ci_high,
        pred_elapsed_sec, pred_elapsed_min}``.

        ``pred_f1`` is the regressor output (calibrated F1).  ``pred_std``
        is the magnitude of disagreement between ranker and regressor
        ordering, expressed on the same [0, 1] scale as ``pred_f1`` —
        higher values mean the two models disagree more about this combo's
        position, which is the v6 stand-in for the v3 calibrator ensemble
        std.  ``ci_low / ci_high`` is ``pred_f1 ± 1.96 * pred_std``.

        ``pred_elapsed_sec`` is the v6 warmup-calibrated training time
        prediction.  If the caller supplies ``anchor_elapsed_sec`` (the
        actual measured runtime of ``self.anchor_combo`` on this
        project), the time predictor recovers LOPO R²(log) ≈ +0.958 /
        MAPE ≈ 14 %.  Without an anchor the physical-only prediction is
        returned (good for *relative* ordering, poor for absolute
        magnitude — display as a rough estimate).  Both fields are
        ``None`` when the time predictor bundle is missing.
        """
        import xgboost as xgb

        combos = candidate_combos if candidate_combos else self.all_combos
        # Guard the explicit-candidates path too: a caller-supplied list must
        # not smuggle a retired architecture back into a recommendation.
        combos = [c for c in combos if combo_is_buildable(c)]
        if not combos:
            return []
        rows: list[list[float]] = []
        parsed: list[dict[str, Any]] = []
        dino_reduced = self._project_dino_vec(dino_vec_768)
        for key in combos:
            combo = self.parse_combo(key)
            rows.append(self._build_row(scalar_features, dino_reduced, combo))
            parsed.append(combo)
        X = np.asarray(rows, dtype=np.float64)
        X = self._apply_zscore(X)
        dmat = xgb.DMatrix(X, feature_names=self.feature_columns)

        reg_preds = np.asarray(self.regressor.predict(dmat), dtype=np.float64)
        rank_preds = np.asarray(self.ranker.predict(dmat), dtype=np.float64)
        reg_norm = _minmax(reg_preds)
        rank_norm = _minmax(rank_preds)
        w = float(self.ensemble_weight_reg)
        ensemble = w * reg_norm + (1.0 - w) * rank_norm

        # Per-row disagreement = |reg_norm - rank_norm|, scaled into the
        # F1 range by multiplying by the empirical regressor span so the
        # ci_low/ci_high band stays in plausible F1 units.
        reg_span = float(reg_preds.max() - reg_preds.min()) if reg_preds.size else 0.0
        disagreement = np.abs(reg_norm - rank_norm) * max(reg_span, 1e-3)

        time_seconds: np.ndarray | None = None
        if self.time_predictor is not None:
            try:
                time_seconds = self.time_predictor.predict_seconds(
                    combos=list(combos),
                    scalar=scalar_features,
                    anchor_elapsed_sec=anchor_elapsed_sec,
                )
            except Exception as e:
                logger.warning("TimePredictor.predict_seconds failed: %s", e)
                time_seconds = None

        order = np.argsort(-ensemble)
        out: list[dict] = []
        for i in order:
            combo = parsed[int(i)]
            pred = float(reg_preds[int(i)])
            std = float(disagreement[int(i)])
            row = {
                "combo": combos[int(i)],
                "arch": combo["arch"],
                "base_channels": int(combo["base_channels"]),
                "patch_size": int(combo["patch_size"]),
                "distill_on": bool(combo["distill_on"]),
                "rank_score": float(ensemble[int(i)]),
                "pred_f1": pred,
                "pred_std": std,
                "ci_low": pred - 1.96 * std,
                "ci_high": pred + 1.96 * std,
                "pred_elapsed_sec": (
                    float(time_seconds[int(i)]) if time_seconds is not None else None
                ),
                "pred_elapsed_min": (
                    float(time_seconds[int(i)]) / 60.0 if time_seconds is not None else None
                ),
            }
            out.append(row)
        return out


_cached_predictor: ComboPredictor | None = None
_load_error: str | None = None


def get_default_predictor() -> ComboPredictor | None:
    """Load the bundled v6 predictor (cached).  Returns None on failure."""
    global _cached_predictor, _load_error
    if _cached_predictor is not None:
        return _cached_predictor
    if not (_DEFAULT_MODEL_DIR / "metadata.json").exists():
        _load_error = f"bundle metadata.json not found in {_DEFAULT_MODEL_DIR}"
        return None
    try:
        _cached_predictor = ComboPredictor.load(_DEFAULT_MODEL_DIR)
    except Exception as e:
        _load_error = str(e)
        logger.warning("ComboPredictor load failed: %s", e)
        return None
    _load_error = None
    return _cached_predictor


def get_default_predictor_load_error() -> str | None:
    """Why the last get_default_predictor() call returned None (None = no error)."""
    return _load_error
