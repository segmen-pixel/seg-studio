# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The Tversky defaults must agree wherever they are written down.

They did not. losses.tversky_loss multiplies alpha by FALSE POSITIVES and its
signature defaults (alpha=0.3, beta=0.7) match its docstring, "alpha < beta
biases toward recall". Every configuration layer defaulted to alpha=0.7 /
beta=0.3 -- the precision-biased direction -- and train_phase_train passes the
config values under a comment reading "FN-biased learning (micro-defect
focus)". Since tversky_weight defaults to 1.0 the term was active in every run,
nothing logged the contradiction, and the tests called tversky_loss with no
alpha/beta, so they exercised 0.3/0.7 while training ran 0.7/0.3.

Same failure mode as the patch-size default: one number, several homes.
"""
from __future__ import annotations

import inspect

import torch

from segcore.training.losses import tversky_loss
from segcore.training.train_config import TrainConfig

PARAMS = ("tversky_alpha", "tversky_beta", "tversky_gamma")

# TrainConfig has required positional args; none of them touch the tversky
# defaults, they just have to be present for it to construct.
_REQUIRED = dict(
    input_size=[64, 64],
    output_stride=2,
    epochs=1,
    batch_size=2,
    lr=1e-3,
    ignore_index=255,
    normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
)


def _loss_signature_defaults() -> dict[str, float]:
    sig = inspect.signature(tversky_loss)
    return {f"tversky_{k}": float(sig.parameters[k].default) for k in ("alpha", "beta", "gamma")}


def test_train_config_matches_the_loss_signature():
    cfg = TrainConfig(**_REQUIRED)
    sig = _loss_signature_defaults()
    got = {k: float(getattr(cfg, k)) for k in PARAMS}
    assert got == sig, f"TrainConfig {got} != tversky_loss signature {sig}"


def test_api_schema_matches_the_loss_signature():
    import sys
    from pathlib import Path as _P
    api = _P(__file__).resolve().parents[3] / "apps" / "trainer_api"
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    from app.schemas import TrainRequest  # noqa: PLC0415

    sig = _loss_signature_defaults()
    fields = TrainRequest.model_fields
    got = {k: float(fields[k].default) for k in PARAMS}
    assert got == sig, f"TrainRequest {got} != tversky_loss signature {sig}"


def test_alpha_weights_false_positives_not_false_negatives():
    """Pin the convention, so the pair cannot be silently transposed again.

    alpha on FP means: raising alpha must punish a false-positive-only
    prediction and leave a false-negative-only one alone.
    """
    torch.manual_seed(0)
    # one pixel of class 1 in GT, prediction misses it entirely -> FN only
    logits_fn = torch.tensor([[[[9.0]], [[-9.0]]]])
    target_fg = torch.tensor([[[1]]])
    # GT background, prediction claims class 1 -> FP only
    logits_fp = torch.tensor([[[[-9.0]], [[9.0]]]])
    target_bg = torch.tensor([[[0]]])

    kw = dict(num_classes=2, ignore_index=255, gamma=1.0)
    fn_low_alpha = tversky_loss(logits_fn, target_fg, alpha=0.1, beta=0.9, **kw)
    fn_high_alpha = tversky_loss(logits_fn, target_fg, alpha=0.9, beta=0.1, **kw)
    fp_low_alpha = tversky_loss(logits_fp, target_bg, alpha=0.1, beta=0.9, **kw)
    fp_high_alpha = tversky_loss(logits_fp, target_bg, alpha=0.9, beta=0.1, **kw)

    assert fp_high_alpha > fp_low_alpha, (
        "raising alpha did not punish false positives more -- alpha is on the "
        f"wrong term ({fp_low_alpha.item():.4f} -> {fp_high_alpha.item():.4f})"
    )
    assert fn_high_alpha < fn_low_alpha, (
        "raising alpha (lowering beta) did not relax the false-negative penalty "
        f"({fn_low_alpha.item():.4f} -> {fn_high_alpha.item():.4f})"
    )


def test_shipped_defaults_are_recall_biased():
    """The shipped pair must favour recall, which is what every comment claims."""
    cfg = TrainConfig(**_REQUIRED)
    assert cfg.tversky_alpha < cfg.tversky_beta, (
        f"alpha={cfg.tversky_alpha} >= beta={cfg.tversky_beta}: alpha weights FP, "
        "so this is a precision bias, the opposite of 'micro-defect focus'"
    )
