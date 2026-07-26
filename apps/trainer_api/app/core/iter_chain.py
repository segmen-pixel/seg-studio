# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Iterative hard-mining chain control: end-of-run fan-out of the next
iteration and the end-of-chain best declaration.

Extracted verbatim from training_runner.py during the pre-OSS refactor;
training_runner re-exports both names.
"""
from __future__ import annotations

import json
from pathlib import Path

from .training_launcher import _launch_single_run


def _summarize_iter_chain(prev_run_path: Path, config: dict, log_fn) -> None:
    """Rank every completed iteration of this chain and declare the best.

    The chain warm-starts each iteration from the previous model and keeps
    the LAST iteration on top of the run list, but hard-boosting can make
    later iterations strictly worse (observed: iter0 F1 0.534 -> iter2
    0.425 on a 38-image project). At chain end this compares the honest
    per-image SW macro F1 across all iterations of the group, writes
    chain_summary.json into the final run dir and a chain_best.json marker
    into the winning run dir, and logs which run to use.

    Never raises — a summary failure must not break run post-processing.
    """
    try:
        group_id = config.get("iter_group_id")
        if not group_id:
            return
        runs_root = prev_run_path.parent
        entries = []
        for run_dir in runs_root.iterdir():
            cfg_path = run_dir / "train_config.json"
            pim_path = run_dir / "per_image_metrics.json"
            if not (run_dir.is_dir() and cfg_path.exists() and pim_path.exists()):
                continue
            try:
                run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(run_cfg.get("iter_group_id")) != str(group_id):
                continue
            try:
                pim = json.loads(pim_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            # Dataset-level micro P/R/F1 — the same judge metric the
            # completion hook uses for the iter-stop decision, and the
            # same aggregation the UI 全画像スコア shows. A mean of
            # per-image macros would let clean-image 0.0 entries dominate.
            from segcore.training.train import _dataset_micro_prf
            micro_prec, micro_rec = _dataset_micro_prf(pim)
            if micro_prec == 0.0 and micro_rec == 0.0:
                continue
            micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                        if (micro_prec + micro_rec) > 0 else 0.0)
            entries.append({
                "run_id": run_dir.name,
                "iter_index": int(run_cfg.get("iter_index", 0) or 0),
                "micro_f1": float(micro_f1),
                "micro_prec": float(micro_prec),
                "micro_rec": float(micro_rec),
                "n_images": len(pim),
                "inference_threshold": run_cfg.get("inference_threshold"),
            })
        if not entries:
            return
        entries.sort(key=lambda e: e["iter_index"])
        best = max(entries, key=lambda e: e["micro_f1"])
        summary = {
            "iter_group_id": str(group_id),
            "metric": "dataset_micro_f1",
            "runs": entries,
            "best_run_id": best["run_id"],
            "best_iter_index": best["iter_index"],
            "best_micro_f1": best["micro_f1"],
        }
        (prev_run_path / "chain_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (runs_root / best["run_id"] / "chain_best.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        scores = ", ".join(
            f"iter{e['iter_index']} F1={e['micro_f1']:.3f}" for e in entries
        )
        log_fn(
            f"Iterative chain summary: {scores} -> BEST: iter "
            f"{best['iter_index']} (run {best['run_id'][:8]}, "
            f"F1={best['micro_f1']:.3f}). Export/infer from that run.\n"
        )
    except Exception as _sum_err:
        log_fn(f"Iterative: chain summary failed ({_sum_err})\n")


def _maybe_launch_next_iteration(
    project_id: str,
    prev_run_id: str,
    prev_run_path: Path,
    config: dict,
    log_fn,
) -> None:
    """If the run was launched in iterative mode and the metric target was
    missed (train.py wrote iterative_hard_ids.json), copy the config, bump
    iter_index, wire the previous model as pretrained_checkpoint + attach the
    hard_ids, and launch the next iteration. Runs at most `iter_max` times."""
    if not bool(config.get("iterative_mode")):
        return
    iter_index = int(config.get("iter_index", 0) or 0)
    iter_max = int(config.get("iter_max", 3) or 3)
    if iter_index + 1 >= iter_max:
        log_fn(f"Iterative: reached iter_max={iter_max}; chain ends.\n")
        _summarize_iter_chain(prev_run_path, config, log_fn)
        return
    hard_path = prev_run_path / "iterative_hard_ids.json"
    if not hard_path.exists():
        log_fn("Iterative: target met; chain ends.\n")
        _summarize_iter_chain(prev_run_path, config, log_fn)
        return
    try:
        hard_data = json.loads(hard_path.read_text(encoding="utf-8"))
        hard_ids = list(hard_data.get("hard_ids") or [])
    except Exception as _read_err:
        log_fn(f"Iterative: could not read hard_ids ({_read_err}); chain ends.\n")
        _summarize_iter_chain(prev_run_path, config, log_fn)
        return
    if not hard_ids:
        _summarize_iter_chain(prev_run_path, config, log_fn)
        return
    next_config = dict(config)
    next_config["iter_index"] = iter_index + 1
    next_config["pretrained_checkpoint"] = str(prev_run_path / "model.pt")
    next_config["hard_ids"] = hard_ids
    # iter_group_id is inherited unchanged.
    # Inherit the RESOLVED inference threshold from the previous run so the
    # whole chain deploys, judges and hard-mines at one threshold. The
    # in-memory config only holds the request value; per-iteration re-search
    # on a handful of val images produced wild swings (0.02 -> 0.38 -> 0.52)
    # that made iterations incomparable.
    try:
        _prev_cfg = json.loads(
            (prev_run_path / "train_config.json").read_text(encoding="utf-8")
        )
        _prev_thr = _prev_cfg.get("inference_threshold")
        if _prev_thr is not None:
            next_config["inference_threshold"] = _prev_thr
    except (json.JSONDecodeError, OSError):
        pass
    # Adaptive epochs: a best_epoch at (or within 2 of) the cap means the
    # model was still improving when training was cut off — observed as
    # best_epoch=80/80 on every iteration of a chain, with low-confidence
    # TPs that longer training would sharpen. Give the next iteration 1.5x
    # epochs (capped at 500) instead of re-hitting the same wall.
    try:
        _mets = json.loads(
            (prev_run_path / "metrics.json").read_text(encoding="utf-8")
        )
        _best_ep = int(_mets.get("best_epoch") or 0)
        # In-run convergence extension may have trained past the requested
        # epochs; judge against what actually ran.
        _prev_epochs = int(_mets.get("epochs_effective") or config.get("epochs") or 0)
        if (bool(config.get("auto_epochs", True))
                and _prev_epochs > 0 and _best_ep >= _prev_epochs - 2):
            _new_epochs = min(500, int(round(_prev_epochs * 1.5)))
            if _new_epochs > _prev_epochs:
                next_config["epochs"] = _new_epochs
                log_fn(
                    f"Iterative: best epoch {_best_ep}/{_prev_epochs} hit the "
                    f"cap (model still improving); raising epochs to "
                    f"{_new_epochs} for the next iteration.\n"
                )
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    log_fn(
        f"Iterative: launching next iteration "
        f"({iter_index + 1}/{iter_max - 1}) with {len(hard_ids)} hard IDs\n"
    )
    try:
        _launch_single_run(project_id, next_config)
    except Exception as _launch_err:
        log_fn(f"Iterative: launch failed ({_launch_err})\n")
