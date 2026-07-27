# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""read_epoch_val_metrics: the per-epoch tail the training monitor streams
into the run log."""
from __future__ import annotations

from segcore.instseg.train_rfdetr import read_epoch_val_metrics

_HEADER = "epoch,step,train/loss,val/segm_mAP_50_95,val/segm_mAP_50,val/F1\n"


def _write(tmp_path, body):
    p = tmp_path / "metrics.csv"
    p.write_text(_HEADER + body, encoding="utf-8")
    return p


def test_missing_file_returns_empty(tmp_path):
    assert read_epoch_val_metrics(tmp_path / "nope.csv") == []


def test_val_rows_parsed_per_epoch(tmp_path):
    p = _write(tmp_path,
               "0,10,5.0,,,\n"          # train-only row: skipped
               "0,10,,0.10,0.20,0.15\n"
               "1,20,4.0,,,\n"
               "1,20,,0.30,0.40,0.35\n")
    rows = read_epoch_val_metrics(p)
    assert [r["epoch"] for r in rows] == [0, 1]
    assert rows[0]["segm_map"] == 0.10
    assert rows[1]["segm_map50"] == 0.40
    assert rows[1]["f1"] == 0.35
    # train loss is merged from the per-epoch train rows
    assert rows[0]["train_loss"] == 5.0
    assert rows[1]["train_loss"] == 4.0


def test_after_epoch_filters_already_reported(tmp_path):
    p = _write(tmp_path,
               "0,10,,0.10,0.20,0.15\n"
               "1,20,,0.30,0.40,0.35\n")
    rows = read_epoch_val_metrics(p, after_epoch=0)
    assert [r["epoch"] for r in rows] == [1]
    assert read_epoch_val_metrics(p, after_epoch=1) == []


def test_optional_columns_absent(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_text("epoch,step,val/segm_mAP_50_95\n0,10,0.5\n", encoding="utf-8")
    rows = read_epoch_val_metrics(p)
    assert rows == [{"epoch": 0, "segm_map": 0.5, "segm_map50": None,
                     "f1": None, "train_loss": None}]


def test_garbage_rows_skipped(tmp_path):
    p = _write(tmp_path,
               "zero,10,,abc,,\n"
               "0,10,,nan,,\n"
               "1,20,,0.30,,\n")
    rows = read_epoch_val_metrics(p)
    assert [r["epoch"] for r in rows] == [1]


def test_resolve_num_workers_env(monkeypatch):
    from segcore.instseg.train_rfdetr import resolve_num_workers

    monkeypatch.delenv("SEG_INSTANCE_NUM_WORKERS", raising=False)
    assert resolve_num_workers() == 0
    monkeypatch.setenv("SEG_INSTANCE_NUM_WORKERS", "2")
    assert resolve_num_workers() == 2
    monkeypatch.setenv("SEG_INSTANCE_NUM_WORKERS", "-1")
    assert resolve_num_workers() == 0
    monkeypatch.setenv("SEG_INSTANCE_NUM_WORKERS", "abc")
    assert resolve_num_workers() == 0
