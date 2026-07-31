# Instance Segmentation Integration (v0.9.8) — Design

Status: **Draft v1** (2026-07-22).
Japanese version: [docs/ja/design_instance_segmentation_v098.md](ja/design_instance_segmentation_v098.md)

Evidence base: a screw-counting trial (2026-07-21/22) — RF-DETR-Seg Nano fine-tuned
on synthetic copy-paste data derived from ordinary semantic masks reached
**32/32 exact-count accuracy** on a held-out real test set with **zero manual
instance labels** (val segm mAP 0.844). The one residual error class (duplicate
detection on a single object) is fully suppressed by mask-IoU dedup.

## 1. Goals / Non-goals

**Goals (v0.9.8)**

- Exact-count instance segmentation as a first-class training mode:
  semantic masks → synthetic instance dataset → RF-DETR-Seg fine-tune →
  instance prediction + counting → ONNX export → serving endpoint.
- Zero new annotation burden: instance ground truth is synthesized, never painted.

**Non-goals (v0.9.8)**

- Manual instance-annotation UI (revisit only if synthesis-first fails on a real case).
- auto_select / model-search integration for instance models.
- TensorRT / tflite export extras; CPU-only training.

## 2. Locked decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Synthesis-first**: no instance annotate UI | PoC: 32/32 with zero manual instance labels; existing semantic annotate flow (255-ignore convention) is unchanged |
| D2 | **rfdetr ships in the core lockfile** (not an optional extra) | Owner decision. Consequences accepted: larger install, larger pip-audit/license surface (§4.5) |
| D3 | **Scope includes ONNX export + serving_api endpoint** | Owner decision. Carries the R1 export-feasibility risk (§8) |
| D4 | **Synthesis config lives in the Training form** (no separate tab) | Appears only when `training_mode="instance"`; keeps UI flat |
| D5 | **RF-DETR *Seg* family only** (Small default; Medium / Large selectable. Nano was offered initially but retired — see D5a) | Seg checkpoints are Apache-2.0 at every size. Detection XL/2XL and the rfdetr "plus" extra are PML-1.0 → prohibited; ban patterns added (§4.5) |
| D5a | **Nano retired for new training (2026-07-23)** | On the reference counting workload Nano reached 0.92 segm mAP50 vs 0.94 (Small) / 0.99 (Medium); not enough for exact counts. The class mapping stays so existing Nano checkpoints remain loadable for prediction |
| D6 | **Count = threshold (val-calibrated) + mask-IoU dedup (0.7)** | Dedup kills the DETR duplicate-mask artifact; adjacent-but-touching objects have near-zero mask IoU, so no false merges (PoC-verified) |

## 3. User flow

1. Annotate semantic masks exactly as today (`PUT …/datasets/annotate/masks/{item}.png`, 255 = unpainted).
2. Training form → mode selector gains **"Instance (counting)"**. Selecting it reveals the synthesis section (counts, objects/image, stack-pair probability, seed, area-band override) plus a **Preview** button that renders 2–3 composed samples inline.
3. Start → the run pipeline extracts cutouts, composes the COCO dataset into the run dir, fine-tunes RF-DETR-Seg, writes `metrics.json`.
4. Results tab → instance overlay (numbered badges, Okabe–Ito palette) + per-image count.
5. Export ONNX → serve counts via serving_api.

## 4. Architecture

### 4.1 segcore — new package `segcore/instseg/`

- `compose.py` — copy-paste composer, ported from the PoC generator:
  - cutout extraction from semantic masks (connected components in the
    single-object area band; band is **auto-estimated** from the blob-area
    histogram with a manual override field — the PoC used a fixed 3200–8500 px band);
  - background plates via inpainting every Nth image;
  - painter's-algorithm compositing → exact per-instance visible masks;
  - **coaxial stack pairs**: PCA principal-axis estimation per cutout, both
    cutouts rotated to a shared axis (random 180° polarity), placed at
    contact-to-slight-overlap along the axis (factor 0.80–0.97) with small
    perpendicular jitter. This is the pattern that fixed the PoC's only miss;
  - real full-GT images (every blob in the single-object band) mixed into
    train/val as in the PoC;
  - deterministic under a seed; COCO writer (roboflow-style layout rfdetr expects).
- `count.py` — confidence filter + greedy mask-IoU dedup (keep higher conf,
  suppress IoU > 0.7) + count. Pure numpy; unit-tested.
- `train_rfdetr.py` — subprocess entry (Windows: `if __name__ == "__main__"`
  spawn guard, `num_workers=0`), maps run config → `RFDETRSegNano/Small.train()`,
  translates rfdetr metrics into our `metrics.json` shape.

### 4.2 trainer_api

- `schemas.py:132` — `training_mode` gains `"instance"`; new optional
  `instance_synthesis` config block (`n_train=500`, `n_val=80`,
  `objects_min=4`, `objects_max=8`, `stack_pair_prob=0.55`, `seed`,
  `area_band=[auto]`). Validation gate mirrors the existing mode gate tests
  (`tests/test_train_mode_gate.py`).
- `training_launcher.py` — instance branch: skips semantic prepare, runs
  compose → rfdetr subprocess; progress streams through the existing run-log
  channel; `train_config.json` written as today.
- Prediction routes (new, run-scoped): batch predict producing per item
  `instances.json` (`{instances: [{id, conf, bbox, rle, area}], count, threshold, dedup_iou}`),
  an `overlay.png`, and a legacy semantic-style composite mask PNG so existing
  viewers keep working. RLE via pycocotools (arrives with `rfdetr[train]`).
- Preview route for the training form: compose 2–3 samples on demand (fast, CPU).
- Export: instance ONNX export route following `export_routes.py` /
  `training_exports.py` conventions (fp32/fp16 first; int8 out of scope).

### 4.3 serving_api

- New instance inference capability (ORT session over the exported ONNX):
  postprocess = threshold + dedup + count, response = instances (RLE) + count.
- serving stays minimal: RLE encode implemented in ~30 lines of numpy —
  **pycocotools is NOT added to serving**.

### 4.4 trainer_ui

- Training form (`training/`): mode selector + conditional synthesis section
  (flat fields, no new nesting) + preview strip.
- Results (`results/`): instance overlay renderer (numbered badges; Okabe–Ito
  palette — the PoC viz palette is already Okabe–Ito, no purple, colorblind-safe),
  count chip; `MeasurementPanel` gains an instance mode that reads
  `instances.json` instead of running union-find on the semantic mask.
- Run-type awareness: semantic-only panels (confidence/error heatmaps, pixel
  histogram) are hidden for instance runs rather than left to error.
- No new canvas interactions: overlays are view-only; zoom/pan/keyboard
  conventions are untouched.

### 4.5 Dependencies & licensing

- `apps/trainer_api/requirements.in`: add `rfdetr==1.8.*` with the `[train]`
  extra (and the export deps once R1 is validated). Compatibility verified
  2026-07-22: `torch>=2.2` (core 2.13.*, cu128 2.11.* both fine),
  `transformers>=5.1,<6` (we pin 5.5.4), `pydantic>=2,<3` (we pin 2.12.*).
  `requirements-cu128.in` needs the same addition.
- New transitive closure licenses (PyPI, checked 2026-07-22): supervision MIT,
  pytorch-lightning Apache-2.0, albumentations MIT, peft Apache-2.0,
  torchmetrics Apache-2.0, pycocotools BSD-2-Clause, roboflow Apache-2.0,
  rf100vl MIT, onnxsim MIT/Apache-2.0/BSD-2, polygraphy Apache-2.0,
  faster-coco-eval **verify at repo (classifiers missing; GitHub says Apache-2.0)**.
  The lockfile bump commit MUST carry the full `LICENSE: <pkg> <ver> <SPDX> confirmed at <URL>`
  trail; regenerate all lockfiles with uv 0.11.11 per the `.in` headers;
  update `THIRD_PARTY_NOTICES.md`.
- **Ban patterns**: add the rfdetr "plus" extra and the detection XL/2XL identifiers to
  `scripts/nc-vendor-patterns.txt` (dev-only file) so pre-commit + public CI
  block reintroduction. The public repo's `NC_VENDOR_PATTERNS` variable must be
  refreshed at the v0.9.8 release (release checklist §1).
- Windows note (verified 2026-07-22): `onnxsim` must be pinned `==0.4.36` —
  it ships a cp311 win_amd64 wheel, while newer sdists fail to install on
  Windows long-path limits.
- Housekeeping found during recon: `CONTRIBUTING.md` §Dependencies still
  documents torch 2.6.0 — stale vs `requirements.in` (2.13.*); fix alongside.

### 4.6 Windows specifics

- rfdetr training runs with `num_workers=0` + spawn guard (also avoids the
  pagefile blow-ups documented for multi-worker sweeps).
- VRAM (measured 2026-07-22, RTX 3090, Nano, 1 epoch, effective batch 16):

  | batch | grad-accum | peak allocated | peak reserved | required GPU |
  |-------|-----------|----------------|---------------|--------------|
  | 8 | 2 | 5.75 GiB | 6.5 GiB | ≥ 8 GiB |
  | 4 | 4 | 3.29 GiB | 3.6 GiB | ≥ 5.5 GiB |
  | 2 | 8 | 2.01 GiB | 2.2 GiB | ≥ 3.5 GiB |

  Auto-reduction (`instance_training._fit_batch_to_vram`): the batch is halved
  (grad-accum doubled, effective batch preserved) until the required tier fits
  `total_memory`; floor is batch 2. Below 3.5 GiB instance training is
  unsupported.

## 5. Data contracts

- Run dir: `train_config.json`, `instseg_dataset/{train,valid}/…` (COCO),
  rfdetr checkpoints, `metrics.json`, `instances/{item}.json` + overlays.
- `metrics.json` (instance runs): `segm_mAP_50_95_val`, `segm_AP50_val`,
  `AR_val`, `count_exact_val` (on the real-image val subset), `best_epoch`,
  `epochs_effective`, `dataset_stats {n_synth, n_real, n_cutouts, stack_pair_ratio}`.
  `MetricsSection.tsx` renders the subset conditionally.

## 6. Testing

- Unit (CI, CPU): composer determinism under seed; stack-pair coaxiality
  (angle delta + gap assertions); dedup (duplicate suppressed, adjacent kept);
  RLE round-trip.
- API: instance mode gate, synthesis validation, preview endpoint, predict
  artifact shapes.
- e2e: training-form spec (fast, render+validation); full instance train smoke
  gated `@heavy` under the existing skip-budget mechanism (needs GPU, dev box only).
- No golden-run pinning for rfdetr training (nondeterministic across envs);
  count-level assertions on a fixed tiny synthetic set instead.

## 7. Milestones

| M | Content | Est. |
|---|---------|------|
| M1 | segcore `instseg/` port (compose/count) + unit tests + **ONNX export spike (R1)** | 1–2 d |
| M2 | trainer_api instance mode end-to-end on the dev box + VRAM measurement | 2–3 d |
| M3 | Results UI (overlay/count) + predict artifacts + preview | 2 d |
| M4 | ONNX export + serving endpoint | 2 d |
| M5 | Lockfiles + license trail + docs/handbook + e2e specs | 1–2 d |

## 8. Risks

- **R1 — rfdetr Seg ONNX export: RESOLVED (2026-07-22 M1 spike).**
  RF-DETR-Seg Nano (the trained PoC checkpoint) exported to ONNX (opset 17)
  and ran under onnxruntime CPU. Verified contract: input `1×3×312×312`
  (export default; `shape=` overridable — dimensions must fit the patch-12
  backbone), outputs `dets 1×100×4` (normalized boxes), `labels 1×100×2`
  (logits), `masks 1×100×78×78` (mask logits at reduced resolution).
  Serving postprocess = sigmoid → confidence threshold → mask resize →
  dedup → count, exactly as §4.3 assumed.
  **M4 addendum (2026-07-22):** the object confidence is
  `sigmoid(labels[:, 0])` — COCO category 1 maps to internal class index 0.
  The full numpy chain (stretch resize → /255 → ImageNet normalize → ORT →
  conf ≥ thr → mask sigmoid → bilinear resize → >0.5 → area ≥ 16 → greedy
  mask-IoU dedup) reproduces the SDK's `model.predict` counts exactly:
  32/32 GT-exact and 32/32 SDK-match on the PoC test set. The serving
  `/count` endpoint and the exported `instance_inference.json` contract
  encode this chain.
- **R2 — roboflow SDK behavior.** Verify no import-time network/telemetry and
  clean offline install; isolate behind lazy import if needed.
- **R3 — core-lockfile growth** (D2, accepted): install size + audit surface.
- **R4 — small-GPU VRAM**: mitigated by M2 measurement + auto grad-accum.
- **R5 — synthesis domain gap** beyond separable rigid parts: mitigated by
  area-band auto-estimation, the preview strip, and mixing real full-GT
  images; documented as a known limitation in the handbook.
