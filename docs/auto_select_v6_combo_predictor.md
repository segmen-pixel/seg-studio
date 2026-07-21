# auto_select v6 — Combo Predictor (XGBoost Ensemble, OSS-clean)

> **Status**: v6.0.  Drop-in replacement for the v3 LightGBM dual-model
> combo predictor in `segcore.auto_select`.
> 2026-05-21: warmup-calibrated **training-time predictor** bundled
> alongside the F1 predictor — see §6.
> 2026-05-22: **VRAM predictor / OOM-avoidance** (WDDM-aware) added —
> see §7.
> 2026-07-07: **bundle refreshed** — feature cache repaired (26/26
> scalar columns, previously 14 were silently zero-filled) and the
> training table extended to wave1-6 (12,362 rows).  See §3 and
> [`auto-config-rationale.md`](auto-config-rationale.md) §7.
>
> **Sibling note**: [`technical_note_auto_select.md`](ja/technical_note_auto_select.md)
> covers the *transfer-learning* side of auto_select (donor checkpoint
> picking, profile library).  This document covers the *combo
> recommendation* side — picking the best `(arch, base_channels,
> patch_size, distill_on, fg_patch_prob, dice_weight, loss_type,
> class_weight_strength)` 8-axis recipe for a new project.

---

## 1. Why v6 exists

The v3 combo predictor used **LightGBM** (MIT) as its boosted-tree
backend.  MIT is OSS-compatible with seg-studio's Apache-2.0 licence,
but ships every seg-studio install with a dependency under a different
licence family.  v6 narrows the runtime stack to **Apache-2.0 +
BSD-3-Clause only**:

| Layer | Package | Licence | Role |
|---|---|---|---|
| Boosted trees | `xgboost` | Apache-2.0 | regressor + ranker |
| Preprocessing / PCA / metrics | `scikit-learn` | BSD-3-Clause | PCA, z-score |
| Bayesian HP search *(research only)* | `scikit-optimize` | BSD-3-Clause | not shipped |
| Numerical stack | `numpy` / `scipy` | BSD-3-Clause | — |
| Embedding teacher | DINOv2 weights | Apache-2.0 | inference-time image embedding |

MIT-licensed combo-predictor alternatives — LightGBM, Optuna, SHAP,
interpret EBM — are intentionally **not** in the shipped lockfiles.
The CI `python-imports` job grep-rejects them (`ci.yml`, “Reject
MIT-licensed combo predictor deps” step).

All training and inference run on CPU.  No GPU is required to install
or use the bundled model.

---

## 2. Architecture

```
                       ┌─────────────────────────────────────┐
   new project ───►    │  feature_extractor.extract_runtime_ │
   (images,            │  features(images_dir, masks_dir)    │
    masks)             └────────────────┬────────────────────┘
                                        │
                       26 scalar features  +  raw DINOv2 768-d
                                        │
                                        ▼
                       ┌─────────────────────────────────────┐
                       │  ComboPredictor.rank(scalar, dino)  │
                       │  (combo_predictor.py)               │
                       │                                     │
                       │   ┌─────────────┐  ┌─────────────┐  │
                       │   │ DINOv2 PCA  │  │ z-score on  │  │
                       │   │ (32 dims)   │  │ 26 scalars  │  │
                       │   └──────┬──────┘  └──────┬──────┘  │
                       │          └──────┬──────────┘        │
                       │                 ▼                   │
                       │      design row per candidate combo │
                       │                 │                   │
                       │   ┌─────────────┴─────────────┐     │
                       │   ▼                           ▼     │
                       │ regressor.json          ranker.json │
                       │ reg:squarederror        rank:ndcg   │
                       │   │                           │     │
                       │   ▼ pred_f1                   ▼ rank│
                       │  min-max norm          min-max norm │
                       │           ╲              ╱          │
                       │            ╲            ╱           │
                       │     ensemble = 0.8·reg + 0.2·rank   │
                       └──────────────────┬──────────────────┘
                                          ▼
                       sorted list of combos with
                       (combo, pred_f1, pred_std, ci_low, ci_high)
```

### 2.1 Feature surface

77 columns total (`feature_columns` in `metadata.json`):

| Block | Count | Source |
|---|---:|---|
| Project-level scalars (BG variance, fg ratio, edge density, frequency bands, geometry stats, scale logs) | 26 | runtime feature extractor |
| DINOv2 PCA components | 32 | per-image global mean DINOv2 ViT-B/14 → PCA fit during packaging |
| Arch one-hot | 3 | `simpleunet` / `stdc` / `deeplabv3plus` |
| Recipe knobs (raw + `log2(base_channels)`, raw + `log2(patch_size)`, distill flag, fg_patch_prob, dice_weight, class_weight_strength) | 9 | combo |
| Loss one-hot | 3 | `focal` / `lovasz` / `ce` |
| Hand-tuned cross-interactions (distill×bc, distill×ps, fp×fg_ratio, focal×imbalance, lovasz×bc) | 5 | combo × scalar |

The scalar block is z-scored using the mean / std persisted in
`metadata.json`.  XGBoost handles the remaining columns raw and treats
missing scalars as `NaN` (its native missing-value default).

### 2.2 Model heads

* **Regressor** — `xgboost.train(..., objective="reg:squarederror")`.
  Predicts F1 directly in `[0, 1]`.  Used for ensembling and to attach
  a calibrated expected-F1 value (`pred_f1`) to each recommendation.
* **Ranker** — `xgboost.train(..., objective="rank:ndcg")` LambdaRank.
  XGBoost requires *integer relevance grades*, not continuous F1, so
  raw F1 is quantile-binned per project to `[0, num_grades-1]` (default
  10 grades) by `f1_to_relevance` in `train_xgb_ranker.py`.

### 2.3 Ensembling

Per-project min-max normalisation collapses each model's outputs into
`[0, 1]`, then a linear mix:

```
ensemble_score  =  w_reg · reg_norm  +  (1 - w_reg) · rank_norm
```

`w_reg = 0.8` is the LOPO-best choice in the 2026-05-21 weight sweep at
PCA dim = 32 / per-fold refit.  Stored in
`metadata.json::ensemble_weight_reg`.

A `pred_std` field is exposed alongside `pred_f1` for downstream
confidence calculation.  In v3 it was the standard deviation of 5
calibrator seeds; v6 ships a single regressor, so `pred_std` is
redefined as the *regressor / ranker disagreement* on the candidate
combo, scaled into F1 units by the regressor's empirical span.  Larger
`pred_std` means the two models disagree more about that combo's
ordering.  `ci_low / ci_high` is the conventional `pred_f1 ± 1.96 ×
pred_std` band — interpreted as a coarse confidence interval rather
than a strict statistical CI.

---

## 3. Training the production model

Source of truth: `scripts/research/combo_predictor_v6/package.py`.

```bash
# from seg-studio-dev/ (uses .venv-windows)
python scripts/research/combo_predictor_v6/package.py \
    --out  seg-studio/packages/segcore/segcore/auto_select/models/best_model_v6/ \
    --dino-dims 32 \
    --w-reg 0.8
```

The script:

1. **Loads** per-project handcrafted scalars + DINOv2 embeddings from
   the dev cache at `research_artifacts/combo_predictor/features_v2_full.json`
   and `features_v2_dino.npz`.  (rev. 2026-07-07: the earlier
   `features_v2.json` cache populated only 12 of the 26 scalar columns
   the model schema declares — `fg_ratio`, `num_train`, `mean_width`,
   `class_imbalance_ratio` etc. were silently zero-filled at train
   *and* inference time.  `features_v2_full.json` fills all 26 from
   each project's `prepared/dataset_stats.json`, falling back to
   `compute_basic_stats_fallback` where that file is absent.)
2. **Fits** a fresh PCA on *all* available project DINOv2 vectors at
   `n_components = 32`.  PCA is fit on the full corpus because the
   production artefact is for end-user inference; there is no LOPO
   holdout to leak into.
3. **Loads** the wave1–6 unified table
   (`research_artifacts/unified/all_waves_w1-6.csv`, 12,362 rows with
   `best_F1 > 0`, 37 projects) and builds the 77-column design matrix.
4. **Fits the regressor and ranker** on the full design matrix.
   Internal validation uses a by-project hold-out (~10% of training
   projects) purely to pick `best_iteration` via XGBoost early
   stopping; the shipped boosters are then **refit on the full dataset
   for `best_iteration + 1` rounds** so they have seen every project.
5. **Writes** the four artefacts to `best_model_v6/`:
   - `regressor.json` (~1.1 MB)
   - `ranker.json` (~10 KB)
   - `dino_pca.pkl` (~102 KB)
   - `metadata.json` (~12 KB; schema, feature column names,
     z-score mean / std, combo catalogue, ensemble weight, dependency
     pins, licence stack)

The legacy v3 bundle (`best_model_dual/`, ~5.6 MB across 5 calibrators
+ 1 ranker + PCA) is **removed** from the tree as part of the v6 swap.

### 3.1 LOPO benchmark (research-only, not for shipping)

`scripts/research/combo_predictor_v6/ensemble.py` runs the leave-one-
project-out evaluation that produced the headline numbers below.  PCA
is refit on the training projects of every fold to avoid an
unsupervised leak — global PCA inflates Spearman by a few percent in
practice but cannot be shipped honestly.

Numbers measured against the **repaired 26/26 feature cache + wave1-6
table** (2026-07-07 refresh, the shipped bundle):

| Configuration | top-1 hit | top-3 hit | top-5 hit | mean F1 gap | median F1 gap | Spearman | within 0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ranker (dim 4) | 0% | 0% | 2.7% | 0.104 | 0.080 | 0.260 | 37.8% |
| **Regressor (dim 4)** | **2.7%** | **2.7%** | **8.1%** | **0.082** | **0.045** | **0.471** | **51.4%** |
| Ensemble (dim 32, auto-tuned `w_reg = 0.60`) | 0% | 0% | 5.4% | 0.089 | 0.064 | 0.448 | 48.6% |

The plain regressor is currently the strongest single configuration —
the ranker gains little from the repaired features, and mixing it in
dilutes the regressor.  The shipped bundle keeps `w_reg = 0.8` in
`metadata.json` (mostly-regressor), with a full re-tune of the
ensemble weight noted as follow-up work.  Research-only aside: adding
a rule filter (patch=256 / focal / distill=off) in front of the
regressor lifts top-3 to 8.1% and top-5 to 10.8%, but collapses the
ranker (Spearman −0.14) — see `auto-config-rationale.md` §7.

Pre-repair baseline for comparison (12/26 features zero-filled,
wave1-4, 9,084 rows):

| Configuration | top-1 hit | top-3 hit | top-5 hit | mean F1 gap | median F1 gap | Spearman | within 0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ranker baseline (LOPO) | 0% | 2.7% | 2.7% | 0.099 | 0.063 | 0.238 | 45.9% |
| Regressor (PCA dim = 32, per-fold) | 2.7% | 2.7% | 5.4% | 0.075 | 0.063 | 0.472 | 48.6% |
| Ensemble (`w_reg = 0.8`, dim = 32) | 2.7% | 2.7% | 8.1% | 0.077 | 0.040 | 0.485 | 56.8% |

Top-k hit rates are inherently low because every project carries
200–326 candidate combos in the 8-axis space — a random baseline scores
~0.4% for top-1 and ~1.5% for top-3, and per-project best combos are
almost unique (33 distinct best combos across 37 projects).  The
recommended-vs-best F1 gap and the `within 0.05` rate are more useful
as ROI indicators.

---

## 4. Inference at runtime (segcore.auto_select integration)

### 4.1 Loading the bundle

```python
from segcore.auto_select.combo_predictor import get_default_predictor

predictor = get_default_predictor()   # cached, lazy load
# Falls back to None if `best_model_v6/metadata.json` is missing or
# xgboost is not importable; callers should branch on that.
```

`config_selector._recommend_via_ml` already handles the `None` path:
when the v6 bundle is unavailable, it falls back to the legacy
similarity-weighted z-score portfolio (see
`ja/technical_note_auto_select.md` §3).

### 4.2 Ranking candidate combos

```python
ranked = predictor.rank(
    scalar_features=runtime_scalar_features_from_extractor,
    dino_vec_768=runtime_dinov2_global_mean,
    candidate_combos=None,   # default: use bundled all_combos
)
# Each entry:
#   {combo, arch, base_channels, patch_size, distill_on,
#    rank_score, pred_f1, pred_std, ci_low, ci_high}
# Sorted descending by rank_score.
```

`rank_score` is the per-project min-max ensemble score in `[0, 1]`.
`pred_f1` is the regressor's raw F1 estimate — kept on its original
scale so callers can attach a calibrated expected-F1 to the
recommendation chip in the UI.

`candidate_combos` can be passed explicitly when the caller wants to
score a subset of recipes (e.g. excluding `distill_on=True` recipes
when a small dataset cannot afford the teacher pass).

### 4.3 Confidence band in `config_selector`

`_recommend_via_ml` retunes the confidence thresholds for the v6
`[0, 1]` ensemble scale:

| level | condition |
|---|---|
| `high` | `pred_std < 0.01 and top1-top2 gap > 0.05` |
| `medium` | `pred_std < 0.03 and top1-top2 gap > 0.02` |
| `low` | `pred_std < 0.06` |
| `none` | otherwise |

The thresholds are deliberately conservative on a brand-new project:
the v6 bundle was trained on 37 projects from one customer base, so
out-of-distribution datasets should not display a `high` chip until
enough donors land in the project library to corroborate the pick.

---

## 5. Operational notes

### 5.1 Dependencies

`apps/trainer_api/requirements.in` (and the cu128 sibling) carry
`xgboost>=2.0.0,<4` as the only combo-predictor runtime dep.  The
lockfiles are regenerated with `uv pip compile`; the lockfile-drift CI
rejects `.in` changes without the matching `.txt` regeneration.

```
# trainer_api/requirements.in
# ── auto_select combo predictor (XGBoost ensemble, Apache-2.0) ─
xgboost>=2.0.0,<4
```

### 5.2 Repackaging cadence

The bundled model is regenerated each time the wave1–4 unified table
grows by a non-trivial amount (e.g. a new ablation wave) or a project
is added.  Re-run `package.py` and commit the new `best_model_v6/`
artefacts together with an updated `CHANGELOG.md` entry.  No
`metadata.json` schema bump is required as long as the feature surface
is unchanged; if `dino_dims`, `archs`, `losses`, or
`scalar_feature_names` change, bump the `schema_version` field and
update `combo_predictor.py` accordingly.

### 5.3 Reproducibility

`package.py` uses `random_state=42` everywhere (PCA + the regressor's
internal val split is deterministic in project order, taken as the
last 10% of `sorted(proj_feats)`).  Re-running on the same input
caches should produce byte-identical artefacts modulo any
non-determinism in XGBoost's tree splits at extreme float boundaries.

### 5.4 CI guarantees

* `ci.yml::python-imports → "Verify auto_select v6 combo predictor
  loads"` end-to-end loads the shipped bundle, calls `rank()`, and
  asserts the output is non-empty and monotonically descending.
* `ci.yml::python-imports → "Reject MIT-licensed combo predictor
  deps"` blocks reintroduction of LightGBM / Optuna / SHAP / interpret
  into the shipped lockfiles.
* `ci.yml::pytest → tests/test_auto_select.py::TestComboPredictorV6`
  carries two smoke tests covering bundle load + rank shape /
  monotonicity.

---

## 6. Time predictor (warmup calibration, v6)

The combo predictor recommends *which* recipe to run; the time predictor
estimates *how long* each candidate will take on the current project's
hardware.  Without a calibration anchor the physical log-linear model
alone is R²(log) ≈ -0.005 — useful only for *ordering* combos by relative
runtime.  With one anchor run the same model jumps to R²(log) ≈ **+0.958
/ MAPE ≈ 14 %** on the wave1–4 LOPO evaluation (see
`research_artifacts/combo_predictor_v4/eval/time_v4_warmup_results.json`).

### 6.1 Bundle layout

`best_model_v6/phys_time.json` (≤ 1 KB) carries:

```json
{
  "status": "ok",
  "version": "v6.0-warmup",
  "feature_names": ["log_num_train", "log_num_total", "log_img_pixels",
                    "log_bc", "log_ps", "arch_simpleunet", "arch_stdc",
                    "arch_deeplabv3plus", "distill_on", "fg_patch_prob"],
  "coefs":  [10 floats],
  "intercept": float,
  "anchor_combo": "simpleunet_bc64_p256_distillOff_fp0.5_dw2.0_lovasz_cws0.8",
  "lopo_metrics_v4": {"r2_log": 0.958, "mape_pct": 13.9, "mae_sec": 158.0,
                       "anchor_label": "full_simpleunet_bc64"}
}
```

The 10-feature log-linear model is fit on the wave4 timings snapshot
(`research_artifacts/ablation_wave4/_snapshot_2026-05-13/wave4_timings.csv`,
1,092 ok-status rows across 37 projects).  In practice only five
coefficients carry real signal — `log_bc` (≈ +0.20), `distill_on`
(≈ +0.13), and the three `arch_*` one-hots; project-size coefficients
(`log_num_train`, `log_num_total`, `log_img_pixels`, `log_ps`) regress
to ~0 because per-project hardware variance dominates that signal.
Warmup calibration is what closes that gap.

### 6.2 Inference math

`segcore.auto_select.time_predictor.TimePredictor.predict_seconds`
implements the v4 recipe in pure numpy (no sklearn at scoring time):

```
log_pred  = X @ coefs + intercept                          # physical
scale_log = log(anchor_elapsed_sec) - log_pred[anchor]     # per-project
return    np.exp(log_pred + scale_log)                     # calibrated
```

When `anchor_elapsed_sec` is `None` the scale term is omitted and the
caller gets the physical-only prediction — fine for ranking by speed,
do **not** show it as an absolute ETA.

### 6.3 Integration with `ComboPredictor.rank()`

`rank()` accepts an optional `anchor_elapsed_sec` argument and appends
two fields to every returned dict:

| field | type | meaning |
|---|---|---|
| `pred_elapsed_sec` | `float \| None` | seconds; `None` only when `phys_time.json` is missing |
| `pred_elapsed_min` | `float \| None` | same value in minutes (UI convenience) |

`ComboPredictor.anchor_combo` exposes the recommended anchor combo
string — UI flows can promote this combo to the front of the queue
and, after it finishes, re-call `rank()` with its actual elapsed_sec
to display calibrated ETAs for the rest of the queue.

### 6.4 `config_selector.recommend_combo()`

The high-level wrapper grew matching optional support:

```python
rec = recommend_combo(
    query_features, library,
    images_dir=…, masks_dir=…,
    anchor_elapsed_sec=660.0,   # optional, seconds (11 min in this example)
)
# rec.pred_elapsed_min          # → calibrated ETA for the top combo
# rec.time_anchor_combo         # → which combo to run for warmup
# rec.time_calibrated           # → True iff anchor_elapsed_sec was used
# rec.reasoning                 # → "ML ensemble v6: pred_f1=0.86 … ~11.3 min (calibrated)"
```

Without `anchor_elapsed_sec` callers still get a physical-only estimate
in the same fields, tagged `(physical-only)` in `reasoning`.

### 6.5 End-to-end UI flow (Phase 7)

1. **Initial recommendation.**  The user opens the Training tab and
   clicks **Model Search**.  The UI calls
   `POST /api/v1/projects/{id}/train/model-search` without
   `anchor_elapsed_sec` — the response carries `pred_elapsed_min` from
   the *physical-only* prediction, and the result panel renders e.g.

       ⏱ estimated training time: ~13.9 min (physical-only)
       → for an accurate ETA, run the warmup anchor combo first:
           simpleunet_bc64_p256_distillOff_fp0.5_dw2.0_lovasz_cws0.8
         then click "Recalibrate ETAs" with its actual elapsed seconds.

2. **Warmup run.**  The user trains the named anchor combo
   (typically ~11 min) and notes the actual elapsed seconds.

3. **Recalibrate.**  The user clicks the **"Recalibrate ETAs"** button
   in the model-search panel; a `prompt()` collects the measured
   seconds, the UI re-issues model-search with
   `?anchor_elapsed_sec=NNN`, and the result panel switches to the
   calibrated band:

       ⏱ estimated training time: ~11.3 min (calibrated)

4. **Subsequent recommendations.**  The chosen anchor seconds persist
   in component state for the session, so any later model-search
   request from the same screen automatically reuses them until the
   user re-enters a value or reloads the page.

Auto-detection of the anchor's elapsed seconds from completed runs is
**not yet implemented** — `TrainingRun` records do not currently carry
`elapsed_sec` and the project-level run schema does not expose combo
identifiers we can match against the bundled `anchor_combo`.  Adding
that round-trip is the natural next step (record elapsed_sec when a
run finishes → match the combo identifier → auto-populate
`anchorElapsedSec` in the UI).

### 6.6 CI guarantees

`ci.yml::python-imports` step "Verify auto_select v6 combo predictor
loads" now asserts that `phys_time.json` is bundled, that
`pred_elapsed_sec` is populated for every combo, and that warmup
calibration round-trips the anchor combo's elapsed_sec exactly
(`|rt - 660| < 1e-3`).  `tests/test_auto_select.py::TestTimePredictorV6`
covers the same surface plus the plausible-band check
(`30 < pred_elapsed_sec < 6 × 3600`).

---

## 7. VRAM predictor (OOM avoidance, v6)

The combo predictor recommends *which* recipe and the time predictor
estimates *how long*; the VRAM predictor estimates *whether it fits in
GPU memory* and issues a **WDDM-aware OOM verdict**.  Implemented in
`segcore/auto_select/vram_predictor.py`, research code under
`scripts/research/vram_predictor_v6/`.

### 7.1 Why it exists

The training pipeline auto-fits a batch size with a static VRAM
dry-run.  On Windows **WDDM** GPUs the OS reserves/recycles VRAM behind
PyTorch's back, cuDNN warms up to heavier algorithms, and the desktop
compositor reclaims surfaces — so the dry-run under-estimates and the
run OOMs mid-epoch.  The wave5 cross-device probe measured this:

- 9,669 VRAM probes on RTX 5090 / 3090 / 3080 Ti × 2.
- **542 OOM events — 100 % on the 12 GB 3080 Ti running Linux with no
  headroom.**  The WDDM 5090/3090 (2 GB profiler headroom) saw zero.
- OOM rate by loss: `lovasz 9.9 % > focal 6.3 % > ce 0 %`; by arch:
  `simpleunet 9 % / stdc 7.8 % / deeplabv3plus 0 %`.

### 7.2 Two heads

| Head | Model | Target | LOPO result |
|---|---|---|---|
| VRAM regressor | XGBoost `reg:squarederror` (log space) | `vram_peak_mb` | MAPE 3.6 %, R²(log) 0.99 |
| OOM classifier | XGBoost `binary:logistic` | `status == error` | AUC 0.99, recall 0.94 |

Feature surface (20 cols): combo 8-axis + `driver_is_wddm` +
`gpu_total_mb` + `num_train` (raw + log1p) + 2 physical interactions
(`bc·ps²`, `is_lovasz·bc`).  No DINOv2 — image *content* does not move
VRAM, only geometry/data-volume/recipe do.

**Why `batch_size` is not a feature.**  wave5 measured each cell at its
*auto-fit* batch size — the trainer's VRAM dry-run picks the largest
batch that fits.  Heavier settings get a *smaller* auto-fit batch yet
still peak *higher*, so `batch_size` and `vram_peak` are *negatively*
correlated (corr = -0.24).  An early v6 build fed `batch_size` in and
XGBoost learned the inverted causality — `predict_vram_mb` then claimed
a light combo needed 17 GB at batch=1.  The predictor therefore
estimates the peak of an *auto-fit* run directly; `gpu_total_mb` (which
determines the auto-fit batch) is the hardware-budget proxy.  It
answers **"will training this combo on this GPU OOM?"**, not "what
batch fits?".

### 7.3 WDDM-aware safety margin

The raw regressor predicts the *expected* peak and under-predicts on
~67 % of rows — shipping it raw would OOM constantly.  `verdict()`
applies three layers:

```
vram_safe = pred_vram * safety_multiplier          # exp(underpred_log_p95) ≈ 1.19
budget    = gpu_total_mb * usable_fraction - headroom_mb
verdict   = "oom_risk" if vram_safe > budget        # layer 1: budget
            or oom_prob >= 0.5                      # layer 2: classifier veto
            else "ok"
```

| driver | usable_fraction | headroom_mb | rationale |
|---|---:|---:|---|
| **WDDM** | 0.92 | 2048 | cuDNN warm-up + compositor reclaim |
| **Linux** | 0.94 | 512 | native driver, small cushion |

The multiplier is the wave5 LOPO 95th-percentile under-prediction band,
so a slightly-low estimate still lands inside the budget.  All constants
live in `vram_metadata.json::safety` — the policy is data-driven, not
hard-coded.

### 7.4 API

```python
from segcore.auto_select import get_default_vram_predictor

vp = get_default_vram_predictor()           # None if bundle/xgboost missing
v = vp.verdict("stdc_bc128_p256_distillOff_fp0.5_dw2.0_lovasz_cws0.8",
               gpu_total_mb=24576, is_wddm=True, num_train=300)
# v = {verdict: "ok"|"oom_risk", reason, pred_vram_mb, vram_safe_mb,
#      budget_mb, headroom_mb, utilization, oom_prob, driver}
```

There is no `batch_size` parameter — the verdict is for an auto-fit
run.  On an `oom_risk` verdict the operator should drop `base_channels`
or switch `lovasz` → `ce`/`focal` (the wave5-measured VRAM-cheapest
losses).

### 7.5 Bundle

`best_model_v6/` gains three artefacts (≈ 3.5 MB total):
`vram_regressor.json`, `oom_classifier.json`, `vram_metadata.json`.
Fit by `scripts/research/vram_predictor_v6/package_vram.py`.

### 7.6 seg-studio integration

The predictor is wired through the same surfaces as the combo + time
predictors:

- **`segcore.auto_select`** re-exports `VramPredictor` and
  `get_default_vram_predictor` as public API.
- **`training_runner.py`** — after the auto-config phase picks a combo,
  it reads the resolved GPU's total VRAM
  (`torch.cuda.get_device_properties`), derives the driver mode
  (`os.name == "nt"` → WDDM), and logs the verdict:
  `Auto-config [VRAM]: … predicted peak ~N MB (safe ~M MB / budget B
  MB) -> ok|oom_risk`.  On `oom_risk` it prints an explicit warning
  naming the cheaper alternatives.
- **`POST /train/model-search`** — the `config_recommendation` payload
  gains a `vram` object: `{gpu_total_mb, driver, pred_vram_mb,
  budget_mb, verdict, oom_risk}` for the top combo on the currently
  configured GPU.
- **trainer_ui** — the model-search result panel renders one extra line:
  `✓ VRAM: fits the current GPU (… MB, WDDM) — predicted peak ~N MB,
  safe budget B MB` or, on risk, `⚠ VRAM: this combo may OOM …`.

All GPU-property reads happen on surfaces that already touch CUDA
(training subprocess, DINOv2 feature extraction), so no new CUDA
context is created on the request path.

### 7.7 Verified against wave5 ground truth

The batch-free predictor reproduces the wave5 measurements closely.
For `Bolt` / `simpleunet_bc64_p256_distillOff_fp0.3_dw1.0_ce_cws0.0`:

| GPU | wave5 actual `vram_peak` | v6 prediction |
|---|---:|---:|
| RTX 5090 (32 GB) | 23,210 MB | 23,145 MB |
| RTX 3080 Ti (12 GB, Linux) | 7,999 MB | 7,961 MB |

The verdict gives an interpolated `oom_risk` on a 24 GB 3090 WDDM
(predicted peak ~18.9 GB → safe ~22.4 GB > 20.6 GB budget), which is
the expected call for a project this heavy.  Note there is no
"recommend a smaller batch" feature — wave5 never varied batch as a
controlled variable, so the data cannot support it; the verdict is the
shipped surface.

---

## 8. Known sharp edges

* **XGBoost rank:ndcg requires integer relevance grades.**  Continuous
  F1 must be quantile-binned per project (`f1_to_relevance`).  Choosing
  a different `num_relevance_grades` is a model-quality knob, not a
  display setting; metadata records the value used at fit time.
* **Per-project min-max ensembling is scale-invariant w.r.t. ranking.**
  Warmup calibration / multiplicative scale factors only move
  ranker-vs-regressor agreement within a project; they do **not**
  improve top-k hit rate (`argmax(p × s) = argmax(p)`).  Warmup
  calibration is therefore not part of the shipped predictor; it lives
  in `scripts/research/combo_predictor_v6/warmup_calibration.py` for
  optional offline MAE/MAPE calibration only.
* **PCA at LOPO holdout time is capped at `min(n_samples,
  n_features)`.**  With 37 projects, the per-fold refit caps PCA at
  ~32 components — hence the production setting of `dino_dims = 32`.
  Increasing the project count meaningfully (>50) would let us
  re-sweep `dino_dims`.
* **Time predictor needs an anchor to be useful.**  Physical-only
  predictions (`anchor_elapsed_sec=None`) are R²(log) ≈ -0.005 —
  reliable only for *ordering* combos by relative cost.  Show them as
  "(physical-only)" in the UI and prompt the user to run the
  recommended `time_anchor_combo` before quoting absolute ETAs.
* **Time predictor coefficients are GPU-and-thermal-coupled.**  The
  shipped `phys_time.json` was fit on a 3080 Ti / 3090 / 5090 mix and
  encodes their power-limit / boost-clock behaviour implicitly.  When a
  significantly different GPU lands (e.g. an H100), refit `package.py`
  against fresh timings; the warmup-calibration scale absorbs same-GPU
  variance per project but cannot rescue a fundamentally different
  architecture.
