# Auto-config rationale

> Why these defaults are these defaults.

Most segmentation toolkits ship defaults that work well on public benchmarks
(DAGM, MVTec AD, COCO-Stuff, etc.) and rely on the user to tune them for
their actual deployment. Seg-Studio's auto-config picks training defaults
from a different source: a large-scale ablation on **real industrial
inspection projects**, not benchmark performance. This document explains
what was measured, what the data showed, and which findings have been
turned into code-level defaults so far.

> **Reading the numbers.** Everything below is a **ΔF1** — a difference between
> two arms of the same sweep, on the same data, under the same pipeline. That
> is what the sweep was designed to measure and it is what the defaults rest
> on. The underlying absolute F1 values are not quoted here and should not be
> reconstructed from these deltas: they were produced before the v0.9.8
> corrections to per-sample weighting, gradient accumulation and best-model
> selection, so they are not comparable with what the current code reports. The
> comparisons survive those corrections because both arms carried the same
> bias; the absolute levels do not.

## Sweep overview

| metric | value |
|---|---|
| projects | 37 (real industrial inspection workloads) |
| domains covered | precision manufacturing, electronics inspection, SEM imagery, consumer-product QA, packaging defects |
| image width range | 377 – 10,400 px (≈ 28×) |
| foreground ratio range | 0.0001 – 0.13 (≈ 1300×) |
| architectures swept | `simpleunet`, `stdc`, `deeplabv3plus` (retired since 0.9.7 — see note below) |
| base channels | 32, 64, 128 |
| distillation | DINOv2 ViT-B/14 (on/off, Apache-2.0) |
| learning rates | 1e-3, 2e-4, 1e-4 |
| other axes | `fg_patch_prob`, `dice_weight`, `loss_type` (ce/focal/lovasz), `class_weight_strength` |
| training budget per cell | 80 epochs |
| total training runs | ~7,000 unique (project × condition) cells |
| compute | 4 GPUs (1× RTX 3090, 2× RTX 3080 Ti, 1× brief RTX PRO 6000) over ~6 weeks |
| success rate | 99.79% (13 errors out of 6,308 success-row sample) |

> **Note on `deeplabv3plus`.** The sweep below measured it, and the numbers are
> reported as they were observed. The architecture was removed from the trainer
> in 0.9.7 and is no longer selectable — `MODEL_REGISTRY` is `simpleunet` and
> `stdc`, and a request naming a retired architecture falls back to
> `simpleunet`. Rows mentioning it are kept as a record of what was measured,
> not as a recommendation you can act on.

We hold the underlying images private (each project is contractually confidential),
so the report below is aggregate statistics only — no per-project values,
no representative image samples from the dataset.

We do not currently cross-validate against a public benchmark: the
widely-known industrial-defect datasets (MVTec AD, DAGM 2007) are released
under research-only / non-commercial licenses that are incompatible with
this project's Apache-2.0 license and with commercial deployment, so we
neither bundle them nor recommend them as a validation step here.
Independent verification of the trends below requires bringing your own
labelled industrial dataset.

## Findings turned into defaults (enforced in code today)

### 1. Scratch epoch budget is set from `min_width`

A depth-2 decision tree fitted to project-median `best_epoch` picked
`min_width` (smallest image width across the project) as the sole strong
predictor (Spearman ρ = +0.50 with `best_epoch`). Other handcrafted
features — `num_train`, `fg_ratio`, `mean_fg_area_px`, image area,
aspect-ratio — added no signal once `min_width` was in the tree. Deeper
trees overfit at n=37 (LOOCV R² turns negative), so the rule stays at
depth=2.

| bucket | rule | n | best_epoch median | % projects still gaining ≥1pt F1 between ep60–80 |
|---|---|---|---|---|
| small | `min_width < 1000` | 12 | 57.5 | 33% |
| mid (legacy default) | `1000 ≤ min_width < 2000` | 15 | 60.0 | 53% |
| large | `min_width ≥ 2000` | 10 | 65.0 | 70% |

Code: [`app/core/auto_select_utils.py::_recommend_scratch_epochs`](../apps/trainer_api/app/core/auto_select_utils.py).
The user can still pin `epochs` explicitly; the auto path only fires when
`epochs` is unset or `<= 0`.

## Empirical findings (not yet enforced in code)

The findings below are statistically robust but have not been turned into
code defaults yet — they live here as guidance for users who tune
manually and as a roadmap for future auto-config changes.

### 2. Capacity-aware learning rate

The single most important hyperparameter we measured. Paired within-condition
F1 differences (same project, same arch, bc, distill, loss, fp, dw, cws —
only `lr` varies) at n=800+ pairs each:

| comparison | mean ΔF1 | Cohen's d | win / tie / loss |
|---|---|---|---|
| `lr=1e-3` vs `lr=1e-4` | **+0.048** | 0.41 (med) | 58% / 18% / 25% |
| `lr=1e-3` vs `lr=2e-4` | +0.019 | 0.20 (small) | 47% / 21% / 32% |
| `lr=2e-4` vs `lr=1e-4` | +0.033 | 0.39 (med) | 57% / 21% / 22% |

**But the effect reverses with model capacity** (same `lr=1e-3 vs 1e-4`
ΔF1, broken down by arch × base_channels):

| arch × bc | mean ΔF1 (1e-3 vs 1e-4) | direction |
|---|---|---|
| `simpleunet` bc=32 | **+0.179** | high-lr essential |
| `deeplabv3plus` bc=64 | +0.107 | high-lr strong |
| `stdc` bc=32 | +0.079 | high-lr helpful |
| `simpleunet` bc=64 | +0.063 | high-lr helpful |
| `deeplabv3plus` bc=128 | +0.045 | small benefit |
| `stdc` bc=64 | +0.009 | indifferent |
| `stdc` bc=128 | **-0.039** | **high-lr hurts** |

**Take-away.** "Default to `lr=1e-3`" is wrong for high-capacity models;
"default to `lr=1e-4`" is wrong for small models. Out of 35 projects with
full lr-triplet coverage, `1e-4` was the best lr for only 2; we recommend
dropping `1e-4` from sweeps entirely.

### 3. Architecture × base-channels Pareto front

Median F1 across all configs per arch × bc × distill bucket:

| arch | bc | distill | n | F1 median | F1 p90 |
|---|---|---|---|---|---|
| **stdc** | 128 | off | 805 | **0.819** | 0.911 |
| **deeplabv3plus** | 128 | on | 749 | **0.815** | 0.907 |
| stdc | 64 | on | 788 | 0.811 | 0.901 |
| stdc | 64 | off | 827 | 0.809 | 0.906 |
| simpleunet | 64 | off | 749 | 0.793 | 0.908 |
| deeplabv3plus | 128 | off | 750 | 0.778 | 0.886 |
| stdc | 32 | off | 849 | 0.763 | 0.878 |
| deeplabv3plus | 64 | off | 772 | 0.763 | 0.885 |
| simpleunet | 32 | off | 749 | **0.690** | 0.873 |

**Take-aways.**
- `stdc bc128` is the production sweet-spot for 24 GB GPUs.
- `deeplabv3plus bc128 + distill` matches it but at higher cost.
- `simpleunet bc32` is consistently the weakest baseline; remove from
  serious sweeps.

### 4. Distillation × learning rate are independent

The `lr=1e-3 vs 1e-4` effect is the same magnitude regardless of whether
DINOv2 distillation is on (n=198 pairs, ΔF1 = +0.025) or off (n=179
pairs, ΔF1 = +0.026). Auto-config can tune them separately.

### 5. Loss type depends on arch (stdc loves focal)

Paired within-condition `focal vs lovasz` (n=716 total):

| arch × bc | mean ΔF1 (focal − lovasz) |
|---|---|
| stdc × 128 | **+0.059** (focal strongly preferred) |
| stdc × 64 | +0.025 |
| simpleunet × 64 | +0.021 |
| stdc × 32 | +0.012 |
| deeplabv3plus × 128 | +0.004 (indifferent) |
| deeplabv3plus × 64 | +0.002 (indifferent) |
| simpleunet × 32 | −0.009 (lovasz slightly better) |

### 6. Convergence: large-image projects undershoot 80 epochs

Saturating-exponential fits (R² 0.90–0.97) to per-arch median curves
predict the F1 ceiling. Most groups are within 1pt of the ceiling at
epoch 80, with two caveats:

- **`stdc bc32`**: median gains another +0.8pt by epoch 200.
- **`deeplabv3plus bc128 + distill`**: observed peak (0.705) is +0.017
  *above* the fit's asymptote — the model starts overfitting in the
  last 10–15 epochs. Best-checkpoint selection (already standard in
  production) hides the regression from end users.

These observations drive findings #1 above (epoch budget by image size).

## What was *not* useful

- `num_train`, `num_val`, `fg_ratio`, `mean_fg_ratio_per_image`,
  `num_classes` — all uncorrelated with growth potential (|ρ| < 0.15)
  once `min_width` was in the model.
- `class_weight_strength`, `dice_weight`, `fg_patch_prob` — within the
  ranges swept, paired diffs were within ±0.005 F1 across all
  comparisons. They likely matter at extreme values not swept here.

## 7. Per-axis best-hit distribution (2026-07-07 update, n=37)

The findings above look at *paired mean-F1 diffs* across sub-sweeps.
A complementary view — how often does each axis level actually **land
on the per-project best-F1 combo** — was computed on the wave1-6
unified table (12,362 rows, 37 projects) after `features_v2.json` was
rebuilt with all 26 scalar feature columns filled. This exposes three
values that the pre-existing defaults sat on but that win the
per-project best in <= 15% of projects:

| axis | value | best-hit | dominant target | best-hit |
|---|---|---:|---|---:|
| `arch` | `simpleunet` | 5/37 (14%) | `deeplabv3plus` | **17/37** |
| `fg_patch_prob` | `0.5` | 1/37 (3%) | `0.7` | **13/37** |
| `class_weight_strength` | `0.8` | 3/37 (8%) | `0.5` | **19/37** |
| `base_channels` | `64` | 5/37 (14%) | `128` | **29/37** |
| `patch_size` | any non-256 | 0/37 (0%) | `256` | **37/37** |

`base_channels=64` and `patch_size=256` were already the code
convention; the other three were nudged in the same update. See
[`apps/trainer_api/app/schemas.py`](../apps/trainer_api/app/schemas.py)
and [`apps/trainer_ui/src/training/hooks/useTrainForm.ts`](../apps/trainer_ui/src/training/hooks/useTrainForm.ts).

### Post-ML sanity rules (`AutoOrchestrator`)

`AutoOrchestrator._apply_evidence_based_sanity_rules` fires after the
recipe layer runs and nudges the three dominated values back to their
majority target. It fires whether the offending value came from a stale
saved config, an explicit user override, or an unusual ML pick — the
recipe layer's picks on other axes are untouched. Unit tests: [`apps/trainer_api/tests/test_auto_sanity_rules.py`](../apps/trainer_api/tests/test_auto_sanity_rules.py).

### Loss tier revised: very-sparse FG now gets focal

The 2026-04-26 auto tier handed `lovasz` to every project with
`fg_ratio < 0.03` — which is most industrial-defect projects (median
fg_ratio across the 37-project library is 0.0026). That tier was set
from the pooled cross-project mean (`ce 0.825 > lovasz 0.819 > focal
0.784`), a lens that penalises focal for its bad cells across all
recipe combos. Comparing per-project mean F1 per loss (paired within
each project, |gain| > 0.01) inverts the ranking, and the result holds
on the wave1-4 table alone — i.e. it is not an artefact of wave6's
focal-heavy sampling:

| contrast (fg < 0.03, wave1-4 only) | wins | mean gain |
|---|---|---:|
| focal vs lovasz | **17 : 1** (17 tie) | **+0.026** |
| focal vs ce | 7 : 4 (24 tie) | +0.002 |
| lovasz vs ce (all fg) | 1 : 17 (19 tie) | −0.023 |

`TuningPolicy.LOSS_TYPE_VERY_SPARSE` is now `focal`; the dense/middle
default stays `ce` (only n=2 projects have fg ≥ 0.03 and both prefer
ce there). The wave4 instability pair (focal + cws=0.8) is not
reachable via auto — the cws tier hands out 0.5/0.3.

### Dynamic fg_patch_prob bounds aligned with sweep evidence

Training adjusts `fg_patch_prob` each epoch from validation P/R
balance (FP-heavy → more background patches, FN-heavy → more FG
patches). The adjuster's old bounds contradicted the sweep evidence in
both directions:

- **Cap was 0.90.** fp above 0.80 was never swept, and fp=0.8 already
  loses to fp=0.7 in paired per-project means (10:6, wave1-4). Above
  0.80 patches lose the background context needed to suppress false
  positives — a documented no-go. The cap is now **0.80**.
- **Floor was fg-tiered at 0.60/0.50/0.30**, locking sparse-FG
  projects (fg < 0.03 — most of the library) out of fp=0.3, which is
  the per-project best in about a third of them (0.3 beats 0.8 by
  9:2, beats 0.5 by 13:2 in the sparse tier on wave1-4). The floor is
  now a flat **0.30**.
- **Bug: the decrease branch dragged low values up.** `max(floor,
  old - step)` yanks a below-floor starting value (e.g. an explicit
  fp=0.3 with the old 0.60 floor) up to the floor through the branch
  whose purpose is to *decrease* it. Bounds now gate movement only —
  a value outside the range is left where the user put it.

fp=0.3 and fp=0.7 split the sparse tier almost exactly (5:7 with 23
ties), which is why this axis stays with the dynamic controller
rather than a static rule: the right value is project-specific, and
the controller now has the full evidence-backed range to find it.

### cws subdivision rule tested and rejected (2026-07-07)

The per-axis EDA left one open thread: within the sparse tier, the
0.3-vs-0.5 `class_weight_strength` split correlates with dataset
features (`class_imbalance_ratio` ρ = −0.33, `edge_canny_density`
ρ = +0.32, `num_active_classes` ρ = −0.28 on wave1-4, n = 37). We
tested whether a feature-threshold rule ("hand out 0.3 when …")
should replace the flat 0.5 default:

- Head-to-head is nearly all noise: 0.3 wins 7, 0.5 wins 10, and 20
  of 37 projects are within |ΔF1| ≤ 0.01.
- The best in-sample decision stump gains only +0.003 mean F1 — and
  that number is inflated by searching 9 features × ~37 thresholds.
- Leave-one-project-out, the stump's realised gain is **−0.0003**
  (16 projects up, 13 down), and the feature chosen per fold is
  unstable (`mean_width` 33/37 folds, the EDA-suggested
  `class_imbalance_ratio` only 2/37).

The correlations are real but too weak to act on at n = 37: a rule
would reshuffle projects by ±0.04 F1 with zero expected value. The
flat **0.5 default stands**; revisit only if a future wave sweeps cws
densely enough to give the stump a stable out-of-sample gain.

### arch geometry rule evaluated — axis left with the ML predictor (2026-07-07)

The other open thread from the per-axis EDA: deeplabv3plus-vs-stdc
correlates strongly with defect geometry on wave1-4 (n = 37;
`g_mean_convexity` ρ = −0.58, `g_mean_elongation` ρ = −0.53 against
the paired F1 diff). Unlike cws, this axis has real money on the
table: heads-up is 12:15 with 10 ties, mean |ΔF1| = 0.033, and an
oracle that always picks the right one of the two gains +0.015 mean
F1 over always-deeplab. A LOPO decision stump captures about a third
of that (+0.0044, stable feature choice: `g_mean_aspect_ratio` in
34/37 folds) — but its misfires cost up to −0.079 on a single
project.

We still do not add a static rule, for an architectural reason
rather than a statistical one: deeplab-vs-stdc is a *contested* axis,
not a dominated value, and the post-ML sanity layer is scoped to
dominated values only. The geometry features that drive this split
(`g_mean_*`) are already among the predictor's 26 scalar inputs, so
arch selection is the ML's job — one it could not do at all until the
xgboost venv regression was fixed (see CHANGELOG: the ML path was
silently falling back to z-score). Revisit only if post-fix LOPO
shows the predictor consistently mis-picking arch on
geometry-extreme projects that the stump gets right.

### Feature cache repaired

The pre-existing `features_v2.json` cache populated 12 of 26 scalar
columns; the remaining 14 (`fg_ratio`, `num_train`, `mean_width`,
`log_img_pixels`, `class_imbalance_ratio`, …) were silently
zero-filled. This crippled the two cross-interactions the model was
built around (`fp × fg_ratio` and `focal × class_imbalance`). Repairing
the cache from `prepared/dataset_stats.json` (fast path) and
`compute_basic_stats_fallback` (fallback for wave6-only projects) and
re-fitting `best_model_v6` on the wave1-6 table lifts LOPO metrics for
the regressor from top-5 hit 5.4% to 10.8% and close-within-0.05 from
43% to 51%.

## Limitations

1. **n = 37 projects.** Decision-tree analysis past depth 2 overfits
   (LOOCV R² goes negative). The 3-bucket rule is the deepest split we
   trust.
2. **Private data and no public cross-check.** We can show aggregate
   statistics but cannot release the underlying images; readers cannot
   fully reproduce. The obvious public industrial-defect benchmarks
   (MVTec AD, DAGM) are research-only and license-incompatible with
   this project, so we have not validated the trends against an
   external dataset here.
3. **Single inspection regime.** All 37 projects are 2D defect
   localisation; the rules above may not transfer to 3D, video,
   medical, or natural-scene segmentation.
4. **80-epoch ceiling.** We never trained past 80 epochs, so the
   extrapolation in finding #6 is a model fit, not measurement.

## How to opt in or out (`auto_mode`)

The recommendations above are enabled through a single `auto_mode` field
on the training request (ADR-005 Phase D). Set it in the training config
before starting a run:

| value | behaviour |
|---|---|
| `"recipe_only"` (default) | Auto-config on — arch/bc/patch/distill recommendations plus the from-scratch epoch budget |
| `"off"` | Nothing runs — the values in the request body are used verbatim |

The former `"full"` mode (automatic donor warm-start from a similar past
project) was retired as a product decision (ADR-005 addendum): explicit
transfer learning via the Transfer mode remains, but weights are never
attached without the user picking them. Stale requests sending `"full"`
coerce to `"recipe_only"`.

The legacy toggle `auto_config` is still honoured when explicitly
present in the request body and wins over `auto_mode`; the legacy
`auto_select` field is accepted but ignored (its donor behaviour is
gone). This is the pre-Phase-D backward-compat window; the legacy
fields are marked for removal in v1.0.0.

Unknown or mistyped `auto_mode` values silently coerce to
`"recipe_only"` — a noisy failure was tempting but a mistyped mode
should not brick a training run.

## Reproducibility

- Auto-config code: [`apps/trainer_api/app/core/auto_select_utils.py`](../apps/trainer_api/app/core/auto_select_utils.py)
- Unit tests for the rule: [`apps/trainer_api/tests/test_recommend_scratch_epochs.py`](../apps/trainer_api/tests/test_recommend_scratch_epochs.py)
- Sweep harness (research): `scripts/research/ablation_sweep_wave4.py` (private dev repo)
- `scripts/cli_train.py` runs the same training recipe on any existing
  Seg-Studio project (`--arch`, `--base-channels`, `--loss`, `--sweep`
  options), so users can verify the architecture ordering on their own
  data. No third-party dataset is bundled.

## Changelog

- 2026-07-07 — Section 7 added. Per-project best-hit distributions
  computed on the wave1-6 unified table after `features_v2.json` was
  rebuilt with 26/26 scalar columns. Defaults updated for `arch`,
  `fg_patch_prob`, `class_weight_strength`, `base_channels`,
  `useDinov2` (distill); `best_model_v6` refreshed with full-feature
  cache + wave1-6; post-ML sanity rules added to `AutoOrchestrator`;
  auto loss tier for very-sparse FG revised lovasz → focal (17:1 on
  bias-free wave1-4 paired means).
- 2026-06-25 — Initial document. Wave6 sweep 79% complete (6,308 unique
  cells); statistics above are from this checkpoint. Numbers will be
  refreshed at sweep completion (expected mid-July).
