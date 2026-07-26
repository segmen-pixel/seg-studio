# Changelog

All notable changes to this project will be documented in this file.

## [0.9.8] - 2026-07-27

### Added — instance segmentation as a training mode ("counting")

Training gains a third mode alongside normal and quick: **Instance
(counting)**. It answers "how many objects are there?" when objects touch
or overlap and a semantic mask alone cannot separate them.

The mode adds no annotation work. Instance ground truth is *synthesized*
from the semantic masks that already exist: cutouts are extracted from the
painted regions and copy-pasted into composed scenes with known instance
identities, and an RF-DETR-Seg model is fine-tuned on that COCO dataset.
On the reference counting workload this reached exact counts on a held-out
real test set with zero manually drawn instance labels.

- **Model sizes**: Small (default), Medium, Large. Batch size is auto-fitted
  to the detected VRAM from a measured per-size table, halving the batch and
  doubling gradient accumulation so the effective batch is preserved.
- **Counting**: a validation-calibrated score threshold plus mask-IoU dedup
  at 0.7, which removes the duplicate-mask artifact DETR-family models
  produce on single objects. Adjacent-but-touching objects have near-zero
  mask IoU, so they are not merged.
- **Multi-class**: one model counts every painted class, and the class
  mapping travels with the exported serving contract.
- **Results**: instance overlay with numbered badges in an Okabe-Ito
  palette, per-class count chips, and per-image counts.
- **Export / serving**: ONNX export into the serving registry and a
  `/count` endpoint in serving_api.

Synthesis is configured inline in the Training form (object counts,
objects per image, stack-pair probability, seed, area-band override) with a
Preview button that composes two or three samples before you commit to a
run. Instance runs report per-epoch progress in the run log and drive the
UI progress bar.

Nano was offered during development and retired before release: on the
reference workload it reached 0.92 segm mAP50 against 0.94 (Small) and
0.99 (Medium), which is not enough for exact counts. Existing Nano
checkpoints stay loadable for prediction.

### Added — RF-DETR dependencies, with the non-commercial surface excluded

`rfdetr` ships in the core lockfile rather than an optional extra. Only the
Apache-2.0 package and the Apache-2.0 **Seg** checkpoints are used; the
PML-1.0 "plus" extra and the detection XL/2XL classes are excluded, and ban
patterns cover them. The GUI `cv2` build that `supervision` requests is
replaced by the existing `opencv-python-headless` through a lockfile
override.

`torchmetrics` needed the same treatment for a different reason. The package
is Apache-2.0, so every metadata-level check passes it, but one module —
the Extended Edit Distance text metric — carries a license derived from the
Qt Non-Commercial License v1.0 and is not licensed for commercial use. We
use torchmetrics only for detection mAP, and `import torchmetrics` loads the
text package eagerly, so the installer swaps that module and its Metric-class
wrapper for Apache-2.0 stubs rather than deleting them. The build fails
closed if a future version moves the code. All added dependencies are
recorded in THIRD_PARTY_NOTICES.md.

### Changed — installer reproducibility and source-release compliance

The installer builds from the repository alone; the dev-venv fallback is
gone and a failing build step aborts instead of silently degrading. Builds
emit `release_manifest.json` with a SHA-256 for every staged file.
`scripts/release/collect_lgpl_sources.py` assembles the LGPL/MPL
corresponding-source bundle for binary releases and fails on unpinned or
unknown libraries.

### Fixed

- Instance runs pin the training device explicitly, gate on checkpoint
  presence, avoid train/val leakage in the synthesized split, and serialize
  prediction requests.
- Composition scales correctly across source resolutions and caps cutouts
  to the canvas; single-object area bands are resolution-relative.
- Instance results no longer show semantic region pills that do not apply.
- Annotation gains a batched per-class label clear across the image-list
  selection.

### Housekeeping

- Every version surface now reports the same number. `config.APP_VERSION`,
  `report_generator`, `segcore` and `seg-sdk` had drifted a release behind
  the UI; `report_generator` now imports the single backend definition.
- The docs no longer present `deeplabv3plus` as a selectable architecture —
  it was removed from the registry in 0.9.7. The sweep records that measured
  it are annotated rather than rewritten.
- `timm` is a declared dependency again (`timm==1.0.*`, Apache-2.0), reversing
  the 0.9.6 removal: MobileSAM and TinySAM import it at module load without
  declaring it themselves, so a clean install failed the moment either SAM
  backend was selected. Recorded in THIRD_PARTY_NOTICES.md.

## [0.9.7] - 2026-07-17

### Security — CVE lockfile refresh + cross-platform path sanitization (2026-07-21)

- `_sanitize_filename` now uses `PureWindowsPath` so backslash traversal
  segments (`..%5C..%5C`) are stripped on Linux/macOS deployments too;
  behavior on Windows is unchanged.
- Dependency pins refreshed to clear all open pip-audit advisories
  (58 advisories across 8 packages): idna 3.18, onnx 1.22.0,
  pillow 12.3.0, python-multipart 0.0.32, setuptools 83.0.0,
  starlette 1.3.1, transformers 5.5.4, weasyprint 69.0.
  fastapi stays at 0.135.4 (allows starlette >=0.46 with no upper bound).
- Golden-run regression fixture regenerated for the refreshed
  environment (verified deterministic across repeated runs) and the test
  now runs only on Windows — the fixture is machine/env-pinned, so
  cross-platform float drift on Linux CI is expected, not a regression.

### Removed — automatic donor warm-start (auto_mode "full")

The backend half of the ADR-005 addendum: training no longer attaches a
similar past project's checkpoint unless the user explicitly picks one
(Transfer mode / model search, both unchanged). `auto_mode` is now
`"recipe_only" | "off"` with `"recipe_only"` the default; stale requests
sending the retired `"full"` coerce to `"recipe_only"`, and the legacy
`auto_select` field is accepted but ignored. The from-scratch epoch
budget (wave6 min_width rule) moved into the recipe phase and now
applies only when the request does not pin an explicit epoch count —
donor-similarity epoch shortening is gone with the donor path.

### Removed — cloud training (parked internally)

The RunPod Serverless cloud-training option (cloud mode toggle, GPU
picker, job widget, credentials store, the six cloud routers, S3
transfer, and the RunPod worker handler) is out of the open-source
distribution. The public scope is local-first; the implementation is
preserved verbatim in the private tree and may return in a later
release. Encrypted-credential storage shipped only for cloud keys and
leaves with it.

### Removed — anomaly training mode

The OK-images-only anomaly mode is retired end to end (mode card, its
hyperparameter subform, the anomaly worker/subprocess path, the
prediction endpoints and result views, `segcore.anomaly`, and the
unwired reconstruction models in `segcore.training.model_vae`). After
the 2026-07 method removal it was down to a single positional z-score
detector — too thin to carry a flagship "train from OK images" promise.
Anomaly detection moves to the companion AnomaLens project.

Legacy runs keep their `anomaly_model.pkl` artifacts on disk (startup
cleanup will not delete them) but are no longer listed or viewable in
the UI, and training requests with the mode are rejected (400).

## [0.9.6] - 2026-07-08

### auto-config — z-score fallback now reports why the ML predictor was skipped (2026-07-07)

When the ML combo predictor cannot run, `recommend_combo` falls back to
the legacy z-score portfolio. That fallback used to be invisible in the
training log: the reason only reached the server console, so a broken
predictor (e.g. `xgboost` missing from the serving venv, which forced
every run onto the z-score path and its stricter confidence gate)
degraded recommendations silently. Now:

- `ConfigRecommendation.ml_fallback_reason` carries the failure reason
  (bundle load error, missing imports, feature-extraction or `rank()`
  exception, or prepared images/masks unavailable).
- Phase 3 logs `Auto-config: ML predictor unavailable (<reason>); using
  z-score fallback` right before the `[zscore]` recommendation line.

### training — auto defaults refreshed from per-axis EDA (2026-07-07)

A per-project best-F1 analysis on the wave1-6 unified table
(n=37 projects, 12,362 rows) showed the pre-existing defaults land on
values that are per-project best in <= 15% of cases. Both backend and
UI defaults were updated to the plurality winners:

- `arch`: `simpleunet` → `deeplabv3plus` (5/37 → 17/37 per-project best)
- `base_channels`: `64` → `128` (5/37 → 29/37)
- `fg_patch_prob`: `0.5` → `0.7` (1/37 → 13/37)
- `class_weight_strength`: `0.45` → `0.5` (0.5 wins 19/37)
- `useDinov2` (distill): `true` → `false` (24/37 per-project best `False`)
- `patch_size`: `256` unchanged (37/37 best in mean-F1, no project
  where `128` or `512` wins in per-project max-F1)

The `best_model_v6` bundle was refreshed on the same data with all 26
feature columns filled. The prior `features_v2.json` cache had 14 of
26 scalar columns silently zero-filled (`fg_ratio`, `num_train`,
`mean_width`, `log_img_pixels`, `class_imbalance_ratio`, …); repairing
it and re-fitting the regressor / ranker over the wave1-6 12,362-row
table lifts LOPO metrics for the regressor:

| metric | before | after |
|---|---:|---:|
| top-5 hit | 5.4% | **10.8%** |
| top-3 hit | 5.4% | **8.1%** |
| close_within_0.05 | 43.2% | **51.4%** |

Schema, `dino_dims`, `feature_columns`, and `all_combos` are unchanged
between the old and new model bundle — only `n_train_rows` (9,084 →
12,362), `n_trees_regressor`, and `n_trees_ranker` differ.

Finally, `AutoOrchestrator` gained a small post-ML sanity layer that
nudges three dominated values (`arch=simpleunet`,
`fg_patch_prob=0.5`, `class_weight_strength=0.8`) toward the
evidence-based majority. The rule fires whether the offending value
came from a stale config, a user override, or an unusual ML pick;
picks on all other axes are untouched.

See `docs/auto-config-rationale.md` §7 for the underlying per-project
best-hit distributions and full methodology.

The dynamic `fg_patch_prob` adjuster (per-epoch P/R balancing) had its
bounds corrected to the swept range: cap 0.90 → **0.80** (fp > 0.80
was never swept and starves patches of background context; 0.8
already loses to 0.7 in paired means), fg-tiered floor 0.60/0.50/0.30
→ flat **0.30** (fp=0.3 is the per-project best in ~1/3 of sparse
projects, which the 0.60 floor locked out). Also fixed: the decrease
branch could drag an explicitly-set below-floor value *up* to the
floor; bounds now gate movement instead of attracting values.

The auto loss tier was also revised: projects with `fg_ratio < 0.03`
(the very-sparse tier — most industrial-defect projects) now get
`focal` instead of `lovasz` when `loss_type` is on auto. Per-project
paired mean-F1 comparison on the selection-bias-free wave1-4 table
puts focal over lovasz 17:1 (mean gain +0.026) in that tier; the old
tier had been set from the pooled cross-project mean, which penalises
focal for its bad cells across unrelated recipe combos. The
dense/middle default stays `ce`. Explicit user loss choices are
unaffected (the tier only fills `loss_type=None`).

### training — GPU selector restored in the training form (2026-05-22)

The training form had lost its GPU `<select>`: the change handler, the
device-list API client and the `torchState` plumbing were all still
present, but the dropdown itself was gone — so training could only run
on whatever the "auto" device resolved to. The dropdown is back, next
to the model-name field; it lists every CUDA device (plus an "Auto"
entry) and sets the device through `PUT /hardware/torch/device`. This
matters on a multi-GPU host where "auto" can pick the wrong card.

### training — batch sizing: dry-run verify instead of maximise (2026-05-22)

The VRAM batch profiler ran a 50-step saturation probe (exponential
ramp + binary search) to find the *largest* batch that fits, and
`training_runner` used that value directly as the training batch. On a
24 GB GPU it picked batch 163, pinning VRAM at the limit: on Windows
WDDM the driver then spills allocations to shared system memory over
PCIe (~10x per-epoch slowdown), and `patches_per_image` was inflated
in lock-step so an epoch processed far more patches than configured.

The profiler no longer maximises. `_profile_max_batch_size` is now a
short **dry run**: it runs a few forward+backward steps at the
configured `batch_size` and returns it if it fits, halving on OOM for
a GPU too small. The wave4-era saturation probe (~350 lines —
`_profile_one_pass`, the `_PROBE_*` constants, the profile cache) is
removed; it only existed to safely run *at* the maximum. The
configured batch is a moderate, GPU-agnostic target — no hardcoded
caps, no maximisation.

### training — explicit loss type & class weight strength honoured (2026-05-22)

`_auto_tune_training` applied its data-driven wave4 recipe
(`loss_type`, `class_weight_strength`) to every run unconditionally,
silently overriding an explicit user choice — only the wave4 ablation
sweep could opt out via `library_compatible_recipe`. A value picked in
the training form was therefore ignored.

- `loss_type` and `class_weight_strength` now follow the None-sentinel
  pattern already used by `dice_weight`: `None` means "auto" and takes
  the data-driven wave4 tier value; a concrete value is an explicit
  user choice and is honoured verbatim.
- The training form gains a **loss function** selector with an
  "Auto (recommended)" default; the class-weight-strength auto/manual
  toggle now actually takes effect in manual mode.
- `TrainConfig`, the `POST /train` schema and `training_runner`
  propagate `None` end to end instead of coercing it to `focal` / a
  fixed `0.80`. The wave4 sweep is unaffected — it passes an explicit
  recipe for every cell and still opts out via the gate.

### auto_select v6 — model-search DINOv2 runs on GPU (2026-05-22)

The model-search endpoint picked the combo-predictor's DINOv2
extraction device by string-matching the configured device against
`"cuda"`. The default configured value is `"auto"`, which never
matches, so DINOv2 silently ran on CPU. The configured device is now
resolved through `resolve_torch_device_or_cpu()` first, so `"auto"`
maps to the selected GPU whenever one is available.

### auto_select v6 — VRAM predictor batch-free correction (2026-05-22)

Live model-search testing exposed a causal-direction bug. wave5
measures VRAM at the *auto-fit* batch size, where `batch_size` and
`vram_peak` are negatively correlated (heavier settings get a smaller
auto-fit batch but still peak higher, corr = -0.24). Feeding
`batch_size` as a feature made XGBoost learn the inverted relationship;
`max_safe_batch` then claimed a light combo would OOM at batch=1.

- **Dropped `batch_size`** and the batch interactions from the feature
  surface (24 → 20 columns). The predictor now estimates the VRAM peak
  of an auto-fit run directly, with `gpu_total_mb` as the
  hardware-budget proxy.
- **`max_safe_batch` removed** — wave5 has no batch-controlled data, so
  back-solving a batch size is not supportable. `verdict()` /
  `predict_vram_mb()` / `predict_oom_prob()` lose their `batch_size`
  parameter.
- **Retrained**: regressor MAPE 3.6 % / R²(log) 0.99; classifier
  AUC 0.99 (the inverted feature had been noise). Verified against
  wave5 ground truth — `Bolt` `simpleunet_bc64` predicted 23,145 MB vs
  actual 23,210 MB on the RTX 5090, 7,961 vs 7,999 on the 3080 Ti.
- Callers (`training_runner`, model-search endpoint, trainer_ui)
  updated to the batch-free verdict; the `vram` payload is now
  `{gpu_total_mb, driver, pred_vram_mb, budget_mb, verdict, oom_risk}`.

### auto_select v6 — VRAM predictor wired through API + UI (2026-05-22)

- **`segcore.auto_select`** now re-exports `VramPredictor` and
  `get_default_vram_predictor` as public API.
- **`training_runner.py`** auto-config phase logs the WDDM-safe batch
  ceiling for the picked combo on the resolved GPU
  (`Auto-config [VRAM]: …`), and prints an explicit OOM warning when
  the combo cannot fit even at batch=1.
- **`POST /train/model-search`** — `config_recommendation` gains a
  `vram` object (`gpu_total_mb`, `driver`, `safe_batch_ceiling`,
  `oom_risk`) computed for the top combo on the configured GPU.
- **trainer_ui** — the model-search panel shows a one-line VRAM verdict
  (`✓ VRAM: fits …` / `⚠ VRAM: this combo may OOM …`).
- No new dependency; GPU-property reads sit on surfaces that already
  use CUDA, so no extra CUDA context is created on the request path.

### auto_select v6 — VRAM predictor / OOM avoidance (2026-05-22)

- **`segcore.auto_select.vram_predictor`** new module: estimates peak
  GPU memory for a candidate `(combo, batch_size, project, GPU)` and
  issues a **WDDM-aware OOM verdict** with an explicit safety margin.
- **Two XGBoost heads**, bundled in
  `packages/segcore/segcore/auto_select/models/best_model_v6/`:
  - `vram_regressor.json` — `reg:squarederror` on log-VRAM. LOPO:
    MAPE 3.3 %, R²(log) 0.99.
  - `oom_classifier.json` — `binary:logistic`. LOPO: AUC 0.99,
    recall 0.96.
  - `vram_metadata.json` — 24-feature schema + data-driven safety
    policy + LOPO metrics.
- **WDDM safety margin**: the raw regressor under-predicts on ~65 % of
  rows, so `verdict()` inflates the estimate by the LOPO 95th-percentile
  under-prediction band (`×1.195`) and subtracts a driver-specific
  headroom — **WDDM** 2048 MB + 0.92 usable fraction (cuDNN warm-up +
  compositor reclaim), **Linux** 512 MB + 0.94.  wave5 measured this
  directly: all 542 OOM events hit the headroom-less Linux 3080 Ti,
  zero on the WDDM cards.
- Trained on the wave5 cross-device VRAM probe (9,669 rows / 37
  projects, RTX 5090 / 3090 / 3080 Ti).  The research training code and
  the production fit script live in the internal research repo and are
  not shipped in the public tree.
- Stack unchanged — XGBoost (Apache-2.0) + numpy (BSD-3); no new
  runtime dependency (xgboost already shipped since the Phase 4 swap).
- CI `python-imports` gains a "Verify auto_select v6 VRAM predictor
  loads" step (verdict shape + WDDM-tighter-than-Linux assertion);
  `tests/test_auto_select.py::TestVramPredictorV6` adds 4 smoke tests.
- Known limitation: `max_safe_batch` is conservative — wave5 collected
  VRAM at the auto-fit batch size, so small-batch estimates are
  extrapolated.  Use `verdict()` as the primary check.

### auto_select v6 — Phase 7: ETA chip in UI + recalibration UX (2026-05-21)

- **`POST /api/v1/projects/{id}/train/model-search`** now accepts an
  optional `anchor_elapsed_sec` query parameter (`gt=0`) and threads it
  through to `recommend_combo()` so the response surfaces *calibrated*
  training-time predictions when the caller has measured the warmup
  anchor.  The returned `config_recommendation` dict gains four fields:
  `pred_elapsed_sec`, `pred_elapsed_min`, `time_anchor_combo`,
  `time_calibrated`; `top_combos_detail[*]` rows pick up the same
  per-combo time pair.
- **`training_runner.py` auto-config phase** logs the predicted training
  time alongside the recommended recipe — `~N.N min (calibrated|
  physical-only)` — and, when no anchor has been measured yet, prints a
  one-line tip naming the warmup anchor combo so the operator can run
  it first.
- **trainer_ui (React)**: `modelSearch()` is now typed end-to-end
  (`ConfigRecommendationApi` + `ModelSearchResponse`).  The model-search
  result panel renders the ETA inline (`⏱ ~N.N min`) and a
  **"Recalibrate ETAs"** button next to it.  Clicking the button
  prompts for the anchor combo's measured seconds and re-fetches
  model-search with `anchor_elapsed_sec` populated — the result
  switches from `(physical-only)` to `(calibrated)` once a valid value
  is supplied.
- Auto-detection of the anchor's elapsed_sec from completed
  `TrainingRun` records is intentionally deferred (run records do not
  yet carry `elapsed_sec` directly; manual entry covers the MVP
  while keeping the change footprint bounded).

### auto_select v6 — warmup-calibrated training-time predictor (2026-05-21)

- **`segcore.auto_select.time_predictor`** new module: numpy-only
  inference for the v4 warmup-calibrated training-time model.  Bundled
  alongside the F1 predictor as
  `packages/segcore/segcore/auto_select/models/best_model_v6/phys_time.json`
  (≤ 1 KB: 10 coefficients + intercept + recommended anchor combo).
- **`ComboPredictor.rank(..., anchor_elapsed_sec=...)`** new optional
  argument and two new fields on every returned dict —
  `pred_elapsed_sec` and `pred_elapsed_min`.  Without an anchor the
  physical-only prediction is returned (R²(log) ≈ -0.005, ordering
  only); with the anchor's actual elapsed time it jumps to
  R²(log) ≈ +0.958 / MAPE ≈ 14 %.  `ComboPredictor.anchor_combo`
  exposes the recommended warmup combo.
- **`config_selector.recommend_combo(..., anchor_elapsed_sec=...)`**
  propagates the anchor argument through to the ML path.  New
  `ConfigRecommendation` fields: `pred_elapsed_sec`,
  `pred_elapsed_min`, `time_anchor_combo`, `time_calibrated`.
  `reasoning` text now includes `~N min (calibrated|physical-only)`.
- The JSON bundle is produced by an internal research packaging script
  (not shipped in the public tree) that fits the log-linear physical
  time model on the 2026-05-13 wave4 timing snapshot
  (1,092 ok-status rows / 37 projects).
- **CI smoke (`python-imports` job)** now also verifies the time
  bundle is present, every combo carries a non-None `pred_elapsed_sec`,
  and the warmup-calibration anchor identity holds to 1e-3 seconds.
- **Tests**: `tests/test_auto_select.py::TestTimePredictorV6` (3
  smoke tests).  26 / 26 pass locally.

### auto_select combo predictor v6 — OSS-clean swap (2026-05-21)

- **Backend changed from LightGBM (MIT) to XGBoost (Apache-2.0)** for the
  `segcore.auto_select` combo predictor.  This drops the only MIT-licensed
  runtime dependency in the recommendation pipeline so the shipped stack is
  exclusively Apache-2.0 / BSD-3-Clause.  No public API breakage:
  `ComboPredictor.load()` / `predictor.rank()` / `get_default_predictor()` keep
  their signatures, and every returned dict carries the same keys
  (`combo`, `arch`, `base_channels`, `patch_size`, `rank_score`, `pred_f1`,
  `pred_std`, `ci_low`, `ci_high`) consumed by `_recommend_via_ml` in
  `config_selector.py`.
- **Model retrained on the wave1-4 unified table** (9,084 rows, 37 projects):
  XGBoost `reg:squarederror` regressor + `rank:ndcg` LambdaRank ranker, mixed
  via per-project min-max normalisation with `weight_reg=0.8` (the LOPO-best
  configuration in the 2026-05-21 sweep).  DINOv2 PCA reduced to 32
  components (was 4 in v3).  (Research training code is not shipped in
  the public tree.)
- **Bundled artefacts** (`packages/segcore/segcore/auto_select/models/best_model_v6/`):
  `regressor.json`, `ranker.json`, `dino_pca.pkl`, `metadata.json` — total
  ~1.2 MB (vs ~5.6 MB for the v3 dual-model bundle).  Legacy
  `best_model_dual/` (5 LightGBM calibrators + ranker + PCA) deleted.
- **Dependency**: `lightgbm>=4.0.0,<5` replaced with `xgboost>=2.0.0,<4` in
  both `apps/trainer_api/requirements.in` and `requirements-cu128.in`;
  lockfiles regenerated.  THIRD_PARTY_NOTICES updated.

### Pre-publish OSS review fixes (2026-05-13 audit)

Six blockers and several should-fix items from the internal pre-publish
review (the full review report is not shipped in the public tree):

- **API token bypass closed** (`apps/trainer_api/app/main.py`): non-versioned router registrations were removed so the optional `SEG_API_TOKEN` middleware cannot be skirted by hitting the prefix-less path. All trainer API routes are now reachable only under `/api/v1/...`. The `/v2/...` and `/ws/v2/...` surfaces used by `seg-sdk` remain prefix-less but are now also guarded by the same middleware. **Breaking change for any caller hitting `/projects/...` etc. directly — switch to `/api/v1/projects/...`.**
- **CoC reports go private** (`CODE_OF_CONDUCT.md`): harassment reports now route to GitHub Security Advisory instead of public Issues.
- **SyntaxError in `make_autoalgorithm/__main__.py`** removed (`$HEADER` placeholder left over from SPDX templating).
- **`eslint` itself added to `apps/trainer_ui/devDependencies`**, plus the matching `package-lock.json` regeneration. `npm ci && npm run lint` now works on a fresh clone. Two `no-constant-condition` errors and one `no-empty` error were fixed inline; `--max-warnings` was raised from 0 to 200 to make the existing `no-explicit-any` debt a non-blocker (a dedicated cleanup pass will bring it back to 0).
- **Debug `console.log` (17 statements with `[DBG]…` / `[Infer]…` prefixes) removed** from `store.ts`, `TiledViewer.tsx`, `useAnnotatorEffects.ts`, `useImageList.ts`, `useInferenceEngine.ts`.
- **CVAT / Annotation reverse-proxy is now opt-in**: `/cvat/*` and `/annotate/*` are mounted only when `SEG_CVAT_URL` / `SEG_ANNOTATION_URL` are set explicitly. With them unset (the new default), the routes simply do not exist — closing the localhost-SSRF surface when `SEG_HOST=0.0.0.0` is used.
- **Raw exception strings no longer leak through error responses**: `str(e)` / `f"...: {e}"` patterns in `apps/trainer_api/app/main.py` (S3 volume validate), `apps/trainer_api/app/routers/reports.py`, and `apps/trainer_api/app/routers/cloud_train.py` (RunPod endpoint + S3 health checks) now return a categorised, generic message; full details land in the server log under a correlation ID.
- **`segcore` public SDK surface filled in**: `from segcore import build_model, MODEL_REGISTRY, TrainConfig, __version__, compute_miou, accumulate_f1_stats, …` now works. Deep imports remain available for the trainer-api and internal scripts.
- **Internal E2E planning docs untracked**: `apps/trainer_ui/e2e/E2E_TEST_PLAN.md` and `apps/trainer_ui/e2e/UX_COVERAGE_EXPANSION.md` (both already in `.gitignore`) are no longer tracked, so they will not flow out via subtree push.
- **`APP_VERSION` synced to 0.9.5** in `apps/trainer_api/app/core/config.py` (was stuck at 0.9.4 — `pyproject.toml` had already been bumped). The MCP-server default `API_BASE` and the `--api` CLI option now auto-append `/api/v1` so existing invocations keep working.

### Security

- **Dependabot**: weekly scans for pip (root + trainer_api + serving_api), npm (trainer_ui), and GitHub Actions
- **CodeQL**: push/PR + weekly schedule for Python and JavaScript/TypeScript with the `security-and-quality` query suite
- **Lockfile-based reproducible installs**: `apps/trainer_api/requirements.in` (human-edited) → `requirements.txt` (auto-generated by `uv pip compile`, fully pinned with all transitive deps). Same pair for `serving_api` and `requirements-dev`. `mobile-sam` and `sam-2` git+ URLs are now pinned to specific commit SHAs.
- **`pip-audit` CI**: every push runs OSV vulnerability scans against trainer_api and serving_api lockfiles; build fails on known CVEs in shipped deps.
- **Lockfile drift CI** (`.github/workflows/lockfile-drift.yml`): re-runs `uv pip compile` and rejects PRs that ship a `.in` change without the matching `.txt` regeneration — closing the loophole where a dependency could be added without flowing through the LICENSE-confirmation trail.
- **SBOM workflow** (`.github/workflows/sbom.yml`): on every `v*` tag, generates CycloneDX 1.6 + SPDX 2.3 Software Bill of Materials, re-checks the SBOM for non-commercial license expressions, and attaches both files to the GitHub Release. Local sanity-check confirmed 173/182 components carry license metadata, **0 NC components** detected (the prior NC-vendor purge is now mechanically verifiable).

### Changes

- Removed unused `timm` dependency from `apps/trainer_api/requirements.txt`
- Declared `xgboost` dependency (used by the auto_select combo predictor v6, replacing the unannounced LightGBM dep that v3 carried) and added it to THIRD_PARTY_NOTICES
- Dropped the video tutorial asset pipeline in favor of the in-app hands-on tutorial introduced in 0.9.4
- Completed SPDX-License-Identifier coverage across tracked Python sources
- Filled in `readme` / `authors` / `keywords` / `classifiers` / `urls` metadata in `packages/segcore/pyproject.toml`
- Optional `SEG_API_TOKEN` shared-secret middleware: when set, all `/api/v1/*` requests must carry `X-API-Token`. Unset by default, keeping localhost-only behavior unchanged.
- `docker-compose.yml` now publishes every port on `127.0.0.1` only to avoid accidental LAN exposure of the unauthenticated default configuration.
- Cloud credential transport: RunPod API key, S3 access key, and S3 secret key are now carried in request headers (`X-Runpod-Api-Key`, `X-Runpod-S3-Access-Key`, `X-Runpod-S3-Secret-Key`) instead of `?api_key=` query strings, so they no longer end up in browser history, reverse-proxy access logs, or error reports. The outgoing RunPod GraphQL calls switched from `?api_key=` to `Authorization: Bearer …`.
- `CloudJobWidget` dropped its legacy `localStorage` reads (`seg_cloud_api_key` / `seg_cloud_s3_*`); the widget now relies entirely on backend-side credential resolution.

### GPU support

- **Parallel Blackwell (sm_120 / RTX 5090) lockfile**: added `apps/trainer_api/requirements-cu128.in` and `requirements-cu128.txt` carrying `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`. All other dependencies identical to the cu124 lockfile (only the torch family changes). Verified: 5090 + 3090 dual-GPU enumeration, `cosine_sim=1.000000` agreement on DINOv2 embeddings vs the cu124 baseline, segcore tests 31/31 PASS. See `docs/BLACKWELL_MIGRATION.md` for the parallel-venv setup procedure. The existing `.venv-windows` (cu124) remains the production environment; `requirements.txt` is unchanged.

## [0.9.5] - 2026-05-08

### License & redistribution hardening (pre-public-release)

- Restricted online distillation teachers to DINOv2 (Apache-2.0) and SAM2 (Apache-2.0). Legacy generic-state-dict teacher loader, the matching pretrained-init checkpoint adapter, and the FastAPI precompute / ensemble endpoints have been removed; UI was already DINOv2-only.
- Stopped bundling the DINOv2 `facebookresearch_dinov2_main/` torch-hub source tree in the installer. Recent upstream versions mix Apache-2.0 with non-commercial license fragments (CC-BY-NC-4.0 in `LICENSE_CELL_DINO_CODE`, FAIR Noncommercial in `LICENSE_XRAY_DINO_MODEL`); only the Apache-2.0 weight (`dinov2_vitb14_pretrain.pth`) is shipped now, and the model-definition source is fetched at runtime via `torch.hub.load(...)` on the user's machine.
- `licenses/third_party/lgpl/` now ships the full text of LGPL-2.0, LGPL-2.1, MPL-1.1 and a GPL-2.0 reference, with a `README` mapping each LGPL component (`libvips`, `Cairo`, `Pango`, `pyphen`, OpenCV-bundled FFmpeg) to its license file and upstream-source URL.
- `scripts/build_installer.py` now copies LICENSE / NOTICE / ThirdPartyNotices files out of every shipped wheel (`torch`, `torchvision`, `opencv-python-headless`, `onnxruntime[-gpu]`, `transformers`, `pyvips`, `Pillow`) into `licenses/third_party/wheels/`. PyTorch's bundled NVIDIA library obligations (cuDNN / cuBLAS / cuFFT / cuRAND / cuSPARSE / NCCL / NVRTC / NVTX) propagate from the wheel's own license bundle.
- `NOTICE`, `THIRD_PARTY_NOTICES.md`, `licenses/third_party/MODEL_WEIGHTS.md`: refreshed obligations text, added trademark disclaimer for NVIDIA / PyTorch / Apple / Microsoft / Meta.

## [0.9.4] - 2026-04-17

### Features

- **Hands-on Tutorial**: 3 modes (Beginner / Intermediate / Expert) with spotlight overlay + 12 animated SVG illustrations explaining each tool and feature
- **Unseen Result Pulse**: Result buttons for completed training runs pulse blue until the user opens them once (desc-mode only, per-project localStorage)
- **Guided Next-Tab Highlight**: With description mode on, the next tab to visit (Projects/Annotate/Training) pulses based on project state (images, annotations, runs)
- **Keyboard Shortcuts in Tutorial**: Enter = next, ←/→ = back/next, Esc = skip, 1/2/3 = choose mode on welcome screen
- **Replay Button in Header**: Restart the hands-on tutorial any time from the header ▶︎ icon

## [0.9.3] - 2026-04-10

### Features

- **Anomaly detection (initial implementation)**: Auto-config, patch training, step progress display (superseded 2026-07, then removed entirely in 0.9.7 — see that entry; anomaly detection moves to AnomaLens)
- **Results Tab**: GT/prediction mask separation, pattern overlay menu (hatching/dots/grid)
- **Desc Mode**: Tooltip rewrite from CSS ::after to JS (layout fix)
- **Live Inspection**: Overlay labels, filters, topbar redesign
- **Mark Clean**: Flag defect-free images
- **Folder Import**: Nested folder support with folder name headers
- **Batch Export**: Multi-project export
- **Training Queue**: Auto-launch reserved runs, resume queue on startup
- **New Models Widget**: Model list with anomaly step progress

### Fixes

- Dual GPU support — busy only when all CUDA devices occupied
- Region labels and class list respect confidence threshold
- SW inference max aggregation and cache invalidation
- Reserved run cascade crash (GPU cooldown + retry)
- Slider UI — thumb reaches edges, wider tracks
- Brush size improvements
- Results tab full Japanese localization

## [0.9.0-beta] - 2026-04-01

### Features

- **Annotator**: Brush, eraser, polygon, wand (flood fill), SAM click segmentation, spot detection, ridge detection, superpixel, crack trace tools
- **SAM Models**: MobileSAM, SAM2 Tiny/Small, TinySAM, EfficientSAM-Ti (all Apache 2.0)
- **MLP Assist**: CPU feature extraction + GPU MLP for interactive annotation assistance
- **Training**: SimpleUNet, STDC, DeepLabV3+ architectures with patch-based training
- **Sliding Window**: Evaluation at native resolution for patch-trained models
- **Export**: ONNX and CoreML model export
- **Results**: Per-image prediction scoring, confidence heatmaps, class overlays
- **Dynamic Result Tabs**: Open multiple model result views simultaneously
- **Distillation**: Teacher-student feature/channel distillation pipeline
- **MCP Server**: 37-tool integration for programmatic operations
- **Data Import**: ZIP import/export, labeling data converter

### Infrastructure

- FastAPI backend with lazy loading (1.2s startup)
- React + Zustand frontend with Web Worker overlay processing
- Subprocess-based training (CUDA crash isolation)
- Structured logging with file output
- Security headers middleware
- Auto-build UI on API startup
