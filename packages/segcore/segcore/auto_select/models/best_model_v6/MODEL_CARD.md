<!-- SPDX-License-Identifier: Apache-2.0 -->
# Model Card — auto-select combo predictor v6 (`best_model_v6`)

## What this model is

An XGBoost ensemble that recommends training configurations
(architecture / loss / patch size / batch-channel combos) for a new
segmentation project from its dataset profile. It powers the
"auto-select" feature; it does **not** segment images itself.

| File | Contents |
|---|---|
| `regressor.json` | XGBoost regressor — predicted F1 per combo |
| `ranker.json` | XGBoost ranker — combo ordering |
| `oom_classifier.json` | XGBoost classifier — out-of-memory risk |
| `vram_regressor.json` / `vram_metadata.json` | VRAM usage estimate |
| `phys_time.json` | Physical training-time anchor model |
| `dino_pca.pkl` | PCA projection for DINOv2 embedding features |
| `metadata.json` | Feature schema, z-score parameters, combo list |

## Training data and provenance

- Trained on **12,362 rows of training-run outcomes across 37 internal
  research projects**, collected on the maintainers' own hardware.
- Each row is *run-level tabular metadata*: the tried configuration, the
  resulting quality metric, runtime, and VRAM — plus a numeric dataset
  profile (13 handcrafted statistics and PCA-projected DINOv2 embedding
  statistics) of the project the run belonged to.
- **No images, masks, or per-pixel data were used as training input, and
  none are recoverable from the shipped artifacts.** The artifacts
  contain only decision-tree split thresholds over aggregate statistics,
  PCA basis vectors, and z-score normalization constants.
- The model files in this directory hold no project names, file paths,
  UUIDs, or free-text fields. The per-project statistics bundled with
  auto-select (`segcore/auto_select/model_combo_library.json`) name each
  project by a `Sample_*` id in place of its internal name; each id keeps
  a generic word for what the images show, such as a part or a kind of
  defect.

## License

Apache-2.0, same as the repository. Runtime dependency licenses:
xgboost (Apache-2.0), scikit-learn (BSD-3-Clause), numpy (BSD-3-Clause);
the DINOv2 weights used for profile embeddings are Apache-2.0.

## Intended use and limitations

- Intended: ranking candidate training configurations for industrial
  segmentation datasets inside Seg-Studio.
- The training distribution is industrial inspection imagery profiled on
  specific GPU classes; recommendations for very different domains or
  hardware are extrapolations (LOOCV on the training projects: Top-5
  contains the best combo ~85% of the time — see
  `docs/auto_select_v6_combo_predictor.md`).
- The model never blocks manual configuration; predictions are
  suggestions surfaced in the UI.

## Retraining / reproducibility

The build pipeline lives in `scripts/make_autoalgorithm/` (library
build, LOOCV validation). The underlying run table is internal and not
distributed; third parties can rebuild an equivalent model from their
own run history with the same scripts.

