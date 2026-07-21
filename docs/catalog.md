# Seg-Studio Feature Catalog

> **All-in-one image segmentation** — annotate, augment, train, evaluate
> and ship, in one tool.

---

## 🖊 Annotation

| Feature | What it does | Tag |
|---|---|:---:|
| Brush / Wand / Eraser | Core painting tools | Core |
| 🆒 **SAM marking** | One click extracts the target (Meta SAM) | Flagship |
| Crack trace | Click a crack, auto-traced candidates to pick from | Useful |
| Spot detect | DoG-based point-defect detection | Useful |
| Superpixel selection | Grab large regions in a single click | Useful |
| Move tool | Drag existing masks to reposition | Useful |
| **Mark Clean** | Tag OK images as training data in one click | Flagship |

![Annotation tools on a defect image](ja/assets/handbook/catalog_annotate.png)

---

## 🧪 Augmentation

| Feature | What it does | Tag |
|---|---|:---:|
| 🆒 **Perlin CutPaste** | Warp and paste existing defects elsewhere | Flagship |
| 🆒 **Lighting variants** | Day / evening / night color re-grading for outdoor | Flagship |
| Auto-config | Suggests optimal settings from data stats | Useful |
| Annotation Patches | Centroid-biased patch sampling | Core |

![Perlin CutPaste augmentation dialog](ja/assets/handbook/catalog_augment.png)

---

## 🎓 Training

| Feature | What it does | Tag |
|---|---|:---:|
| Training modes | Standard / Quick / Transfer, selected before start | Core |
| Local GPU training | Train on the same PC | Core |
| Transfer learning | Warm-start from a past run's checkpoint you pick | Useful |
| 🆒 **DINOv2 distillation** | Distill features from a 142M-image teacher | Flagship |
| Lovász-Softmax loss | Selectable boundary-sensitive loss | Useful |
| Multi-arch | SimpleUNet / STDC / DeepLabV3+ | Core |
| Deep supervision | Auxiliary losses on intermediate layers | Core |

![Training tab with mode cards and metrics](ja/assets/handbook/catalog_training.png)

> Anomaly detection (learning from OK images only) is deliberately out of
> Seg-Studio's scope: it moves to **AnomaLens**, our companion project for
> industrial visual anomaly detection (publication upcoming).

---

## 📊 Evaluation & Visualization

| Feature | What it does | Tag |
|---|---|:---:|
| Metric suite | F1 / mIoU / Precision / Recall | Core |
| Heatmap | Confidence-coloured overlay | Useful |
| **CCA analysis** | Region counts and size histograms | Useful |
| **Pattern overlay** | Defect-only against the original scene | Useful |
| 🆒 **Live Inspect** | Camera-fed real-time inference | Flagship |
| Threshold slider | Instant confidence-threshold tuning | Useful |
| Batch export | Bulk prediction dump | Useful |

![Result view with metrics and prediction overlay](ja/assets/handbook/catalog_results.png)

---

## 📦 Export & Distribution

| Feature | What it does | Tag |
|---|---|:---:|
| ONNX export | For Python/C++ server inference | Core |
| CoreML export | For iOS/macOS app bundles | Core |
| 🆒 **CoreML Updatable** | On-device re-training on iOS | Flagship |
| OpenVINO export | FP32/FP16/INT8 for Intel CPU/iGPU/NPU | Useful |
| Python SDK | `pip install` and go | Core |
| REST API | `/v2/infer` for single-frame calls | Core |
| 🆒 **WebSocket inference** | Low-latency streaming for continuous frames | Flagship |

![Trained-model export dialog (ONNX / CoreML)](ja/assets/handbook/catalog_export.png)

---

## 🔧 UX / Operations

| Feature | What it does | Tag |
|---|---|:---:|
| Localized UI | Japanese + English | Core |
| Deep Zoom tiling | 8K images stay smooth (DZI) | Useful |
| Multi-project dashboard | Switch projects at a glance | Core |
| Size-adaptive uploads | Batch size auto-tuned per file | Useful |
| Hands-on onboarding | In-app interactive tour | Useful |

---

## Tag legend

| Tag | Meaning |
|:---:|---|
| Flagship | Major differentiators |
| Useful | Productivity boosters |
| Core | Industry-standard essentials |
| 🆒 | Highlight feature worth a demo |

---

## Related docs

- 📘 [Beginner's handbook](handbook.md) — 14-chapter linear walkthrough
- 📖 [Detailed user guide](user-guide.md)
- 🛠 [Troubleshooting](troubleshooting.md)
- 💻 [Developer quickstart](dev-quickstart.md)

---

_Seg-Studio Catalog v1.0 — 2026-04_
