---
marp: true
theme: default
paginate: true
size: 16:9
header: "Seg-Studio Feature Catalog"
footer: "Seg-Studio v0.9.8 — 2026-07"
style: |
  section {
    font-family: "Inter", system-ui, sans-serif;
    background: #fafafa;
  }
  section.title {
    background: linear-gradient(135deg, #1565c0, #4fc3f7);
    color: white;
    justify-content: center;
    text-align: center;
  }
  section.title h1 { font-size: 64px; margin: 0; }
  section.title h2 { font-size: 24px; margin-top: 16px; opacity: 0.9; font-weight: normal; }
  h1 { color: #1565c0; border-bottom: 3px solid #4fc3f7; padding-bottom: 8px; }
  h2 { color: #1976d2; }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: bold;
  }
  .star { background: #ffca28; color: #424242; }
  .good { background: #66bb6a; color: white; }
  .std { background: #90a4ae; color: white; }
  table { font-size: 18px; }
  table th { background: #e3f2fd; }
---

<!-- _class: title -->

# Seg-Studio

## All-in-one Image Segmentation
### Annotate → Train → Evaluate → Ship, in one tool

---

# Why Seg-Studio?

| Feature | Seg-Studio | Typical alternatives |
|------|:---:|:---:|
| Browser-only workflow | ✅ | — |
| One-click SAM extraction | ✅ | partial |
| Perlin CutPaste synthesis | ✅ | rare |
| Outdoor lighting variants | ✅ | no |
| CoreML Updatable export | ✅ | no |
| Localized UI (JA / EN) | ✅ | rare |

<br>

**In short** — a full-stack GUI tailored for real-world inspection work, not just research demos.

---

# 🖊 Annotation

| Feature | What it does | Tag |
|---|---|:---:|
| Brush / Wand / Eraser | Core painting tools | <span class="badge std">Core</span> |
| 🆒 **SAM marking** | One click extracts the target (Meta SAM) | <span class="badge star">Flagship</span> |
| Crack trace | Click a crack, auto-traced candidates | <span class="badge good">Useful</span> |
| Spot detect | DoG-based point defect detection | <span class="badge good">Useful</span> |
| Superpixel selection | Click-once to grab whole regions | <span class="badge good">Useful</span> |
| **Mark Clean** | Tag OK images as training data | <span class="badge star">Flagship</span> |

<!-- Screenshot (Marp bg): ja/assets/handbook/catalog_annotate.png -->

---

# 🧪 Augmentation

| Feature | What it does | Tag |
|---|---|:---:|
| 🆒 **Perlin CutPaste** | Warp & paste existing defects elsewhere | <span class="badge star">Flagship</span> |
| 🆒 **Lighting variants** | Day / evening / night color re-grading | <span class="badge star">Flagship</span> |
| Auto-config | Suggests optimal settings from data stats | <span class="badge good">Useful</span> |
| Annotation Patches | Centroid-biased patch sampling | <span class="badge std">Core</span> |

<!-- Screenshot (Marp bg): ja/assets/handbook/catalog_augment.png -->

---

# 🎓 Training

| Feature | What it does | Tag |
|---|---|:---:|
| Training modes | Standard / Quick / Transfer / Counting | <span class="badge std">Core</span> |
| Local GPU training | Train on the same PC | <span class="badge std">Core</span> |
| Transfer learning | Warm-start from a past run's checkpoint you pick | <span class="badge good">Useful</span> |
| 🆒 **DINOv2 distillation** | Distill features from a 142M-image teacher | <span class="badge star">Flagship</span> |
| Lovász-Softmax loss | Selectable boundary-sensitive loss | <span class="badge good">Useful</span> |
| Multi-arch | SimpleUNet / STDC | <span class="badge std">Core</span> |
| Deep supervision | Auxiliary losses on intermediate layers | <span class="badge std">Core</span> |
| 🆒 **Object counting** | Count parts per image, trained from existing masks | <span class="badge star">Flagship</span> |
| Tiled training & inference | Small objects stay at capture resolution | <span class="badge good">Useful</span> |

<!-- Screenshot (Marp bg): ja/assets/handbook/catalog_training.png -->

---

# 📊 Evaluation & Visualization

| Feature | What it does | Tag |
|---|---|:---:|
| Metric suite | F1 / mIoU / Precision / Recall | <span class="badge std">Core</span> |
| Heatmap | Confidence-colored overlay | <span class="badge good">Useful</span> |
| **CCA analysis** | Region counts & size histograms | <span class="badge good">Useful</span> |
| **Pattern overlay** | Defect-only against the original scene | <span class="badge good">Useful</span> |
| 🆒 **Live Inspect** | Camera-fed real-time inference | <span class="badge star">Flagship</span> |
| Threshold slider | Instant confidence-threshold tuning | <span class="badge good">Useful</span> |
| Batch export | Bulk prediction dump | <span class="badge good">Useful</span> |

<!-- Screenshot (Marp bg): ja/assets/handbook/catalog_results.png -->

---

# 📦 Export & Distribution

| Feature | What it does | Tag |
|---|---|:---:|
| ONNX | For Python / C++ server inference | <span class="badge std">Core</span> |
| CoreML | For iOS / macOS app bundles | <span class="badge std">Core</span> |
| 🆒 **CoreML Updatable** | On-device re-training on iOS | <span class="badge star">Flagship</span> |
| OpenVINO | FP32/FP16/INT8 for Intel edge devices | <span class="badge good">Useful</span> |
| Python SDK | `pip install` and go | <span class="badge std">Core</span> |
| REST API | `/v2/infer` for single-frame calls | <span class="badge std">Core</span> |
| Counting API | `/count` returns per-class counts and per-object boxes | <span class="badge good">Useful</span> |
| 🆒 **WebSocket inference** | Low-latency streaming | <span class="badge star">Flagship</span> |

<!-- Screenshot (Marp bg): ja/assets/handbook/catalog_export.png -->

---

# The workflow in 5 steps

1. **Create a project** → define classes (e.g. scratch / stain)
2. **Drop images** → video-frame extraction also works
3. **Annotate** (SAM / Brush / Crack trace) → Mark Clean for OK samples
4. **Pick a mode → Train** (the dataset split runs automatically)
5. **Inspect Results** → Export (ONNX / CoreML / Updatable)

<!-- Screenshot (Marp bg): ja/assets/handbook/01_overview.png -->

**Full tour** — see 📘 `docs/handbook.md` (16 chapters, linear).

---

# Recommended use cases

| Use case | Features that shine |
|---|---|
| 🏭 Factory inspection (scratch / stain / misc.) | SAM / Perlin CutPaste / Live Inspect |
| 🌳 Outdoor field inspection (handheld) | Lighting / Mark Clean / BG bulk ingest |
| 📱 Mobile apps | CoreML Updatable / lightweight STDC |
| 🔬 Research prototypes | Multi-arch / Lovász / DINOv2 distillation |
| 🏢 In-house servers | REST / WebSocket SDK |

---

<!-- _class: title -->

# Start Here

## 📘 [Beginner's Handbook](handbook.md)
## 📗 [Feature Catalog](catalog.md)
## 🏠 [README](../README.md)

### Apache 2.0 / Copyright 2026 Contributors
