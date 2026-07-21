<div align="center">

# Seg-Studio

**Train, annotate, and deploy semantic segmentation models -- all in one desktop app.**

![License](https://img.shields.io/badge/license-Apache_2.0-blue)
![Version](https://img.shields.io/badge/version-0.9.7-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/status-beta-yellow)

Seg-Studio is an open-source, all-in-one semantic segmentation workbench that runs entirely on your local machine. Annotate images with SAM-powered smart tools, train PyTorch models with real-time monitoring, and export to CoreML or ONNX for edge deployment -- no cloud account required.

[Quick Start](#quick-start) | [Features](#key-features) | [Japanese](README.ja.md)

</div>

<p align="center">
  <img src="docs/images/hero.gif" alt="Annotate with SAM, train, evaluate, and export — end to end in Seg-Studio" width="900" />
</p>

<p align="center"><sub>End-to-end in 44 s: annotate with SAM &rarr; train with auto-tuned recipe &rarr; inspect heatmap and report &rarr; export to ONNX / CoreML.</sub></p>

---

## Screenshots

<table>
  <tr>
    <td align="center">
      <img src="docs/images/screenshot_projects.png" width="400" /><br />
      <b>Projects</b> -- Manage datasets and runs
    </td>
    <td align="center">
      <img src="docs/images/screenshot_annotate.png" width="400" /><br />
      <b>Annotate</b> -- Brush, Wand, SAM click, and more
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/screenshot_training.png" width="400" /><br />
      <b>Training</b> -- Real-time loss and F1 curves
    </td>
    <td align="center">
      <img src="docs/images/screenshot_results.png" width="400" /><br />
      <b>Results</b> -- Predictions, heatmaps, and export
    </td>
  </tr>
</table>

---

## Why Seg-Studio?

| | **Seg-Studio** | LabelMe | CVAT | Label Studio |
|---|:---:|:---:|:---:|:---:|
| Annotation tools | Yes | Yes | Yes | Yes |
| SAM click segmentation | 5 models built-in | No | Built-in | Via ML Backend |
| Model training (built-in) | Yes | No | No | Via ML Backend |
| Real-time training monitor | Yes | N/A | N/A | N/A |
| CoreML / ONNX export | Yes | N/A | N/A | N/A |
| Single GPU, no cloud needed | Yes | Yes | Docker | Docker or Cloud |
| Auto-tuning (loss, LR, weights) | Yes | N/A | N/A | N/A |

---

## Key Features

**Annotation**
- Brush, polygon, wand (flood fill), spot detect, crack trace, superpixel, and Perlin-based ridge detection
- Move tool to drag-reposition mask regions
- Mark Clean flag for defect-free images
- SAM click segmentation with 5 model variants (MobileSAM, SAM2 Tiny/Small, TinySAM, EfficientSAM)
- MLP Assist for semi-automatic labeling
- Recipe-based auto-labeling pipeline

**Training**
- PyTorch training with real-time WebSocket monitoring (loss, F1, mIoU)
- Training mode selection: Standard / Quick / Transfer learning
- Auto-config v2: recommends architecture, patch size, and base channels from project statistics
- Centroid-based annotation patch sampling
- Lovász-Softmax loss
- Auto-tuning of loss function, learning rate, and class weights
- Sliding window validation for high-resolution images
- Knowledge distillation support (DINOv2 feature distillation — the Apache-2.0 weight ships with the installer; the model-definition source is fetched at runtime via `torch.hub` on your machine, not bundled)
- Global training queue with auto-launch for multi-project workflows

**Results & Deploy**
- GT / prediction mask separation with overlay patterns (hatching, dots, grid)
- Pixel-level confidence heatmap, histogram, and per-image scoring
- Post-processing CCA (connected components, minimum-area filter)
- Live inspection mode and batch export across projects
- Evaluation report generation (HTML / PDF / Excel)
- CoreML export for iPad / iPhone deployment
- ONNX export for cross-platform inference
- OpenVINO IR export with optional INT8 quantization (opt-in install: `--with-openvino`)
- Count and area measurement in Results view

**Interface**
- Bilingual UI (Japanese / English), switchable from the header

**Interactive onboarding**
- In-app hands-on tutorial with three modes (Beginner / Intermediate / Expert),
  a spotlight overlay, animated SVG illustrations, and full keyboard control.
  Replayable any time from the header ▶ button.
- Guided next-tab highlights and unseen-result pulses steer first-time users
  through each stage of the workflow.

---

## Quick Start

### 1. Get the code — no git required

Download the latest **Source code (zip)** from the
[Releases page](https://github.com/segmen-pixel/seg-studio/releases)
(or use the green **Code → Download ZIP** button on the repository page)
and extract it anywhere. If you prefer git:
`git clone https://github.com/segmen-pixel/seg-studio.git`

### 2. Install and start

**Windows (NVIDIA GPU):**
```bash
install-windows.bat
start-windows.bat
```

No terminal needed: double-click `install-windows.bat`, then
`start-windows.bat`, right in the extracted folder. The installer auto-detects your GPU (pass `cpu` or
`cuda124` to override) and prints step-by-step guidance if Python 3.10+ is
missing. The start script opens the UI in your browser once the server is
ready.

**macOS (Apple Silicon / Intel):**
```bash
bash install-macos.sh
bash start-macos.sh
```

Then open **http://localhost:8002/ui/** in your browser.
Stop everything later with `stop-windows.bat` / `bash stop-macos.sh`.

> git is optional: the installer only uses it to fetch the SAM assist
> libraries. Without git, everything except SAM click segmentation works.
> macOS: MPS (Metal) is used automatically for inference on Apple Silicon. Training requires an NVIDIA CUDA GPU (Windows).

**Docker (docker compose):**
```bash
docker compose up --build
```

Then open **http://localhost:5173/** — the UI container's nginx proxies `/api`,
`/v2`, and `/ws` to the trainer API. All ports are published on `127.0.0.1` only.

---

## Workflow

```
Projects  -->  Annotate  -->  Train  -->  Results  -->  Deploy
   |              |             |            |            |
 Create or    Label with    Configure    Evaluate     Export to
 import       SAM, brush,   and run      predictions  CoreML or
 dataset      wand, MLP     training     and compare  ONNX
```

---

## Architecture Selection

| Architecture | Params | Model size | Inference (RTX 3090) | Strengths |
|---|---:|---:|---:|---|
| **SimpleUNet** (bc=64) | 1.9 M | 7.3 MB | 2.9 ms · 339 img/s | Stable, high F1, GroupNorm + SE attention |
| **STDC** (bc=32) | 2.9 M | 11.2 MB | 1.3 ms · 758 img/s | Lightweight, fastest inference |
| **DeepLabV3+** (bc=32) | 4.8 M | 18.5 MB | 5.1 ms · 198 img/s | ASPP + MobileNetV3 encoder, large receptive field |

All models support GroupNorm, JIT tracing, and configurable output stride.
Inference latency is single-image at 256×256, batch size 1. See
[BENCHMARKS.md](BENCHMARKS.md) for the full GPU + CPU benchmark and how to reproduce it.

The training default is **DeepLabV3+** — on a 37-project factory
library it was the per-project best architecture most often (17/37,
vs 15 for STDC and 5 for SimpleUNet). Pick STDC for the fastest
inference and SimpleUNet for the smallest memory footprint.

---

## MCP Bridge

Connect Seg-Studio to MCP-compatible tools for programmatic project inspection:

```bash
pip install fastmcp httpx
python scripts/mcp_server.py --api http://localhost:8002 --policy read
```

37 tools across dataset, annotation, training, prediction, export, and system categories. Policy levels: `read`, `write`, `full`.

---

## Requirements

- **OS:** Windows 10 / 11 (64-bit) or macOS 12+ (Apple Silicon recommended)
- **GPU (training):** NVIDIA GPU with CUDA required — Windows (4 GB+ VRAM recommended). Apple Silicon (MPS) / CPU can annotate and run inference only.
  - The Windows installer defaults to CUDA 12.8 PyTorch wheels (Turing /
    RTX 20xx and newer, including Blackwell RTX 5090); run
    `install_windows.bat cuda124` to use CUDA 12.4 wheels on older GPUs
    (Maxwell / Pascal / Volta).
  - A parallel pinned lockfile `requirements-cu128.txt` is provided for
    Blackwell (sm_120) environments — see
    [`docs/BLACKWELL_MIGRATION.md`](docs/BLACKWELL_MIGRATION.md).
- **Python:** 3.10+
- **Node.js:** 18+ (for UI build only)

---

## Project Structure

```
seg-studio/
  apps/
    trainer_api/     # FastAPI backend
    serving_api/     # ONNX inference API
    trainer_ui/      # React frontend
  packages/
    segcore/         # Training core (models, dataset, train loop)
    seg-sdk/         # Python client SDK for the inference API
  models/
    sam_checkpoints/ # SAM model checkpoints
  scripts/
    windows/         # Windows setup/start scripts
    macos/           # macOS setup/start scripts
```

---

## Community

- **Contributing** -- Pull requests are welcome. Please open an issue first for major changes.
- **Discussions** -- Use [GitHub Discussions](https://github.com/segmen-pixel/seg-studio/discussions) for questions and ideas.
- **Security** -- Report vulnerabilities privately via [GitHub Security Advisories](https://github.com/segmen-pixel/seg-studio/security/advisories).

---

## Documentation

### Getting started

- 📘 **[Beginner's Handbook](docs/handbook.md)** — 14-chapter linear walkthrough (images → model → inference)
- 📗 **[Feature Catalog](docs/catalog.md)** — One-page overview of every feature

### Reference

- [User Guide](docs/user-guide.md)
- [Developer Quickstart](docs/dev-quickstart.md)
- [Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Import / Export](docs/import_export.md)
- [Roadmap](docs/ROADMAP.md)
- [API Reference](http://localhost:8002/docs) (available when the server is running)

For Japanese documentation, see [README.ja.md](README.ja.md).

---

<div align="center">

Copyright 2026 Segmen-Pixel and Seg-Studio contributors.
Licensed under the [Apache License 2.0](LICENSE).

Third-party licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) /
Upstream attributions: [NOTICE](NOTICE) /
Per-release SBOMs (CycloneDX + SPDX) attached to each
[GitHub Release](https://github.com/segmen-pixel/seg-studio/releases)

</div>

---

## Disclaimer

This software and the bundled or referenced pretrained models (SAM family,
DINOv2, etc.) are distributed on an "AS IS" basis under
[Apache License 2.0](LICENSE), Section 7. The authors and contributors make no
warranty regarding the accuracy of inference results or the relationship of
those results to any third-party rights. Users are responsible for their own
validation when applying the software to industrial or safety-critical
workflows.

All trademarks referenced (Apple, CoreML, PyTorch, NVIDIA, CUDA, ONNX, SAM,
DINOv2, etc.) are the property of their respective owners — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the complete attribution.
