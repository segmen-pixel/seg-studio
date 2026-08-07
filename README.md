<div align="center">

# Seg-Studio

**Train, annotate, and deploy image segmentation models -- all in one desktop app.**

![License](https://img.shields.io/badge/license-Apache_2.0-blue)
![Version](https://img.shields.io/badge/version-0.9.8-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/status-beta-yellow)

Seg-Studio is an open-source, all-in-one image segmentation workbench that runs entirely on your local machine. Annotate images with SAM-powered smart tools, train PyTorch models with real-time monitoring, and export to CoreML or ONNX for edge deployment -- no cloud account required. Alongside semantic segmentation it can count individual objects, reusing the masks you already drew rather than asking for new annotation.

[Quick Start](#quick-start) | [Features](#key-features) | [Japanese](README.ja.md)

</div>

<p align="center">
  <img src="docs/images/hero.gif" alt="Annotate with SAM, train, evaluate, and export — end to end in Seg-Studio" width="900" />
</p>

<p align="center"><sub>End-to-end in 44 s: annotate with SAM &rarr; train with auto-tuned recipe &rarr; inspect heatmap and report &rarr; export to ONNX / CoreML.</sub></p>

---

## Where to start

Three steps: **download the ZIP &rarr; run `install` then `start` &rarr; open
`http://localhost:8002/ui/`.** The exact commands are in
[Quick Start](#quick-start) below.

| If you are... | Go here |
|---|---|
| **New here** | [Quick Start](#quick-start) to install, then the [First Run Walkthrough](docs/first-run-manual.md) — the shortest single path, from opening the app to your first prediction in about 10 minutes |
| **Stuck installing or starting the server** | [Troubleshooting](docs/troubleshooting.md) |
| **Looking up one feature** | [User Guide](docs/user-guide.md) — reference for every tab, tool and setting |
| **After the full detail** | [Beginner's Handbook](docs/handbook.md) — the same workflow carried end to end on a sample dataset, 16 chapters |
| **Running Seg-Studio on a shared machine or LAN** | [Deployment](docs/deployment.md) — token sign-in, reverse proxy, backups |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md), plus the [Developer Quickstart](docs/dev-quickstart.md) |

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

The short version: annotate with brush, wand or SAM click; train a PyTorch
segmentation model on your own machine while loss and F1 update live; review
the predictions; export to ONNX or CoreML. It can also count objects, reusing
the masks you already drew.

Everything beyond that — DINOv2 distillation, Lovász-Softmax, CCA
post-processing, OpenVINO INT8, Perlin CutPaste, MLP Assist, the global
training queue — is optional, and is folded away below so the install steps
stay near the top. The one-page tagged version is the
[Feature Catalog](docs/catalog.md).

<details>
<summary><b>Full feature list</b> — click to expand</summary>

**Annotation**
- Brush, eraser, wand (flood fill), spot detect, crack trace, superpixel
- Move tool to drag-reposition mask regions
- Mark Clean flag for defect-free images
- SAM click segmentation with 5 model variants (MobileSAM, SAM2 Tiny/Small, TinySAM, EfficientSAM)
- MLP Assist for semi-automatic labeling
- Recipe-based auto-labeling pipeline

**Training**
- PyTorch training with real-time WebSocket monitoring (loss, F1, mIoU)
- Training mode selection: Standard / Quick / Transfer learning / Counting
- Object counting (instance segmentation): separates touching objects, trained from your existing masks via synthetic composition -- no extra annotation
- Tiled training and inference for counting, so small objects stay at capture resolution (training and inference always share one patch size)
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
- Count and area measurement in Results view (connected components), plus per-object counting with a trained counting model
- `POST /count` serving endpoint returning per-class counts and per-object boxes

**Interface**
- Bilingual UI (Japanese / English), switchable from the header

**Interactive onboarding**
- In-app hands-on tutorial with three modes (Beginner / Intermediate / Expert),
  a spotlight overlay, animated SVG illustrations, and full keyboard control.
  Replayable any time from the header ▶ button.
- Guided next-tab highlights and unseen-result pulses steer first-time users
  through each stage of the workflow.

</details>

---

## Quick Start

### 0. Install pre-flight — check these first

- **Disk space.** The installer's own guidance for the CUDA PyTorch step is
  **~5 GB free**, and that download alone is ~2.5 GB. Add ~300 MB if you opt
  into OpenVINO, plus the five SAM checkpoints. Your images, runs, and exported
  models then live in `projects/` inside the same folder, so extract it
  somewhere with room to grow.
- **Administrator rights: not needed** for the normal path. Everything is
  written inside the folder you extracted — the virtualenv (`.venv-windows` /
  `.venv-macos`), `models/`, `logs/`, and `projects/`. Elevation only comes up
  if you let the Windows installer fetch a missing Python, Node.js or git
  through `winget`, or if a permission error stops `npm install`.
- **Keep the path short on Windows.** Creating the virtualenv fails against the
  260-character path limit; the installer's advice is to move the folder
  somewhere like `C:\seg-studio`. If antivirus blocks the virtualenv instead,
  add that folder to your antivirus exclusions.
- **Python 3.10 or later must be installed first.** Both installers stop if they
  cannot find it. The Windows installer looks for 3.11 first, then 3.12, 3.13
  and 3.10: the dependency lockfile is compiled against 3.11, so that is the
  version every pinned package is guaranteed to have a prebuilt wheel for. As a last
  resort Windows tries `winget install Python.Python.3.11`; if the new Python is
  not on PATH yet the installer asks you to close the terminal and rerun. On
  macOS, install it yourself (`brew install python@3.11`).
- **Node.js 18+ is what builds the browser UI.** No pre-built UI bundle ships in
  the repository (`dist/` is not committed), so without `npm` the API starts but
  `http://localhost:8002/ui/` has nothing to serve. If `npm` is missing, the
  Windows installer tries to install Node.js 22 LTS via `winget`; on macOS it
  warns and skips the UI build. Pass `--skip-ui` if you deliberately only want
  the API.
- **git: required on macOS, installed for you on Windows.** `install-macos.sh`
  treats a missing `git` as a fatal prerequisite and stops (`brew install git`).
  `install-windows.bat` installs it through `winget` like Python and Node.js,
  and if that does not work it carries on and skips only the SAM assist
  libraries. Windows also uses `curl` (shipped with Windows 10 1803 and later)
  to download the SAM checkpoints.
- **Double-clicking is enough.** `install-windows.bat` and `start-windows.bat`
  are at the top of the extracted folder and hold the console open when you
  double-click them, so you can read the result. Set `SEG_NO_PAUSE=1` (or run
  under CI) if you are driving them from a script and want them to return
  immediately.
- **NVIDIA driver.** This repository does not pin a minimum driver version, so
  we will not quote one. What the installer actually checks is whether
  `nvidia-smi` runs: if it does you get the CUDA 12.8 wheels (Turing / RTX 20xx
  and newer, including Blackwell); if it does not, you get the CPU build. On
  Maxwell / Pascal / Volta run `install-windows.bat cuda124`. After installing,
  confirm with `python -c "import torch; print(torch.cuda.is_available())"` — it
  must print `True`.
- **Without the SAM weights, everything except SAM click assist still works.**
  Brush, eraser, wand, spot detect, crack trace, superpixel, MLP assist,
  training, evaluation, and every export are independent of them. The five
  checkpoints are downloaded during the Windows install, and otherwise on first
  use; each is checked against a SHA-256 recorded in the source.
- **Internet: needed to install, not to work.** Installing downloads PyTorch and
  the Python dependencies from PyPI, the SAM assist libraries from GitHub, the
  SAM checkpoints, and the UI's npm packages. After that, annotating, training,
  evaluating, and exporting all run on your machine. Two things still reach out
  on demand: a SAM checkpoint you have not downloaded yet, and the DINOv2
  distillation model definition (fetched via `torch.hub` the first time you
  enable it). For an air-gapped machine, build a bundle on a connected one
  first: `python scripts/install.py --offline-pack <dir>`.

### 1. Get the code — no git required

**Windows, with nothing to install first:** download
**Seg-Studio-v0.9.8-win64.zip** from the
[Releases page](https://github.com/segmen-pixel/seg-studio/releases), extract it
anywhere, and double-click `Seg-Studio.bat`. The package brings its own Python
and CUDA PyTorch, so there is no Python, git, or CUDA toolkit to install and
step 2 below does not apply to you. Those bundled runtimes are why it is a
~3 GB download.

**Every other platform, and Windows if you would rather build it yourself:**
Download the latest **Source code (zip)** from the
[Releases page](https://github.com/segmen-pixel/seg-studio/releases)
(or use the green **Code → Download ZIP** button on the repository page)
and extract it anywhere. If you prefer git:
`git clone https://github.com/segmen-pixel/seg-studio.git`

Every Release also carries `SHA256SUMS.txt`, so you can check a download before
running it:

```powershell
# Windows PowerShell
(Get-FileHash Seg-Studio-v0.9.8-win64.zip -Algorithm SHA256).Hash
```

```bash
# macOS / Linux
shasum -a 256 -c SHA256SUMS.txt --ignore-missing
```

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

Stop everything later with `stop-windows.bat` / `bash stop-macos.sh`.

> On Windows the installer fetches what it needs: if Python, Node.js or git is
> missing it installs it for you through winget, so a machine with none of the
> three still gets a working setup from one double-click. Without winget
> (some Windows 10 installs lack the App Installer package) it stops and tells
> you where to get Python; Node.js and git only degrade -- no Node.js means the
> UI is not rebuilt, no git means SAM click segmentation is unavailable.
> On macOS git is required up front: `install-macos.sh` stops without it.
> macOS uses MPS (Metal) automatically on Apple Silicon, for training as well as
> inference; object counting still needs an NVIDIA GPU. See
> [Platform support](#platform-support) for what runs where.

### 3. Open the UI

Open **http://localhost:8002/ui/** in your browser — that is the whole install.

New to segmentation? Continue with the
[First Run Walkthrough](docs/first-run-manual.md): from opening the app to your
first prediction in about 10 minutes.

### Alternative: Docker (docker compose)

The Linux route — CPU only, no training.

```bash
# Windows: use `python` instead of `python3`
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up --build
```

Then open **http://localhost:5173/** — the UI container's nginx proxies `/api`,
`/v2`, and `/ws` to the trainer API. All ports are published on `127.0.0.1` only.
The `.env` step is required, not optional: each container binds `0.0.0.0` inside
its own network namespace, and the trainer refuses to start off-loopback without
a token. The stack requests no GPU, so annotation, inference and the UI work but
training does not — details in [Deployment](docs/deployment.md#docker-optional).

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

All models support GroupNorm, JIT tracing, and configurable output stride.
Inference latency is single-image at 256×256, batch size 1. See
[BENCHMARKS.md](BENCHMARKS.md) for the full GPU + CPU benchmark and how to reproduce it.

The training default is **SimpleUNet** — small, stable, and a safe
starting point. Auto-config will suggest an architecture from your
dataset's profile, so you rarely have to choose by hand. On a 37-project
factory library STDC was the per-project best more often than SimpleUNet
(15 vs 5), so try STDC when you want more accuracy or the fastest
inference.

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

- **OS:** Windows 10 / 11 (64-bit), macOS 12+ (Apple Silicon recommended), or
  Linux via `docker compose` (the `.bat` / `.sh` installers cover Windows and
  macOS only)
- **Python:** 3.10+ (the dependency lockfile is compiled against 3.11)
- **Node.js:** 18+ (for the UI build only)
- **Disk:** ~5 GB free for the CUDA install — see "0. Install pre-flight" under
  Quick Start

### Platform support

| | Windows + NVIDIA | Apple Silicon (MPS) | CPU only |
|---|---|---|---|
| Annotate, SAM assist | Yes | Yes — 4 of the 5 SAM models (TinySAM is not installed on macOS) | Yes |
| Semantic segmentation training | Yes | Yes | Yes, much slower |
| Object counting (instance) training | Yes | No | No |
| ONNX export | Yes | Yes | Yes |
| ONNX inference | Yes (CUDA provider) | Yes (CPU provider) | Yes (CPU provider) |
| Core ML export | Only if `coremltools` is importable | Yes | Only if `coremltools` is importable |

- The training device selector defaults to `auto`, which picks CUDA, then MPS,
  then CPU. 4 GB+ VRAM is the recommended minimum for semantic segmentation
  training on NVIDIA.
- The Windows installer defaults to CUDA 12.8 PyTorch wheels (Turing /
  RTX 20xx and newer, including Blackwell RTX 5090); run
  `install-windows.bat cuda124` to use CUDA 12.4 wheels on older GPUs
  (Maxwell / Pascal / Volta).
- On MPS, mixed precision is disabled, so a run is slower than on a comparable
  NVIDIA card. MPS shares unified memory with the rest of the system — see
  [Troubleshooting](docs/troubleshooting.md) if a run runs out of memory.
- **Object counting (instance segmentation) training needs an NVIDIA GPU.** The
  VRAM auto-fit only runs on CUDA devices; measured on an RTX 3090, the `small`
  model needs 8 GiB at the default batch 8, auto-reducing to batch 4 (5.5 GiB)
  and batch 2 (3.5 GiB). Below 3.5 GiB it is unsupported.
- Core ML export requires `coremltools` (pinned at 8.3.0); without it the export
  endpoint returns HTTP 501. The macOS installer installs it explicitly.
- OpenVINO IR export is an opt-in extra of the Windows installer
  (`install-windows.bat --with-openvino`, ~300 MB). The macOS installer has no
  equivalent flag.
- There is no prebuilt macOS package, by choice. An unsigned `.app` would make
  every user clear it through Gatekeeper by hand, which is more friction than
  running `install-macos.sh`; Windows has no equivalent obstacle, so the Windows
  package is shipped prebuilt.

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

- 🚀 **[First Run Walkthrough](docs/first-run-manual.md)** — The shortest single path: from opening the app to your first prediction, about 10 minutes
- 📘 **[Beginner's Handbook](docs/handbook.md)** — 16-chapter worked tutorial on a sample dataset (images → model → inference → SDK)
- 📗 **[Feature Catalog](docs/catalog.md)** — One-page overview of every feature

### Reference

- [User Guide](docs/user-guide.md)
- [Developer Quickstart](docs/dev-quickstart.md)
- [Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Import / Export](docs/import_export.md)
- [OpenVINO Export](docs/openvino_export.md) — running an exported IR on Intel CPU / iGPU / NPU
- [Feature Catalog slides](docs/slides-catalog.md) — Marp source for the overview deck
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
