# Seg-Studio Handbook — Your First Run

This handbook walks a **complete beginner** through the full pipeline:
drop images in → annotate → train → export → run inference.
It's a 14-step linear tour, not a reference manual.

Rough budget:

- **⏱ Time**: 3–15 min per chapter
- **📸 Screenshots**: ~30 total
- **🎯 Goal**: End up with a working model and a Python client that calls it

> 💡 The running sample is the `SIM` project (classes = scratch / stain,
> 163 images). Substitute your own data and the flow is identical.

---

## Contents

1. [Welcome](#1-welcome) — What Seg-Studio is for
2. [Launch & UI Tour](#2-launch--ui-tour) — Tabs and panels
3. [Create a Project](#3-create-a-project) — From scratch
4. [Add Images](#4-add-images) — Drag & drop / video / zip
5. [Define Classes](#5-define-classes) — What you want to detect
6. [Annotate](#6-annotate) — 🆒 **SAM marking** and other power tools
7. [Mark Clean for OK images](#7-mark-clean-for-ok-images) — Teach "no defect"
8. [Augment Your Data](#8-augment-your-data) — 🆒 **Perlin CutPaste / Lighting**
9. [Prepare the Dataset](#9-prepare-the-dataset) — train / val / test split
10. [Choose Training Settings](#10-choose-training-settings) — Auto-config defaults
11. [Run Training](#11-run-training) — Watch it live
12. [Inspect the Results](#12-inspect-the-results) — Heatmaps / CCA / Live Inspect
13. [Export the Model](#13-export-the-model) — 🆒 **CoreML Updatable**
14. [Call it from the SDK](#14-call-it-from-the-sdk) — Python in 5 lines

---

## 1. Welcome

Seg-Studio is an **all-in-one GUI for image segmentation** —
collecting images, annotating them, training a model, evaluating it
and shipping it as ONNX / CoreML. No Python required to use it.

### Who it's for

- Anyone replacing manual visual inspection with ML
- Users who prefer GUIs over notebooks
- Teams that need business-ready outputs, not research demos

### What sets it apart

| Feature | Seg-Studio | Typical alternatives |
|------|:---:|:---:|
| Runs entirely in a browser | ✅ | |
| One-click SAM extraction | ✅ | partial |
| Perlin CutPaste synthesis | ✅ | rare |
| Outdoor Lighting variants | ✅ | no |
| CoreML Updatable export | ✅ | no |
| Japanese UI (EN switchable) | ✅ | rare |

![Seg-Studio tab overview](ja/assets/handbook/01_overview.png)
<!-- screenshot: Projects tab open; the full tab bar (Projects / Annotate / Training / Live Inspect) visible with at least the SIM project in the list. -->

> **Tip**: The tabs are ordered left-to-right along the workflow.
> When lost, go back one tab to the left.

---

## 2. Launch & UI Tour

### Start the app

```bash
# Windows
scripts\windows\start_local_windows.bat

# macOS
bash scripts/macos/start_local_macos.sh
```

The browser opens `http://localhost:8002/ui/` automatically once the API is ready.

<!-- 図（スクリーンショット未収録）: Landing screen [02_launch.png] -->
<!-- screenshot: Fresh launch state — browser just opened localhost:8002/ui/, no project selected, ⚙️ button visible top-right. -->

### Tabs

| Tab | Purpose |
|---|---|
| **Projects** | Project dashboard |
| **Annotate** | Draw masks on images |
| **Training** | Configure and run training |
| **Results tab** | Auto-opens per trained run; holds metrics / heatmap / CCA / Live Inspect views |
| **Live Inspect** | Camera-fed real-time inference |

> **Tip**: The top-right ⚙️ opens settings. Most defaults are fine.

---

## 3. Create a Project

A project bundles images, masks and training runs into one folder.

### Steps

1. Open the **Projects** tab
2. Type a name into the project-name field (e.g. `SIM`)
3. Click **Create Project**

![New project form](ja/assets/handbook/03_create_project.png)
<!-- screenshot: The project-name field filled with 'SIM', about to click Create Project. -->

### About the storage layout

- Each project gets a UUID, stored under `projects/{uuid}/`
- **Renaming is safe**; the UUID doesn't change

> **Gotcha**: Two projects can share a name. They'll still be
> independent on disk. Pick from the list to switch.

---

## 4. Add Images

### Option A: Drag & drop (recommended)

<!-- 図（スクリーンショット未収録）: Drag & drop onto the list [04a_drag_drop.png] -->
<!-- screenshot: Annotate tab with the image list empty or partially populated; ideally a folder drop cursor mid-drag, or the progress bar mid-upload. -->

- **Annotate** tab → drop a folder onto the left list
- From 1 to several thousand files is fine
- Batches automatically adapt to file size

> **Tip**: Batch sizes are computed from each file's bytes — no
> setting to tune. Places365 256px JPGs pack 100 per request;
> 4K photos pack 5 per request.

### Option B: Extract frames from a video

<!-- 図（スクリーンショット未収録）: Video frame extraction [04b_video_import.png] -->
<!-- screenshot: Video drop dialog showing the frame-interval input (extract every N frames). -->

- Drop a `.mp4` etc.
- Set "extract every N frames"
- Decoding happens server-side automatically

### Option C: ZIP import

For restoring a previously exported project.

---

## 5. Define Classes

Label design — decide what's being detected.

### Example (SIM project)

| ID | Name | Color |
|---:|---|---|
| 0 | background | black |
| 1 | scratch | red |
| 2 | stain | green |

![Class panel](ja/assets/handbook/05_classes.png)
<!-- screenshot: Class panel on the right of Annotate with SIM's three classes (background / scratch / stain) and their color swatches. -->

### Guidelines

- **ID 0 must stay "background"**. Don't change it
- Too many classes = explosive annotation load. **3–5 is the sweet spot**
- Merge visually similar defects. Over-splitting hurts more than under-splitting

---

## 6. Annotate

### 6-a. Brush / Wand basics

![Brush in action](ja/assets/handbook/06a_brush.png)
<!-- screenshot: Brush mid-stroke on a scratch defect — circular brush cursor on the defect, partial red fill in progress. -->

- **Brush** (`B`): drag to paint. `[` / `]` to resize
- **Wand** (`W`): click to select similar colors → fill
- **Eraser** (`E`): paint to remove

### 6-b. 🆒 SAM marking (flagship)

Meta's SAM is bundled. **One click extracts the target**.

<!-- 図（スクリーンショット未収録）: SAM result [06b_sam.png] -->
<!-- screenshot: SAM tool active after a left-click — positive-point marker (green +) and the proposed mask highlighted, pre-Enter preview. -->

#### Usage

1. Select the SAM tool (🪄 icon)
2. **Left-click** on a defect — positive point
3. **Right-click** on non-defect — negative point (optional)
4. Press Enter to commit

#### Why it matters

- 10-second initial load, then instant clicks
- Fine boundaries fall out automatically → 10× faster than brushing

### 6-c. Crack trace / Spot detect

<!-- 図（スクリーンショット未収録）: Crack & spot tools [06c_crack_spot.png] -->
<!-- screenshot: Crack trace result after two endpoint clicks, or spot detect highlighting point defects. -->

- **Crack trace** (`C`): click on a linear defect → candidates are auto-traced; click to select, Enter to commit
- **Spot detect** (`D`): DoG (Difference-of-Gaussians) blob detection for point defects — paint a sample, press **Detect**, tune sensitivity, Enter to commit
- **Superpixel** (`P`): regions with similar color, one click grabs a chunk

### 6-d. Save / Undo

- `Ctrl+Z` / `Ctrl+Y` for undo / redo
- Auto-save happens on tab switch or image change
- `Ctrl+S` saves explicitly

---

## 7. Mark Clean for OK images

Clean (defect-free) images are **very important** training signal.

### What Mark Clean does

Writes an **all-zero mask** for the image and flags it as "this is OK".
The network learns "here, background wins everywhere."

### Steps

1. Select OK images in the list (Shift-click for range)
2. Click **OK** at the top (or press `Shift+C`)

![Mark Clean button](ja/assets/handbook/07_mark_clean.png)
<!-- screenshot: Image list with several clean images multi-selected (blue highlight); the OK (Mark Clean) button is highlighted. -->

### Outdoor use case

Handheld outdoor cameras see wildly diverse backgrounds → many false
positives. Counter-measure: drop **general indoor/outdoor scene
images** (e.g. Places365) and Mark Clean them all. The model learns
"none of this is a defect."

> **Tip**: Keep Mark Clean at **30–50% of train** at most. Flooding
> it drops recall.

> **Gotcha**: Mark Clean is reversible via the **Clear OK** button
> next to OK.

---

## 8. Augment Your Data

### 8-a. 🆒 Perlin CutPaste

Cuts a random defect from an existing annotation, warps it with Perlin
noise, pastes it somewhere else.

<!-- 図（スクリーンショット未収録）: Perlin CutPaste dialog [08a_perlin.png] -->
<!-- screenshot: Augment dialog with Perlin CutPaste enabled; preview canvas shows a warped defect and the class dropdown is open. -->

#### How

1. Annotate tab → **Augment** button
2. Check **Perlin CutPaste**
3. Pick target class (or "All classes")
4. Set count and warp strength
5. **Generate** — new images appear in seconds

### 8-b. 🆒 Lighting variants (outdoor)

Same scene re-colored into **daytime / evening / night**. Masks are
preserved verbatim; only colors shift.

<!-- 図（スクリーンショット未収録）: Lighting variants [08b_lighting.png] -->
<!-- screenshot: Thumbnail gallery after generating Lighting variants — same scene in daytime / evening / night side by side. -->

#### How

1. In Augment dialog, enable **Lighting**
2. Pick any combination of day/evening/night
3. Can combine with Perlin CutPaste in the same run

> **Tip**: Outdoor datasets are usually small. This one trick can
> triple your effective training size.

---

## 9. Prepare the Dataset

### Role

Splits images into **train / val / test** and materializes the layout
training will consume.

### How it works now

You don't press anything — the split runs **automatically when you
press Start Train**. A "Preparing dataset..." placeholder appears in
the run list, followed by the resulting train / val counts.

To change the ratios (default val 15%, test 10%), open **⚙️ Settings**
top-right and adjust **Validation Ratio** / **Test Ratio**.

<!-- 図（スクリーンショット未収録）: Prepare report [09_prepare.png] -->
<!-- screenshot: Run list showing the "Preparing dataset..." placeholder and the resulting train / val counts. -->

### Stratified split (important)

- **Foreground images** and **Mark Clean images** are split independently
- Both end up in val / test at the right ratio
- Prevents the "val is all Mark Clean, F1 = 0" failure mode

---

## 10. Choose Training Settings

### Pick a training mode first

The Training tab starts with three mode buttons — **Standard** (annotate
& train), **Quick** (fast training on already-labeled images) and
**Transfer** (fine-tune an existing model).
**The Start Train button stays disabled until you pick one.** For this
walkthrough, pick **Standard**.

### Auto-config recommendations

<!-- 図（スクリーンショット未収録）: Auto-config panel [10a_auto_config.png] -->
<!-- screenshot: Auto-config recommendation panel with suggested arch / patch_size / base_channels badges. -->

Image count, defect size and pre-analysis give you recommended `arch`,
`patch_size`, `base_channels`. **When in doubt, take the suggestion.**

### Manual tuning (once you have a feel)

| Knob | Guide |
|---|---|
| arch | deeplabv3plus (default, best on 17/37 projects) / stdc (best on 15/37) / simpleunet (small model, best on 5/37) |
| patch_size | 256 (best on 37/37 library projects; 128/512 never won a per-project comparison) |
| output_stride | 2 or 4 (lower = finer boundaries) |
| epochs | 50–100, early stopping handles the rest |
| transfer learning | **On recommended** — seeds from prior runs |

![Training form](ja/assets/handbook/10b_training_form.png)
<!-- screenshot: Full hyperparameter form in Training tab — epochs, loss_type, transfer learning toggle all visible. -->

---

## 11. Run Training

### Local GPU

<!-- 図（スクリーンショット未収録）: Local training live [11a_local_training.png] -->
<!-- screenshot: Local training in progress — loss / F1 / mIoU line chart updating live with a bottom progress bar. -->

- **Start Train** → training runs on the NVIDIA CUDA GPU (required for training)
- loss / F1 / mIoU update live
- Per-epoch checkpoints are saved automatically

---

## 12. Inspect the Results

### Metrics

<!-- 図（スクリーンショット未収録）: Metrics on Results [12a_metrics.png] -->
<!-- screenshot: Results tab metric header — F1 / mIoU / Precision / Recall rendered as large badges. -->

- **F1**: the headline number — 0.8+ is practical
- **mIoU**: lower than F1 hints at loose boundaries
- **Precision / Recall**: diagnose FP vs FN trade-offs

### Visualizations

| Feature | When to use |
|---|---|
| **Heatmap** | See where confidence is high vs borderline |
| **CCA (connected components)** | Region counts and size histograms |
| **Pattern overlay** | Isolate defects against the scene |
| **Live Inspect** | Feed a camera for real-time inference |

<!-- 図（スクリーンショット未収録）: Heatmap [12b_heatmap.png] -->
<!-- screenshot: One image open in Results with heatmap overlay on; high-confidence regions glow red/yellow. -->
<!-- 図（スクリーンショット未収録）: Live Inspect [12c_live_inspection.png] -->
<!-- screenshot: Live Inspect tab with webcam feed + real-time mask overlay; fps counter visible. -->

### Tune the threshold

The right-hand slider lets you shift the **inference confidence threshold**.
FP/FN trade-offs respond instantly.

---

## 13. Export the Model

### Formats

| Format | Use for |
|---|---|
| **ONNX** | Server-side inference, Python/C++ |
| **CoreML** | Bundling into iOS/macOS apps |
| **🆒 CoreML Updatable** | **On-device re-training** on iOS |
| **OpenVINO** (FP32/FP16/INT8) | Intel CPU / iGPU / NPU edge inference |

![Export menu](ja/assets/handbook/13_export.png)
<!-- screenshot: Export menu for a trained run showing ONNX / CoreML / CoreML Updatable buttons. -->

### Steps

1. In the Training tab, pick a completed run (or use **Export Model** in its Results tab)
2. Click **Export** → choose format
3. Download the `.onnx` / `.mlmodel`

> **Tip**: CoreML Updatable lets users fine-tune just the final Conv
> layer on-device — perfect for apps that personalize.

---

## 14. Call it from the SDK

A Python client, `seg-sdk`, ships with the repo.

### Install

```bash
pip install ./packages/seg-sdk
```

### Minimal (5 lines)

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")
result = client.predict(open("frame.jpg", "rb").read())
print(result.judgement, "| centroids:", [r.centroid for r in result.regions])
```

### What you get

| Field | Content |
|---|---|
| `judgement` | `"OK"` / `"NG"` |
| `regions[i].bbox` | Bounding box `(x, y, w, h)` |
| `regions[i].centroid` | Centroid `(cx, cy)` — great for robot pick |
| `summary.fg_ratio` | Fraction of pixels flagged as defect |
| `latency_ms.total` | Server processing time |

Full docs: [`packages/seg-sdk/README.md`](../packages/seg-sdk/README.md).

<!-- 図（スクリーンショット未収録）: SDK output [14_sdk_output.png] -->
<!-- screenshot: Terminal after running quick_start.py — judgement, region count, centroid and latency printed. -->

---

## That's it 🎉

You've just walked the full path from raw images to a working model
callable over HTTP. Next:

- Re-run the same loop on your own business images
- Skim the [catalog](catalog.md) for every other feature
- If FPs are your pain point, lean on Mark Clean with diverse BG data
- For the last few points of F1, leave Auto-config and hand-tune

### Stuck?

- [Troubleshooting](troubleshooting.md)
- [Detailed user guide](user-guide.md)
- [Developer quickstart](dev-quickstart.md)

---

_Seg-Studio Handbook v1.0 — 2026-04_
