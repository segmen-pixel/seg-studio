# Seg-Studio Handbook — Your First Run

> **Which document is this?** The long-form tutorial: one sample dataset
> carried in order from raw images through annotation, training, export and
> inference, advanced features included. For the quickest route to a first
> result use the [First Run Walkthrough](first-run-manual.md); to look up a
> single feature use the [User Guide](user-guide.md).

This handbook walks a **complete beginner** through the full pipeline:
drop images in → annotate → train → export → run inference.
It's a 16-chapter linear tour, not a reference manual. Chapters 1–14 are the
core path from images to inference; 15 and 16 are an optional add-on covering
object counting.

Rough budget:

- **⏱ Time**: 3–15 min per chapter
- **📸 Screenshots**: 7 so far — the rest are listed in `contributing/screenshots.md`
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
15. [Count Objects (Instance Mode)](#15-count-objects-instance-mode) — 🆕 Exact counting with zero instance labels
16. [The counting API](#16-the-counting-api-post-count) — 🆕 `POST /count`: counts, centroids, RLE

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

### About the storage layout

- Each project gets a UUID, stored under `projects/{uuid}/`
- **Renaming is safe**; the UUID doesn't change

> **Gotcha**: Two projects can share a name. They'll still be
> independent on disk. Pick from the list to switch.

---

## 4. Add Images

### Option A: Drag & drop (recommended)

- **Annotate** tab → drop a folder onto the left list
- From 1 to several thousand files is fine
- Batches automatically adapt to file size

> **Tip**: Batch sizes are computed from each file's bytes — no
> setting to tune. Places365 256px JPGs pack 100 per request;
> 4K photos pack 5 per request.

### Option B: Extract frames from a video

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

### Guidelines

- **ID 0 must stay "background"**. Don't change it
- Too many classes = explosive annotation load. **3–5 is the sweet spot**
- Merge visually similar defects. Over-splitting hurts more than under-splitting

---

## 6. Annotate

### 6-a. Brush / Wand basics

![Brush in action](ja/assets/handbook/06a_brush.png)

- **Brush** (`B`): drag to paint. `[` / `]` to resize
- **Wand** (`W`): click to select similar colors → fill
- **Eraser** (`E`): paint to remove

### 6-b. 🆒 SAM marking (flagship)

Meta's SAM is bundled. **One click extracts the target**.

#### Usage

1. Select the SAM tool (🪄 icon)
2. **Left-click** on a defect — positive point
3. **Right-click** on non-defect — negative point. This is **off by default**:
   tick **Enable SAM exclude-click (right click)** in the top-right ⚙️ Settings
   dialog first, otherwise right-click is ignored
4. Press Enter to commit

#### Why it matters

- 10-second initial load, then instant clicks
- Fine boundaries fall out automatically → 10× faster than brushing

### 6-c. Crack trace / Spot detect

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

#### How

1. Annotate tab → **Augment** button
2. Check **Perlin CutPaste**
3. Pick target class (or "All classes")
4. Set count and warp strength
5. **Generate** — new images appear in seconds

### 8-b. 🆒 Lighting variants (outdoor)

Same scene re-colored into **daytime / evening / night**. Masks are
preserved verbatim; only colors shift.

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

### Prepared images are copied losslessly (PNG by default)

Dataset preparation writes a training copy of every image into
`prepared/images/` as **PNG**, byte for byte identical to the source. Nothing
is re-encoded, so the pixels training sees are the pixels you captured. The
prepared folder is correspondingly large.

**Trading fidelity for disk and decode speed.** JPEG quality 95 is available as
an opt-in, for projects where the prepared folder does not fit or where image
decoding is starving the GPU. Set:

```
SEG_PREPARED_IMAGE_FORMAT=jpeg
```

and re-run dataset preparation. Copies are then written as q95 JPEG with
**4:4:4 chroma** (no colour subsampling), which lands at roughly **3.6x smaller
than PNG** on real 20 MPix inspection images — the 4:2:0 encoder that used to be
the default measured 5.9x there, and 4:4:4 roughly doubles the bytes.

**What the JPEG option costs.** On those same images the mean absolute error is
about **2.7 / 255** inside the defect region — and about **2.7 in the background
too**, so the loss is uniform quantisation noise rather than something that
erases defects selectively. But that measurement came from images whose defects
covered ~42% of the frame. If your defects are one or two pixels wide, or very
low contrast, their signal-to-quantisation ratio is far worse — and **the
reported score will not tell you**, because training and evaluation both read
the re-encoded copies while inference reads your original files. Writing 4:4:4
is what keeps colour-only defects intact: at libjpeg's 4:2:0 default a one-pixel
colour line lost its contrast **16.25 → 5.21**, against 16.25 → 16.21 at 4:4:4.

The format in force is recorded in `prepared/report.json` as
`prepared_image_format`.

### Stratified split (important)

- **Foreground images** and **Mark Clean images** are split independently
- Both end up in val / test at the right ratio
- Prevents the "val is all Mark Clean, F1 = 0" failure mode

> **The split has no notion of groups.** Items are ranked by SHA1 of the image
> filename, which is effectively random with respect to capture session, lot or
> workpiece. Burst frames of one part, or the same part shot from several
> angles, can therefore land on both sides of the split — and a near-duplicate
> straddling train and val makes the val score look better than the model is.
> If your data contains such groups, set each image's **train / test**
> assignment yourself to keep a whole group on one side; the prepared
> `report.json` records how many you pinned (`manual_train_count`,
> `manual_test_count`) alongside `split_grouping: "none"`.

---

## 10. Choose Training Settings

### Pick a training mode first

The Training tab starts with four mode buttons — **Standard** (annotate
& train), **Quick** (fast training on already-labeled images),
**Transfer** (fine-tune an existing model) and **Instance (count)**
(count objects from ordinary semantic masks — section 15).
**The Start Train button stays disabled until you pick one.** For this
walkthrough, pick **Standard**.

### Auto-config recommendations

Image count, defect size and pre-analysis give you recommended `arch`,
`patch_size`, `base_channels`. **When in doubt, take the suggestion.**

### Manual tuning (once you have a feel)

| Knob | Guide |
|---|---|
| arch | simpleunet (default, small and fast) / stdc (best on 15/37 library projects) |
| patch_size | 256 (best on 37/37 library projects; 128/512 never won a per-project comparison) |
| output_stride | 2 or 4 (lower = finer boundaries) |
| epochs | 50–100, early stopping handles the rest |
| transfer learning | **On recommended** — seeds from prior runs |

![Training form](ja/assets/handbook/10b_training_form.png)

---

## 11. Run Training

### Local GPU

- **Start Train** → training runs on the NVIDIA CUDA GPU (required for training)
- loss / F1 / mIoU update live
- Per-epoch checkpoints are saved automatically

---

## 12. Inspect the Results

### Metrics

- **F1**: the headline number — compare it across runs on the same dataset; there is no universal pass mark
- **mIoU**: lower than F1 hints at loose boundaries
- **Precision / Recall**: diagnose FP vs FN trade-offs

### Visualizations

| Feature | When to use |
|---|---|
| **Heatmap** | See where confidence is high vs borderline |
| **CCA (connected components)** | Region counts and size histograms |
| **Pattern overlay** | Isolate defects against the scene |
| **Live Inspect** | Feed a camera for real-time inference |

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

### Steps

1. In the Training tab, pick a completed run and click **Results** to open its Result tab
2. Click **Export Model**, choose ONNX / CoreML / CoreML (Updatable), and confirm
3. Download the `.onnx` / `.mlmodel`

> **OpenVINO is the exception**: export it from the OpenVINO menu on the run in
> the Training tab, which offers FP32 / FP16 / INT8.

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

---

## 15. Count Objects (Instance Mode)

Need to know **how many** screws / parts / tablets are in the image, even when
they touch or overlap? Instance mode trains an RF-DETR-Seg counting model
**from the semantic masks you already have** — no per-instance annotation,
ever.

### How it works

Your painted masks are cut into per-object patches, and a synthetic training
set is composed from them: copy-paste with exact per-instance ground truth,
including coaxial touching pairs, the case counting models usually miss.
Validation always uses source images that never fed the training composition,
and the count threshold is calibrated afterwards on the checkpoint that is
actually exported.

**Every active class is counted.** You do not pick one — the model learns all
of them and reports a count per class.

### Steps

1. Annotate semantic masks as usual (section 6). **Four images** containing
   the target class is the enforced minimum; more is better, and the
   preview strip below will tell you whether you have enough.
2. In **Training**, pick the **個数カウント / Count** mode card.
3. Open the **Settings** disclosure in the Training tab (not the top-right
   ⚙️ Settings dialog from section 9) if you want to adjust synthesis (objects
   per image, stack-pair probability, area band — `0` = auto) and press
   **Generate** in the preview strip to sanity-check a few composed samples.
4. Start training. **Small** is the default, **Medium** is noticeably more
   accurate, and **Large** is selectable as a third size. Smaller GPUs are
   handled automatically for Small and Medium — the batch halves and gradient
   accumulation doubles to keep the effective batch, with about 3.5 GB the
   floor for Small. **Large has no measured VRAM table**, so it is not
   auto-fitted: the run uses the batch you asked for unchecked and may OOM.
   Lower it by hand if it does.

   | Size | batch 8 | batch 4 | batch 2 |
   |------|---------|---------|---------|
   | Small | 8.0 GB | 5.5 GB | 3.5 GB |
   | Medium | 16.0 GB | 9.5 GB | 6.0 GB |

5. In **Results**, run inference. The overlay numbers each instance, and the
   header shows the count per class. **検出ハイライト / Detection highlight**
   switches to a view that greys the background and gives every object its own
   colour, which is the quickest way to see whether two touching parts were
   counted as one.
6. **Export → ONNX (Serving)** registers the model; after activation,
   `POST /count` on the serving API returns per-image counts (see section 16).

### Large images are tiled automatically

The detector has a fixed input, so a whole photo has to be shrunk to reach it —
and a 110px screw in a 2560px frame arrives at 18px, which is most of it gone.
So training composes **768×768 patches at capture resolution** and inference
runs the same 768 patches across the image, stitching the detections back
together.

Measured on a real 2560×2048 project, counting four photos against their
annotation (40 objects in each):

| | tiles | counts | mean error | time |
|---|---|---|---|---|
| patch 384 | 63 | 36, 36, 36, 36 | 4.0 | 1.6 s |
| **patch 768** (default) | 20 | 40, 40, 40, 40 | **0.0** | **0.5 s** |

The truth here is 40 per photo, not the 39, 39, 40, 39 an earlier version of
this table reported: three of the photos contain a pair of screws lying against
each other, which the annotation records as one region. Counting regions
undercounts those images by one each.

**This needs nothing from you.** Images smaller than 768 are handled as a
single padded patch — exactly what they did before tiling existed — so a
project of 512px photos behaves as it always has. Only larger frames tile.

If you do want to change it, `instance_patch_size` accepts a pixel size
(0-4096), or `0` to turn tiling off — the default 768 is already twice the
Small model's 384 input. Whatever you choose is recorded in the exported
contract, because **training and inference must use the same patch size** — a
mismatch does not raise, it just makes the count wrong.

### Known limits

- **Synthesis-first works best for separable rigid parts** (screws, pins,
  stamped parts). Heavily deformable or densely piled objects may need more
  real annotated images. Watch the preview strip: if the composed samples do
  not look like your real scenes, adjust the area band or annotate more.
- **The query budget** caps how many instances one forward pass can emit,
  across all classes combined. It is whatever the exported model carries, not a
  fixed 200, and it bounds the whole frame only when tiling is off
  (`instance_patch_size` 0) — with the default tiling every tile gets its own
  budget. On a non-tiled model, past 90% of the budget the response carries a
  truncation warning rather than a confident number — split the frame or use a
  tighter crop.
- Objects **wider than the tile overlap** can straddle a seam. At the default
  768 patch the overlap is 192px, so anything under that is safe; larger
  objects want a larger patch.

---

## 16. The counting API (`POST /count`)

Once a counting model is activated in the serving registry, `POST /count` takes
one image and returns what it found. Runs on CPU; no GPU required.

```bash
curl -F "image=@tray.png" http://localhost:8001/count
```

```json
{
  "model_id": "b42d4695-...",
  "count": 40,
  "counts_by_class": {"1": 38, "2": 2},
  "class_names": {"1": "screw", "2": "washer"},
  "threshold": 0.3,
  "dedup_iou": 0.7,
  "image_size": [2560, 2048],
  "inference_time_sec": 0.52,
  "instances": [
    {
      "id": 1,
      "class_id": 1,
      "class_name": "screw",
      "conf": 0.9812,
      "bbox": [412, 233, 118, 121],
      "centroid": [470.4, 293.1],
      "area": 8842,
      "rle": {"size": [2048, 2560], "counts": []}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `count` | Total instances across every class |
| `counts_by_class` | Per class, keyed by the project's class id |
| `bbox` | `[x, y, width, height]` in original image pixels |
| `centroid` | **Area centroid** of the mask (first moment), not the box centre - this is the one to use for pick-and-place |
| `area` | Mask pixels |
| `rle` | COCO uncompressed RLE, column-major, the same encoding the trainer writes |
| `threshold`, `dedup_iou` | What the model was calibrated with, echoed so a caller can record it |

### When the count may be short

A detector can only emit as many detections per forward pass as it has queries,
shared across all classes. That budget is not fixed at 200 - serving reads it
from the activated model's export and reports whatever that model carries. Near
the ceiling the response gains:

```json
"truncation_warning": "191 instances is close to this model's per-image limit of 200; the real count may be higher",
"max_instances_per_image": 200
```

Treat that as "at least 191", not "191". Split the frame or crop tighter.

**Only whole-frame models produce this.** A patch-trained model - the default,
since `instance_patch_size` is 768 - tiles the image and gives each tile its own
query budget, so there is no per-frame ceiling to report and neither key is
returned. Train with `instance_patch_size` 0 if you want the whole-frame
behaviour these keys describe.

### When the threshold was never measured

The confidence threshold is chosen at training time by counting your annotated
validation photos at each candidate value. That needs at least one annotated
image in the validation split; without one the model ships the default instead,
and the response says so:

```json
"threshold_warning": "threshold 0.3 was not calibrated: the training run had no annotated validation image to measure against, so this is the default. Counts may be systematically high or low."
```

The usual cause is that every annotated image had a region outside the
single-object area band. Annotate one more image, keeping each object as its
own region, and retrain.

### Errors

| Status | Meaning |
|---|---|
| 503 | No model activated |
| 409 | The active model is a semantic-segmentation export - use `/segment` |

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
