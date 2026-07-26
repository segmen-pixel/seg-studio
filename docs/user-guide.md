# Seg-Studio User Guide

> **Which document is this?** The feature reference — every tab, tool and
> setting, written to be dipped into rather than read straight through. Part 1
> below is a condensed overview of the workflow; to be walked through it step
> by step instead, use the [First Run Walkthrough](first-run-manual.md), or the
> [Handbook](handbook.md) for the same ground at full length on a sample
> dataset.

Seg-Studio is a visual inspection tool that lets you teach an AI to recognize regions in images. You label areas of interest on your inspection images (scratches, defects, parts), train a segmentation model, review the results, and export the trained model for deployment on edge devices including iPad and iPhone.

---

## Part 1: Getting Started (10-Minute Quick Start)

### The Workflow at a Glance

Every project follows six steps. On a machine with an NVIDIA GPU you can complete a basic run-through in under 10 minutes.

**Step 1 — Create a Project.** Open the **Projects** tab, type a descriptive name such as "Gear Defect" or "PCB Scratch Detection", and click **Create Project**. One project per inspection target is recommended.

**Step 2 — Upload Images.** Drag and drop image files (JPEG, PNG, BMP, TIFF, WebP) directly onto the browser window, or upload a ZIP file containing multiple images. You need at least a few images to get started; 20-50 is a good starting point.

**Step 3 — Annotate.** Switch to the **Annotate** tab. Select a class (label category) from the right panel, pick a tool, and paint over the regions you want the AI to learn. Annotations save automatically when you switch images.

**Step 4 — Train.** Switch to the **Training** tab. Select a training mode first — **Standard** is the right choice for a first run (the **Start Train** button stays disabled until a mode is selected). Keep the default architecture (**SimpleUNet**), then click **Start Train**. Watch the loss curve go down and the F1 score go up in real time. Training stops on its own once the model stops improving, so how long it runs depends on your hardware and your data: on an NVIDIA GPU a first run over a few small images is usually a matter of minutes, while many images, large images, a larger patch size, or slower hardware can stretch it to tens of minutes or more.

**Step 5 — Review Results.** After training completes, open the **Results** tab and click **Run Inference** to see the AI's predictions on every image. Compare the predictions against your annotations using overlays and scores.

**Step 6 — Export.** Once you are satisfied with accuracy, click **Export Model** in the run's **Results** tab and choose ONNX (general-purpose) or CoreML (Apple devices).

---

## Part 2: Working with Each Tab

### Projects Tab

The **Projects** tab is your home screen. Each project appears as a tile showing the project name, image count, and training status.

**Creating a project.** Type a name in the project-name box and click **Create Project**. The project is ready to receive images immediately.

**Importing a project.** Click **Import** and select a ZIP file — either one exported from another Seg-Studio instance, or one you assembled yourself containing an `images/` folder (and optionally `masks/` and `classes.json`). A new project is created from the ZIP file name and everything inside is imported. See [import_export.md](import_export.md) for the exact format.

**Exporting a project.** Select a project and click **Export**. An export dialog opens showing the original data size, an optional foreground analysis (how small your smallest defect regions are), and a "shrink to reduce size" option that can create a resized copy of the project. Confirm to download a ZIP containing all images, annotation masks, class definitions, train/val splits, training runs (models), and metadata. This is your portable backup.

> 💡 Tip: Export your project after every major milestone (initial annotation, first successful training, final model). The ZIP contains everything needed to reproduce your work.

---

### Annotate Tab — Labeling Your Images

The left panel shows image thumbnails. The center is your canvas. The right panel shows classes and tool settings. Select a class first, then use a tool to paint that class onto the image.

#### Painting areas by hand

**Brush** (`B`) — The default tool. Click and drag to paint. Adjust size with the sidebar slider or `[` and `]` keys. Use this for any freeform region.

**Eraser** (`E`) — Removes annotations. Works like the brush but clears pixels back to background. Same size controls apply.

> 💡 Tip: Start with a large brush to fill broad areas, then switch to a smaller brush for edges. You do not need pixel-perfect labels; the AI generalizes from approximate annotations.

#### Selecting similar regions

**Wand** (`W`) — Click on a uniformly colored area and the tool automatically selects all connected pixels of similar color. The tolerance is calculated automatically from the color variation around your click point. Best for regions with consistent color such as metal surfaces or uniform coatings.

#### AI-assisted selection

**SAM Click** (`S`) — Uses a Segment Anything Model to identify object boundaries from your clicks. Left-click marks what you want (foreground). Right-click marks what you do not want (background), but exclude-clicks are off by default — switch on **Enable SAM exclude-click (right click)** in the Settings dialog first, otherwise only foreground clicks are accepted. The AI refines the selection with each click. Press `Enter` to confirm or `Esc` to cancel.

**SAM Box** (`X`) — Draw a bounding box around the object you want to select. The AI finds the precise boundary within the box. Press `Enter` to confirm or `Esc` to cancel.

You can select different SAM models from the sidebar dropdown (MobileSAM, SAM2 Tiny, SAM2 Small, and others). **MobileSAM** is fast and works well for most cases. Checkpoints are placed in `models/sam_checkpoints/` by the installer.

> 💡 Tip: For SAM Click, start with one or two foreground clicks. If the AI includes unwanted areas, add a background click (right-click) on those areas to refine — this needs **Enable SAM exclude-click (right click)** switched on in the Settings dialog.

#### Finding small defects

**Spot Detect** (`D`) — Detects small spots and blobs across the whole image. Paint one example spot with the brush, press **Detect**, then adjust the **Sensitivity** slider (1-50) in the sidebar — higher values detect more spots. Press `Enter` to accept or `Esc` to discard.

#### Tracing linear defects

**Crack Trace** (`C`) — Detects ridge-like and linear structures such as cracks, scratches, or wiring patterns. Useful for defects that are long and thin. Press `Enter` to confirm or `Esc` to cancel.

#### Region-based labeling

**Superpixel** (`P`) — Divides the image into small regions based on color and texture boundaries. Click or drag across regions to label them in bulk. Efficient for images where defect boundaries follow natural color edges.

#### Class management

Classes are the label categories your AI will learn to distinguish (e.g., "scratch", "dent", "normal").

- **Add a class:** Click **Add Class** in the class panel on the right.
- **Rename a class:** Click the class name and type — it is an editable field.
- **Change color:** Click the color swatch to open the color picker.
- **Delete a class:** Select the class, then click **Del Class**. All pixels of that class revert to background. The **Clear Marks** button beside a class name only erases its painted pixels; the class itself stays.

> **Warning:** The **background** class (ID 0) cannot be deleted or renamed. Every unlabeled pixel is automatically background.

#### Brightness / Contrast controls

Use the brightness and contrast sliders to make features easier to see while annotating.

> **Warning:** These adjustments are display-only. They do not change the actual image data and have no effect on training.

#### Common operations

| Action | Shortcut |
|--------|----------|
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Save | `Ctrl+S` |
| Zoom in / out | Mouse wheel |
| Pan (move image) | Middle-click + drag |
| Increase brush size | `]` |
| Decrease brush size | `[` |
| Confirm AI tool result | `Enter` |
| Cancel AI tool result | `Esc` |
| Previous / next image | `↑` / `↓` |

---

### Training Tab — Teaching the AI

#### When to start training

You need annotations on at least 3-5 images before training will produce useful results. More annotated images give better accuracy. If you have 50 images, annotating 15-20 of them is a good starting point.

#### Choosing a training mode

Before anything else, pick one of the four training modes. **The Start Train button is disabled until a mode is selected.**

| Mode | What it does |
|------|--------------|
| **Standard** | Annotate and train a segmentation model — the fundamental workflow |
| **Quick** | Fast segmentation training on images that already have labels |
| **Transfer** | Transfer learning from an existing model in a similar past project |
| **Instance (count)** | Train a model that counts individual objects, reusing the masks you already drew |

#### Counting objects

Standard segmentation tells you *where* a class is, as a region of pixels. It
cannot tell you that two parts touching each other are two parts.
**Instance (count)** mode trains a second kind of model that outputs one
detection per object.

You do not annotate anything new for it. It composes its own training images by
cutting objects out of the masks you already painted and scattering them over
your own backgrounds, so it sees crowded and touching arrangements that your
real photos may not contain.

**What you need:** at least 4 images that actually contain the class. Images
marked clean do not count, and neither does a mask that was never painted.

**Small objects.** The detector has a fixed input size, so a whole photo would
be shrunk to reach it and small parts would lose most of their pixels. Instead
the image is cut into overlapping tiles at capture resolution, and detections
are stitched back together. This is on by default; images smaller than one tile
simply go through in a single pass.

The one setting worth knowing is **Patch size** in the training form
(`instance_patch_size` over the API, default 768). Raise it
if your objects are larger than the tile overlap — an object bigger than the
overlap can straddle a seam with no tile holding all of it. Whatever you set is
used for **both training and inference**; they are never allowed to disagree,
because a mismatch produces no error at all, only a quietly wrong count.

Results appear per class, and **Detection highlight** in the Results tab gives
each object its own colour so you can see at a glance whether two touching
parts were counted as one.

For the API and the exact response shape, see the handbook.

#### Choosing an architecture

| Architecture | Best for | Speed | Accuracy |
|-------------|----------|-------|----------|
| **STDC** | Real-time / edge deployment, compact defect shapes | Fast | High |
| **SimpleUNet** | General purpose (default) — smallest model, memory-constrained environments | Medium | Good |

If you are unsure, keep **SimpleUNet** (the default) or let auto-config
pick for you. On our 37-project factory library STDC was the per-project
best more often than SimpleUNet (15 vs 5), so STDC is worth trying when
you need more accuracy or faster inference.

#### Key parameters

Click the **Settings** button next to **Start Train** to see the hyperparameter panel. The most important ones are:

| Parameter | What it means | Recommended |
|-----------|--------------|-------------|
| **Epochs** | How many times the AI reviews all training images | 60–100 (auto-set from image size: 60 below 1000 px wide, 80 below 2000 px, 100 above) |
| **Patch Size** | Size of image crops used for training (pixels) | 256 |
| **Learning Rate** | How aggressively the AI updates itself | 0.0005 (default) |

Most other settings are tuned automatically. You do not need to change them unless you have specific reasons.

#### Auto-tuning

Seg-Studio automatically configures many advanced settings for you:

- Loss function (Focal + Dice when the foreground is sparse, Cross-Entropy + Dice otherwise)
- Class weights based on how much of each class appears in your data
- Learning rate schedule (Cosine Annealing with warm-up)
- Early stopping (training halts if no improvement for 15 epochs)

#### Reading the training charts

Two charts update in real time during training:

- **Loss chart** — The loss value should go down over time. This means the AI is learning. If it stays flat, your annotations may be insufficient.
- **F1 chart** — The F1 score should go up over time. This measures how closely the AI matches your annotations. Watch the trend rather than the value: what counts as a good F1 depends on how small your defects are and how much of each image they cover, so a number that means "ready" on one project can mean "unusable" on another.

> 💡 Tip: Loss goes down = the AI is learning. F1 goes up = the AI is getting accurate. If loss goes down but F1 stays low, you may need more diverse annotations.

#### Stopping and resuming

Click **Stop** to halt training early. The best checkpoint up to that point is automatically saved. You can start a new training run at any time; each run is saved separately.

#### Training queue

If you start training on multiple projects, they are queued and run one at a time. The Training Pulse widget in the bottom-right corner shows which project is currently training and how many are waiting.

#### Advanced Accuracy Mode (DINOv2 distillation)

The **DINOv2 Distill** toggle (off by default) uses a powerful pre-trained vision model as a teacher. Enabling it can improve accuracy, especially with limited training data, at the cost of longer training. The teacher weights are installed automatically — no manual setup needed.

---

### Results Tab — Checking the AI's Work

Each training run creates its own Results tab. You can have multiple tabs open to compare different runs.

#### Running inference

Click **Run Inference** to apply the trained model to every image in your project. A progress indicator shows how many images have been processed. Run Inference always computes fresh predictions, so results are never stale after a re-train.

#### Restoring cached results

If you re-open a Results tab, click **Restore** to reload previously computed predictions from the cache instead of re-running inference. This is much faster when nothing has changed.

#### Understanding scores

| Score | What it tells you | How to read it |
|-------|-------------------|----------------|
| **F1** | Balance between finding all defects and not over-detecting | Compare runs on the same dataset; the absolute value is not portable |
| **mIoU** | How precisely the predicted region overlaps the true region | Much lower than F1 suggests loose boundaries |

There is no universal pass mark. F1 is summed over pixels, so a project with
hair-thin defects scores lower than one with large blobs at the same practical
quality, and the same model scores differently on a different image mix. Use
the scores to compare runs **on the same dataset with the same settings**, and
decide what is good enough from your own inspection requirement — how many
missed defects and how many false alarms you can live with — not from a
threshold in a table.

Scores are shown both as a summary across all images and per-image in the detail view of the selected image. Low-scoring images need more or better annotations.

> 💡 Tip: To find the images the AI is least sure about, sort the score table in the **Training** tab by **Confidence** (its sort keys are image name, confidence and FG patch ratio). Add more annotations to those images, re-train, and your overall score will improve.

#### Overlay modes

- **Class Overlay** — Colors each pixel by its predicted class, overlaid on the original image. Lets you visually check if the AI found the right regions.
- **Confidence Heatmap** — Shows how sure the AI is about each pixel. Blue = uncertain, red = confident. Look for uncertain areas to find where the AI needs more training data.

#### Count mode

Switch to **Count** mode to see how many separate regions the AI found in each image. This counts connected blobs of pixels, so two objects touching each other count as one. When you need individual objects separated, train a counting model instead (see [Counting objects](#counting-objects)).

#### Area mode + calibration

Switch to **Area** mode to measure the total area of detected regions in each image.

To convert pixel measurements to real-world units, draw a **Calibration Line** on an image: click two points of known distance, enter the real distance, and select the unit (mm, cm, or m). All area measurements update automatically.

#### Tab locking

Click the lock icon (🔓/🔒) on a Results tab to prevent it from being accidentally closed. Locked tabs persist across sessions (stored in your browser). In the Training tab, locked runs are marked "Locked in Results tab" and cannot be deleted.

#### Apply to label

Double-click an image in the Results image list to copy its prediction back into your annotations, or click **Apply results**, tick the images you want, and click **Apply**. This is useful for bootstrapping: train a rough model, apply its predictions, then manually correct the mistakes. This "human-in-the-loop" cycle dramatically speeds up annotation.

---

### Live Inspection Tab

The Live Inspect tab connects to a camera for real-time inference.

**Camera setup.** Open **Camera Settings** to set the device ID, resolution, and FPS, then start the camera preview. The dialog accepts a numeric USB device index only; to use a network camera, pass its stream URL as the device id through the API.

**Real-time inference.** Select a project and a completed model, load it, and start inspection. The AI processes each frame, overlays the predicted segmentation on the live feed, and shows OK/NG results, latency, and running statistics.

**File-based inspection.** No camera? Drop image files onto the tab to run the same inference on saved images.

**Detection display settings.** Toggle the region-count display, area display (calibration-aware), and a detail view listing each region's class, area, and confidence.

---

## Part 3: Reference

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `B` | Brush tool |
| `E` | Eraser tool |
| `G` | Fill (bucket) tool |
| `V` | Move tool |
| `W` | Wand tool |
| `S` | SAM Click tool |
| `X` | SAM Box tool |
| `M` | Measure tool |
| `D` | Spot Detect tool |
| `C` | Crack Trace tool |
| `P` | Superpixel tool |
| `[` | Decrease brush/eraser size |
| `]` | Increase brush/eraser size |
| `1`-`9` | Switch active class |
| `Enter` | Confirm AI tool result |
| `Esc` | Cancel AI tool result |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+S` | Save |
| `Shift+C` | Mark Clean (mark image as OK) |
| `Delete` / `Backspace` | Clear annotations on the selected image |
| `↑` / `↓` | Previous / next image |

> The **?** button in the header toggles description mode (tooltips for UI elements).

### Model Export

**ONNX** — Universal format compatible with most inference engines (ONNX Runtime, TensorRT, OpenVINO). Open a completed run's **Results** tab, click **Export Model**, then pick **ONNX** and confirm.

**CoreML** — For Apple devices (iPad, iPhone, Mac). In the same **Export Model** dialog, pick **CoreML** and confirm. The exported `.mlmodel` file can be integrated directly into iOS/iPadOS apps. A **CoreML (Updatable)** variant, offered in the same dialog, supports on-device incremental training.

**OpenVINO** — For Intel CPU / iGPU / NPU edge devices. Available from the run's export menu in FP32, FP16 (recommended), and INT8 precision.

Exported files download to your browser's default download folder.

> 💡 Tip: Always test your exported model on a few representative images in the target deployment environment before rolling out to production.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1650 (4 GB VRAM) | NVIDIA RTX 3060+ (8 GB+ VRAM) |
| RAM | 8 GB | 16 GB+ |
| Storage | SSD (10 GB free) | SSD (50 GB+ free for large projects) |
| Browser | Chrome, Edge, or Firefox (latest) | Chrome |

Training requires an NVIDIA CUDA GPU (Windows is the supported platform). Without a CUDA GPU you can still annotate images and run inference, but not train.

### Data Management and Backup

**Project data location.** All project data is stored in the `projects/` folder relative to the Seg-Studio installation. Each project has its own subfolder containing images, masks, model checkpoints, and configuration files.

**Backup strategy.** Copy the entire `projects/` folder to create a full backup. Alternatively, use the **Export** feature in the Projects tab to create portable ZIP archives of individual projects.

**Restoring from backup.** To restore, either copy the project folders back into `projects/`, or use **Import Project** with an exported ZIP file.

> **Warning:** Back up your data regularly, especially after completing annotation milestones or training successful models. There is no "undo" for a deleted project.

### Troubleshooting

#### The screen is blank (white page)

The UI may not be built. Ask your system administrator to run:
```
cd apps/trainer_ui
npm install
npm run build
```
Then refresh your browser at `http://localhost:8002/ui/`.

#### "Connecting to API server..." message won't go away

The backend is not running or is still starting up. Seg-Studio takes about 1-2 seconds to start, but loading AI modules takes longer (up to 30 seconds). Wait for the loading screen to finish. If it persists, check that port 8002 is not used by another application.

#### Training loss does not decrease

- Ensure you have annotated at least 3-5 images with clear examples of each class.
- Verify your annotations are correct (wrong labels confuse the AI).
- Try reducing the learning rate from 0.0005 to 0.0001.
- Increase epochs to 80-100 to give the AI more time to learn.

#### GPU out of memory (CUDA OOM)

- Reduce model size (the **Model Size** setting in the hyperparameter panel).
- Reduce patch size (256 to 192 to 128).
- Close other GPU-heavy applications (games, video editors, other AI tools).
- Disable GPU acceleration in your browser (Settings > System > Hardware acceleration).

#### SAM tool shows an error or no models available

- SAM checkpoints are downloaded by the installer and stored in `models/sam_checkpoints/`. Verify the folder is not empty.
- If checkpoints are missing, re-run the install script (an internet connection is required); check your firewall or proxy settings if downloads fail.

#### Annotations are not saving

Annotations auto-save when you switch to another image. If you close the browser without switching images, the most recent strokes may be lost. To be safe, switch to any other image before closing the browser.

#### Export fails (ONNX or CoreML)

- Make sure you selected a training run that completed successfully (not one that was stopped early or errored).
- For CoreML export, the `coremltools` Python package must be installed.
- For ONNX export, the `onnx` package must be installed.
- Check the server logs for detailed error messages.

#### Inference is slow or returns errors

- Seg-Studio processes inference requests one at a time to avoid GPU memory conflicts. If you have many images, inference may take several minutes.
- Do not navigate away from the Results tab during inference; the progress indicator shows current status.
- If errors occur on specific images, those images may be corrupted. Try re-uploading them.

#### Colors or labels look wrong after import

- Verify your mask files are single-channel (grayscale) PNGs with pixel values matching class IDs.
- If your masks are RGB, only the red channel is used for class IDs.
- Include a `classes.json` file in your import folder to define class names and colors.

---

### Glossary

| Term | Meaning |
|------|---------|
| **Annotation / Label** | The colored regions you paint on images to show the AI what to look for |
| **Class** | A category of object or region (e.g., "scratch", "dent", "background") |
| **Epoch** | One full pass through all training images. More epochs = more learning time |
| **F1 Score** | A number from 0 to 1 measuring accuracy. Combines precision (not over-detecting) and recall (not missing things) |
| **Inference** | Running the trained model on images to generate predictions |
| **Loss** | A number that decreases as the AI learns. Lower is better |
| **Mask** | A grayscale image where each pixel's value is a class ID. This is how annotations are stored |
| **mIoU** | Mean Intersection over Union. Measures how well predicted regions overlap with true regions |
| **ONNX** | Open Neural Network Exchange. A universal model format supported by many platforms |
| **CoreML** | Apple's model format for iOS, iPadOS, and macOS deployment |
| **Patch** | A small crop of a larger image, used during training to fit within GPU memory |
| **Segmentation** | Classifying every pixel in an image (unlike object detection which draws boxes) |
| **SAM** | Segment Anything Model. An AI that finds object boundaries from click or box prompts |

### Supported Image Formats

| Format | Import | Notes |
|--------|--------|-------|
| JPEG (.jpg, .jpeg) | Yes | Most common; lossy compression |
| PNG (.png) | Yes | Recommended for masks; lossless |
| BMP (.bmp) | Yes | Large file size; uncompressed |
| TIFF (.tif, .tiff) | Yes | Common in industrial imaging |
| WebP (.webp) | Yes | Accepted for upload and ZIP import |

### Best Practices for Accurate Models

1. **Annotate diverse examples.** Include images with different lighting, angles, and defect sizes. The AI can only learn what you show it.

2. **Be consistent.** If you label a scratch in one image, label all visible scratches in that image. Partial annotations teach the AI to ignore defects.

3. **Use the bootstrap loop.** Annotate 3-5 images manually, train a rough model, use **Apply results** in the Results tab to copy its predictions onto more images, correct the mistakes, then re-train. Each cycle improves accuracy faster than annotating everything from scratch.

4. **Check low-scoring images first.** After inference, go through the per-image scores and start with the weakest. The worst images tell you exactly where to focus your annotation effort.

5. **Start simple.** Begin with 2-3 classes (e.g., background + defect). Add more granular classes later if needed.

6. **Keep projects focused.** One project per inspection target. Mixing unrelated images (e.g., gears and PCBs) in one project will confuse the model.

> 💡 Tip: A model trained on 20 well-annotated images almost always outperforms a model trained on 100 poorly-annotated images. Quality over quantity.
