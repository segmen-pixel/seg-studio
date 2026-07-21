# Seg-Studio User Guide

Seg-Studio is a visual inspection tool that lets you teach an AI to recognize regions in images. You label areas of interest on your inspection images (scratches, defects, parts), train a segmentation model, review the results, and export the trained model for deployment on edge devices including iPad and iPhone.

---

## Part 1: Getting Started (10-Minute Quick Start)

### The Workflow at a Glance

Every project follows six steps. You can complete a basic run-through in under 10 minutes.

**Step 1 — Create a Project.** Open the **Projects** tab and click **New Project**. Give it a descriptive name such as "Gear Defect" or "PCB Scratch Detection". One project per inspection target is recommended.

**Step 2 — Upload Images.** Drag and drop image files (JPEG, PNG, BMP, TIFF) directly onto the browser window, or upload a ZIP file containing multiple images. You need at least a few images to get started; 20-50 is a good starting point.

**Step 3 — Annotate.** Switch to the **Annotate** tab. Select a class (label category) from the right panel, pick a tool, and paint over the regions you want the AI to learn. Annotations save automatically when you switch images.

**Step 4 — Train.** Switch to the **Training** tab. Select a training mode first — **Standard** is the right choice for a first run (the **Start Train** button stays disabled until a mode is selected). Keep the default architecture (**DeepLabV3+**), then click **Start Train**. Watch the loss curve go down and the F1 score go up in real time. A typical first training takes 5-15 minutes.

**Step 5 — Review Results.** After training completes, open the **Results** tab and click **Run Inference** to see the AI's predictions on every image. Compare the predictions against your annotations using overlays and scores.

**Step 6 — Export.** Once you are satisfied with accuracy, export the model in ONNX (general-purpose) or CoreML (Apple devices) format from the **Training** tab.

---

## Part 2: Working with Each Tab

### Projects Tab

The **Projects** tab is your home screen. Each project appears as a tile showing the project name, image count, and training status.

**Creating a project.** Click **New Project**, type a name, and press Enter. The project is ready to receive images immediately.

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

**SAM Click** (`S`) — Uses a Segment Anything Model to identify object boundaries from your clicks. Left-click marks what you want (foreground). Right-click marks what you do not want (background). The AI refines the selection with each click. Press `Enter` to confirm or `Esc` to cancel.

**SAM Box** (`X`) — Draw a bounding box around the object you want to select. The AI finds the precise boundary within the box. Press `Enter` to confirm or `Esc` to cancel.

You can select different SAM models from the sidebar dropdown (MobileSAM, SAM2 Tiny, SAM2 Small, and others). **MobileSAM** is fast and works well for most cases. Checkpoints are placed in `models/sam_checkpoints/` by the installer.

> 💡 Tip: For SAM Click, start with one or two foreground clicks. If the AI includes unwanted areas, add a background click (right-click) on those areas to refine.

#### Finding small defects

**Spot Detect** (`D`) — Detects small spots and blobs across the whole image. Paint one example spot with the brush, press **Detect**, then adjust the **Sensitivity** slider (1-50) in the sidebar — higher values detect more spots. Press `Enter` to accept or `Esc` to discard.

#### Tracing linear defects

**Crack Trace** (`C`) — Detects ridge-like and linear structures such as cracks, scratches, or wiring patterns. Useful for defects that are long and thin. Press `Enter` to confirm or `Esc` to cancel.

#### Region-based labeling

**Superpixel** (`P`) — Divides the image into small regions based on color and texture boundaries. Click or drag across regions to label them in bulk. Efficient for images where defect boundaries follow natural color edges.

#### Class management

Classes are the label categories your AI will learn to distinguish (e.g., "scratch", "dent", "normal").

- **Add a class:** Click the **+** button in the class panel on the right.
- **Rename a class:** Double-click the class name to edit it.
- **Change color:** Click the color swatch to open the color picker.
- **Delete a class:** Click the delete icon next to the class name. All pixels of that class revert to background.

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

Before anything else, pick one of the three training modes. **The Start Train button is disabled until a mode is selected.**

| Mode | What it does |
|------|--------------|
| **Standard** | Annotate and train a segmentation model — the fundamental workflow |
| **Quick** | Fast segmentation training on images that already have labels |
| **Transfer** | Transfer learning from an existing model in a similar past project |

#### Choosing an architecture

| Architecture | Best for | Speed | Accuracy |
|-------------|----------|-------|----------|
| **DeepLabV3+** | General purpose (default) — wide context, irregular defect shapes | Slow | High |
| **STDC** | Real-time / edge deployment, compact defect shapes | Fast | High |
| **SimpleUNet** | Smallest model, memory-constrained environments | Medium | Good |

If you are unsure, choose **DeepLabV3+** — on our 37-project factory
library it was the per-project best architecture most often (17/37,
vs 15 for STDC and 5 for SimpleUNet).

#### Key parameters

Click the **Settings** button next to **Start Train** to see the hyperparameter panel. The most important ones are:

| Parameter | What it means | Recommended |
|-----------|--------------|-------------|
| **Epochs** | How many times the AI reviews all training images | 60–100 (auto-set from image size: 60 below 1000 px wide, 80 below 2000 px, 100 above) |
| **Batch Size** | Images processed at once (higher = faster but uses more GPU memory) | 4-8 |
| **Patch Size** | Size of image crops used for training (pixels) | 256 |
| **Learning Rate** | How aggressively the AI updates itself | 0.0005 (default) |

Most other settings are tuned automatically. You do not need to change them unless you have specific reasons.

#### Auto-tuning

Seg-Studio automatically configures many advanced settings for you:

- Loss function (Focal Loss + Dice)
- Class weights based on how much of each class appears in your data
- Learning rate schedule (Cosine Annealing with warm-up)
- Early stopping (training halts if no improvement for 15 epochs)

#### Reading the training charts

Two charts update in real time during training:

- **Loss chart** — The loss value should go down over time. This means the AI is learning. If it stays flat, your annotations may be insufficient.
- **F1 chart** — The F1 score should go up over time. This measures how accurately the AI matches your annotations. Higher is better; 0.7+ is good, 0.9+ is excellent.

> 💡 Tip: Loss goes down = the AI is learning. F1 goes up = the AI is getting accurate. If loss goes down but F1 stays low, you may need more diverse annotations.

#### Stopping and resuming

Click **Stop** to halt training early. The best checkpoint up to that point is automatically saved. You can start a new training run at any time; each run is saved separately.

#### Training queue

If you start training on multiple projects, they are queued and run one at a time. The status bar shows which project is currently training and how many are waiting.

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

| Score | What it tells you | Good value |
|-------|-------------------|------------|
| **F1** | Overall accuracy — balance between finding all defects and not over-detecting | 0.7+ good, 0.9+ excellent |
| **mIoU** | How precisely the predicted region overlaps the true region | 0.6+ good, 0.8+ excellent |

Scores are shown both as a summary across all images and per-image in the image table. Click column headers to sort. Low-scoring images need more or better annotations.

> 💡 Tip: Sort by F1 (ascending) to find the images where the AI struggles most. Add more annotations to those images, re-train, and your overall score will improve.

#### Overlay modes

- **Class Overlay** — Colors each pixel by its predicted class, overlaid on the original image. Lets you visually check if the AI found the right regions.
- **Confidence Heatmap** — Shows how sure the AI is about each pixel. Blue = uncertain, red = confident. Look for uncertain areas to find where the AI needs more training data.

#### Count mode

Switch to **Count** mode to see how many separate objects the AI detected in each image. Uses connected-component analysis to count distinct regions.

#### Area mode + calibration

Switch to **Area** mode to measure the total area of detected regions in each image.

To convert pixel measurements to real-world units, draw a **Calibration Line** on an image: click two points of known distance, enter the real distance, and select the unit (mm, cm, or m). All area measurements update automatically.

#### Tab locking

Click the lock icon (🔓/🔒) on a Results tab to prevent it from being accidentally closed. Locked tabs persist across sessions (stored in your browser). In the Training tab, locked runs are marked "Locked in Results tab" and cannot be deleted.

#### Apply to label

Click **Apply to label** on an image (or select several images and use **Apply all**) to copy the AI's predictions back into your annotations. This is useful for bootstrapping: train a rough model, apply its predictions, then manually correct the mistakes. This "human-in-the-loop" cycle dramatically speeds up annotation.

---

### Live Inspection Tab

The Live Inspect tab connects to a camera for real-time inference.

**Camera setup.** Open **Camera Settings** to set the device ID, resolution, and FPS (USB cameras and network camera streams are supported), then start the camera preview.

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

**ONNX** — Universal format compatible with most inference engines (ONNX Runtime, TensorRT, OpenVINO). Select a completed training run in the **Training** tab and click **Export > ONNX**.

**CoreML** — For Apple devices (iPad, iPhone, Mac). Select a completed training run and click **Export > CoreML**. The exported `.mlmodel` file can be integrated directly into iOS/iPadOS apps. A **CoreML Updatable** variant supports on-device incremental training.

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

- Reduce batch size (8 to 4 to 2 to 1).
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

### Best Practices for Accurate Models

1. **Annotate diverse examples.** Include images with different lighting, angles, and defect sizes. The AI can only learn what you show it.

2. **Be consistent.** If you label a scratch in one image, label all visible scratches in that image. Partial annotations teach the AI to ignore defects.

3. **Use the bootstrap loop.** Annotate 3-5 images manually, train a rough model, use **Apply to label** to copy its predictions onto more images, correct the mistakes, then re-train. Each cycle improves accuracy faster than annotating everything from scratch.

4. **Check low-scoring images first.** After inference, sort by F1 score ascending. The worst images tell you exactly where to focus your annotation effort.

5. **Start simple.** Begin with 2-3 classes (e.g., background + defect). Add more granular classes later if needed.

6. **Keep projects focused.** One project per inspection target. Mixing unrelated images (e.g., gears and PCBs) in one project will confuse the model.

> 💡 Tip: A model trained on 20 well-annotated images almost always outperforms a model trained on 100 poorly-annotated images. Quality over quantity.
