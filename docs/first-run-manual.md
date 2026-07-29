# First Run Walkthrough

> **Which document is this?** The shortest single path — six steps from a
> running server to your first prediction. Installing Seg-Studio is covered by
> the [README Quick Start](../README.md#quick-start), individual features by
> the [User Guide](user-guide.md), and the same workflow at full length on a
> sample dataset by the [Handbook](handbook.md).

Welcome to Seg-Studio, a semantic segmentation tool built for factory inspection.
This guide walks you through the full workflow — from opening the app to seeing
your first AI prediction — in about 10 minutes on a machine with an NVIDIA GPU.

No machine learning experience required. Just follow the steps.

> Japanese version: [ja/first-run-manual.md](ja/first-run-manual.md)
>
> **Prerequisite:** Seg-Studio has to be installed and running before step 1.
> If it isn't, install it first — [README → Quick
> Start](../README.md#quick-start) — then come back here.

---

## Before You Start

You need two things:

1. **The Seg-Studio server running** on your machine (or a machine on your network).
   You should be able to reach `http://localhost:8002/ui/` in a browser.
2. **A handful of test images** — 5 to 10 photos of the parts you want to inspect.
   Standard formats (PNG, JPEG) work fine.

If someone else set up the server for you, just confirm the URL and move on.

---

## Step 1: Open Seg-Studio (30 seconds)

Open your browser and go to:

```
http://localhost:8002/ui/
```

You should see a header bar with four tabs: **Projects**, **Annotate**,
**Training**, and **Live Inspect**. The Projects tab is selected by default.

> **Tip:** If the page won't load, the server probably isn't running. Ask your
> setup person to check, or look at the terminal where the server was started.

---

## Step 2: Create Your First Project (1 minute)

A project holds your images, labels, and trained models for one inspection task.

1. Type a name in the text field — something descriptive like
   `Surface Scratch Test` or `Solder Joint Check`.
2. Click **Create Project**.
3. A project card appears and is selected automatically.

That's it. You now have a workspace ready for images.

---

## Step 3: Upload Test Images (1 minute)

1. Click the **Annotate** tab at the top (shown as ラベル when the UI language
   is set to Japanese).
2. Click the **+** button in the left image panel, or drag and drop image files
   directly into the panel.
3. Wait for the upload progress to finish.
4. Click any image thumbnail to load it onto the canvas.

You should see your image fill the center of the screen, with a class list on
the right side.

---

## Step 4: Label a Few Images (3 minutes)

This is where you teach the AI what to look for. You paint over defect areas
so the model can learn the pattern.

1. **Create a class** — Click **Add Class** on the right panel. Give it a name
   like `scratch` or `crack`. This is the defect category.
2. **Select the Brush tool** — Press **B** on your keyboard, or click the brush
   icon in the toolbar.
3. **Paint over a defect** — Click and drag on the image to mark the defect area.
   - Press **[** and **]** to shrink or grow the brush.
   - Press **E** to switch to the eraser if you make a mistake.
4. **Move to the next image** — Click another thumbnail in the left panel. Your
   work is saved automatically when you switch images.
5. **Repeat** for at least 3 images. The more images you label, the better the results.

> **Tip:** For faster labeling, try the **SAM click tool** (press **S**).
> Left-click on a defect and the AI will try to select its outline for you.
> Press Enter to confirm or Esc to cancel.

---

## Step 5: Train Your First Model (a few minutes on an NVIDIA GPU)

Now let the AI learn from your labels.

1. Click the **Training** tab at the top.
2. Select a training mode — pick **Standard** for a first run. The **Start
   Train** button stays disabled until a mode is selected.
3. Click **Start Train**. The default settings work well for a first run — no
   need to change anything.
4. Watch the **loss chart** — you should see the line trend downward. This means
   the model is learning.
5. Training stops automatically when the model stops improving, so how long it
   runs depends on your hardware and your data. On an NVIDIA GPU, a first run
   over the handful of small images in this tutorial usually finishes in a few
   minutes. More images, larger images, a larger patch size, or slower hardware
   can stretch it to tens of minutes or more.

> **Note:** If the Start button is disabled, select a training mode first.
> For useful results you also want at least 3 labeled images with at least
> one class painted.

---

## Step 6: Check the Results

When training finishes, a new **Result tab** appears in the tab bar.

1. Click the result tab to open it.
2. Click **Run Inference** to generate predictions on your images.
3. Browse through images in the left panel. For each image you'll see:
   - The AI's prediction overlay (colored regions)
   - A confidence score

Compare the prediction against your original labels. If the AI is missing
defects or marking too much, that's normal for a first run with only a few
images.

---

## What's Next?

- **Label more images and retrain.** Accuracy improves significantly with 10-20+
  labeled images. Focus on images where the model got it wrong.
- **Try different tools** for faster labeling:
  - **Wand** (press **W**) — auto-selects similar colored regions
  - **SAM click** (press **S**) — AI-assisted selection with just a click
- **Count objects, not just find them.** If you need to know *how many* parts
  are in each image, pick the **Instance (count)** training mode. It reuses
  the masks you just drew — no new labelling — and separates objects that
  touch each other. You need 4 or more images containing the class.
- **Export your model** — open the run's **Result tab** and click **Export Model**
  to save it as ONNX or CoreML for deployment on inspection devices.

---

## Quick Troubleshooting

| Problem | What to check |
|---|---|
| Page won't load | Is the server running? Try `http://localhost:8002/ui/` again. |
| Training won't start | Select a training mode first (e.g. Standard); then check you have labeled images. |
| Results look bad | Label more images and retrain. 10+ labeled images recommended. |
| Everything feels slow | Check if another training run or heavy process is using the GPU. |
| Image upload fails | Make sure files are PNG or JPEG and not corrupted. |

---

## Keyboard Shortcuts Reference

| Key | Action |
|---|---|
| **B** | Brush tool |
| **E** | Eraser tool |
| **S** | SAM click tool |
| **W** | Wand tool |
| **D** | Spot detect tool |
| **[** / **]** | Decrease / increase brush size |
| **Enter** | Confirm (SAM / Spot detect) |
| **Esc** | Cancel current operation |
| **Ctrl+Z** | Undo |

---

That's the complete workflow: upload, label, train, check. Each cycle gets you
a better model. Start simple, iterate, and let the results guide where to
label next.
