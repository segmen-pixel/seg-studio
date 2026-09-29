# Import / Export Format Specification

Specification for interoperating Seg-Studio project data with external tools.

---

## Export

### Endpoint

```
GET /api/v1/projects/{project_id}/datasets/export
```

Optional query parameter: `resize_scale` (0.1–1.0) — downscales images (Lanczos) and masks (nearest-neighbor) on export.

Triggered by the **Export** button on a project tile. The export dialog shows the original data size, an optional foreground analysis, and a "shrink to reduce size" option before the ZIP download starts.

### Output Format

ZIP archive. Filename: `{project_name}_{YYYYMMDD_HHMM}.zip` (or `{project_name}_s{scale}_{YYYYMMDD_HHMM}.zip` when resized).

```
{prefix}/
├── images/          # Original images (original filenames preserved)
│   ├── sample_001.png
│   ├── sample_002.jpg
│   └── ...
├── masks/           # Annotation masks (grayscale PNG); images without
│   ├── {item_id}.png  # annotations get an all-zero (background) mask
│   └── ...
├── train.txt        # Training item ID list (one ID per line)
├── val.txt          # Validation item ID list (one ID per line)
├── training/        # Training runs (checkpoints, configs, metrics)
│   ├── runs/{run_id}/...   # not read back by Import
│   └── pretrained/  # imported pretrained checkpoint, when present
└── metadata.json    # Project metadata and each image's marks
```

### masks/ Format

| Field | Value |
|-------|-------|
| Filename | `{item_id}.png` (item ID) |
| Channels | Single channel (grayscale, PIL mode `"L"`) |
| Pixel values | Class ID (0 = background, 1+ = user-defined classes) |
| Ignore index | 255 (legacy unpainted regions) |

> **255 is not excluded from training.** Despite the field name, mask value 255
> marks pixels left unpainted by older versions and is converted to background
> (class 0) during dataset preparation, before any loss or metric sees it — the
> model learns those pixels as background and they count in the score. The
> `ignore_index` field is a fixed contract value (it must be 255) reserved for an
> internal safety clamp on out-of-range class ids. If you need genuinely ignored
> regions, do not rely on 255.

A large image's mask, which is stored in tiles rather than as a PNG, is written from its tiles like any other mask, a band of rows at a time, and `resize_scale` shrinks it the same way (nearest-neighbor). Before 0.9.9 such an image was exported with the all-zero placeholder, or with an out-of-date PNG left beside the tiles.


### metadata.json

```json
{
  "project_id": "uuid-string",
  "project_name": "Bolt",
  "exported_at": "2026-04-01T12:00:00",
  "num_images": 97,
  "num_train": 78,
  "num_val": 19,
  "classes": [
    { "id": 0, "name": "background", "color": [0, 0, 0] },
    { "id": 1, "name": "scratch", "color": [242, 36, 36] }
  ],
  "ignore_index": 255,
  "items": [
    { "id": "sample_001", "filename": "sample_001.png", "name": "sample_001.jpg",
      "hasMask": true, "markedClean": false, "set": "train" },
    { "id": "sample_002", "filename": "sample_002.png", "name": "sample_002.png",
      "hasMask": true, "markedClean": true },
    { "id": "sample_003", "filename": "sample_003.png", "name": "sample_003.png",
      "hasMask": false, "markedClean": false,
      "draft": true, "draftRun": "review", "draftReason": "edge unclear" }
  ]
}
```

Each entry of `items` describes one image. `id` and `filename` are always there; the other fields record what was decided about the image, which the pixels in `masks/` cannot say, and Import reads them back (see [Per-image marks](#per-image-marks)).

| Field | Written | Meaning |
|-------|---------|---------|
| `id` | always | Item ID; the image's mask is `masks/{id}.png` |
| `filename` | always | File name under `images/` |
| `name` | when set | Name shown in the image list; without it, Import shows the file name |
| `hasMask` | always | `false`: the image has no mask, and its file in `masks/` is the all-zero placeholder the export writes |
| `markedClean` | always | `true`: marked **OK** — someone checked the image and there is nothing to label. Its all-zero mask is a negative sample, not an unlabelled image |
| `set` | `train` / `test` only | Manual train/test assignment |
| `draft` | when set | `true`: flagged **Review**, or a mask a training run drafted that nobody has checked yet |
| `draftRun` | with `draft` | `"review"` for a Review flag, otherwise the run that drafted the mask |
| `draftReason` | with `draft`, when given | Why it was flagged |
| `by` | when an agent saved the mask | The agent that wrote it; a person's save carries none |
| `synthetic` | synthesized samples | `true`: made by the synthesis tool |

Exports from earlier versions carry only `id` and `filename`.

### train.txt / val.txt

- If a split exists in `prepared/splits/`, it is used
- Otherwise, exported items are randomly split 80/20
- One item ID per line

---

## Import

### Endpoint

```
POST /api/v1/projects/{project_id}/datasets/annotate/import_zip
```

Select a **ZIP file** via the **Import** button in the Projects tab. A new project is created from the ZIP file name, then the archive contents are imported into it.

### Supported ZIP Structures

Import reads one place in the archive: its root or, when the root holds no pictures and just one folder (an export, or a zipped folder), that folder. There:

- if an `images/` folder holds pictures, those are the images, and the masks are the `.png` files in the `masks/` folder beside it;
- otherwise the pictures lying directly in that place are the images, with masks from a `masks/` folder beside them if there is one.

Only files directly inside `images/` and `masks/` count, and **pictures in any deeper folder are not imported**: an export's training runs under `training/` (a run's reliability charts are PNGs), `prepared/` copies, sub-folders of pictures. `__MACOSX/`, `._*`, `Thumbs.db`, `.DS_Store` and `desktop.ini` are ignored. An archive with no images in that place is refused, and the error names a picture that was passed over.

When the one top folder in turn holds no pictures and just one folder, Import goes on into that one, up to three folders deep in all: Windows' **Extract All** puts an export inside a second folder of the same name, and zipping that folder again keeps both (Pattern D). It never goes into `images/` or `masks/` this way, nor, below the top folder, into `training/`, `runs/` or `prepared/`.

#### Pattern A: Flat Structure

```
MyProject.zip
├── images/
│   ├── img_001.png
│   └── ...
├── masks/
│   ├── img_001.png
│   └── ...
└── classes.json
```

#### Pattern B: One Top Folder (a Seg-Studio export)

```
sample_20260401_1200.zip
└── sample_20260401_1200/
    ├── images/
    ├── masks/
    ├── metadata.json
    ├── train.txt / val.txt
    └── training/        # not imported
```

#### Pattern C: A Folder of Pictures

```
photos.zip
└── photos/              # or the pictures directly at the ZIP root
    ├── img_001.jpg
    ├── img_002.jpg
    └── masks/           # optional
        └── img_001.png
```

#### Pattern D: An Export Extracted and Zipped Again

```
sample_20260401_1200.zip
└── sample_20260401_1200/          # the folder Extract All made
    └── sample_20260401_1200/      # the export's own folder
        ├── images/
        ├── masks/
        └── metadata.json
```

> Earlier versions looked for `images/` and `masks/` at **any depth** and took every picture outside `masks/` as an image, so an export came back with its training charts as extra images. An archive built around a deeper folder (for example `datasets/prepared/images/`) now needs its `images/` and `masks/` moved to the top.

### Image Files (images/)

| Field | Value |
|-------|-------|
| Supported extensions | `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp` |
| Case sensitivity | Case-insensitive |
| Note | Non-PNG images are converted to PNG on import |

### Mask Files (masks/)

| Field | Value |
|-------|-------|
| Supported extensions | `.png` only |
| Channels | Single channel (grayscale) recommended. For RGB images the import reads the first OpenCV channel (blue in BGR order) as the class ID, so supply single-channel grayscale to avoid ambiguity |
| Pixel values | Class ID (0 = background, 1+ = user-defined classes) |

### Mask-to-Image Matching Rules

If the ZIP contains a `metadata.json` (i.e. it is a Seg-Studio export), masks are matched through its `items` array (original filename → item ID). Otherwise images and masks are matched by **filename stem** (filename without extension).

```
images/bolt_001.png  <-->  masks/bolt_001.png   (stem: bolt_001)
images/sample.jpg    <-->  masks/sample.png     (stem: sample)
```

Images without a matching mask are imported as unannotated.

### Per-image marks

When the `items` of `metadata.json` carry the fields listed under [metadata.json](#metadatajson), Import puts them back:

- `markedClean: true` — the image is **OK** again, provided its mask was imported and has nothing painted in it (a mark over paint is ignored)
- `hasMask: false` — the all-zero placeholder in `masks/` is dropped and the image stays unlabelled
- `set`, `name`, `draft` / `draftRun` / `draftReason` (the **Review** flag) and `synthetic` — restored as they were; `by` only when the mask came along

Without these fields — an export from an earlier version, or a ZIP from another tool — every supplied mask is taken as saved, and an all-zero mask is an image not labelled yet, not an **OK** one.

### Class Definition File

A `classes.json` beside `images/` (or at the ZIP root) is used first. Otherwise the `classes` array of `metadata.json` is used — in an export, the project's classes when it was exported. Only when neither exists is a `classes.json` from deeper in the archive used, such as the copy a training run keeps.

```json
{
  "version": 1,
  "ignore_index": 255,
  "classes": [
    { "id": 0, "name": "background", "color": [0, 0, 0], "active": true },
    { "id": 1, "name": "scratch",    "color": [242, 36, 36], "active": true }
  ]
}
```

- `color`: RGB array `[R, G, B]`
- `version`, `ignore_index`, `active` fields are optional

### Import Processing Flow

```
1. Select ZIP -> a project is created from the ZIP file name
2. Archive is scanned: images/, masks/, classes.json, metadata.json (deeper folders are skipped)
3. Images are converted to PNG (in parallel) and registered
4. Masks are matched via metadata.json items or stem name, and each image's marks are restored
5. Class definitions are registered (classes.json or metadata.json)
6. Orphan class IDs found in masks are auto-reconciled
7. Project list refreshes
```

### Response

```json
{ "status": "ok", "image_count": 5, "mask_count": 4, "clean_count": 2,
  "classes_imported": true, "reconciled_classes": 0,
  "passed_over": 2, "passed_over_sample": ["images/old/img_006.png", "img_007.png"] }
```

`passed_over` counts the pictures in the archive that lay outside the place images are read from and were not imported, and `passed_over_sample` names up to five of them. An export's own `training/` folder is not counted: its runs' charts are left behind by design. The import message in the Projects tab, and in the image list for a ZIP dropped on it, adds the count.

### Constraints

- An error is raised if the ZIP contains no images where they are read from
- Import brings back the dataset: images, masks, classes and each image's marks. Training runs, models and the train/val split files in the archive are not imported
- Importing masks without images is not supported
- The class definition file is optional (import works without it)
- Mask files must be `.png`; other formats in `masks/` are ignored

---

## External Tool Integration Guide

### Importing from Other Tools into Seg-Studio

Prepare a ZIP archive with the following structure:

```
project_name.zip
├── images/    # Original images
├── masks/     # Grayscale PNG with class IDs (match stem names with images)
└── classes.json  # Class definitions (optional)
```

### Exporting from Seg-Studio to Other Tools

When you extract the export ZIP:

- `images/`: Original images (original filenames)
- `masks/`: Grayscale masks (filenames are item IDs)
- `metadata.json`: The `items` array maps item IDs to original filenames and records each image's marks (OK, Review, train/test)
- `train.txt` / `val.txt`: Training/validation split information
- `training/`: Training runs (model checkpoints, configs, metrics) — ignore if you only need the dataset. Import does not read it back
