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
│   └── runs/{run_id}/...
└── metadata.json    # Project metadata
```

### masks/ Format

| Field | Value |
|-------|-------|
| Filename | `{item_id}.png` (item ID) |
| Channels | Single channel (grayscale, PIL mode `"L"`) |
| Pixel values | Class ID (0 = background, 1+ = user-defined classes) |
| Ignore index | 255 (unannotated regions) |

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
    { "id": "item-uuid", "filename": "sample_001.png" }
  ]
}
```

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

#### Pattern B: Nested Structure

```
MyProject.zip
├── datasets/
│   └── prepared/
│       ├── images/
│       │   ├── img_001.png
│       │   └── ...
│       └── masks/
│           ├── img_001.png
│           └── ...
└── classes.json
```

> `images/` and `masks/` are detected at **any depth** inside the archive — what matters is the immediate parent folder name. Image files at the ZIP root (with no `images/` folder) are also accepted.

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
| Channels | Single channel (grayscale) recommended. For RGB images, the R channel is used as the class ID |
| Pixel values | Class ID (0 = background, 1+ = user-defined classes) |

### Mask-to-Image Matching Rules

If the ZIP contains a `metadata.json` (i.e. it is a Seg-Studio export), masks are matched through its `items` array (original filename → item ID). Otherwise images and masks are matched by **filename stem** (filename without extension).

```
images/bolt_001.png  <-->  masks/bolt_001.png   (stem: bolt_001)
images/sample.jpg    <-->  masks/sample.png     (stem: sample)
```

Images without a matching mask are imported as unannotated.

### Class Definition File

A `classes.json` file is auto-detected (at any depth within the archive). If it is absent but `metadata.json` contains a `classes` array, the classes are taken from there instead.

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
2. Archive is scanned: images/, masks/, classes.json, metadata.json
3. Images are converted to PNG (in parallel) and registered
4. Masks are matched via metadata.json items or stem name
5. Class definitions are registered (classes.json or metadata.json)
6. Orphan class IDs found in masks are auto-reconciled
7. Project list refreshes
```

### Constraints

- An error is raised if the ZIP contains no images
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
- `metadata.json`: The `items` array maps item IDs to original filenames
- `train.txt` / `val.txt`: Training/validation split information
- `training/`: Training runs (model checkpoints, configs, metrics) — ignore if you only need the dataset
