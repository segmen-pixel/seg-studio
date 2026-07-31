# segcore

`segcore` is the training core of [Seg-Studio](https://github.com/segmen-pixel/seg-studio):
a self-contained PyTorch library for semantic-segmentation training and
inference. The Trainer API and the bundled CLI scripts are thin layers on
top of it, and it can be used on its own without the web application.

## What's inside

| Module | Purpose |
| --- | --- |
| `segcore.training` | Models (`simpleunet`, `stdc`), datasets, losses, the training loop, metrics (mIoU / F1 / calibration), sliding-window inference, and knowledge distillation |
| `segcore.augment` | Synthetic data generation (Perlin CutPaste, lighting variants) |
| `segcore.auto_select` | Automatic model/config selection from dataset image profiles |
| `segcore.runtime` | Host resource probing and DataLoader planning |
| `segcore.postprocess` / `segcore.image_io` | Mask post-processing and Unicode-safe image IO |

## Install

From the repository root:

```bash
pip install -e packages/segcore
```

Requires Python >= 3.10 and installs `torch`, `torchvision`, `numpy`,
`pillow`, `scikit-learn`, `scipy`, and `opencv-python-headless`.
Distillation extras: `pip install -e "packages/segcore[distill]"`.
Supported distillation teachers are DINOv2 (`dinov2_vitb14` /
`dinov2_vitl14`) and SAM2 (`sam2.1_hiera_*`) — both Apache-2.0.

## Quick start

The stable public surface is re-exported at the package top level:

```python
from segcore import MODEL_REGISTRY, TrainConfig, build_model, __version__

print(__version__)           # e.g. "0.9.8"
print(list(MODEL_REGISTRY))  # ["simpleunet", "stdc"]

cfg = TrainConfig(
    input_size=[512, 512],
    output_stride=4,
    epochs=100,
    batch_size=8,
    lr=1e-3,
    ignore_index=255,
    normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
)
model = build_model(
    "simpleunet",
    num_classes=2,
    output_stride=cfg.output_stride,
    base_channels=32,
)
```

Metric helpers (`compute_miou`, `accumulate_confusion_matrix`,
`accumulate_f1_stats`, `finalize_f1`, `finalize_metrics`) are also exported
from the top level. Anything not re-exported in `segcore/__init__.py` is
internal and may change between releases without notice.

## License

Apache License 2.0 — see the repository root [LICENSE](../../LICENSE).
Copyright 2026 Segmen-Pixel and Seg-Studio contributors.
