---
title: "Seg-Studio — local GUI for image segmentation"
description: "Open-source desktop workbench for semantic segmentation: SAM-assisted annotation, PyTorch training, object counting, and ONNX / CoreML export. Runs fully offline on Windows and macOS."
---

# Seg-Studio

**Annotate, train and export image segmentation models — all on your own machine.**

Seg-Studio is an open-source, all-in-one workbench for semantic segmentation.
It runs as a local application on Windows and macOS: you annotate images with
SAM-assisted tools, train PyTorch models with live monitoring, and export the
result to ONNX or CoreML for edge deployment. There is no cloud account, no
upload step, and no per-seat licence — the images and the trained weights stay
on the machine you run it on.

Alongside semantic segmentation it can count individual objects, reusing the
masks you already drew rather than asking for a second round of annotation.

Current release: **v0.9.8 (beta)** · Apache-2.0 ·
[Download](https://github.com/segmen-pixel/seg-studio/releases/latest) ·
[Source](https://github.com/segmen-pixel/seg-studio) ·
[日本語](ja/)

![Annotate with SAM, train, evaluate and export — end to end in Seg-Studio](docs/images/hero.gif)

## Where to start

| If you are… | Go here |
|---|---|
| New here | [First Run Walkthrough](docs/first-run-manual.md) — from install to your first prediction in about 10 minutes |
| Looking up one feature | [User Guide](docs/user-guide.md) — reference for every tab, tool and setting |
| After the full detail | [Handbook](docs/handbook.md) — the same workflow carried end to end, 16 chapters |
| Stuck installing | [Troubleshooting](docs/troubleshooting.md) |
| Running it on a shared machine or LAN | [Deployment Guide](docs/deployment.md) |
| Contributing | [Contributing](CONTRIBUTING.md) · [Developer Quickstart](docs/dev-quickstart.md) |

## What it does

- **SAM-assisted annotation** — click-to-segment, brush and polygon tools, class
  management, autosave. Masks are plain indexed PNGs, so nothing is locked in.
- **Training without writing code** — pick a task, and the recipe (backbone,
  epochs, augmentation, learning rate) is chosen for you; every field stays
  editable if you want to override it.
- **Live monitoring and evaluation** — loss and metric curves during training,
  then per-image F1 / precision / recall / IoU, confidence distributions and a
  heatmap view for inspecting what the model actually learned.
- **Object counting** — count instances from the masks you already annotated.
- **Export for deployment** — ONNX, OpenVINO IR and CoreML, so a model trained
  here can run on an edge device, an iPhone or a PC without Seg-Studio installed.
- **Runs offline** — no cloud account, no telemetry, no upload of your images.

See the [Feature Catalog](docs/catalog.md) for the full list, and
[Benchmarks](BENCHMARKS.md) for measured numbers.

## Install

Three steps: download the ZIP, run `install` then `start`, and open
`http://localhost:8002/ui/` in a browser. The exact commands for Windows and
macOS are in the [README](README.md#quick-start), and the
[First Run Walkthrough](docs/first-run-manual.md) carries you from there to a
trained model.

Python 3.10+ is required. A CUDA GPU speeds training up considerably but is not
required — training on CPU works, and Apple Silicon is supported through MPS.

## Reference

- [Import / Export format specification](docs/import_export.md)
- [OpenVINO IR export](docs/openvino_export.md)
- [Auto-Select technical note](docs/technical_note_auto_select.md) ·
  [Auto-config rationale](docs/auto-config-rationale.md)
- [Roadmap](docs/ROADMAP.md) · [Changelog](CHANGELOG.md) ·
  [Security policy](SECURITY.md)

## Licence

Apache-2.0. Third-party components and their licences are listed in
[THIRD_PARTY_NOTICES](https://github.com/segmen-pixel/seg-studio/blob/main/THIRD_PARTY_NOTICES.md).
