# Third-Party Notices

Seg-Studio is licensed under Apache License 2.0 (see `LICENSE`). It bundles or
depends on the following third-party components. Each component is governed by
its own license; the terms below are summaries — refer to each project's source
for the authoritative text.

**Scope:** This file lists components that are *redistributed* with Seg-Studio,
either bundled in the Windows installer or fetched at install time on macOS.
Build- and development-only tools (pytest, ruff, mypy, eslint, type stub
packages, etc.) that are never shipped to end users are not listed here.

For the upstream NOTICE attributions required by Apache License 2.0 §4(d),
see `NOTICE` at the repository root.

Last updated: 2026-05-21 (for Seg-Studio v0.9.x — auto_select v6 swap).

---

## Bundled at runtime (included in installer)

### Embedded Python runtime
- **python-build-standalone** — MIT License / PSF
  - https://github.com/indygreg/python-build-standalone

### Python: core server
| Package | License | Source |
|---|---|---|
| fastapi | MIT | https://github.com/fastapi/fastapi |
| starlette | BSD-3-Clause | https://github.com/encode/starlette |
| uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| pydantic | MIT | https://github.com/pydantic/pydantic |
| sqlmodel | MIT | https://github.com/fastapi/sqlmodel |
| sqlalchemy | MIT | https://github.com/sqlalchemy/sqlalchemy |
| python-multipart | Apache-2.0 | https://github.com/Kludex/python-multipart |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| jinja2 | BSD-3-Clause | https://github.com/pallets/jinja |
| python-dotenv | BSD-3-Clause | https://github.com/theskumar/python-dotenv |

### Python: ML / inference
| Package | License | Source |
|---|---|---|
| torch (PyTorch) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchvision | BSD-3-Clause | https://github.com/pytorch/vision |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy |
| scipy | BSD-3-Clause | https://github.com/scipy/scipy |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-image | BSD-3-Clause | https://github.com/scikit-image/scikit-image |
| pillow | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |
| opencv-python-headless | Apache-2.0 | https://github.com/opencv/opencv-python |
| onnx | Apache-2.0 | https://github.com/onnx/onnx |
| onnxruntime / onnxruntime-gpu | MIT | https://github.com/microsoft/onnxruntime |
| xgboost | Apache-2.0 | https://github.com/dmlc/xgboost |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| coremltools | BSD-3-Clause | https://github.com/apple/coremltools |
| matplotlib | PSF-based / Matplotlib License | https://github.com/matplotlib/matplotlib |
| openpyxl | MIT | https://foss.heptapod.net/openpyxl/openpyxl |
| weasyprint | BSD-3-Clause | https://github.com/Kozea/WeasyPrint |
| onnxscript | MIT | https://github.com/microsoft/onnxscript |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| tqdm | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| openvino (optional, `--with-openvino`) | Apache-2.0 | https://github.com/openvinotoolkit/openvino |
| nncf (optional, `--with-openvino`) | Apache-2.0 | https://github.com/openvinotoolkit/nncf |
| pywinpty (Windows installer) | MIT | https://github.com/andfoy/pywinpty |
| label-studio (optional, installer opt-in) | Apache-2.0 | https://github.com/HumanSignal/label-studio |

### Python: data / storage
| Package | License | Source |
|---|---|---|
| pyvips | MIT | https://github.com/libvips/pyvips |
| zarr | MIT | https://github.com/zarr-developers/zarr-python |

### Python: `seg-inference-sdk` (client-side)
| Package | License | Source |
|---|---|---|
| requests | Apache-2.0 | https://github.com/psf/requests |
| websocket-client | Apache-2.0 | https://github.com/websocket-client/websocket-client |
| websockets | BSD-3-Clause | https://github.com/python-websockets/websockets |

### Segment Anything family
| Component | License | Source | Notes |
|---|---|---|---|
| Segment Anything Model (SAM) | Apache-2.0 | https://github.com/facebookresearch/segment-anything | Meta / FAIR |
| SAM 2 | Apache-2.0 | https://github.com/facebookresearch/sam2 | Meta / FAIR |
| MobileSAM | Apache-2.0 | https://github.com/ChaoningZhang/MobileSAM | ChaoningZhang et al. |
| TinySAM | Apache-2.0 | https://github.com/xinghaochen/TinySAM | Huawei Noah's Ark Lab |
| EfficientSAM | Apache-2.0 | https://github.com/yformer/EfficientSAM | Yunyang Xiong et al. |

SAM/SAM2/MobileSAM/TinySAM/EfficientSAM **model weights (checkpoints)** are
redistributed from HuggingFace (`segmen-pixel/seg-studio`) under the same
licenses as their upstream repositories.

### DINOv2 (feature distillation teacher)
- **DINOv2** — Apache-2.0 — Meta / FAIR
  - https://github.com/facebookresearch/dinov2
  - Pretrained weights (`dinov2_vitb14_pretrain.pth`) are redistributed under
    the Apache-2.0 license.

---

## Frontend (`apps/trainer_ui`)

Only the packages whose code is compiled into the shipped UI bundle are
listed; the build toolchain (vite, typescript, eslint, playwright, type
stubs) is development-only and out of scope per the note above.

| Package | License | Source |
|---|---|---|
| react | MIT | https://github.com/facebook/react |
| react-dom | MIT | https://github.com/facebook/react |
| zustand | MIT | https://github.com/pmndrs/zustand |
| immer | MIT | https://github.com/immerjs/immer |
| openseadragon | BSD-3-Clause | https://github.com/openseadragon/openseadragon |

---

## Fonts, icons, and media

- Seg-Studio icon (`build/installer/launcher/seg-studio.ico`,
  `apps/trainer_ui/public/favicon.ico`): © Seg-Studio contributors,
  distributed under Apache-2.0 (part of this project).
- Any web fonts loaded at runtime are served by the user's browser from
  their standard sources; no fonts are redistributed by this project.

---

## Attribution notes

External components are either:
- installed at runtime from public package registries (PyPI, npm),
- redistributed in binary form (model checkpoints) from upstream repositories
  under their original licenses, or
- adapted in source form with attribution preserved in the file header
  (e.g. `apps/trainer_api/app/core/colormap.py` adapts the Turbo colormap
  LUT from Anton Mikhailov / Google LLC, Apache-2.0).

No proprietary or commercially-licensed third-party source code is embedded
in this project.

## NVIDIA libraries via PyTorch CUDA wheel (Windows installer)

The Windows installer ships the PyTorch CUDA wheel
(`torch==2.x.x+cuXXX`), which transitively bundles NVIDIA libraries
(cuDNN, cuBLAS, cuFFT, cuRAND, cuSPARSE, NCCL, NVRTC, etc.) under their
respective redistributable licenses. Seg-Studio does not download or
package these libraries from NVIDIA directly; it inherits them from the
PyTorch wheel that PyTorch's maintainers redistribute under the
permissions granted by NVIDIA.

The aggregated obligations text is reproduced verbatim from each wheel's
own `LICENSE` / `NOTICE` / `ThirdPartyNotices.txt` files, copied into
`licenses/third_party/wheels/` at installer build time.

CUDA, cuDNN, cuBLAS, NCCL, NVRTC, NVTX, NVIDIA, the NVIDIA logo, and
related marks are trademarks of NVIDIA Corporation. PyTorch is a
trademark of The Linux Foundation.

## Transitive LGPL components (Windows installer)

The Windows installer ships pre-built binaries that include the following
LGPL-licensed components as dynamically-linked libraries. These can be
replaced by the user with compatible builds, as required by the LGPL.
Full license texts are bundled under `licenses/third_party/lgpl/` (see
the README in that directory for the obligation→file mapping and the
upstream-source URL for each component).

| Component | License | Upstream source |
|---|---|---|
| libvips | LGPL-2.1+ | https://github.com/libvips/libvips (used by `pyvips`) |
| Cairo | LGPL-2.1 / MPL-1.1 | https://gitlab.freedesktop.org/cairo/cairo (used by `WeasyPrint`) |
| Pango | LGPL-2.0+ | https://gitlab.gnome.org/GNOME/pango (used by `WeasyPrint`) |
| HarfBuzz | MIT | https://github.com/harfbuzz/harfbuzz (no LGPL obligation; co-shipped) |
| FFmpeg (cv2-bundled) | LGPL-2.1+ | LGPL build shipped inside `opencv-python` as `opencv_videoio_ffmpegXXXX_64.dll`; upstream https://github.com/FFmpeg/FFmpeg |
| pyphen | LGPL-2.1+ / MPL-1.1 / GPL-2.0+ tri-license; we elect **LGPL** | https://github.com/Kozea/Pyphen (used by `WeasyPrint`) |

## Trademarks

All trademarks, service marks, trade names, product names, and logos appearing
in this software are the property of their respective owners:

- Apple, CoreML, macOS, Apple Silicon — trademarks of Apple Inc.
- PyTorch — trademark of The Linux Foundation.
- NVIDIA, CUDA, cuDNN — trademarks of NVIDIA Corporation.
- ONNX — trademark of the LF AI Foundation.
- HuggingFace — trademark of Hugging Face, Inc.
- Microsoft, Windows, Playwright, ONNX Runtime — trademarks of Microsoft Corporation.
- SAM, SAM 2, DINOv2 — research models from Meta Platforms, Inc.

The use of these names is for identification and informational purposes only
and does not imply endorsement by their respective owners.

Corrections or omissions: please open an issue at
https://github.com/segmen-pixel/seg-studio/issues.
