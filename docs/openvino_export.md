# OpenVINO IR Export

Seg-Studio can export a trained model as an **OpenVINO Intermediate
Representation (IR)** — the `.xml` + `.bin` pair consumed by Intel's
[OpenVINO Runtime](https://docs.openvino.ai/) on CPU, integrated GPU
(Iris/UHD), and NPU. This is the recommended deployment format for
**Intel-based edge inspection PCs** where CUDA is unavailable.

This page covers:

1. When to pick OpenVINO over CoreML / ONNX
2. The three precisions and what they buy you
3. Installing the optional dependencies
4. Triggering the export from the UI or the API
5. Loading the IR on a deployment box

For CoreML (Apple) see `handbook.md` §13 (Export the Model) or `user-guide.md`.

---

## When OpenVINO is the right format

| Format     | Target              | Notes                                       |
| ---------- | ------------------- | ------------------------------------------- |
| CoreML     | macOS / iOS         | Apple-native; supports on-device update.    |
| ONNX       | Cross-vendor        | Largest portability, no automatic INT8.     |
| **OpenVINO** | **Intel CPU/iGPU/NPU** | **Best CPU throughput; INT8 4× speedup.** |

Use OpenVINO when:

- The inference machine has no NVIDIA GPU.
- You want the lightest install for an Intel inspection PC.
- INT8 quantization (≈4× faster, ≈4× smaller) is acceptable.

---

## Precisions

| Mode   | File size (relative) | Latency (relative) | Accuracy        | When to pick                              |
| ------ | -------------------- | ------------------ | --------------- | ----------------------------------------- |
| FP32   | 1×                   | 1×                 | Baseline        | Sanity-check, debugging.                  |
| FP16   | ½×                   | ~1× CPU / faster iGPU/NPU | Near-zero loss | **Default for most edge deployments.**   |
| INT8   | ¼×                   | ~3–4× CPU          | Small drop      | When CPU throughput is the bottleneck.    |

INT8 uses **post-training quantization via NNCF** with calibration
patches automatically sampled from `prepared/splits/val.txt`. No extra
upload is required — the same val images used during training are
re-used here. The first 100 patches are sufficient for stable
calibration.

> **Tip.** If INT8 accuracy is unacceptable, switch back to FP16 for the
> same model — no re-training needed; only the export step changes.

---

## Installing OpenVINO

OpenVINO and NNCF are **optional dependencies** (~300 MB of wheels) and
not bundled by default. Install them at setup time:

```bat
scripts\windows\install_windows.bat --with-openvino
```

…or manually, into an existing venv:

```bat
.venv-windows\Scripts\python.exe -m pip install -r apps\trainer_api\requirements-openvino.txt
```

(`requirements-openvino.txt` is the lockfile compiled from
`requirements-openvino.in` — install from the lockfile, not the `.in`.)

The packages are Apache-2.0 licensed:

- `openvino` — [github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE)
- `nncf` (INT8 quantization) — [github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf/blob/develop/LICENSE)

If `openvino` is missing at export time, the API returns HTTP 501 with a
clear message rather than crashing.

---

## Triggering the export

### From the UI

1. Open the **Training** tab and select a completed run.
2. Click the **OpenVINO export** dropdown next to the regular Export
   button.
3. Choose **FP32**, **FP16**, or **INT8**.
4. A `<project>_openvino_<precision>.zip` is downloaded. Unzip it on
   the deployment machine to get `model.xml` + `model.bin`.

### From the API

```bash
# FP32 / FP16
curl -X POST -o model.zip \
  "http://localhost:8002/api/v1/projects/<project_id>/train/runs/<run_id>/export/openvino?precision=fp16"

# INT8 (uses prepared/splits/val.txt for calibration)
curl -X POST -o model_int8.zip \
  "http://localhost:8002/api/v1/projects/<project_id>/train/runs/<run_id>/export/openvino?precision=int8"
```

The response is always a zip containing exactly two files:

```
model.xml   # OpenVINO graph description
model.bin   # quantized / packed weights
```

OpenVINO IR is a two-file format; both files are required at inference
time and must sit alongside each other.

---

## Loading the IR on an edge device

Minimal Python sample. The same script works on any Intel CPU, on iGPU
(Gen9.5+), and on NPU (Core Ultra series).

```python
# pip install openvino numpy pillow
import numpy as np
import openvino as ov
from PIL import Image

core = ov.Core()
print("Available devices:", core.available_devices)
# Typically ["CPU"], or ["CPU", "GPU"] when Intel Graphics drivers are present.

# Pick a device. "AUTO" lets the runtime pick the fastest; "CPU" forces CPU.
DEVICE = "AUTO"
model = core.read_model("model.xml")        # finds model.bin automatically
compiled = core.compile_model(model, DEVICE)
infer_request = compiled.create_infer_request()

# Preprocess to whatever input_size the model was trained at (see
# train_config.json in the run directory). For a 256x256 RGB model:
img = Image.open("sample.jpg").convert("RGB").resize((256, 256))
arr = np.asarray(img, dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW

infer_request.infer({0: arr})
logits = infer_request.results[compiled.output(0)]  # shape [1, num_classes, H, W]
mask = logits.argmax(axis=1)[0]
print("Mask shape:", mask.shape, "unique classes:", np.unique(mask))
```

### Choosing a device

- `"CPU"` — works everywhere; INT8 here gives the biggest speedup
  thanks to AVX2/AVX-512 VNNI.
- `"GPU"` — Intel iGPU (UHD / Iris Xe). FP16 typically beats CPU; INT8
  is **not** supported on most iGPUs (older generations lack the XMX
  instructions). The runtime will refuse to compile and you should
  fall back to FP16.
- `"NPU"` — Core Ultra ("Meteor Lake" and newer). Lowest power; FP16
  recommended.
- `"AUTO"` — lets OpenVINO pick the fastest available device for the
  given precision.

### Performance hints

If the same model is benchmarked at FP32 / FP16 / INT8 on the same
machine, FP16 ≈ FP32 on plain CPU but **iGPU/NPU gain ~2×**. INT8 on
CPU is the one that moves the needle (often **3–4× speedup over FP32**)
because it activates VNNI integer dot-product instructions on Intel Core
8th-gen and later.

---

## Troubleshooting

| Symptom                                              | Cause / fix                                                                   |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| `HTTP 501 openvino is required ... not installed`    | Rerun installer with `--with-openvino`, or `pip install openvino nncf`.       |
| `HTTP 400 INT8 quantization requires a prepared val` | Run **Prepare dataset** first; INT8 needs `prepared/splits/val.txt`.          |
| INT8 mask noticeably worse than FP16                 | Try FP16 — it ships near-FP32 accuracy. INT8 isn't free; calibrate carefully. |
| iGPU refuses INT8 model                              | Older Intel Graphics lack XMX. Use FP16 on iGPU or INT8 on CPU.               |
| Inference output has wrong shape                     | Confirm input shape matches the run's `train_config.json:input_size`.         |

---

## License attribution

Models exported from Seg-Studio inherit your project's license. The
OpenVINO Runtime and NNCF packages themselves are Apache-2.0; their
license texts are included verbatim in the wheel metadata under
`<venv>/Lib/site-packages/openvino-*.dist-info/` and
`nncf-*.dist-info/`.
