# Benchmarks

Architecture-level inference benchmarks for the three Seg-Studio segmentation
backbones. These numbers are produced by [`scripts/benchmark.py`](scripts/benchmark.py),
which runs on **synthetic input and requires no dataset**, so anyone can
reproduce them on their own hardware:

```bash
python scripts/benchmark.py --device cuda:0    # GPU
python scripts/benchmark.py --device cpu       # CPU
```

## Test setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| CPU | Intel Core i9-13900KF (32 threads) |
| Framework | PyTorch 2.6.0 + CUDA 12.4 (Windows) |
| Input | 256×256 RGB, `output_stride=2`, `num_classes=2` |
| Batch size | 1 (single-image latency) |
| Method | 20 warmup + 100 timed iterations (GPU); 5 + 30 (CPU); **median** reported |
| VRAM | peak `torch.cuda.max_memory_allocated` (accurate on WDDM, where `nvidia-smi` is not) |

Parameter counts and model sizes are measured on the instantiated model with
default settings (`base_channels` as shipped, `use_se=True`), i.e. the exact
weights that get serialized into a checkpoint.

## Inference performance

Single image, 256×256, batch size 1.

| Architecture | Params | Model size | GPU latency | GPU throughput | GPU VRAM | CPU latency | CPU throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SimpleUNet** (bc=64) | 1.91 M | 7.3 MB | 2.95 ms | 339 img/s | 94 MB | 35.8 ms | 28 img/s |
| **STDC** (bc=32) | 2.92 M | 11.2 MB | **1.32 ms** | **758 img/s** | **20 MB** | **8.2 ms** | **122 img/s** |
| **DeepLabV3+** (bc=32) | 4.83 M | 18.5 MB | 5.06 ms | 198 img/s | 44 MB | 40.8 ms | 25 img/s |

*GPU = RTX 3090, CPU = Core i9-13900KF. Lower latency / higher throughput is better.*

**Takeaways**

- Every architecture runs in **single-digit milliseconds on a consumer RTX 3090** — fast
  enough for interactive annotation assist and live inspection.
- **STDC** leads on throughput (758 img/s on GPU, and **122 img/s even on CPU**) at the
  smallest VRAM footprint (20 MB), making it the default pick for edge and CPU-only deployments.
- **SimpleUNet** balances accuracy and speed; **DeepLabV3+** trades speed for a larger
  receptive field on high-resolution targets.
- Model sizes are 7–19 MB — small enough to bundle inside CoreML / ONNX exports for
  on-device inference.

## Accuracy (coming soon)

Accuracy figures (F1 / mIoU) will be added here, measured on a **public,
redistributable dataset** so they are fully reproducible. Numbers from internal
datasets are intentionally excluded from this document.

## Notes

- Latency is GPU/CPU compute only and excludes image decode and pre/post-processing.
- Throughput is derived from median single-image latency at batch size 1; larger
  batches achieve higher aggregate throughput.
- Re-run `scripts/benchmark.py --help` for all options (`--input`, `--runs`, `--warmup`, `--out`).
