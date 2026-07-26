# Troubleshooting

## Server Won't Start / Cannot Connect to API

**Symptom:** The browser keeps showing "Connecting to API server..."

1. Check the service status:
   - Windows: `scripts\windows\status_windows.bat`
   - macOS: `lsof -iTCP:8002 -sTCP:LISTEN`
2. Check the log files:
   - Windows: `logs\windows\trainer_<timestamp>.log` (one per start; `scripts\windows\status_windows.bat` tails the newest)
   - macOS: `logs/macos/trainer.log`
3. Verify port 8002 is not occupied by another process:
   ```bash
   # Windows
   netstat -ano | findstr :8002
   # macOS
   lsof -i :8002
   ```
4. Ensure the Python virtual environment is properly activated

## Blank White Screen

**Symptom:** Navigating to `http://localhost:8002/ui/` shows a blank page

1. Check if the UI has been built: verify that the `apps\trainer_ui\dist\` folder exists
2. If it hasn't been built:
   ```bat
   cd apps\trainer_ui
   npm install
   npm run build
   ```
3. If using the dev server, navigate to `http://localhost:5173` instead

## OOM (Out of Memory) Error

**Symptom:** CUDA out of memory error during training

1. **Reduce the patch size:** Lower it in the Training tab settings (e.g., 256 → 192 → 128)
2. **Reduce the batch size** (default 8; try 4 → 2 → 1)
3. **Close other GPU-intensive apps:** Disable GPU acceleration in your browser
4. **4 GB VRAM GPUs:** Low-VRAM mode is applied automatically, but if you still get OOM errors, reduce the patch size to 128
5. Windows/Linux: Ensure the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (enabled by default)
6. **macOS (MPS):** Unified memory is shared with other apps. If you run out of memory, reduce the input size or close unnecessary applications

## CUDA Not Detected (Windows / Linux)

**Symptom:** Only CPU appears as an option even though a GPU is available

1. Verify your NVIDIA driver is up to date: `nvidia-smi`
2. Check that the CUDA-enabled PyTorch is installed:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. If it prints `False`, reinstall PyTorch with CUDA support:
   ```bash
   # cu128 for Turing/RTX 20xx and newer (incl. Blackwell); use cu124 only
   # on Maxwell/Pascal/Volta
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   On Windows you can instead re-run `install-windows.bat cuda`, or
   `install-windows.bat cuda124` on older GPUs.
4. Manually select the device with the **Device** selector in the Training tab

## MPS Not Detected (macOS)

**Symptom:** MPS device does not appear on an Apple Silicon Mac

1. Verify you are running macOS 12 (Monterey) or later
2. Check that MPS-compatible PyTorch is installed:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```
3. If it prints `False`, reinstall PyTorch:
   ```bash
   pip install --upgrade torch torchvision
   ```
4. MPS is not available on Intel Macs (CPU only)

## SAM Assist Not Working

**Symptom:** Errors when using SAM click / box segmentation

1. Verify checkpoint files exist: `models\sam_checkpoints\`
2. Required files: `mobile_sam.pt`, `sam2.1_hiera_tiny.pt`, etc.
3. Confirm `checkpoint_exists: true` via the `/api/v1/sam/models` endpoint
4. Check the logs for specific errors: `logs\trainer_errors.log`

## Export Errors

**Symptom:** CoreML / ONNX export fails

1. **CoreML:** Ensure `coremltools` is installed (macOS recommended)
   ```bash
   pip install coremltools
   ```
2. **ONNX:** Ensure `onnx` and `onnxruntime` are installed
3. Make sure you have selected a run that completed successfully (status: completed)

## Counting: Training Won't Start

**Symptom:** Instance (count) training fails immediately with "instance mode
needs at least 4 annotated images containing one of classes [...]"

1. **Count what actually has the class**, not how many masks exist. A mask
   painted entirely with value 255 is "unpainted", not foreground, and counts
   for nothing. Open a few masks in the Annotate tab and confirm the class
   colour is really there.
2. **Mark Clean images do not count** — they are all-background by definition.
3. The check runs before any GPU work, so this fails in seconds. The message
   names the classes it looked for; if that list is not what you expected,
   check which classes are **active** in the Classes panel.

## Counting: The Count Is Wrong

**Symptom:** The reported count is consistently too low or too high

1. **Look at the overlay first.** Turn on **Detection highlight** in Results —
   background greys out and every object gets its own colour. Two touching
   parts sharing one colour means they were merged; one part with two colours
   means it was split.
2. **Consistently ~2x too high:** objects are being counted once per tile.
   Check the export contract has `patch_size`; a run trained with tiling but
   inferred without it (or the reverse) miscounts silently.
3. **Consistently low, near a round number:** you may be at the model's
   per-image detection ceiling — 100 for the default Small model, 200 for
   Medium/Large. The `/count` response flags this with `truncation_warning`,
   but only for models exported with tiling off (`instance_patch_size = 0`);
   with the default 768 tiling no warning is emitted, so judge saturation
   from the count itself. Crop tighter or split the frame.
4. **Low by a few:** objects larger than the tile overlap can straddle a seam
   and be dropped. At the default 768 patch the overlap is 192px. If your
   parts are bigger than that, raise `instance_patch_size`.
5. **Check the threshold.** `/count` echoes the `threshold` it used. It is
   calibrated on held-out images at training time; a very different scene may
   want a different one.

## Counting: Synthetic Samples Look Wrong

**Symptom:** The preview strip shows composed images that do not resemble your
real scenes

1. **Objects too large or too small:** set the area band explicitly instead of
   `0` (auto). Auto estimates it from your blobs, and a few stray specks or
   one over-large blob skews it.
2. **Too few cutouts:** the composition log reports `n_cutouts` and how many
   blobs were excluded by the area band. If most were excluded, the band is
   wrong for your data.
3. **Backgrounds look repetitive:** only annotated images become background
   plates. Annotate a few more, from different areas of the scene.

## Counting: `POST /count` Returns 409

**Symptom:** `the active model is a semantic-segmentation export; use /segment`

The activated model in the serving registry is a semantic model, not a
counting one. Export the counting run with **Export → ONNX (Serving)** and
activate that model id. Counting exports carry an `instance_inference.json`
alongside `model.onnx`; semantic ones do not.

## Checking Logs

The API writes its own rotating log files; the launchers separately capture
whatever the process prints to the console. Both are useful, and they are not
the same file.

| What | Where |
|---|---|
| Everything the API logs | `logs\app.log` under the repo root (10 x 20 MB rotation) |
| Warnings and errors only | `logs\trainer_errors.log` under the repo root (5 x 10 MB rotation) |
| Both, on a packaged Windows install | `%LOCALAPPDATA%\Seg-Studio\logs\` |
| Console capture, Windows launcher | `logs\windows\trainer_<timestamp>.log` and `serving_<timestamp>.log` (one pair per start) |
| Console capture, macOS launcher | `logs/macos/trainer.log`, `logs/macos/serving.log` |
| Console capture, `scripts/start_local.sh` | `/tmp/seg_trainer.log`, `/tmp/seg_serving.log` |
| One training run | `projects/<project_id>/training/runs/<run_id>/train.log` |

- **JSON logs:** Set environment variable `LOG_FORMAT=json` for JSON output
- **Log level:** Set environment variable `LOG_LEVEL=DEBUG` for debug output

## Dependency Audit

To check for security vulnerabilities:
```bash
# Windows
scripts\audit.bat
# macOS / Linux
bash scripts/audit.sh
```

## Reporting a Problem

Nothing here helped? Open an issue with the
[bug report template](../.github/ISSUE_TEMPLATE/bug_report.md) at
<https://github.com/segmen-pixel/seg-studio/issues/new/choose>.

**Please include:**

- **Seg-Studio version** — shown next to the name in the app header, and
  returned as the `X-API-Version` header on every API response
- **OS and version** — e.g. Windows 11 23H2, macOS 14.5
- **GPU and driver** — paste `nvidia-smi`, or say "Apple Silicon M2" / "CPU only"
- **How you started it** — `start-windows.bat`, `bash start-macos.sh`,
  `scripts\windows\start_local_windows.bat`, `docker compose up`, ...
- **How you installed it** — release ZIP, git clone, or the Windows installer,
  plus `python --version`
- **The exact error text**, copied as text rather than photographed
- **The last ~50 lines of the log** — `trainer_errors.log` first, then
  `app.log` (see [Checking Logs](#checking-logs) for where they live):

```bash
# Windows (PowerShell)
Get-Content logs\trainer_errors.log -Tail 50
# macOS / Linux
tail -n 50 logs/trainer_errors.log
```

**Please do NOT include:**

- Your `.env` file, `SEG_API_TOKEN`, or the `api_token` value inside
  `projects/runtime_settings.json` — those are credentials, and an issue is
  public
- Customer or production images, masks, or exported models — crop, redact, or
  reproduce the problem with an image you are free to share
- Project names, file paths, host names or user names that identify a customer
  or a site — rename them before pasting
- Personal data of any kind

Log lines carry file paths and project names, so skim what you paste. For a
**security vulnerability**, do not open a public issue — follow
[SECURITY.md](../SECURITY.md).
