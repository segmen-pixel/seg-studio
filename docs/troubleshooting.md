# Troubleshooting

## Server Won't Start / Cannot Connect to API

**Symptom:** The browser keeps showing "Connecting to API server..."

1. Check the service status:
   - Windows: `scripts\windows\status_windows.bat`
   - macOS: `lsof -iTCP:8002 -sTCP:LISTEN`
2. Check the log files:
   - Windows: `logs\windows\trainer.log`
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
   # cu124 for most GPUs; use cu128 for RTX 50-series (Blackwell)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
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

## Checking Logs

- **API server logs:** Console output (with timestamps)
- **Error logs:**
  - Windows: `logs\trainer_errors.log`
  - macOS: `logs/macos/trainer.log`
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
