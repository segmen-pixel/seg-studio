# Developer Quickstart

Get the Seg-Studio development environment running on Windows, macOS, or Linux/WSL.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | 3.10 works but 3.11+ recommended |
| Node.js | 18+ | Required for UI build and dev server |
| npm | 9+ | Bundled with Node.js |
| NVIDIA GPU (CUDA) | Required for training | CUDA 12.8 (cu128); cu124 for older GPUs. Annotation and inference work without one. |
| Apple Silicon | Optional | MPS acceleration on macOS (M1/M2/M3/M4) |

Verify prerequisites:

```bash
python --version      # 3.11+
node --version        # v18+
npm --version         # 9+
nvidia-smi            # optional — confirms CUDA driver
```

## Clone and Setup

### 1. Create a virtual environment

**Windows:**
```bat
cd seg-studio
python -m venv .venv-windows
.venv-windows\Scripts\activate
```

**macOS:**
```bash
cd seg-studio
python3 -m venv .venv-macos
source .venv-macos/bin/activate
```

**Linux / WSL:**
```bash
cd seg-studio
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r apps/trainer_api/requirements.txt
pip install -e packages/segcore
```

(`segcore` is the local training-core package; the editable install is the
documented path. If it is missing, the API falls back to a `sys.path` shim.)

For CUDA-enabled PyTorch (Windows/Linux with NVIDIA GPU; use `cu124` instead
of `cu128` for older Maxwell/Pascal/Volta GPUs):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For macOS (default PyPI wheels include MPS support on Apple Silicon):

```bash
pip install torch torchvision
```

For CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install UI dependencies

```bash
cd apps/trainer_ui
npm install
cd ../..
```

## Starting the API Server

The API auto-builds the UI on first startup if `dist/` is missing or stale.

**Windows (quick start):**
```bat
scripts\windows\start_api_only.bat
```

**Windows (full stack with serving API + UI dev server):**
```bat
scripts\windows\start_local_windows.bat
```

**macOS (full stack):**
```bash
bash scripts/macos/start_local_macos.sh
```

**macOS (API only):**
```bash
bash scripts/macos/start_api.sh
```

**Linux / WSL:**
```bash
# Activate venv first, then from the repo root:
export SEG_PROJECTS_DIR="$(pwd)/projects"
export SEG_DB_PATH="$(pwd)/projects/app.db"
python -m uvicorn apps.trainer_api.app.main:app --port 8002
# To allow LAN access: add --host 0.0.0.0 (or set SEG_HOST=0.0.0.0)
```

The API binds to port **8002**. On first launch, heavy modules (PyTorch, OpenCV,
scikit-learn) are loaded in the background. A loading screen is shown in the
browser until startup completes (~5-15 seconds depending on hardware).

## Starting the UI Dev Server

For frontend development with hot-reload:

```bash
cd apps/trainer_ui
npm run dev
```

This starts Vite on **http://localhost:5173** with proxy rules that forward API
calls to `localhost:8002`. Edit React/TypeScript source files and see changes
instantly.

For production builds served directly by the API:

```bash
cd apps/trainer_ui
npm run build
```

Then access the UI at **http://localhost:8002/ui/**.

## Verify Connection

```bash
curl http://localhost:8002/api/v1/health
```

Expected response (JSON with version, disk, RAM, GPU info):

```json
{
  "status": "ok",
  "version": "1.0.0",
  ...
}
```

Other useful endpoints:

| URL | Description |
|-----|-------------|
| http://localhost:8002/ui/ | Built UI (production) |
| http://localhost:8002/docs | Swagger API docs |
| http://localhost:8002/startup-status | Startup progress |
| http://localhost:5173 | Vite dev server (if running) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEG_PROJECTS_DIR` | `<repo>/projects` | Root directory for all project data |
| `SEG_DB_PATH` | `<repo>/projects/app.db` | SQLite database path |
| `SEG_MODELS_DIR` | `<repo>/models` | Model registry and SAM checkpoints |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `text` | Log format: `text` or `json` |
| `SEG_HOST` | `127.0.0.1` | Bind address. Set to `0.0.0.0` for LAN access |
| `SEG_TORCH_DEVICE` | `auto` | Force device: `cuda:0`, `cpu`, or `auto` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA memory allocator config (set by start scripts) |

## Common Issues

### ECONNREFUSED on port 8002

The API server is not running or has not finished starting.

1. Check that the API process is alive (look for `uvicorn` in your process list).
2. Wait for background startup to complete. The `/startup-status` endpoint
   returns `{"ready": false}` until all routers are registered.
3. Check logs for errors:
   - Windows: `logs\windows\trainer.log`
   - macOS: `logs/macos/trainer.log`
   - Linux: console output or `/tmp/seg_trainer.log`

### CUDA Not Found

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`:

1. Verify NVIDIA driver is installed: `nvidia-smi`
2. Reinstall CUDA-enabled PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
3. Ensure `CUDA_HOME` / `CUDA_PATH` are set if using a non-standard install.

### Vite Proxy Errors (404 on API calls from dev server)

The Vite dev server proxies API calls to `localhost:8002`. If the API is running
on a different host or port, edit `apps/trainer_ui/vite.config.mjs`:

```js
proxy: {
  "/api": { target: "http://YOUR_API_HOST:8002", ws: true },
  // ... other routes (/v2, /ws, /health, /version, /startup-status)
}
```

### TypeScript Build Errors

```bash
cd apps/trainer_ui
npx tsc --noEmit
```

Fix any reported type errors before committing. The production build
(`npm run build`) runs `tsc -b` first and will fail on type errors.

### Port Already in Use

```bash
# Windows
netstat -ano | findstr :8002

# macOS / Linux
lsof -i :8002
```

Kill the conflicting process or choose a different port with `--port`.

### Blank White Page at /ui/

The UI has not been built. Run:

```bash
cd apps/trainer_ui
npm install
npm run build
```

Or start the Vite dev server (`npm run dev`) and access `http://localhost:5173`.

## Stopping Services

**Windows:**
```bat
scripts\windows\stop_local_windows.bat
```

**macOS:**
```bash
bash scripts/macos/stop_local_macos.sh
```

**Linux:**
```bash
# If started in foreground, Ctrl+C stops uvicorn.
# If backgrounded:
pkill -f "uvicorn apps.trainer_api"
```

## Running Tests

See `scripts/test.sh` (Linux/WSL) or `scripts/test.bat` (Windows) for the
unified test runner that checks TypeScript, linting, Python imports, and
optionally E2E tests.

```bash
# Linux / WSL
bash scripts/test.sh

# Windows
scripts\test.bat
```
