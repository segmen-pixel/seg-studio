# Troubleshooting

**How to read this page.** Each problem starts with what you see on the screen.
Under it:

- **Try this first** — steps you can do with the mouse: restarting Seg-Studio,
  closing other apps, reloading the page, changing a setting. No commands.
- **Digging deeper (uses commands)** — steps that need a command window
  (Command Prompt / PowerShell on Windows, Terminal on macOS). If you are not
  comfortable with that, ask whoever set up Seg-Studio, or skip to
  [Reporting a Problem](#reporting-a-problem) at the end of this page.

Error messages are quoted exactly as Seg-Studio shows them, so you can search
this page for the text you see (Ctrl+F / Cmd+F).

> **Restarting Seg-Studio** comes up often below. On Windows, double-click
> `stop-windows.bat` and then `start-windows.bat` in the Seg-Studio folder
> (`restart-windows.bat` also
> restarts it, but refuses while a training run is running or queued). On macOS, run `bash stop-macos.sh`
> and then `bash start-macos.sh` in the Seg-Studio folder, the same way you
> started it.
> If you opened the **Seg-Studio** app, close its taskbar button (Windows) or
> quit it from its Dock icon (macOS), then open the app again.

## The Seg-Studio App Does Not Start

**Symptom:** Opening **Seg-Studio** from the Start menu, the desktop or
`~/Applications` shows no screen, or shows "Seg-Studio could not start" or
"Another program is using port 8002".

**Try this first**

1. Open the log folder: **Open the log folder** in the app's tray menu
   (Windows) or its Dock icon's menu (macOS), or `logs\windows` (`logs/macos`
   on macOS) in the Seg-Studio folder.
   `app_start.log` holds what the start script printed, `app.log` what the
   app itself did, and on macOS `app_console.log` what its launcher printed.
   The trainer API's own logs are one folder up, in `logs` (see
   [Checking Logs](#checking-logs)).
2. **"Another program is using port 8002"**: a program that is not
   Seg-Studio already serves that port; quit it, then open the app again. (A
   Seg-Studio from another folder is not this case: the app asks whether to
   stop it and start its own.)
3. **"The running Seg-Studio could not be stopped"**: stop it with
   `stop-windows.bat` / `bash stop-macos.sh` in the folder it was started
   from, then open the app again.
4. **Windows, no icon in the tray**: Windows 11 keeps a new tray icon in the
   hidden area; click the **^** arrow next to the clock. The app's button is in
   the taskbar either way.
5. **macOS, no Seg-Studio in the Dock**: the library behind the Dock icon and
   the menu bar item (pyobjc) is missing, so the app asks with a dialog
   instead; run `bash install-macos.sh` again to add it. (With the Dock icon
   there but no menu bar item, the menu bar is full: macOS hides the items
   that do not fit.) If the Dock shows Python rather than Seg-Studio, Python
   is not a framework build (pyenv's or conda's, for example): the app works
   the same. If the Seg-Studio folder is in Desktop, Documents or Downloads,
   allow access when macOS asks (or in **System Settings → Privacy & Security
   → Files & Folders**); when it was refused, the app says it cannot write to
   its folder.
6. Start it with `start-windows.bat` / `bash start-macos.sh` instead: the
   console shows each step and any error.

## Server Won't Start / Cannot Connect to API

**Symptom:** The browser keeps showing "Connecting to server..." or
"Connecting to server. Please wait."

The loading screen usually clears in about 5-15 seconds, depending on the PC.
If it has not changed after a minute, start Seg-Studio again.

**Try this first**

1. Restart Seg-Studio (see the box at the top of this page), then reload the
   browser page.
2. Look at the newest log file. You can open it in Notepad / TextEdit from the
   Seg-Studio folder:
   - Windows: `logs\windows\trainer_<timestamp>.log` (one per start; pick the newest)
   - macOS: `logs/macos/trainer.log`

**Digging deeper (uses commands)**

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

**Try this first**

1. Reload the page.
2. Check that the Seg-Studio folder contains `apps\trainer_ui\dist\`. That
   folder is the built screen of the app. If it is missing, run the installer
   again (`install-windows.bat` on Windows) — its last step builds the screen —
   then start Seg-Studio again.

**Digging deeper (uses commands)**

1. Check if the UI has been built: verify that the `apps\trainer_ui\dist\` folder exists
2. If it hasn't been built:
   ```bat
   cd apps\trainer_ui
   npm install
   npm run build
   ```
3. If using the dev server, navigate to `http://localhost:5173` instead
4. On macOS, the installer skips building the screen when Node.js (`npm`) is
   not installed. Check that Node.js is installed (`brew install node` if it
   is not), run `bash install-macos.sh` again, then start Seg-Studio again.

## Training Stops with "Out of Memory" (OOM)

**Symptom:** Training stops with a "not enough memory" error: `CUDA out of memory`

**Try this first** (all in the app)

1. **Reduce the patch size** — the size of the image pieces the AI learns from
   at one time. Lower it in the Training tab settings (e.g., 256 → 192 → 128)
2. **Pick a smaller Model Size** — Compact or Standard in the Training tab
   settings. There is no batch-size setting: the app fits the batch (how many
   pieces it learns from at once) to the free GPU memory itself, and a run
   that still runs out of memory on an NVIDIA GPU is retried once with
   lower-memory settings
3. **Close other GPU-intensive apps:** Disable GPU acceleration in your browser
4. **4 GB VRAM GPUs** (graphics cards with 4 GB of memory): Low-VRAM mode is
   applied automatically, but if you still get OOM errors, reduce the patch
   size to 128
5. **macOS (MPS):** Unified memory is shared with other apps. If you run out of
   memory, reduce the patch size or close unnecessary applications

**Digging deeper (uses commands)**

- Windows/Linux: Ensure the environment variable
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (enabled by default)

## GPU Not Used: Only "CPU" Appears (Windows / Linux, CUDA)

**Symptom:** Only CPU appears as an option even though a GPU is available
(CUDA not detected)

**Try this first**

1. Update your NVIDIA graphics driver.
2. On Windows, run `install-windows.bat` again by double-clicking it. When the
   NVIDIA driver is installed, it picks the GPU (CUDA) version of PyTorch on
   its own.
3. Manually select the device with the **Device** selector in the Training tab

**Digging deeper (uses commands)**

1. Verify your NVIDIA driver is up to date: `nvidia-smi`
2. Check that the CUDA-enabled PyTorch is installed:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. If it prints `False`, reinstall PyTorch with CUDA support:
   ```bash
   # cu128 for Turing/RTX 20xx and newer (incl. Blackwell); use cu124 only
   # on Maxwell/Pascal/Volta
   pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128 torch torchvision
   ```
   This is the command `install-windows.bat` runs. `--force-reinstall`
   replaces the torch already installed (a plain `pip install` keeps it and
   reports it as already satisfied), and `--no-deps` leaves the other
   packages as installed. On Windows you can instead re-run
   `install-windows.bat cuda`, or
   `install-windows.bat cuda124` on older GPUs.

## Mac GPU Not Used: "MPS" Does Not Appear (macOS)

**Symptom:** MPS device (the Mac's built-in GPU) does not appear on an Apple
Silicon Mac

**Try this first**

1. Verify you are running macOS 12 (Monterey) or later (Apple menu → About This Mac)
2. MPS is not available on Intel Macs (CPU only)

**Digging deeper (uses commands)**

1. Check that MPS-compatible PyTorch is installed:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```
2. If it prints `False`, reinstall PyTorch:
   ```bash
   pip install --upgrade torch torchvision
   ```

## SAM Assist Not Working

**Symptom:** Errors when using SAM click / box segmentation (the AI-assisted
outline tools in the Annotate tab)

**Try this first**

1. Open the `models\sam_checkpoints\` folder in the Seg-Studio folder and check
   that the model files are there: `mobile_sam.pt`, `sam2.1_hiera_tiny.pt`, etc.
2. The model files are downloaded during the Windows install, and on macOS
   and Linux the first time you use the SAM tool. If they are missing, connect
   the computer to the internet and use the SAM tool again. On Windows you can
   also run `install-windows.bat` again — it downloads them — then restart
   Seg-Studio. When a file cannot be downloaded, the SAM tool says "The SAM
   model file (...) could not be downloaded" (NSS-5011) with the file's name,
   and an AI support run stops with the same message.
3. Open `logs\trainer_errors.log` in Notepad / TextEdit and look for the
   specific error.

**Digging deeper (uses commands)**

1. Verify checkpoint files exist: `models\sam_checkpoints\`
2. Required files: `mobile_sam.pt`, `sam2.1_hiera_tiny.pt`, etc.
3. Confirm `downloaded: true` via the `/api/v1/sam/models` endpoint
   (`checkpoint_exists` is also `true` for a file that is missing but can be
   downloaded automatically)
4. Check the logs for specific errors: `logs\trainer_errors.log`

## Model Export Fails

**Symptom:** CoreML / ONNX export fails (saving the trained model for use on
inspection devices)

**Try this first**

1. Make sure the run you selected has a saved model. A run that failed, or one that was stopped before any model was saved, cannot be exported
2. **CoreML** export is recommended on macOS

**Digging deeper (uses commands)**

1. **CoreML:** Ensure `coremltools` is installed (macOS recommended)
   ```bash
   pip install coremltools
   ```
2. **ONNX:** Ensure `onnx` and `onnxruntime` are installed

## Counting: Training Won't Start

**Symptom:** Instance (count) training fails immediately with "instance mode
needs at least 4 annotated images containing one of classes [...]"

This means fewer than 4 images actually have that class painted on them.

**Try this first** (all in the app)

1. **Count what actually has the class**, not how many masks exist. A mask
   painted entirely with value 255 is "unpainted", not foreground, and counts
   for nothing. Open a few masks in the Annotate tab and confirm the class
   colour is really there.
2. **Images marked OK do not count** — they are all-background by definition.
3. The check runs before any GPU work, so this fails in seconds. The message
   names the classes it looked for; if that list is not what you expected,
   check which classes are **active** in the Classes panel.

## Counting: The Count Is Wrong

**Symptom:** The reported count is consistently too low or too high

**Try this first** (in the app)

1. **Look at the overlay first.** Turn on **Detection highlight** in Results —
   background greys out and every object gets its own colour. Two touching
   parts sharing one colour means they were merged; one part with two colours
   means it was split.
2. **Consistently low, near a round number:** you may have hit the most objects
   the model can report in one image — 100 for the default Small model, 200 for
   Medium/Large. Crop tighter or split the frame.

**Digging deeper (settings and API details)**

1. **Consistently ~2x too high:** objects are being counted once per tile (the
   image is processed in overlapping pieces).
   Check the export contract has `patch_size`; a run trained with tiling but
   inferred without it (or the reverse) miscounts silently.
2. **Consistently low, near a round number:** you may be at the model's
   per-image detection ceiling — 100 for the default Small model, 200 for
   Medium/Large. The `/count` response flags this with `truncation_warning`,
   but only for models exported with tiling off (`instance_patch_size = 0`);
   with the default 768 tiling no warning is emitted, so judge saturation
   from the count itself. Crop tighter or split the frame.
3. **Low by a few:** objects larger than the tile overlap can straddle a seam
   and be dropped. At the default 768 patch the overlap is 192px. If your
   parts are bigger than that, raise `instance_patch_size`.
4. **Check the threshold** (the confidence cut-off). `/count` echoes the
   `threshold` it used. It is calibrated on held-out images at training time;
   a very different scene may want a different one.

## Counting: Synthetic Samples Look Wrong

**Symptom:** The preview strip shows composed images that do not resemble your
real scenes

**Try this first** (in the app)

1. **Objects too large or too small:** set the area band (the allowed object
   size range) explicitly instead of `0` (auto). Auto estimates it from your
   blobs (painted objects), and a few stray specks or one over-large blob
   skews it.
2. **Too few cutouts:** the composition log reports `n_cutouts` and how many
   blobs were excluded by the area band. If most were excluded, the band is
   wrong for your data.
3. **Backgrounds look repetitive:** only annotated images become background
   plates. Annotate a few more, from different areas of the scene.

## Counting: `POST /count` Returns 409

**Symptom:** `active model is a semantic-segmentation export; use /segment`

This means the model that is switched on for serving is the wrong kind: one
that marks areas (semantic segmentation), not one that counts objects.

**Try this first**

The activated model in the serving registry is a semantic model, not a
counting one. Export the counting run with **Export Model → ONNX (Serving)** and
activate that model id. Counting exports carry an `instance_inference.json`
alongside `model.onnx`; semantic ones do not.

## Checking Logs

Log files are plain text. You can open them in Notepad / TextEdit.

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
| The Seg-Studio app | `logs\windows\app.log` and `app_start.log` (macOS: `logs/macos/app.log`, `app_start.log` and `app_console.log`) |
| One training run | `projects/<project_id>/runs/<run_id>/train.log` |

"Repo root" means the Seg-Studio folder.

- **JSON logs:** Set environment variable `LOG_FORMAT=json` for JSON output
- **Log level:** Set environment variable `LOG_LEVEL=DEBUG` for debug output

## Dependency Audit

An optional check that uses commands. To check for security vulnerabilities:
```bash
# Windows
scripts\audit.bat
# macOS / Linux
bash scripts/audit.sh
```

## Reporting a Problem

Nothing here helped? Open an issue with the
[bug report template](https://github.com/segmen-pixel/seg-studio/blob/main/.github/ISSUE_TEMPLATE/bug_report.md) at
<https://github.com/segmen-pixel/seg-studio/issues/new/choose>.

**Please include:**

- **Seg-Studio version** — shown next to the name in the app header, and
  returned as the `X-API-Version` header on every API response
- **OS and version** — e.g. Windows 11 23H2, macOS 14.5
- **GPU and driver** — paste `nvidia-smi`, or say "Apple Silicon M2" / "CPU only"
- **How you started it** — the **Seg-Studio** app, `start-windows.bat`,
  `bash start-macos.sh`, `scripts\windows\start_local_windows.bat`,
  `docker compose up`, ...
- **How you installed it** — release ZIP, git clone, or the Windows installer,
  plus `python --version`
- **The exact error text**, copied as text rather than photographed
- **The last ~50 lines of the log** — `logs\trainer_errors.log` first, then
  `logs\app.log`, and with the Seg-Studio app also
  `logs\windows\app_start.log` (see [Checking Logs](#checking-logs) for where
  they live). You can
  copy them from Notepad / TextEdit, or use:

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
