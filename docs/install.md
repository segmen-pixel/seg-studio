# Installation in detail

Read this page when the "Try it" steps in the [README](../README.md) were not
enough. It covers what to check before installing, the steps for each OS,
Docker, offline machines, and what runs where.

## 0. Install pre-flight — check these first

- **Disk space.** The installer's own guidance for the CUDA PyTorch step is
  **~5 GB free**, and that download alone is ~2.5 GB. Add ~300 MB if you opt
  into OpenVINO, plus the five SAM checkpoints. Your images, runs, and exported
  models then live in `projects/` inside the same folder, so extract it
  somewhere with room to grow.
- **Administrator rights: not needed** for the normal path. Everything is
  written inside the folder you extracted — the virtualenv (`.venv-windows` /
  `.venv-macos`), `models/`, `logs/`, and `projects/`. Elevation only comes up
  if you let the Windows installer fetch a missing Python, Node.js or git
  through `winget`, or if a permission error stops `npm install`.
- **Keep the path short on Windows.** Creating the virtualenv fails against the
  260-character path limit; the installer's advice is to move the folder
  somewhere like `C:\seg-studio`. If antivirus blocks the virtualenv instead,
  add that folder to your antivirus exclusions.
- **Python 3.11 or later is required.** The Windows installer installs it for
  you when no Python is found, and stops if that is not possible or if the one
  it finds is older than 3.11; the macOS installer stops if it cannot find it.
  The Windows installer looks for 3.11 first, then 3.12 and
  3.13: the dependency lockfile is compiled against 3.11, so that is the
  version every pinned package is guaranteed to have a prebuilt wheel for. As a last
  resort Windows tries `winget install Python.Python.3.11`; if the new Python is
  not on PATH yet the installer asks you to close the terminal and rerun. On
  macOS, install it yourself (`brew install python@3.11`), then check that
  `python3 --version` prints 3.11 or later: the macOS installer runs the
  `python3` command and stops if that is older.
- **Node.js 18+ is what builds the browser UI.** No pre-built UI bundle ships in
  the repository (`dist/` is not committed), so without `npm` the API starts but
  `http://localhost:8002/ui/` has nothing to serve. If `npm` is missing, the
  Windows installer tries to install Node.js 22 LTS via `winget`; on macOS it
  warns and skips the UI build. Pass `--skip-ui` if you deliberately only want
  the API.
- **git: required on macOS, installed for you on Windows.** `install-macos.sh`
  treats a missing `git` as a fatal prerequisite and stops (`brew install git`).
  `install-windows.bat` installs it through `winget` like Python and Node.js,
  and if that does not work it carries on and skips only the SAM assist
  libraries. Windows also uses `curl` (shipped with Windows 10 1803 and later)
  to download the SAM checkpoints.
- **Double-clicking is enough.** `install-windows.bat` and `start-windows.bat`
  are at the top of the extracted folder and hold the console open when you
  double-click them, so you can read the result. Set `SEG_NO_PAUSE=1` (or run
  under CI) if you are driving them from a script and want them to return
  immediately.
- **NVIDIA driver.** This repository does not pin a minimum driver version, so
  we will not quote one. What the installer actually checks is whether
  `nvidia-smi` runs: if it does you get the CUDA 12.8 wheels (Turing / RTX 20xx
  and newer, including Blackwell); if it does not, you get the CPU build. On
  Maxwell / Pascal / Volta run `install-windows.bat cuda124`. After installing,
  confirm with `python -c "import torch; print(torch.cuda.is_available())"` — it
  must print `True`.
- **Without the SAM weights, everything except SAM click assist still works.**
  Brush, rectangle, eraser, spot detect, crack trace, training,
  evaluation, and every export are independent of them. The five
  checkpoints are downloaded during the Windows install, and otherwise on first
  use; each is checked against a SHA-256 recorded in the source.
- **Internet: needed to install, not to work.** Installing downloads PyTorch and
  the Python dependencies from PyPI, the SAM assist libraries from GitHub, the
  SAM checkpoints, and the UI's npm packages. After that, annotating, training,
  evaluating, and exporting all run on your machine. Three things still reach
  out on demand: a SAM checkpoint you have not downloaded yet; the DINOv2
  weights, downloaded the first time a training run needs them (the Auto
  training setting, on by default, reads DINOv2
  features, and so do DINOv2 distillation and the DINOv2 feature split); and
  the optional [AI support panel](ai-support.md), which sends
  the pictures a run looks at, your instruction, the project's name and the
  tool results to the model server it uses (a server on this machine
  unless you give it another address or choose the `openai` backend). For
  an air-gapped machine, build a bundle first on a connected one with the
  same operating system and Python version:
  `python scripts/install.py --offline-pack <dir>`. The bundle holds the
  packages pinned in the trainer API's lockfile
  (`apps/trainer_api/requirements.txt`) apart from torch, torchvision and
  its git lines; torch and torchvision themselves (from PyTorch's CUDA 12.8
  index on Windows and Linux, from PyPI on macOS); and the SAM checkpoints.
  The pack's `install_offline.py` installs them as pinned, without resolving
  dependencies again. It does not hold ONNX Runtime (pinned in the serving
  API's lockfile, which the pack does not read), the EfficientSAM and
  TinySAM libraries, the UI's npm packages, or the DINOv2 weights
  (`dinov2_vitb14_pretrain.pth`, into torch's `hub/checkpoints` folder);
  bring over whichever of those that machine needs. The MobileSAM and SAM 2
  libraries, which the lockfile names by git URL, are still fetched from
  GitHub; on a machine that cannot reach it, `install_offline.py` says so
  and installs everything else.

## 1. Get the code — no git required

Download the latest **Source code (zip)** from the
[Releases page](https://github.com/segmen-pixel/seg-studio/releases)
(or use the green **Code → Download ZIP** button on the repository page)
and extract it anywhere. If you prefer git:
`git clone https://github.com/segmen-pixel/seg-studio.git`

No pre-built package is published for Windows or macOS: a release on that
page is the source archive (with its SBOMs), so step 2 applies to everyone.
A package can be built from a checkout with `python scripts/build_installer.py`;
when one is published here it will be listed on the Releases page with
`SHA256SUMS.txt` beside it to check the download against.

## 2. Install and start

**Windows (NVIDIA GPU):**
```bash
install-windows.bat
start-windows.bat
```

No terminal needed: double-click `install-windows.bat`, then
`start-windows.bat`, right in the extracted folder. The installer auto-detects your GPU (pass `cpu` or
`cuda124` to override) and installs Python through `winget` if it is missing,
printing step-by-step guidance if that is not possible.
The start script opens the UI in your browser once the server is
ready. After the install, the **Seg-Studio** shortcut does the same without
a console window: see *The Seg-Studio app* below.

**macOS (Apple Silicon / Intel):**
```bash
bash install-macos.sh
bash start-macos.sh
```

**Linux (NVIDIA GPU or CPU):**
```bash
python3.11 -m venv .venv
.venv/bin/python scripts/install.py
bash scripts/start_local.sh
```

Python 3.11, Node.js 18 or later (22 LTS recommended) and git must already be
installed (on Ubuntu:
`apt install python3.11 python3.11-venv git`, Node.js from nodejs.org or your
distribution). The script refuses to run outside a virtual environment and
checks for those three before it downloads anything. It builds the same
environment as the Windows installer -- CUDA 12.8 PyTorch wheels, the CUDA
ONNX Runtime, all five SAM models -- and was verified on Ubuntu 22.04 with an
RTX 3080 Ti: annotation, GPU training and CUDA inference.

Stop everything later with `stop-windows.bat` / `bash stop-macos.sh` /
`bash scripts/stop_local.sh`.

**The Seg-Studio app.** The installers also create an app for the checkout:
on Windows a **Seg-Studio** shortcut in the Start menu and on the desktop, on
macOS **Seg-Studio** in `~/Applications`. Opening it starts the trainer and
serving APIs without a console window, opens the screen once the API is
ready, and stays in the taskbar and the tray (Windows), or in the Dock and the
menu bar (macOS), with **Open Seg-Studio**, **Open the log folder** and **Stop
and quit**. Stop and quit runs the stop script, and asks first when a training,
an inference or an AI support run is going, on any device; closing the taskbar
button (**Close window**) on Windows, or quitting the app from the Dock or with
Cmd-Q on macOS, does the same. Opening the app while it runs, or clicking its
taskbar button or Dock icon, opens the screen. If a Seg-Studio from another
folder, or an older version, is already running, the app asks whether to stop
it and start its own. Stop and quit stops only this folder's Seg-Studio, or
what the app itself started: a program found on the port, or a Seg-Studio from
another folder you chose to keep, is left running. The stop script stops
whatever listens on ports 8002, 8001, 5173 and 8081, as `stop-windows.bat`
does.

The app runs the same start and stop scripts as the `.bat` / `.sh` files,
without the UI dev server. Its log is `logs/windows/app.log`
(`logs/macos/app.log` on macOS), and the start script's output goes to
`app_start.log` beside it; on macOS the app's launcher also writes
`app_console.log` there. With LAN access on, the start script writes where the
LAN token is kept (the `api_token` entry of `runtime_settings.json` in the
projects folder), not the token itself.

Both are written on your machine by the installer, not downloaded, so
Windows and macOS have no signature to check. Pass `--no-shortcuts` to
`install-windows.bat` or `--no-app` to `install-macos.sh` to skip them. On
macOS the Dock icon and the menu bar item use pyobjc (MIT), which the
installer adds from `apps/trainer_api/requirements-macos-app.txt`; without it
the app still starts and opens the screen, and asks with a dialog instead of a
menu. The Dock says Seg-Studio when Python is a framework build (python.org or
Homebrew); with another Python (pyenv, conda, uv) it says Python. If the
checkout is in Desktop, Documents or Downloads, macOS asks once whether
Seg-Studio may use that folder; if that is refused, the app says so when it
opens (allow it in **System Settings → Privacy & Security → Files & Folders**,
or move the folder). The app opens the checkout that installed it last; to
remove it, delete the shortcuts or `Seg-Studio.app`. A Seg-Studio pinned to
the taskbar keeps the folder it was pinned from: after installing a new
version into a new folder, unpin it and pin it again. Without a built screen
(`--skip-ui`, or a UI build that failed) the installers make neither, since
the app serves the built screen.

On Windows, `restart-windows.bat` restarts just the trainer API on port 8002 --
which is what a settings change that says "takes effect on next server restart"
is waiting for. It reuses the LAN token, so browser sessions keep working, and
the new server is detached, so a restart issued over SSH survives the session
ending. It refuses while training is running or queued unless you pass
`--force`; `--dry-run` shows what it would do. The serving API and the UI dev
server are left alone -- use stop + start for those.

> On Windows the installer fetches what it needs: if Python, Node.js or git is
> missing it installs it for you through winget, so a machine with none of the
> three still gets a working setup from one double-click. Without winget
> (some Windows 10 installs lack the App Installer package) it stops and tells
> you where to get Python; Node.js and git only degrade -- no Node.js means the
> UI is not rebuilt, no git means SAM click segmentation is unavailable.
> On macOS git is required up front: `install-macos.sh` stops without it.
> macOS uses MPS (Metal) automatically on Apple Silicon, for training as well as
> inference; object counting still needs an NVIDIA GPU. See
> [Platform support](#platform-support) for what runs where.

## 3. Open the UI

Open **http://localhost:8002/ui/** in your browser — that is the whole install.

New to segmentation? Continue with the
[First Run Walkthrough](first-run-manual.md): from opening the app to your
first prediction in about 10 minutes.

## Alternative: Docker (docker compose)

For a container: CPU only. Semantic training runs but is slow; object counting
(instance) training needs a native NVIDIA install.

```bash
# Windows: use `python` instead of `python3`
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up --build
```

Then open **http://localhost:5173/** — the UI container's nginx proxies `/api`,
`/v2`, and `/ws` to the trainer API. All ports are published on `127.0.0.1` only.
The `.env` step is required, not optional: each container binds `0.0.0.0` inside
its own network namespace, and the trainer refuses to start off-loopback without
a token. The stack requests no GPU, so annotation, inference and the UI work,
semantic training runs on the CPU (much slower), and object counting training
is unavailable — details in [Deployment](deployment.md#docker-optional).


## Requirements

- **OS:** Windows 10 / 11 (64-bit), macOS 12+ (Apple Silicon recommended), or
  Linux (verified on Ubuntu 22.04; `scripts/install.py`)
- **Python:** 3.11+ (the dependency lockfile is compiled against 3.11; 3.10 cannot resolve it)
- **Node.js:** 18+ (for the UI build only)
- **Disk:** ~5 GB free for the CUDA install — see "0. Install pre-flight" above

## Platform support

| | Windows + NVIDIA | Linux + NVIDIA | Apple Silicon (MPS) | CPU only |
|---|---|---|---|---|
| Annotate, SAM assist | Yes | Yes | Yes — 4 of the 5 SAM models (TinySAM is not installed on macOS) | Yes |
| Semantic segmentation training | Yes | Yes | Yes | Yes, much slower |
| Object counting (instance) training | Yes | Yes | No | No |
| ONNX export | Yes | Yes | Yes | Yes |
| ONNX inference | Yes (CUDA provider) | Yes (CUDA provider) | Yes (CPU provider) | Yes (CPU provider) |
| Core ML export | Only if `coremltools` is importable | Only if `coremltools` is importable | Yes | Only if `coremltools` is importable |

- The training device selector defaults to `auto`, which picks CUDA, then MPS,
  then CPU. 4 GB+ VRAM is the recommended minimum for semantic segmentation
  training on NVIDIA.
- The Windows installer defaults to CUDA 12.8 PyTorch wheels (Turing /
  RTX 20xx and newer, including Blackwell RTX 5090); run
  `install-windows.bat cuda124` to use CUDA 12.4 wheels on older GPUs
  (Maxwell / Pascal / Volta). The Linux script uses the same CUDA 12.8 wheels
  and has no switch; on a CPU-only machine they run on the CPU.
- On MPS, mixed precision is disabled, so a run is slower than on a comparable
  NVIDIA card. MPS shares unified memory with the rest of the system — see
  [Troubleshooting](troubleshooting.md) if a run runs out of memory.
- **Object counting (instance segmentation) training needs an NVIDIA GPU.** The
  VRAM auto-fit only runs on CUDA devices; measured on an RTX 3090, the `small`
  model needs 8 GiB at the default batch 8, auto-reducing to batch 4 (5.5 GiB)
  and batch 2 (3.5 GiB). Below 3.5 GiB it is unsupported.
- Core ML export requires `coremltools` (pinned at 8.3.0); without it the export
  endpoint returns HTTP 501. The macOS installer installs it explicitly.
- OpenVINO IR export is an opt-in extra of the Windows installer
  (`install-windows.bat --with-openvino`, ~300 MB). The macOS installer has no
  equivalent flag.
- No prebuilt package is published for either platform: a release is the
  source archive, and the install scripts build the environment on your
  machine (step 2 above). For macOS that is by choice: an unsigned
  `.app` would make every user clear it through Gatekeeper by hand, which is
  more friction than running `install-macos.sh`.
