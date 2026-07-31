@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Seg-Studio  --  Windows Install Script
REM  Usage:  install_windows.bat [cpu|cuda|cuda124]
REM                              [--with-label-studio] [--with-openvino]
REM                              [--skip-ui] [--skip-sam] [--help]
REM ============================================================

REM ---- Handle --help early (before repo root detection) ------
for %%A in (%*) do (
  if /I "%%~A"=="--help" goto :show_help
  if /I "%%~A"=="-h"     goto :show_help
  if /I "%%~A"=="/?"     goto :show_help
)

REM ---- Locate repo root ------------------------------------
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_ABS=%%~fI"
call :find_repo_root "%SCRIPT_ABS%"
if not defined REPO_ROOT call :find_repo_root "%CD%"
if not defined REPO_ROOT (
  echo.
  echo  [ERROR] Could not find repository root.
  echo          This script expects to be located at:
  echo            ^<repo^>\scripts\windows\install_windows.bat
  echo          Or run it from the repository root directory.
  echo.
  goto :fail
)
cd /d "%REPO_ROOT%"

REM ---- Log setup --------------------------------------------
set "LOG_DIR=%REPO_ROOT%\logs\windows"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\install_windows.log"
echo ============================================================ > "%LOG_FILE%"
echo [%date% %time%] install_windows.bat started >> "%LOG_FILE%"
echo Repo root: %REPO_ROOT% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM ---- Parse arguments --------------------------------------
REM Auto-detect NVIDIA GPU: default to cuda if nvidia-smi is available
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  set "TORCH_FLAVOR=cpu"
) else (
  set "TORCH_FLAVOR=cuda"
)
REM CUDA build: cu128 (default; Turing/RTX 20xx and newer, incl. Blackwell)
REM             or cu124 (older GPUs: Maxwell/Pascal/Volta) via the "cuda124" arg.
set "TORCH_CUDA_INDEX=cu128"
set "WITH_LABEL_STUDIO=0"
set "WITH_OPENVINO=0"
set "SKIP_UI=0"
set "SKIP_SAM=0"

REM ---- SAM git dependency pins ------------------------------
REM Repo policy (CONTRIBUTING.md, "Pinning git+URL dependencies"):
REM git installs must reference a commit SHA, never branch HEAD.
REM MOBILE_SAM_SHA / SAM2_SHA mirror apps\trainer_api\requirements.txt
REM -- keep them in sync whenever that lockfile is regenerated.
REM EFFICIENT_SAM_SHA / TINYSAM_SHA are not in the lockfile; they pin
REM upstream HEAD as of 2026-07. Bumping a SHA is a dependency
REM upgrade: re-confirm the upstream LICENSE and re-run smoke tests.
set "MOBILE_SAM_SHA=b01a9ccef3b9e10b099b544efe004d0871802c3b"
set "SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4"
set "EFFICIENT_SAM_SHA=d525f622e6f640acf5a0fc37c7ca1f243da5bde0"
set "TINYSAM_SHA=11589bc1d98c16cff046c31d5ad4cd90a30f0897"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="cpu" (
  set "TORCH_FLAVOR=cpu"
) else if /I "%~1"=="cuda" (
  set "TORCH_FLAVOR=cuda"
  set "TORCH_CUDA_INDEX=cu128"
) else if /I "%~1"=="cuda124" (
  set "TORCH_FLAVOR=cuda"
  set "TORCH_CUDA_INDEX=cu124"
) else if /I "%~1"=="--with-label-studio" (
  set "WITH_LABEL_STUDIO=1"
) else if /I "%~1"=="--with-openvino" (
  set "WITH_OPENVINO=1"
) else if /I "%~1"=="--skip-ui" (
  set "SKIP_UI=1"
) else if /I "%~1"=="--skip-sam" (
  set "SKIP_SAM=1"
) else (
  echo [WARN] Unknown option: %~1
  echo        Run with --help for usage information.
)
shift
goto parse_args

:args_done

REM ---- Validate repo root -----------------------------------
if not exist "apps\trainer_api\app\main.py" (
  echo [ERROR] Repository structure validation failed.
  echo         Expected file not found: apps\trainer_api\app\main.py
  echo         Detected root: %REPO_ROOT%
  goto :fail
)

REM ============================================================
REM  STEP 1: Prerequisites check
REM ============================================================
echo.
echo ============================================================
echo  Seg-Studio Windows Installer
echo ============================================================
echo  Repo root : %REPO_ROOT%
echo  Mode      : %TORCH_FLAVOR%
echo  Log file  : %LOG_FILE%
echo ============================================================
echo.
echo [STEP 1/7] Checking prerequisites...
echo.

set "PREREQ_OK=1"

REM ---- Python detection -------------------------------------
set "PY_BOOTSTRAP="
set "PY_VERSION="
call :resolve_python
if errorlevel 1 goto :python_not_found
goto :python_found

:python_not_found
set "PREREQ_OK=0"
echo   Python ...... NOT FOUND
echo.
echo   [ERROR] Python 3.10 or later is required but was not found.
echo.
echo   How to install Python:
echo     1. Download from https://www.python.org/downloads/windows/
echo        (Recommended: Python 3.11.x)
echo     2. IMPORTANT: Check "Add Python to PATH" during installation
echo     3. After installing, CLOSE this terminal and open a new one
echo     4. Verify: python --version
echo.
echo   Alternatively, if you have winget:
echo     winget install Python.Python.3.11
echo.
goto :fail

:python_found
echo   Python ...... OK  (!PY_VERSION!)

REM ---- Check Python version is 3.10+ -----------------------
call :check_python_version
if errorlevel 1 (
  echo.
  echo   [ERROR] Python version %PY_VERSION% is too old.
  echo           Python 3.10 or later is required.
  echo.
  echo   Please install Python 3.10, 3.11, 3.12, or 3.13 from:
  echo     https://www.python.org/downloads/windows/
  echo.
  goto :fail
)

REM ---- Node/npm detection -----------------------------------
set "HAS_NPM=1"
where npm >nul 2>nul
if errorlevel 1 goto :npm_not_found
goto :npm_found

:npm_not_found
if "%SKIP_UI%"=="1" (
  set "HAS_NPM=0"
  echo   npm ......... SKIPPED ^(--skip-ui^)
  goto :npm_done
)
echo   npm ......... NOT FOUND ^(attempting auto-install...^)
call :install_nodejs
where npm >nul 2>nul
if errorlevel 1 goto :npm_auto_failed
for /f "tokens=*" %%V in ('npm --version 2^>nul') do echo   npm ......... OK  ^(v%%V - just installed^)
goto :npm_done

:npm_auto_failed
set "HAS_NPM=0"
echo   npm ......... NOT FOUND ^(auto-install failed^)
echo.
echo   [WARN] Node.js/npm is needed for the Trainer UI.
echo          The API will work without it, but the UI will not be built.
echo.
echo   How to install Node.js:
echo     1. Download Node.js 22 LTS from https://nodejs.org/
echo     2. Run the installer (includes npm)
echo     3. Close and reopen this terminal
echo     4. Verify: npm --version
echo.
goto :npm_done

:npm_found
for /f "tokens=*" %%V in ('npm --version 2^>nul') do echo   npm ......... OK  ^(v%%V^)

:npm_done

REM ---- Git detection (needed for SAM libraries) -------------
set "HAS_GIT=1"
where git >nul 2>nul
if errorlevel 1 goto :git_not_found
echo   git ......... OK
goto :git_done

:git_not_found
if "%SKIP_SAM%"=="1" (
  set "HAS_GIT=0"
  echo   git ......... SKIPPED ^(--skip-sam^)
  goto :git_done
)
REM Auto-install like Python and Node.js above. git was the only
REM prerequisite that merely warned: a first-time user with none of the
REM three installed got Python and Node handed to them, then silently lost
REM SAM click segmentation because they had no git.
echo   git ......... NOT FOUND ^(attempting auto-install...^)
call :install_git
where git >nul 2>nul
if errorlevel 1 goto :git_auto_failed
for /f "tokens=*" %%V in ('git --version 2^>nul') do echo   git ......... OK  ^(%%V - just installed^)
goto :git_done

:git_auto_failed
set "HAS_GIT=0"
echo   git ......... NOT FOUND ^(auto-install failed^)
echo.
echo   [WARN] git is required to install SAM libraries from GitHub.
echo          Everything except SAM click segmentation still works.
echo.
echo   How to install git:
echo     1. Download from https://git-scm.com/download/win
echo     2. Run the installer (the defaults are fine)
echo     3. Close and reopen this terminal
echo     4. Verify: git --version
echo.

:git_done

REM ---- curl detection (needed for checkpoint downloads) -----
set "HAS_CURL=1"
where curl >nul 2>nul
if errorlevel 1 goto :curl_not_found
echo   curl ........ OK
goto :curl_done

:curl_not_found
set "HAS_CURL=0"
echo   curl ........ NOT FOUND
echo.
echo   [WARN] curl is required to download model checkpoints.
echo          curl is included in Windows 10 1803+ by default.
echo          If missing, install from: https://curl.se/windows/
echo.

:curl_done

REM ---- CUDA check (only when cuda mode) --------------------
if /I not "%TORCH_FLAVOR%"=="cuda" goto :cuda_skip
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :cuda_not_found
REM The comma is escaped: inside for /f, cmd treats a bare comma as an
REM argument separator, so --format=csv,noheader reached nvidia-smi as two
REM arguments and it answered "Option noheader is not recognized". That
REM text goes to stdout, so 2^>nul did not hide it -- it was captured and
REM printed as the GPU name.
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo   CUDA ........ OK  ^(%%G^)
goto :cuda_done

:cuda_not_found
echo   CUDA ........ NOT FOUND (nvidia-smi not in PATH)
echo.
echo   [WARN] You specified 'cuda' but nvidia-smi was not found.
echo          This usually means:
echo            - NVIDIA drivers are not installed
echo            - Or NVIDIA tools are not in PATH
echo          PyTorch CUDA wheels will be installed, but GPU
echo          acceleration may not work at runtime.
echo.
echo          If you don't have an NVIDIA GPU, use: install_windows.bat cpu
echo.
goto :cuda_done

:cuda_skip
echo   CUDA ........ SKIPPED (cpu mode; use 'cuda' arg for GPU)

:cuda_done

echo.

REM ============================================================
REM  STEP 2: Virtual environment
REM ============================================================
echo [STEP 2/7] Setting up Python virtual environment...

set "VENV_DIR=%REPO_ROOT%\.venv-windows"

REM ---- Handle existing venv ---------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" goto :venv_check_partial
REM Verify the existing venv is functional
"%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo   [WARN] Existing venv appears broken. Recreating...
  echo   [WARN] Removing broken venv: %VENV_DIR% >> "%LOG_FILE%"
  rmdir /s /q "%VENV_DIR%" 2>nul
  goto :create_venv
)
echo   [INFO] Using existing venv: %VENV_DIR%
for /f "tokens=*" %%V in ('"%VENV_DIR%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2^>nul') do (
  echo   [INFO] Venv Python version: %%V
)
goto :venv_ready

:venv_check_partial
if exist "%VENV_DIR%" (
  echo   [WARN] Incomplete venv found ^(no python.exe^). Recreating...
  rmdir /s /q "%VENV_DIR%" 2>nul
)

:create_venv
echo   [INFO] Creating virtualenv: %VENV_DIR%
%PY_BOOTSTRAP% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_create_failed
echo   [INFO] Virtualenv created successfully.
goto :venv_ready

:venv_create_failed
echo.
echo   [ERROR] Failed to create virtual environment.
echo.
echo   Common causes:
echo     - Python was installed without the 'venv' module
echo       Fix: Reinstall Python and ensure "pip" and "tcl/tk" are checked
echo     - Antivirus blocking file creation
echo       Fix: Add %VENV_DIR% to your antivirus exclusions
echo     - Path too long (Windows 260 char limit)
echo       Fix: Move the repo to a shorter path (e.g., C:\seg-studio)
echo.
goto :fail

:venv_ready
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] venv python not found: %PYTHON_EXE%
  goto :fail
)

REM ============================================================
REM  STEP 3: Python dependencies
REM ============================================================
echo.
echo [STEP 3/7] Installing Python dependencies...
echo   (This may take several minutes on first install)
echo.

echo   [INFO] Upgrading pip/setuptools/wheel...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [ERROR] Failed to upgrade pip/setuptools/wheel.
  echo          Check your internet connection and try again.
  echo          See log: %LOG_FILE%
  goto :fail
)

echo   [INFO] Installing trainer API dependencies (requirements.txt)...
echo          Around 150 packages. This is the longest step of the install;
echo          pip prints each download below, so the window is working even
echo          when a single line sits there for a while.
REM torch/torchvision pins are excluded here: the pinned versions may have
REM no wheels for the local Python (e.g. torch==2.6.0 had none for 3.14).
REM STEP 4 installs the right torch build explicitly (CUDA index or PyPI
REM CPU) at the newest version compatible with this Python.
REM The SAM git+ lines are excluded too: STEP 5 installs them one by one,
REM non-fatally (mirroring macOS), so a machine without git still gets a
REM working install instead of failing the whole requirements step.
findstr /v /r /c:"^torch==" /c:"^torchvision==" /c:"^mobile-sam @" /c:"^sam-2 @" apps\trainer_api\requirements.txt > "%VENV_DIR%\requirements-notorch.txt"
REM --no-deps: requirements.txt is a fully resolved uv lockfile, and compile time
REM is the only point where apps\trainer_api\overrides.txt applies. Letting pip
REM re-resolve here undoes it: supervision declares an unbounded
REM opencv-python>=4.5.5.64, so the GUI wheel (now major 5) lands next to the
REM pinned opencv-python-headless. Both wheels own cv2/, which makes
REM `import cv2` nondeterministic. lockfile-drift.yml keeps the lockfile a
REM complete resolution, so installing it verbatim is safe.
"%PYTHON_EXE%" -m pip install -r "%VENV_DIR%\requirements-notorch.txt" --no-deps --log "%LOG_FILE%"
if errorlevel 1 goto :pip_trainer_failed
goto :pip_trainer_ok

:pip_trainer_failed
echo.
echo   [ERROR] Failed to install trainer API dependencies.
echo.
echo   Common causes:
echo     - No internet connection
echo     - Firewall/proxy blocking pip
echo     - Visual C++ Build Tools missing (needed by some packages)
echo       Fix: Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo     - Incompatible Python version
echo.
echo   Details in log: %LOG_FILE%
goto :fail

:pip_trainer_ok

echo   [INFO] Installing scikit-learn (MLP assist)...
"%PYTHON_EXE%" -m pip install "scikit-learn>=1.4.0" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] scikit-learn install failed. MLP assist will be unavailable.
  echo   [WARN] scikit-learn install failed >> "%LOG_FILE%"
)

echo   [INFO] Installing scikit-image (superpixel)...
"%PYTHON_EXE%" -m pip install "scikit-image>=0.22.0" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] scikit-image install failed. Superpixel feature will be unavailable.
  echo   [WARN] scikit-image install failed >> "%LOG_FILE%"
)

echo   [INFO] Installing pywinpty (terminal PTY)...
"%PYTHON_EXE%" -m pip install "pywinpty>=2.0.0" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] pywinpty install failed. Terminal features may be limited.
  echo   [WARN] pywinpty install failed >> "%LOG_FILE%"
)

REM transformers is installed pinned (5.x) via requirements.txt above.
REM Do NOT reinstall or cap it here: an older "transformers<5" step
REM used to downgrade the lockfile-pinned version right after install.

echo   [INFO] Installing serving API dependencies...
"%PYTHON_EXE%" -m pip install -r apps\serving_api\requirements.txt --log "%LOG_FILE%"
if errorlevel 1 (
  echo   [WARN] Serving API dependencies install failed.
  echo          The serving API may not work, but the trainer API will.
  echo   [WARN] serving API requirements install failed >> "%LOG_FILE%"
)

REM ============================================================
REM  STEP 4: PyTorch (CPU or CUDA)
REM ============================================================
echo.
echo [STEP 4/7] Configuring PyTorch (%TORCH_FLAVOR%)...

if /I not "%TORCH_FLAVOR%"=="cuda" goto :torch_cpu
echo   [INFO] Installing CUDA-enabled PyTorch wheels (%TORCH_CUDA_INDEX%)...
echo          This download is ~2.5 GB and may take a while.
"%PYTHON_EXE%" -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/%TORCH_CUDA_INDEX% torch torchvision --log "%LOG_FILE%"
if errorlevel 1 goto :torch_cuda_failed
echo   [INFO] Verifying CUDA availability in PyTorch...
"%PYTHON_EXE%" -c "import torch; avail=torch.cuda.is_available(); print(f'  CUDA available: {avail}')"
echo.
echo   [INFO] Installing CUDA-enabled ONNX Runtime...
REM The serving API lockfile installs the CPU onnxruntime wheel, and both wheels
REM own the onnxruntime/ package directory, so uninstalling one deletes files the
REM other still needs. Remove both, then force the GPU wheel back in: on a re-run
REM onnxruntime_gpu's dist-info can survive the uninstall above, and a plain
REM install would report "already satisfied" and leave onnxruntime/ holding only
REM capi\, which imports as an empty namespace package.
"%PYTHON_EXE%" -m pip uninstall -y onnxruntime onnxruntime-gpu --log "%LOG_FILE%"
REM Pinned to the CUDA 12.8 line so it matches the torch %TORCH_CUDA_INDEX% wheels.
REM PyPI onnxruntime-gpu 1.27+ is built against CUDA 13: its provider DLL needs
REM cublasLt64_13.dll, which the CUDA 12 torch wheels do not ship, so the CUDA EP
REM fails to load and every inference silently runs on CPU at 50-100 s per image.
REM Keep in sync with scripts/build_installer.py and apps/serving_api/requirements.in.
"%PYTHON_EXE%" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.25.1" --log "%LOG_FILE%"
if errorlevel 1 goto :ort_gpu_install_failed
REM Two separate checks with different severities. First: is the package itself
REM intact? A gutted onnxruntime/ takes the trainer API's startup down, so that is
REM a hard failure, not a warning.
"%PYTHON_EXE%" -c "import onnxruntime as ort;assert ort.get_available_providers()" 2>nul
if errorlevel 1 goto :ort_gpu_broken
REM Second: get_available_providers() lists CUDA even when the provider DLL cannot
REM load, so load the DLL itself to verify its CUDA dependencies resolve. This one
REM only warns: torch GPU still covers training. torch must be imported first, as
REM it registers its bundled CUDA DLL directory via os.add_dll_directory().
"%PYTHON_EXE%" -c "import torch,ctypes,os,onnxruntime as ort;ctypes.CDLL(os.path.join(os.path.dirname(ort.__file__),'capi','onnxruntime_providers_cuda.dll'))" 2>nul
if errorlevel 1 goto :ort_gpu_dll_failed
echo   [INFO] onnxruntime-gpu installed, CUDA provider verified.
goto :torch_done

:ort_gpu_broken
echo.
echo   [ERROR] The onnxruntime package is present but not importable.
echo           onnxruntime/ is missing its module files, which leaves the trainer
echo           API unable to start. Remove the venv and run this installer again.
echo   [ERROR] onnxruntime package is gutted >> "%LOG_FILE%"
goto :fail

:ort_gpu_install_failed
echo   [WARN] onnxruntime-gpu install failed. Inference will use CPU.
echo   [WARN] onnxruntime-gpu install failed >> "%LOG_FILE%"
goto :torch_done

:ort_gpu_dll_failed
echo   [WARN] onnxruntime-gpu is installed but its CUDA provider DLL will not load.
echo          ONNX inference will fall back to CPU at 50-100 s per image.
echo          Check that the onnxruntime-gpu CUDA major version matches PyTorch.
echo   [WARN] ORT CUDA provider DLL failed to load >> "%LOG_FILE%"
goto :torch_done

:torch_cuda_failed
echo.
echo   [ERROR] CUDA PyTorch wheel installation failed.
echo.
echo   Possible fixes:
echo     - Check internet connection (large download ~2.5 GB)
echo     - Try CPU mode instead: install_windows.bat cpu
echo     - Check disk space (needs ~5 GB free)
echo.
echo   Details in log: %LOG_FILE%
goto :fail

:torch_cpu
echo   [INFO] Installing CPU PyTorch wheels - newest for this Python...
"%PYTHON_EXE%" -m pip install torch torchvision --log "%LOG_FILE%"
if errorlevel 1 goto :torch_cpu_failed
echo   [INFO] To enable GPU later, rerun with: install_windows.bat cuda
goto :torch_done

:torch_cpu_failed
echo.
echo   [ERROR] CPU PyTorch wheel installation failed.
echo   Details in log: %LOG_FILE%
goto :fail

:torch_done

REM ============================================================
REM  STEP 5: SAM libraries & checkpoints
REM ============================================================
echo.
if "%SKIP_SAM%"=="1" (
  echo [STEP 5/7] SAM libraries... SKIPPED ^(--skip-sam^)
  goto :sam_done
)
echo [STEP 5/7] Installing SAM libraries and downloading checkpoints...

if not "%HAS_GIT%"=="1" goto :sam_no_git

REM ---- SAM dependency: timm (required by MobileSAM, TinySAM) ----
"%PYTHON_EXE%" -c "import timm" >nul 2>nul
if not errorlevel 1 (
  echo   [INFO] timm .......... already installed
) else (
  echo   [INFO] Installing timm...
  "%PYTHON_EXE%" -m pip install timm >> "%LOG_FILE%" 2>&1
  if errorlevel 1 echo   [WARN] timm install failed. MobileSAM/TinySAM may not work.
)

REM ---- SAM library: MobileSAM ----
"%PYTHON_EXE%" -c "import mobile_sam" >nul 2>nul
if not errorlevel 1 (
  echo   [INFO] MobileSAM ..... already installed
) else (
  echo   [INFO] Installing MobileSAM from GitHub...
  "%PYTHON_EXE%" -m pip install "git+https://github.com/ChaoningZhang/MobileSAM.git@!MOBILE_SAM_SHA!" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 echo   [WARN] MobileSAM install failed. MobileSAM will be unavailable.
)

REM ---- SAM library: SAM2 ----
"%PYTHON_EXE%" -c "import sam2" >nul 2>nul
if not errorlevel 1 (
  echo   [INFO] SAM2 ......... already installed
) else (
  echo   [INFO] Installing SAM2 from GitHub...
  "%PYTHON_EXE%" -m pip install "git+https://github.com/facebookresearch/sam2.git@!SAM2_SHA!" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 echo   [WARN] SAM2 install failed. SAM2 models will be unavailable.
)

REM ---- SAM library: EfficientSAM ----
"%PYTHON_EXE%" -c "import efficient_sam" >nul 2>nul
if not errorlevel 1 (
  echo   [INFO] EfficientSAM ... already installed
) else (
  echo   [INFO] Installing EfficientSAM from GitHub...
  "%PYTHON_EXE%" -m pip install "git+https://github.com/yformer/EfficientSAM.git@!EFFICIENT_SAM_SHA!" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 echo   [WARN] EfficientSAM install failed. EfficientSAM will be unavailable.
)

REM ---- SAM library: TinySAM ----
REM Copying the package directory installs no dependencies. tinysam imports
REM timm at module load, as MobileSAM does, which is why timm is declared in
REM requirements.in and installed by STEP 3 above rather than here.
"%PYTHON_EXE%" -c "import tinysam" >nul 2>nul
if not errorlevel 1 (
  echo   [INFO] TinySAM ...... already installed
) else (
  echo   [INFO] Installing TinySAM from GitHub ^(manual copy^)...
  set "TINYSAM_TMP=%TEMP%\tinysam_install"
  if exist "!TINYSAM_TMP!" rmdir /s /q "!TINYSAM_TMP!"
  git init "!TINYSAM_TMP!" >> "%LOG_FILE%" 2>&1 && git -C "!TINYSAM_TMP!" fetch --depth 1 https://github.com/xinghaochen/TinySAM.git !TINYSAM_SHA! >> "%LOG_FILE%" 2>&1 && git -C "!TINYSAM_TMP!" checkout !TINYSAM_SHA! >> "%LOG_FILE%" 2>&1
  if not errorlevel 1 (
    xcopy /E /I /Y "!TINYSAM_TMP!\tinysam" "%VENV_DIR%\Lib\site-packages\tinysam" >> "%LOG_FILE%" 2>&1
    if exist "!TINYSAM_TMP!\LICENSE" copy /Y "!TINYSAM_TMP!\LICENSE" "%VENV_DIR%\Lib\site-packages\tinysam\LICENSE" >> "%LOG_FILE%" 2>&1
    rmdir /s /q "!TINYSAM_TMP!" 2>nul
    echo   [INFO] TinySAM installed.
  ) else (
    echo   [WARN] TinySAM clone failed. TinySAM will be unavailable.
  )
)
goto :sam_checkpoints

:sam_no_git
echo   [WARN] git not found. Skipping SAM library installs.
echo          Install git from https://git-scm.com/download/win and rerun.

:sam_checkpoints
REM ---- SAM checkpoints ------------------------------------
if not "%HAS_CURL%"=="1" (
  echo   [WARN] curl not found. Skipping SAM checkpoint downloads.
  goto :sam_done
)
echo.
echo   [INFO] Checking SAM checkpoints...
echo   [INFO] All checkpoints are downloaded from official sources only
echo   [INFO] ^(GitHub releases, Meta CDN, HuggingFace^) and verified with SHA-256.
set "SAM_DIR=%REPO_ROOT%\models\sam_checkpoints"
if not exist "!SAM_DIR!" mkdir "!SAM_DIR!"

REM ---------------------------------------------------------------------------
REM SHA-256 checksums for integrity verification.
REM These model weights are non-executable PyTorch tensors loaded with
REM torch.load(weights_only=True^) which blocks arbitrary code execution.
REM
REM Source repositories (all Apache 2.0 licensed^):
REM   MobileSAM     — https://github.com/ChaoningZhang/MobileSAM
REM   SAM2          — https://github.com/facebookresearch/sam2
REM   TinySAM       — https://github.com/xinghaochen/TinySAM
REM   EfficientSAM  — https://github.com/yformer/EfficientSAM
REM ---------------------------------------------------------------------------
set "SHA_mobile_sam=6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
set "SHA_sam2_tiny=7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
set "SHA_sam2_small=6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
set "SHA_tinysam=4b8edcf93af46e2a658ae455574de62873778a5cc3fd8e8adf094dcdfa957cf2"
set "SHA_effvitt=dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a"

if not exist "!SAM_DIR!\mobile_sam.pt" (
  echo   [INFO]   Downloading MobileSAM checkpoint...
  curl -L --progress-bar -o "!SAM_DIR!\mobile_sam.pt" "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 ( echo   [WARN]   MobileSAM download failed. ) else (
    call :verify_sha256 "!SAM_DIR!\mobile_sam.pt" "!SHA_mobile_sam!" "MobileSAM"
  )
) else (
  echo   [INFO]   MobileSAM .......... already downloaded
)

if not exist "!SAM_DIR!\sam2.1_hiera_tiny.pt" (
  echo   [INFO]   Downloading SAM2 Tiny checkpoint...
  curl -L --progress-bar -o "!SAM_DIR!\sam2.1_hiera_tiny.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 ( echo   [WARN]   SAM2 Tiny download failed. ) else (
    call :verify_sha256 "!SAM_DIR!\sam2.1_hiera_tiny.pt" "!SHA_sam2_tiny!" "SAM2 Tiny"
  )
) else (
  echo   [INFO]   SAM2 Tiny .......... already downloaded
)

if not exist "!SAM_DIR!\sam2.1_hiera_small.pt" (
  echo   [INFO]   Downloading SAM2 Small checkpoint...
  curl -L --progress-bar -o "!SAM_DIR!\sam2.1_hiera_small.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 ( echo   [WARN]   SAM2 Small download failed. ) else (
    call :verify_sha256 "!SAM_DIR!\sam2.1_hiera_small.pt" "!SHA_sam2_small!" "SAM2 Small"
  )
) else (
  echo   [INFO]   SAM2 Small ......... already downloaded
)

if not exist "!SAM_DIR!\tinysam.pth" (
  echo   [INFO]   Downloading TinySAM checkpoint...
  curl -L --progress-bar -o "!SAM_DIR!\tinysam.pth" "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/tinysam.pth" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 ( echo   [WARN]   TinySAM download failed. ) else (
    call :verify_sha256 "!SAM_DIR!\tinysam.pth" "!SHA_tinysam!" "TinySAM"
  )
) else (
  echo   [INFO]   TinySAM ............ already downloaded
)

if not exist "!SAM_DIR!\efficient_sam_vitt.pt" (
  echo   [INFO]   Downloading EfficientSAM-Ti checkpoint...
  curl -L --progress-bar -o "!SAM_DIR!\efficient_sam_vitt.pt" "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/efficient_sam_vitt.pt" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 ( echo   [WARN]   EfficientSAM-Ti download failed. ) else (
    call :verify_sha256 "!SAM_DIR!\efficient_sam_vitt.pt" "!SHA_effvitt!" "EfficientSAM-Ti"
  )
) else (
  echo   [INFO]   EfficientSAM-Ti .... already downloaded
)

:sam_done

REM ============================================================
REM  STEP 5.5: OpenVINO IR export (optional, Intel edge deployment)
REM ============================================================
echo.
if not "%WITH_OPENVINO%"=="1" goto :openvino_skip
echo [STEP 5.5] Installing OpenVINO + NNCF (Intel edge export)...
echo            ~300 MB download; required only for IR-format export.
"%PYTHON_EXE%" -m pip install -r apps\trainer_api\requirements-openvino.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] OpenVINO install failed. IR export will return HTTP 501.
  echo   [WARN] OpenVINO install failed >> "%LOG_FILE%"
) else (
  echo   [INFO] OpenVINO + NNCF installed successfully.
)
goto :openvino_done

:openvino_skip
echo [STEP 5.5] OpenVINO... SKIPPED (use --with-openvino to install)

:openvino_done

REM ============================================================
REM  STEP 6: Label Studio (optional)
REM ============================================================
echo.
if not "%WITH_LABEL_STUDIO%"=="1" goto :labelstudio_skip
echo [STEP 6/7] Installing Label Studio...
"%PYTHON_EXE%" -m pip install label-studio >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] Label Studio install failed. It can be installed later.
  echo   [WARN] Label Studio install failed >> "%LOG_FILE%"
) else (
  echo   [INFO] Label Studio installed successfully.
)
goto :labelstudio_done

:labelstudio_skip
echo [STEP 6/7] Label Studio... SKIPPED (use --with-label-studio to install)

:labelstudio_done

REM ============================================================
REM  STEP 7: Trainer UI (Node.js / React)
REM ============================================================
echo.
if "%SKIP_UI%"=="1" (
  echo [STEP 7/7] Trainer UI... SKIPPED ^(--skip-ui^)
  goto :after_ui
)
if not "%HAS_NPM%"=="1" (
  echo [STEP 7/7] Trainer UI... SKIPPED ^(npm not available^)
  echo          Install Node.js 22 LTS from https://nodejs.org/ and rerun.
  goto :after_ui
)
echo [STEP 7/7] Building Trainer UI...

echo   [INFO] Installing npm dependencies...
cd /d "%REPO_ROOT%\apps\trainer_ui"
call npm ci >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] 'npm ci' failed. Trying 'npm install' instead...
  call npm install >> "%LOG_FILE%" 2>&1
  if errorlevel 1 goto :npm_install_failed
)
goto :npm_install_ok

:npm_install_failed
echo.
echo   [ERROR] npm install failed.
echo.
echo   Common causes and fixes:
echo     - node-gyp errors: Install Visual C++ Build Tools
echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo     - EACCES / permission errors: Run terminal as Administrator
echo     - Network errors: Check proxy settings
echo       npm config set proxy http://your-proxy:port
echo     - Corrupted cache: npm cache clean --force
echo     - node_modules conflict: delete apps\trainer_ui\node_modules
echo       and rerun this script
echo.
echo   Details in log: %LOG_FILE%
echo.
cd /d "%REPO_ROOT%"
REM Don't fail the whole install for UI issues
goto :after_ui

:npm_install_ok
echo   [INFO] Building UI (npm run build)...
call npm run build >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [WARN] UI build failed. The UI can still run via Vite dev server.
  echo   [WARN] To build manually: cd apps\trainer_ui ^&^& npm run build
  echo   [WARN] UI build failed >> "%LOG_FILE%"
) else (
  echo   [INFO] UI built successfully.
)
cd /d "%REPO_ROOT%"

:after_ui

cd /d "%REPO_ROOT%"
if not exist "logs\windows" mkdir "logs\windows"

REM ============================================================
REM  Summary
REM ============================================================
echo.
echo ============================================================
echo  Installation Complete
echo ============================================================
echo.
echo  Next steps:
echo    Start:  scripts\windows\start_local_windows.bat
echo    Stop:   scripts\windows\stop_local_windows.bat
echo    Status: scripts\windows\status_windows.bat
echo.
echo  Endpoints (after starting):
echo    Trainer API : http://localhost:8002/docs
echo    Trainer UI  : http://localhost:8002/ui/
echo    Serving API : http://localhost:8001/docs
echo.
echo  Full log: %LOG_FILE%
echo ============================================================
echo.

echo [%date% %time%] install_windows.bat completed successfully >> "%LOG_FILE%"
exit /b 0

REM ============================================================
REM  Subroutines
REM ============================================================

:show_help
echo.
echo  Seg-Studio Windows Installer
echo.
echo  Usage:
echo    install_windows.bat [OPTIONS]
echo.
echo  Options:
echo    cpu                   Install CPU-only PyTorch
echo    cuda                  CUDA PyTorch, cu128 build (default for NVIDIA GPUs^).
echo                          Turing/RTX 20xx and newer, incl. Blackwell (RTX 50xx, RTX PRO 6000^).
echo    cuda124               CUDA PyTorch, cu124 build for older GPUs
echo                          (Maxwell/Pascal/Volta: GTX 10xx, Tesla V100^).
echo                          (Default: auto-detect — cuda/cu128 if NVIDIA GPU found^)
echo    --with-label-studio   Also install Label Studio annotation tool
echo    --with-openvino       Also install OpenVINO + NNCF (~300 MB^)
echo                          Enables Intel edge export (.xml/.bin, FP32/FP16/INT8^)
echo    --skip-ui             Skip Node.js/npm UI build
echo    --skip-sam            Skip SAM model library and checkpoint downloads
echo    --help, -h            Show this help message
echo.
echo  Examples:
echo    install_windows.bat                     CPU mode, full install
echo    install_windows.bat cuda                GPU mode with CUDA
echo    install_windows.bat cuda --skip-sam     GPU mode, skip SAM downloads
echo    install_windows.bat --skip-ui           Skip UI build (API only)
echo.
echo  Prerequisites:
echo    Required:  Python 3.10+ (with pip and venv)
echo    Optional:  Node.js 22 LTS (for Trainer UI)
echo    Optional:  git (for SAM library installs)
echo    Optional:  NVIDIA GPU + drivers (for CUDA mode)
echo.
echo  If you encounter issues, check the log at:
echo    logs\windows\install_windows.log
echo.
exit /b 0

:fail
echo.
echo ============================================================
echo  [FAILED] Setup did not complete successfully.
echo ============================================================
if defined LOG_FILE (
  echo  Check the log for details:
  echo    %LOG_FILE%
)
echo.
echo  Common troubleshooting steps:
echo    1. Ensure Python 3.10+ is installed and in PATH
echo    2. Ensure you have internet access
echo    3. Try running as Administrator if permission errors occur
echo    4. If the venv is corrupted, delete .venv-windows and retry
echo    5. Run with --help for usage information
echo.
pause
exit /b 1

:find_repo_root
set "CANDIDATE=%~f1"
:find_repo_loop
if exist "%CANDIDATE%\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%"
  goto :eof
)
if exist "%CANDIDATE%\seg-studio\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\seg-studio"
  goto :eof
)
if exist "%CANDIDATE%\seg-sutie\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\seg-sutie"
  goto :eof
)
if exist "%CANDIDATE%\windows\seg-studio\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\windows\seg-studio"
  goto :eof
)
if exist "%CANDIDATE%\windows\seg-sutie\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\windows\seg-sutie"
  goto :eof
)
for %%P in ("%CANDIDATE%\..") do set "PARENT=%%~fP"
if /I "%PARENT%"=="%CANDIDATE%" goto :eof
set "CANDIDATE=%PARENT%"
goto :find_repo_loop

:install_nodejs
where winget >nul 2>nul
if errorlevel 1 (
  echo   [INFO] winget is unavailable. Cannot auto-install Node.js.
  goto :eof
)
echo   [INFO] Attempting to install Node.js 22 LTS via winget...
winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [INFO] Automatic Node.js install did not succeed.
  goto :eof
)
REM Refresh PATH to pick up newly installed Node.js
set "PROG_NODE=%ProgramFiles%\nodejs"
if exist "%PROG_NODE%\npm.cmd" (
  set "PATH=%PROG_NODE%;%PATH%"
  echo   [INFO] Node.js installed: %PROG_NODE%
)
goto :eof

:install_git
where winget >nul 2>nul
if errorlevel 1 (
  echo   [INFO] winget is unavailable. Cannot auto-install git.
  goto :eof
)
echo   [INFO] Attempting to install Git via winget...
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [INFO] Automatic git install did not succeed.
  goto :eof
)
REM Refresh PATH to pick up newly installed git
set "PROG_GIT=%ProgramFiles%\Git\cmd"
if exist "%PROG_GIT%\git.exe" (
  set "PATH=%PROG_GIT%;%PATH%"
  echo   [INFO] Git installed: %PROG_GIT%
)
goto :eof

:resolve_python
REM Try Python Launcher first (supports version selection)
REM NOTE: We use explicit sequential calls instead of a for-loop
REM       because batch errorlevel is sticky inside parenthesized blocks.
set "PY_BOOTSTRAP="
set "PY_VERSION="

where py >nul 2>nul
if errorlevel 1 goto :resolve_python_no_py

REM Newest-first was wrong. requirements.txt is compiled with
REM --python-version 3.11, so 3.11 is the only interpreter every pinned
REM package is guaranteed to have a wheel for. On 3.13 the ones without a
REM cp313 wheel fall back to building from sdist -- a path nothing tested
REM -- and antlr4-python3-runtime 4.9.3 (sdist-only on PyPI) dies there
REM with "No such file or directory: 'bin\pygrun'", taking the whole
REM install down. The same run also built asciitree, coremltools, iopath
REM and pyvips from source, so antlr4 was simply the first to fall.
REM Prefer the version the lockfile was built for; keep the others as
REM fallbacks so a machine without 3.11 still installs.
REM Try py -3.11 (what the lockfile targets)
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.11
REM Try py -3.12
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.12
REM Try py -3.13
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.13
REM Try py -3.10
if defined PY_BOOTSTRAP goto :resolve_python_done_py
call :try_py_version 3.10

REM Try py -3 (default Python 3)
if defined PY_BOOTSTRAP goto :resolve_python_done_py
set "PY_VERSION="
for /f "tokens=*" %%O in ('py -3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')" 2^>nul') do set "PY_VERSION=%%O"
if not defined PY_VERSION goto :resolve_python_no_py
set "PY_BOOTSTRAP=py -3"

:resolve_python_done_py
if defined PY_BOOTSTRAP exit /b 0

:resolve_python_no_py
REM Try 'python' command
where python >nul 2>nul
if errorlevel 1 goto :resolve_python_no_python
set "PY_VERSION="
for /f "tokens=*" %%O in ('python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')" 2^>nul') do set "PY_VERSION=%%O"
if not defined PY_VERSION goto :resolve_python_no_python
set "PY_BOOTSTRAP=python"
exit /b 0

:resolve_python_no_python
REM Try 'python3' command
where python3 >nul 2>nul
if errorlevel 1 goto :resolve_python_auto_install
set "PY_VERSION="
for /f "tokens=*" %%O in ('python3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')" 2^>nul') do set "PY_VERSION=%%O"
if not defined PY_VERSION goto :resolve_python_auto_install
set "PY_BOOTSTRAP=python3"
exit /b 0

:resolve_python_auto_install
REM Try auto-install via winget as last resort
where winget >nul 2>nul
if errorlevel 1 exit /b 1

echo   [INFO] Attempting to install Python 3.11 via winget...
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo   [INFO] Automatic Python install did not succeed.
  exit /b 1
)

REM Check standard install location
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  set "PY_BOOTSTRAP=%LocalAppData%\Programs\Python\Python311\python.exe"
  set "PY_VERSION=3.11 (just installed)"
  exit /b 0
)

REM Try py launcher after install
where py >nul 2>nul
if errorlevel 1 goto :resolve_python_post_install_python
set "PY_VERSION="
for /f "tokens=*" %%O in ('py -3.11 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')" 2^>nul') do set "PY_VERSION=%%O"
if not defined PY_VERSION goto :resolve_python_post_install_python
set "PY_BOOTSTRAP=py -3.11"
set "PY_VERSION=%PY_VERSION% (just installed)"
exit /b 0

:resolve_python_post_install_python
where python >nul 2>nul
if errorlevel 1 goto :resolve_python_need_restart
set "PY_VERSION="
for /f "tokens=*" %%O in ('python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')" 2^>nul') do set "PY_VERSION=%%O"
if not defined PY_VERSION goto :resolve_python_need_restart
set "PY_BOOTSTRAP=python"
set "PY_VERSION=%PY_VERSION% (just installed - may need terminal restart)"
exit /b 0

:resolve_python_need_restart
echo.
echo   [INFO] Python was installed but is not yet in PATH.
echo          Please CLOSE this terminal, open a new one, and rerun this script.
echo.
exit /b 1

:try_py_version
REM %1 = version like 3.12
REM Trust a captured version string, never the exit code: the python.org
REM Python Install Manager's "py" prints "No runtime installed that
REM matches %1" for a missing runtime but still exits 0, which used to
REM select a non-functional interpreter here.
set "PY_VERSION="
for /f "tokens=*" %%O in ('py -%1 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')" 2^>nul') do set "PY_VERSION=%%O"
if not defined PY_VERSION exit /b 1
set "PY_BOOTSTRAP=py -%1"
exit /b 0

:check_python_version
REM Verify Python version is 3.10+
if not defined PY_BOOTSTRAP exit /b 1
%PY_BOOTSTRAP% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 exit /b 1
exit /b 0

:verify_sha256
REM Verify SHA-256 hash of a downloaded file using certutil.
REM Usage: call :verify_sha256 "filepath" "expected_hash" "display_name"
REM %~1 = file path, %~2 = expected SHA-256 (lowercase hex), %~3 = display name
if not exist "%~1" (
  echo   [WARN]   %~3: file not found, skipping hash check.
  exit /b 1
)
set "EXPECTED_HASH=%~2"
set "ACTUAL_HASH="
for /f "skip=1 tokens=*" %%H in ('certutil -hashfile "%~1" SHA256 2^>nul') do (
  if not defined ACTUAL_HASH set "ACTUAL_HASH=%%H"
)
REM Remove spaces from certutil output
set "ACTUAL_HASH=!ACTUAL_HASH: =!"
if /I "!ACTUAL_HASH!"=="!EXPECTED_HASH!" (
  echo   [OK]     %~3: SHA-256 verified.
) else (
  echo   [WARN]   %~3: SHA-256 mismatch!
  echo            Expected: !EXPECTED_HASH!
  echo            Got:      !ACTUAL_HASH!
  echo            The file may be corrupted or tampered with.
  echo            Consider deleting it and re-downloading.
)
exit /b 0
