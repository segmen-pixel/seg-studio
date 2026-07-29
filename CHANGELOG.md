# Changelog

All notable changes to this project will be documented in this file.

## [0.9.8.post1] - 2026-07-27

### Security

- **A server started without an API token could answer the whole network.**
  The startup check that refuses an unauthenticated off-box bind resolves the
  interface from `SEG_HOST` and the persisted LAN-access setting, so
  `uvicorn --host 0.0.0.0` with neither of those set never reached it. The
  request guard then judged the caller by the `Host` header, which the caller
  writes: a request from anywhere on the network claiming `Host: localhost`
  was served, and omitting `Origin` cleared the CSRF check as well. A server
  with no token configured now answers only clients on the machine it runs
  on. Deployments that set `SEG_HOST` or enable LAN access from the interface
  are unaffected -- both already refuse to start without a token.

Install fixes, plus one change to the startup check that reports whether GPU
inference is actually available. The version reported inside the app stays
0.9.8.

### Fixed

- **Every ONNX inference ran on the CPU provider, at 50-100 seconds per image.**
  The installer asked pip for `onnxruntime-gpu>=1.19.0` with `--upgrade`, so a
  clean install took the newest wheel. onnxruntime-gpu 1.27 and later are built
  against CUDA 13 on PyPI, and their provider DLL needs `cublasLt64_13.dll`,
  which the CUDA 12 PyTorch wheels do not ship. The CUDA execution provider
  failed to load and ONNX fell back to CPU. `onnxruntime-gpu` is now pinned to
  1.25.1, matching the portable build and the serving lockfile.
  Because the CPU and GPU wheels share the `onnxruntime/` package directory,
  both are uninstalled and the GPU wheel is reinstalled with
  `--force-reinstall`, so the outcome does not depend on what an earlier run
  left behind.
- **On GPUs older than Ampere, every validation score read 0.0000 while the
  model was fine.** Sliding-window inference enabled fp16 autocast for any CUDA
  device, but training gates it on compute capability, because Turing (GTX 16xx
  and RTX 20xx, capability 7.5) has no fp16 Tensor Cores. On a GTX 1650 the
  model then trained in fp32 and was evaluated in fp16, and autocast returns
  all-NaN logits there once the forward batch reaches four tiles: batch 1 and 2
  are clean, batch 4 and up are entirely NaN. Sliding-window batches ten. Every
  probability map came back NaN, argmax picked background everywhere, and
  validation F1 read exactly 0.0000 with nothing raising an error. `ECE` was
  reported as `nan`, the threshold sweep bottomed out at its floor, reloading
  the best checkpoint scored 0.0000 on weights that contain no NaN at all, and
  the metrics endpoint returned 500 with "Out of range float values are not
  JSON compliant". The rule now lives in one place and both training and
  inference read it. Measured on the affected machine with the same checkpoint
  and data: F1 0.0000 before, 0.8759 after.
- **Instance-mode training ran about eighteen times slower than it needed to on
  GPUs older than Ampere.** The fp16 autocast rule that sliding-window
  inference was missing had a third gap: rfdetr defaults to `amp=True` and
  resolves `amp_dtype="auto"` to fp16 below Ampere, and none of the four places
  that build the detector -- training, threshold calibration, prediction and
  ONNX export -- said otherwise. All four now read the same policy. Measured on
  a GTX 1650 Max-Q through the app, same project and composed dataset at batch
  2 with gradient accumulation 8: 79.8 minutes per epoch before, 4.5 after,
  taking an 80-epoch run from an estimated 106 hours to about six. Epoch-one
  quality is unchanged in kind (segm mAP50-95 0.700 to 0.636, F1 0.667 to
  0.623). Ampere and newer keep autocast on and are unaffected.
- **The VRAM fit reported a reduced batch as though it had fitted.** The batch
  fitter halves the batch until it fits the card's budget and stops at 2, which
  is the smallest configuration rfdetr handles well rather than one that
  necessarily fits the budget. On a 4 GiB Windows card the budget works out at
  1.7 GiB and the smallest model on offer is tabled at 3.5, and the log said
  only that the batch had been reduced. The dry-run now records the shortfall
  as well. Note that the budget is a deliberately pessimistic policy -- a flat
  2 GiB WDDM headroom off a 4 GiB card -- so being over it does not mean the
  run will fail: the machine this was measured on trains fine.
- **A failed CUDA session left no trace, and then cost twice.** Creating the
  ONNX Runtime session with the CUDA provider can throw -- a missing or
  mismatched DLL is the usual reason -- and the fallback quietly installed a
  CPU session instead. The outcome was indistinguishable from a machine that
  never had CUDA: the only sign was `provider=cpu` in a later line, so the
  reason had to be reconstructed by hand from outside the application. It is
  now logged with its traceback and the device that was asked for. The
  single-image path compounded it by running the entire CPU inference first
  and only then looking at the provider, discarding the result and repeating
  it on torch, so on such a machine every prediction was paid for twice. The
  provider is now read when the session is loaded, before any inference runs,
  which is what the batch path already did.
- **Nothing showed which device had run a prediction.** The score payload has
  always carried `inference_device` and `inference_ms`. The interface declared
  both fields and rendered neither, so an ONNX session that had fallen back to
  the CPU provider looked exactly like a healthy GPU run and surfaced only as
  "inference feels slow". Results now shows the device and the elapsed time
  under the image prediction panel. The training widget separately labelled
  `cuda:N` as "GPU N", which collides with Windows Task Manager, where GPU 0
  is usually the integrated adapter; it now reads `CUDA:N`.
- **`large` composed its training canvases at the wrong scale.** Each model's
  input size was restated locally and had drifted from the SDK: nano was
  listed as 384 where it takes 312, and large as 432 where it takes 504.
  Composition doubles that number to size the canvas, and inference tiles to
  match, so `large` -- one of the three selectable sizes -- built 864 px
  canvases for a model that resizes them to 504. That is a 1.7x reduction in
  the one mode whose entire purpose is to have none. The value is now read
  from the SDK's own configuration, which loads no weights.
- **Nothing told you inference had fallen back to CPU.** The startup check did
  detect the failed provider load, but downgraded it to a log line whenever
  PyTorch CUDA was working, on the assumption that the torch GPU path covered
  it. It did not: ONNX inference stayed on CPU. The check now raises a startup
  warning, and names the DLL that failed to resolve instead of guessing at
  cuDNN. It also no longer aborts startup when onnxruntime is present but not
  importable.
- The OpenVINO sample in `docs/openvino_export.md` opened with
  `DEVICE = "AUTO"`, which selects the integrated GPU whenever Intel Graphics
  drivers are present -- the one device the same document warns cannot run
  INT8. It now defaults to `"CPU"`, and the note on `AUTO` says what it picks.
- **A clean install pulled in the GUI OpenCV build next to the pinned headless
  one.** The overrides in `apps/trainer_api/overrides.txt` only apply while the
  lockfile is being compiled. At install time pip re-resolved and honoured
  supervision's unbounded `opencv-python>=4.5.5.64`, which today means
  opencv-python 5.0.0.93, an untested major version. Both wheels own `cv2/`, so
  which one answered `import cv2` came down to install order. The trainer
  lockfile is now installed with `--no-deps`.
- **The Windows install failed outright on machines whose newest Python is
  3.13.** The installer preferred the newest interpreter it could find, but
  `requirements.txt` is compiled with `--python-version 3.11`, so on 3.13 every
  pinned package without a cp313 wheel fell back to building from source, a
  path nothing had tested. `antlr4-python3-runtime` 4.9.3 is sdist-only on
  PyPI and dies there with `No such file or directory: 'bin\pygrun'`, ending
  the install. The same run also built asciitree, coremltools, iopath and
  pyvips from source, so antlr4 was only the first to fall. The installer now
  looks for 3.11 first and keeps 3.12 / 3.13 / 3.10 as fallbacks.
- **`CUDA ........ OK (ERROR: Option noheader is not recognized...)`.** Inside
  `for /f`, cmd treats a bare comma as an argument separator, so
  `--format=csv,noheader` reached `nvidia-smi` as two arguments. The tool
  writes that complaint to stdout, so redirecting stderr did not hide it and
  the error text was captured and printed where the GPU name belongs.

### Changed

- The long install steps print their progress again. Dependency installs,
  the CUDA/CPU PyTorch download and the serving requirements had all of their
  output redirected into the log, so the window sat silent for minutes and
  looked frozen. They now use pip's `--log`, which appends the verbose log to
  the same file while leaving pip's own output on screen; because nothing is
  piped, the exit code is still pip's and failures are still caught.
- The installer verifies ONNX Runtime after installing it, at two severities.
  Whether the package imports at all is a hard failure, since the application
  cannot start without it. Whether the CUDA provider DLL loads is a warning,
  since PyTorch GPU still covers training. `get_available_providers()` lists
  CUDA even when the provider DLL cannot be loaded, so it is not a usable check
  on its own; the DLL is loaded directly instead.

### Upgrading an existing 0.9.8 install

A full reinstall is not needed. In the installed venv:

```
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --force-reinstall --no-deps onnxruntime-gpu==1.25.1
```

Do not remove `opencv-python` on its own. It shares `cv2/` with
`opencv-python-headless`, and uninstalling it leaves `import cv2` broken. If
both are present, reinstall the headless wheel afterwards:

```
pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84
```

## [0.9.8] - 2026-07-27

### Added — instance segmentation as a training mode ("counting")

Training gains a third mode alongside normal and quick: **Instance
(counting)**. It answers "how many objects are there?" when objects touch
or overlap and a semantic mask alone cannot separate them.

The mode adds no annotation work. Instance ground truth is *synthesized*
from the semantic masks that already exist: cutouts are extracted from the
painted regions and copy-pasted into composed scenes with known instance
identities, and an RF-DETR-Seg model is fine-tuned on that COCO dataset.
On the reference counting workload this reached exact counts on a held-out
real test set with zero manually drawn instance labels.

- **Model sizes**: Small (default), Medium, Large. Batch size is auto-fitted
  to the detected VRAM from a measured per-size table, halving the batch and
  doubling gradient accumulation so the effective batch is preserved.
- **Counting**: a validation-calibrated score threshold plus mask-IoU dedup
  at 0.7, which removes the duplicate-mask artifact DETR-family models
  produce on single objects. Adjacent-but-touching objects have near-zero
  mask IoU, so they are not merged.
- **Multi-class**: one model counts every painted class, and the class
  mapping travels with the exported serving contract.
- **Results**: instance overlay with numbered badges in an Okabe-Ito
  palette, per-class count chips, and per-image counts.
- **Export / serving**: ONNX export into the serving registry and a
  `/count` endpoint in serving_api.

Synthesis is configured inline in the Training form (object counts,
objects per image, stack-pair probability, seed, area-band override) with a
Preview button that composes two or three samples before you commit to a
run. Instance runs report per-epoch progress in the run log and drive the
UI progress bar.

Nano was offered during development and retired before release: on the
reference workload it reached 0.92 segm mAP50 against 0.94 (Small) and
0.99 (Medium), which is not enough for exact counts. Existing Nano
checkpoints stay loadable for prediction.

### Added — RF-DETR dependencies, with the non-commercial surface excluded

`rfdetr` ships in the core lockfile rather than an optional extra. Only the
Apache-2.0 package and the Apache-2.0 **Seg** checkpoints are used; the
PML-1.0 "plus" extra and the detection XL/2XL classes are excluded, and ban
patterns cover them. The GUI `cv2` build that `supervision` requests is
replaced by the existing `opencv-python-headless` through a lockfile
override.

`torchmetrics` needed the same treatment for a different reason. The package
is Apache-2.0, so every metadata-level check passes it, but one module —
the Extended Edit Distance text metric — carries a license derived from the
Qt Non-Commercial License v1.0 and is not licensed for commercial use. We
use torchmetrics only for detection mAP, and `import torchmetrics` loads the
text package eagerly, so the installer swaps that module and its Metric-class
wrapper for Apache-2.0 stubs rather than deleting them. The build fails
closed if a future version moves the code. All added dependencies are
recorded in THIRD_PARTY_NOTICES.md.

### Changed — installer reproducibility and source-release compliance

The installer builds from the repository alone; the dev-venv fallback is
gone and a failing build step aborts instead of silently degrading. Builds
emit `release_manifest.json` with a SHA-256 for every staged file.
`scripts/release/collect_lgpl_sources.py` assembles the LGPL/MPL
corresponding-source bundle for binary releases and fails on unpinned or
unknown libraries.

### Fixed

- Instance runs pin the training device explicitly, gate on checkpoint
  presence, avoid train/val leakage in the synthesized split, and serialize
  prediction requests.
- Composition scales correctly across source resolutions and caps cutouts
  to the canvas; single-object area bands are resolution-relative.
- Instance results no longer show semantic region pills that do not apply.
- Annotation gains a batched per-class label clear across the image-list
  selection.

### Housekeeping

- Every version surface now reports the same number. `config.APP_VERSION`,
  `report_generator`, `segcore` and `seg-sdk` had drifted a release behind
  the UI; `report_generator` now imports the single backend definition.
- `timm` is a declared dependency again (`timm==1.0.*`, Apache-2.0), reversing
  the 0.9.6 removal: MobileSAM and TinySAM import it at module load without
  declaring it themselves, so a clean install failed the moment either SAM
  backend was selected. Recorded in THIRD_PARTY_NOTICES.md.
