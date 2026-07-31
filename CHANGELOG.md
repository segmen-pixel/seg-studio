# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **The mask noise filter is a setting again, on the Training tab.** The form
  has sent `postprocess_min_area: 0` ever since the toggle was taken out, and 0
  does not mean off: it asks the trainer to measure a 6-sigma threshold from the
  training masks and drop every component below it. Auto is exactly that
  behaviour and stays the default, so nothing changes for anyone who leaves it
  alone. Manual takes a size in pixels and starts at 1, which removes nothing,
  so switching the mode cannot delete anything on its own.

  What gets filtered is the prepared copy of the masks, rebuilt from the
  annotations at the start of every run. The annotations themselves are never
  touched, and the filtering cannot accumulate from one run to the next.

### Changed

- **Files are named with your clock, everywhere.** Exported datasets were named
  in local time and reports and imports in UTC, so two files produced in the
  same minute by two features were named hours apart. They all use local time
  now, through one function rather than a choice made at each site. Timestamps
  that are stored, compared, or sent over the API are unaffected and remain
  UTC -- local time is for names.

  The reports list used to be ordered by the report directory's name, so it
  would have reshuffled around the moment that name stopped being UTC. It is
  ordered by the timestamp inside each report instead, which does not depend on
  what the directory is called.

### Fixed

- **The deprecated UTC constructors are gone, and nothing on the wire moved.**
  `datetime.utcnow()` and `datetime.utcfromtimestamp()` are deprecated from
  Python 3.12 and both return a datetime with no zone attached. The three
  remaining calls -- a run's `created_at` from the CLI, the pretrained model's
  `updated_at`, and a `Last-Modified` header -- now use the tz-aware forms.

  The values they produce are unchanged, deliberately. The header renders
  identically because its format carries the zone itself, and the two
  serialised fields keep their unsuffixed shape: readers on both sides take an
  unsuffixed value as UTC, existing `train_config.json` files on disk carry
  that shape, and removing a deprecation is not how an API format should
  change.

- **A GPU lock whose heartbeat could not be read was treated as abandoned.**
  The staleness check parsed the timestamp inside a bare `except` that fell
  through to "older than the window", and a stale lock is deleted and
  re-claimed -- while the process that owns it, which that branch has already
  established is alive, carries on training. A naive timestamp was enough to
  trigger it, because subtracting one from a tz-aware datetime raises.

  A timestamp with no zone is now read as UTC, as it is everywhere else here; a
  missing one falls back to the claim time, which bounds the lock's age since
  the check only covers the gap between claiming a device and the worker
  starting; and a lock with nothing readable at all is treated as held, with a
  warning. That costs one idle GPU until the owning process exits, against two
  jobs on one card.

- **The export manifest recorded local time where everything else records UTC.**
  `exported_at` was the one naive local timestamp the application wrote, and the
  convention everywhere else -- including the reader in the UI -- is that a
  value with no zone suffix is UTC. An export made at 18:00 JST therefore read
  back as 09:00, nine hours out in the opposite direction from the display bugs
  fixed in 0.9.8.post2, which is how it survived them.

- **Reported latencies were measured on the wall clock.** The inference time in
  the serving response, the train and predict times from the assist endpoints,
  and the trace timings were all `time.time()` differences. A clock step -- an
  NTP correction, or someone setting the clock -- makes such a difference wrong
  and can make it negative. They read `time.perf_counter()` now, as the
  pipelined inference runtime beside them already did.

- **Two in-process caches expired on the wall clock.** The projects summary and
  the library stats set their deadline with `time.time()`, so a backward step
  held them unexpired for the length of the step and a forward one discarded
  them early. Both are module globals that never leave the process, so
  `time.monotonic()` is the clock they want -- with an empty-cache deadline
  outside its range rather than `0.0`, which is inside it.

  The caches that are written to disk keep the wall clock deliberately: a
  timestamp another process has to read cannot come from a clock that has no
  origin and no meaning outside the process that read it.

- **The pipelined inference runtime held its GPU sessions through every idle
  release.** The release added in 0.9.8.post2 covers the ORT session cache, and
  neither half of the v2 runtime goes through it: each GPU worker loads a
  session onto its own thread, and Live Inspection and single-image prediction
  share a second one. A machine that ran a camera session or a batch of
  predictions and was then left alone kept the card until the process ended or
  a training run happened along and cleared a different cache.

  Both now hand the memory back after the same 300 seconds of idleness, under
  the same `SEG_ORT_IDLE_RELEASE_SECONDS` setting -- `0` still switches the
  whole thing off. Measured on an RTX 3090 with a 512x512 model: the worker's
  session went from 23,972 MiB back to 1,044 MiB 46 seconds after its last
  batch, and the stream session returned 271 of the 324 MiB it had taken.

  A worker releases from inside its own loop rather than from the poller,
  because that thread is the only one that ever runs its session. Dropping it
  from outside would free nothing while the batch below still held the
  reference, and would leave the next batch to build a second session beside
  the first -- which on a 4 GB card is the failure the release exists to
  prevent. The stream session is deliberately used outside its lock, for a
  whole sliding-window pass over one frame, so it counts the passes in flight
  for the same reason.

- **The batch size a GPU worker profiles is remembered across a release.** The
  search runs real inferences until they fail, and took 24.3 seconds for that
  model on that card. Paying it once per process was reasonable while a session
  was loaded once per process; paying it after every quiet spell would have put
  those seconds in front of the first prediction. Recalling the number promises
  nothing that was not already true -- a sub-batch that fails to allocate is
  still halved and retried, and the smaller number replaces the remembered one,
  so a card that is tighter than it was corrects itself on the first batch. The
  same batch measured 24.3 seconds cold and 0.1 seconds after a release.

## [0.9.8.post2] - 2026-07-30

### Changed

- **Inference now uses the sliding-window stride each run measured for itself.**
  Post-training stride optimisation scores several strides on the run's own
  validation images and stores the winner, and the threshold that ships with the
  model is calibrated at that same geometry -- but inference read none of it and
  always stepped by three quarters of the patch. Where the foreground is sparse
  the two are not equivalent: on a project with 1.9% foreground the stored stride
  scored F1 0.9714 against 0.9176 for the default. Predicting that project's
  images through the application, precision goes from 0.929 to 0.962 and
  false-positive pixels halve (35,055 to 18,186) at unchanged recall, while
  detections that sit on no real object drop from 14 to 2.

  **This costs time.** The finer stride is 5.5 times as many patches on a
  1280x720 image, and on a GTX 1650 the same images went from 1.2 to 7.6 seconds
  each. Runs whose optimisation kept the three-quarter stride are unaffected, as
  are runs from before stride optimisation existed. The Live Inspection tab keeps
  the coarse stride, because it runs the window on every frame.

- **Runs moved out of `training/` and the predictions directory got shorter.**
  A run now lives at `projects/<project id>/runs/<run id>/pred/` instead of
  `projects/<project id>/training/runs/<run id>/predictions/`. Sixteen more
  characters of a 260-character Windows path, on top of the shorter ids below,
  taking the worst case from 294 to 218.

  Projects migrate themselves the first time they are opened, in one step that
  leaves `training/runs` in place until everything under it has been renamed --
  so an interrupted migration simply runs again. `training/pretrained` and the
  `training/archive_*` directories stay where they are. Nothing has to be done
  by hand, and a project copied back from an old backup migrates whenever it
  next appears, whatever version it claims to be stamped with.

### Fixed

- **A prediction kept the graphics card until the next training run.** Loading a
  model for inference caches its ONNX Runtime session, and the cache is only
  dropped by the release that runs when a training run starts -- so on a machine
  used for predictions and then left alone, the session stayed for the life of
  the process. Measured on a 4 GB card while tracking down the previous release's
  bug: 3,915 of 4,096 MiB still held an hour and a half after the GPU had gone
  idle at 2.34 W. The pre-training release, further down this same list, reclaims
  that memory when the next training run starts -- but nothing reclaimed it if no
  training run followed.

  Cached sessions are now handed back after five minutes without use, checked
  every thirty seconds. `SEG_ORT_IDLE_RELEASE_SECONDS` changes the five minutes
  and `0` switches the release off. An inference in flight holds it off, so a
  long sliding-window prediction cannot have its session released from under it.
  The cost is that the first prediction after a release rebuilds the session.

- **One validation reading could move the patch-sampling balance by a quarter of
  its range.** Training nudges `fg_patch_prob` -- how often a training patch is
  taken from an annotated region -- by 0.05 toward whichever of precision and
  recall is behind. The nudge ran on every epoch, but the precision and recall it
  reads only change on an epoch that validates, and past the tenth epoch that is
  one epoch in five. The same reading therefore drove five steps: 0.25 of the
  0.30-0.80 range the value may move within, so two readings leaning the same way
  pinned it to a bound and left it there. It now moves once per measurement.

  Runs will sample differently from the tenth epoch onward. A run that was being
  walked to a bound by this stays nearer where auto-tuning put it.

- **The early-stopping message counted in the wrong unit.** Training evaluates
  every epoch up to the tenth and every fifth epoch after that, and the
  no-improvement counter behind early stopping is in those evaluations rather
  than in epochs -- so a stop after 25 of them was announced as "no improvement
  for 25 epochs" when around 85 had passed. The message now says validation
  rounds. When training stops is unchanged.

- **Every time the application displayed was nine hours out in JST.** Timestamps
  arrive from the API without a timezone marker, and the model list, the best
  model card, the new-model notification, the project tile's last-trained line
  and the Live Inspection model picker all read them as local time. A run started
  at 19:08 was listed as 10:08, and a model that had just finished was reported
  as nine hours old. The helper that normalises these values had been in the code
  the whole time with no callers. The time beside each model is when the run was
  created, which for a queued run is when it was queued rather than when training
  began, and its tooltip now says so.

- **A model trained before 09:00 was named with yesterday's date.** The automatic
  name took its calendar date from the UTC clock, so in JST the daily sequence
  number also rolled over at nine in the morning instead of at midnight. Names
  already assigned are left as they are.

- **Windows path limit: some projects had already run out of room.** Every
  artifact lived at `projects/<project id>/training/runs/<run id>/predictions/`,
  and the two ids alone spent 72 of the 260 characters Windows allows. The
  deepest path on a real install measured 250, ten short of the limit, and the
  worst case a long image filename could produce was already past it -- which
  surfaces tens of minutes into a run as a `FileNotFoundError` naming a
  prediction nobody asked about. Three changes, together taking the worst case
  from 294 to 234:

  - New project, run and model ids are 12 hex characters instead of 36-character
    UUIDs. **Existing ids are untouched and keep working** -- nothing migrates,
    and both shapes are recognised everywhere.
  - Filenames longer than 48 characters are shortened on disk, keeping a
    readable prefix and a digest so two similar names stay distinct. The image
    keeps its original name for display. Files already on disk are left alone.
    48 rather than a longer cap because the cap is what decides how deep an
    installation may sit: it fits any projects directory rooted within 132
    characters.

  - **Importing a zip ignored that cap entirely, and a long name inside an
    archive failed the import.** Uploading applied the cap; importing took the
    archive member's name as it found it, so a name that did not fit a Windows
    path raised an unhandled error rather than being shortened. Both routes now
    go through the same function, and both are covered by tests -- the cap had
    none, which is how one route came to have it and the other not.
  - Startup reports the remaining budget for the actual installation, naming
    any project that is already over, instead of letting it fail later.

  A directory in `projects/` is now recognised as a project only if its name is
  one of the two id shapes. It used to be anything that parsed as a UUID, and
  that check was also what kept the orphan sweep from deleting `.library` --
  which holds the only surviving copy of a deleted project's best weights --
  along with `.gpu_locks` and hand-made directories.

- **Ids are checked for collision before anything is written.** Project and run
  creation both `mkdir(exist_ok=True)`, write their own metadata over whatever
  is there, and delete the whole directory if the database insert then fails,
  so a collision would have destroyed the project it collided with. Shorter ids
  make that arithmetic real rather than notional, so each id is now confirmed
  free first.

- **The unique indexes on `trainingrun.run_id` and `modelrecord.model_id` now
  exist.** They were declared in the models but `create_all()` only creates
  missing tables, so no database that predates the declaration ever got them.

- **The Live Inspection tab could not open a camera.** Its camera endpoints live
  at the server root (`/v2/camera/...`), not under the `/api/v1` prefix, and the
  hook that starts capture built its URLs from the prefixed base while the
  sibling hook that polls status used the unprefixed one. Every call it made --
  config, start, stop, and the preview WebSocket -- went to `/api/v1/v2/camera/...`
  and returned 404, so the tab could read camera state but never start it.

- **Stopping an instance run threw away the model it had already trained.**
  RF-DETR exposes no in-training stop hook, so a stop terminates the training
  child inside `model.train()`, and every line after it — including the write
  of `instance_inference.json` — never ran. The checkpoints were on disk and
  the run still reported no model, because that file is what marks one
  available: 90 minutes of training sat next to a `checkpoint_best_regular.pth`
  nothing could reach. The contract is now written after the terminate, so a
  stopped run keeps its best checkpoint and can be predicted with. A stop
  before the first evaluation has nothing to keep and says so rather than
  failing the run.

- **Instance training ran at a different input size than instance inference.**
  RF-DETR's multi-scale training keeps only its largest candidate unless
  random resize is enabled — 504 px for a 384 px model — while validation,
  `predict()` and the tiled inference path all use the model's own 384.
  Composition sizes each canvas at twice the model input so a tile halves to
  reach the model on a clean 2:1, which is the whole point of patch mode;
  training was taking 1.52:1 instead and seeing every object 1.31x larger than
  inference would ever show it. Training is now pinned to the model input, so
  both halves agree.

### Changed

- **Instance training stops when it stops improving.** RF-DETR has carried an
  early-stopping callback all along and nothing switched it on, so every run
  spent its full epoch count on a fine-tune that typically plateaus in
  single-digit epochs -- one measured here reached 0.999 mAP50 at epoch 3 and
  had 77 epochs left to go. It now honours `early_stopping_patience`, the key
  the semantic path already reads, and stops after that many epochs without a
  meaningful improvement in segmentation mAP. Set it to 0 to run every epoch.

- Instance training evaluates every 5 epochs instead of every epoch, the
  cadence semantic runs already use. RF-DETR ran a full COCO evaluation over
  the whole validation split each epoch, and ran it twice — a second forward
  pass through the EMA weights. The best checkpoint is now chosen among
  evaluated epochs, and the final epoch always evaluates.

  Both changes measured together on a 4 GB GTX 1650 Max-Q, three epochs over
  the same 325-image composed dataset, one configuration after the other:
  **2616 s before, 845 s after** — 3.1x. The input size accounts for most of
  it (932 s on its own, without this cadence change): the backbone is a ViT,
  so 1.72x the pixels costs rather more than 1.72x the time. Epochs get
  cheaper later in a long run under either configuration, so this is the ratio
  over the first three rather than a whole-run average.

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
