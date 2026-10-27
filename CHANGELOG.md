# Changelog

All notable changes to this project will be documented in this file.

## [0.9.13] - 2026-10-27

Seg-Studio opens as an app: the installers make a Start menu and desktop
shortcut on Windows and `Seg-Studio.app` on macOS, and the running app sits
in the taskbar and the tray, or the Dock and the menu bar, until it is
stopped from there. The release also updates Pillow to 13.0.0 and other
dependencies for known vulnerabilities; run the installer again.

### Added

- **Seg-Studio as an app, made by the installers.** `install-windows.bat`
  ends by creating a **Seg-Studio** shortcut in the Start menu and on the
  desktop, and `install-macos.sh` by creating **Seg-Studio.app** in
  `~/Applications`, when the screen is built (the app serves the built
  screen, so without one neither is made). Opening it starts the trainer and
  serving APIs with no console window, opens the screen once the API is
  ready, and stays in the taskbar and the tray (Windows), or in the Dock and
  the menu bar (macOS), with **Open Seg-Studio**, **Open the log folder** and
  **Stop and quit**. Closing the taskbar button, or quitting the macOS app
  from the Dock or with Cmd-Q, stops it the same way, and a quit while the
  app is still starting does not wait for the start. Opening the app while it
  runs, or clicking its taskbar button or Dock icon, opens the screen.
- **Stop and quit asks first when a training, an inference or an AI support
  run is going, on any device**, the Mac's GPU and the CPU included: the
  trainer API's `/train/global-status` gains `training_runs`, the number of
  trainings it is running, which a run on the CPU does not show anywhere
  else. It stops only this folder's Seg-Studio, or what the app itself
  started: a program found on the port, or a Seg-Studio from another folder
  the person chose to keep, is left running. The stop script it runs stops
  whatever listens on ports 8002, 8001, 5173 and 8081, as `stop-windows.bat`
  and `stop-macos.sh` do.
- **A Seg-Studio from another folder, or an older version, on the port** is
  stopped only when the person says so, and the question says when that one
  is running a training. For this the trainer API's `/startup-status` names
  the checkout it runs from, as a short hash of the folder's device and file
  number: the same however the path is spelt, and nothing of the path or the
  user name in it.
- **Windows:** the taskbar button is a minimized window with the shortcuts'
  AppUserModelID, so it shows and pins as the Start menu's Seg-Studio
  (Windows 11 hides a new tray icon behind the ^ arrow). The app runs the
  scripts by their path relative to the checkout, so a folder whose path
  holds a space together with `(` `)` or `&` works.
- **macOS:** with a framework build of Python (python.org or Homebrew) the
  Dock shows Seg-Studio, not Python: the app runs the venv with a copy of
  that Python's executable kept inside `Seg-Studio.app`, replaced by a
  rename when Python is updated. When macOS refuses the app access to its
  folder (in Downloads, Desktop or Documents), the app says so.
- **The app is `scripts/desktop_app/seg_studio_app.py`.** It runs the same
  start and stop scripts as the `.bat` / `.sh` files, with three new
  switches those scripts honour: `SEG_NO_DEV_UI=1` (no Vite dev server),
  `SEG_NO_BROWSER=1` (the caller opens the screen) and `SEG_HIDE_TOKEN=1`
  (with LAN access on, say where the token is kept instead of printing it:
  the app's log is a file). The shortcut and the app are written on the
  machine, not downloaded, so there is nothing to sign or notarize;
  `--no-shortcuts` and `--no-app` skip them. The Windows tray is the Win32
  API through ctypes, with no third-party package; the macOS Dock icon and
  menu bar item use pyobjc (MIT), pinned in
  `apps/trainer_api/requirements-macos-app.txt` and checked by the licence
  gate, and without it the app asks with a dialog instead.

### Changed

- **One icon everywhere: the browser tab's white S on indigo.** The
  shortcuts, the tray, the packaged launcher and the browser tab use the
  same drawing, and the macOS menu bar shows it as a template image that
  follows the light or dark menu bar. `scripts/installer_assets/make_icon.py`
  draws all of them from shapes, and draws the macOS app's icon at every
  size when the app is made.

### Fixed

- **A mask that is not 8-bit goes through a resized export and a resize
  clone.** Both read a mask PNG that is 1-bit or 16-bit through an array in
  its own data type, and Pillow 13 refuses to change that type to fit the
  mode it is given: with the new Pillow both would stop at such a mask. The
  array is made 8-bit first. A 1-bit mask comes out as before (0 and 1); a
  16-bit mask comes out with its class ids, where its raw bytes used to be
  read as pixels.
- **`sam-segment` with `mask_side` answers.** Asked for masks smaller than
  the picture, the endpoint scaled them to booleans, and a boolean mask times
  255 is a 64-bit array the PNG encoder refuses: every such call on a
  picture larger than `mask_side` was a 500. The scaled masks are kept as
  8-bit, as the full-size ones are. Nothing in the app or the bridge sends
  `mask_side`.

### Security

- **Pillow 13.0.0.** The update addresses 41 known vulnerabilities published
  from 2026-10-20 to 2026-10-23, among them heap buffer overflows in the
  JPEG, TIFF and FlashPix decoders; no 12.x release carries the fixes.
  Pillow is the image library of the trainer and the serving API, so
  `apps/trainer_api/requirements.in` and `apps/serving_api/requirements.in`
  now ask for `pillow>=13.0.0,<14`, and the trainer, serving and dev
  lockfiles pin 13.0.0 (from 12.3.0). The licence is unchanged (MIT-CMU).
  Pillow 13 removes a few long-deprecated functions that the code does not
  use; its change to `Image.fromarray` is the first item under Fixed.
- **aiohttp 3.14.5, brotli 1.2.1, certifi 2026.10.23, filelock 4.2.4 and
  setuptools 84.2.1.** The updates address known vulnerabilities (aiohttp
  3.14.3, brotli 1.2.0 / brotlicffi 1.2.0.1, certifi 2026.4.22, filelock
  3.29.0, setuptools 83.0.0). None of them is a package this project picks:
  aiohttp arrives through fsspec (and brings yarl 1.25.1 with it), brotli
  through fonttools, certifi through httpx and requests, filelock through
  torch and `huggingface_hub`, setuptools through torch. The security-floors
  section of `apps/trainer_api/requirements.in` states a floor for each.
  The certifi update takes `emSign ECC Root CA - G3` out of the CA bundle:
  eight intermediate CAs under it had their keys compromised and are
  revoked. The licences are unchanged (aiohttp Apache-2.0 AND MIT, yarl
  Apache-2.0, brotli and brotlicffi MIT, certifi MPL-2.0, filelock MIT,
  setuptools MIT).
- **fastmcp 4.1.2.** The update addresses two known vulnerabilities in
  4.0.1 (fastmcp and fastmcp-slim 4.0.1 → 4.1.2). The trainer lockfile pins
  it, and `FASTMCP_PIN` in `scripts/mcp_server.py` and the install line the
  AI panel shows name the same version. The licence is unchanged
  (Apache-2.0).
- **vite 6.4.5 in the UI's development dependencies.** The update addresses
  known vulnerabilities in the development server and in the code it
  carries to open a file in an editor (vite 6.4.3 → 6.4.5). The start
  scripts run that server on 127.0.0.1:5173 unless `SEG_NO_DEV_UI=1`;
  nothing that ships in `dist/` changes. The licence is unchanged (MIT).

### Upgrading

- **As for any release:** extract the new Source zip, move `projects/` (and
  `models/registry/` if you serve exported models) into it, and run
  `install-windows.bat` again (`bash install-macos.sh` on macOS). The
  security updates are in the environment the installer builds, so they need
  the installer, not only the new code.
- **The installer now makes the Seg-Studio shortcut** in the Start menu and
  on the desktop (`--no-shortcuts` skips it); on macOS `install-macos.sh`
  makes `~/Applications/Seg-Studio.app` (`--no-app` skips it). The `.bat`
  and `.sh` start and stop scripts work as before. A Seg-Studio pinned to
  the taskbar keeps the folder it was pinned from: after installing a new
  release into a new folder, unpin it and pin it again.
- **An environment put together by hand** needs `pillow>=13.0.0` and, for
  AI support and the MCP bridge, `python -m pip install fastmcp==4.1.2`.

## [0.9.12] - 2026-10-15

AI support works from a fresh install and labels a project image by image:
the MCP library is part of the install, the panel finds a model server on its
own machine with nothing configured, and a run rehearses on the images a
person labelled before it writes anything. The release also fixes photo
orientation on import, class deletion in the annotator, crack tracing and the
macOS installer.

### Added

#### AI support: a run goes through the images one by one

- **The program opens the pictures, and the model answers about one at a
  time.** Asked to label, a run works through every unlabelled image of the
  project, or only the images the person named themselves (in the
  instruction, a reply or an earlier turn); an image the model picks on its
  own does not narrow the run. The program
  (`apps/trainer_api/app/core/vlm_agent/walk.py`) opens each image, whole as
  the canvas's Fit shows it or in window-sized parts at a zoom, and asks the
  model one question per picture: where the targets are in it. The answer is
  asked for as a small JSON object (`chat_json` in `backends.py`: Ollama's
  `format`, `response_format` with a JSON schema on OpenAI-compatible
  servers, and a plain chat with the first JSON object taken when a server
  refuses the schema), with boxes on a 0-1000 scale across the picture. Each
  box goes to SAM on the part it was read off, the image is written, and the
  parts that hold mask are shown again for a check, with up to two rewrites.
  An image the run is unsure about gets the Review flag with the reason.
  These short conversations are not added to the run's own history, so
  trimming that history never drops a picture of the walk. Three failed
  calls to the model server in a row end the run where it is.

- **A picture looked at whole may hold none.** The model's brief for a part
  said that most parts hold none; the brief for a whole picture now says
  that some pictures hold none and that the answer is then no boxes. It is a
  default and a small one. On the tutorial sample with one image labelled (a
  black spot on a contact pad), of 16 pictures with no spot the run marked a
  mounting hole on 11 without the sentence and on 9 with it; an instruction
  that says what is not a target ("the round holes are not spots; leave a
  picture with no spot unlabelled") brought it to 1, with every spot still
  found. The user guide says to write that sentence.

- **One of the person's own images is shown as a worked example.** Before
  each picture it is asked about, the model is shown one image a person
  labelled together with the boxes their labels give for it (bridge tool
  `teacher_example`), fetched once per run for each zoom in use. A project
  nobody has labelled, or a model server that does not take the extra
  picture, goes on without it, and the panel says so once.

- **The run asks before it labels.** It says how many images are left and
  about how long they will take, and asks whether to go ahead. The choices
  are buttons under the question; a typed answer works as well. Once the
  person goes ahead, nothing more is asked until the end (unless "Check each
  image with me" is on). Anything but going ahead ends the run with nothing
  labelled, and so does a question nobody answers within twenty minutes. A
  reply that says none of the choices plainly is asked once more.

- **The zoom is rehearsed on the person's labels.** Where no zoom is filed
  for the project, the run first tries Fit and then 50 %, 100 %, 200 % and
  400 % of the picture's own pixels on up to three images a person labelled,
  through the same steps as the real walk and writing nothing, and scores
  what it kept against their masks object by object (bridge tools
  `view_plan(rehearse=true)` and `rehearse(views_json=...)`). The lowest
  zoom whose recall and precision both reach 0.8 is filed with the project
  (`view_choose`), and later runs reuse it. A zoom whose rehearsal would
  take more than 18 looks is skipped, and the panel says so. When no zoom
  reaches 0.8, the best one is filed as not passing, the first unlabelled
  image is labelled at it as a sample and shown, and the person is asked
  whether to go on; stopping takes the sample and the filing back. On a
  later run the go-ahead question also offers "Choose the zoom again".

- **Objects go to SAM, scatterings of small marks to spot detection.** How a
  project's targets are labelled is decided once and filed with the project
  (`way_choose`: `objects` or `specks`). On a project a person has labelled
  it is read from their labels: a median object under 1.5 % of its picture's
  long side, more than 20 a picture and at most 32 px across in the
  picture's own pixels is a scattering of specks; 4 % or more and at most 20
  a picture are objects; anything between is tried both ways on their images
  and the better one kept. On a project nobody has labelled, the person is
  asked once (objects, small marks, or stop). Specks are found image by
  image with `spot_detect`, with the person's marks as the example where
  there are any, written with `spot_write` and looked at once.

- **The parts can be walked by the program or by the model.** A "Parts
  walked by" choice in the panel (the program / the model, remembered in the
  browser) and a `driver` field on the run request (`harness`, the default,
  or `model`) say who moves across an image that is looked at in parts. With
  `model`, the model calls the view tools itself (`next_view`, `image_view`,
  `accept_masks(view=...)`, `kept_drop`, `write_kept`), and the program
  keeps the order and the coverage: a picture out of order is refused with
  the next one named, and the write waits until every part has been shown
  and answered. The worked example is shown only when the program walks.
  Where neither goes -- specks when the model walks -- the run labels with
  the earlier flow and says so; STEP view applies to those runs only.

- **Objects on a project nobody has labelled: the run asks for one image
  first.** A run measures SAM and the size of an object on masks a person
  drew, and checks what it finds against them. The panel says so before the
  model says anything and waits while one image is labelled by hand (**I
  have labelled one** / **Stop**); the image is read and the run goes on
  from it. Where nobody can be asked (the command line, `--yes`), the screen
  says that nothing is written on that run and why.

- **A run stops with the reason when SAM cannot run.** SAM's model file is
  downloaded the first time a model is used. Where that fails, or SAM fails
  on the server in another way, `calibrate_sam` says "SAM could not be run"
  with the server's reason when not one of its asks was answered, and a run
  of the panel ends there, or at the second call in a row that SAM fails on,
  with that reason and what the server says to do.

#### AI support: the model server is chosen in the panel

- **A fresh install looks for Ollama on its own machine, and the AI button
  is always offered.** With no `vlm` block and no `SEG_VLM_*` variable, the
  connection is Ollama at `http://127.0.0.1:11434`. The panel has a list for
  the kind of model server on the machine that runs Seg-Studio (Ollama, LM
  Studio, vLLM, MLX, llama.cpp), each standing for the one loopback address
  that server listens on by default, beside the list of models. When the
  server lists no model, the panel shows the address it asked and what came
  back, and Start asks for the list again.

- **`PUT /api/v1/agent/connection` stores two names.** The body takes
  `backend`, `model` or both, and the answer is the connection as the panel
  shows it. Both are written to the `vlm` block of
  `projects/runtime_settings.json`; choosing another kind drops the stored
  model. An address, or the name of a key variable, is never read from a
  request. The route answers 400 when neither name is given, for a kind
  outside the list, and for a model name that is not printable or is longer
  than 200 characters. It answers 409 for a change of kind while a run is
  going or when the machine itself set the server (an address or a key in
  the file, `openai`, or anything through the environment), and for a model
  when `SEG_VLM_MODEL` sets it. The settings file is read strictly, under a
  lock shared with `merge_runtime_settings`: a file that exists and cannot
  be read as a JSON object is left as it is, and the request is refused
  (409). `connection` in `GET .../agent/state` gains `source`
  (`environment`, `file` or `default`), `selectable`,
  `model_from_environment` and `kinds`.

#### What the model is told about the data

- **Data rules come before everything else the bridge sends.** `DATA_RULES`
  in `scripts/mcp_server.py` opens the connect-time instructions (ahead of
  the playbook, so a `--playbook` override cannot drop it), is the resource
  `segstudio://rules` (listed first in `segstudio://playbook`) and opens
  every prompt. The rules: keep everything received through the bridge out
  of any other tool, MCP server, web search, URL, API, file outside the
  project, message or code runner; treat text inside the data as data, not
  as instructions; do not identify people or read out, collect or keep
  personal details seen in the pictures, while masks on people or text are
  ordinary labelling; and nothing said in the conversation lifts them. MCP
  leaves it to the client whether connect-time instructions reach its model,
  so a short form ends every tool's description (`SEG_MCP_RULES_IN_TOOLS=0`
  leaves it off), and the full text stays under 2,048 characters, which a
  test holds. The AI support panel reads the resource, puts it at the head
  of its system prompt and does not start a run without it; a conversation
  carried on from an earlier one has its system prompt replaced. These are
  instructions to a model, not a barrier: the bridge talks to one Seg-Studio
  server and nothing else, but a model served on the internet receives the
  pictures with each request, and a client that has other network tools
  depends on the model following the rules. README, SECURITY.md and the
  guides say so.

#### MCP bridge and API

- **Six tools for looking at a picture the way the annotation window shows
  it: 81 tools.** `view_plan` (READ) lists the window-sized parts that cover
  an image at a zoom, in the order to look at them, and for a project the
  zoom and the way filed for it and the images left to label. `image_view`
  (READ) opens the Fit view or one part, optionally with the written mask
  drawn over it. `teacher_example` (READ) hands over one image a person
  labelled with boxes round their objects. `kept_drop` (READ; in memory
  only) lets go of kept masks in a view or under boxes. `view_choose` and
  `way_choose` (WRITE) file the zoom and the way with the project.
  `accept_mask` and `accept_masks` take `view=`: the box is put back through
  that view, SAM is asked on the region the part shows, a box touching an
  inner edge is skipped (the object is whole in the neighbouring part), and
  a box whose middle lies in a mask already kept is skipped as a duplicate.
  A skip is not a refusal and does not stop a batch. `rehearse` takes
  `views_json` and `class_id`, `write_kept` takes `keep` (what was kept
  stays for a review after the write), and `teacher_band` takes
  `at_filed_zoom` and reports `object_side_pct`.

- **Pictures arrive as MCP image content.** `image_get_b64`, `image_view`,
  `teacher_view`, `teacher_example`, `accept_mask`, `accept_masks`,
  `accept_points` and `spot_detect` answer with their picture as an image
  block, beside a JSON part that carries no base64 and names, under
  `pictures`, the key each image stands for (`image_base64`,
  `candidates_jpeg`) in order. A picture that is neither JPEG nor PNG stays
  as base64 in the text. Called from Python, the functions return the dict
  as before.

- **Coordinates can be given on a 0-1000 scale.** `units="norm1000"` on
  `accept_mask`, `accept_masks`, `accept_points`, `kept_drop`,
  `spot_detect`, `zoom_score`, `zoom_plan` and `image_get_b64` (its
  `crop_json`) reads coordinates as 0 to 1000 across the picture they were
  read off, each axis by its own side, and `grid_units="norm1000"` on
  `image_view` and `image_get_b64` numbers the ruler the same way. The
  default stays pixels; the AI support panel sends both.

- **`sam-segment` takes a region.** An optional `region` `[x0, y0, x1, y1]`
  in the picture's pixels shows SAM that part of the picture alone. The
  prompt stays in the picture's pixels, the masks come back the size of the
  region, and the reply adds `region`, `full_width` and `full_height` so the
  caller can put them back. The embedding cache is keyed by image and
  region. A region that is not four finite numbers with `x1 > x0`,
  `y1 > y0` and `x0, y0 >= 0`, or that lies off the picture, is answered
  with 400; a region that is the whole picture is treated as none. Labels
  are 1, as before.

#### Annotation

- **The class delete confirmation says what it erases.** It names how many
  images the class is painted in and how many of those a person drew, read
  from the new `GET /api/v1/projects/{project_id}/classes/{class_id}/usage`
  (`{class_id, images, person_images}`; 400 for an id outside 1..254). A
  stroke still waiting for the autosave timer is saved first, so it is in
  the count.

#### Documentation

- **Three new pages, in English and Japanese.** `docs/ai-support.md` covers
  the panel, connecting an MCP client of your own (one section for every
  client that starts MCP servers as local programs), what the model is told
  about the data, and the bridge reference. `docs/ai-support-ollama.md`
  walks through one setup with Ollama from an empty machine to a finished
  run, on Windows and macOS; the page records the machine, the Ollama
  version and the model each step was run with (Windows 11 with a 24 GB GPU,
  and macOS on Apple silicon with 64 GB of memory). `docs/install.md` holds
  the install pre-flight, the steps per OS, Docker, offline machines and
  platform support.

### Changed

#### AI support and the MCP bridge

- **fastmcp 4.0.1 is part of the install.** It is a direct dependency in
  `apps/trainer_api/requirements.in` (`fastmcp==4.0.1`, the version
  `FASTMCP_PIN` and `FASTMCP_VERSION` name; tests hold all three together),
  so AI support and the MCP bridge work without a further step. The trainer
  lockfile gains 49 pins and no existing pin moves: fastmcp and fastmcp-slim
  4.0.1 (Apache-2.0), mcp and mcp-types 2.3.0 (MIT) and the 45 packages they
  depend on. The serving, dev and OpenVINO lockfiles are unchanged. The
  licences are Apache-2.0, MIT, BSD-3-Clause, ISC and Unlicense, with
  cryptography under `Apache-2.0 OR BSD-3-Clause` and exceptiongroup MIT
  with PSF-2.0 portions. `THIRD_PARTY_NOTICES.md` lists every one under
  "Python: AI support and the MCP bridge", and notes the Docutils subset
  rich-rst carries (public domain, two files BSD-2-Clause) and the OpenSSL
  statically linked into the cryptography wheels (Apache-2.0). The PyPI
  licence gate and `pip-audit` pass on the new lockfile. In an environment
  put together without fastmcp the API still starts, and the run endpoint
  answers 503 with the line that installs it.

- **Every picture a model is shown fits a 1344x896 window.** A copy used to
  have its long side set to 1280 px. It is now fitted to the window the way
  the annotation canvas's Fit does it, enlarged when the picture is smaller,
  with each side rounded to the nearest multiple of 32 (`scripts/mcp_view.py`,
  standard library only, shipped with the installer build): 2448x2048 is
  shown at 1056x896, 12000x4096 at 1344x448 and 100x100 at 896x896.
  `image_get_b64` keeps `max_side` and `min_side` in its signature and no
  longer applies them; the reply says so. A box read off a copy is put back
  per axis, so the rounding does not move it. The model is told where a crop
  or a part is in words, not in frame pixels. `teacher_view` and the sheets
  of SAM's answers fit the same window, and `zoom_plan` measures at the Fit
  scale.

- **A policy lists only the tools it can run.** Each tool carries the tier
  its docstring opens with as a tag, and the bridge disables the tiers the
  policy does not allow when it starts: `--policy read` lists 53 tools,
  `write` 79 and `full` all 81. Every call is still checked against the
  policy, and replacing a person's work still needs `--policy full`. The AI
  support panel runs its bridge under `write`, so what it is offered does
  not change. The start-up line reports how many tools are offered.

- **The run endpoint follows the new defaults.** `POST
  /api/v1/projects/{project_id}/agent/run` no longer answers 503 when no
  model server is configured: the connection is then the default one. A run
  with no model named is refused with 400 before it starts, with where to
  choose one. The body takes `driver` (`harness` or `model`; anything else
  400). The command-line example `scripts/examples/qwen_mcp_agent.py` gains
  `--yes` (go ahead without the question, which nobody is there to answer)
  and `--driver`.

- **"Ask me before writing" is "Check each image with me".** On a project a
  person has labelled, the run writes an image's mask and then puts the
  image to the person: keep it, take it back, or stop. With no answer for
  twenty minutes it is kept as written. The label and a tooltip now say
  that; the behaviour is unchanged.

- **A SAM model file that cannot be downloaded has an error of its own.**
  `sam-segment` answers 503 with code `NSS-5011`, the file's name and a hint
  (check the machine's connection, or place the file in
  `models/sam_checkpoints`), where it answered 500 "SAM inference failed".
  The SAM tool of the annotation screen shows it, and
  `docs/troubleshooting.md` names it. The bridge passes on the hint the
  trainer sends with any error, for every tool.

- **An error body says what its message is to its code.** `error.kind` is
  `usual` (the code's registered message), `specific` (the code names the
  error and the message says more) or `uncoded` (an `HTTPException`, whose
  code is only the one its status falls under). `error.code`,
  `error.message` and the top-level `detail` are as they were.

#### Annotation

- **Crack trace finds light and wide lines, and refuses a click that is not
  on a line.** The tracer selected regions that were not connected to the
  clicked crack, found little on light scratches, and from the middle of the
  sensitivity slider merged surface texture into candidates the size of the
  frame. `apps/trainer_api/app/core/crack_trace.py` is rebuilt around a
  centre-line test. Oriented second-derivative line filters run at effective
  scales of 1 to 12 px on a three-level pyramid, for dark and for light
  lines on equal terms. A pixel is a line centre only where the smoothed
  grey is a minimum (or maximum) across the line and both sides rise (or
  fall) by comparable amounts, which leaves out the side lobes beside a
  line, a crack's bright rim and the foot of an edge. Hysteresis runs along
  the centre lines with a low threshold that never falls under the picture's
  own 90th percentile (sliding to the 80th between sensitivity 60 and 100).
  Candidates shorter than 40 px are dropped before widening, so the width
  slider changes thickness only, and a line that ends on another is its own
  candidate, so the side branches of a crack network take one click each. A
  click is refused when nothing under it is a line centre, when the line is
  no stronger than the surface texture, when the selection is a spot (a
  centre line under 16 px) or when even the tightest threshold would take
  more than max(20,000 px, 2.5 % of the frame). `crack-trace/adaptive`
  returns the reason in `why`, and the status line shows it. A result for an
  image the user has left is dropped before anything is shown. The filters
  run in four threads, and the module uses OpenCV, NumPy and Pillow only.

#### Install, CI and release tooling

- **The Docker images, CI and the SBOM job install the lockfiles as they
  are.** Both Dockerfiles, the two CI jobs, `sbom.yml` and the devcontainer
  pass `--no-deps`, as the installers do, and so do the commands in
  `docs/dev-quickstart.md`, `docs/deployment.md` and `CONTRIBUTING.md`. A
  lockfile is a complete resolution; resolved a second time, pip adds what a
  pinned package asks for and the lockfile overrides, which here is
  supervision's `opencv-python` beside the pinned `opencv-python-headless`,
  two packages that both own `cv2/`. A test covers every route that installs
  a lockfile. Checked on Linux with Python 3.11: the three lockfiles install
  with `--no-deps` in one pass, `opencv-python-headless` is the only OpenCV
  build, and fastmcp imports.

- **The SBOMs name the project and say what they cover.** Both CycloneDX
  files take `metadata.component` (name and version) from `pyproject.toml`.
  The trainer SBOM describes the trainer lockfile as installed on Linux with
  the CPU build of PyTorch; what an installer adds outside the lockfiles
  (the CUDA builds of PyTorch and ONNX Runtime, pywinpty on Windows,
  EfficientSAM and TinySAM) is listed in `THIRD_PARTY_NOTICES.md`, and the
  workflow header and `CONTRIBUTING.md` say so. The SPDX file names the
  project and its version as well: syft names a scanned folder after its
  path, so the document was called "." and described a package "." with no
  version; the workflow now hands syft the name and the version from
  `pyproject.toml`. The SPDX step no longer attaches a copy of its own to
  the Release (`upload-release-assets: false`): the attach step, which runs
  after the licence audit, is the only one that does.

#### Japanese on the screen

- **The screen, the AI support panel and the reports read as Japanese.** 509
  of the 992 Japanese screen strings, about 190 AI support messages, the
  release notes, the tutorial, the PDF and Excel reports and the bundled
  examples are rewritten: no word-for-word English, full-width punctuation,
  no space between a number and its counter (counts joined in code drop it
  in Japanese too), and one name for one thing -- しきい値 for threshold,
  確信度 for confidence, ひび割れ検出 for the crack tool, 人がラベルを付けた画像
  for an image a person labelled. Text sent to a model, and the words a reply
  is matched against, are unchanged.

#### Documentation

- **README is a plain entry point, and the guides read plainly.** README
  (en / ja) leads with what the app does and three install steps for Windows
  and macOS; the engineer material sits under one "More detail" section, and
  the install detail, the MCP bridge reference and the project structure
  move to `docs/install.md`, `docs/ai-support.md` and
  `docs/dev-quickstart.md`. The user guide, the handbook, the first-run
  walkthrough and troubleshooting explain terms where they appear (the user
  guide and the handbook gain a glossary), start each problem from what is
  on screen, and carry the same sections in both languages. They describe an
  AI support run as it works in this release.

- **Corrections checked against the code.** Image counts, the epoch default
  and its automatic extension, the hardware table, the names on the screen
  and where the SAM files come from are written from the code. The macOS
  steps ask for Node.js as well. `docs/deployment.md` says what Docker can
  train and gains the htpasswd and checklist sections the Japanese page has;
  both checklists name port 8001. `docs/auto-config-rationale.md` gives the
  condition under which the `min_width` epoch rule runs (a config with no
  epoch count; the screen and the HTTP API always send one). `cli_train.py`
  options that do not exist (`--sweep`, `--no-crop`) are no longer offered,
  and version stamps that go stale are dropped from footers and samples.

- **The Japanese pages read as Japanese.** They read as a word-for-word
  rendering of the English, and GitHub showed each line break inside a
  paragraph as a space in the middle of a Japanese sentence. Every Japanese
  page is rewritten: a paragraph on one line, full-width punctuation, one
  spacing rule, and the screen's own Japanese names (OK, not Mark Clean).
  Where the two languages said different things, both now say what the code
  does; the Japanese README's disclaimer says, as the English does, that the
  authors make no warranty.

- **A reverse proxy needs `SEG_API_TOKEN`.** `docs/deployment.md` said that a
  server with no token answers a proxied request 403 and that
  `SEG_ALLOWED_HOSTS` lets it through. The proxy's `X-Real-IP` or
  `X-Forwarded-For` header marks the request as not from this machine, so
  the answer is 401 and only the token lets it through; the page says so.

### Fixed

#### Import and annotation

- **Photos are stored upright.** A phone or camera held upright writes the
  sensor's rows and records the turn in the EXIF orientation tag. The
  encoder that writes `images/` re-encoded through PIL, which neither
  applies the tag nor keeps it, so a portrait photo came into a `png` or
  `jpg` project lying on its side, and a mask drawn on the upright photo
  (what any other tool produces) landed on the wrong pixels. In a `raw`
  project the tag survived, but the width and height recorded for the image
  were the sensor's. `image_encode` now applies the turn before encoding and
  does not keep the tag. An existing PNG is still kept byte for byte unless
  its own eXIf chunk turns it, which is read from the chunk headers without
  a decode; `raw` keeps an upright photo as it came and stores a turned one
  as PNG. Every import route goes through this encoder: upload, ZIP import,
  video frames, resize-clone and synthetic generation.

- **Deleting a class no longer turns unlabelled pixels into background.**
  After a class delete, and after clearing a class from the selected images,
  the annotator marked every pixel of each cached mask as touched, so the
  next save of that image wrote 255 (not labelled) as 0 (background) over
  everything the person had left unlabelled. Only the pixels that held the
  class are marked now.

- **A mask rewritten by a class delete or merge counts as changed.** Purge
  and merge rewrote a person's masks without moving their revision or
  `lastSavedAt`: the index had been synced before the item was refreshed, so
  the refresh saw nothing to do. A rewritten mask now moves one revision on
  and gets a new `lastSavedAt`.

- **Class routes answer 404 for a project that is gone.** `PUT .../classes`
  wrote the class list whatever the project's state, so an annotator still
  open on a deleted project put a `classes.json` back into its folder with
  its autosave, and got 200. That route, and the reconcile, merge and purge
  routes, answer 404 when the folder is missing or carries the mark a delete
  leaves on a folder it could not remove whole.

- **The Enter that confirms a Japanese conversion no longer sends the
  reply.** Answering the AI support panel's question in Japanese sent the
  reply at the Enter that confirmed the first word, and the bulk tag field
  on the projects page committed its tags the same way. Both ignore an Enter
  that belongs to the input method (`isComposing`, or `keyCode` 229 for
  Safari on macOS).

- **Server errors are shown in the screen's language, and a refusal says
  what was refused.** The trainer answers in English. Every error code now
  has Japanese and English words on the screen (17 of 59 had), used when the
  message is the code's usual one; a message that says more is kept beside
  the Japanese words. A request refused without a code of its own was shown
  as "[NSS-1006] A required parameter is missing" whatever the reason --
  when creating, deleting or importing a project failed, for one -- and is
  now shown with the server's reason. A failure said in Japanese ("作成失敗:
  ...") is drawn as a failure and stays until it is clicked, as one said in
  English did; it was drawn as an ordinary note and cleared after four
  seconds. A test keeps the screen's list and the trainer's in step.

- **The AI panel stays where it was put.** Opened in a tab or a pane that
  had no size yet, it was pushed past the right and bottom edges and stayed
  there. Its place is now kept apart from where it is drawn: a window made
  smaller moves the panel in, and made larger again lets it back.

- **A SAM click after `sam-label-assist` is answered from the right
  image.** `sam-label-assist` encodes its image on the predictor
  `sam-segment` uses, and `sam-segment` remembers which image it encoded
  last. That record was not cleared, so the next `sam-segment` call for the
  image it had encoded before was answered from the other image's embedding.
  The route now encodes under the same lock and clears the record. This
  concerns callers of the API: the app's own screens do not call
  `sam-label-assist`.

#### AI support and the MCP bridge

- **Images that share a name are counted and labelled.** The bridge dropped
  rows whose display name repeated, so on a project where the same files had
  been uploaded twice the tools saw fewer images than there are, reported
  the unlabelled count short, and a run never labelled the rest. Rows are
  told apart by id; `annotation_status`, `dataset_images` and `view_plan`
  count with one function; a name two images share is answered as ambiguous
  with both ids (`ambiguous`, `ids`) and no picture; and the next image is
  fetched by id.

- **A box read off a crop lands where it was drawn.** After `image_get_b64`
  with `crop_json`, the panel's loop filled in `from_width` and
  `from_height` from the size of the last copy without saying where the crop
  sat, and the bridge then mapped the box as if the copy showed the whole
  frame: the box was refused when the crop's shape differed from the
  frame's, and written in the wrong place when it matched. The loop fills in
  nothing now, and the bridge maps a box sent without `from_*` through the
  last copy it handed over, for `accept_mask`, `accept_masks` and
  `accept_points`. The review picture after a write is cut to the crop as
  well, and says that it shows only that part.

#### Training

- **A SAM2 distillation teacher finds its checkpoint.** `load_sam2_teacher`
  looked in `<root>/packages/models/sam_checkpoints` only. The app keeps its
  downloads in `<root>/models/sam_checkpoints`, or under `SEG_MODELS_DIR`,
  so a run through the API or `cli_train.py` with a `sam2.1_hiera_*` teacher
  ended in `FileNotFoundError`. (The training screen sends `dinov2_vitb14`
  and was not affected.) `sam2_checkpoint_dirs()` looks in
  `$SEG_MODELS_DIR/sam_checkpoints` when the variable is set, then
  `<root>/models/sam_checkpoints`, then the earlier place, and a missing
  checkpoint names every place that was looked in. The feature tap `s4` ran
  off the end of SAM2's three FPN scales (`IndexError`): `s1` to `s3` read
  the scales in order and `s4` the coarsest. An unknown tap is refused
  before anything is loaded.

- **`train_config.json` follows a distillation fallback.** The file is
  written before the distillation mode is settled. A run that fell back to
  `distill_mode="off"` trained without distillation while the file, and the
  report read from it, kept the mode that was asked for. The runner rewrites
  the file when the mode changed.

- **`transformers` is labelled for what loads it.** The start-up library
  check and the `requirements.in` section said it is there for distillation.
  Distillation does not import it; the instance-segmentation stack (rfdetr,
  peft) does.

#### Install and samples

- **The macOS installer runs to the end, with the lockfile's own
  packages.** `packages/segcore` names sam-2 by its pinned commit in its
  `distill` extra, and hatchling refuses a direct reference unless the
  metadata allows it: `pip install -e packages/segcore`, which the macOS
  installer runs, stopped with a `ValueError` and the installer with it.
  `allow-direct-references` is set. `scripts/macos/install_macos.sh` also
  installs the two lockfiles and segcore with `--no-deps`, as the Windows
  installer does; resolved again, pip added `opencv-python` beside the
  pinned `opencv-python-headless`, and `import cv2` was whichever was
  installed last. Checked by running `install-macos.sh` in a fresh folder on
  an Apple silicon Mac: the installer finishes, every pinned package is at
  its lockfile version (torch and torchvision are installed unpinned on
  macOS by design), `opencv-python-headless` is the only OpenCV build, the
  UI is built, and the start script brings up the API.

- **With an NVIDIA GPU, the installed packages are the lockfile's.** The
  Windows installer reinstalled PyTorch from the CUDA index together with
  everything PyTorch depends on, and that index was the only one named:
  numpy, filelock, sympy, fsspec and setuptools were replaced by the
  versions it hosts (setuptools 78.1.0, where the lockfile pins 83.0.0). The
  two wheels are now swapped alone. The steps after it that let pip resolve
  (the SAM libraries, OpenVINO) check what PyTorch declares, and the newest
  build on the CUDA index (2.11 on cu128) declares `setuptools<82`, so the
  lockfile is installed once more at the end, without resolving, and the
  licence exclusions are applied after it -- by the Windows installer and by
  `scripts/install.py`. `pip check` reports that one declaration; PyTorch
  2.11 imports setuptools only to build C++ extensions.

- **A package built with `scripts/build_installer.py` carries the lockfile's
  packages.** The build let pip resolve the trainer lockfile a second time,
  the way `opencv-python` came to sit beside the pinned
  `opencv-python-headless` on the other routes, and brought in PyPI's CPU
  build of PyTorch only to remove it again before the CUDA one. It now
  installs the lockfile with `--no-deps`, swaps in the two PyTorch wheels
  alone, takes the pin for what ONNX Runtime needs beyond the trainer
  lockfile (flatbuffers) from the serving lockfile, and installs the
  lockfile once more after the steps that let pip resolve. Checked with a
  lean win64 build: every package the lockfile pins for Windows is at its
  pinned version except torch and torchvision (2.11.0+cu128 and
  0.26.0+cu128, from the CUDA index by design), `opencv-python-headless` is
  the only OpenCV build, and the licence exclusions are in place. Started
  through its own launcher, the package imports the sample, answers a SAM
  click, completes two training runs and labels images through the bundled
  bridge. No pre-built package is published; what is still open before one
  can be is listed in `docs/ROADMAP.md`.

- **On Linux, the macOS wrapper scripts name the Linux commands.**
  `install-macos.sh`, `start-macos.sh` and `stop-macos.sh` run on another OS
  now point at `scripts/install.py`, `scripts/start_local.sh` and
  `scripts/stop_local.sh`, and at `docs/install.md`.

- **The sample's clean images arrive marked clean.**
  `samples/seg-studio-tutorial.zip` carries a `metadata.json` (`hasMask` and
  `markedClean` per image). A clean image is an all-background mask plus the
  mark, and an import reads the mark from that file; without it the 16 clean
  images of the sample arrived as images nobody had labelled, so they were
  left out of training and counted as work to do for an AI support run. The
  images, the masks and `classes.json` are carried over byte for byte. An
  import reports 38 images, 38 masks and 16 clean, and a test imports the
  shipped archive.

#### Found in the pre-release review

- **The Docker screen can save.** nginx forwarded `Host` without the port,
  so the origin check refused every POST, PUT and DELETE from the screen at
  `http://localhost:5173`. It forwards `$http_host` now.
- **A SAM click made while label assist trains gets its own picture.** Both
  shared one predictor, and label assist took one of its two locks, so its
  next image could replace the clicked one between encoding and decoding.
  Both now go through one function that holds both locks.
- **AI support stops offering an image it has already redone.** The path
  that writes an image when a run is cut short kept it on the list of
  images to redo, and the run kept asking for it until its no-progress
  guard ended it.
- **The start scripts keep `SEG_PROJECTS_DIR`, `SEG_DB_PATH` and
  `SEG_MODELS_DIR` when they are set,** as the user guide says, and print
  where the projects are; a restart passes them on.
- **ZIP import removes its temporary file however it ends,** a refused
  archive included.
- **Reports.** An unknown report type answers 400 (it was logged as a
  failure and answered 500), and a batch report on a run with no
  predictions answers 404 with the message in the screen's language. The
  missed-instance table is headed Coverage, the rule it reports, and the
  report says an instance counts as found when the prediction covers half
  of it.
- **The start-up repair and `install.py --offline-pack` install the lockfile
  as pinned** (`--no-deps`), with the same exclusions as the installers, so
  they no longer pull the GUI build of OpenCV.
- **Errors.** A class merge or delete that cannot rewrite a tiled mask logs
  it and counts it; an input error that is not a missing field is sent
  without the English field list on a Japanese screen.
- **MCP tool descriptions match the tools.** `run_metrics_get`,
  `image_upload`, `run_agreement`, `prelabel_run` and `teacher_band` say
  what they return and what their words mean; `view_choose` answers that a
  zoom was not filed (or not taken back) when it was not, and the walk says
  so instead of reporting it done.
- **Docs.** The README's comparison table is a list of the app's own
  features. The first-run walkthrough, the user guide and the handbook say
  that the screen opens in Japanese, that a run's Result tab opens from its
  **Results** button, and that progress shows in the model list and the
  log (there is no loss chart). CONTRIBUTING and the developer quickstart
  install CUDA torch the way the installer does; troubleshooting no longer
  mentions batch-size and input-size settings the screen does not have.

### Security

- **LAN mode keeps the development screen on this machine.** With LAN access
  on, the start scripts ran the Vite development server (port 5173) on every
  interface and the firewall script opened that port. Its proxy relayed a
  LAN request to the API from 127.0.0.1 with no forwarding header, so the
  API took it as this machine's own and the token did not apply. The
  development server now always listens on 127.0.0.1, port 5173 is no
  longer opened (an old rule for it is removed), and the proxy marks a
  request that did not come from this machine as relayed. The LAN reaches
  the app at `http://<address>:8002/ui/`, as the Settings hint says.
- **The Trainer API sends no CORS headers.** It let any page on localhost
  or a private address read its answers; the screen, the development
  server, nginx and the chat example are all same-origin or server-side,
  and need none.
- **The `/annotate/*` and `/cvat/*` proxies are guarded like the API, and
  set only when asked for.** The start scripts set `SEG_ANNOTATION_URL` on
  every start; they now do so only with `SEG_START_LABEL_STUDIO=1`. The
  proxies pass the request guard, and do not forward `X-API-Token` or the
  session cookie.
- **A camera is chosen by its index.** `device_id` takes an integer (a
  string of digits counts as one); any other string, which the capture
  library would open as a URL or a file, is refused.
- **An unreadable `runtime_settings.json` is not written over.** Changing the
  device, the LAN setting or the import format read the file leniently and
  wrote the merge back, so a hand edit that broke the file lost the API
  token and the AI support choice with it. A write now reads it strictly
  and answers 409 when it cannot, leaving the file as it is; a file in
  another encoding is read as unreadable rather than raising.

- **A model server and a Trainer API on this machine are reached directly,
  whatever proxy the machine names.** Both HTTP clients follow the proxy the
  environment or the system names: the panel's urllib client (to the model
  server) and the bridge's httpx client (to the Trainer API). Where a proxy
  is set and local addresses are not excepted, a request for `127.0.0.1`
  was sent to the proxy: the local server never saw it, the run failed, and
  the request, pictures included, was handed to that other host. A loopback
  model server is now opened with an opener that knows no proxy, and the
  bridge builds its client with `trust_env` off when the API is on this
  machine. Any other address is reached as the machine is configured. The
  Windows and macOS start scripts ask the API whether it is ready the same
  way: through the proxy, that check waited its full 60 s and warned about a
  server that was already up.

- **Starting the bridge makes no request to the internet.** The MCP library
  asks PyPI for its newest release when it prints its start-up banner, and
  keeps the answer in a file under the user's home. The bridge switches the
  check off before the library reads its settings
  (`FASTMCP_CHECK_FOR_UPDATES=off`) and starts it without that banner; its
  own `[MCP]` start-up lines are unchanged. The version stays the one the
  lockfile pins.

- **@typescript-eslint 8.70.1 in the UI's development dependencies.**
  `@typescript-eslint/eslint-plugin` and `@typescript-eslint/parser` move
  from 7.18.0 to 8.70.1 (MIT, unchanged). The 8.x line does not depend on
  globby or fast-glob, so `braces`, for which `npm audit` reported an
  advisory, leaves the lockfile together with micromatch, fast-glob, globby
  and what only they needed (14 packages out, 4 in). `npm audit` and
  `npm audit --omit=dev` report none. Development only: the packages lint
  the source and take no part in the build. `.eslintrc.json` gives caught
  errors the same underscore convention as arguments and variables.

### Removed

- **`TrainConfig(distill_teacher_hf_repo=...)`.** The argument was stored
  and never read. segcore's `TrainConfig` no longer takes it, and the
  trainer API no longer forwards it; a stored config that carries the key is
  accepted and the key ignored.

- **Error code `NSS-7002` and `DistillError` ("distillation precompute").**
  No such step exists and nothing raised them. `NSS-7002` is not assigned.

### Upgrading

- **As for any release:** extract the new Source zip, move `projects/` (and
  `models/registry/` if you serve exported models) into it, and run
  `install-windows.bat` again. The installer adds the MCP library this time,
  so AI support needs the new environment, not only the new code.
- **A copy of fastmcp installed by hand is no longer needed.** The install
  has no `pip install fastmcp==4.0.1` step any more: the installer installs
  fastmcp and the packages it depends on at the lockfile's versions.
- **The AI panel is offered without configuration.** The AI button is shown
  on every install; with no model server running, the panel says which
  address it asked. A stored `vlm` block in
  `projects/runtime_settings.json` and the `SEG_VLM_*` variables work as
  before and need no editing. With `base_url` or `api_key_env` in the block,
  `backend: "openai"`, or `SEG_VLM_BACKEND`, `SEG_VLM_BASE_URL` or
  `SEG_VLM_API_KEY_ENV` in the environment, the panel shows that server and
  offers no list of kinds. With only a local `backend` and/or a `model` in
  the block, the panel can now change them: it writes `backend` and `model`
  into the same block and leaves the file's other keys as they are.
- **Callers of the run endpoint:** a start with nothing configured is no
  longer 503; a start with no model chosen is 400; `driver` is optional. The
  zoom and the way a run files stay with the project and move with
  `projects/`.
- **MCP clients:** under `--policy read` and `write` the tools outside the
  policy are absent from the list (53 and 79 of 81), where they used to be
  listed and refuse. The picture tools answer with MCP image content: a
  client or script that read `image_base64` or `candidates_jpeg` out of the
  JSON text reads the image blocks instead, in the order `pictures` gives.
  `max_side` and `min_side` on `image_get_b64` are accepted and not applied.
  Coordinates stay pixels unless a call says `units="norm1000"`.
- **Training:** saved settings need no editing. Code that builds a
  `TrainConfig` itself has to stop passing `distill_teacher_hf_repo`.
- **Photos imported before this release stay as they were stored.** Import
  the original again to have it stored upright.
- **The tutorial sample:** a project made from the earlier archive is not
  changed. Mark its 16 clean images clean on the Annotate screen, or import
  the sample again.
- **macOS:** use `bash stop-macos.sh`, `bash install-macos.sh` and
  `bash start-macos.sh` in place of the `.bat` files, with Python 3.11 or
  later, Node.js and git installed first. The installer reuses a
  `.venv-macos` it finds and does not remove packages already in it, so
  install into the newly extracted folder rather than over an earlier
  attempt.
- **Docker:** rebuild the images (`docker compose up --build`). Inside the
  compose stack the panel's default address is the trainer container itself;
  for a model server on the host or on another machine, set
  `SEG_VLM_BACKEND` and `SEG_VLM_BASE_URL` in the trainer service's
  environment.

## [0.9.11] - 2026-10-13

A maintenance release: dependency updates that address known
vulnerabilities, and the removal of training options that could not be
selected in the app. It adds no features, and nothing changes in how the app
is used.

### Security

- **urllib3 2.8.0, multidict 6.9.1 and fsspec 2026.9.0.** The updates address
  known vulnerabilities (urllib3 2.7.0 → 2.8.0, multidict 6.7.1 → 6.9.1,
  fsspec 2026.4.0 → 2026.9.0). None of the three is a package this project
  picks: urllib3 arrives through `requests`, multidict through `aiohttp` /
  `yarl`, and fsspec through torch, `huggingface_hub` and
  `pytorch-lightning`. `apps/trainer_api/requirements.in` states a floor for
  each (`urllib3>=2.8.0`, `multidict>=6.9.1`, `fsspec>=2026.6.0`), and the
  trainer lockfile moves these three packages only; the serving, dev and
  OpenVINO lockfiles are unchanged. The licences are unchanged (MIT,
  Apache-2.0, BSD-3-Clause).
- **source-map-js 1.2.2 and brace-expansion 5.0.12 in the UI's development
  dependencies.** The updates address known vulnerabilities (source-map-js
  1.2.1 → 1.2.2, brace-expansion 5.0.9 → 5.0.12). Both are patch bumps of
  packages the UI lockfile already had; neither appears in the production
  dependency tree, so nothing that ships in `dist/` changed, and the
  dependency ranges in `package.json` are unchanged. The licences are
  unchanged (BSD-3-Clause, MIT).

### Removed

- **Distillation options that could not be selected in the app.**
  Distillation is feature distillation with an online teacher:
  `distill_mode="feature"` with `distill_teacher_model_dir` set to a
  `dinov2_*` or `sam2*` teacher. `distill_mode` takes `off` and `feature`.
  The app sends one of those two, so none of the following could be selected
  in it, and they are gone:
  - the channel mode (`distill_mode="channel"`);
  - the cached-teacher path (`distill_teacher_cache_dir`);
  - the ensemble-logits path (the `distill_ensemble*` options).

  Feature distillation, the path Auto uses, behaves as before: a short DINOv2
  run on synthetic data on the GPU, before and after the cached-teacher and
  channel removal, agrees in per-epoch loss, distillation loss, peak memory,
  log lines and metrics keys within run-to-run noise.
- **segcore.** `TrainConfig` no longer takes `distill_teacher_cache_dir` or
  the four `distill_ensemble*` arguments, so code that builds a `TrainConfig`
  itself has to stop passing them; a `distill_mode` other than `off` /
  `feature` is read as `off`. Feature distillation with no teacher set logs
  one line and trains without distillation. The functions and classes in
  `segcore.training.distill` that served the removed paths are gone, and
  the metrics keys of the ensemble-logits path are no longer written.
- **Trainer API.** `TrainRequest` loses its three `distill_ensemble*` fields.
  A request or a stored config that still carries any of the removed keys is
  accepted, and the keys are ignored. A `distill_mode` other than `off` /
  `feature` is logged and the run goes ahead with `off`.
- **`scripts/cli_train.py`.** `--distill` takes `off` and `feature`
  (`--distill channel` is an argparse error), and `--distill feature` needs
  `--distill-teacher-dir` naming a `dinov2_*` or `sam2*` teacher.

### Upgrading

- Extract the new Source zip and run `install-windows.bat` again, as for any
  release. Saved training settings need no editing: the removed keys are
  ignored.

## [0.9.10] - 2026-09-29

A licensing release, following a file-level licence scan (ScanCode) of the
source tree, the four git dependencies and 135 of the 156 PyPI packages the
lockfiles pin (their Windows wheels, or sdists where no wheel fits; torch,
torchvision, onnxruntime, opencv-python-headless, triton and the NVIDIA GPU
runtime wheels are binaries and were not scanned file by file).

### Changed

- **DINOv2 is built from a model definition that ships with Seg-Studio.** The
  Apache-2.0 model definition (`vision_transformer.py` and `layers/` from
  facebookresearch/dinov2 at the pinned revision `7764ea0f`) is in
  `packages/segcore/segcore/_vendor/dinov2/` with the upstream licence, and
  `segcore.dinov2_loader` builds the backbone from it. No other upstream code
  is loaded: `torch.hub.load` is not used, so the repository's `hubconf.py`
  never runs (torch's checkpoint cache and download helper still fetch the
  weight file). The weights are the same published
  file (a copy already in torch's cache is reused, and `dinov2_vitb14` is
  checked against its SHA-256), and the outputs are bit-identical, so Auto's
  recommendations do not change.
- **Every install route applies the licence exclusions.**
  `install-windows.bat`, `scripts/macos/install_macos.sh`,
  `scripts/install.py`, the trainer Docker image, the offline pack and the
  packaged build apply one table (`scripts/_nc_stubs.py`) right after
  installing the lockfile, and a failure stops the install. The table
  replaces torchmetrics' Extended Edit Distance module (licensed for
  non-commercial use only; Seg-Studio uses torchmetrics for detection mAP)
  with an Apache-2.0 stub, and deletes the pyphen hyphenation dictionaries
  whose licence is GPL-only, unclear or not stated: pyphen's LGPL election
  covers its code, not the dictionaries, and no report turns hyphenation on.
  The trainer API checks at startup and applies the table again when a later
  `pip install` has put the files back (unless `SEG_STARTUP_NO_INSTALL` is
  set).
- The installers pin `timm` (1.0.28) and `coremltools` (8.3.0) to the
  lockfile's versions where they name them.

### Dependencies

- cuda-bindings 13.3.1 → 13.4.3 (Linux only, pulled in by torch); Apache-2.0
  from 13.4.

### CI

- The git-dependency licence gate scans all four git dependencies, EfficientSAM
  and TinySAM included (`scripts/ci/git-deps.txt`).
- In the PyPI licence gate, a package under a proprietary licence
  (`LicenseRef-*`, "Other/Proprietary License") needs a reviewed entry in
  `scripts/ci/dep-license-allowlist.txt`. The NVIDIA CUDA runtime wheels torch
  installs on Linux are listed there.

### Documentation

- `THIRD_PARTY_NOTICES.md`, `NOTICE`, `licenses/third_party/lgpl/README.md`
  and `licenses/third_party/MODEL_WEIGHTS.md` describe the vendored DINOv2
  model definition, the install routes that apply the torchmetrics stub,
  pyphen's dictionaries, timm's pretrained weights (not loaded) and
  cuda-bindings.

### Upgrading

- Extract the new Source zip and run `install-windows.bat` again, as for any
  release. Starting the new version is enough on its own: the trainer API
  applies the exclusions to the existing environment at startup.

## [0.9.9] - 2026-09-29

### Added

#### Labelling with a vision model (AI support)

- **AI support: a vision model labels the open project.** An AI button
  at the foot of the Annotate tool column opens a panel where you type an
  instruction. The trainer API runs the labelling loop against a model server
  you configure and drives the MCP bridge (below) as a subprocess, so every
  mask the model writes arrives through the same routes the browser uses,
  carrying the header that records who made it. The conversation streams into
  the panel as it happens: the model's words on the left, yours on the right,
  and its tool calls and the pictures it was shown folded behind a count you
  can open. Before a run writes anything it works through four steps -- look
  at the pictures, read the images a person already drew and name the object,
  measure how to cut it, and rehearse on one of those images, scored against
  that person's own mask -- and the model can ask you a question, which you
  answer in the same box. The panel can be moved and resized and remembers
  both per project, and on a Japanese screen the run's step names and
  rehearsal verdict are in Japanese. The button appears only once a model
  server is configured.

- **Stop, Pause, Clear and STEP view.** Stop ends a run between steps, so
  nothing is half-written; a question the model is waiting on is answered at
  once (within about half a second) when you press Stop. Pause holds the run,
  and a question standing while the run is paused does not time out.
  Otherwise an unanswered question stands for twenty minutes, and then the
  run carries on and says what it assumed. Clear (or `/clear` / `/reset` in
  the box) empties the panel and the conversation the next run would
  continue, and leaves the run log alone. STEP view (or `/debug-step` /
  `/debug-off`) holds the next run after each of its setup steps until you
  say to carry on; it is remembered per project.

- **A run survives a reload and a restart.** Its events are kept whether or
  not a screen is listening, so reopening the panel or reloading the page
  rejoins a run in progress. The conversation is filed in
  `projects/<project id>/agent_thread.json` as the run goes (the last 400
  events), and every event is appended to
  `projects/<project id>/agent_logs/<started>.jsonl`, pictures left out. The
  panel shows the stored conversation when it opens, a finished run's
  included, and a project switch clears it.

- **The model server is configuration, not a request.** Set
  `SEG_VLM_BACKEND`, `SEG_VLM_BASE_URL`, `SEG_VLM_MODEL` and, for a server
  that wants a key, `SEG_VLM_API_KEY_ENV` -- the *name* of the environment
  variable that holds the key -- or the same four as `backend`, `base_url`,
  `model` and `api_key_env` in a `vlm` block of
  `projects/runtime_settings.json`; the environment wins. No endpoint writes
  them: where the server sends a project's images is decided on the machine,
  not from a browser. `backend` is one of `ollama`, `openai`, `mlx`, `vllm`,
  `lmstudio` and `llamacpp`, and without a `base_url` each uses its own
  default address: Ollama `http://127.0.0.1:11434`, LM Studio
  `http://127.0.0.1:1234/v1`, vLLM `http://127.0.0.1:8000/v1`, MLX and
  llama.cpp `http://127.0.0.1:8080/v1`, OpenAI `https://api.openai.com/v1`.
  The panel lists the models the configured server holds, and a run may name
  one of them. When the server lists its models, a name it does not hold is
  refused before the run starts; when the listing comes back empty or fails,
  any name is accepted and sent to the server as it is. Listing the models
  loads nothing and gives up after ten seconds.

- **fastmcp is optional, and pinned when you add it.** AI support and the
  MCP bridge need the MCP client, not shipped with Seg-Studio. Install it into
  the Seg-Studio Python -- `<that python> -m pip install fastmcp==4.0.1`
  (`bin/python` on macOS and Linux); a bare `pip` in a new shell usually
  belongs to another interpreter. Its licence, and the packages it brings
  with it, are recorded in THIRD_PARTY_NOTICES.md under "Optional, installed
  by the user (not bundled)". Until it is installed, the run endpoint answers
  503 with that exact command; it also answers 503 when
  `scripts/mcp_server.py` is missing or no model server is configured. The
  installer build now stages `scripts/mcp_server.py`, `scripts/mcp_recipe.py`
  and `scripts/mcp_playbook/`.

- **What a run can and cannot touch.** The bridge a run starts is limited to
  `--policy write`, never `full`, so nothing started from the screen deletes
  a run or clears a class. Every call is held to the project the run was
  started on, whatever project the model names, and an image id that is not
  in that project's image list is refused before anything is sent. A run
  cannot replace a mask a person drew or an image a person marked clean:
  `overwrite` is not offered to the model and is dropped if it sends one.
  Replacing a person's work is done on the Annotate screen. The bridge is
  started against the port and socket address the request arrived on, always
  over plain HTTP, with only the MCP SDK's list of safe environment variables
  plus `SEG_API_TOKEN`. One limit: with LAN access (or `SEG_HOST=0.0.0.0`)
  configured but the server started by hand with `--host` set to a single LAN
  address, the bridge is sent to loopback, which that server does not answer;
  start it with the configured host, or set `SEG_HOST` to that address.

- **A long job runs as long as the work.** The step limit is high enough not
  to be anyone's limit; what ends a run is standing still, forty steps with
  nothing kept and nothing written. The conversation counts each picture at
  its real size and is trimmed before the model server's window is full; a
  trim keeps a tool call together with its results and any picture between
  them, and never leaves a result without the call it answers, which
  OpenAI-compatible servers refuse. The model server is called off the API's
  event loop, so the Annotate screen, Stop and Pause stay responsive while
  the model thinks.

- **See which projects are being labelled, from any project.** A widget
  beside the training one lists every project with a run going, the image it
  is on, and whether it is paused or waiting for your answer -- marked by a
  glyph and a word as well as by colour. It reads the new
  `GET /api/v1/agent/runs` every three seconds while the tab is visible.

- **The endpoints behind it**, under `/api/v1/projects/{project_id}/agent/`:
  `run` (POST, streams NDJSON, one run per project -- a second start gets
  409, also when two arrive at the same moment), `stop`, `pause`, `reply` and
  `clear` (POST), and `state`, `thread` and `models` (GET).

#### Watching an agent work

- **The screen shows the bridge at work.** The MCP bridge labels through the
  same routes the browser uses, so a person with the project open used to see
  nothing until the next project switch. The bridge now names itself on every
  request (`X-Seg-Agent: mcp/<policy>`, `X-Seg-Agent-Tool: <tool>`), the
  annotation routes record those in an in-memory feed served at
  `GET /api/v1/agent/activity`, and an agent can post a step note to
  `POST /api/v1/agent/note` (the action, the image and a count; a request
  without the agent header is ignored). The UI polls the feed once a second
  while the tab is visible. A chip beside the brand names the tool at work
  and stays up for ninety seconds after the last event, so it is there
  through a vision model's thinking. A browser request carries no agent
  header and leaves no trace, so the feed shows only what an agent did.
  Vermilion with a pulsing dot rather than a colour alone, as elsewhere in
  the UI.

- **Follow the bridge (LIVE ON / LIVE OFF).** A switch on the chip, on by
  default, moves the annotator to the image the bridge is working on and
  shows each mask as it is written. An image the bridge wrote is read again
  when it is on screen, never one with unsaved paint, and a mask is never
  painted onto another image. The canvas shows results -- the masks written
  -- not the prompts the model tries on the way; `sam-segment` records that
  an agent is working on an image, not where it pointed.

- **An agent can flag an image for a person's eyes.**
  `POST .../datasets/annotate/review` sets the index's draft flag with a
  reason; the bridge tool `mark_review` calls it; the image list shows a
  vermilion "Review" badge, and a save clears it. Deleting a class, or
  merging it into another, clears the flag on every image left with nothing
  painted, and the list is read again at once; an image still carrying
  another class keeps its flag.

- **What an agent writes carries its author.** The index entry of a mask an
  agent writes records the agent (`by`); a person saving the mask takes it
  back. The mask route, mark-clean, unmark-clean and recipe apply refuse an
  agent's write over a person's work -- a mask they drew, or an image they
  marked clean -- with a 409 that says so, unless `overwrite=1` is passed. An
  agent's own writes and drafts stay its own to replace. An agent that marks
  clean an image a person already marked clean leaves the mark the person's.

#### MCP bridge

- **The bridge can annotate, not only look: 75 tools, up from 37.** Under
  `--policy write` an MCP client can now propose a mask with SAM
  (`sam_segment`, which returns all three granularities with their pixel
  areas), write one (`mask_put`), declare images clean (`mark_clean`), add an
  image (`image_upload`), flag an image for a person (`mark_review`), and let
  a finished run draft every unannotated image (`prelabel_candidates`,
  `prelabel_run`). It can create and split a project (`project_create`,
  `dataset_set_split`), read the split a run used (`run_splits`), predict and
  judge its own work (`predict_batch`, `predict_status`, `predict_verdicts`,
  `predict_operating_points`, `run_agreement`, `mask_stats`,
  `instance_counts`, `instance_preview`, `class_presence`, `layout_doctor`,
  `report_generate`), and find and repair masks left pointing at deleted
  classes (`classes_reconcile_check`, `classes_reconcile_fix`). `clear_class`
  erases one class and, like `train_run_delete`, needs `--policy full`. The
  startup banner counts 49 READ, 24 WRITE and 2 DESTRUCTIVE tools;
  `teacher_band` and `calibrate_sam` are WRITE because they record something
  with the project (class 1 when the class list lacks it, and the
  measurement). A test reads the source to prove that every tool's declared
  tier and the policy check it runs agree, and another checks every path the
  bridge calls against the app's own route table.

- **Under `--policy write` no tool writes over a mask a person drew or an
  image a person marked clean.** `mask_put`, `write_kept`, `spot_write`,
  `mark_clean`, `recipe_apply` and `prelabel_run` leave such a mask and such
  an image as they are, and `mark_review` does not flag them; `overwrite=true`
  on either needs `--policy full`. The first five keep no copy of what they
  replace; `prelabel_run` keeps a copy of the mask file in `masks_replaced`
  only until the next draft over that image, and none of a clean mark. Its
  `overwrite=true` is checked against every image it would draft over -- the
  images named, or every image in the project when none are -- and under
  `write` the whole call is refused if any of them carries a person's work;
  otherwise it drafts over them, agents' masks and earlier drafts included. A
  clean mark with no recorded author counts as a person's. `mask_put` takes
  `class_id` to write a `sam_segment` level (0 and 255) as that class; a 0/255
  mask without it is refused, and `class_id=255` writes it unchanged as
  background and ignore. The rule covers masks and clean marks and nothing
  else: other WRITE tools still change what everyone using the trainer shares:
  `hardware_set_device` switches the compute device for every project,
  `classes_set` and `assistant_context_set` rewrite a project's classes and
  assistant notes, `dataset_prepare_annotate` and `dataset_set_split` move
  images between splits, and `train_start` starts training.

- **The counting recipe as tools.** `teacher_band` reads the masks a person
  drew and computes the acceptance band, the expected count and a shrink
  calibrated on those masks; `accept_mask`, `accept_masks` (every box you can
  see, in one call) and `accept_points` (a point per object, or a group of
  points for an object made of parts that look different) ask SAM and keep
  the level that fits; `write_kept` unions, shrinks and writes what was kept,
  using the background value the person's masks use. `teacher_view` shows the
  model the person's own objects outlined and cut out; `calibrate_sam`
  measures which SAM model and which way of prompting fit the project and
  files the answer with the project, so a later run reads it back instead of
  measuring again; `zoom_plan` and `zoom_score` measure how close a model has
  to look on an image whose answer is known; `propose_boxes` finds where the
  person's own objects turn up again in an image. The arithmetic lives in
  `scripts/mcp_recipe.py`, tested on small shapes and end to end against a
  fake trainer.

- **How the counting recipe decides.** It starts from a vision model's boxes
  passed to `sam_segment` as `box_json`, which fit better than clicks on
  template matches and, on touching objects, keep SAM off the neighbour;
  crowded frames still defeat the model and keep the templates, and a second
  ask with the objects found so far painted over recovers an under-count. The
  acceptance band is taught from reference images taken across the whole
  image list (ends first, then by halves), widens when the person has drawn
  only a few objects and says so, and gives the object size in pixels as well
  as in fractions of the frame. The texture and contrast floors are measured
  as a share of each picture's own texture and contrast, and a floor that
  would admit the background of the frames it was set on is dropped, with the
  reason. The shrink is calibrated in SAM's own pixels on a sample of the
  person's objects, and a picture's gradient is computed once. Refusals in a
  row are counted per image: from four the answer says that moving the box
  does not change a texture floor, from eight it says to stop and write what
  is kept. An image with fewer objects than the reference images asks for a
  person only when the gap is wide enough to mean something was missed. The
  playbook records the overlap fallback, a comparison of the SAM models (stay
  with `mobile_sam`), and the kinds of project the recipe does not suit:
  particles and fibres, "the odd one out", tiny things on texture, defects
  named as objects, and zoom without a normal reference.

- **Many small marks, from one example.** `spot_detect` finds the specks that
  look like one example -- a point, or every speck a person painted on an
  image they drew (`like_item_id`) -- and `spot_write` writes what it found
  as the image's mask; an example image may be tiled, and its stored mask is
  read rather than a PNG export that may be older. `steps` reports how far a
  run has got through the setup steps, and `rehearse` scores a run's work on
  an image a person drew against that person's mask and writes nothing.

- **Drafts say they are drafts, and `mask_stats` says which need a person.**
  A draft from `prelabel_run` carries `draft: true` and the run that made it
  on its index entry (a save through the mask route clears it). `mask_stats`
  (`GET .../datasets/annotate/mask-stats`) puts every mask's area and region
  count on the table, grouped by who made it -- a person, an agent or a draft
  -- and lists the masks that differ from the person's, with the reason. It
  reads the files and runs no inference.

- **A run has to earn the right to draft.** `run_agreement`
  (`GET .../predict/agreement`) scores a run's predictions against every mask
  a person drew in its own project -- mean and median IoU, how many images
  score zero, regions drawn per region the person drew, whether the clean
  images came back empty -- and ends with a verdict. A score on a handful of
  images can be far better than on all of them, and a draft from a weak run
  costs the person more than a blank image. It reads masks as the annotator
  writes them (class ids in the first channel, never through a palette),
  counts every painted class as foreground, scores tiled images, and counts a
  labelled image whose mask cannot be read in `n_missing_prediction`, so
  `n_labelled = n_scored + n_missing_prediction`.

- **Projects by name.** Every tool takes a project's id or the name a person
  sees in the browser, Japanese included; an ambiguous name is refused with
  the matches, and a name that matches nothing lists what exists. A WRITE or
  DESTRUCTIVE tool takes an id, an exact name or a name ignoring case, never
  a fragment of one.

- **A Seg-Studio on another machine.** `--token` (or `SEG_API_TOKEN`) sends
  the shared secret as `X-API-Token`, so the bridge can reach a server bound
  to the LAN; it sent none, and such a server answered 401.

- **A playbook the bridge hands to every client.** Eight markdown pages under
  `scripts/mcp_playbook/` reach a client three ways: the overview as the
  server's instructions on connect, every page as a resource under
  `segstudio://playbook`, and four prompts (`count_objects_from_teacher`,
  `mark_single_object`, `find_defect_candidates`, `bootstrap_loop`) that
  render a page with a project filled in. The pages say which tool to reach
  for on which kind of project, record the method comparisons behind that
  advice, and say where a recipe does not work; one says what can be read off
  the images a person already annotated, which tool reads it and what it
  decides, including what was tried and found not to help. `--playbook DIR`
  (or `SEG_MCP_PLAYBOOK`) layers a site's own pages over the shipped ones. A
  test checks that every tool a page names exists.

- **Example scripts for a local vision model.**
  `scripts/examples/qwen_mcp_agent.py` drives the bridge from the command
  line, and `scripts/examples/qwen_mcp_chat.py` is a chat page over the same
  loop. Neither is part of the UI -- the AI support panel is -- and any MCP
  client can do the same. Which model answers is a flag: `--backend ollama`,
  `mlx`, `vllm`, `lmstudio`, `llamacpp` or `openai` (anything speaking
  `/v1/chat/completions`), with `--base-url` to say where it is and
  `--api-key-env` naming the variable that holds a key, which is sent only to
  that server. Beside the conversation the page lists every tool call and
  shows four kinds of picture. An image a tool hands back (`image_get_b64`, or
  `teacher_view`'s outlines of a person's objects) is shown as it came back,
  shrunk to at most 640 px on its long side, with none of the model's boxes on
  it. SAM's answers and what `spot_detect` found are the pictures the bridge
  draws for the model: for SAM, a panel per answer cut around the box (the
  whole image instead for an answer that spills out of that view), with the
  box, points or outline the model sent drawn in orange (for `accept_masks`,
  a row each for up to three of its boxes); for `spot_detect`, the whole
  image with each find ringed and, when it is shown at less than half size,
  numbered close-ups at full size below. After `write_kept` writes a new mask,
  the page shows that mask on the copy the model looked at, the rest dimmed
  and its edge in white (what `spot_write` writes is not shown back). And when
  the model puts a question about an image it has fetched to you, the page
  adds that copy with the boxes of the `accept_mask` and `accept_masks` calls
  that kept something since the image was last written or reset ("the boxes so
  far") -- the only picture with the model's boxes over the whole image, and
  one the model itself is not shown. The divider between the two columns can
  be dragged, and the page is in Japanese or English (`--lang`). It has stop
  (which also answers to typing "stop"), pause and resume, an "Ask each time"
  box that, unticked for a batch, keeps the model's questions from coming to
  you (it decides and says in its report what it assumed), questions whose
  answers are offered as buttons, a status line, a context meter, one
  conversation per tab with a new-conversation button, and an instruction sent
  mid-run interrupts rather than being refused; the page and the server's own
  messages follow `--lang`. The conversation is written down as it happens and
  survives a reload and a restart of the page's server. The Connection button
  picks the server and the model; the address and the key variable come from
  `--base-url` and `--api-key-env` and are shown read-only, and only the
  server's name and the model are remembered.

#### Annotation

- **A SAM click can be read at three granularities.** A point on a bolt
  thread could mean the mark, the thread or the bolt, and SAM proposes all
  three nested masks. The annotator kept whichever scored highest, which is
  often the whole object and can cover little of a small defect. Three
  buttons under the SAM tool -- sub-part, part and whole -- switch between
  the candidates without another round trip, for a click and for a box alike;
  the class is applied when a reading is chosen, and the default is
  unchanged. `sam-segment` adds `levels`, one entry per candidate with name,
  mask, score and pixel area, plus `default_level`.

- **Invert a SAM preview.** When SAM answers with the surround rather than
  the thing, the Invert button beside Confirm and Cancel -- or I -- takes
  everything the preview did not, inside the box when a box was drawn and
  across the image otherwise; press it again to go back. Nothing is written
  until Confirm.

- **A rectangle tool sits between the brush and the eraser.** Drag a box and
  it is filled with the active class; the shortcut is R. While dragging, the
  box shows as a solid outline with a translucent fill, against the dashed
  outline of SAM Box, so the line style says which box tool is in hand
  without relying on colour, and the box stays visible if the pointer wanders
  off the image. Pixels that already hold the class are not re-emitted, so a
  single undo takes the whole box back exactly. It is not available on the
  tiled viewer used for very large images, which handles the brush and eraser
  only, as every other tool there already is.

- **A finished training run can draft annotations for the images that have
  none.** The Draft from a trained model panel uses a project run's
  predictions as a starting point, replacing the auto-label button. Apply to
  this image lays the prediction over the open image as a draft to confirm or
  cancel; Apply to all images writes drafts into every unannotated image. An
  image that already carries a mask is left alone, even one marked all
  background, and a prediction with no foreground is reported rather than
  adopted. Adoption reads the prediction already on disk when there is one
  and needs a model only for the images without one, so a run trained here
  and never exported to ONNX can still draft. The draft is written where the
  mask lives, under one project lock with its "is there a mask already"
  check. Its index entry records that it is a draft and the run that made it
  (`draft`, `draftRun`), not who asked for it; an agent that asks shows in
  the activity feed. `POST .../train/runs/{run_id}/prelabel` streams NDJSON
  and takes `overwrite`, which first copies each mask it replaces into
  `masks_replaced` -- one copy per image, which a later overwrite of the same
  image replaces.

- **Merge one class into another.** Deleting a class purges its pixels;
  merging moves them. For the state ordinary use produces -- a class deleted
  and remade under a new id, two classes with one meaning and the paint split
  between them -- the browser offers a dialog that picks both sides and
  renames the survivor in the same step. Masks, prepared copies, the index
  and the undo stacks are all renumbered. The class buttons stand in one
  column: add, merge, delete.

- **The panes either side of the image can be resized on the Annotate and
  Results tabs.** Both dividers drag; they sit exactly where the old gaps
  were, so nothing moves until you move it, and each width is remembered per
  tab in the browser. Double-click a divider to put it back, or nudge it with
  the arrow keys. The training monitor's divider behaves the same way, and on
  narrow screens the layouts still collapse to one column.

#### Results and reports

- **Every image on the results tab is judged against its annotation.** A
  badge says whether the model found the annotated defects, missed them,
  over-detected or left a clean image clean. The figure is the share of the
  annotated area the prediction covered, with the area predicted outside it
  as excess, so a stray annotated pixel cannot turn a well-covered defect
  into a miss. The judgement follows the confidence slider and the size
  filter, and older predictions are judged in the background.
  `GET .../predict/verdicts` gives the same answer, with counts and a
  `pending` tally, and the evaluation report uses the same rule, so the two
  cannot disagree.

- **The image list marks every row with its result and can be filtered by
  class.** Each row carries one of five silhouettes (square, triangle,
  circle, bar, slash) so states differ by shape, not only colour, with a mark
  for not yet judged. A Match column shows the area match rate: an em dash
  means nothing is annotated, a blank means not yet computed. The per-class
  dots become a filter strip in the header -- click a class to show only
  images where it was predicted, and Up and Down walk that order -- and a
  count strip says how many images are detected, missed, over-detected and
  clean. Long filenames truncate in the middle, keeping the serial number and
  extension.

- **Operating points are swept over confidence and minimum area together.**
  The recall-first, balanced and precision-first buttons come from a sweep
  over the run's own predictions on disk (no inference), set the confidence
  and the area filter in one click, and say what each choice costs: the floor
  it applies, defects found out of defects annotated, and when the area floor
  cannot go higher. The floor never exceeds the smallest predicted component
  that sits on a real defect, so a preset cannot tell the line to ignore
  everything below a large size. The size filter (Min and Max area) sits
  beside Confidence and applies to the picture and every number as soon as it
  is typed. Served at `GET .../predict/operating-points`; a run with no
  predictions yet falls back to the threshold-only presets.

- **The end-of-run report measures how much of each defect region was
  found.** Pixel F1 charges a defect for its outline, so a prediction whose
  only error is a thin boundary band still loses F1. PRO is the share of each
  annotated region the prediction covered, averaged per region so one large
  defect cannot drown out small ones. Since predicting everything scores 1.0
  on PRO, it is paired with the false-positive rate and integrated as AUPRO.
  Beside F1 on the results tab, PRO and FPR tell a boundary a few pixels off
  from small regions missed. metrics.json gains `pro_curve` and `aupro`;
  older and full-image runs lack them.

- **Counting runs get the results view that segmentation runs have.** A count
  run now records count precision, count recall and objects counted
  correctly, so the balanced, recall-first and precision-first presets are
  offered, labelled Objects rather than Defects, and a preset that counts
  fewer objects than balanced warns how many. Composition records both sides
  of the source split, so the train/val badge on each row is filled. Older
  runs are unchanged.

- **Save a report as a file.** The report viewer has a button that saves the
  report as one self-contained HTML file, images included, named after the
  run and the report.

#### Training

- **Training can shift each patch by a small random offset, for fixed-camera
  setups.** With a fixed camera and jig a model can learn where defects
  usually sit rather than what they look like. A new `augment_translate`
  field on the training request (0 to 0.5, the largest shift as a fraction of
  the side) moves the image and its mask together by a whole number of
  pixels, so labels are never resampled. The band the shift leaves empty is
  mirror-filled in the image and marked ignore in the mask, so the frame edge
  does not read as empty. It is off by default and has no control on the
  Training tab yet; the run log reports the value in use.

- **Count training can say how the objects lie, Scattered or Aligned.** Every
  synthetic cutout used to be turned to a random angle, which is wrong for
  parts always placed the same way up: the model learns poses it never meets,
  and a run can score well on its own composites while counting the real
  photograph badly. The count-training panel gains a How objects lie setting
  with two drawn options. Scattered is the default and composes exactly as
  before; Aligned keeps each cutout on its own axis with a little slack, and
  stacked pairs join along it. It shows in the synthesis preview and travels
  as `instance_arrangement`.

- **Composed count datasets record each cutout's full mask from before it was
  occluded.** The composer pastes one cutout at a time, so at each paste it
  holds that instance's complete extent, a label no annotator of a real
  photograph can produce. The full mask is now written to the composed COCO
  as `amodal_segmentation`, `amodal_area` and `amodal_bbox`. The visible
  segmentation is unchanged and remains what training reads, the fields are
  absent on instances from real photographs, and composed images are
  byte-identical to before. Nothing in the application reads them yet.

- **A prepared dataset says which file every image is, and a run records what
  it read.** Preparation writes `prepared/dataset.json` beside the splits,
  naming the image directory and giving every split id the filename, size and
  modification time it resolved to. Everything that reads images after
  preparation looks them up through it instead of guessing an extension, so
  an id that does not resolve is named rather than silently dropped. A
  finished run's `train_config.json` carries `images_layout`, `image_format`
  and `dataset_schema`, with a copy of the descriptor beside the run.
  Existing projects prepare once more on their first run after upgrading.

#### Images and projects

- **Duplicate a project without copying its pictures.** Duplicate asks for a
  name and whether the labels come too, then hard-links the images and copies
  only what is small -- the class list, the masks (a tiled mask is copied for
  real, since it is written in place) and the index -- so it is near enough
  instant at any size. Memo, tags and sort order come along; training runs,
  reports, exports, prepared copies, caches and conversations stay with the
  original. Served at `POST /api/v1/projects/{id}/duplicate`.

- **Each project stores its imported images in a format you can read and
  set.** `GET`/`PUT /api/v1/system/import-defaults` sets the format new
  projects start with and `GET`/`PUT /api/v1/projects/{id}/image-store` a
  project's own. The choices are `png` (lossless, the default), `jpg`
  (quality 95 at full 4:4:4 chroma, since 4:2:0 subsampling visibly dims a
  one-pixel tinted defect) and `raw`, which keeps PNG, JPEG and WebP uploads
  byte for byte and turns TIFF, BMP and GIF into PNG. A project's format
  freezes once it holds an image, with a 409 pointing at the converter below;
  existing projects are stamped PNG on the first start after upgrading.

- **A project's images can be converted to another format in place, without
  losing any.** `POST /api/v1/projects/{id}/datasets/convert-images`
  re-stores every image in the project's configured format, or the `format`
  given as a query parameter; it is the one way to change the format of a
  project that already holds images. It refuses to overwrite a file whose
  target name is taken, saves the annotation index before removing any source
  so a crash leaves a duplicate rather than a broken project, keeps alpha,
  and stamps the new format only when every file made it.

- **A doctor reports what a project's images really are.**
  `GET /api/v1/projects/{id}/layout/doctor` opens each file and counts what
  it is from the bytes -- PNG, JPEG with its chroma subsampling, WebP -- plus
  names their contents contradict, split ids that resolve under the layout in
  force and under the originals, index ids unreachable through whitespace,
  files the index names that are gone, and files nothing names. It reads
  headers, not pixels, so tens of thousands of files fit one request.
  `POST .../layout/measure` stores the census on the project, and
  `scripts/migrate_prepared_images.py doctor` prints it from the command
  line, writing nothing else.

- **A project can train from its original images instead of a second copy.**
  The re-encoded copies under `prepared/images/` can be as large as the
  originals. `project.json` now records which copy a project reads, as
  `images_layout`, and a project pointed at its originals copies nothing.
  `POST /api/v1/projects/{id}/layout/flip` switches to the originals and
  `.../layout/rollback` back; neither copies or deletes an image. A flip is
  refused while a run is running or reserved, when a split id has no file
  under `images/`, or for a CVAT-exported dataset; a rollback is refused once
  the copies are gone. Nothing changes until a project is flipped.

- **The prepared copies can be reclaimed from the command line, with a
  backup.** `scripts/migrate_prepared_images.py flip`, `rollback` and
  `reclaim` do from a shell what the layout endpoints do; `reclaim` is the
  one irreversible step and has no URL. Every subcommand is a dry run unless
  given `--apply`, and `--all --apply` is rejected. `reclaim` requires
  `--backup`, verifies each backup copy before deleting its source, re-checks
  that the original exists at the moment of each delete, and quarantines a
  file whose original cannot be found under the project's exports rather than
  deleting it. The manifest records whether a run had already trained on the
  originals.

#### Platform and release tooling

- **Linux installs from `scripts/install.py` and starts with
  `scripts/start_local.sh`.** The script existed but had never been run on a
  clean machine: run with the system Python it installed several gigabytes
  into the user site-packages, then spent that download before noticing `npm`
  was missing, and treated a failed step as done; it also installed the SAM
  libraries from the tip of each repository after the lockfile had installed
  the audited commits, let the resolver replace the CUDA torch with the PyPI
  build, installed no ONNX Runtime and no TinySAM, and the start script ran
  uvicorn with `--reload` watching the tree a training run writes into, so
  the server bounced itself mid-run. It now refuses a non-venv interpreter
  and checks for git, node and npm first, installs the lockfile as pinned
  with the SAM libraries at the same commits as the other installers, torch
  from the CUDA 12.8 index, the CUDA ONNX Runtime and TinySAM, and stops at
  the first failing step. Verified on Ubuntu 22.04 with an NVIDIA GPU: all
  five SAM models, GPU training, CUDA inference.

- **Windows gets a restart script for the trainer API.** Settings says a
  change to LAN access takes effect on the next restart, but only start and
  stop scripts existed, and a bare uvicorn restart fails because the LAN
  token lives in the launcher's environment. `restart-windows.bat` stops
  whatever holds port 8002 and starts the API through the same launcher, so
  the token is reused, browser sessions stay signed in and the process
  survives the SSH session ending. It refuses while a run is running or
  queued unless given `--force`, `--dry-run` shows what it would do, and if
  the port does not free, nothing is started.

- **A release gate for data that must not be published.**
  `scripts/ci/check_no_customer_data.py` has two passes. The structural pass
  needs no data -- an id shaped like a project id, an address on a private
  network, a home directory with a name in it -- and runs in CI through
  `tests/test_no_customer_data.py`. The local pass reads every project id,
  project name and distinctive image name from the projects of the machine
  doing the release (`--projects-dir`, else `$SEG_PROJECTS_DIR`, else
  `projects/`) and looks for them in the tree, so the repository never holds
  the list it protects; it fails (exit 2) when it finds no projects, since a
  check against nothing is not a pass. `--root` points either pass at an
  export of the tree. A line may opt out with a trailing
  `# allow-customer-data: <why>`.

- **`make_release_artifacts.py` runs that gate before it writes anything.**
  Both passes read the tree of `--ref` exported with `git archive` -- the
  files the source archive will hold -- rather than the working tree. New
  options: `--projects-dir` (default `$SEG_PROJECTS_DIR`, else `projects/`)
  and `--skip-local-data-check`, which runs the structural pass alone and
  says so. Without either a projects directory or that flag it stops, and it
  refuses to run when the checker is missing.

- **The SBOMs record the commit they were built from.** The SBOM workflow
  attaches a fourth asset, `seg-studio-sbom.commit`. A re-run on a tag whose
  SBOMs came from the same commit leaves the published ones as they are and
  attaches only any that a first run left out; a moved tag, or a release
  whose SBOMs have no recorded commit, fails the job instead of replacing
  what people may have pinned.

### Changed

#### MCP bridge: replies and rules a caller will notice

- **`image_get_b64` hands over a copy to look at, ruled with a grid.** By
  default the reply is now a JPEG with a coordinate grid drawn on it -- the
  numbers along the top and down the left edge are pixel coordinates of the
  lines beside them -- where it used to be the stored file as it is. Pass
  `grid=false` for the picture unmarked. `max_side` scales a large picture
  down and `min_side` enlarges a small one or a crop, `crop_json` cuts part
  of it, and the reply gives the size of the copy (`width`, `height`) and of
  the picture (`full_width`, `full_height`); `accept_mask` and the other
  tools take `from_width` / `from_height` (and `from_box_json` for a crop)
  and put coordinates read off the copy back where they belong. A box that
  cannot be from the copy is refused. The tool takes an image's id or its
  file name, whatever format the image is stored in, and a name that is not
  in the project is answered with ids that are (`not_found`, `unlabelled`)
  rather than a 404.

- **`dataset_images` answers with ids.** It used to return the trainer's
  per-image records; it now returns `total`, `labelled`, `unlabelled`,
  `blank_mask` and `ids: {done, todo}`, where done means painted or marked
  clean, plus `not_listed` when a project is too large to list in one reply
  (the counts stay whole). `annotation_status` counts the same way: a mask
  file with nothing painted in it, not marked clean, is unannotated, and the
  reply says when it has cut its list.

- **Smaller rule changes.** `recipe_apply` takes `item_ids_json` only as a
  JSON array of ids. `classes_set` and `assistant_context_set` refuse text
  that carries the placeholder the bridge shows in place of withheld text. A
  failed call says what the server said was wrong, not only the status and
  the URL. Imported without fastmcp, the bridge raises ImportError; run as a
  script, it exits with the install command.

- **The bridge pools its HTTP connections.** Every request opened its own
  client, one TCP connection per tool call, and parallel sweeps exhausted
  Windows' ephemeral ports (WinError 10048).

#### Annotation

- **SAM takes positive points only.** Every click is a point on the object,
  and clicking inside the preview confirms it; the client always sends label
  1 for each point. `sam-segment` answers 422 for any other label, with a
  sentence that says so, rather than dropping or changing it.

- **Spot detection runs on the server, with one threshold.** The detector
  that ran in the browser -- in two copies that disagreed about where a spot
  ends -- is replaced by `POST .../annotate/{item_id}/spot-detect`, where the
  search and the mask are made by the same function, at the image's own
  resolution. It judges a spot against the surface it sits on rather than the
  whole frame, reads the painted stroke as where an example is rather than as
  the size of the object, and scores its automatic threshold on recall. The
  browser sends the painted pixels and paints the mask that comes back; the
  score map is cached per image on the server, so a slider move is a short
  round trip. The slider is one Threshold scale of whole numbers from 1 to 60
  in every mode, where higher finds fewer; the server rounds the value, holds
  it to that range, refuses a non-number with 422 and returns the range as
  `sensitivity_range`. The separate tolerance slider that one of the two
  detection modes had is gone.

- **A mask is revalidated, not re-sent.** The mask route answers a matching
  `If-None-Match` with 304 and no body, so switching between images does not
  download an unchanged mask again, while a mask the bridge has just written
  is sent whole.

#### Training and results

- **The Training tab's top row is the two tasks, defect detection and count
  inspection.** The row used to mix what to train with how to train it. It
  now offers the two tasks only, each with its own illustration and help
  pop-up, and choosing defect detection selects the standard pipeline. What
  is sent to the server is unchanged, so saved configurations and older
  clients keep working, and the run badges read the same state as before.

- **Only painted images and images marked clean go into a training run.** An
  empty mask used to mean two things -- marked clean or not yet painted --
  and both trained as background, so a model could learn to answer background
  everywhere and still report a high F1. Now paint is foreground, Mark Clean
  is a negative, and an undeclared empty mask or no mask leaves the image
  out; the prepare report counts `clean_declared`, `undeclared_excluded` and
  `unmasked_excluded`. The `include_unmasked` request flag is gone and
  ignored if sent. Re-preparing an existing project changes its split and
  scores, and that is the correction.

- **The command-line trainer checks that the split's images resolve instead
  of looking for `prepared/images`.** `cli_train` refused to start unless
  `prepared/images` existed, which a project that trains from its originals
  no longer has. It now asks whether the train and validation split ids
  resolve to files, and prints where the images were found and which mode
  answered. When some ids do not resolve it warns and continues, because
  those images are dropped from the run and every score afterwards is for the
  smaller set; only when none resolve does it exit.

- **The epoch that ships is chosen with overlapping windows, as inference
  uses.** Best-model selection scored each epoch with the sliding window
  stepping a whole patch, so no windows overlapped and seams went unblended.
  The monitoring pass now steps half a patch, which recovers most of what
  full overlap gains at a fraction of its cost. `F1_val` and `mIoU_val` do
  not move, but `best_F1_val` and `best_mIoU_val` are now measured
  differently and are not comparable with earlier runs; they describe the
  selection pass, not the model.

- **Stride optimisation judges candidates on precision and recall rather than
  F1.** A coarse stride costs false positives -- unblended seams, speckle
  over the threshold -- while recall barely moves, and F1 can average a real
  precision gain away to inside the margin. A finer stride now has to gain
  more than 0.005 precision and give up no more than 0.002 recall, and the
  log prints both per candidate. New runs may keep a finer stride and predict
  more slowly; existing runs are untouched.

- **Count-inspection training sizes its data-loading workers from the machine
  it runs on.** The instance fine-tune ran without loader workers everywhere.
  Two workers help and more give the gain back, so the plan caps at two, and
  further by physical cores, by free memory against the measured per-worker
  cost (higher where processes are spawned, as on Windows and macOS), and by
  batches per epoch. A four-core, 8 GB laptop plans zero, as before.
  `SEG_INSTANCE_NUM_WORKERS` sets an exact count, `SEG_DISABLE_AUTO_PLAN=1`
  turns the plan off, and the run log lists every cap considered.

- **The note before a count run starts says what happens when no tile can
  hold the largest object whole.** The training log used to claim the count
  would read low. It now says counting will fall back to stitching fragments
  across tile seams, which can join two objects that genuinely touch, and
  names the patch size that would show each object whole in one tile, since a
  stitched reading is still weaker than seeing the object in one tile.

- **The score row reads the model that ships.** The all-image score row led
  with the monitoring numbers, which are measured on part of the validation
  set at a coarser stride to rank epochs, and could read well below the
  shipped model. It now reads F1 at the shipped threshold, with the threshold
  on the row, and the monitoring pair moves to the tooltip on the epoch. mIoU
  is the exception and does not follow the threshold, as its tooltip says.
  The card is labelled as measured on the validation set at training time,
  unaffected by post-processing, and gains the PRO and FPR columns described
  under Added.

- **Live Inspection keeps the last two models it ran loaded.** Switching
  between two runs used to unload one model and rebuild the other on every
  switch. The runtime now holds two slots and drops the least recently used,
  so switching back to a model used a moment ago is immediate. Each slot is
  released on its own after the usual idle period, and a slot is never
  dropped while a frame is being inferred through it.

#### Install, dependencies and CI

- **Python 3.11 is the floor, and the installers say so.** The dependency
  lockfile is compiled for 3.11 and pins a package with no 3.10 build, so a
  3.10 interpreter never could install it; it failed deep inside pip with a
  message naming that package rather than the cause. The Windows installer
  still accepted 3.10 at its gate, `scripts/install.py` had no gate at all,
  and the READMEs and `pyproject.toml` advertised 3.10 or later. All now say
  3.11, and both installers stop before installing anything, with the reason.

- **The serving and OpenVINO lockfiles are compiled under the trainer
  lockfile as a constraint.** Both are installed into the same virtual
  environment after the trainer lockfile, and each moved shared pins on the
  way in: the serving lockfile raised pydantic, and the OpenVINO extra bumped
  numpy and scikit-learn and downgraded networkx. They now resolve to the
  trainer's pins, and the drift check recompiles them the same way.

- **CI runs the Serving API tests.** The pytest job installs the serving
  lockfile as well and runs `apps/serving_api/tests`.

- **Dependency version updates no longer open pull requests on the published
  repository.** Each release replaces the published `main` with one commit
  built from the development tree, so a pull request merged there was
  discarded by the next release. Version updates are now raised in the
  development repository and arrive with the next release; the published
  Dependabot configuration keeps its ecosystems at a pull-request limit of
  zero. Alerts and security updates are switched on there.

- **The READMEs describe what a release actually carries.** They told a
  beginner to download a Windows zip and a `SHA256SUMS.txt`, and no release
  has carried either: a release holds the source archive and the SBOMs. Both
  READMEs describe the source download, and say that a package will be listed
  on the Releases page, with its checksums, once one is published.

- **THIRD_PARTY_NOTICES.md has a section for optional packages.** "Optional,
  installed by the user (not bundled)" lists fastmcp 4.0.1 (Apache-2.0) and
  mcp 2.1.1 (MIT), which AI support and the MCP bridge load once a user
  installs them, with the licence checked at the upstream release tag and a
  note on the unpinned packages fastmcp brings with it.

- **The feature catalogue and the About dialog point at Cls-Studio for
  anomaly detection.** Both had said that anomaly detection moves to
  AnomaLens, with publication upcoming. That project was published as
  Cls-Studio, and a reader following the old name would have landed on
  unrelated repositories. The catalogue links the real repository in both
  languages, and the 0.9.7 entry in the About dialog names the same project.

### Deprecated

- **`SEG_PREPARED_IMAGE_FORMAT`.** The variable still works but governs only
  the training copies under `prepared/images/`; the per-project storage
  format decides what every import writes, and startup warns, naming the
  variable, whenever it is set. The handbook said the prepared copies were
  byte for byte identical to the source. That was not true: preparation
  converts the source to RGB and writes a new PNG, so alpha is dropped, and a
  JPEG source was lossy before preparation ran. The handbooks and deployment
  reference now say so and point at the layout doctor.

- **`POST .../datasets/migrate-to-png`** stays as an alias that converts to
  PNG through `convert-images`, described under Added.

### Removed

- **Transfer learning by project similarity, and the Standard / Quick /
  Transfer method buttons.** The three buttons were declared identical -- no
  overrides, no floors, no hooks, nothing branching on their ids -- so they
  named three things that did one thing while the Auto switch beneath them
  made every decision they appeared to make. Transfer was the one that could
  have meant something: a library of per-run feature profiles, a similarity
  search over past projects, and a recommendation with a donor checkpoint.
  All of that shipped on the backend; none of it was reachable, because the
  button set a label and the search had no trigger in the form, and the
  result, when it did run, was printed and never applied. Removed rather than
  finished: the Auto path already picks arch, patch, base channels and epochs
  from the combo predictor, which keeps working -- it never read the library.
  Gone with it: `POST /train/model-search`, the `/train/library-*` endpoints,
  per-run `feature_profile.npz` writes, the `.library` best-run archive, and
  the Recalibrate-ETAs panel. `quick` and `transfer` are still accepted on
  the wire and run as `standard`, so saved configs and older clients keep
  working.

- **The Danger zone in Settings.** Its two actions rebuilt and deleted the
  transfer-learning library, which is gone; the heading and its warning had
  nothing left under them.

- **The magic-wand (auto-select) tool.** A flood fill bounded by colour
  tolerance stops where the colour changes, not where the defect does: it ran
  past the boundary into background of a similar shade as soon as the
  tolerance was wide enough to fill the defect, so the selection needed more
  correcting than painting did. A SAM click at the same point is what the
  wand was for, and the granularity buttons described under Added pick how
  much of the object it takes. The wand's cost-map variant went with it.

- **The auto-label button.** It built a colour histogram from everything
  labelled so far, which finds nothing where the subject and its background
  share a colour, as metal parts on metal fixtures usually do. Draft from a
  trained model (under Added) takes its place; the endpoint stays for callers
  of the API.

- **The superpixel tool, from the palette.** A SAM box reaches the same
  shapes in fewer clicks. The tool is gone from the toolbar, its tutorial
  step with it, and P is free again.

- **SAM exclude-click.** The opt-in "exclude-click (right click)" setting and
  right-click for the SAM and SAM Box tools are gone, as part of making SAM
  positive-only (under Changed).

### Fixed

- **Count training and count prediction work with non-ASCII image names on Windows.**
  Image files in the count pipeline were written and read with OpenCV's
  path-based calls, which mangle or miss a name that is not ASCII on
  Windows; training on the bundled cell-count sample (Japanese file names)
  stopped at threshold calibration. They now go through segcore.image_io.

#### Annotation, masks and the image list

- **One place decides where a mask goes.** A mask belongs either in a PNG or,
  for a very large image, in a chunked array, and everything that reads one
  prefers the array -- so the drafts, recipes and agent writes that painted a
  PNG beside an array were writing files nothing read, while the image list
  described them. Writers go through a single place now, which writes where
  the mask actually lives; mask-stats, clear-class and spot-detect's example
  read from there too, and the paths that cannot sensibly draft over a tiled
  mask say so instead of pretending.

- **The image list cannot drift from the masks again.** What each image is
  annotated with used to be whatever the last writer left in the index, so
  one writer that forgot to update it left rows marked for paint that was
  gone. Each entry now carries the mask's size and modification time beside
  the ids read from it: a mask that changed since is read again, whoever
  changed it and whether or not it went through the API, and one that did not
  costs a stat rather than a decode. Existing projects are repaired as they
  are opened, a bounded number of images per load so a large one never
  stalls.

- **A tiled mask says what is painted in it.** A mask stored in chunks for a
  very large image is never read to build the image list -- it can be a
  gigabyte -- so one that existed was simply reported as having foreground,
  with no classes named. It now keeps a tally beside the array, updated a
  tile at a time as you paint, so the list shows the right classes and an
  erased stroke stops being counted.

- **Deleting a class clears the image list's marks for good.** The purge
  erased the pixels but left each index entry still naming the class, so
  adding a class back brought every mark with it, and the green bar on a row
  stayed while the dots went. The entries are recomputed from the masks, and
  only ids the class list still holds count.

- **A class purge or merge keeps a tiled image's clean mark, review flag and
  author.** Purge and merge refreshed every index entry against its PNG; a
  tiled image has none, so each one read as a deleted mask and lost those
  three. The refresh reads the tally for a tiled image, and a PNG export
  beside an array is no longer counted as a second mask. Merge and purge
  accept only class ids 1 to 254 (400 otherwise), so the ignore value 255 can
  no longer be merged into a class, and they answer 404 for an unknown
  project instead of creating its directory; review and mask-stats do the
  same.

- **A file that is not a mask is not an annotation.** A half-written PNG used
  to be recorded as an image annotated with nothing, which is a confirmed
  negative to training. It is reported as unannotated until it can be read.

- **A deleted image leaves nothing behind for the next one.** Deleting
  removed the PNG but could leave the tiled array or its tally, and the next
  image given that name inherited them, arriving already annotated. Delete,
  bulk delete and the startup sweep now clear every form a mask takes.

- **Marking clean skips what it could not write.** An image whose size the
  index did not know had its row set to clean while its paint stayed on disk.

- **A mask can only be saved for an image that exists.** The route wrote the
  file before looking the image up, leaving a mask no entry described.

- **Two files that share a name no longer share a row.** `scan_01.png` and
  `scan_01.jpg` took one id between them, and painting either marked both.

- **Cleared or deleted annotations no longer come back.** Server-side bulk
  actions -- clearing a class from selected images, marking images OK,
  deleting a class or images, applying an auto-label recipe -- now redraw the
  canvas immediately and let in-flight saves settle first. Previously an
  autosave scheduled just before the action could land after it and silently
  write the old mask back, resurrecting the cleared markings; the canvas also
  kept showing them until the next image switch. Saves of one image are now
  strictly ordered, and a mask download that started before the action can no
  longer re-insert pre-clear pixels into the client cache.

- **Four more ways the annotator could silently erase or bring back pixels
  are closed.** A tool shortcut mid-move switched tools before the lifted
  region was put back; shortcuts now wait for the move. A slow spot or crack
  result could land on an image opened since; it is discarded. A class the
  server failed to purge was removed locally and undo could revive it; only a
  class the server does not know is removed, and undo history is scrubbed. A
  redraw still in flight could roll a fresh stroke back; such responses are
  discarded.

- **A class list can no longer be written into another project during a
  project switch.** Four of the five places that send the class list paired
  the project open now with whatever class draft was in memory, which match
  only once a switch has finished loading, exactly when they fire. The server
  accepts a superset of the ids a project has, so a foreign list was
  accepted: it renamed the class the masks were painted with and added the
  other project's classes beside it, and putting it back was refused, since
  dropping the added ids counts as removal. The draft now carries the id of
  the project it was loaded from, and a write for any other project is
  dropped.

- **Switching projects blanks the canvas.** The previous project's picture
  and mask stayed up until the first image of the new one had decoded.

- **A crack seed that is not on a ridge is refused.** Clicking a smooth
  surface with the crack tool could select most of the frame -- the
  thresholds came from the seed alone. The seed is now tested against the
  frame's own distribution, and one that is not a ridge is refused with a
  sentence saying what the tool is for.

- **Adjacent selected rows in the image list no longer show a double
  border.** Each shared edge is now drawn once, so every boundary is a single
  line and the active row keeps its complete ring. The status bar on a row's
  left edge is painted above the ring, and a selected annotated row now shows
  the selection ring at all, which a style-ordering clash had been dropping.

- **Uploads are stored by what they are, not what they were called.** Every
  route that adds pixels -- upload, ZIP import, video frames, resize-clone
  and synthetic generation -- now goes through one encoder, and the stored
  suffix follows the bytes rather than the filename: a JPEG uploaded under a
  `.png` name used to be stored verbatim. A resized clone inherits its
  source's format instead of the global default, a file that cannot be
  decoded is skipped with a line in the log instead of becoming an empty
  item, a `.tif` inside a ZIP is no longer dropped, and two uploads differing
  only by extension no longer claim the same item and mask.

- **A project exported and imported again comes back as it left.** Import
  took every picture outside a `masks/` folder as an image, at any depth, and
  an export carries its training runs under `training/runs/`, whose
  reliability charts are PNGs: a project came back with its runs' charts
  among its photographs. Images are now read only from an `images/` folder at
  the top of the archive or inside its one top folder, or from the pictures
  lying directly there; an archive built around a deeper folder such as
  `datasets/prepared/images/` is refused with the name of a picture it
  passed over. The export's `metadata.json` now also records, per image,
  whether it was marked OK, flagged for review, placed in train or test by
  hand, drafted by a run or saved by an agent, and whether its mask was only
  the export's blank; Import puts those back. An OK image used to return as
  an empty mask nobody had declared, which counts as unlabelled and is left
  out of training. The classes come from `metadata.json` rather than from
  whichever run's copy was read last. Exports from earlier versions import
  as before, and so does an export extracted and zipped again, which
  Windows' Extract All leaves inside a second folder of the same name. A
  large image whose mask is stored in tiles was exported with the all-zero
  placeholder or an out-of-date PNG in place of its mask, and now carries
  the mask itself.

- **Every part of the application agrees on which files are images.** Eight
  places kept their own list of image extensions: mining and per-image
  evaluation skipped `.tif`, INT8 calibration and automatic configuration
  skipped `.webp`, and the project list counted four types. A WebP or TIFF
  project showed zero images, fed the combo predictor zeros, and could leave
  the Results tab blank. One list -- `.png`, `.jpg`, `.jpeg`, `.webp`,
  `.bmp`, `.tif`, `.tiff` -- is used everywhere, so such a project's
  recommended combination can change, and INT8 export and automatic
  configuration no longer stand down because `prepared/images` is absent.

- **Tabs you are not looking at stop drawing.** A results tab left open
  repainted its whole mask every frame, and the Annotate blink timer kept
  firing behind another tab; both now wait until their tab is shown.

- **An upload no longer crashes its tile task when pyvips is installed
  without libvips.** That import fails with an OSError from cffi, not an
  ImportError, so it escaped the guard and every image upload logged a long
  "Unhandled exception" traceback from the DeepZoom background task. One
  warning with the reason now, tiles skipped, uploads quiet.

#### MCP bridge

- **The MCP bridge starts again with a current fastmcp.** Its startup banner
  counted tools through FastMCP's private registry, which moved in 4.x, so an
  unpinned install raised `AttributeError` before serving anything. The count
  is read from the tools' own docstrings now.

- **`server_version` and `startup_status` answer instead of returning 404.**
  Both routes are mounted at the server root, outside the `/api/v1` prefix,
  but both tools sent them through the versioned helper. A test now builds
  the app's own route table and asserts every literal path in the bridge
  resolves against it.

- **`export_onnx` percent-encodes the run id in its query.**

#### Training and inference

- **Turning automatic configuration off now actually turns it off.** The
  training screen's automatic-configuration switch sends a single mode value,
  but the request also carried the older, separate on/off flag at its default
  of "on", and that older flag took priority whenever it was present. Since
  it was always present, the switch never had any effect: automatic
  configuration ran on every training run, and could replace the
  architecture, the number of base channels and the patch size you had chosen
  -- most visibly by training an STDC model when SimpleUNet was selected. The
  older flag is now absent unless a caller sets it deliberately, so the mode
  value decides. Callers that still send the older flag keep their existing
  behaviour, and runs left on automatic are unchanged.

- **A model trained with SE attention off now exports with SE attention
  off.** `--no-se` is a CLI training flag; the checkpoint it produces has no
  SE weights. Every place that rebuilt the model around a checkpoint -- the
  ONNX export behind prediction, the PyTorch fallback, the Core ML export,
  and the FP16 speed-optimise copy -- built it with SE blocks enabled
  regardless and loaded the weights non-strictly, so the missing SE layers
  kept their random initial values and the model that shipped was not the
  model that trained. The speed-optimise copy even saved those random weights
  back to disk. Training through the application was never affected: the API
  always trains with SE on. The rebuild now reads the flag from the
  checkpoint itself, and a checkpoint must match the rebuilt model key for
  key -- any mismatch is a clear error at export time instead of a silently
  different model.

- **A prepare that finds no masks no longer switches every class off for
  good.** Preparing a project whose masks were missing found no painted
  pixels for any class, wrote every class as inactive and never wrote it
  back: the next run trained on background alone and reported loss, mIoU and
  F1 of exactly 0.0000 with no error anywhere, and fixing the data did not
  help, because nothing re-enabled the classes. A prepare that sees no masks
  at all is now treated as a broken prepare, not as evidence about classes,
  and changes nothing. A class this pass switches off is marked, and the next
  prepare that finds masks for it switches it back on; a class switched off
  by hand carries no mark and stays off.

- **A prepared dataset is reused only for the orders it was built to.**
  Training skips preparation when the prepared directory is newer than the
  annotation index, but a timestamp cannot see what the cache was built to,
  so changing the validation or test ratio, the folds, the split method or
  pseudo-label settings reused the old dataset as a no-op. The prepare report
  now records those parameters under `prep_params`, the training-start gate
  compares them before trusting the cache, and the log names the difference.
  A report from before the field existed is compared on the two ratios it
  carries, and a gate error is reported rather than re-preparing silently.

- **Batch sizing measures the tensor the model actually receives.** Patch
  training resizes every crop to the model input size before the forward
  pass, but the VRAM estimate and dry run profiled the patch size instead.
  The two are equal in the default configuration, which is why this never
  showed; a run that sets them apart was sized for the wrong shape -- patch
  128 into input 512 is sixteen times the activation memory of what was
  measured.

- **The command-line trainer verifies the batch it raised.** The CLI raises
  the batch size to fill the GPU from an estimate, and the estimate can be
  optimistic enough to run out of memory in the first epoch. The dry run the
  app performs -- a real forward and backward, halved until it fits -- now
  runs whenever the batch was raised (`--auto-batch` still forces it for a
  hand-picked batch). The flag itself had been calling the profiler with a
  missing argument and failed on every invocation.

- **Epochs no longer slow down over the course of a run on Windows.**
  Sampling a training patch reflect-padded the whole image and mask once per
  sample -- tens of megabytes of allocation and copy on a large image,
  repeated for every patch of every image in every epoch. On Windows the
  churn showed as a flood of page faults, epoch time grew steadily across a
  run, and training could stall inside the padding call. The crop window is
  now folded through the same reflection by index arithmetic, touching only
  the pixels of the patch; sampling is bit-identical to before on a set of
  reference patches, and a sample costs a fraction of a millisecond.

- **The DataLoader planner no longer remembers a dataset profile built from
  nothing.** The planner decides cache mode, worker count and prefetch depth
  from a sample of 32 images. A sample in which no image resolved reported
  zero bytes per image, was cached under a fingerprint that matched for 24
  hours, and sized every run that day. A profile with no bytes and no pixels
  behind it is now returned but not cached, with a warning in the log. The
  fingerprint also keyed on the image directory's modification time, which
  changes with every saved annotation once a project trains from its
  originals; it keys on the dataset descriptor and the split files instead.

- **The DataLoader planner survives a Japanese Windows console.** Its sizing
  report contained a character (U+2248) that the cp932 console encoding
  cannot represent, so on Japanese Windows every training run crashed the
  planner mid-report and silently fell back to legacy sizing. The report now
  uses a plain tilde.

- **Deleting a run or a project while it trains stops the training process.**
  Deleting a run removed its directory before the monitor could write the
  stop sentinel there, dropping the only reference to the training process.
  It kept training with the device reported free and the next queued run
  started; deleting a project did the same. One supervisor now returns only
  when the child is dead: the stop sentinel is placed before the directory
  goes, so a run between polls stops cleanly and keeps its checkpoint, and
  one that misses the window is terminated. `model.pt` is written via a
  temporary file, so a stop mid-save cannot truncate it.

- **Training runs start on Linux.** The training worker is a separate process
  written for the spawn start method, which Windows uses by default; Linux
  defaults to fork, and by the time a run is launched the API has already
  initialised CUDA in its startup health check, so the forked child died with
  "Cannot re-initialize CUDA in forked subprocess" and no run could start at
  all. The API now asks for spawn when it loads, and a host that had already
  chosen spawn is left as it is. DataLoader workers on Linux become spawned
  processes too.

- **Count training reads images and masks through the Unicode-safe reader.**
  The plain OpenCV call fails silently on non-ASCII paths on Windows, so a
  project with Japanese filenames lost those images from the training set
  without a message.

- **Small projects no longer compose every synthetic image on a single
  background.** Background plates for the synthetic dataset were taken from
  every eighth source image, so a project with four to seven annotated
  sources yielded no plate at all and the fallback took one, from the first
  usable source. Every composite then shared a single inpainted background
  while the other sources sat unused. The fallback now takes every usable
  source, up to the usual plate count.

- **A real photo larger than one tile no longer trains the detector at the
  wrong scale.** Fine-tuning resized every training image to the model's own
  resolution while inference tiles at patch size, so a large photo trained
  squashed, each object a fraction of its inference size, and upsampling its
  predicted masks back roughly doubled peak VRAM. A real photo that fits the
  patch stays in the training set; a larger one moves intact to
  `_real_annotations.coco.json`, which count calibration evaluates through
  the tiled path, as inference does. Stats report `n_train_oversized` and
  `n_val_oversized`.

- **Objects longer than the tile overlap are counted instead of being dropped
  at the seams.** Tiled counting dropped any detection a tile edge cut,
  trusting a neighbouring tile to see it whole, which needs an overlap wider
  than the object: an object nearly as long as the patch was cut in every
  view and vanished. When the largest annotated object exceeds the overlap,
  fragments are now joined by the pixels neighbouring tiles share, so an
  object across seams counts once and two abreast stay two. The serving
  container's /count does the same, and older runs and bundles are unchanged.

- **The counting overlay and the count chips follow the confidence slider.**
  Raising the confidence on a count run changed the number beside the picture
  and left the picture alone: the overlay was rendered once at inference and
  the chips read totals baked in with it. `overlay.png` now takes a `score`
  from 0 to 1 and leaves out instances below it; above every confidence the
  source image shows untouched. The chips count at the same threshold, and a
  class the run detected stays on screen at 0 rather than vanishing. A score
  outside 0 to 1 is rejected.

- **The size filter reaches the picture and the numbers.** Min and Max area
  changed nothing until an apply button was pressed; they now apply to the
  image on screen, through the server's own post-processing, as soon as they
  are set.

- **A run shows the same F1 wherever it appears.** The metrics file carries
  three F1s: per-epoch monitoring, the report pass at plain argmax, and the
  shipped weights at the shipped threshold. Only the last describes the
  model, yet the run list and the results screen could show different numbers
  for the same run, and the project summary took a third route. One function
  now answers, preferring the shipped-threshold score and falling back only
  for older runs, and the run list, new-model notification and project
  summary all ask it. A run with no F1 reports none rather than 0.0, and the
  listed mIoU can read lower, now measured on the whole validation set.

- **Per-image results are measured at the stride the run ships.** After
  training, the step between inference windows is tuned on the validation
  set, and the winner is what the prediction engine uses from then on. The
  per-image results written at the end of the run -- and the pass that picks
  hard images and decides whether an iterative chain continues -- ignored
  that winner and stepped at the default stride instead.

- **Served models infer at the stride they were tuned for.** The serving
  container received the winner inside the exported `train_config.json` and
  ignored it, stepping at the default stride while applying the inference
  threshold that was calibrated at the tuned one. The `/segment` endpoint now
  reads the shipped stride, with the same guard and output-stride alignment
  the trainer applies, and reports it in `meta.json`; `/count` is unchanged,
  since instance tiling has no tuned stride to read.

- **Sliding-window inference stops asking the driver how much memory is free,
  once per image.** Sizing the tile batch shelled out to `nvidia-smi` on the
  per-image path, which cost tens of milliseconds every time and returned the
  same answer; it weighs most on small images predicted in bulk. The probe is
  cached per (patch size, class count). The caller already catches an
  out-of-memory from the forward, halves the batch and retries the chunk,
  which is what makes a cached reading safe to rely on.

- **A prediction started while a training run holds the card no longer
  freezes the machine.** Training took the GPU lock before using the card;
  the ONNX Runtime inference path never did, so a prediction fired during a
  run landed on top of the trainer, and on Windows a saturated card takes the
  desktop compositor with it. Prediction now claims the same device lock. If
  the card is held, the prediction runs on the CPU instead -- slower, but it
  cannot push the trainer out of memory, and the results tab stays usable
  during a long run. A run that finds the card held by a prediction is
  reserved and starts when it frees.

- **ONNX inference on Linux runs on the CUDA provider.** The DLL preload that
  precedes a CUDA session is a Windows concern; on Linux it tripped over a
  namespace package and every batch prediction fell back to PyTorch.

- **The embedding-stratified split can load DINOv2 again.** It imported a
  module path that exists on no installation and fell back to the hash split
  on every project, quietly.

- **Speed-optimising a run no longer fails after writing its files.** The
  FP16 copy was registered in the database under the id factory itself rather
  than the id it had minted, so the commit raised after `model.pt`,
  `model.onnx` and the name had been written: every attempt answered 500 and
  left an orphan run directory.

- **A SAM model whose package is not installed answers 501 (NSS-5010) with an
  install hint** instead of 500 "SAM inference failed", so a caller can tell
  "pick another model" from a crash.

#### API, install and packaging

- **The OpenAPI document describes the whole API, not the part registered
  when it was first read.** Most routers are registered from the background
  startup, and FastAPI builds the schema once, so anything that asked for
  `/openapi.json` during that window pinned a truncated document while every
  route it left out answered normally. The startup gate now answers the
  schema URL with the same 503 it gives the rest of the API until startup is
  complete, and the cached document is discarded just before the gate opens.

- **Class changes, model exports and model activations are recorded in the
  audit log.** Six routes opened a database session purely to write their
  audit entry and never committed it, so the row was discarded with the
  session and those actions never appeared in the log. Those entries are now
  written in a transaction of their own; routes that already batched the
  entry with its change are unchanged.

- **`POST .../recipes/apply` checks its body.** It answers 400 when the body
  is JSON but not an object, or when `item_ids` is not a list; a bare string
  used to match ids by substring.

- **The installer's launcher honours "Allow access from LAN".** It bound
  loopback regardless of the setting, and had it bound the LAN it would have
  hit the app's refusal to serve a non-loopback address without a token. It
  now resolves the host as the scripts do and mints the token on the first
  LAN start.

- **The offline pack replaces the torchmetrics module that is not licensed
  for commercial use.** The installer's staged tree and
  `install.py --offline-pack` both replace the Extended Edit Distance module
  through `scripts/_nc_stubs.py`, and the pack's wheels themselves are
  rewritten, RECORD included.

- **The Lovász-Softmax MIT licence text ships.** It is in
  `licenses/third_party/`, and the attribution sits at the function as well
  as in THIRD_PARTY_NOTICES.md.

- **The UI container builds.** Its Dockerfile flattened the app while the
  Vite config copies `../../THIRD_PARTY_NOTICES.md`; it keeps the
  repository's depth now, as the installer's staging already did.

- **The test suite leaves the checkout as it found it.** The test client runs
  the app's startup, which installs missing packages and builds the UI;
  `SEG_STARTUP_NO_INSTALL=1` now makes startup skip the pip and npm
  self-install and the UI build, and the trainer tests set it.

### Security

- **The example chat page answers only its own page.**
  `scripts/examples/qwen_mcp_chat.py` has no login, and before this change
  any web page open in the same browser could make it send a named
  environment variable, as a bearer token, to a URL of that page's choosing,
  save that URL for later runs' images, and start a write run. It now listens
  on loopback only (`--host` must be `127.0.0.1`, `localhost` or `::1`; reach
  it from another machine through an SSH tunnel), refuses a request whose
  `Host` is not loopback, and refuses a POST that is not JSON or comes from
  another origin. The model server's address (`--base-url`) and the
  environment variable holding its key (`--api-key-env`) come from the
  command line only: a request that names either is refused with 400, the
  page can only switch between the known servers and pick a model, and an
  address saved by an earlier version is ignored.

- **The MCP bridge checks the ids it puts into a URL.** An id with a slash, a
  backslash or a question mark, or one that is `.` or `..`, is refused before
  any request is made, and every path segment is percent-encoded, so an id
  cannot reach another route and an image named with `#` or `%` is still
  reached by its name.

- **Deleting a recipe checks the id before touching the disk.** The route
  joined the raw id onto the recipes directory; on Windows an id containing
  backslashes resolved outside it. Recipe ids are validated as slugs first,
  and the image-lookup fallback and the draft-from-run adopt path refuse an
  id that is not a bare name.

- **Error responses no longer carry the server's file paths.** The three
  "checkpoint incompatible with the current architecture" responses quoted
  the checkpoint's absolute path, the INT8-calibration error named the
  prepared-images directory, and a failed spot-detect read named the file.
  The path now stays in the server log.

- **The Serving API caps `/count` and `/segment` uploads at 50 MB.** It has
  no authentication, and both routes read the whole body before decoding it.

- **The nanoid advisory (GHSA-2v37-7h3g-55p8) is out of the build-time
  tree.** It sat under vite's CSS tooling, a path that never ships, but the
  release gate runs `npm audit` and read it as high. The pin moved; the built
  UI is unchanged.

- **Dependency advisories against the 0.9.9 pins are closed.** anyio moves to
  4.14.2 (GHSA-82r6-8w77-94w6, GHSA-5p39-cfhj-2xmp), transformers to 5.10.4
  (GHSA-xrqw-3rrv-vx5w), hydra-core to 1.3.7 (GHSA-2cp2-2r3c-7p7r), weasyprint
  to 70.0 (GHSA-jf6q-chmf-3h3v) and pytorch-lightning to 2.6.6
  (GHSA-qqmf-gpg7-g8gw), and js-yaml in the UI's build-time tree to 4.3.2
  (GHSA-2883-xcg3-v3hh). anyio and hydra-core are not packages this project
  picks, so their floors go in the security-floors section of
  `requirements.in`. No other pin in the four lockfiles moved, and
  `package.json` is unchanged.

- **The accelerate advisory (GHSA-4j2p-28q2-5m79) is accepted until a fix
  ships.** accelerate has no fixed release. The flaw is in two of its
  checkpoint loaders, which trust the file names listed in a sharded
  checkpoint's index. accelerate arrives only through peft, and neither peft,
  transformers, rfdetr nor Seg-Studio calls those loaders, so the CVE scan in
  CI ignores this one id, with a note to drop the ignore when accelerate ships
  a fix.

## [0.9.8.post2] - 2026-08-07

*First published 2026-07-30 and withdrawn; re-released 2026-08-07. The
entries down to the note further below are new in the re-release.*

### Security

- **aiohttp is raised to 3.14.3 (PYSEC-2026-3545).** It is not a dependency this
  project chooses -- it arrives through `fsspec` -- so the lockfile carried
  whatever the resolver had settled on, and an advisory published against that
  version made the CVE scan a release blocker. `requirements.in` now states a
  floor for it, in a section for exactly that: versions raised on packages we do
  not otherwise pick. Recompiling moved one package and added none.

- **Three advisories in the build-time dependencies are closed.** `js-yaml`
  (high), `brace-expansion` (high) and `postcss` (moderate), with `nanoid`
  carried along by the same resolution. All four are patch bumps of packages the
  lockfile already had; none of them appears in the production dependency tree,
  so nothing that ships in `dist/` changed and neither did `package.json`. The
  exposure was to whoever builds the UI, not to anyone running it.

### Changed

- **Stride optimisation no longer buys a small score with a large bill.** After
  training, a run scores several sliding-window strides on its own validation
  images and keeps the winner, and inference runs at that stride from then on.
  The winner was whichever scored highest, with nothing said about what it cost.
  Patches scale with the inverse square of the stride, so a finer stride that
  scored a hair better could cost nine times the patches -- paid on every
  prediction for the life of that model. Starting from a 128-pixel patch, which
  auto-config does choose, the candidate list reaches sixteen times.

  A finer stride now has to beat the coarser one by more than 0.005 F1. The case
  stride optimisation was added for is untouched: where the foreground is
  sparse, a finer stride wins by far more than that margin. **A run trained
  from now on may keep a coarser stride than the same run would have kept
  before, and predicts faster for it.** The log names the
  stride that scored highest, what it scored, and what taking it would have
  cost.

### Added

- **A run now records when it actually started.** `created_at` is when the run
  row was written, which for a run whose GPU was busy is when it went into the
  queue -- often minutes before training began, sometimes hours. Everything
  that wanted a start time was reading a reservation, and the summary card said
  "Started" while showing one.

  `started_at` is stamped at the moment training begins, whether that is
  immediately or when a card frees up later. It is nullable and deliberately
  not backfilled: for a run written before the column existed, and for one still
  waiting in the queue, there is no honest value, and copying `created_at` in
  would state a start that never happened. Where it is missing the display falls
  back to the creation time and labels it as the creation time.

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
  whole thing off. On release a worker gives back nearly all the memory its
  session held, and the stream session most of what it had taken.

  A worker releases from inside its own loop rather than from the poller,
  because that thread is the only one that ever runs its session. Dropping it
  from outside would free nothing while the batch below still held the
  reference, and would leave the next batch to build a second session beside
  the first -- which on a 4 GB card is the failure the release exists to
  prevent. The stream session is deliberately used outside its lock, for a
  whole sliding-window pass over one frame, so it counts the passes in flight
  for the same reason.

- **The batch size a GPU worker profiles is remembered across a release.** The
  search runs real inferences until they fail, which on a large card takes tens
  of seconds. Paying it once per process was reasonable while a session was
  loaded once per process; paying it after every quiet spell would have put
  that wait in front of the first prediction. Recalling the number promises
  nothing that was not already true -- a sub-batch that fails to allocate is
  still halved and retried, and the smaller number replaces the remembered one,
  so a card that is tighter than it was corrects itself on the first batch. The
  first batch after a release now starts without searching again.

*The entries below are from the original 2026-07-30 publication.*

### Changed

- **Inference now uses the sliding-window stride each run measured for itself.**
  Post-training stride optimisation scores several strides on the run's own
  validation images and stores the winner, and the threshold that ships with the
  model is calibrated at that same geometry -- but inference read none of it and
  always stepped by three quarters of the patch. Where the foreground is sparse
  the two are not equivalent: the stored stride can score clearly higher, and
  predicting through the application at it raises precision and cuts
  false-positive pixels and detections that sit on no real object, at
  unchanged recall.

  **This costs time.** A finer stride is several times as many patches, and on
  a small GPU each image takes several times as long. Runs whose optimisation
  kept the three-quarter stride are unaffected, as
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
  the process: on a 4 GB card nearly all of the memory stayed held long after
  the GPU had gone idle. The pre-training release, further down this same list,
  reclaims that memory when the next training run starts -- but nothing
  reclaimed it if no
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
  and the two ids alone spent 72 of the 260 characters Windows allows. Deep
  paths came within a few characters of the limit, and the worst case a long
  image filename could produce was already past it -- which
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
  available: a long run's training sat next to a `checkpoint_best_regular.pth`
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
  single-digit epochs. It now honours `early_stopping_patience`, the key
  the semantic path already reads, and stops after that many epochs without a
  meaningful improvement in segmentation mAP. Set it to 0 to run every epoch.

- Instance training evaluates every 5 epochs instead of every epoch, the
  cadence semantic runs already use. RF-DETR ran a full COCO evaluation over
  the whole validation split each epoch, and ran it twice — a second forward
  pass through the EMA weights. The best checkpoint is now chosen among
  evaluated epochs, and the final epoch always evaluates.

  Together the two changes make the first epochs several times faster on a 4 GB
  GPU. The input size accounts for most of it: the backbone is a ViT, so 1.72x
  the pixels costs rather more than 1.72x the time. Epochs get cheaper later in
  a long run under either configuration, so this describes the first few epochs
  rather than a whole-run average.

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

- **Every ONNX inference ran on the CPU provider, far slower than on the GPU.**
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
  inference read it, and the same checkpoint and data now score normally.
- **Instance-mode training ran many times slower than it needed to on GPUs
  older than Ampere.** The fp16 autocast rule that sliding-window inference was
  missing had a third gap: rfdetr defaults to `amp=True` and resolves
  `amp_dtype="auto"` to fp16 below Ampere, and none of the four places that
  build the detector -- training, threshold calibration, prediction and ONNX
  export -- said otherwise. All four now read the same policy. On such a GPU an
  epoch now runs an order of magnitude faster, turning a multi-day run into a
  few hours, and the first epoch's quality is comparable. Ampere and newer keep
  autocast on and are unaffected.
- **The VRAM fit reported a reduced batch as though it had fitted.** The batch
  fitter halves the batch until it fits the card's budget and stops at 2, which
  is the smallest configuration rfdetr handles well rather than one that
  necessarily fits the budget. On a 4 GiB Windows card the budget works out at
  1.7 GiB and the smallest model on offer is tabled at 3.5, and the log said
  only that the batch had been reduced. The dry-run now records the shortfall
  as well. Note that the budget is a deliberately pessimistic policy -- a flat
  2 GiB WDDM headroom off a 4 GiB card -- so being over it does not mean the
  run will fail.
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
No instance labels are drawn by hand.

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

Nano was offered during development and retired before release: it
counted less exactly than the larger sizes. Existing Nano
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
