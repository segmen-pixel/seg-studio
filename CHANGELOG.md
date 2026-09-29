# Changelog

All notable changes to this project will be documented in this file.

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

- **The torchmetrics module that is not licensed for commercial use is
  replaced in the offline pack too.** The installer swapped the Extended Edit
  Distance module for stubs in its staged tree, and nothing else did, so
  `install.py --offline-pack` saved the wheels as pip left them. The
  replacement now lives in `scripts/_nc_stubs.py`, called by both, and the
  wheels themselves are rewritten, RECORD included.

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
