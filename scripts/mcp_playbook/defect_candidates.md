# Finding defect candidates

For a project where each image shows a part, and the annotation marks the
occasional flaw on it: a chip on a machined surface, a pit, a deposit.

## Step 1: if a trained run exists, use it and stop

`train_runs_list` → a run with `status: completed` and `has_model: true` →
`prelabel_run` drafts every unannotated image, copying aside anything it
replaces. On held-out images a trained model can find each defect closely
and paint nothing on the clean images. Scored against the hand masks of its
own project (images the run may have trained on, so this is closeness of the
draft, not accuracy on new parts), a run's existing predictions can fit less
closely than that and still leave the clean images empty. Nothing below
matches that.

**Measure the run before you let it draft.** `run_agreement` scores its
predictions against every hand mask in the project and ends with a verdict.
Every labelled image, not the first few: a run can score well on the first
few and far worse on all of them, many at zero, drawing two regions
where the human drew one. A draft from such a run costs the person
more than an empty image. Below roughly 0.5, do not `prelabel_run`; say the
run is not good enough and quote the numbers. `predict_operating_points`
gives the recall-first threshold to use when it is.

`predict_status` first: a run can exist with no prediction on disk for any
image, and both the measurement and `prelabel_run` need them.
`predict_batch` fills them in -- and needs the item ids; it does not take
"all". `has_model` on a run means its torch checkpoint; the prediction
paths want ONNX, and a run trained here may never have been exported. A
draft adopts whatever `predict_batch` left on disk without a model, so the
order is `predict_batch` (every id, ONNX) → `run_agreement` →
`prelabel_run`. If `predict_batch` itself answers "model checkpoint not
found", `export_onnx` the run first.

End to end, on a counting project of touching objects whose run scored well
in `run_agreement`, `prelabel_run` wrote every unlabelled image in under a
second from the predictions on disk, and the drafts sat in the range of the
hand masks. One difference to know about: a draft writes 0 for background where
the person's masks in a counting project carry 255 (ignore). A 0 asserts
"nothing here" to training; a 255 says "not labelled". If the project's
own masks use 255, say so when you hand the drafts over, or convert the
draft's background before training on it.

**After drafting, read `mask_stats` before you report.** A draft is
indexed as a draft (`draft: true`, with the run that made it) until a person
saves over it, and `mask_stats` puts the drafts' area and region count
against the range of the hand masks and lists the ones outside it, with why.
A run can score well on its hand masks and then draft images far outside
the hand masks' range on one side or the other -- when the unlabelled
images are a different kind of scene. Hand those to the person first, by
name; do not call the
project labelled.

And a project can carry labelled files with nothing in them: "labelled"
masks that are all background and ignore, where `run_agreement` answers
"no defect image to score". There is nothing to draft from there; the run
was trained on empty targets, and the person has to label first. The rest of this playbook is for a project with **no**
run yet, to produce the first few annotations a run can be trained from.

Template matching (the counting playbook) does not transfer here: a defect is
a one-off shape, and the cut-out of one chip does not match the next:
measured, it matched next to nothing.

## Step 2: zoom, or the model cannot see it

A defect under 1% of a frame of about a megapixel is invisible to a vision
model at native size: none found, with or without a marked example. Cut the
frame into tiles
sized so the defect fills a quarter to a half of the tile, and upscale each
tile to ~1024 px. From the teacher masks: tile side ≈ 1.5 × the largest defect
bounding-box side, floor 200 px. With that the same model found the defect on
every image tested.

## Step 3: give it normal, not the defect

Zoom alone is not enough: the model flagged tiles whether or not a defect was
present, on a defect image and a clean one alike. It had no idea what this
surface looks like when nothing is wrong.

Build a **reference sheet**: four crops of the surface at the same tile size
and zoom, taken from a frame the human left unannotated, or from annotated
frames well away from the annotation. Send it as the first image and the tile
as the second, with the prompt in `segstudio://playbook/prompts` under
"defect with a normal reference". Ask for a *difference from normal*, not for
a defect.

Compared on the same tiles at the same zoom, on defect frames and clean ones:

| model build | defects | false flags on a defect frame | on a clean frame |
|---|---|---|---|
| "find a defect", no reference (MLX 27B) | found | many | many |
| normal sheet, MLX 27B on a laptop | all found | a few | a few |
| normal sheet, GGUF 27B on an RTX 3090, `num_ctx` 8192 | one missed | none | none |

Same prompt, same tiles: one build finds everything and over-flags, the other
never flags a clean frame and misses a defect. A handful of frames is not
enough to call either the rule; it is enough to say the answer depends on the
build, so measure yours on a clean frame and a defect frame before trusting
it. Either
way the flagged points are **candidates for a person or a trained model**,
never annotations to write.

## Step 4: from a candidate to a mask

`sam_segment` at the candidate point, filter the three levels by a band built
from the teacher defects exactly as in the counting playbook (range ± slack;
texture floor). A candidate that produces no acceptable mask is dropped.
Present the survivors with `instance_preview`-style crops or an overlay for a
human to accept; write with `mask_put` only what they accept, with
`class_id` set -- SAM's masks are 0 and 255, and 255 is ignore.

## Cost

About forty tiles per frame of a megapixel at 200 px. A 27B model on an RTX
3090 with
`num_ctx 8192`: about 2 minutes per frame, warm. On a laptop MLX build about
12. A coarse pass over larger tiles to pick regions, then the fine pass only
there, cuts most of that; it is not implemented in the bridge.
