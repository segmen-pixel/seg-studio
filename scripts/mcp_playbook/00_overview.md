# Seg-Studio MCP playbook

You are connected to a Seg-Studio trainer through its MCP bridge. The bridge
exposes the tools; this playbook is the measured part, so you do not have to
learn it again on someone's dataset.

## Measure once, then label

`calibrate_sam` and `zoom_plan` are worth their half minute the first time and
nothing after it: the answer is filed with the project and `calibrate_sam`
says `measured_before` when it has already been done. A run that spends its
turns measuring and writes no mask has done nothing for the person who asked.
If nothing can be labelled, say which images were looked at and what refused
them, and `mark_review` one with that reason -- a measurement is not an
answer to "label these".

## What the policy tiers mean

Every tool declares a tier in its description: `[READ]`, `[WRITE]` or
`[DESTRUCTIVE]`. The bridge was started with `--policy read`, `write` or
`full`; a tool outside the policy refuses instead of degrading, and every call
is logged to the operator's stderr with its tier. Do not retry a refused tool
with different arguments -- the operator has to restart the bridge to widen
the policy, and that is their decision.

Every request you make is labelled with the bridge's name and the tool, and
a person with the project open sees a chip in their browser while you write.
Work in the open: do not spread writes out to stay under it.

Nothing under `--policy write` replaces what a person drew or marked clean:
`mask_put`, `write_kept`, `spot_write`, `mark_clean`, `recipe_apply` and
`prelabel_run` leave those images as they are and say so. `overwrite` sends
them anyway -- only when the person asked for exactly that -- and needs
`--policy full`, because what it replaces is not kept: the first five keep
no copy, and `prelabel_run` keeps the mask it replaced only until the next
draft over that image, and nothing of a clean mark. Under `write`,
`prelabel_run` with `overwrite` is refused as a whole if any image it would
reach -- the ones named, or with none named every image in the project --
carries a person's work. `clear_class` and `train_run_delete` cannot be
undone from here.

## Which playbook to use

Ask `annotation_status` and `classes_get` first -- and do not read
`with_mask` as "labelled". It counts mask files, and a mask file can hold
nothing: a project can report many labelled images of which most are
background and ignore only. `class_presence` reads the pixels and says
which images actually carry a class. Then read two or three teacher masks
and count their foreground components -- and check whether any
**annotated** image has none. A project whose hand-labelled images include
empty ones is a defect project whatever the object count looks like: when
every marked image carries the same number of objects, they can be the
defective ones of a feature every part has that many of, and templates cut
from those match the healthy ones on a clean image. Objects per image says
nothing on its own; a clean annotated image says everything.

Many hand-annotated projects are defects on a part, and many of those
already have a finished run -- so the first question on such a project is
not "how do I draw this" but "is the run good enough to draw it for me".
`run_agreement` answers it.

| The images show | Use | Playbook |
|---|---|---|
| The same manufactured object many times, up to ~15 per frame (small parts, fasteners, tablets) | a vision model's boxes → SAM, teacher band | `segstudio://playbook/counting_from_teacher` |
| The same object, crowded (dozens per frame) | teacher templates → SAM, or a trained run | `segstudio://playbook/counting_from_teacher` |
| Particles, powders, fibres -- every blob a different shape | a trained run only; the counting recipe scores close to zero on them | `segstudio://playbook/counting_from_teacher`, last section |
| At most one large object on a plain surface (one part on a tray), nothing annotated | the model's box → SAM's default level | `segstudio://playbook/single_object_from_vlm` |
| Many small marks of the same kind scattered on a plain surface (specks, pinholes, dust, pitting) | `spot_detect` from the person's own marks | `segstudio://playbook/spots_from_one_example` |
| A part with an occasional defect on it | a trained run if one exists, else the zoomed defect search | `segstudio://playbook/defect_candidates` |
| A new project with nothing annotated yet | the bootstrap loop | `segstudio://playbook/bootstrap_loop` |

The prompts `count_objects_from_teacher`, `mark_single_object`,
`find_defect_candidates` and `bootstrap_loop` render those playbooks with a
project filled in.

## Many small marks, in full, because it is four steps

The page is `segstudio://playbook/spots_from_one_example`, and a client that
cannot read resources has the whole of it here:

1. Find an image the person drew the marks on: `teacher_band` names the
   images they drew, and `class_presence` says which images carry a class.
   Their marks are the example, because they are the operator's own
   judgement of what counts.
2. On an image that still has to be labelled, `spot_detect(project_id,
   item_id, like_item_id=<the image they drew>)`. It measures their marks on
   their own mask first, says how well that carries, and finds the same kind
   of mark on this image. Leave sensitivity out: it is measured, and a
   number you pick is a guess against a scale you cannot see.
3. Look at the picture of what it found, and read `count`. Finds on
   something that is not the surface you label -- its edge, a fixture, the
   stand -- are what a region is for: `spot_detect` again with `region_json`,
   the surface as a box or an outline read off the copy you were shown, and
   only what lies inside it is kept.
4. `spot_write` on the same image. The class is the one the teachers paint
   with; you do not pass it, and the mask never travels through this
   conversation. `spot_write` leaves an image a person drew as it is.

With no image of theirs to take the example from, point instead: open the
image you are labelling with `image_get_b64`, crop in with `crop_json` and
`min_side` until you can see one mark, and `spot_detect(project_id, item_id,
x, y)` with a point inside it, read off that copy. If its `next` says the
point is not on a mark, point again -- no sensitivity mends a miss.

Do not zoom around looking for the marks. Every crop shows a few specks that
look like every other speck, so there is nothing to decide and nothing gets
written. Do not use this for one large object, for scratches or lines, or to
decide which marks matter -- it finds everything of the kind it is shown.

## Three things that do not work, measured

1. **Asking a vision model for coordinates in a crowded frame.** On a frame
   of dozens of touching parts the model either never finished thinking (its
   whole token cap spent, an empty answer) or, when forced past its thinking,
   returned an evenly spaced grid of round numbers; on a crowded frame of
   small objects it claimed far more than there were and put them anywhere.
   This held for the 8B and the 27B model. On a handful of well-separated
   parts the same models put every point on a part, and asked for **boxes**
   on a dozen or so touching objects they came close to the hand masks. The
   difference is the picture, not the model.
2. **Asking a vision model for a judgement instead of an object.** "The odd
   one among similar objects", "the one in the wrong position", "a scratch"
   on a patterned surface: it boxes everything of the kind (every similar
   object, every grid line) and the teacher band cannot sort them. What works
   is an object that is the thing in the picture; judgements and tiny things
   on texture do not. Tiny things on a PLAIN surface are the exception, and
   they are not done by looking either: `spot_detect` measures the person's
   own marks and finds the rest --
   `segstudio://playbook/spots_from_one_example`.
3. **Painting a whole image from one click.** SAM's own default level can
   cover most of the frame -- an answer with the background -- and a
   single such mask ruins the union. Every playbook below filters SAM's
   proposals through the sizes the human actually drew.

## Before choosing a playbook, read the teachers

`segstudio://playbook/from_the_teachers` is what to measure off the images a
person has already annotated, and what each measurement decides -- including
the two that take two minutes and say whether this project can be labelled
this way at all. It also lists what was measured and did not work, so the same
ladder is not climbed twice.

## How to end

With a request, not a claim of completion: the images you flagged with
`mark_review` and why, first, as "please look at these". The person opens
those; the rest they trust because you said which ones not to.

## What a good result looks like

With the counting recipe and a vision model's boxes, the masks come close to
a person's: separated objects of one kind nearly exactly, touching ones a
little less closely, few false objects, and the object count exactly right
on most images. If `predict_verdicts` or a hand check puts you far below
that, stop and report rather than write more masks.
