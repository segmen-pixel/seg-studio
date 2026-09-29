# One object on a plain background

For a project where each image shows at most one thing on a uniform surface
-- a part on a tray, a single item in a container -- and nothing is annotated
yet. This is the one case where a vision model's coordinates are worth taking:
one large, distinct object, nothing to enumerate.

## Steps

1. `annotation_status`, `classes_get`, `dataset_images`. If file names or
   folders already say which images are empty (`empty_*`, say), use
   that: those images get `mark_clean`, a deliberate
   negative, and the model is only asked about the rest.
2. Ask the vision model for a bounding box and a centre, with the prompt in
   `segstudio://playbook/prompts` under "one object, box and centre". **Say
   in the prompt that coordinates are on a 0-1000 grid**, then always scale
   by width and height. Do not try to guess from the values whether they are
   pixels: a box that happens to fit inside the image size reads as pixels
   and lands in the wrong place -- a thin object's box comes back well off
   that way while the big objects look roughly right.
3. `sam_segment` with `box_json`. Take the level SAM names in `default_level`
   -- on a plain surface it is usually the object -- and drop connected
   components under 0.05% of the frame; the `whole` level of a shiny object
   carries specks of the floor. Do not pick the largest level: it can be the
   floor of the container the object lies in.
4. Sanity band before any teacher exists: 0.5% – 60% of the frame, at most a
   few pieces. Then a band **per kind of object**, keyed on the model's own
   `what`, once three masks of that kind are in: range ± slack as in
   `segstudio://playbook/counting_from_teacher`. Not one band for the
   project: when the first images hold two kinds of object, every object of
   a third kind a hair larger than the band they made is rejected. A kind
   the band has not seen yet gets the sanity band.
5. `mask_put` for accepted masks, with `class_id` set: SAM's masks are 0 and
   255, and 255 is ignore. `mark_clean` for images the file names or the
   model (count 0, on a name that says empty) mark as empty; leave the rest
   unlabelled and list them.
6. Look at an overlay sheet before writing the bulk. Five images tell you
   whether the boxes are placed and the level is right; forty do not tell you
   more.

## Where it fails: frames with nothing in them

A model asked to find something finds it. Where the labelled frames were
mostly clean, the box prompt -- with "count 0 if nothing is damaged" in it --
returned a damaged region on every clean one, while a real chip came back
boxed well enough. File names that say which frames
are empty are worth more than the prompt's promise; where they do not exist
and clean frames are possible, this recipe cannot be trusted on its own --
use the normal-reference comparison in
`segstudio://playbook/defect_candidates`, or a trained run.

## Expected numbers (measured, 27B model on an RTX 3090, warm)

Box and centre in 1-2 s per image; SAM 30-50 ms; the model named every object
it was shown and answered count 0 on an empty frame. Each mask came out one
piece after speck removal.
