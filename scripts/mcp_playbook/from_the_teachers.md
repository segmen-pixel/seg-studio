# What to read off the teachers

Everything about how to label a project is a property of that project, and the
images a person has already annotated are where it is written down. Measure it;
do not assume it. Each measurement below decides something, and the tool that
takes it is named.

The order matters: the first two decide whether this project can be labelled
this way at all, and they take about two minutes.

## 1. Can the model see the objects? -- `zoom_plan`, `zoom_score` with points

Views of an annotated image, widest first, each with how many of the person's
objects are inside it. Look at each, say where you think the objects are, and
`zoom_score` marks the points against their mask. Take the widest view whose
recall is 0.8 or better and use crops that size.

Measured: on large photographs of medium-sized objects, the whole frame
missed a fair share of them and half the frame -- twice the magnification --
found most. Parts on a tray were nearly all found in the whole frame, so no
crop was needed.

## 2. Can one object be expressed as a rectangle? -- `zoom_score` with boxes

The same tool, given boxes instead of points, scored against the box around
each of their objects. This is the question that decides the method.

| mean box IoU | what it means |
|---|---|
| 0.5 and up | box prompts are sound; go ahead |
| 0.35 - 0.5 | expect fusion; check the first written mask by eye |
| under 0.35 | **this project is not for box prompts.** Stop here |

On elongated parts lying across each other it stayed at the stop line or under
it, and did not move with zooming in (which made it worse), with being shown the
teacher's own mask first, or with any segmenter. The model put down a grid of
similar rectangles that matched nothing; it could say where the parts were and
not how far one extended. Hours can go into the rest of the ladder before this
is measured.
Measure it first.

## 3. What one object is -- `teacher_band`, `teacher_view`

`teacher_band` reads their masks for the numbers: how many objects a frame
carries, area and box as fractions of the frame, the size of one object **in
pixels of the picture in hand** (a model cannot box with a fraction), what they
paint with (the class id: writing 1 where they paint 2 trains on two names for
one thing), and what they leave unpainted (0, or 255 for ignore).

`teacher_view` renders their image with their objects outlined, plus single
objects cut out. Numbers do not say where one object ends; this does. It is
also where to decide whether one object spans parts that look different -- a
dark head on a bright shaft, say -- because a point on one part returns
that part. When it does, give `accept_points` a group of points per object
rather than one.

## 4. Which segmenter, and how to point -- `calibrate_sam`

Each model and each way of pointing, asked about their own objects and scored
against what they drew: fit (IoU) and how often one mask swallows the
neighbour. Half a minute, and it settles an argument that is otherwise endless.

On those parts mobile_sam and tinysam fitted well with a box and close to each
other; every model was much worse with a point alone, and sam2 worse than both.
The spread between the best and the worst *model* was
smaller than the spread between a good box and a sloppy one.

## 5. How much SAM over-paints -- the shrink in `teacher_band`

Asked of a sample of their objects, scored against their mask, and measured in
pixels of SAM's own 1024 px working image -- the same error is more pixels on a
bigger picture.

## What did not work, and need not be tried again

All measured against the same teachers, all worse than doing nothing:

- **Appearance floors as absolute numbers.** Teachers from brightly lit frames
  turned away many of the same person's objects in dim ones. As a share of
  each picture's own contrast and texture they carry; and a floor that admits
  the background of the frames it was set on is dropped rather than kept
  (objects on a plain surface keep both floors, admitting none of the
  background patches; objects on a surface that looks like them keep neither).
- **Preprocessing the image.** Unsharp, bilateral, specular inpainting,
  black-hat valley deepening, illumination flattening, edge darkening: every
  one fitted worse than the untouched picture. SAM is a ViT trained on
  ordinary photographs; what makes a boundary obvious to a person moves the
  image away from what it was trained on.
- **Brightness and contrast.** No difference worth having. (In the browser these are a display
  filter and never reach the server at all.)
- **Rotating the picture so an object lies level.** Worse than the untouched
  picture. A turned box holds barely less of the neighbours than an upright
  one, because a headed part is a T and no rectangle fits it.
- **Template matching from the teachers' cut-outs.** Excellent where the same
  manufactured thing repeats at one scale, and no better than the model where
  poses vary: a poor fit, at minutes an image, on parts lying across each
  other, and many missed on large photographs, because rotation alone does
  not cover perspective.
- **Showing the object's size as a picture.** A dark strip beside the copy
  carrying one rectangle the size of a teacher's object, so the size is seen
  rather than read. Box IoU rose only a little and recall fell sharply: the
  boxes became the right shape and stopped covering the frame. The strip costs
  a few percent of the bytes and is kept for the objects-per-view it also
  answers, but it does not place a box.
- **Looking closer at a small object.** `max_side` only ever shrank a view, so
  a small defect in a large frame reached the model at a few dozen pixels
  however it was cropped; `min_side` was added to enlarge a crop, and on the
  same views it found fewer objects with the enlargement than without it.
  Cropping to where the objects are is worth doing -- the same measurement
  found far more -- but the pixels of an enlargement are
  not what was missing.
- **Shape as an acceptance test.** Solidity, and the spread of grey inside a
  mask, both overlap between a whole object and a part of one: a floor at the
  teachers' lowest admitted every fused region, and most part-masks.
  Neither tells one object from two.

## When the answer is that this is not the job for it

Parts lying across each other, measured: it cannot bound them (box IoU at or
under the stop line), one point returns the head or the shaft rather than the
whole part, and a correct box returns the object almost perfectly. The one
recall figure that looked good -- nearly everything, from dozens of boxes --
was a grid laid over the frame, not the parts found; rendering the boxes is what showed that,
and no number in the reply would have.

Everything needed was there except the box, and nothing but a person could
draw it.

Three or four images annotated by hand, then `train_start` and `prelabel_run`,
is the shorter road. A trained run beat every proposal method measured here.

## 6. One frame at a time is too much to point at

Asked to put one point on each of a few dozen parts in a wide frame, every
point came back inside the top of the picture: a small share of the parts
found, and many points on nothing. The same picture in two halves, each
shown as a strip, found far more of them, and not one point on nothing. Boxing behaved the same way: whole frame, the boxes crowded into the
upper half.

So the y coordinates collapse toward the top of a busy frame, and the fix is the
view rather than the prompt: pass `crop_json` and work through the frame in
horizontal strips, scoring each with the same `crop_json`. Cheap to check on any
new project -- two strips, two minutes -- and worth checking before concluding
that a picture is too crowded to point at.

## 7. When the person's definition is the distinction

A project where what is marked is one kind of surface defect, and the frames
also carry cracks, joints, seams and other ragged edges, none of which is
marked.

Every appearance measure said as much before a single view was asked for: the
texture floor and the contrast floor would each admit nearly all of the
background patches from the teachers' own frames. When both floors are
useless, the band has nothing left but size, and size does not tell that
defect from a seam.

Rendering the points showed the same thing from the other side: they land on
cracks and other edges, which are real defects and are not the one the person
is labelling. Neither cropping nor enlarging moves that, because it is not a
question of seeing.

Where the teachers already number in the tens, this is the point to stop
prompting and start training: dozens of hand-labelled frames against a few
times as many to do is a ratio no proposal method here has beaten.


## How many teachers is enough

The band is the observed range widened by a slack, and the slack now grows when
there are few objects to have seen (`1 + 1/sqrt(n)`): half again as wide at
eight objects, a tenth wider at a hundred. `teacher_band` reports the count and
says when it is thin.

On a defect project, a handful of hand-drawn images holding only a few
objects gave a band whose top was well below the largest objects the same
person drew elsewhere in the project -- several of their own objects fell
outside a band drawn from their own hand. Every mask the recipe wrote came back
far smaller than what they drew, with recall far below precision against their
own masks, because only the small ones fit.

So: if boxes are refused for area on objects that are plainly there, the answer
is more hand-drawn images, not smaller boxes. Twenty objects is where the band
stops moving much; eight is a guess with a number attached to it.
