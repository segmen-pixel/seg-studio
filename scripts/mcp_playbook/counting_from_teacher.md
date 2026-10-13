# Counting the same object many times

For a project where every image shows copies of one manufactured thing --
parts on a tray, fasteners in a bin, tablets on a plate. The human has drawn
at least one image; that annotation is the specification.

## The short way: three tools hold the arithmetic

Everything below is what those tools do; read it to know why they answer
as they do. To run the recipe from any client:

1. `teacher_band(project_id)` once. It reads the teacher masks, computes
   the acceptance band, calibrates the shrink, and says how many objects a
   frame carries when the teachers agree.
2. Per image: `image_get_b64`, look, and for every object you see call
   `accept_mask` with its box in pixel coordinates (or a point). It asks
   SAM, judges the levels against the band, and keeps the right one --
   or tells you why not.
3. `write_kept` for the image. It unions what was kept, applies the
   shrink, writes the mask, and tells you if the count fell short; then
   `mark_review` with that reason.
4. Paint what was accepted flat grey on the image and ask once more for
   anything not painted (the prompts page); accept and write again.

Measured with a local vision model driving these tools on held-out frames of
separated parts, the masks agreed with the hand annotation about as well as
the scripted recipe's, and it found a part of a different finish that the
template recipe misses. A model takes minutes a frame where a Python driver
of the same tools takes seconds, so batch work belongs to the driver and the
model to the judgement calls.

## Why the teacher mask is the whole method

A teacher mask is not just "class 1 here". It says how big the object is, how
its bounding box runs, how much texture and contrast it has, and -- as a set
of cut-outs -- what it looks like at every length it comes in. Everything
below reads those from the mask instead of guessing.

## Steps

The comparisons below are held out against hand annotation, on three kinds
of frame: **separated** parts (a handful per frame), **touching** objects
(up to a dozen or so per frame) and **crowded** small objects (many dozens
per frame).

1. `annotation_status` -- how many images have a mask. You need at least one
   with paint in it (`class_presence`). If the project has a completed run,
   ask `run_agreement` first: a run that scores well beats this recipe, and
   the margin is not small: on touching objects a trained run fitted the
   hand masks clearly better than templates did, and on crowded frames far
   better. Templates are for the project that has a teacher and no run yet.
2. `mask_get_b64` on each annotated image, and `image_get_b64` on the same
   images. Connected components of class 1 are the teacher objects. Drop
   specks **relative to the objects** -- anything under about 8% of the median
   component -- not relative to the frame. A frame-relative floor that
   looked reasonable on objects of a few percent each threw away every
   object on a crowded frame of tiny ones: zero teachers, zero templates, and
   a stack trace instead of a label. Refuse to continue with zero teachers and
   say why.
3. From the teacher objects compute the **acceptance band**:
   - area as a fraction of the frame: `[min / 1.8, max × 1.8]`
   - bounding-box width and height as fractions: the same
   - texture (mean gradient magnitude under the mask) floor: `min / 1.5`
   - contrast (grey std under the mask) floor: `min / 1.3`

   Use the observed **range**, not percentiles. A p10-p90 band called the one
   long part in each frame an outlier and threw away most of an image's
   annotation. The band exists to reject what is nothing like the object -- a
   proposal that was the surface under the objects, or the whole frame -- and
   those are many times the median, not 1.5×.
4. Propose where the objects are. How close each comes to the hand
   annotation, held out:

   | proposals | separated | touching | crowded |
   |---|---|---|---|
   | **a vision model's boxes** → SAM box prompt (+ calibrated shrink) | **best** | **best** | poor |
   | a vision model's centres → SAM point | as good | clearly worse | poor |
   | teacher templates → SAM point | nearly as good | worse still | poor, the least so |
   | teacher templates → SAM box | nearly as good | worst | – |
   | grid of points → SAM | finds most objects | – | – |

   - **A vision model's boxes** (best up to ~15 objects per frame): send the
     frame, or tiles of it, zoomed to ~1024 px, with the prompt under
     "boxes for separated objects" in `segstudio://playbook/prompts`. Pass
     each box to `sam_segment` as `box_json` together with its centre as the
     point. The box is what makes the difference on touching objects: with a
     centre alone SAM's largest level swallowed the neighbour; with the box
     it stays inside, with high precision and recall.
     Costs 2–4 s per frame warm on a 27B model.
   - **Templates** (no model needed; the fallback when the frame is crowded):
     cut every teacher object out with its mask, match the cut-outs back
     against the target over a full turn of rotations (every 20°), take the
     top peaks. Match on the **gradient image**, not brightness: masked
     template matching only offers `TM_CCORR_NORMED`, which does not
     subtract the mean, and on brightness the flat background out-scored
     every object. On the gradient the same templates hit every object. The
     click is the template's **deepest interior point** (max of
     the distance transform), not the centroid -- a long part lying
     diagonally has its centroid off it. A box from the template's
     footprint was worse: the rotated cut-out's box is loose and SAM
     fragments inside it.
   - **Grid** (finds most objects): a point every ~100 px. Simpler, slower, and it
     needs the texture floor to keep the surface out.

   **Ask twice.** The model under-counts: on a frame of a few objects it
   tends to name one fewer. After the first pass, paint
   every accepted mask flat grey on the image, send the same tiles again
   with the prompt under "anything not painted over" in
   `segstudio://playbook/prompts`, and keep only boxes whose centre is not
   on a painted mask. Measured: it recovered the objects missed where the
   first pass under-counted, left the other frames unchanged, added nothing
   false, and took about twice the model time.

   **Crowded frames** (many dozens of small objects per image, many tiles)
   defeat the vision model both ways -- it claimed far more objects than
   there were and put them anywhere -- and the templates barely do better.
   Those need a trained run; see
   `segstudio://playbook/defect_candidates` step 1.
5. For every proposal, `sam_segment`. It returns three levels. Keep the
   largest level that passes the band; skip the proposal if none does.
   Ignore SAM's own score and its `default_level` -- both chose the
   background on real images. (The bridge's `accept_mask` does this for
   you, and once `calibrate_sam` has measured a rung for the project it
   uses that rung unless you name another.)

   "Largest that passes" was measured against two alternatives, because on
   one image it swallowed a neighbour: SAM's `default_level` when it passes,
   and the median-area level. Held out (mobile_sam), the largest passing
   level came out ahead of both on separated and on touching objects, and
   on crowded frames all three were equally poor.

   The band already removes the "whole surface" answers; among what is left,
   the smaller levels are parts of the object (a part's head without its
   shaft), and picking them costs more than the occasional neighbour.
6. Dedupe with a fallback: a sweep clicks the same object several times, and
   the largest level sometimes covers two touching objects. When a mask
   overlaps an already-kept mask by more than 55% of the smaller one, do not
   drop the proposal outright -- take the next smaller level that passes the
   band and does not overlap; drop only if none does. On touching objects
   this raised the score for templates, model centres and model boxes alike;
   separated ones were unchanged or up.
   **Count the kept masks, not the union's connected components.** Objects
   that touch merge in the union: on stacked objects the union held fewer
   blobs than kept masks on one image and several times as many on another,
   while the kept count stayed within one of the truth on most images. The
   dedupe itself merges stacked objects that overlap more than half, so it
   under-counts them; for objects that pile up, lower that threshold and
   expect to under-count.
7. Shrink each kept mask by the amount the teachers ask for, then union.
   SAM paints a little outside a person's brush, and on touching objects it
   fills the gap between them, so the union merges neighbours. Calibrate on
   the teacher images: for each hand-drawn object, `sam_segment` from its own
   bounding box and centre, keep the largest passing level, and score the
   union against the hand mask after eroding every mask by 0, 1, 2, 3, 4 px;
   take the best. On touching objects the teachers asked for the smallest
   shrink, which raised the fit on the teachers and held out alike and made
   the count match for the first time; separated parts asked for none and
   were unchanged. Never pick the erosion by eye; the teachers decide, and 0
   is a valid answer.
8. Union the kept masks. Write with `mask_put`, passing the class to write
   the objects as: SAM's levels are 0 and 255, and 255 is ignore. Write `255`
   (ignore) for the background in counting projects -- that is the
   convention the human used -- unless `mask_get_b64` on a teacher shows `0`.
   (`write_kept` does all of this in one call.)
9. **Check your count against the teachers' and say so.** The teachers show
   how many objects a frame carries (five or six, say). When fewer were
   kept than that -- most often because the model boxed one fewer than
   there were -- `mark_review`
   the image with the reason rather than leaving a mask that looks finished.
   Then **ask**: the last thing you write is the list of those images, each
   with its reason, as a request -- "please look at these" -- before any
   number. The person would rather open a few images than trust hundreds.
10. **Do not write an all-background mask when nothing was accepted.** That
   tells training the image is empty. Leave it unlabelled and say so.
11. Verify: `predict_verdicts` needs a run; without one, compare against any
   held-out teacher with IoU and component count, and look at the overlay.

## What to expect

| | model boxes → SAM | templates → SAM |
|---|---|---|
| separated parts, held out | close to the hand masks; count exact on most frames | about as close |
| touching objects, held out | close, and a little closer with the calibrated shrink; high precision and recall | clearly worse |
| time per small frame | 2–4 s model + ~7 ms per SAM click | ~11 s (transport, not SAM) |

A trained run, where one exists, is about level with the model's boxes on
the same touching objects. End to end through the bridge, writing into a
copy of the project, both kinds held up (teachers included), and the count
of touching objects first came out exact once the calibrated shrink
separated them.

## Which SAM model

All five, same images. With the full recipe (a vision model's boxes, the
overlap fallback, the calibrated shrink), mobile_sam fitted best on
separated and on touching objects, tinysam and efficient_sam_ti were just
behind it, and the two sam2 models came last. With teacher templates and a
point (the crowded-frame fallback) the order on separated parts was much
the same; on touching and crowded frames every model was about equally
poor, and efficient_sam_ti did not finish.

Stay with mobile_sam. tinysam is a close second and answers with a single
level, so the level rule has nothing to choose from; efficient_sam_ti gets
the count of touching objects right most often but paints them less
accurately; the sam2 models are behind on both. The calibrated shrink chose
the same amount for nearly every model (none on separated parts, the
smallest on touching objects), so it is a property of the annotation, not
of the model. A model whose package is not installed answers NSS-5010 (501)
with an install hint -- pick another model rather than retrying.

**Small objects: the count holds, the mask does not.** On crowded frames of
small objects, the kept count came close to the truth but the masks fitted
poorly -- SAM's levels do not follow an edge that small -- and a frame took
well over a minute. Use the recipe for the number, not for the
pixels, and say so; a mask that poor is worse than none for training.

What it misses: objects of a different colour or finish from the teacher --
one part of another finish among the rest, dark-headed ones among bright
ones. If the project has such a variant, ask for a teacher image that
includes it.

## Where the recipe works, and where it does not

- It works when the object is **the thing in the picture**. A judgement
  about it has to be **written into the object word**, not left to the
  band: naming the state the person means ("upright", say, rather than the
  bare noun) fixed the frames where the wrong state was counted, at no extra
  cost. What does not work is a separate yes/no verification of each
  candidate crop ("is this one?"): it rejected real objects, because the
  model's answer hangs on the word. Judgements the word cannot carry ("the
  one in the wrong position") stay out of reach.
- **Defects are not objects.** Named as "scratch", the model boxes every
  edge on a structured surface. Use `segstudio://playbook/defect_candidates`
  with its normal reference; and do not zoom without one -- tiles sized to
  the object made scratches worse, because every tile then had edges to call
  scratches.
- **Tiny objects on a textured surface** (well under 0.1% of the frame) are
  not found at native size, and zoom alone barely helps -- the texture
  wins. A trained run is the answer there.
- Frames at or under 640 px are shown whole by the tiler; that is right for
  objects of a few percent of the frame and wrong for tiny ones. When the
  median object is under about 0.1% of the frame, tile by object size
  anyway, and expect false boxes.

## Tried and rejected: a colour band from the teachers

Where the false boxes were objects in the wrong state, a band on colour
looked obvious: mean Lab colour of every teacher object, reject a candidate
farther from all of them than 2.5 × the teachers' own spread (floor 15).
Measured, it threw away the real objects too -- much of the agreement
lost, some frames left empty -- because a handful of teachers do not span
the colours the objects come in, and it cost the touching objects a little as
well. Size and texture generalise from a few teachers; colour does not. Do
not add it.

## Tried and rejected: a template fallback driven by the teachers' count

When fewer objects were kept than the teachers' median count per frame,
match the teacher cut-outs on the unpainted area and add what passes the
band. On separated parts it never fired (the kept count met the teachers'
even where one more was missed). On touching objects, whose count varies a
lot from frame to frame, the median of the teachers was a false yardstick:
it fired on every frame and painted false objects, and the fit fell. A count
from the teachers is a check only when the teachers agree with each other
(within one); otherwise the model's own count is the check, and the honest
action on a shortfall is `mark_review`, not more paint.

## Tried and found to change nothing: prompts along a thin object

For elongated objects, a second SAM pass with three points spread along the
first mask's principal axis (so a diagonal scratch is prompted along its
length). Elongated parts were unchanged; scratches drawn as broad strokes
were unaffected because their hand masks are not thin lines (by area over
mean width squared they came out no longer than a disc), and their loss is
elsewhere
-- the model naming one scratch where there are two, and the band rejecting
both boxes on one frame. Not adopted.

## Where the recipe stops working (measured)

Particles and fibres are not "the same object many times" in the sense the
templates need: every grain is a different blob, and a fibre is a different
curve on every image. Held out against hand annotation, same code as above,
a particle project and a fibre project, each with plenty of teacher
objects, scored a mean IoU near zero.

The band accepted everything (areas over orders of magnitude, boxes as
wide as the frame), the matcher put its peaks anywhere, and the union was
noise. For these, the only
things that worked were a person drawing a few images and a trained run
(`prelabel_run`). Do not spend a sweep on them; say so and ask for teachers.

## Data problems this surfaced

- An annotate index can list the same filename twice, and the repeat may
  carry another image's mask. Dedupe by `name` before doing anything.
- `with_mask` counts files, not paint. Most of a project's "labelled"
  masks can hold only background and ignore. Pick teachers from
  `class_presence`, not from `annotation_status`.
- The class you are counting is not always 1. A project may paint only
  class 2. Take the painted class from the teacher masks; treat everything
  but 0 and 255 as foreground when you score.
- `image_get_b64` takes an image's id as `dataset_images` lists it and finds
  the stored file whatever its format; `mask_get_b64` wants the `id` too.
  An id is not always the upload's `name`.
- A teacher mask can be unfinished (part of the frame left untouched). Check
  that the annotated region spans the frame before trusting it as ground
  truth.
