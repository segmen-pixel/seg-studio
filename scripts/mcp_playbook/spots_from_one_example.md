# Many small marks, from one example

The pictures are mostly one flat surface, and the target is dozens to hundreds
of small marks scattered over it: specks, pinholes, dust, pitting, porosity.
Every mark is the same kind of thing and a few pixels across.

Drawing them by hand is the whole job -- one frame can hold over a hundred --
and this is the one case where a single tool call does it.

## Why the other playbooks stall here

The overview says asking a vision model for coordinates in a crowded frame does
not work, and that tiny things on texture do not. Both apply, and the zoomed
defect search makes it worse rather than better: it walks crop after crop, and
each crop shows a few specks that look like every other speck, so there is
nothing to decide and nothing gets written: a run can read crop after crop
of the same image and write no mask.

`spot_detect` does not look for specks. It measures an example -- the marks a
person drew, or the one you point at -- and then finds everything in the frame
that resembles it.

## The call

1. Take the example from an image the person drew: `teacher_band` names the
   images they drew, and `class_presence` says which images carry a class.
   Their marks are the best example you will get, because they are the
   operator's own judgement of what counts.
2. On an image still to be labelled, `spot_detect(project_id, item_id,
   like_item_id=<the image they drew>)`. It measures their marks on their own
   mask first -- `measured` says how many of theirs it finds there, and how
   many when chosen on half of the mask and scored on the other half, which
   is what to expect on a frame nobody drew -- then finds the same kind of
   mark on this image.
3. Look at the picture it answers with, and read `count`, before writing
   anything.
4. `spot_write` on the same image. The class is the one the teachers paint
   with, taken from the project rather than asked for, and the mask is written
   from where it was found -- it does not travel back through the conversation
   to get there. An image a person drew is left as it is.

With no drawn image to take the example from, point: open the image you are
labelling with `image_get_b64`, crop in until you can see one mark, and call
`spot_detect(project_id, item_id, x, y)` with a point inside it, read off the
grid of that copy. The point is a place, not an outline -- any pixel of the
mark. A point finds marks on its own image only, and a pixel read off a mask
is in the picture's own coordinates, not the copy's.

Leave `sensitivity` out. It is measured for you -- the tightest threshold that
still recovers the example -- and reported back; a number you choose is a
guess against a scale you cannot see. If `next` says your point is not on a
mark, point again: no sensitivity mends a miss.

## Reading the answer

The picture is the check, and `count` with it. A handful where the picture
clearly holds hundreds, or thousands where you expected dozens, means the
example was not the marks -- for a point, that it landed on the surface
instead of a mark. Finds on something that is not the surface you label -- its
edge, a fixture, the stand -- are what `region_json` is for: the surface as a
box or an outline, read off the copy you were shown. Only what lies inside it
is kept, and `outside_region` says how many were left out.

`size_range` and `color_tolerance` are what it read off the example. They are
deliberately loose: a worse instance of the same defect is bigger and brighter
than the one you pointed at, and those are the ones worth finding: an example
a few pixels across can go on to find marks many times its size.

On a frame of many specks on a flat background, from one point: nearly all of
them found, with a handful of extras. The same picture through the annotator's
older detector found far fewer.

## What this is not for

* One large object, or a handful of large objects -- SAM, through the counting
  or single-object playbooks.
* Scratches, cracks, lines -- `crack-trace` in the annotator; a line is not a
  blob and this will chase its brightest points.
* Deciding which marks matter. It finds everything of the kind it is shown.
  If only some of them are defects, that judgement is not in this tool, and the
  overview's second measured failure applies.
* A surface that is not plain. The threshold it works against is one number for
  the whole frame, so a patterned half of a picture sets the bar for the flat
  half. Take the example on the plain part, and treat counts near an edge, a
  bright strip or a hole with suspicion.
