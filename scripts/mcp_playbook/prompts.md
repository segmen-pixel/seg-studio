# Prompts that were measured

Every line in these was added because a measurement showed the model doing
the wrong thing without it. Shorten them only against a measurement.

Send them to a vision model with `temperature 0`, thinking off, and a token
cap; parse the numbers with a regular expression rather than `json.loads`,
because the answer sometimes arrives inside a code fence or with a trailing
sentence. **State the coordinate system in every prompt** (the 0 – 1000 grid
line below) and scale by the width and height of what was sent. Guessing
the scale from the values does not work: a box whose numbers happen to fit
inside the image size reads as pixels and lands in the wrong place. Pin the
context (`num_ctx 8192` on Ollama): the default 32k context plus the vision
encoder overflowed a 24 GB card with the trainer resident and fell back to
CPU, minutes per image instead of seconds.

## Put the judgement in the object word

`{object}` is the only place a distinction can live: a word that names the
state the person means ("upright part", not "part") scored clearly higher
than the bare noun and stopped counting the wrong state. A separate yes/no
question per candidate crop is not a substitute: asked "is it one?" the model
said no to real objects.

## Boxes for separated objects (one image or tile)

The best proposal for up to ~15 objects per frame, separated or touching,
with SAM given the box. Do not use it on a crowded frame.

```
This is part of a photo of {object}s.
Find every {object} you can see, including any partly hidden behind another one.
Coordinates are on a 0-1000 grid over the image you see: (0,0) is the
top-left corner, (1000,1000) the bottom-right corner.
Reply with ONLY this JSON, nothing else:
{"count": N, "boxes": [[x0, y0, x1, y1], ...]}
```

Pass each box to `sam_segment` as `box_json` with its centre in
`points_json`. About twice the tokens of the centres prompt (2 – 4 s per
frame warm and alone on a 27B model).

## Anything not painted over (the second ask, same tile)

Send after the first pass, with every accepted mask painted flat grey
(128,128,128) on the image. The model under-counts on the first pass and
this recovers the miss without adding false boxes (frames that were short
recovered, the others unchanged).

```
This is part of a photo of {object}s. The {object}s already found have
been painted over with flat grey.
Are there any {object}s NOT painted over? Include any partly hidden
behind another one.
Coordinates are on a 0-1000 grid over the image you see: (0,0) is the
top-left corner, (1000,1000) the bottom-right corner.
If there are none left, reply exactly {"count": 0, "boxes": []}.
Reply with ONLY this JSON, nothing else:
{"count": N, "boxes": [[x0, y0, x1, y1], ...]}
```

Drop any returned box whose centre lands on a painted mask; the model
sometimes re-reports what it can no longer see. A `count` of 0 is an
answer, not a stall.

## Centres of separated objects (one image or tile)

Faster than boxes and nearly as good on objects that do not touch; on
touching objects SAM's mask from a centre spills onto the neighbour and fits
clearly worse than with boxes.

```
This is part of a photo of {object}s.
Mark the centre of every {object} you can see, including any that is
partly hidden behind another one.
Coordinates are on a 0-1000 grid over the image you see: (0,0) is the
top-left corner, (1000,1000) the bottom-right corner.
Reply with ONLY this JSON, nothing else:
{"count": N, "centres": [[x, y], ...]}
```

Measured before the grid line was added: every point on a part, count
exactly right, on a few images of a handful of parts each (27B model,
~1.5 s per image warm on an RTX 3090).

## Defect with a normal reference (two images: sheet, then tile)

```
The FIRST image is a reference sheet: four crops of this {surface} in its
normal, acceptable condition. Note how the {normal features} look when nothing
is wrong.

The SECOND image is one crop of the same kind of surface, at the same
magnification.
Decide whether the SECOND image contains a DEFECT -- {defect kinds} -- that
does NOT appear anywhere in the reference sheet.
Ordinary variation in {normal features} is NOT a defect.
If the second crop is within normal variation, answer exactly
{"count": 0, "centres": []}.
Reply with ONLY JSON: {"count": N, "centres": [[x, y], ...]}
```

Fill `{surface}` (e.g. "machined metal surface with a regular striped
texture"), `{normal features}` (e.g. "the stripes, brightness, staining or
speckle") and `{defect kinds}` (e.g. "a chip, a pit, a dent or a foreign
deposit") from what the project's images actually show.

Why each part is there:

| line | what happened without it |
|---|---|
| the reference sheet and "note how ... look when nothing is wrong" | false flags on clean frames |
| naming the defect kinds | "defect" alone matched anything |
| "ordinary variation ... is NOT a defect" | every dark band in the stripes was a chip |
| "answer exactly {count: 0}" | it wrote something even for an empty tile |
| "ONLY JSON" | prose and code fences around the answer |

Naming a defect as if it were an object ("scratch") and skipping the
reference sheet does not work: on patterned surfaces -- grids, pads,
printed text, component edges -- the model boxed every grid line, pad, glyph
and edge, scoring next to nothing, and zooming in without the sheet made it
worse.

## One object, box and centre (one image)

For at most one large object on a plain surface. Feed the box to
`sam_segment` as `box_json` and take `default_level`.

```
This photo shows {scene}. There may be one {thing} lying in it. Find it.
Give coordinates on a 0-1000 grid where [0,0] is the top-left corner of the
image and [1000,1000] is the bottom-right corner, regardless of the image size.
Reply with ONLY this JSON, nothing else:
{"count": N, "box": [x1, y1, x2, y2], "centre": [x, y], "what": "short name"}
Use count 0 and an empty box if there is nothing.
```

Measured: several kinds of object boxed and named, count 0 on an empty
frame, 1-2 s each warm.
