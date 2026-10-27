# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The walk: each image labelled part by part at the filed zoom, by the program or by the model.

A person labelling a large picture at a zoom pans across it a window at a
time and looks at each window as it comes. A model handed the whole frame and
a dozen tools did neither: it looked at a copy shrunk to fit, boxed what it
could make out there, and spent its turns on the bookkeeping in between --
which copy a box was read off, what to open next, when to write -- and got
some of that wrong on every image, in a different way each time.

Here the program does the bookkeeping, at the zoom filed for the project
(view_choose) -- or, with none filed, at the one it chooses first by
rehearsing the walk on the person's own pictures, lowest first, and files
(_decide_rung): it opens every window-sized part of every unlabelled image
in order, the empty ones too, hands each box to SAM on the part it was read off,
writes each image once, and then shows the parts that hold the mask, with the
mask drawn on them. The model is asked one thing at a time about one picture:
where the objects are in it -- usually nowhere -- or whether the mask on it
is right. Each question is a conversation of its own: the data rules, a
short brief, the picture with where it is said in words, and the answer
asked for as a small JSON object, its boxes from 0 to 1000 across the
picture -- the scale the model answers in, whatever it is told. A question
where the objects are comes after one of the person's own pictures, asked
the same way and answered with the boxes their labels give it: one example
says what one object is where the words did not (EXAMPLE_SAID). None of it
enters the run's own conversation, so no trim of that ever touches a
picture of the harness's walk.

The person can have the model walk instead (driver "model", model_call): it
calls the view tools itself, in the run's own conversation, and is handed
each picture there, one at a time -- the one before is let go of as the next
arrives. The program then does only what keeps the walk whole: it hands the
pictures over in the same order, refuses any other, takes nothing about a
picture before it has reached the model, and holds each write back until
every picture of the image has been answered. The question before walking,
the preparation, the flags and the screen are the same for both.

What a project's target is decides the tool, and that is settled first,
once a project (decide_way): distinct objects, each boxed by the model and
handed to SAM -- the walk above -- or a scattering of small alike marks,
found together with spot_detect and written with spot_write. On one
4096x2160 frame of 134 marks a few pixels across, boxes and SAM found none,
where spot_detect with the person's own as the example found 129 of them
(precision 0.93) in 26 s; screws and flowers came out whole with boxes at
Fit. Where nothing is filed it is measured on the person's own, both ways,
or -- with nothing of theirs -- they are asked, and the answer is filed
(way_choose). A project of specks is walked by the program as well, an
image at a time: found, written, and looked at once (_speck_image).

Nothing here imports the loop, which imports this: what the walk needs of
the run -- the bridge, the model server, the way to ask the person, the
run's own books -- comes in a Walk.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

#: Where a walk that failed leaves its traceback. The run says what happened
#: and ends, so the failure is not raised to the run's own handler, and the
#: screen and the run's event log get a repr cut to 300 characters: a bug in
#: the walk was "The walk failed: KeyError('id')" and nothing else.
_log = logging.getLogger(__name__)

#: The bridge tools whose presence turns the walk on, with way_choose, which
#: files how a project is labelled (decide_way). They are not in the loop's
#: TOOLS, so a bridge from before them -- and every fake that lists TOOLS --
#: runs the old way.
WALK_TOOLS = ("view_plan", "image_view", "view_choose", "kept_drop", "way_choose")
#: Who walks the parts. "harness": this module, the model only answering one
#: picture at a time. "model": the model itself, in the run's own
#: conversation, with the view tools -- this module hands it the pictures in
#: order and holds the write back until every one has been answered (see
#: model_call). The rest is the same for both: the question before walking,
#: the preparation, the flags, one picture per image on the screen.
DRIVERS = ("harness", "model")
#: Boxes per accept_masks call: the bridge's own guidance (BATCH_HINT).
ACCEPT_BATCH = 15
#: Failed model-server calls in a row that end a walk. A server that has gone
#: away answers every one of a few thousand pictures the same way, and an
#: hour of failures is worse than a stop that says where it was.
WALK_LOOK_FAILURES_STOP = 3
#: The bridge's words for the trainer's SAM route failing on the server
#: (mcp_server._raise_for: "<status> from <path>: <why>"): SAM itself could
#: not run -- its model file not downloaded on a machine with no way out, a
#: device out of memory -- which is not a box SAM made nothing of. A 4xx is
#: the caller's mistake about one box, and stays with that box.
_SAM_DOWN = re.compile(r"\b5\d\d from \S*sam-segment\b")
#: How much of a failed tool's words the walk keeps (_call): the route it
#: failed on, then what the server said and what it says to do about it.
TOOL_ERROR_CHARS = 700
#: How many calls in a row SAM may fail on before the walk ends (_accept):
#: every box of every image after would be lost the same way, each after a
#: look of the model's, and a rehearsal of them reads as a zoom that finds
#: nothing.
SAM_FAILURES_STOP = 2
#: Refused boxes on one image at which it is put down: written as it is,
#: flagged for a person. PROBE_REFUSALS_STOP's ten, or half the boxes sent if
#: that is more: on an image of many objects ten refusals can be a few percent.
WALK_REFUSALS_MIN = 10
#: Boxes taken from one answer. A model that lists more has lost count of
#: what it is looking at.
BOXES_PER_LOOK = 40
#: A box reaching past the picture it was read off by more than this was not
#: read off it.
VIEW_SLACK = 1.02
#: What the time a walk will take is counted from, until this run has timed
#: its own looks: a median picture turn of a vision model of the default size
#: on one workstation GPU. The first image of a walk replaces it (_reestimate).
LOOK_SECONDS_DEFAULT = 3.5
#: What the person's example adds to a look for the objects, in the estimate
#: before this run has timed any (_estimate). Such a look is two pictures
#: (EXAMPLE_SAID), measured 3 to 5 s longer; counted at 3.5 s, the time the
#: person agreed to on a taught project was some half of what the walk took.
#: The first image's time has it in (_reestimate), and not the fetch of the
#: example, which comes before it (label_images).
EXAMPLE_LOOK_EXTRA_SECONDS = 4.0
#: The share of parts that hold mask, each of them looked at again in review.
REVIEW_SHARE_DEFAULT = 0.5
#: SAM, writing and fetching, per image.
PER_IMAGE_SECONDS_DEFAULT = 5.0
#: What the run's own conversation is told of a walk, at most.
SUMMARY_CHARS = 1500
#: What the loop keeps of a call that failed (loop._short_error): its first
#: 200 characters, and none of its other keys. A copy, held equal by a test:
#: the walk does not import the loop. A refusal of the model driver's says
#: what to do instead inside it (_refused).
REFUSAL_CHARS = 200
#: The run's own turns a model walking the parts itself is allowed for a
#: picture it looks at (_looks_of: each part, the whole picture after them
#: for what is too large for a part, and the share of parts checked again
#: after the write): one turn to send what it found, one to move on.
MODEL_TURNS_PER_LOOK = 2
#: And per image, besides: its write, a rewrite for each review_redos, and a
#: refused call or two. With these the run's step limit is not what ends a
#: walk the model drives (model_turns): each of its pictures is a turn of the
#: run's, and an image of 255 parts at 400% is some 400 of them.
MODEL_TURNS_PER_IMAGE = 8
#: How a project is labelled -- its way, filed with it (way_choose) and read
#: back by view_plan. "objects": each one boxed by the model and handed to
#: SAM, the walk above. "specks": a scattering of small alike marks, found
#: together with spot_detect and written with spot_write (_speck_image).
WAYS = ("objects", "specks")
#: Specks, without a trial: the person's objects a median of less than this
#: share of their picture's long side (teacher_band's object_side_pct). On
#: the whole picture fitted to the window that is under some 20 px across --
#: about where teacher_band's own guess calls one too small to box (16 px).
#: Small at Fit is not a speck by itself: five bolts 150 px across on a
#: 12000 px frame are 1.25% of it, and a closer look boxes them. So also more
#: than WAY_FEW a picture -- a scattering -- and no more than
#: WAY_SPECK_NATIVE_PX across in the picture's own pixels; anything else
#: small is measured both ways.
WAY_SPECK_SIDE_PCT = 1.5
#: spot_detect scores marks at two fixed scales of the picture's own pixels
#: (sigma 1.5 and 3 px: a mark some 4 to 8 px across answers best), so an
#: object many times that is outside both, however small it comes out at
#: Fit. A person's brush paints round a speck generously: some four times
#: the larger scale's mark. teacher_band's object_size, the median object.
WAY_SPECK_NATIVE_PX = 32
#: Objects, without a trial: at least this share -- some 54 px across at Fit,
#: one to box -- and no more than WAY_FEW a picture. Everything between is
#: measured both ways: a crowd of large ones (45 cases a frame were found 45
#: of 45 with boxes at Fit), or of ones small at Fit and large in their own
#: pixels, which spot_detect does not see as marks.
WAY_OBJECT_SIDE_PCT = 4.0
WAY_FEW = BOXES_PER_LOOK // 2
#: The estimate for a walk of specks before this run has timed one:
#: spot_detect and spot_write on one image, and, once a run, the detector's
#: measurement of the person's own marks -- some 20 s on a 4K frame.
SPECK_IMAGE_SECONDS_DEFAULT = 8.0
SPECK_CALIBRATION_SECONDS = 25.0

#: The zooms one is chosen from, lowest first: Fit, then the bridge's rungs
#: of the picture's own pixels (mcp_view.RUNGS), as the zoom slider counts
#: them. A copy, held equal by a test: the walk does not import the bridge.
RUNGS = ("fit", "50%", "100%", "200%", "400%")
#: What a zoom is to find of the person's own to be chosen, both of them:
#: their objects found, and of what was kept, the share on one of theirs. A
#: box on a part with none of theirs is one on nothing.
RECALL_MIN = 0.8
PRECISION_MIN = 0.8
#: The person's pictures a zoom is rehearsed on, at most: not those its
#: examples are made from, Fit's and the zoom's own -- a model shown its own
#: answer is not measured -- unless every image they drew is one an example
#: is made of. One that cannot be planned at the zoom -- none of the class
#: asked for on it -- gives way to the next of theirs.
REHEARSE_TEACHERS = 3
#: The looks one zoom's rehearsal may come to: on each of those pictures, the
#: bridge's parts with their objects and without (mcp_view's
#: REHEARSE_OBJECT_VIEWS, REHEARSE_EMPTY_VIEWS: 3 and 2) and the whole
#: picture after them. A zoom whose plan comes to more -- from a bridge that
#: hands every part, 255 of them on a large frame at 400% -- is not
#: rehearsed, and that is said.
REHEARSE_LOOKS_MAX = REHEARSE_TEACHERS * (3 + 2 + 1)

#: What the screen answers a question with when it stood unanswered
#: (routers/agent_run.py): not the person's reply. Read as one, it was a no,
#: and the run ended quoting the screen's own words back as theirs. A copy,
#: held equal by a test: the walk does not import the route, which imports it.
NOBODY_ANSWERED = ("Nobody answered. Carry on with what you judge best, and say in your "
                   "report what you assumed and which images it affects.")

#: The key of the bridge's picture of what spot_detect found, each find
#: ringed (the loop's SHEET_KEY). A copy, held equal by a test: the walk does
#: not import the loop.
SHEET_KEY = "candidates_jpeg"

#: How every coordinate of the walk is said: 0 to 1000 across the picture
#: the model is shown, each axis by its own side, whatever the picture's size
#: in pixels -- the bridge's units='norm1000'. It is how the vision model
#: answers whatever it is asked: told to read its boxes off a ruler in the
#: picture's pixels, it answered on a 0-1000 scale all the same, and every
#: box read as pixels landed somewhere else.
UNITS = "norm1000"
NORM = 1000
#: Said in every question of the walk, after what to look for. The most
#: boxes one answer is taken with (BOXES_PER_LOOK) is said too: on a crowded
#: picture at Fit, an answer listing every object ran past the room an answer
#: has (ANSWER_TOKENS), and was cut off mid-list.
COORDS = ("Give each box as [x0, y0, x1, y1] in coordinates from 0 to 1000 across this picture (0,0 is its "
          "top left corner, 1000,1000 its bottom right), whatever its size in pixels; sizes said with the "
          f"picture are on the same scale. At most {BOXES_PER_LOOK} boxes. ")
#: The shape of the answer, said in words as well as held to by the server
#: (FOUND, CHECKED): a server that takes no schema is asked plainly, and told
#: only to "give an empty list", a model answered with the list alone --
#: [[x0, y0, x1, y1], ...] or [] -- and no object round it. The braces are
#: doubled: the briefs go through str.format.
ANSWER_FOUND = ('Answer with the JSON only: {{"boxes": [[x0, y0, x1, y1], ...]}} -- an empty list, '
                '{{"boxes": []}}, when there is none.')
ANSWER_CHECKED = ('Answer with the JSON only: {{"missed": [[x0, y0, x1, y1], ...], "wrong": [[x0, y0, x1, y1], '
                  '...]}} -- each an empty list, [], when there is none.')

#: What the model is told in each conversation about one part. There is no
#: "box EVERY object": most parts hold none, and a model told that every
#: picture has objects to box finds some.
WALK_BRIEF = (
    "You look at one part of a larger picture, at {rung}: the picture is looked at part by part, and "
    "each part is asked about on its own. Look for: {target}. Box each one you see WHOLE in this part. "
    + COORDS + "Anything cut by the edge of this part is taken whole in the neighbouring part: leave it out. "
    "Most parts hold none of them. Do not box what you are unsure of. " + ANSWER_FOUND)
#: The same, for a picture looked at whole: its own Fit is as close as the zoom.
#: It says that a picture may hold none, as the brief for a part does. It did
#: not, and the example shown before each picture always holds one: on an
#: inspection set, where many pictures are clean, the model boxed whatever
#: looked most like the target. Measured on the tutorial sample with one image
#: labelled, a black spot on a contact pad: of 16 pictures with no spot, 11
#: had a mounting hole boxed and written; with this sentence, 9. It is a
#: default, not a cure -- what brought it to 1 was the person saying in the
#: instruction what is not a target ("the round holes are not spots"), which
#: reaches this brief as {target}.
WALK_BRIEF_FIT = (
    "You look at a whole picture. Look for: {target}. Some pictures hold none of them: the answer for "
    "those is no boxes. Box each one you see. " + COORDS + "Do not box what you are unsure of. "
    + ANSWER_FOUND)
#: The whole picture after its parts, for what is too large to be whole in
#: any part: the parts' overlap stops at half the window, and an object larger
#: than that is cut in every part that holds it, and skipped in each.
WALK_BRIEF_LARGE = (
    "You look at a whole picture, shrunk to fit. Look for: {target} -- only those as large as the "
    "message says. Smaller ones are looked for closer, part by part: leave them out. Box each large one "
    "you see. " + COORDS + "Most pictures hold none that large. " + ANSWER_FOUND)
#: The review of a part after its image was written.
REVIEW_BRIEF = (
    "You check a mask just written on a picture. What is not mask is dimmed, and the edge of the mask "
    "is drawn in white. The objects labelled: {target}. Look for two things. An object that is there "
    "with no mask on it: a box round it in missed. A mask on something that is not one of them, or one "
    "taking in much more than its object: a box round that mask in wrong. " + COORDS + "An object cut by "
    "the edge of a part is checked in the neighbouring part, and one too large to lie whole in any part on "
    "the whole picture: leave those out. If the mask is right, both lists are empty. " + ANSWER_CHECKED)
#: The review of the whole picture, for what was taken from it: an object too
#: large for any part is cut by the edge of every part that holds it, and the
#: review of the parts leaves it out in each of them.
REVIEW_BRIEF_LARGE = (
    "You check the masks just written on a picture, shown whole and shrunk to fit. What is not mask is "
    "dimmed, and the edge of the mask is drawn in white. The objects labelled: {target} -- only those as "
    "large as the message says: smaller ones are checked closer, part by part, so leave them out. A large "
    "one with no mask on it: a box round it in missed. A large mask on something that is not one of them, "
    "or taking in much more than its object: a box round it in wrong. " + COORDS + "If they are right, "
    "both lists are empty. " + ANSWER_CHECKED)

#: One point inside one mark, for spot_detect to measure that mark and find
#: the rest like it, on a project of specks nobody has drawn on. A box round
#: each is not asked: there are too many, too small, and the detector finds
#: them together from one.
SPECK_POINT_BRIEF = (
    "You look at a whole picture. Look for: {target} -- a scattering of small marks. Give the middle of ONE of "
    "them that you see clearly, as [x, y] in coordinates from 0 to 1000 across this picture (0,0 is its top "
    "left corner, 1000,1000 its bottom right), whatever its size in pixels: a program finds the others like "
    "it. Do not point at what you are unsure of. "
    'Answer with the JSON only: {{"point": [x, y]}} -- an empty list, {{"point": []}}, when there is none.')
#: The look at an image after spot_write, on spot_detect's own picture of
#: what it found: each find ringed, close-ups beside it. That picture has no
#: coordinates a miss could be boxed in, and nothing would write one: right
#: or not, and not right is the image flagged for a person.
SPECK_REVIEW_BRIEF = (
    "You check what a program found on a picture: each find is ringed -- or outlined in white, with the rest "
    "dimmed -- and close-ups of parts of the picture may be beside or under it. Asked for: {target}. Are the "
    "marked ones those, with few of them left unmarked? "
    'Answer with the JSON only: {{"right": true}} or {{"right": false}}.')

#: Said before the person's example and before the picture asked about after
#: it, when the walk shows the model one of the person's pictures first
#: (teacher_example): that picture with where it is said in its own words,
#: then its answer as the model's own -- the boxes the person's labels give
#: it -- then the picture asked about. The target said in words alone told
#: the model too little of what one object is: measured at Fit, it found none
#: of 23 small cases and none of 36 thin boards under the loads they carry;
#: with one example before each picture, all 23 and 28 to 32 of the 36, for 3
#: to 5 s more a look. That was measured on whole pictures alone. Before each
#: part (a part of theirs at the same zoom), and before the whole picture
#: after the parts (_overview), it is the same question asked closer, and
#: has not been measured there.
EXAMPLE_SAID = "An example the person labelled, with the answer for it. "
NOW_SAID = "Now this picture. "

#: What each question of the walk is answered with: a JSON object of this
#: shape, the server held to it (Backend.chat_json). Asked to answer with a
#: tool call, the model answered most busy pictures with a description of
#: them and no call, read as nothing found; held to the object, it answered
#: every one. No other keys: a strict json_schema (api.openai.com's) refuses
#: an object schema that does not say so.
_BOXES = {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}
FOUND = {"type": "object", "properties": {"boxes": _BOXES}, "required": ["boxes"], "additionalProperties": False}
CHECKED = {"type": "object", "properties": {"missed": _BOXES, "wrong": _BOXES}, "required": ["missed", "wrong"],
           "additionalProperties": False}
#: A point (SPECK_POINT_BRIEF), and the look at specks written (SPECK_REVIEW_BRIEF).
POINT = {"type": "object", "properties": {"point": {"type": "array", "items": {"type": "number"}, "maxItems": 2}},
         "required": ["point"], "additionalProperties": False}
RIGHT = {"type": "object", "properties": {"right": {"type": "boolean"}}, "required": ["right"],
         "additionalProperties": False}
#: A point in an answer's words: two numbers in brackets (_answer_of).
_POINT_SAID = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
#: A box in an answer's words: four numbers in brackets (_answer_of).
_BOX_SAID = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
                       r"\s*(-?\d+(?:\.\d+)?)\s*\]")

#: How a reply to one of the walk's questions is read when it names no choice
#: by its number or its label. It is go ahead when it says so and says
#: nothing against it, stop when it says that and not go ahead, and anything
#: else is asked once more: the consent is the person's to give. Read word by
#: word against a short list, "yes please", the same in Japanese and "no need
#: to ask, go ahead" were each read as a no, and the run ended with nothing
#: labelled; "proceed, I choose 100%" was read as choosing the zoom again, and
#: took the filing back. Choosing again is a label or its number, nothing less.
#: Said first, these mean go ahead.
_YES_FIRST = re.compile(r"(yes|y|yeah|yep|ok|okay|sure|fine|please)(?![a-z])|" + "\u306f\u3044|\u3046\u3093|\u3048\u3048|\u4e86\u89e3|\u627f\u77e5")
#: Anywhere in a reply, these do too.
_GO_WORDS = re.compile(r"(?<![a-z])(proceed|go ahead|go on|carry on|continue|keep going|do it|please do)"
                       r"(?![a-z])|" + "\u9032\u3081\u3066|\u9032\u3081\u308b|\u9032\u3081\u307e|\u7d9a\u3051\u3066|\u7d9a\u3051\u308b|\u304a\u9858\u3044|\u3088\u308d\u3057\u304f|\u3069\u3046\u305e")
#: Phrases that hold a word for no and mean go ahead. Taken out before the
#: rest is read, and read as a yes.
_GO_PHRASES = re.compile(r"(?<![a-z])(no need to (ask|check|confirm|wait)|no problem|no worries|"
                         r"(do not|don't|dont) (ask|stop|wait)|without (asking|stopping))(?![a-z])"
                         "|(\u3084\u3081|\u6b62\u3081)(\u305a|\u306a\u3044\u3067|\u306a\u304f\u3066)|\u4e2d\u6b62(\u305b\u305a|\u3057\u306a\u3044\u3067)|\u805e\u304b\u306a\u3044\u3067|\u78ba\u8a8d\u3057\u306a\u3044\u3067|\u78ba\u8a8d\u305b\u305a|\u78ba\u8a8d\u306a\u3057"
                         "|\u69cb\u3044\u307e\u305b\u3093|\u304b\u307e\u3044\u307e\u305b\u3093|\u69cb\u308f\u306a\u3044|\u554f\u984c\u3042\u308a\u307e\u305b\u3093|\u554f\u984c\u306a\u3044")
_STOP_WORDS = re.compile(r"(?<![a-z])(stop|cancel|no(?!\s+(need|problem|worries)))(?![a-z])|"
                         "\u3084\u3081|\u6b62\u3081|\u4e2d\u6b62|\u3044\u3044\u3048")
#: The same without "no": answered "no" to "keep it, take it back, or
#: stop?", a person may mean either of the other two.
_STOP_ONLY = re.compile(r"(?<![a-z])(stop|cancel)(?![a-z])|" + "\u3084\u3081|\u6b62\u3081|\u4e2d\u6b62")
#: Words that turn round what they stand by. A reply holding one says
#: neither go ahead nor anything else plainly: "I don't want to proceed".
_NOT_WORDS = re.compile(r"(?<![a-z])(not|never|no need|dont|cant|wont)(?![a-z])|n't(?![a-z])"
                        "|\u306a\u3044|\u307e\u305b\u3093|\u305a\u306b|\u305b\u305a")
#: For confirm_each: keep the image as written, or take it back.
_KEEP_WORDS = re.compile(r"(?<![a-z])(keep|leave it)(?![a-z])|" + "\u3053\u306e\u307e\u307e|\u305d\u306e\u307e\u307e|\u6b8b\u3057\u3066|\u6b8b\u3059")
_BACK_WORDS = re.compile(r"(?<![a-z])(take (it )?back|undo|remove it|erase it)(?![a-z])"
                         "|\u53d6\u308a\u6d88|\u6d88\u3057\u3066|\u6d88\u3059|\u623b\u3057\u3066|\u5143\u306b\u623b")
#: For the way of a project nobody has drawn on (_way_asked): distinct
#: objects, or a scattering of small marks.
_OBJECT_WORDS = re.compile(r"(?<![a-z])(objects?|distinct|separate|box(es)?)(?![a-z])|" + "\u7269\u4f53|\u500b\u3005|\u500b\u5225|\u3072\u3068\u3064\u305a\u3064")
_SPECK_WORDS = re.compile(r"(?<![a-z])(specks?|marks?|dots?|spots?|dust|scatter(ed|ing)?|pinholes?)(?![a-z])|"
                          + "\u70b9|\u7c92|\u307b\u3053\u308a|\u57c3|\u6563\u3089")
#: What is let go of round a reply or a label before they are compared.
_ENDS = "\"'`()[]{}<>.,!?;:\u3001\u3002\u300c\u300d\u300e\u300f\u30fb"
#: An image's file name, given where its id is wanted.
_ID_EXT = re.compile(r"\.(png|jpe?g|bmp|tiff?|webp)$", re.I)


class _Ended(Exception):
    """The walk ends here: stopped, declined, or unable to go on."""

    def __init__(self, outcome: str, said: str = "") -> None:
        super().__init__(outcome)
        self.outcome, self.said = outcome, said


@dataclass
class Walk:
    """What the walk needs of the run, and the run's books it keeps."""

    c: Any                                  # the bridge
    bk: Any                                 # the model server
    rules: str                              # the bridge's data rules, first in every conversation
    project_id: str
    lang: str
    instruction: str
    emit: Any                               # emit(event)
    gate: Any                               # await gate(): True when the run is to stop; holds a pause
    ask: Any                                # await ask(question) -> reply, or None: nobody to ask
    ask_gate: Any                           # the loop's ask, which stops asking after "do them all"
    no_more_questions: dict
    text: Any                               # the loop's reading of a tool's answer (_text)
    render: Any                             # the loop's small picture for the screen (_render)
    wants_no_more_questions: Any
    plan: dict                              # the project's view_plan that turned the walk on
    written: list
    done_with: set
    reviewed: set
    ever_flagged: set
    handed_over: dict
    images_seen: dict
    seen_crop: dict
    kept_high: dict
    progress: Any
    answers: list
    #: Images the run took the review flag off to redo (the loop's books): one
    #: the walk writes or flags again is redone.
    redo: set = field(default_factory=set)
    #: The images the run is for, when its caller named them (the command
    #: line does): the walk labels those alone, not the rest of the project.
    only: frozenset = frozenset()
    assume_consent: bool = False
    confirm_each: bool = False
    review_redos: int = 2
    prepared: bool = False
    #: The images the person said to go ahead with. A later call for others
    #: asks again: a yes to one image is not a yes to the project.
    consented_ids: set = field(default_factory=set)
    #: Every image of the walk's plan has been dealt with: the rest are a person's.
    covered: bool = False
    looks: list = field(default_factory=list)       # seconds per picture asked about
    failed_in_a_row: int = 0
    #: accept_masks calls in a row that SAM could not run for (_accept).
    sam_failed_in_a_row: int = 0
    unplanned_in_a_row: int = 0                     # images the bridge could not plan, in a row
    results: list = field(default_factory=list)     # (item_id, what became of it, stats)
    #: Who walks the parts (DRIVERS). The rest is the model driver's place in
    #: its walk, kept between the model's calls: the images still to walk, in
    #: order, how many there were, the one it is in now (_Open), the class it
    #: is written as and the classes the project has (None until read), the
    #: loop's turn the call in hand is from, and how many turns of the run's
    #: the walk has been allowed beyond its step limit.
    driver: str = "harness"
    begun: bool = False
    queue: list = field(default_factory=list)
    walk_total: int = 0
    open: Any = None
    class_id: int | None = None
    known_classes: frozenset | None = None
    turn: int = 0
    model_turns: int = 0
    #: The person's worked example for each rung the harness asks at and the
    #: class it writes, (rung, class_id) -> {"b64", "boxes", "words",
    #: "item_id"} (_examples), or None where there is none to show -- nobody
    #: drew, the bridge gave none, or the model server would not take it.
    #: Fetched once a run.
    examples: dict = field(default_factory=dict)
    #: The model server answered a picture only without the person's example
    #: (_without_examples): none is fetched or shown for the rest of the run.
    examples_refused: bool = False
    #: The person's images the screen was told the examples are made of
    #: (_examples): not again for every rung a rehearsal fetches one for, but
    #: again for a rung whose example is another of theirs -- said once, at
    #: Fit, the image of the zoom chosen after it went unnamed.
    example_said: set = field(default_factory=set)
    #: The zoom this run chose by rehearsing on the person's own (_decide_rung):
    #: {"rung", "recall", "precision", "passed", "filed", "on"}, or {"failed":
    #: why} when it could not be rehearsed -- tried once a run either way.
    rehearsed: dict | None = None
    #: How the project is labelled, chosen on this run (decide_way): what was
    #: said of it, and the teachers as read for it (teacher_band, the shrink
    #: not measured), which a walk of specks needs no more of.
    way_said: str = ""
    band: dict | None = None
    #: A walk of specks: the person's image whose marks are the example
    #: (spot_detect like_item_id), and what spot_detect measured of them there.
    like: str = ""
    like_score: dict | None = None

    def say(self, ja: str, en: str) -> str:
        return ja if self.lang == "ja" else en


async def engages(c, project_id: str, text, lang: str = "en",
                  driver: str = "harness") -> tuple[dict | None, str]:
    """(the project's walk plan, or None: the old way runs), and why not, where that is worth saying.

    What a run does with the plan is walks()'s: a plan whose way is None --
    nothing filed says how the project is labelled -- is walked once the way
    is chosen (decide_way), before anything else, and one marked "untaught"
    once the person has drawn an image (teach_first). A bridge that cannot
    plan the project runs the old way.
    """
    try:
        plan = text(await c.call_tool("view_plan", {"project_id": project_id}))
    except Exception:                                    # noqa: BLE001 - an older bridge, or none: the old way
        return None, ""
    if not isinstance(plan, dict) or plan.get("error"):
        return None, ""
    return walks(plan, lang, driver)


def _way_filed(plan: dict) -> str | None:
    """How the project is filed as labelled: "objects", "specks", or None where nothing says."""
    if plan.get("specks") or plan.get("way") == "specks":
        return "specks"
    return "objects" if plan.get("way") == "objects" else None


def walks(plan: dict, lang: str = "en", driver: str = "harness") -> tuple[dict | None, str]:
    """What a run does with the project's plan: (the plan it walks, or None: the old way), and why not.

    Objects are walked at the zoom filed. With none, such a plan carries a
    filing of its own, {"rung": "fit", "default": True}: the program, once it
    is asked to label, first chooses the zoom by rehearsing on the person's
    own, Fit first (label_images, _decide_rung) -- measured, a model shown
    the whole picture fitted to the window found every object of a crowded
    frame in seconds, where closer looks took longer and lost some -- and the
    model walking the parts looks at each image once, whole, at Fit. Specks
    are walked an image at a time, whole, whatever zoom is
    filed: the detector finds them on the picture itself. A plan whose way
    is None comes back with it None, to be chosen (decide_way) and passed
    here again.

    The old way runs, and says so, where the walk does not go yet: specks
    with the model walking (driver "model" walks parts, and specks are found
    whole), and a zoom filed without teachers on a project since drawn on,
    which needs the untaught labelling that comes with choosing one. A run
    labelling the old way with nothing on the screen about it looks like one
    that is not using what was chosen for it.

    Objects on a project nobody has drawn on come back marked "untaught":
    the walk measures SAM and the band on the person's own, and so do the
    steps before labelling of the old way, which on such a project could
    write nothing. The person is asked for one image first (teach_first).
    """
    ja = lang == "ja"
    filed = plan.get("filed")
    zoom = isinstance(filed, dict) and bool(filed.get("rung")) and not filed.get("default")
    rung = filed["rung"] if zoom else "fit"
    default = {"rung": "fit", "default": True, "taught": bool(plan.get("taught"))}
    way = _way_filed(plan)
    if way == "specks":
        if driver == "model":
            return None, ("\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u3067\u306f\u3001\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9\u3068\u3057\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002\u70b9\u306f\u90e8\u5206\u3054\u3068\u306b\u898b\u3066\u56de\u3089\u305a\u3001spot_detect \u3067\u753b\u50cf\u5168\u4f53\u304b\u3089\u63a2\u3059\u305f\u3081\u3001"
                          "\u30e2\u30c7\u30eb\u304c\u898b\u3066\u56de\u308b\u8a2d\u5b9a\u306e\u3053\u306e\u5b9f\u884c\u3067\u306f\u3001\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002" if ja else
                          "This project is labelled as a scattering of small marks, which are found on the whole "
                          "picture with spot_detect, not walked part by part: with the model walking, this run "
                          "labels them the way it did before")
        said = ""
        if zoom:
            said = (f"\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u500d\u7387 {rung} \u304c\u8a18\u9332\u3055\u308c\u3066\u3044\u307e\u3059\u304c\u3001\u70b9\u3068\u3057\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308b\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306a\u306e\u3067\u3001"
                    "\u3053\u306e\u5b9f\u884c\u3067\u306f\u90e8\u5206\u3054\u3068\u306b\u898b\u305a\u3001spot_detect \u3067\u5404\u753b\u50cf\u306e\u5168\u4f53\u304b\u3089\u63a2\u3057\u307e\u3059\u3002" if ja else
                    f"{rung} is filed for this project, but it is labelled as specks, which are found on each "
                    "whole image with spot_detect: this run labels them that way, not part by part")
        return {**plan, "filed": default, "way": "specks", "specks": True}, said
    if zoom:
        if filed.get("taught") is False and not plan.get("taught"):
            return {**plan, "way": "objects", "untaught": True}, ""
        if filed.get("taught") is False:
            return None, (f"\u500d\u7387 {rung} \u306f\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u306a\u3044\u72b6\u614b\u3067\u8a18\u9332\u3055\u308c\u3066\u3044\u307e\u3059\u3002\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u306a\u3057\u3067\u90e8\u5206\u3054\u3068\u306b\u898b\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308b\u6a5f\u80fd\u306f\u307e\u3060\u306a\u3044\u305f\u3081\u3001"
                          "\u3053\u306e\u5b9f\u884c\u3067\u306f\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002" if ja else
                          f"{rung} was filed for this project without teachers, and labelling part by part "
                          "without teachers is not here yet: this run labels the way it did before")
        return {**plan, "way": "objects"}, ""
    if way is None:
        return {**plan, "filed": default, "way": None}, ""
    if not plan.get("taught"):
        return {**plan, "filed": default, "way": "objects", "untaught": True}, ""
    if str(plan.get("rung") or "fit").lower() != "fit":
        return None, ""
    return {**plan, "filed": {**default, "taught": True}, "way": "objects"}, ""


async def decide_way(w: Walk) -> dict | None:
    """How the project is labelled, where nothing is filed: chosen first, filed, and said.

    On a project the person has drawn on, by their own: obvious from what
    teacher_band reads of their objects -- a scattering of marks too small
    to box at Fit and small in their own pixels are specks; a few large ones
    are objects -- and otherwise measured both ways on their images
    (_way_measured). On one nobody has drawn on, the person is asked, once
    (_way_asked). Filed with the project (way_choose) for the runs after it,
    and said on the screen with the numbers it rests on. Before the question
    before walking, whose estimate depends on it.

    Not on a run with no image to label in its scope: every image has a mask,
    or those the run is for do. Nothing is spent on choosing -- a spot_detect
    measurement, SAM and a look -- and nobody is asked about a way for a run
    that labels nothing; the old way runs, and the next run that labels
    chooses.

    None when the run goes on: w.plan's way chosen -- or left None where it
    could not be (said), and the run labels the old way. Otherwise the end
    of the run, as label_images returns one: stopped, declined, or nobody to
    say.
    """
    if not _left_in_scope(w):
        return None
    try:
        if _taught(w):
            way, said, why = await _way_from_teachers(w)
        else:
            way, said, why = await _way_asked(w)
    except _Ended as end:
        return _out(w, end.outcome, "fit", end.said)
    except Exception as exc:                             # noqa: BLE001 - said, and the old way runs
        way, why = "", ""
        said = w.say(f"\u30e9\u30d9\u30eb\u306e\u4ed8\u3051\u65b9\u3092\u6c7a\u3081\u3089\u308c\u306a\u304b\u3063\u305f\u305f\u3081\uff08{exc!r}\uff09\u3001\u3053\u306e\u5b9f\u884c\u3067\u306f\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002"[:300],
                     f"How this project is labelled could not be chosen ({exc!r}): this run labels the way it "
                     "did before"[:300])
    if not way:
        w.emit({"type": "auto", "text": said})
        return None
    if why:
        got = await _call(w, "way_choose", {"way": way, "why": why[:300]})
        if got.get("error"):
            # Chosen for this run all the same; the next one chooses again.
            err = " ".join(str(got["error"]).split())[:160]
            said = said.replace(_WAY_FILED[0 if w.lang == "ja" else 1],
                                w.say(f"\u8a18\u9332\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff08{err}\uff09\u3002", f" Not filed: {err}."))
    w.plan = {**w.plan, "way": way, "specks": way == "specks"}
    w.way_said = said
    w.emit({"type": "auto", "text": said})
    return None


#: How many times a project nobody has drawn on is asked about (teach_first):
#: once, once more after a reply that says neither answer plainly, and once
#: more after "I have labelled one" with none found -- a mask still being
#: saved, or painted in another project.
TEACHER_ASKS = 3


async def teach_first(w: Walk) -> dict | None:
    """Objects on a project nobody has drawn on: the person is asked for one image of theirs, first.

    A run measures SAM and the size of an object on masks a person drew
    (teacher_band, calibrate_sam) and checks itself against them (rehearse).
    The old way reads the same in its steps before labelling, so a run
    handed to it on such a project wrote nothing: the model put the same
    question, in its own words, until the run gave up. So the program says
    what is missing and asks, once, before the model says anything: an image
    labelled by hand while the question waits is read (view_plan) and the
    run goes on from it. Stop, no answer, or none found after being asked
    again end the run, which says why nothing was labelled.

    Not on a run with nothing to label in its scope, and not where nobody
    can be asked (no ask, or the command line's --yes): said on the screen,
    and the run goes on without labelling.

    None when the run goes on: w.plan read again and taught, or left
    untaught where nobody was asked. Otherwise the end of the run, as
    label_images returns one.
    """
    if not _left_in_scope(w):
        return None
    try:
        await _teacher_asked(w)
    except _Ended as end:
        return _out(w, end.outcome, "fit", end.said)
    return None


async def _teacher_asked(w: Walk) -> None:
    """teach_first's question; w.plan read again once the person has drawn one."""
    why = w.say(
        "\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306f\u7269\u4f53\u3068\u3057\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u304c\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u307e\u3060\u3042\u308a\u307e\u305b\u3093\u3002AI \u30b5\u30dd\u30fc\u30c8\u306f\u3001"
        "\u4eba\u304c\u63cf\u3044\u305f\u30de\u30b9\u30af\u3092\u4f7f\u3063\u3066 SAM \u3068\u7269\u4f53\u306e\u5927\u304d\u3055\u3092\u6e2c\u308a\u3001\u898b\u3064\u3051\u305f\u3082\u306e\u3082\u305d\u306e\u30de\u30b9\u30af\u3068\u7167\u3089\u3057\u5408\u308f\u305b\u308b\u305f\u3081\u3001"
        "\u4eba\u304c\u624b\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c1\u679a\u5fc5\u8981\u3067\u3059",
        "This project is labelled as objects, and nobody has drawn on it. A run measures SAM and the size of "
        "an object on masks a person drew, and checks what it finds against them, so it needs one image "
        "labelled by hand")
    if w.ask is None or w.assume_consent:
        w.emit({"type": "auto", "text": why + w.say(
            "\u3002\u3053\u306e\u5b9f\u884c\u3067\u306f\u5c0b\u306d\u308b\u76f8\u624b\u304c\u3044\u306a\u3044\u305f\u3081\u3001\u4f55\u3082\u66f8\u304d\u8fbc\u307f\u307e\u305b\u3093\u30021\u679a\u306e\u753b\u50cf\u3067\u5bfe\u8c61\u3092\u3059\u3079\u3066\u5857\u3063\u3066\u304b\u3089\u3001"
            "\u3082\u3046\u4e00\u5ea6\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
            ". Nobody is asked on this run, so nothing is written on it: paint every target on one image, "
            "then run it again")})
        return
    choices = [w.say("\u63cf\u304d\u307e\u3057\u305f", "I have labelled one"), w.say("\u3084\u3081\u308b", "Stop")]
    text = why + w.say("\u3002\u300c\u30e9\u30d9\u30eb\u300d\u30bf\u30d6\u30671\u679a\u306e\u753b\u50cf\u306e\u5bfe\u8c61\u3092\u3059\u3079\u3066\u5857\u3063\u3066\u304b\u3089\u3001\u3053\u3053\u3067\u7b54\u3048\u3066\u304f\u3060\u3055\u3044\u3002",
                       ". Paint every target on one image in the Annotate tab, then answer "
                       "here.") + _numbered(w, choices)
    asked, looked = text, False
    for _ in range(TEACHER_ASKS):
        w.emit({"type": "question", "item_id": None, "text": asked, "choices": choices})
        reply = str(await w.ask(asked))
        if await w.gate():
            raise _Ended("stopped")
        if reply == NOBODY_ANSWERED:
            raise _Ended("nobody", w.say(
                "\u624b\u30671\u679a\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u307b\u3057\u3044\u3068\u3044\u3046\u304a\u9858\u3044\u306b\u8ab0\u3082\u7b54\u3048\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
                "Nobody answered the question about labelling one image by hand, so nothing was labelled."))
        w.progress.moved("the person answered")
        w.answers.append((asked, reply))
        got = _consent_of(w, reply, choices, ("proceed", "stop"))
        if got == "stop":
            raise _Ended("declined", w.say(
                "\u6307\u793a\u3069\u304a\u308a\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305a\u306b\u6b62\u3081\u307e\u3057\u305f\u3002\u624b\u30671\u679a\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308c\u3070\u3001\u6b21\u306e\u5b9f\u884c\u306f\u305d\u3053\u304b\u3089\u9032\u3081\u3089\u308c\u307e\u3059\u3002",
                "Stopped as told, with nothing labelled. Once one image is labelled by hand, a run goes on "
                "from it."))
        if got is None:
            asked = _again_please(w, reply, text)
            continue
        plan = await _call(w, "view_plan", {})
        if plan.get("error"):
            raise _Ended("failed", w.say(
                f"\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u3092\u8aad\u307f\u76f4\u305b\u307e\u305b\u3093\u3067\u3057\u305f\uff08{plan['error']}\uff09\u3002\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
                f"The project could not be read again ({plan['error']}). Nothing was labelled."))
        if plan.get("taught"):
            w.plan = {**plan, "way": _way_filed(plan) or "objects"}
            w.emit({"type": "auto", "text": w.say(
                "\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u78ba\u8a8d\u3057\u307e\u3057\u305f\u3002\u3053\u306e\u753b\u50cf\u3092\u3082\u3068\u306b\u9032\u3081\u307e\u3059\u3002",
                "An image labelled by hand is in the project now: the run goes on from it.")})
            return
        looked = True
        asked = w.say("\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u307e\u3060\u898b\u3064\u304b\u308a\u307e\u305b\u3093\uff08\u4f55\u3082\u5857\u3089\u308c\u3066\u3044\u306a\u3044\u30de\u30b9\u30af\u306f\u6570\u3048\u307e\u305b\u3093\uff09\u3002",
                      "No image labelled by hand is found in this project yet -- a mask with nothing painted "
                      "in it does not count. ") + text
    if looked:
        raise _Ended("declined", w.say(
            "\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u898b\u3064\u304b\u3089\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u30021\u679a\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u304b\u3089\u3001"
            "\u3082\u3046\u4e00\u5ea6\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
            "No image labelled by hand was found in this project, so nothing was labelled. Label one, then "
            "run this again."))
    quoted = " ".join(reply.split())[:120]
    raise _Ended("declined", w.say(
        f"\u7b54\u3048\uff08\u300c{quoted}\u300d\uff09\u304c\u3069\u3061\u3089\u3092\u9078\u3093\u3060\u306e\u304b\u5206\u304b\u3089\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
        f"The answer (\"{quoted}\") said neither, so nothing was labelled."))


def _left_in_scope(w: Walk) -> bool:
    """Whether the plan has an image with no mask among those the run is for (Walk.only), as _scope counts them."""
    left = [str(e["id"]) for e in w.plan.get("unlabelled") or [] if isinstance(e, dict) and e.get("id")]
    if w.only:
        keys = _keys(w.only)
        left = [i for i in left if i in keys]
    return bool(left)


#: How the line that says the way chosen ends, when it is filed (ja, en).
_WAY_FILED = ("\u8a18\u9332\u3057\u305f\u306e\u3067\u3001\u6b21\u56de\u4ee5\u964d\u306e\u5b9f\u884c\u3067\u3082\u4f7f\u3044\u307e\u3059\u3002", " Filed for the runs after this one.")


def _way_line(w: Walk, way: str, ja: str, en: str, filed: bool = True,
              tail: tuple[str, str] | None = None) -> str:
    """What the screen is told of the way chosen, with what it rests on; tail (ja, en): why it is not filed."""
    if way == "specks":
        head = w.say("\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u3067\u306f\u3001\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9\u3068\u3057\u3066 spot_detect \u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059",
                     "This project is labelled as a scattering of small marks, found with spot_detect")
    else:
        head = w.say("\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u3067\u306f\u3001\u7269\u4f53\u3068\u3057\u30661\u3064\u305a\u3064\u56f2\u307f\u3001SAM \u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059",
                     "This project is labelled as objects, each boxed and handed to SAM")
    tail = (w.say(*tail) if tail else w.say(*_WAY_FILED) if filed else
            w.say("\u8a18\u9332\u306f\u3057\u307e\u305b\u3093\u3002", " Nothing is filed."))
    return w.say(f"{head}\u3002{ja}\u3002{tail}", f"{head}: {en}.{tail}")


async def _way_from_teachers(w: Walk) -> tuple[str, str, str]:
    """(way, what is said of it, why it is filed) from the person's own; ("", why not, "") where it cannot say."""
    if await w.gate():
        raise _Ended("stopped")
    # Not the shrink: measured with SAM on a dozen of their objects, it is
    # what a walk of objects reads again before it starts (_prepare).
    band = await _call(w, "teacher_band", {"calibrate_shrink": False})
    if band.get("error"):
        why = " ".join(str(band["error"]).split())[:160]
        return "", w.say(f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u3081\u306a\u304b\u3063\u305f\u305f\u3081\uff08{why}\uff09\u3001\u30e9\u30d9\u30eb\u306e\u4ed8\u3051\u65b9\u3092\u6c7a\u3081\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u3053\u306e\u5b9f\u884c\u3067\u306f\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002",
                         f"The teachers could not be read ({why}), so how this project is labelled could not be "
                         "chosen: this run labels the way it did before"), ""
    w.band = band
    counts = sorted(int(n) for n in band.get("objects_per_frame") or []
                    if isinstance(n, int) and not isinstance(n, bool))
    per = counts[(len(counts) - 1) // 2] if counts else 0
    side = band.get("object_side_pct")
    side = float(side) if isinstance(side, (int, float)) and not isinstance(side, bool) else None
    native = _native_side(band)
    seen_en = f"the person's objects are {per} a picture" + (
        f", a median {side:g}% of their picture's long side" if side is not None else "") + (
        f" and {native} px across in its own pixels" if native is not None else "")
    seen_ja = f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u7269\u4f53\u306f1\u679a\u306b{per}\u500b" + (f"\u3001\u9577\u8fba\u306f\u753b\u50cf\u306e\u9577\u8fba\u306e{side:g}%\uff08\u4e2d\u592e\u5024\uff09" if side is not None else "") + (
        f"\u3001\u5143\u306e\u753b\u50cf\u3067\u306e\u5927\u304d\u3055\u306f {native} px" if native is not None else "") + "\u3067\u3059"
    if (side is not None and side < WAY_SPECK_SIDE_PCT and per > WAY_FEW
            and native is not None and native <= WAY_SPECK_NATIVE_PX):
        why = (f"{seen_en}: more than {WAY_FEW}, under {WAY_SPECK_SIDE_PCT:g}% and at most "
               f"{WAY_SPECK_NATIVE_PX} px, a scattering too small to box at Fit")
        return "specks", _way_line(
            w, "specks", f"{seen_ja}\u30021\u679a\u306b{WAY_FEW}\u500b\u3088\u308a\u591a\u304f\u3001{WAY_SPECK_SIDE_PCT:g}%\u672a\u6e80\u304b\u3064 {WAY_SPECK_NATIVE_PX} px \u4ee5\u4e0b\u3067\u3001Fit \u30671\u3064\u305a\u3064\u56f2\u3080\u306b\u306f\u5c0f\u3055\u3059\u304e\u308b\u6563\u3089\u3070\u3063\u305f\u70b9\u3067\u3059",
            f"{seen_en} -- more than {WAY_FEW} a picture, under {WAY_SPECK_SIDE_PCT:g}% and at most "
            f"{WAY_SPECK_NATIVE_PX} px: a scattering too small to box one by one at Fit"), why
    if side is not None and side >= WAY_OBJECT_SIDE_PCT and 0 < per <= WAY_FEW:
        why = f"{seen_en}: {WAY_FEW} or fewer, each {WAY_OBJECT_SIDE_PCT:g}% or more"
        return "objects", _way_line(w, "objects", f"{seen_ja}\u3002{WAY_FEW}\u500b\u4ee5\u4e0b\u3067\u3001\u3069\u308c\u3082{WAY_OBJECT_SIDE_PCT:g}%\u4ee5\u4e0a\u3067\u3059",
                                    f"{seen_en} -- {WAY_FEW} or fewer, each {WAY_OBJECT_SIDE_PCT:g}% or more"), why
    return await _way_measured(w, band, seen_ja, seen_en)


def _native_side(band: dict) -> int | None:
    """The median object's long side in its own picture's pixels (teacher_band's object_size); None where not said."""
    size = band.get("object_size")
    if not isinstance(size, dict):
        return None
    sides = [size.get("median_width_px"), size.get("median_height_px")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in sides):
        return None
    return int(round(max(sides)))


def _specks_measured(w: Walk, sp: dict) -> tuple[float, str, str]:
    """spot_detect's measure of itself on the person's own: (F1, said, said for the file); held out where it could be.

    held_out is a threshold chosen on one half of their image and scored on
    the other -- what to expect on an image nobody drew; on_teacher, chosen
    and scored on the same marks, is how well it can fit them at best.
    """
    like = sp.get("like") if isinstance(sp.get("like"), dict) else {}
    held, on_t = like.get("held_out") or {}, like.get("on_teacher") or {}
    got = held if isinstance(held, dict) and held.get("theirs") else on_t
    if sp.get("error") or not isinstance(got, dict) or not got.get("theirs"):
        why = " ".join(str(sp.get("error") or "no measurement came back").split())[:160]
        return 0.0, w.say(f"\u6e2c\u308c\u307e\u305b\u3093\u3067\u3057\u305f\uff08{why}\uff09", f"not measured: {why}"), f"not measured ({why[:80]})"
    f1 = float(got.get("f1") or 0.0)
    held_ja, held_en = (("\u3057\u304d\u3044\u5024\u3092\u534a\u5206\u3067\u9078\u3073\u3001\u6b8b\u308a\u306e\u534a\u5206\u3067\u63a1\u70b9", "chosen on one half, scored on the other") if got is held else
                        ("\u3057\u304d\u3044\u5024\u3092\u9078\u3093\u3060\u306e\u3068\u540c\u3058\u70b9\u3067\u63a1\u70b9", "chosen and scored on the same marks"))
    return f1, w.say(f"F1 {f1:.2f}\u3002\u4eba\u304c\u4ed8\u3051\u305f\u70b9{got.get('theirs')}\u500b\u306e\u3046\u3061{got.get('found')}\u500b\u3092\u898b\u3064\u3051\u3001\u4eba\u304c\u4f55\u3082\u4ed8\u3051\u3066\u3044\u306a\u3044\u6240\u306b"
                     f"\u4ed8\u3051\u305f\u3082\u306e\u306f{got.get('on_nothing')}\u500b\u3067\u3057\u305f\uff08{held_ja}\uff09",
                     f"F1 {f1:.2f}: {got.get('found')} of their {got.get('theirs')} found, {got.get('on_nothing')} "
                     f"on none of theirs ({held_en})"), f"F1 {f1:.2f} ({'held out' if got is held else 'in sample'})"


async def _way_measured(w: Walk, band: dict, seen_ja: str, seen_en: str) -> tuple[str, str, str]:
    """Both ways tried on the person's own, and the one that found theirs better: (way, said, why).

    Specks: spot_detect with their marks on one of their images as the
    example, which measures itself there (_specks_measured). Objects: the
    walk's own look at Fit, that image shown first as the worked example,
    at ANOTHER of theirs -- a model shown its own answer is not measured --
    each box handed to SAM and judged by the band, and the boxes kept
    scored against their objects as zoom_score scores boxes, one to one.
    With one image of theirs, the look is at that one, with no example. Each
    is an F1: their objects found, and what was found on none of theirs.

    Filed only when the model was shown their example: without one it is
    told what to look for in the run's instruction alone -- "label the rest",
    "carry on" -- and a look that boxes nothing for want of a target counted
    against objects, which any F1 of spot_detect's then beat, and every run
    after walked a project of objects as specks. Without the example the way
    is this run's, and the next run measures again.
    """
    teachers = [str(t) for t in band.get("teachers") or [] if t]
    if await w.gate():
        raise _Ended("stopped")
    example = _example_of(await _call(w, "teacher_example", {"rung": "fit", "units": UNITS, "grid": False,
                                                              "max_objects": BOXES_PER_LOOK}))
    first = (example or {}).get("item_id") or (teachers[0] if teachers else "")
    if not first:
        return "", w.say("\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u898b\u3064\u304b\u3089\u305a\u3001\u30e9\u30d9\u30eb\u306e\u4ed8\u3051\u65b9\u3092\u6e2c\u308c\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u3053\u306e\u5b9f\u884c\u3067\u306f\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002",
                         "No image of the person's could be had to measure how this project is labelled on: this "
                         "run labels the way it did before"), ""
    if await w.gate():
        raise _Ended("stopped")
    sp = await _call(w, "spot_detect", {"item_id": first, "like_item_id": first})
    f_specks, specks_said, specks_why = _specks_measured(w, sp)
    if not sp.get("error"):
        w.like, w.like_score = first, {"f1": f_specks, "said": specks_said}
    other = next((t for t in teachers if t != first), "")
    on = other or first
    if await w.gate():
        raise _Ended("stopped")
    pic = await _open(w, on, "fit")
    got = None
    shown_example = example if other and not w.examples_refused else None
    if pic is not None:
        got = await _look(w, WALK_BRIEF_FIT.format(target=_target_of(w, "")), _words(w, "", "fit", pic),
                          pic["image_base64"], FOUND, shown_example)
    # The model server took the picture only without it (_look): not shown.
    fair = shown_example is not None and not w.examples_refused
    if got is None:
        return "", w.say(f"{on} \u3092\u30e2\u30c7\u30eb\u306b\u898b\u305b\u3089\u308c\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u30e9\u30d9\u30eb\u306e\u4ed8\u3051\u65b9\u3092\u6e2c\u308c\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u3053\u306e\u5b9f\u884c\u3067\u306f\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002",
                         f"{on} could not be put to the model, so how this project is labelled could not be "
                         "measured: this run labels the way it did before"), ""
    boxes = _boxes_in(got.get("boxes"))
    kept: list = []
    for i in range(0, len(boxes), ACCEPT_BATCH):
        res = await _call(w, "accept_masks", {"item_id": on, "boxes_json": json.dumps(boxes[i:i + ACCEPT_BATCH]),
                                              "view": "fit", "units": UNITS})
        kept += [r["box"] for r in res.get("results") or []
                 if isinstance(r, dict) and r.get("accepted") and isinstance(r.get("box"), list)]
    scored: dict = {}
    if kept:
        scored = await _call(w, "zoom_score", {"item_id": on, "boxes_json": json.dumps(kept), "units": UNITS,
                                               "from_width": int(pic.get("width") or 0),
                                               "from_height": int(pic.get("height") or 0)})
    # Kept for the score alone: nothing of it is written, and a walk of the
    # image after this starts from nothing.
    await _call(w, "kept_drop", {"item_id": on, "boxes_json": "all"})
    if scored.get("error"):
        why = " ".join(str(scored["error"]).split())[:160]
        return "", w.say(f"{on} \u3067\u56f2\u3093\u3060\u30dc\u30c3\u30af\u30b9\u3092\u63a1\u70b9\u3067\u304d\u306a\u304b\u3063\u305f\u305f\u3081\uff08{why}\uff09\u3001\u30e9\u30d9\u30eb\u306e\u4ed8\u3051\u65b9\u3092\u6c7a\u3081\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u3053\u306e\u5b9f\u884c\u3067\u306f\u3053\u308c\u307e\u3067\u306e\u65b9\u6cd5\u3067\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002",
                         f"The boxes on {on} could not be scored ({why}), so how this project is labelled could "
                         "not be chosen: this run labels the way it did before"), ""
    matched, theirs = int(scored.get("matched") or 0), int(scored.get("objects_in_view") or 0)
    f_objects = 2.0 * matched / (len(kept) + theirs) if kept and (len(kept) + theirs) else 0.0
    way = "specks" if f_specks > f_objects else "objects"
    shown = w.say("\u624b\u672c\u3042\u308a", "the example shown first") if fair else w.say("\u624b\u672c\u306a\u3057", "no example")
    objects_ja = (f"F1 {f_objects:.2f}\u3002{on} \u3092 Fit \u3067\u898b\u305f\u30e2\u30c7\u30eb\u306e\u30dc\u30c3\u30af\u30b9\uff08\u8a31\u5bb9\u7bc4\u56f2\u306b\u5408\u3046\u3082\u306e\uff09\u306f\u3001\u4eba\u304c\u4ed8\u3051\u305f{theirs}\u500b\u306e\u3046\u3061{matched}\u500b\u3092\u898b\u3064\u3051\u3001"
                  f"\u4eba\u304c\u4f55\u3082\u4ed8\u3051\u3066\u3044\u306a\u3044\u6240\u306b\u4ed8\u3051\u305f\u3082\u306e\u306f{len(kept) - matched}\u500b\u3067\u3057\u305f\uff08{shown}\uff09") if kept else \
        f"F1 0.00\u3002{on} \u3092 Fit \u3067\u898b\u305f\u30e2\u30c7\u30eb\u306e\u30dc\u30c3\u30af\u30b9\u306f\u30011\u3064\u3082\u8a31\u5bb9\u7bc4\u56f2\u306b\u5408\u3044\u307e\u305b\u3093\u3067\u3057\u305f\uff08{shown}\uff09"
    objects_en = (f"F1 {f_objects:.2f}: the model's boxes at Fit on {on}, kept by the band, found {matched} of "
                  f"their {theirs}, {len(kept) - matched} on none of theirs ({shown})") if kept else \
        f"F1 0.00: none of the model's boxes at Fit on {on} was kept by the band ({shown})"
    why = (f"measured on the person's own: specks {specks_why} on {first}; objects F1 {f_objects:.2f} at Fit "
           f"on {on}; {seen_en}") if fair else ""
    unfiled = None if fair else (
        "\u4eba\u306e\u624b\u672c\u3092\u898b\u305b\u305a\u3001\u6307\u793a\u306e\u8a00\u8449\u3060\u3051\u3067\u30e2\u30c7\u30eb\u306b\u63a2\u3055\u305b\u305f\u305f\u3081\u3001\u8a18\u9332\u306f\u3057\u307e\u305b\u3093\u3002\u6b21\u306e\u5b9f\u884c\u3067\u3082\u3046\u4e00\u5ea6\u6e2c\u308a\u307e\u3059\u3002",
        " Not filed: with no example of the person's shown, the model looked for the instruction's words alone; "
        "the next run measures again.")
    return way, _way_line(w, way, f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u4e21\u65b9\u306e\u65b9\u6cd5\u3092\u6e2c\u308a\u307e\u3057\u305f\u3002\u70b9\u3068\u3057\u3066\uff08{first}\uff09\uff1a{specks_said}\u3002\u7269\u4f53\u3068\u3057\u3066\uff1a{objects_ja}\u3002{seen_ja}",
                          f"measured on the person's own -- as specks on {first}, {specks_said}; as objects, "
                          f"{objects_en}; {seen_en}", tail=unfiled), why


async def _way_asked(w: Walk) -> tuple[str, str, str]:
    """(way, what is said of it, why it is filed), asked of the person: nothing of theirs to measure on.

    Asked once, as the question before walking is, and like it never answered
    for them: the command line's --yes takes objects and files nothing, for
    nobody chose; with nobody to ask and no --yes, nothing is chosen, and the
    run labels the way it did before. A reply that says none of the answers
    plainly is asked once more; a second one, or Stop, ends the run.
    """
    choices = [w.say("\u500b\u3005\u306e\u7269\u4f53", "Distinct objects"), w.say("\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9", "A scattering of small marks"),
               w.say("\u3084\u3081\u308b", "Stop")]
    text = w.say(
        "\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u307e\u3060\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u306a\u3044\u305f\u3081\u3001\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308b\u5bfe\u8c61\u306e\u7a2e\u985e\u3092\u4e00\u5ea6\u3060\u3051\u304a\u805e\u304d\u3057\u307e\u3059\uff08\u6b21\u56de\u4ee5\u964d\u306e\u5b9f\u884c\u306e\u305f\u3081\u306b\u8a18\u9332\u3057\u307e\u3059\uff09\u3002"
        "1\u3064\u305a\u3064\u56f2\u3093\u3067 SAM \u3067\u8f2a\u90ed\u3092\u53d6\u308b\u500b\u3005\u306e\u7269\u4f53\u3067\u3059\u304b\u3002\u305d\u308c\u3068\u3082\u3001\u50b7\u3084\u307b\u3053\u308a\u306e\u3088\u3046\u306a\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9\u3067\u30011\u3064\u3092\u624b\u304c\u304b\u308a\u306b\u6b8b\u308a\u3092\u307e\u3068\u3081\u3066\u63a2\u3059\u3082\u306e\u3067\u3059\u304b\u3002",
        "Nobody has drawn on this project yet, so what its targets are is asked once, and filed for the runs "
        "after. Distinct objects -- each boxed, its outline taken by SAM -- or a scattering of small marks, such "
        "as specks or dust, found together from one of them?") + _numbered(w, choices)
    if w.assume_consent:
        return "objects", _way_line(w, "objects", "\u78ba\u8a8d\u306a\u3057\u3067\u9032\u3081\u308b\u6307\u5b9a\uff08--yes\uff09\u304c\u3042\u308a\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3082\u306a\u3044\u305f\u3081\u3001\u3053\u306e\u5b9f\u884c\u3067\u306f\u7269\u4f53\u3068\u3057\u3066\u6271\u3044\u307e\u3059",
                                    "--yes, and nobody has drawn on it: objects, for this run", filed=False), ""
    if w.ask is None:
        return "", w.say(
            "\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u306a\u304f\u3001\u5bfe\u8c61\u304c\u500b\u3005\u306e\u7269\u4f53\u304b\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9\u304b\u3092\u7b54\u3048\u308b\u4eba\u3082\u3044\u306a\u3044\u305f\u3081\u3001"
            "\u3053\u306e\u5b9f\u884c\u3067\u306f\u4f55\u3082\u66f8\u304d\u8fbc\u307f\u307e\u305b\u3093\u3002\u624b\u30671\u679a\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308b\u304b\u3001\u4eba\u304c\u7b54\u3048\u3089\u308c\u308b\u753b\u9762\u304b\u3089\u5b9f\u884c\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
            "Nobody has drawn on this project and nobody is here to say whether its targets are distinct objects "
            "or a scattering of small marks: nothing is written on this run. Label one image by hand, or run it "
            "where a person can answer"), ""
    asked, reply, got = text, "", None
    for _ in range(2):
        w.emit({"type": "question", "item_id": None, "text": asked, "choices": choices})
        reply = str(await w.ask(asked))
        if await w.gate():
            raise _Ended("stopped")
        if reply == NOBODY_ANSWERED:
            raise _Ended("nobody", w.say(
                "\u5bfe\u8c61\u304c\u500b\u3005\u306e\u7269\u4f53\u304b\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9\u304b\u3068\u3044\u3046\u8cea\u554f\u306b\u8ab0\u3082\u7b54\u3048\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
                "Nobody answered whether this project's targets are distinct objects or a scattering of small "
                "marks, so nothing was labelled."))
        w.progress.moved("the person answered")
        w.answers.append((asked, reply))
        got = _way_of(reply, choices)
        if got is not None:
            break
        asked = _again_please(w, reply, text)
    if got in (None, "stop"):
        quoted = " ".join(reply.split())[:120]
        raise _Ended("declined", w.say(
            f"\u7b54\u3048\uff08\u300c{quoted}\u300d\uff09\u304b\u3089\u306f\u30e9\u30d9\u30eb\u306e\u4ed8\u3051\u65b9\u3092\u9078\u3079\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
            f"The answer (\"{quoted}\") chose no way of labelling, so nothing was labelled."))
    # The label of the choice, where the reply named it by its number: "2"
    # on the screen and in the file says nothing of what was chosen.
    named = _pick(reply, choices)
    quoted = choices[named] if named is not None else " ".join(reply.split())[:80]
    return got, _way_line(w, got, f"\u4eba\u306e\u7b54\u3048\u306f\u300c{quoted}\u300d\u3067\u3059", f"the person's answer, \"{quoted}\""), \
        f"the person's answer: {quoted}"


def _way_of(reply: str, choices: list[str]) -> str | None:
    """Which of objects / specks / stop a reply names; None for one that names none of them plainly."""
    i = _pick(reply, choices)
    if i is not None:
        return ("objects", "specks", "stop")[i]
    words = _words_of(reply)
    if _NOT_WORDS.search(_GO_PHRASES.sub(" ", words)):
        return None
    said = [k for k, found in (("objects", _OBJECT_WORDS), ("specks", _SPECK_WORDS), ("stop", _STOP_ONLY))
            if found.search(words)]
    return said[0] if len(said) == 1 else None


async def _call(w: Walk, name: str, args: dict) -> dict:
    """A bridge tool for this project, its answer read as the loop reads one; {"error"} when it raised."""
    try:
        res = w.text(await w.c.call_tool(name, {"project_id": w.project_id, **args}))
    except Exception as exc:                             # noqa: BLE001 - said, and the walk decides
        return {"error": " ".join(str(exc).split())[:TOOL_ERROR_CHARS]}
    if not isinstance(res, dict):
        return {"error": f"{name} answered with no object: {str(res)[:200]}"}
    return res


def _sam_down(err: Any) -> bool:
    """Whether a bridge tool's error is SAM failing to run on the server (_SAM_DOWN)."""
    return bool(_SAM_DOWN.search(str(err or "")))


def _no_sam(w: Walk, err: Any) -> _Ended:
    """The end of a walk SAM cannot run for, with what the server said of it -- and nothing of how it was asked."""
    said = " ".join(str(err).split())
    m = _SAM_DOWN.search(said)
    why = said[m.end():].lstrip(": ").rstrip() if m else said
    if why.endswith(")") and why.count(")") > why.count("("):
        why = why[:-1]                                   # calibrate_sam says it in brackets
    why = why or said
    stop = "" if why.endswith((".", "!", "?")) else "."
    return _Ended("failed", w.say(
        f"SAM \u3092\u5b9f\u884c\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff08{why}\uff09\u3002\u7269\u4f53\u306e\u30e9\u30d9\u30eb\u4ed8\u3051\u306b\u306f SAM \u304c\u5fc5\u8981\u306a\u305f\u3081\u3001\u3053\u3053\u3067\u6b62\u3081\u307e\u3057\u305f\u3002",
        f"SAM could not be run: {why}{stop} Labelling objects needs it, so the run stopped here."))


async def _look(w: Walk, brief: str, words: str, b64: str, schema: dict,
                example: dict | None = None) -> dict | None:
    """One picture in a conversation of its own; the answer, or None when the model server failed twice.

    The answer is a JSON object of the schema's shape (FOUND, CHECKED, POINT,
    RIGHT), or what its words say of it (_answer_of): {} when they say
    nothing plainly. Asked once more
    after a failure, as the loop asks a turn again. A failure twice is
    counted, and WALK_LOOK_FAILURES_STOP of them in a row end the walk.

    example, for a look for the objects: the person's picture with where it
    is in its own words, and the boxes their labels give it as the model's
    answer, before the picture asked about (EXAMPLE_SAID). A review is a
    question about a mask, which no example of the person's answers.
    """
    alone = [{"role": "system", "content": w.rules + "\n\n" + brief},
             {"role": "user", "content": words, "images": [b64]}]
    messages = alone
    if example:
        messages = [alone[0],
                    {"role": "user", "content": EXAMPLE_SAID + example["words"], "images": [example["b64"]]},
                    {"role": "assistant", "content": json.dumps({"boxes": example["boxes"]})},
                    {"role": "user", "content": NOW_SAID + words, "images": [b64]}]
    t0, why = time.time(), ""
    for _ in range(2):
        try:
            got, said = await asyncio.to_thread(w.bk.chat_json, messages, schema)
        except Exception as exc:                         # noqa: BLE001 - asked once more, then counted
            why = " ".join(str(exc).split())[:200]
            continue
        w.looks.append(time.time() - t0)
        w.failed_in_a_row = 0
        return _answer_of(got, said, schema)
    if example:
        # Two pictures in one question, failed twice. A server, or the model
        # behind it, may take one picture a question (a vLLM engine's
        # limit_mm_per_prompt of 1; a model that reads one image): every look
        # failed, and the walk ended after three where it had walked before
        # there was an example. Asked with the picture alone and answered, the
        # rest of the run goes without the example, and says so.
        t1 = time.time()
        try:
            got, said = await asyncio.to_thread(w.bk.chat_json, alone, schema)
        except Exception:                                # noqa: BLE001 - counted with the two before it
            pass
        else:
            _without_examples(w, why)
            w.looks.append(time.time() - t1)
            w.failed_in_a_row = 0
            return _answer_of(got, said, schema)
    w.failed_in_a_row += 1
    w.emit({"type": "failed", "step": 0, "name": "walk",
            "text": w.say(f"\u30e2\u30c7\u30eb\u304c\u5fdc\u7b54\u3057\u307e\u305b\u3093\u3067\u3057\u305f\uff08{why}\uff09\u3002", f"the model server did not answer ({why})")})
    if w.failed_in_a_row >= WALK_LOOK_FAILURES_STOP:
        raise _Ended("failed", w.say(
            f"\u30e2\u30c7\u30eb\u304c{w.failed_in_a_row}\u56de\u7d9a\u3051\u3066\u5fdc\u7b54\u3057\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3053\u3053\u3067\u6b62\u3081\u307e\u3057\u305f\uff08{why}\uff09\u3002",
            f"The model server failed {w.failed_in_a_row} times in a row, so the walk stopped here ({why})."))
    return None


def _without_examples(w: Walk, why: str) -> None:
    """The person's example let go of for the rest of the run: the model server answered a picture only without it.

    Fetched later -- for a rung not asked at yet, or after the look that
    chose how the project is labelled (decide_way) -- it is not shown either.
    """
    w.examples.update({k: None for k in w.examples})
    w.examples_refused = True
    w.emit({"type": "auto", "text": w.say(
        f"\u4eba\u306e\u624b\u672c\u3092\u6dfb\u3048\u305f\u8cea\u554f\u306b\u30e2\u30c7\u30eb\u304c2\u56de\u5fdc\u7b54\u305b\u305a\u3001\u753b\u50cf\u3060\u3051\u306a\u3089\u5fdc\u7b54\u3057\u305f\u305f\u3081\uff08{why}\uff09\u3001\u3053\u3053\u304b\u3089\u5148\u306f\u30e2\u30c7\u30eb\u306b\u624b\u672c\u3092\u898b\u305b\u305a\u306b\u5404\u753b\u50cf\u3092\u898b\u305b\u307e\u3059\u3002",
        "The model server failed twice on a question with the person's example in it, and answered the picture "
        f"alone ({why}), so from here on the model is shown each picture without an example.")})


def _answer_of(got: Any, said: Any, schema: dict) -> dict:
    """The answer to one of the walk's questions: the object, or else what its words say of it.

    A server held to the schema answers with the object. Not every one is:
    asked plainly (Backend.chat_json), a model answered with the boxes alone
    -- [[x0, y0, x1, y1], ...], no object round them -- and on a crowded
    picture an answer ran out of room (ANSWER_TOKENS) part-way through its
    list. Each was read as nothing found, and the image flagged with nothing
    kept. Every whole box in the words is taken instead, under the key named
    before it where the answer has two (CHECKED); a box cut off is not one.
    A point is the first two numbers in brackets (POINT). Right or not
    (RIGHT) is the one of true and false the words hold, said plainly: "not
    true" is no answer, nor are words holding both, and an image whose look
    gave none is flagged as not looked at, not taken as right.
    """
    keys = list(schema.get("properties") or {})
    if isinstance(got, dict) and any(k in got for k in keys):
        return got
    text = str(said or "")
    if keys == ["point"]:
        m = _POINT_SAID.search(text)
        return {"point": [float(m.group(1)), float(m.group(2))]} if m else {}
    if keys == ["right"]:
        low = text.lower()
        said = list(re.finditer(r"(?<![a-z])(true|false)(?![a-z])", low))
        if len({m.group(1) for m in said}) != 1 or any(_NOT_WORDS.search(low[max(0, m.start() - 12):m.start()])
                                                        for m in said):
            return {}
        return {"right": said[0].group(1) == "true"}

    def boxes(start: int, end: int) -> list[list[float]]:
        return [[float(v) for v in m.groups()] for m in _BOX_SAID.finditer(text, start, end)]
    if len(keys) == 1:
        found = boxes(0, len(text))
        return {keys[0]: found} if found else {}
    named = sorted((text.find(f'"{k}"'), k) for k in keys if f'"{k}"' in text)
    ends = [at for at, _ in named[1:]] + [len(text)]
    return {k: boxes(at, end) for (at, k), end in zip(named, ends)}


def _boxes_in(raw: Any, width: float = NORM, height: float = NORM) -> list[list[int]]:
    """The boxes of an answer that lie on the picture they were read off, at most BOXES_PER_LOOK.

    In the walk's units, 0 to 1000 across the picture: a box past 1000, or
    before 0, by more than VIEW_SLACK of it was not read off it.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    if len(raw) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
        raw = [raw]                                      # one box written flat
    out: list[list[int]] = []
    for b in raw:
        if not (isinstance(b, (list, tuple)) and len(b) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in b)):
            continue
        x0, x1 = sorted((float(b[0]), float(b[2])))
        y0, y1 = sorted((float(b[1]), float(b[3])))
        if x1 > width * VIEW_SLACK or y1 > height * VIEW_SLACK or x0 < -0.02 * width or y0 < -0.02 * height:
            continue                                     # not read off this picture
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append([int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))])
        if len(out) >= BOXES_PER_LOOK:
            break
    return out


def _words_of(reply: Any) -> str:
    """A reply as it is read: full-width letters and digits as ASCII, lower case, one space between words."""
    return " ".join(unicodedata.normalize("NFKC", str(reply or "")).lower().split())


def _plain(text: Any) -> str:
    """A reply or a label as they are compared: as read, without spaces or the punctuation round it.

    "100%\u3067\u9032\u3081\u308b" is the label "100% \u3067\u9032\u3081\u308b" typed without its space.
    """
    return _words_of(text).replace(" ", "").strip(_ENDS)


def _pick(reply: str, choices: list[str]) -> int | None:
    """Which of the choices a reply names: its number, its label, or the one label inside it.

    Inside only in a reply that turns nothing round: "don't stop" holds the
    label Stop.
    """
    low = _plain(reply)
    if not low:
        return None
    m = re.fullmatch("([1-9])(\u756a\u76ee?|\u3064\u76ee)?", low)
    if m and int(m.group(1)) <= len(choices):
        return int(m.group(1)) - 1
    labels = [_plain(c) for c in choices]
    if low in labels:
        return labels.index(low)
    words = _words_of(reply)
    if _NOT_WORDS.search(_GO_PHRASES.sub(" ", words)):
        return None
    rest = _plain(_GO_PHRASES.sub(" ", words))
    inside = [i for i, c in enumerate(labels) if c and c in rest]
    return inside[0] if len(inside) == 1 else None


def _consent_of(w: Walk, reply: str, choices: list[str],
                outcomes: tuple[str, ...] = ("proceed", "again", "stop")) -> str | None:
    """'proceed', 'again' or 'stop' from the person's reply; None for one that says none of them plainly.

    outcomes: what each of the choices means, in their order.
    """
    i = _pick(reply, choices)
    if i is not None:
        return outcomes[i]
    words = _words_of(reply)
    go = bool(_GO_PHRASES.search(words)) or bool(w.wants_no_more_questions(str(reply)))
    rest = _GO_PHRASES.sub(" ", words).strip(" " + _ENDS)
    go = go or bool(_YES_FIRST.match(rest)) or bool(_GO_WORDS.search(rest))
    stop = bool(_STOP_WORDS.search(rest))
    if go and not stop and not _NOT_WORDS.search(rest):
        return "proceed"
    if stop and not go:
        return "stop"
    return None


def _confirm_of(w: Walk, reply: str, choices: list[str]) -> int | None:
    """Which of keep / take back / stop a reply to confirm_each says; None for one it does not say plainly."""
    i = _pick(reply, choices)
    if i is not None:
        return i
    if w.wants_no_more_questions(str(reply)):
        return 0                                         # "do them all": this one as it is, and on
    rest = _GO_PHRASES.sub(" ", _words_of(reply))
    said = {k for k, words in enumerate((_KEEP_WORDS, _BACK_WORDS, _STOP_ONLY)) if words.search(rest)}
    if len(said) != 1 or (_NOT_WORDS.search(rest) and said != {2}):
        return None
    return said.pop()


def _no_more_asked(w: Walk) -> tuple[str, str]:
    """What the consent question says of the rest of the run: nothing more, or each image written.

    'One look each' was read as one press each: a person shown it took the
    question for the first of hundreds and stopped the run.
    """
    if w.confirm_each:
        return ("\u753b\u50cf\u30921\u679a\u66f8\u304d\u8fbc\u3080\u305f\u3073\u306b\u3001\u6b21\u3078\u9032\u3080\u524d\u306b\u78ba\u8a8d\u3057\u307e\u3059\u3002", "Each image written is put to you before the next.")
    return ("\u9032\u3081\u308b\u3068\u3001\u6700\u5f8c\u307e\u3067\u4f55\u3082\u805e\u304d\u307e\u305b\u3093\u3002", "Once you go ahead, nothing more is asked until the end.")


def _numbered(w: Walk, choices: list[str]) -> str:
    """The choices with their numbers, said after a question: a screen without buttons is answered by typing."""
    listed = " / ".join(f"{k} {c}" for k, c in enumerate(choices, 1))
    return w.say(f"\uff08{listed}\uff09", f" ({listed})")


def _again_please(w: Walk, reply: str, text: str) -> str:
    """The same question once more, after a reply that said none of its answers plainly."""
    quoted = " ".join(str(reply).split())[:80]
    return w.say(f"\u300c{quoted}\u300d\u304c\u3069\u308c\u3092\u6307\u3059\u306e\u304b\u5206\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u756a\u53f7\u3067\u7b54\u3048\u3066\u304f\u3060\u3055\u3044\u3002{text}",
                 f"I could not tell which of these \"{quoted}\" means; answer with its number. {text}")


def _looks_of(todo: list[dict]) -> tuple[int, float]:
    """(parts, pictures looked at) of a walk of these images: every part, an overview, and reviews."""
    parts = sum(max(1, int(e.get("parts") or 1)) for e in todo)
    overviews = sum(1 for e in todo if int(e.get("parts") or 1) > 1)
    return parts, parts * (1 + REVIEW_SHARE_DEFAULT) + overviews


def _estimate(w: Walk, todo: list[dict]) -> tuple[int, int]:
    """(parts, minutes) a walk of these images comes to.

    Before this run has timed a look, each part's carries the person's
    example when the walk will show one (EXAMPLE_LOOK_EXTRA_SECONDS). The
    whole picture after the parts has it only when the example holds an
    object too large for a part (_overview), which is not known until it is
    fetched -- after this is asked -- and seldom so: those are counted
    without.

    Specks: per image, spot_detect and spot_write, and the look after them --
    with nothing of the person's, a look for one point before it too -- and,
    once a run, the detector's measurement of their marks, unless it was
    measured already on choosing the way.
    """
    if _specks(w):
        looks = len(todo) * (1 if _taught(w) else 2)
        seconds = (looks * (statistics.median(w.looks) if w.looks else LOOK_SECONDS_DEFAULT)
                   + len(todo) * SPECK_IMAGE_SECONDS_DEFAULT
                   + (SPECK_CALIBRATION_SECONDS if _taught(w) and w.like_score is None else 0.0))
        return len(todo), max(1, int(round(seconds / 60.0)))
    parts, looks = _looks_of(todo)
    if w.looks:
        seconds = looks * statistics.median(w.looks)
    else:
        seconds = looks * LOOK_SECONDS_DEFAULT + (parts * EXAMPLE_LOOK_EXTRA_SECONDS if _shows_example(w) else 0.0)
    seconds += len(todo) * PER_IMAGE_SECONDS_DEFAULT
    return parts, max(1, int(round(seconds / 60.0)))


def _model_turns(todo: list[dict]) -> int:
    """The turns of the run's own a walk of these images by the model is allowed (Walk.model_turns)."""
    return int(MODEL_TURNS_PER_LOOK * _looks_of(todo)[1]) + MODEL_TURNS_PER_IMAGE * len(todo)


async def _consent(w: Walk, filed: dict, todo: list[dict]) -> tuple[str, str]:
    """Whether to walk these at the filed zoom -- or the one this run chose -- or at Fit: (what was said, the reply).

    Asked once the zoom is known, so that how long it takes is counted at it.
    'proceed', 'again', 'stop', 'nobody' (no one to ask), 'unanswered' (the
    question stood until the screen answered it itself) or 'stopped' (the Stop
    button). Always put to the person, also after a reply that turned the
    run's other questions off: the zoom was chosen on an earlier run, and how
    long the rest will take depends on what is left now. A run with nobody to
    answer does not go ahead on its own; the command line's --yes says yes for
    it. A reply that says none of the answers plainly is asked once more; a
    second such reply is not a yes.
    """
    if _specks(w):
        return await _consent_specks(w, todo)
    rung = str(filed.get("rung"))
    parts, minutes = _estimate(w, todo)
    how = str(filed.get("how") or w.say("\u524d\u306e\u5b9f\u884c", "an earlier run"))
    # No label holds a word that turns the run's questions off (GO_AHEAD):
    # pressed, the button would also have answered every question after it.
    choices = [w.say(f"{rung} \u3067\u9032\u3081\u308b", f"Proceed at {rung}"), w.say("\u500d\u7387\u3092\u9078\u3073\u76f4\u3059", "Choose the zoom again"),
               w.say("\u3084\u3081\u308b", "Stop")]
    if filed.get("default"):
        # Nothing filed: Fit, one look at each image. Asked all the same: how
        # long it takes depends on what is left. Go ahead or stop, and nothing
        # else: there is no zoom to take back, and nothing in a run chooses
        # one -- offered, "choose the zoom again" ended the run with "file a
        # zoom and start again", which nobody at the Assist panel can do.
        choices = [choices[0], choices[2]]
        text = w.say(
            "\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u500d\u7387\u304c\u8a18\u9332\u3055\u308c\u3066\u3044\u306a\u3044\u305f\u3081\u3001\u5404\u753b\u50cf\u3092\u5168\u4f53\u306e\u307e\u307e\uff08Fit \u3067\uff09\u898b\u307e\u3059\u3002"
            f"\u672a\u30e9\u30d9\u30eb\u306e\u753b\u50cf\u306f{len(todo)}\u679a\u3042\u308a\u3001\u30e2\u30c7\u30eb\u304c1\u679a\u305a\u3064\u898b\u3066\u3044\u304f\u3068\u3001\u304a\u3088\u305d{minutes}\u5206\u304b\u304b\u308b\u898b\u8fbc\u307f\u3067\u3059\u3002"
            f"{_no_more_asked(w)[0]}{rung} \u3067\u9032\u3081\u307e\u3059\u304b\u3001\u3084\u3081\u307e\u3059\u304b\u3002",
            f"No zoom is filed for this project, so each image is looked at whole, as Fit shows it. "
            f"{len(todo)} images are left to label, the model looking at each in turn: about {minutes} minutes. "
            f"{_no_more_asked(w)[1]} Proceed at {rung}, or stop?") + _numbered(w, choices)
        return await _ask_consent(w, rung, text, choices, ("proceed", "stop"))
    # A zoom at Fit is one look at each image, filed or chosen: "parts in all"
    # counted the images again as parts of themselves.
    left_ja, left_en = ((f"\u672a\u30e9\u30d9\u30eb\u306e\u753b\u50cf\u306f{len(todo)}\u679a\u3042\u308a\u3001\u30e2\u30c7\u30eb\u304c1\u679a\u305a\u3064\u898b\u3066\u3044\u304f\u3068\u3001\u304a\u3088\u305d{minutes}\u5206\u304b\u304b\u308b\u898b\u8fbc\u307f\u3067\u3059\u3002",
                         f"{len(todo)} images are left to label, the model looking at each in turn: about {minutes} minutes.")
                        if rung == "fit" else
                        (f"\u672a\u30e9\u30d9\u30eb\u306e\u753b\u50cf\u306f{len(todo)}\u679a\uff08\u90e8\u5206\u306b\u5206\u3051\u308b\u3068{parts}\u679a\uff09\u3067\u3001\u304a\u3088\u305d{minutes}\u5206\u304b\u304b\u308b\u898b\u8fbc\u307f\u3067\u3059\u3002",
                         f"{len(todo)} images are left to label: {parts} parts in all, about {minutes} minutes."))
    left_ja, left_en = left_ja + _no_more_asked(w)[0], left_en + " " + _no_more_asked(w)[1]
    chose = w.rehearsed or {}
    if chose.get("passed") and chose.get("rung") == rung:
        # Chosen on this run, after this was asked for: go ahead or stop. The
        # same rehearsal again would choose the same zoom, so there is nothing
        # to choose again; a later run asks with that choice (below).
        choices = [choices[0], choices[2]]
        scored = f"{chose['recall']:.2f}", f"{chose['precision']:.2f}"
        filed_ja, filed_en = (("\u3053\u306e\u500d\u7387\u306f\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u8a18\u9332\u3057\u307e\u3057\u305f", "and it is filed for this project") if chose.get("filed") else
                              ("\u8a18\u9332\u3067\u304d\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3053\u306e\u500d\u7387\u306f\u3053\u306e\u5b9f\u884c\u3067\u3060\u3051\u4f7f\u3044\u307e\u3059", "for this run: it could not be filed"))
        text = w.say(
            f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u3057\u305f\u7d50\u679c\u3001\u30e2\u30c7\u30eb\u304c\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u7269\u4f53\u3092\u898b\u3064\u3051\u3089\u308c\u305f\u6700\u3082\u4f4e\u3044\u500d\u7387\u306f {rung} \u3067\u3057\u305f"
            f"\uff08\u518d\u73fe\u7387 {scored[0]}\u3001\u9069\u5408\u7387 {scored[1]}\uff09\u3002{filed_ja}\u3002{left_ja}{rung} \u3067\u9032\u3081\u307e\u3059\u304b\u3001\u3084\u3081\u307e\u3059\u304b\u3002",
            f"Rehearsed on the person's own images, {rung} is the lowest zoom at which the model found theirs "
            f"(recall {scored[0]}, precision {scored[1]}), {filed_en}. {left_en} Proceed at {rung}, or stop?"
        ) + _numbered(w, choices)
        return await _ask_consent(w, rung, text, choices, ("proceed", "stop"))
    text = w.say(
        f"\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u500d\u7387 {rung} \u304c\u8a18\u9332\u3055\u308c\u3066\u3044\u307e\u3059\uff08{how}\uff09\u3002{left_ja}"
        f"{rung} \u3067\u9032\u3081\u307e\u3059\u304b\u3001\u500d\u7387\u3092\u9078\u3073\u76f4\u3057\u307e\u3059\u304b\u3001\u3084\u3081\u307e\u3059\u304b\u3002",
        f"The zoom filed for this project is {rung} (chosen by {how}). {left_en} "
        f"Proceed at {rung}, choose the zoom again, or stop?") + _numbered(w, choices)
    return await _ask_consent(w, rung, text, choices)


async def _consent_specks(w: Walk, todo: list[dict]) -> tuple[str, str]:
    """_consent for a project of specks: go ahead or stop, with how long it takes."""
    n, minutes = _estimate(w, todo)
    choices = [w.say("\u9032\u3081\u308b", "Proceed"), w.say("\u3084\u3081\u308b", "Stop")]
    how_ja, how_en = (("\u4eba\u304c\u4ed8\u3051\u305f\u70b9\u3092\u624b\u672c\u306b", "the person's own marks as the example") if _taught(w) else
                      ("\u30e2\u30c7\u30eb\u304c\u6307\u3057\u305f1\u3064\u306e\u70b9\u3092\u3082\u3068\u306b", "from one mark the model points at in it"))
    text = w.say(
        f"\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u3067\u306f\u3001\u6563\u3089\u3070\u3063\u305f\u5c0f\u3055\u306a\u70b9\u3068\u3057\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u307e\u3059\u3002\u5404\u753b\u50cf\u306e\u5168\u4f53\u3092 spot_detect \u3067\u63a2\u3057\uff08{how_ja}\uff09\u3001\u66f8\u304d\u8fbc\u3093\u3067\u304b\u30891\u56de\u898b\u76f4\u3057\u307e\u3059\u3002"
        f"\u672a\u30e9\u30d9\u30eb\u306e\u753b\u50cf\u306f{n}\u679a\u3042\u308a\u3001\u304a\u3088\u305d{minutes}\u5206\u304b\u304b\u308b\u898b\u8fbc\u307f\u3067\u3059\u3002\u9032\u3081\u307e\u3059\u304b\u3001\u3084\u3081\u307e\u3059\u304b\u3002",
        f"This project is labelled as a scattering of small marks: each image is searched whole with spot_detect, "
        f"{how_en}, written, and looked at once more. {n} images are left to label: about {minutes} minutes. "
        "Proceed, or stop?") + _numbered(w, choices)
    return await _ask_consent(w, "fit", text, choices, ("proceed", "stop"))


def _specks(w: Walk) -> bool:
    """Whether the project is walked as specks (spot_detect), not as objects."""
    return _way_filed(w.plan) == "specks"


def _at(w: Walk, rung: str) -> str:
    """How going ahead is said: at the zoom, or as specks."""
    return w.say("\u70b9\u3068\u3057\u3066", "as specks") if _specks(w) else w.say(f"{rung} \u3067", f"at {rung}")


async def _ask_consent(w: Walk, rung: str, text: str, choices: list[str],
                       outcomes: tuple[str, ...] = ("proceed", "again", "stop")) -> tuple[str, str]:
    """_consent's question put to the person, and what they said to it (outcomes: what each choice means)."""
    if w.assume_consent:
        at = _at(w, rung)
        w.emit({"type": "auto", "text": w.say(f"\u78ba\u8a8d\u306a\u3057\u3067\u9032\u3081\u308b\u6307\u5b9a\uff08--yes\uff09\u304c\u3042\u308b\u305f\u3081\u3001{at}\u9032\u3081\u307e\u3059\uff1a{text}",
                                              f"going ahead {at}, as --yes says: {text}")[:400]})
        return "proceed", ""
    if w.ask is None:
        w.emit({"type": "auto", "text": w.say(f"\u7b54\u3048\u308b\u4eba\u304c\u3044\u306a\u3044\u305f\u3081\u3001\u9032\u3081\u307e\u305b\u3093\uff1a{text}",
                                              f"nobody to answer, so not going ahead: {text}")[:400]})
        return "nobody", ""
    asked, said, reply = text, None, ""
    for _ in range(2):
        w.emit({"type": "question", "item_id": None, "text": asked, "choices": choices})
        reply = str(await w.ask(asked))
        if await w.gate():
            return "stopped", reply
        if reply == NOBODY_ANSWERED:
            return "unanswered", ""
        w.progress.moved("the person answered")
        w.answers.append((asked, reply))
        said = _consent_of(w, reply, choices, outcomes)
        if said is not None:
            break
        asked = _again_please(w, reply, text)
    if said == "proceed" and _pick(reply, choices) is None and w.wants_no_more_questions(reply):
        # Their own words, not a button: they are for the rest of the run, and
        # the questions per image stop.
        w.no_more_questions.update(flag=True, said=reply, to=text)
    return said or "stop", reply


async def _prepare(w: Walk) -> None:
    """What labelling at the filed zoom -- or at Fit -- stands on, in this run's bridge: the teachers and SAM.

    The bridge is a new process every run and holds nothing of the last one;
    what was filed tells it the zoom, and teacher_band and calibrate_sam then
    measure with SAM shown the parts -- calibrate_sam answers from the file
    when it was measured at this zoom before.

    The steps before labelling (the loop's STEPS_BEFORE_LABELLING) are not
    asked, at a filed zoom or at Fit. What they are for, the walk does
    itself, or does not need: the program reads the teachers and measures SAM
    here; every box goes to SAM and is judged against the band the person's
    own masks give, never written as the model sent it; every write is
    looked at again in review; and the person is asked first, with how long
    it takes (_consent). At a filed zoom the steps were passed when it was
    chosen; a zoom chosen on this run was chosen by rehearsing the walk on
    the person's own (_decide_rung), which is what the steps ask. At Fit with
    nothing chosen -- the model walking the parts, or a rehearsal that could
    not be had -- nothing was rehearsed. Fit is walked only on a project
    labelled as objects -- by an earlier run, whose writes the loop held
    until its steps were passed, or chosen on this one from the person's own
    (decide_way); the person's go-ahead stands in for the rest, and the
    screen says so. A project of specks stands on less (_prepare_specks).
    """
    if _specks(w):
        await _prepare_specks(w)
        return
    # at_filed_zoom: the zoom filed on an earlier run is taken up only when a
    # walk labels at it. A run labelling the old way -- the whole frame -- had
    # its shrink measured on parts it never asks SAM on.
    band = await _call(w, "teacher_band", {"at_filed_zoom": True})
    if band.get("error"):
        raise _Ended("failed", w.say(f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f\uff08{band['error']}\uff09\u3002\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
                                     f"The teachers could not be read ({band['error']}). Nothing was labelled."))
    cal = await _call(w, "calibrate_sam", {})
    if cal.get("error"):
        w.emit({"type": "failed", "step": 0, "name": "calibrate_sam", "text": str(cal["error"])[:300]})
        if _sam_down(cal["error"]):
            raise _no_sam(w, cal["error"])
    filed = w.plan.get("filed") or {}
    rung = filed.get("rung")
    if filed.get("default"):
        # Not rehearsed at Fit (see above): said, so nobody takes the steps
        # for passed at it.
        how_ja, how_en = (("\u3053\u306e\u5b9f\u884c\u3067\u3001\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u7269\u4f53\u3068\u3057\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308b\u3068\u6c7a\u3081\u305f", "is labelled as objects, as chosen on this run")
                          if w.way_said else
                          ("\u524d\u306e\u5b9f\u884c\u3067\u3001\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u7269\u4f53\u3068\u3057\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u308b",
                           "was labelled as objects on an earlier run"))
        w.emit({"type": "auto", "text": w.say(
            "\u500d\u7387\u304c\u8a18\u9332\u3055\u308c\u3066\u3044\u306a\u3044\u305f\u3081\u3001\u5404\u753b\u50cf\u3092\u5168\u4f53\u306e\u307e\u307e\u898b\u307e\u3059\u3002"
            f"{how_ja}\u305f\u3081\u3001\u30e9\u30d9\u30eb\u4ed8\u3051\u524d\u306e\u6bb5\u968e\u306f\u6539\u3081\u3066\u884c\u3044\u307e\u305b\u3093\uff08Fit \u3067\u306e\u8a66\u3057\u306f\u3057\u3066\u3044\u307e\u305b\u3093\uff09\u3002"
            "\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u307f\u3001\u753b\u50cf\u5168\u4f53\u3067 SAM \u3092\u6e2c\u308a\u307e\u3057\u305f\u3002",
            f"No zoom is filed, so each image is looked at whole. This project {how_en}, so the steps before "
            "labelling are not asked again (nothing was rehearsed at Fit): the teachers were read and SAM "
            "measured on whole pictures.")})
        return
    # Said as what SAM was measured on: a zoom of Fit filed, now what a
    # rehearsal files most often, was said to be measured "on parts at fit".
    on_ja, on_en = (("\u753b\u50cf\u5168\u4f53", "on whole pictures") if rung == "fit" else
                    (f"{rung} \u306e\u90e8\u5206", f"on parts at {rung}"))
    chose = w.rehearsed or {}
    if chose.get("rung") == rung and not chose.get("passed"):
        # None passed: what stands in for the steps is the sample at the best
        # rung, looked at by the person, and their answer to it (_sample).
        w.emit({"type": "auto", "text": w.say(
            f"\u500d\u7387 {rung} \u306f\u3001\u3053\u306e\u5b9f\u884c\u3067\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u3057\u305f\u4e2d\u3067\u306f\u6700\u3082\u826f\u3044\u7d50\u679c\u3067\u3057\u305f\u304c\u3001\u57fa\u6e96\u306b\u306f\u5c4a\u3044\u3066\u3044\u307e\u305b\u3093\u3002\u30e9\u30d9\u30eb\u4ed8\u3051\u524d\u306e\u6bb5\u968e\u306e\u4ee3\u308f\u308a\u306b\u3001\u3053\u306e\u500d\u7387\u3067\u8a66\u3057\u306b\u4ed8\u3051\u305f\u30e9\u30d9\u30eb\u3092\u898b\u3066\u3044\u305f\u3060\u304d\u3001\u305d\u306e\u304a\u7b54\u3048\u3067\u9032\u3081\u308b\u304b\u3069\u3046\u304b\u3092\u6c7a\u3081\u307e\u3059\u3002"
            f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u307f\u3001{on_ja}\u3067 SAM \u3092\u6e2c\u308a\u307e\u3057\u305f\u3002",
            f"{rung} was the best of the zooms rehearsed on this run, short of what is wanted: in place of the "
            "steps before labelling, a sample is labelled at it for you to look at, and your answer decides "
            f"whether to go on. The teachers were read and SAM measured {on_en}.")})
        return
    if chose.get("rung") == rung:
        w.emit({"type": "auto", "text": w.say(
            f"\u500d\u7387 {rung} \u306f\u3001\u3053\u306e\u5b9f\u884c\u3067\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u3057\u3066\u6c7a\u3081\u305f\u305f\u3081\u3001\u30e9\u30d9\u30eb\u4ed8\u3051\u524d\u306e\u6bb5\u968e\u306f\u884c\u3044\u307e\u305b\u3093\u3002"
            f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u307f\u3001{on_ja}\u3067 SAM \u3092\u6e2c\u308a\u307e\u3057\u305f\u3002",
            f"{rung} was chosen on this run by rehearsing on the person's own images, so the steps before "
            f"labelling are not asked: the teachers were read and SAM measured {on_en}.")})
        return
    # The steps before labelling were passed when the zoom was chosen. The loop
    # holds the model's writes until they are, and the walk writes nothing the
    # model sends, so they are not asked again -- said, so nobody wonders.
    w.emit({"type": "auto", "text": w.say(
        f"\u500d\u7387 {rung} \u306f\u524d\u306e\u5b9f\u884c\u3067\u6c7a\u3081\u305f\u305f\u3081\u3001\u30e9\u30d9\u30eb\u4ed8\u3051\u524d\u306e\u6bb5\u968e\u306f\u6539\u3081\u3066\u884c\u3044\u307e\u305b\u3093\u3002"
        f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u307f\u3001{on_ja}\u3067 SAM \u3092\u6e2c\u308a\u307e\u3057\u305f\u3002",
        f"{rung} was chosen on an earlier run, so the steps before labelling are not asked again: "
        f"the teachers were read and SAM measured {on_en}.")})


async def _prepare_specks(w: Walk) -> None:
    """What a walk of specks stands on: the person's marks as the example, measured -- or, with none, a point a picture.

    teacher_band without the shrink -- the count a write is checked against,
    and the images the person drew; SAM, which specks never ask, is not
    measured. The example is one of their images, the one whose count is
    nearest the middle of theirs (teacher_example), and spot_detect measures
    it on itself before the first image: some 20 s on a 4K frame, paid once
    a run, said with its numbers, and not counted in the first image's time,
    from which the rest is estimated (_reestimate). Already done when the way
    was measured on this run (decide_way). An image of theirs the detector
    cannot take is passed over for the next of theirs; with none, each image
    is labelled from a point the model gives, as with nothing of theirs.
    """
    if not _taught(w):
        w.emit({"type": "auto", "text": w.say(
            "\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u306a\u3044\u305f\u3081\u3001\u5404\u753b\u50cf\u3067\u30e2\u30c7\u30eb\u304c\u70b9\u30921\u3064\u6307\u3057\u3001spot_detect \u304c\u305d\u308c\u306b\u4f3c\u305f\u6b8b\u308a\u306e\u70b9\u3092\u63a2\u3057\u307e\u3059\u3002",
            "Nobody has drawn on this project, so in each image the model points at one mark, and spot_detect "
            "finds the rest like it.")})
        return
    if await w.gate():
        raise _Ended("stopped")
    if w.band is None:
        band = await _call(w, "teacher_band", {"calibrate_shrink": False})
        if band.get("error"):
            raise _Ended("failed", w.say(f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f\uff08{band['error']}\uff09\u3002\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
                                         f"The teachers could not be read ({band['error']}). Nothing was labelled."))
        w.band = band
    failed: list[str] = []
    if w.like_score is None:
        if not w.like:
            got = await _call(w, "teacher_example", {"rung": "fit", "units": UNITS, "grid": False})
            w.like = str(got.get("item_id") or "")
        # Their image whose count is nearest the middle first, then the rest of
        # theirs: one the trainer would not take as the example (its mask
        # refused, the picture gone) ended the run, and every run after it in
        # the same place, the way being filed.
        for like in dict.fromkeys(t for t in [w.like, *(str(t) for t in w.band.get("teachers") or [])] if t):
            if await w.gate():
                raise _Ended("stopped")
            sp = await _call(w, "spot_detect", {"item_id": like, "like_item_id": like})
            if sp.get("error"):
                failed.append(f"{like}: " + " ".join(str(sp["error"]).split())[:120])
                continue
            f1, said, _ = _specks_measured(w, sp)
            w.like, w.like_score = like, {"f1": f1, "said": said}
            break
        else:
            # None of theirs: as on a project nobody drew on, a point an image.
            w.like = ""
            why = "; ".join(failed)[:300] or w.say("\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093", "no image the person drew could be had")
            w.emit({"type": "auto", "text": w.say(
                f"spot_detect \u304c\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u306e\u3069\u308c\u3082\u624b\u672c\u306b\u3067\u304d\u306a\u304b\u3063\u305f\u305f\u3081\uff08{why}\uff09\u3001\u5404\u753b\u50cf\u3067\u30e2\u30c7\u30eb\u304c\u70b9\u30921\u3064\u6307\u3057\u3001"
                "spot_detect \u304c\u305d\u308c\u306b\u4f3c\u305f\u6b8b\u308a\u306e\u70b9\u3092\u63a2\u3057\u307e\u3059\u3002",
                f"spot_detect could take the person's marks as the example on no image of theirs ({why}), so in "
                "each image the model points at one mark, and spot_detect finds the rest like it.")})
            return
    passed = w.say(f"\uff08{'; '.join(failed)[:300]} \u306f\u624b\u672c\u306b\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff09", f" ({'; '.join(failed)[:300]}: not "
                   "taken as the example)") if failed else ""
    w.emit({"type": "auto", "text": w.say(
        f"\u5404\u753b\u50cf\u306e\u5168\u4f53\u3092 spot_detect \u3067\u63a2\u3057\u307e\u3059\u3002\u624b\u672c\u306f {w.like} \u306b\u4eba\u304c\u4ed8\u3051\u305f\u70b9\u3067\u3059{passed}\u3002\u305d\u306e\u753b\u50cf\u3067\u6e2c\u3063\u305f\u7d50\u679c\uff1a{w.like_score['said']}\u3002"
        "\u898b\u3064\u3051\u305f\u3082\u306e\u306f\u66f8\u304d\u8fbc\u3093\u3067\u304b\u30891\u56de\u898b\u76f4\u3059\u305f\u3081\u3001\u30e9\u30d9\u30eb\u4ed8\u3051\u524d\u306e\u6bb5\u968e\u306f\u884c\u3044\u307e\u305b\u3093\u3002",
        f"Each image is searched whole with spot_detect, the person's marks on {w.like} as the example{passed} "
        f"-- measured there, {w.like_score['said']}. What is found is written and looked at once more, so the "
        "steps before labelling are not asked.")})


def _taught(w: Walk) -> bool:
    """Whether the person has drawn on the project: what a worked example is made of."""
    return bool(w.plan.get("taught")) and (w.plan.get("filed") or {}).get("taught") is not False


def _shows_example(w: Walk) -> bool:
    """Whether the looks for the objects will carry the person's example: the harness's, on a project they drew on.

    The model driver asks nothing in a conversation of its own, and shows
    none. Once fetched, what came back says.
    """
    if w.driver != "harness" or not _taught(w):
        return False
    return not w.examples or any(w.examples.values())


def _example_words(got: dict) -> str:
    """What is said with the person's example: where it is, in its own words.

    Not the words of the picture asked about after it: said with those, an
    example from row 2 of 2 of another image was "row 1 of 3, column 2 of 4"
    of this one, and the size one object looks there was this picture's.
    The boxes of its answer show how large one is.
    """
    rung, where = str(got.get("rung") or "fit"), " ".join(str(got.get("where") or "").split())
    if rung == "fit":
        return "The whole picture."
    return f"This part: {where}, at {rung}." if where else f"A part of a picture, at {rung}."


async def _examples(w: Walk, rung: str, class_id: int | None = None) -> None:
    """The person's worked example for Fit and for the filed rung's parts, fetched once a run (teacher_example).

    Their objects of the class the walk writes, when it is known: an answer
    boxing another class of theirs would show the model what the words did
    not ask for, and SAM would write it as this class. At most as many boxes
    as one answer is taken with (BOXES_PER_LOOK): an example answer cut
    there leaves the rest of its objects unboxed.

    Fetched before the first image, not inside it: the first image is timed
    for what the rest will take (_reestimate), and a fetch counted in it was
    counted again for every image after it. A project nobody drew on has no
    example to show, and a bridge may give none -- one from before
    teacher_example, one that cannot read the person's masks, or one with no
    view of theirs at the rung that holds a whole object: the walk goes on
    without, and says so once.
    """
    wanted = [r for r in dict.fromkeys(("fit", str(rung or "fit"))) if (r, class_id) not in w.examples]
    if not wanted:
        return
    if w.examples_refused:
        w.examples.update({(r, class_id): None for r in wanted})   # said when it was let go of
        return
    if not _taught(w):
        w.examples.update({(r, class_id): None for r in wanted})
        w.emit({"type": "auto", "text": w.say(
            "\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306f\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c\u306a\u3044\u305f\u3081\u3001\u30e2\u30c7\u30eb\u306b\u306f\u624b\u672c\u3092\u898b\u305b\u305a\u306b\u5404\u753b\u50cf\u3092\u898b\u305b\u307e\u3059\u3002",
            "Nobody has drawn on this project, so the model is shown each picture without an example of theirs.")})
        return
    without, why = [], ""
    for r in wanted:
        if await w.gate():
            raise _Ended("stopped")
        args: dict = {"rung": r, "units": UNITS, "grid": False, "max_objects": BOXES_PER_LOOK}
        if class_id is not None:
            args["class_id"] = int(class_id)
        got = await _call(w, "teacher_example", args)
        w.examples[(r, class_id)] = _example_of(got)
        if w.examples[(r, class_id)] is None:
            without.append(r)
            why = why or " ".join(str(got.get("error") or (
                f"no picture with 1 to {BOXES_PER_LOOK} boxes in 0-1000 came back")).split())[:160]
    ids = list(dict.fromkeys(w.examples[(r, class_id)]["item_id"] for r in ("fit", str(rung or "fit"))
                             if w.examples.get((r, class_id))))
    if set(ids) - w.example_said:
        # Not again for each rung a rehearsal tries, but for one whose example
        # is another of theirs: the images the examples of this rung are made of.
        w.example_said.update(ids)
        w.emit({"type": "auto", "text": w.say(
            f"\u5404\u753b\u50cf\u306e\u524d\u306b\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\uff08{', '.join(ids)}\uff09\u3068\u3001\u305d\u306e\u30e9\u30d9\u30eb\u304b\u3089\u4f5c\u3063\u305f\u6b63\u89e3\u3092\u3001\u624b\u672c\u3068\u3057\u3066\u30e2\u30c7\u30eb\u306b\u898b\u305b\u307e\u3059\u3002",
            f"Before each picture, the model is shown one the person labelled ({', '.join(ids)}), with the answer "
            "their labels give for it.")})
    if without:
        at = ", ".join(without)
        w.emit({"type": "auto", "text": w.say(
            f"\u4eba\u304c\u4ed8\u3051\u305f\u30e9\u30d9\u30eb\u304b\u3089\u624b\u672c\u3092\u7528\u610f\u3067\u304d\u306a\u304b\u3063\u305f\u305f\u3081\uff08{why}\uff09\u3001{at} \u3067\u306f\u30e2\u30c7\u30eb\u306b\u624b\u672c\u3092\u898b\u305b\u305a\u306b\u5404\u753b\u50cf\u3092\u898b\u305b\u307e\u3059\u3002",
            f"No example from the person's labels could be had ({why}), so at {at} the model is shown each "
            "picture without one.")})


def _example_of(got: dict) -> dict | None:
    """teacher_example's answer as the walk shows it, {"b64", "boxes", "words", "item_id"}; None: none to show.

    More boxes than one answer is taken with is no example: cut to the first
    of them, it would box the top of its picture and leave the rest.
    """
    boxes = got.get("boxes")
    kept = _boxes_in(boxes) if isinstance(boxes, list) and len(boxes) <= BOXES_PER_LOOK else []
    if got.get("error") or not got.get("image_base64") or got.get("units", UNITS) != UNITS or not kept:
        return None
    return {"b64": got["image_base64"], "boxes": kept, "words": _example_words(got),
            "item_id": str(got.get("item_id") or "")}


def _target_of(w: Walk, given: str) -> str:
    """What is looked for: as the call said it, or as filed with the zoom, or the instruction itself."""
    return (" ".join(str(given or "").split())[:200] or str((w.plan.get("filed") or {}).get("target") or "")
            or " ".join(str(w.instruction or "").split())[:200])


def _class_of(plan: dict, class_name: str, filed: dict) -> int | None:
    """The class to write as: the one named, if the project has it; else what was filed; else the teachers'."""
    name = " ".join(str(class_name or "").split()).lower()
    for c in plan.get("classes") or []:
        if name and str(c.get("name") or "").strip().lower() == name:
            return int(c["id"])
    got = filed.get("class_id")
    return int(got) if isinstance(got, int) and not isinstance(got, bool) else None


def _blank_png(width: int, height: int) -> str:
    """An all-background mask, as base64 PNG: what takes a written image back to having nothing on it."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (int(width), int(height)), 0).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _flag(w: Walk, iid: str, why: str) -> bool:
    """Hand an image to a person, with the reason; put down for the rest of the run either way."""
    got = await _call(w, "mark_review", {"item_ids_json": json.dumps([iid]), "reason": why[:200], "review": True})
    flagged = (bool(got.get("updated")) and not got.get("error")
               and iid not in (got.get("not_flagged") or got.get("teachers") or []))
    w.done_with.add(iid)
    # Flagged again, as the loop's own mark_review counts it: a flag taken off
    # to redo the image is answered.
    w.redo.discard(iid)
    if flagged:
        w.reviewed.add(iid)
        w.ever_flagged.add(iid)
        w.handed_over[iid] = why[:160]
    w.progress.moved(f"flagged {iid}")
    return flagged


def _wrote(w: Walk, iid: str) -> None:
    """An image written: the run's books, as the loop keeps them after write_kept.

    redo and handed_over too: an image whose flag the model took off and the
    walk then wrote is redone. Left in redo, the run was told after the walk
    that it had not been, and carried on until the no-progress guard ended it.
    """
    if iid not in w.written:
        w.written.append(iid)
    w.done_with.add(iid)
    w.reviewed.discard(iid)
    w.redo.discard(iid)
    w.handed_over.pop(iid, None)
    w.kept_high.pop(iid, None)
    w.progress.moved(f"wrote {iid}")


async def _open(w: Walk, iid: str, view: str, **more) -> dict | None:
    """One view of an image, asked once more after a failure; None when both failed.

    A trainer API that missed one request answers the next, and a part that
    is not opened is a part nobody looked at. With no ruler: measured with
    the boxes asked for as 0-1000 of the picture, the model found the most
    objects with none drawn. The size of one object comes in the walk's
    units (grid_units).
    """
    for _ in range(2):
        pic = await _call(w, "image_view", {"filename": iid, "view": view, "grid": False, "grid_units": UNITS,
                                            **more})
        if not pic.get("error") and pic.get("image_base64"):
            return pic
    return None


async def _accept(w: Walk, iid: str, view: str, boxes: list, st: dict, **more) -> int:
    """Hand boxes read off one view to SAM through the bridge, in batches; how many were kept.

    SAM failing to run SAM_FAILURES_STOP calls in a row ends the walk
    (_no_sam): on a machine that could not download its model, a rehearsal
    took its ten minutes and reported that Fit found none of the person's.
    """
    took = 0
    for i in range(0, len(boxes), ACCEPT_BATCH):
        batch = boxes[i:i + ACCEPT_BATCH]
        got = await _call(w, "accept_masks", {"item_id": iid, "boxes_json": json.dumps(batch), "view": view,
                                              "units": UNITS, **more})
        if got.get("error"):
            st["lost"] += len(batch)                     # seen and never asked of SAM: said with the image
            if _sam_down(got["error"]):
                w.sam_failed_in_a_row += 1
                if w.sam_failed_in_a_row >= SAM_FAILURES_STOP:
                    raise _no_sam(w, got["error"])
            continue
        w.sam_failed_in_a_row = 0
        st["sent"] += len(batch)
        took += int(got.get("accepted") or 0)
        st["refused"] += int(got.get("refused") or 0)
        # Cut by an inner edge, or kept already from a neighbour: taken in the
        # other part, not missed. On the whole picture, a box round a group of
        # what the parts kept: each of them taken in its part.
        st["skipped"] += int(got.get("skipped") or 0)
        st["kept"] = int(got.get("kept_so_far") or 0)
    return took


def _words(w: Walk, where: str, rung: str, pic: dict, after: str = "") -> str:
    """What is said with a picture: where it is, in words, and how large one object looks in it.

    The size is on the scale the boxes are asked in, 0 to 1000 across the
    picture (the brief says so): a size in pixels beside boxes on that scale
    is two scales for one answer.
    """
    head = "The whole picture" if rung == "fit" else f"This part: {where}, at {rung}"
    return head + after + "." + _one_said(pic, "One of them")


def _one_said(pic: dict, who: str) -> str:
    """' One of them looks about WxH here.', from the picture's one_object_looks_like; '' without it."""
    one = pic.get("one_object_looks_like")
    return (f" {who} looks about {int(one[0])}x{int(one[1])} here."
            if isinstance(one, list) and len(one) == 2 else "")


def _large_on(large: list | None, pic: dict) -> list[int] | None:
    """Too large for a part (large_from_px, px of the Fit view), in 0-1000 of the whole picture as shown.

    Each axis by its own side: the Fit view's sides are rounded to 32 apart.
    """
    width, height = int(pic.get("width") or 0), int(pic.get("height") or 0)
    if not large or not (width and height):
        return None
    return [max(1, int(round(NORM * float(large[0]) / width))), max(1, int(round(NORM * float(large[1]) / height)))]


def _joined(w: Walk, reasons: list[str]) -> str:
    return w.say("\u3002", "; ").join(r for r in reasons if r)


def _doubts(w: Walk, wrote: dict) -> list[str]:
    """What the last write of an image said is worth a person's eyes: far short of the teachers, or in pieces.

    The model acted on these in the old way (needs_review, then a look, then
    a flag); in a walk it never sees a write's answer, so the walk does.
    """
    out = []
    objects, expected = int(wrote.get("objects") or 0), int(wrote.get("expected_per_frame") or 0)
    if wrote.get("hint") or (wrote.get("needs_review") and expected and objects * 2 < expected):
        out.append(w.say(f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u306f1\u679a\u306b{expected}\u500b\u3042\u308b\u306e\u306b\u3001{objects}\u500b\u3057\u304b\u66f8\u304d\u8fbc\u3081\u3066\u3044\u307e\u305b\u3093",
                         f"written with {objects} objects where the teachers show {expected}"))
    if wrote.get("in_pieces"):
        out.append(w.say(f"\u7269\u4f53{objects}\u500b\u304c{wrote.get('blobs')}\u500b\u306e\u304b\u305f\u307e\u308a\u306b\u5206\u304b\u308c\u3066\u66f8\u304d\u8fbc\u307e\u308c\u3066\u3044\u307e\u3059\uff081\u3064\u306e\u7269\u4f53\u306e\u90e8\u5206\u304c\u96e2\u308c\u305f\u307e\u307e\u3067\u3059\uff09",
                         f"written as {wrote.get('blobs')} blobs for {objects} objects: parts of one object "
                         "kept apart"))
    if wrote.get("needs_review") and not out:
        out.append(w.say("\u66f8\u304d\u8fbc\u307f\u306e\u7d50\u679c\u3001\u8981\u78ba\u8a8d\u3068\u3055\u308c\u307e\u3057\u305f", "the write said it needs a review"))
    return out


def _gaps(w: Walk, st: dict, parts: int, eff: str) -> list[str]:
    """What of an image was never looked at, or never asked of SAM: each is said with the image."""
    out = []
    if st["unseen"]:
        out.append(w.say(f"{eff} \u306e\u90e8\u5206{parts}\u679a\u306e\u3046\u3061{st['unseen']}\u679a\u3092\u898b\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f",
                         f"{st['unseen']} of {parts} parts not looked at, at {eff}"))
    if st["overview_unseen"]:
        out.append(w.say("\u90e8\u5206\u306b\u53ce\u307e\u3089\u306a\u3044\u5927\u304d\u306a\u7269\u4f53\u3092\u63a2\u3059\u305f\u3081\u306e\u753b\u50cf\u5168\u4f53\u3092\u898b\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f",
                         "the whole picture was not looked at for objects too large for a part"))
    if st["lost"]:
        out.append(w.say(f"\u898b\u3064\u3051\u305f\u30dc\u30c3\u30af\u30b9\u306e\u3046\u3061{st['lost']}\u500b\u3092 SAM \u306b\u6e21\u305b\u307e\u305b\u3093\u3067\u3057\u305f",
                         f"{st['lost']} boxes found could not be handed to SAM"))
    return out


def _stats() -> dict:
    """What is counted of one image as it is walked, and said with it."""
    return {"parts": 0, "unseen": 0, "sent": 0, "kept": 0, "objects": 0, "refused": 0, "skipped": 0,
            "lost": 0, "large": 0, "overview_unseen": False, "reviewed": 0, "unreviewed": 0,
            "rewritten": 0, "review_left": None}


async def _plan_image(w: Walk, iid: str, rung: str) -> dict:
    """How one image is walked at the rung: {"eff", "views", "where_of", "large"}, or {"flagged": why}.

    An image the bridge cannot plan is flagged here; _unplanned ends the walk
    when that happens to images in a row.
    """
    plan = await _call(w, "view_plan", {"filename": iid, "rung": rung})
    got = _planned(plan)
    if got is None:
        why = str(plan.get("error") or w.say("\u90e8\u5206\u304c\u3042\u308a\u307e\u305b\u3093", "no views"))
        await _flag(w, iid, w.say(f"\u90e8\u5206\u306b\u5206\u3051\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f\uff1a{why}", f"the walk could not plan it: {why}"))
        return {"flagged": why}
    w.unplanned_in_a_row = 0
    return got


def _planned(plan: dict) -> dict | None:
    """view_plan's answer for one image as the walk reads it: {"eff", "views", "where_of", "large"}; None: no views.

    One reading for the walk of an image and the rehearsal of a zoom on one
    of the person's (_rehearse_rung): read twice, the two would differ the
    first time either changed.
    """
    views = [v for v in (plan.get("views") or []) if isinstance(v, dict) and v.get("view")]
    if plan.get("error") or not views:
        return None
    eff = str(plan.get("rung") or "fit")
    large = plan.get("large_from_px")
    large = ([int(large[0]), int(large[1])]
             if eff != "fit" and isinstance(large, list) and len(large) == 2 else None)
    return {"eff": eff, "views": [str(v["view"]) for v in views], "large": large,
            "where_of": {str(v["view"]): str(v.get("where") or "") for v in views}}


def _unplanned(w: Walk, why: str) -> None:
    """One more image the bridge could not plan; a bridge that has stopped answering fails every image the same way."""
    w.unplanned_in_a_row += 1
    if w.unplanned_in_a_row >= WALK_LOOK_FAILURES_STOP:
        raise _Ended("failed", w.say(
            f"{w.unplanned_in_a_row}\u679a\u7d9a\u3051\u3066\u753b\u50cf\u3092\u90e8\u5206\u306b\u5206\u3051\u3089\u308c\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3053\u3053\u3067\u6b62\u3081\u307e\u3057\u305f\uff08{why}\uff09\u3002",
            f"{w.unplanned_in_a_row} images in a row could not be planned, so the walk stopped here "
            f"({why})."))


def _refused_too_many(st: dict) -> bool:
    """WALK_REFUSALS_MIN refused boxes on the image, or half of those sent if that is more: put it down."""
    return st["refused"] >= max(WALK_REFUSALS_MIN, st["sent"] / 2.0)


def _put_down_said(w: Walk, st: dict, eff: str) -> str:
    return w.say(f"{st['sent']}\u500b\u306e\u30dc\u30c3\u30af\u30b9\u306e\u3046\u3061{st['refused']}\u500b\u304c\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u7269\u4f53\u304b\u3089\u6c42\u3081\u305f\u8a31\u5bb9\u7bc4\u56f2\u306b\u5408\u308f\u305a\u65ad\u3089\u308c\u305f\u305f\u3081\u3001"
                 f"{eff} \u3067\u6b8b\u3057\u305f\u3082\u306e\u3092\u66f8\u304d\u8fbc\u3093\u3067\u3001\u3053\u306e\u753b\u50cf\u3092\u9014\u4e2d\u3067\u624b\u653e\u3057\u307e\u3057\u305f",
                 f"{st['refused']} of {st['sent']} boxes refused by the teacher band: "
                 f"put down at {eff}, with what was kept written")


def _nothing_kept(w: Walk, st: dict, eff: str, put_down: str, gaps: list[str]) -> str:
    """Why an image the walk kept nothing on is flagged."""
    if put_down:
        return put_down
    if not st["parts"]:
        return w.say(f"{eff} \u306e\u90e8\u5206\u30921\u3064\u3082\u898b\u3089\u308c\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u4f55\u3082\u66f8\u304d\u8fbc\u3093\u3067\u3044\u307e\u305b\u3093",
                     f"none of its parts could be looked at, at {eff}: the walk wrote nothing")
    return _joined(w, [w.say(f"{eff} \u3067\u4f55\u3082\u898b\u3064\u304b\u3089\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u4f55\u3082\u66f8\u304d\u8fbc\u3093\u3067\u3044\u307e\u305b\u3093",
                             f"nothing found at {eff}: the walk wrote nothing")] + gaps)


async def _first_write(w: Walk, iid: str, wrote: dict, st: dict) -> str:
    """What the first write of an image made of it: written, unchanged, theirs, or flagged when it failed."""
    if wrote.get("written"):
        _wrote(w, iid)
        st["objects"] = int(wrote.get("objects") or st["kept"])
        return "written"
    if wrote.get("unchanged") or wrote.get("written_before") or wrote.get("already_there"):
        w.done_with.add(iid)
        st["objects"] = st["kept"]                       # what is on it is what was kept
        return "unchanged"
    if "person" in str(wrote.get("why") or ""):
        w.done_with.add(iid)                             # a person's mask or clean mark: theirs
        return "theirs"
    why = str(wrote.get("error") or wrote.get("why"))
    await _flag(w, iid, w.say(f"\u66f8\u304d\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f\uff1a{why}", f"not written: {why}"))
    return "flagged"


async def _judge(w: Walk, iid: str, st: dict, last: dict, put_down: str, gaps: list[str]) -> str:
    """A written image after its review: flagged with every reason a person should look at it, or written."""
    reasons = []
    if put_down:
        reasons.append(put_down)
    elif not st["kept"]:
        reasons.append(w.say("\u898b\u76f4\u3057\u3067\u6b63\u3057\u3044\u30de\u30b9\u30af\u304c1\u3064\u3082\u306a\u304f\u3001\u4ee3\u308f\u308a\u306e\u3082\u306e\u3082\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f",
                             "the review found none of the masks right, and nothing in their place"))
    elif st["review_left"]:
        missed, wrong = st["review_left"]
        reasons.append(w.say(
            f"{st['rewritten']}\u56de\u66f8\u304d\u76f4\u3057\u305f\u3042\u3068\u3082\u3001\u898b\u76f4\u3057\u3067\u898b\u843d\u3068\u3057{missed}\u500b\u3068\u8aa4\u308a{wrong}\u500b\u304c"
            "\u898b\u3064\u304b\u308a\u307e\u3057\u305f",
            f"after {st['rewritten']} rewrites the review still saw {missed} missed and {wrong} wrong"))
    reasons += _doubts(w, last) + gaps
    if st["unreviewed"]:
        reasons.append(w.say(f"\u30de\u30b9\u30af\u306e\u3042\u308b\u90e8\u5206\u306e\u3046\u3061{st['unreviewed']}\u679a\u3092\u898b\u76f4\u305b\u307e\u305b\u3093\u3067\u3057\u305f",
                             f"{st['unreviewed']} parts with mask not looked at in review"))
    if reasons:
        await _flag(w, iid, _joined(w, reasons))
        return "flagged"
    return "written"


async def _let_go(w: Walk, iid: str, n: int, total: int, what: str, st: dict, interrupted: bool) -> None:
    """The end of an image, however it came: the bridge lets go of its masks, and the screen is told.

    Kept for the review, the masks would otherwise stay in memory for the
    rest of the run. An image the run was stopped in is not shown.
    """
    await _call(w, "kept_drop", {"item_id": iid, "boxes_json": "all"})
    w.results.append((iid, what, dict(st)))
    if not interrupted:
        await _shown(w, iid, n, total, what, st)


def _review_views(wrote: dict, eff: str, large: list | None, st: dict) -> list[str]:
    """The views a written image is checked on, in order: the whole picture first when large ones were taken off it."""
    check = list(wrote.get("views_with_mask") or []) or (["fit"] if eff == "fit" else [])
    if bool(large) and bool(st["large"]) and eff != "fit":
        check = ["fit"] + [v for v in check if v != "fit"]
    return check


def _large_only(boxes: list, large: list) -> list:
    """The boxes at least as wide or as tall as an object too large for a part."""
    return [b for b in boxes if b[2] - b[0] >= large[0] or b[3] - b[1] >= large[1]]


def _reestimate(w: Walk, seconds: float, first: dict, rest: list[dict]) -> None:
    """The time left, counted again from the first image.

    The estimate the person agreed to was counted from defaults; the first
    image is a measurement of this machine and model.
    """
    per_part = seconds / max(1, int(first.get("parts") or 1))
    left = sum(max(1, int(e.get("parts") or 1)) for e in rest)
    minutes = max(1, int(round(per_part * left / 60.0)))
    w.emit({"type": "auto", "text": w.say(
        f"1\u679a\u76ee\u306b\u304b\u304b\u3063\u305f\u6642\u9593\u304b\u3089\u898b\u7a4d\u3082\u308a\u76f4\u3059\u3068\u3001\u6b8b\u308a\u306e{len(rest)}\u679a\u3067\u304a\u3088\u305d{minutes}\u5206\u3067\u3059\u3002",
        f"counted again from the first image: about {minutes} minutes for the "
        f"{len(rest)} images left")})


async def _walk_views(w: Walk, iid: str, eff: str, views: list[str], where_of: dict, large: list | None,
                      target: str, class_id: int | None, st: dict, example: bool = True,
                      stop: bool = True) -> tuple[list[str], str]:
    """Each view in order, its boxes handed to SAM, then the whole picture for the large ones: (looked at, put down).

    What a walk of an image and the rehearsal of a zoom on one of the
    person's (_rehearse_image) both are, up to the write: the same briefs,
    the same example, the same order and the same rules for what is kept.
    looked: the views the model answered, the whole picture after the parts
    included; put down: why the image was left part way, or "". example:
    False where the picture asked about is the one the example is made of.
    stop: False for a rehearsal, which goes on to the last view all the same
    and says put down at the end: stopped there, it was scored on the views
    before it alone, and a zoom whose boxes the band refused by the dozen
    passed on the few of theirs in the first part (_passes fails it instead).
    """
    brief = (WALK_BRIEF_FIT.format(target=target) if eff == "fit"
             else WALK_BRIEF.format(rung=eff, target=target))
    put_down, looked = "", []
    for v in views:
        if await w.gate():
            raise _Ended("stopped")
        pic = await _open(w, iid, v)
        if pic is None:
            st["unseen"] += 1
            continue
        if v == "fit":
            w.images_seen[iid] = pic["image_base64"]
            w.seen_crop[iid] = None
        # The person's own picture at the rung this image is walked at,
        # before each one asked about: Fit's for an image looked at whole
        # -- under a zoom, one whose Fit is as close -- a part's for the
        # parts. Read each time: the model server may have refused it.
        got = await _look(w, brief, _words(w, where_of.get(v, ""), eff, pic),
                          pic["image_base64"], FOUND, w.examples.get((eff, class_id)) if example else None)
        if got is None:
            st["unseen"] += 1                            # the model server failed on it twice
            continue
        st["parts"] += 1
        looked.append(v)
        boxes = _boxes_in(got.get("boxes"))
        if boxes:
            await _accept(w, iid, v, boxes, st)
        if stop and _refused_too_many(st):
            put_down = _put_down_said(w, st, eff)
            break
    if large and not put_down:
        if await _overview(w, iid, target, large, st, class_id, example):
            looked.append("fit")
    if not put_down and _refused_too_many(st):
        put_down = _put_down_said(w, st, eff)
    return looked, put_down


async def _label_image(w: Walk, entry: dict, n: int, total: int, rung: str, target: str,
                       class_id: int | None, confirm: bool = True) -> None:
    """One image: every part in order, the large ones on the whole picture, one write, a review of the parts with mask.

    The parts first, and the whole picture after them: shown a stack of
    small objects shrunk to fit, the model boxed the whole stack as one large
    object, and with that kept first every part's box inside it was skipped
    as kept already -- a third of the objects written where every one was
    found at Fit. After the parts, the bridge skips a box on the whole
    picture round two or more of what they found (a group, not one object).

    A part that could not be opened, or that the model server failed on, is
    a part nobody looked at, and whatever lies in it is written as background.
    What was kept elsewhere is still written, and the image is flagged with
    how much was not seen -- as it is when the write itself says it is far
    short of the teachers or in pieces, or the review still finds fault after
    the last rewrite it is allowed. confirm: False for a sample, whose own
    question comes after it (_sample).
    """
    iid = str(entry["id"])
    st = _stats()
    what = "nothing"
    interrupted = False
    try:
        plan = await _plan_image(w, iid, rung)
        if plan.get("flagged"):
            what = "flagged"
            _unplanned(w, plan["flagged"])
            return
        eff, views, where_of, large = plan["eff"], plan["views"], plan["where_of"], plan["large"]
        _, put_down = await _walk_views(w, iid, eff, views, where_of, large, target, class_id, st)
        gaps = _gaps(w, st, len(views), eff)
        if not st["kept"]:
            await _flag(w, iid, _nothing_kept(w, st, eff, put_down, gaps))
            what = "flagged"
            return
        wrote = await _write(w, iid, class_id)
        what = await _first_write(w, iid, wrote, st)
        if what == "written":
            last = wrote
            if not put_down:
                last = await _review(w, iid, eff, where_of, target, wrote, class_id, st, large)
            what = await _judge(w, iid, st, last, put_down, gaps)
        if what == "written" and w.confirm_each and confirm:
            what = await _confirm(w, iid, entry, st, what)
    except BaseException:
        interrupted = True
        raise
    finally:
        await _let_go(w, iid, n, total, what, st, interrupted)


async def _speck_image(w: Walk, entry: dict, n: int, total: int, target: str) -> None:
    """One image of a project of specks: found on the whole picture, written, and looked at once.

    With the person's own, spot_detect takes their marks on w.like as the
    example and finds the like of them here; the model is not asked where.
    With nothing of theirs, the model is shown the whole picture, as Fit
    shows it, and asked for one point inside one mark -- one look -- and
    spot_detect measures that mark and finds the rest like it. Either way
    the detector's own picture of what it found, each find ringed, is looked
    at once after spot_write (_speck_judge). Nothing found, a mark the point
    missed, or a look that says it is wrong is the image flagged for a
    person; a picture of it goes on the screen as for objects.
    """
    iid = str(entry["id"])
    st = {**_stats(), "marks": 0}
    what, interrupted = "nothing", False
    try:
        if w.like:
            found = await _call(w, "spot_detect", {"item_id": iid, "like_item_id": w.like})
        else:
            found = await _speck_point(w, iid, target, st)
            if found is None:
                what = "flagged"
                return
        st["parts"] = 1
        if found.get("error") or not found.get("ready_to_write"):
            why = " ".join(str(found.get("error") or found.get("next")
                               or w.say("\u4f55\u3082\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f", "it found nothing")).split())[:200]
            await _flag(w, iid, w.say(f"spot_detect \u3067\u66f8\u304d\u8fbc\u3081\u308b\u3082\u306e\u304c\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f\uff1a{why}",
                                      f"spot_detect found nothing to write: {why}"))
            what = "flagged"
            return
        st["marks"] = st["kept"] = int(found.get("count") or 0)
        wrote = await _call(w, "spot_write", {"item_id": iid})
        what = await _first_write(w, iid, wrote, st)
        if what == "written":
            st["marks"] = int(wrote.get("spots") or st["marks"])
            what = await _speck_judge(w, iid, found, wrote, target, st)
        if what == "written" and w.confirm_each:
            what = await _confirm(w, iid, entry, st, what)
    except BaseException:
        interrupted = True
        raise
    finally:
        w.results.append((iid, what, dict(st)))
        if not interrupted:
            await _shown(w, iid, n, total, what, st)


async def _speck_point(w: Walk, iid: str, target: str, st: dict) -> dict | None:
    """With nothing of the person's: one point inside one mark, from the model, put to spot_detect; None when flagged.

    The whole picture as Fit shows it, which is what the point is read off:
    spot_detect is told so (from_width, from_height) and puts it back on the
    picture. A point off the picture, or none, is no mark: flagged.
    """
    if await w.gate():
        raise _Ended("stopped")
    pic = await _open(w, iid, "fit")
    if pic is None:
        st["unseen"] = 1
        await _flag(w, iid, w.say("\u753b\u50cf\u3092\u958b\u3051\u307e\u305b\u3093\u3067\u3057\u305f", "the picture could not be opened"))
        return None
    w.images_seen[iid] = pic["image_base64"]
    w.seen_crop[iid] = None
    got = await _look(w, SPECK_POINT_BRIEF.format(target=target), "The whole picture.", pic["image_base64"], POINT)
    if got is None:
        st["unseen"] = 1
        await _flag(w, iid, w.say("\u30e2\u30c7\u30eb\u304c\u5fdc\u7b54\u305b\u305a\u3001\u753b\u50cf\u3092\u898b\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f", "the model server failed on it: not looked at"))
        return None
    st["parts"] = 1
    point = got.get("point")
    if not (isinstance(point, list) and len(point) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and -0.02 * NORM <= float(v) <= NORM * VIEW_SLACK for v in point)):
        await _flag(w, iid, w.say("\u30e2\u30c7\u30eb\u304c\u70b9\u3092\u6307\u3055\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u4f55\u3082\u66f8\u304d\u8fbc\u3093\u3067\u3044\u307e\u305b\u3093",
                                  "the model pointed at no mark: the walk wrote nothing"))
        return None
    x, y = (int(round(min(max(float(v), 0.0), float(NORM)))) for v in point)
    return await _call(w, "spot_detect", {"item_id": iid, "x": x, "y": y, "units": UNITS,
                                          "from_width": int(pic.get("width") or 0),
                                          "from_height": int(pic.get("height") or 0)})


async def _speck_judge(w: Walk, iid: str, found: dict, wrote: dict, target: str, st: dict) -> str:
    """A written image of specks after its look: flagged with every reason a person should see it, or written."""
    reasons = []
    if wrote.get("needs_review"):
        expected = int(wrote.get("expected_per_frame") or 0)
        reasons.append(w.say(f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u306f1\u679a\u306b{expected}\u500b\u3042\u308b\u306e\u306b\u3001\u70b9\u3092{st['marks']}\u500b\u3057\u304b\u66f8\u304d\u8fbc\u3081\u3066\u3044\u307e\u305b\u3093",
                             f"{st['marks']} marks written where the teachers show {expected}"))
    right = await _speck_review(w, iid, found, target, st)
    if right is None:
        reasons.append(w.say("\u66f8\u304d\u8fbc\u3093\u3060\u70b9\u3092\u898b\u76f4\u305b\u307e\u305b\u3093\u3067\u3057\u305f", "what was written could not be looked at again"))
    elif not right:
        reasons.append(w.say("\u66f8\u304d\u8fbc\u307f\u5f8c\u306e\u898b\u76f4\u3057\u3067\u3001\u898b\u3064\u3051\u305f\u70b9\u304c\u6307\u793a\u3055\u308c\u305f\u3082\u306e\u3067\u306f\u306a\u3044\u304b\u3001\u591a\u304f\u3092\u898b\u843d\u3068\u3057\u3066\u3044\u308b\u3068\u5224\u65ad\u3055\u308c\u307e\u3057\u305f",
                             "the look after the write says the marks found are not those asked for, or many "
                             "are missed"))
    if reasons:
        await _flag(w, iid, _joined(w, reasons))
        return "flagged"
    return "written"


async def _speck_review(w: Walk, iid: str, found: dict, target: str, st: dict) -> bool | None:
    """One look at what spot_detect found, as its own picture rings it: right or not; None when not looked at.

    That picture is the whole image with each find ringed and close-ups of
    where it found most and least -- a mark missed is seen only at full
    size. Without it, the whole picture with the mask written on it.
    """
    if await w.gate():
        raise _Ended("stopped")
    sheet, words = found.get(SHEET_KEY), "What was found on the whole picture, each find ringed."
    if not sheet:
        pic = await _open(w, iid, "fit", overlay=True)
        sheet, words = (pic or {}).get("image_base64"), "The whole picture, with what was written on it."
    if not sheet:
        st["unreviewed"] += 1
        return None
    got = await _look(w, SPECK_REVIEW_BRIEF.format(target=target), words, sheet, RIGHT)
    if got is None or not isinstance(got.get("right"), bool):
        st["unreviewed"] += 1
        return None
    st["reviewed"] += 1
    return bool(got["right"])


async def _overview(w: Walk, iid: str, target: str, large: list, st: dict, class_id: int | None = None,
                    shown: bool = True) -> bool:
    """The whole picture after the parts, for objects too large to lie whole in any part: those are boxed here.

    Whether the model answered it. shown: False where the picture is the one
    the example is made of (_walk_views).
    """
    if await w.gate():
        raise _Ended("stopped")
    pic = await _open(w, iid, "fit")
    if pic is None:
        st["overview_unseen"] = True
        return False
    w.images_seen[iid] = pic["image_base64"]
    w.seen_crop[iid] = None
    big = _large_on(large, pic) or [NORM, NORM]
    said = f"The whole picture. Box only those at least {big[0]} wide or {big[1]} tall here."
    # Fit's example, its answer the large ones alone, said with the same
    # words: they ask for those, and an answer beside them boxing every object
    # of the person's would show the model the words are not meant. With none
    # of theirs that large -- the usual at a zoom, where the objects are small
    # -- none is shown: an answer of nothing says nothing of what one large
    # object is, may be read as "answer nothing", and is a second picture for
    # every image, at a look that was never measured with one.
    example = w.examples.get(("fit", class_id)) if shown else None
    big_ones = _large_only(example["boxes"], big) if example else []
    got = await _look(w, WALK_BRIEF_LARGE.format(target=target), said, pic["image_base64"], FOUND,
                      {**example, "boxes": big_ones, "words": said} if big_ones else None)
    if got is None:
        st["overview_unseen"] = True
        return False
    boxes = _large_only(_boxes_in(got.get("boxes")), big)
    if boxes:
        st["large"] += len(boxes)
        await _accept(w, iid, "fit", boxes, st)
    return True


async def _write(w: Walk, iid: str, class_id: int | None) -> dict:
    """write_kept, keeping what was kept for the review."""
    args: dict = {"item_id": iid, "keep": True}
    if class_id is not None:
        args["class_id"] = int(class_id)
    return await _call(w, "write_kept", args)


async def _review(w: Walk, iid: str, eff: str, where_of: dict, target: str, wrote: dict,
                  class_id: int | None, st: dict, large: list | None = None) -> dict:
    """Each part that holds the mask, the mask drawn on it, at the zoom it was labelled at; the last write.

    What it says was missed is added, what it says is wrong is let go of, and
    the image is written again: at most review_redos times, as the loop allows
    a redo after its own review. Only what lies in the part is touched. The
    parts are looked at once more after the last rewrite allowed, and what
    that look still finds is not applied -- nothing would write it, and the
    count said with the image would be of masks it does not have -- but kept
    in st["review_left"], for the image to be flagged with.

    What was taken from the whole picture is checked on it, among the large
    ones alone: such an object is cut by the edge of every part that holds
    it, and each part's review leaves it to its neighbour.
    """
    check = _review_views(wrote, eff, large, st)
    whole = bool(large) and bool(st["large"]) and eff != "fit"
    looked: set = set()
    rewrites, last = 0, wrote
    while check:
        final = rewrites >= w.review_redos
        changed: list[str] = []
        left = [0, 0]
        for v in check:
            if await w.gate():
                raise _Ended("stopped")
            big = whole and v == "fit"
            pic = await _open(w, iid, v, overlay=True)
            if pic is None:
                st["unreviewed"] += 1
                continue
            least = (_large_on(large, pic) or [NORM, NORM]) if big else None
            if big:
                got = await _look(w, REVIEW_BRIEF_LARGE.format(target=target),
                                  f"The whole picture, with the masks just written on it. Check only those at "
                                  f"least {least[0]} wide or {least[1]} tall here.",
                                  pic["image_base64"], CHECKED)
            else:
                got = await _look(w, REVIEW_BRIEF.format(target=target),
                                  _words(w, where_of.get(v, ""), "fit" if v == "fit" else eff, pic,
                                         ", with the mask just written on it"),
                                  pic["image_base64"], CHECKED)
            if got is None:
                st["unreviewed"] += 1
                continue
            st["reviewed"] += 1
            wrong = _boxes_in(got.get("wrong"))
            missed = _boxes_in(got.get("missed"))
            if least:
                wrong, missed = _large_only(wrong, least), _large_only(missed, least)
            if final:
                left[0] += len(missed)
                left[1] += len(wrong)
                continue
            if wrong:
                # On the whole picture a box round a large mask holds the
                # middles of small ones too: only the large go.
                gone = await _call(w, "kept_drop", {"item_id": iid, "view": v, "boxes_json": json.dumps(wrong),
                                                    "units": UNITS,
                                                    **({"min_w": least[0], "min_h": least[1]} if least else {})})
                if int(gone.get("dropped") or 0):
                    st["kept"] = int(gone.get("kept_so_far") or 0)
                    changed.append(v)
            if missed and await _accept(w, iid, v, missed, st):
                changed.append(v)
        looked.update(check)
        if final:
            st["review_left"] = left if (left[0] or left[1]) else None
            return last
        if not changed or not st["kept"]:
            return last
        again = await _write(w, iid, class_id)
        rewrites += 1
        if not again.get("written"):
            return last
        last = again
        st["rewritten"] += 1
        st["objects"] = int(again.get("objects") or st["kept"])
        w.progress.moved(f"wrote {iid}")
        check = list(dict.fromkeys(changed + [v for v in (again.get("views_with_mask") or []) if v not in looked]))
    return last


async def _confirm(w: Walk, iid: str, entry: dict, st: dict, what: str) -> str:
    """confirm_each: keep a written image, take it back, or stop -- asked unless the person said not to ask.

    A reply that says none of them plainly is asked once more; a second one
    leaves the image as written, flagged so the person finds it.
    """
    choices = [w.say("\u3053\u306e\u307e\u307e\u306b\u3059\u308b", "Keep it"), w.say("\u53d6\u308a\u6d88\u3059", "Take it back"), w.say("\u6b62\u3081\u308b", "Stop")]
    text = w.say(f"{iid} \u306b\u7269\u4f53\u3092{st['objects']}\u500b\u66f8\u304d\u8fbc\u307f\u307e\u3057\u305f\u3002\u3053\u306e\u307e\u307e\u306b\u3057\u307e\u3059\u304b\u3001\u53d6\u308a\u6d88\u3057\u307e\u3059\u304b\u3001\u6b62\u3081\u307e\u3059\u304b\u3002",
                 f"{iid}: {st['objects']} objects written. Keep it, take it back, or stop?") + _numbered(w, choices)
    asked, i, reply = text, None, ""
    for _ in range(2):
        will_ask = not w.no_more_questions["flag"] and w.ask is not None
        if will_ask:
            # Shown as a question only when it is one: one the loop answers itself
            # has nothing waiting on it, and a reply to it would be refused.
            w.emit({"type": "question", "item_id": iid, "text": asked, "choices": choices})
        reply = str(await w.ask_gate(asked, choices))
        if await w.gate():
            raise _Ended("stopped")
        if not will_ask or reply == NOBODY_ANSWERED:
            return what                     # the person said to go on, or nobody is there: kept as written
        w.progress.moved("the person answered")
        w.answers.append((asked, reply))
        i = _confirm_of(w, reply, choices)
        if i is not None:
            break
        asked = _again_please(w, reply, text)
    if i is None:
        quoted = " ".join(reply.split())[:80]
        await _flag(w, iid, w.say(f"\u66f8\u304d\u8fbc\u3093\u3060\u307e\u307e\u6b8b\u3057\u307e\u3057\u305f\u3002\u3053\u306e\u307e\u307e\u306b\u3059\u308b\u304b\u53d6\u308a\u6d88\u3059\u304b\u306e\u7b54\u3048\uff08\u300c{quoted}\u300d\uff09\u3092\u8aad\u307f\u53d6\u308c\u307e\u305b\u3093\u3067\u3057\u305f",
                                  f"kept as written: the answer to keep it or take it back (\"{quoted}\") "
                                  "could not be read"))
        return "flagged"
    if i == 2:
        raise _Ended("declined", w.say(f"\u6307\u793a\u3069\u304a\u308a\u3001{iid} \u306e\u3042\u3068\u3067\u6b62\u3081\u307e\u3057\u305f\u3002",
                                       f"Stopped after {iid}, as asked."))
    if i == 1:
        width, height = int(entry.get("w") or 0), int(entry.get("h") or 0)
        if width and height:
            got = await _call(w, "mask_put", {"item_id": iid, "mask_png_base64": _blank_png(width, height)})
            if not got.get("error") and got.get("status") == "written":
                if iid in w.written:
                    w.written.remove(iid)
                w.progress.moved(f"took back {iid}")
                return "taken back"
        await _flag(w, iid, w.say("\u53d6\u308a\u6d88\u3059\u3088\u3046\u6307\u793a\u3055\u308c\u307e\u3057\u305f\u304c\u3001\u53d6\u308a\u6d88\u305b\u307e\u305b\u3093\u3067\u3057\u305f",
                                  "the person asked for it to be taken back, and the walk could not"))
        return "flagged"
    return what


#: What became of an image, as said on a screen in Japanese.
_SAID_JA = {"written": "\u66f8\u304d\u8fbc\u307f\u307e\u3057\u305f", "unchanged": "\u524d\u3068\u540c\u3058\u305f\u3081\u66f8\u304d\u8fbc\u3093\u3067\u3044\u307e\u305b\u3093", "flagged": "\u8981\u78ba\u8a8d\u306e\u5370\u3092\u4ed8\u3051\u307e\u3057\u305f",
            "theirs": "\u4eba\u306e\u30de\u30b9\u30af\u306e\u305f\u3081\u3001\u305d\u306e\u307e\u307e\u306b\u3057\u307e\u3057\u305f", "taken back": "\u53d6\u308a\u6d88\u3057\u307e\u3057\u305f", "nothing": "\u4f55\u3082\u898b\u3064\u304b\u308a\u307e\u305b\u3093\u3067\u3057\u305f",
            "not unlabelled": "\u30de\u30b9\u30af\u304c\u3042\u308b\u304b\u3001\u3053\u306e\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306b\u306a\u3044\u753b\u50cf\u306e\u305f\u3081\u898b\u3066\u3044\u307e\u305b\u3093"}


async def _shown(w: Walk, iid: str, n: int, total: int, what: str, st: dict) -> None:
    """One picture of the image on the screen -- the whole of it, the mask on it -- and one line.

    The count is of the objects on the image -- what the last write put
    there -- not of what is held for it.
    """
    pic = await _call(w, "image_view", {"filename": iid, "view": "fit", "overlay": True, "grid": False})
    what_said = w.say(_SAID_JA.get(what, what), what)
    if pic.get("image_base64"):
        try:
            w.emit({"type": "image", "item_id": iid, "caption": f"{iid}: {what_said}",
                    "jpeg_b64": w.render(pic["image_base64"], [])})
        except Exception:                                # noqa: BLE001 - a picture missing is not the walk failing
            pass
    if "marks" in st:
        w.emit({"type": "auto", "text": w.say(f"{total}\u679a\u4e2d{n}\u679a\u76ee\uff08{iid}\uff09\uff1a\u70b9{st['marks']}\u500b\u3001{what_said}\u3002",
                                              f"image {n} of {total}, {iid}: {st['marks']} marks, {what_said}")})
        return
    w.emit({"type": "auto", "text": w.say(
        f"{total}\u679a\u4e2d{n}\u679a\u76ee\uff08{iid}\uff09\uff1a\u90e8\u5206{st['parts']}\u679a\u3001\u7269\u4f53{st['objects']}\u500b\u3001{what_said}\u3002",
        f"image {n} of {total}, {iid}: {st['parts']} parts, {st['objects']} objects, {what_said}")})


def _summary(w: Walk, rung: str, outcome: str) -> str:
    """What the run's own conversation is told: what was written and, first, what is for a person."""
    written = [i for i, what, _ in w.results if what == "written"]
    flagged = [i for i, what, _ in w.results if what == "flagged"]
    other = [f"{i} ({w.say(_SAID_JA.get(what, what), what)})" for i, what, _ in w.results
             if what not in ("written", "flagged")]
    looked = [i for i, what, _ in w.results if what != "not unlabelled"]
    who = "label_images" if w.driver == "harness" else w.say("\u90e8\u5206\u3054\u3068\u306e\u30e9\u30d9\u30eb\u4ed8\u3051", "the walk")
    at = (w.say("\uff08\u70b9\u3068\u3057\u3066\uff09", " as specks") if _specks(w) else
          w.say(f"\uff08\u500d\u7387 {rung}\uff09", f" at {rung}"))
    head = w.say(f"{who}{at}: {len(looked)} \u679a\u3092\u898b\u3066\u3001{len(written)} \u679a\u306b\u66f8\u304d\u307e\u3057\u305f\u3002",
                 f"{who}{at}: {len(looked)} images looked at, {len(written)} written.")
    parts = [head]
    if flagged:
        parts.append(w.say(f"\u4eba\u306b\u78ba\u8a8d\u3057\u3066\u3082\u3089\u3046\u753b\u50cf\uff08\u8981\u78ba\u8a8d\u306e\u5370\u3092\u4ed8\u3051\u307e\u3057\u305f\uff09: {', '.join(flagged)}\u3002",
                           f"For a person (flagged for review): {', '.join(flagged)}."))
    if written:
        parts.append(w.say(f"\u66f8\u3044\u305f\u753b\u50cf: {', '.join(written)}\u3002", f"Written: {', '.join(written)}."))
    if other:
        parts.append(w.say(f"\u305d\u306e\u307b\u304b: {', '.join(other)}\u3002", f"Others: {', '.join(other)}."))
    if outcome == "done":
        parts.append(w.say("\u6700\u5f8c\u306e\u5831\u544a\u3067\u306f\u3001\u4eba\u306b\u78ba\u8a8d\u3057\u3066\u3082\u3089\u3046\u753b\u50cf\u3092\u5148\u306b\u6319\u3052\u3066\u304f\u3060\u3055\u3044\u3002\u5370\u306e\u4ed8\u3044\u305f\u753b\u50cf\u306f\u4f5c\u308a\u76f4\u3055\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002",
                           "Report the images for a person first; do not redo the flagged ones."))
    text = " ".join(parts)
    return text if len(text) <= SUMMARY_CHARS else text[:SUMMARY_CHARS - 1] + "\u2026"


def _out(w: Walk, outcome: str, rung: str, said: str = "") -> dict:
    summary = _summary(w, rung, outcome)
    return {"outcome": outcome, "summary": ((said + " ") if said else "") + summary,
            "final": (said or summary).strip()}


def _names(raw: Any) -> list[str]:
    """Image ids as a call gives them: a list, a JSON list, or words between commas."""
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except ValueError:
                raw = text.strip("[]")
        if isinstance(raw, str):
            raw = [p for p in re.split("[,\\s\u3001]+", raw) if p]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return []
    return list(dict.fromkeys(str(v).strip().strip("\"'") for v in raw if str(v).strip()))


def _keys(names) -> set[str]:
    """What an image named by id or by its file name is matched by: the name, and the name without its extension."""
    out: set[str] = set()
    for n in names:
        out.add(str(n))
        out.add(_ID_EXT.sub("", str(n)))
    return out


async def _scope(w: Walk, item_ids: Any) -> tuple[dict, list[dict]]:
    """The project's walk plan, and the images to walk: those with no mask, as far as the run and the call name them.

    Images named that are not unlabelled are said so with the results.
    The bridge not answering ends the walk.
    """
    asked = _names(item_ids)
    plan = await _call(w, "view_plan", {})
    if plan.get("error"):
        raise _Ended("failed", w.say(f"\u672a\u30e9\u30d9\u30eb\u306e\u753b\u50cf\u306e\u4e00\u89a7\u3092\u8aad\u307f\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f\uff08{plan['error']}\uff09\u3002",
                                     f"The unlabelled images could not be read ({plan['error']})."))
    listed = [e for e in plan.get("unlabelled") or [] if isinstance(e, dict) and e.get("id")]
    # A run given its images, or a call naming some, is about those. The
    # command line names them, and a walk of the whole project after it
    # was told "label img7" took hours nobody had asked for.
    for scope in (w.only, asked):
        if scope:
            keys = _keys(scope)
            listed = [e for e in listed if str(e["id"]) in keys]
    have = {str(e["id"]) for e in listed}
    said_of = {r[0] for r in w.results}
    for name in asked or sorted(w.only):
        if not (_keys([name]) & (have | said_of)):
            w.results.append((name, "not unlabelled", {}))
    return plan, [e for e in listed if str(e["id"]) not in w.done_with]


async def _agree(w: Walk, filed: dict, rung: str, todo: list[dict]) -> None:
    """The person's go-ahead for walking these images, unless they gave it already; then the preparation, once a run.

    Anything but going ahead ends the walk (_Ended): stopped, declined, the
    zoom to be chosen again, or nobody to say.
    """
    ids = {str(e["id"]) for e in todo}
    if not ids <= w.consented_ids:
        said, reply = await _consent(w, filed, todo)
        if said == "stopped":
            raise _Ended("stopped")
        if said in ("nobody", "unanswered"):
            parts, minutes = _estimate(w, todo)
            at = _at(w, rung)
            raise _Ended("nobody", w.say(
                f"{at}\u9032\u3081\u3066\u3088\u3044\u304b\u3092\u7b54\u3048\u308b\u4eba\u304c\u3044\u306a\u3044\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093"
                f"\uff08{len(todo)}\u679a\u3001\u304a\u3088\u305d{minutes}\u5206\uff09\u3002\u30b3\u30de\u30f3\u30c9\u30e9\u30a4\u30f3\u304b\u3089\u5b9f\u884c\u3059\u308b\u5834\u5408\u306f\u3001--yes \u3092\u4ed8\u3051\u308b\u3068\u78ba\u8a8d\u306a\u3057\u3067\u9032\u3081\u307e\u3059\u3002",
                f"Nobody is here to say whether to go ahead {at} ({len(todo)} images, about {minutes} "
                f"minutes), so nothing was labelled. From the command line, --yes goes ahead without asking.")
                if said == "nobody" else w.say(
                f"{at}\u9032\u3081\u3066\u3088\u3044\u304b\u3068\u3044\u3046\u8cea\u554f\u306b\u8ab0\u3082\u7b54\u3048\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093"
                f"\uff08{len(todo)}\u679a\u3001\u304a\u3088\u305d{minutes}\u5206\uff09\u3002",
                f"Nobody answered whether to go ahead {at} ({len(todo)} images, about {minutes} "
                f"minutes), so nothing was labelled."))
        if said == "again":
            # Taken back, the next run finds no zoom filed (engages): the
            # program then rehearses the zooms again on the person's own, and
            # the model walking the parts asks about looking at each image
            # whole. Said as what that run does.
            got = await _call(w, "view_choose", {"rung": "none"})
            if got.get("error"):
                # Not taken back -- the bridge could not read or write the
                # project's context, or did not answer. Said as "taken back"
                # all the same, the next run found the zoom still filed and
                # asked about it, after this one had said it was gone.
                why = " ".join(str(got["error"]).split())[:200]
                raise _Ended("failed", w.say(
                    f"\u8a18\u9332\u3055\u308c\u3066\u3044\u305f\u500d\u7387\u3092\u53d6\u308a\u6d88\u305b\u307e\u305b\u3093\u3067\u3057\u305f\uff08{why}\uff09\u3002\u500d\u7387\u306f\u8a18\u9332\u3055\u308c\u305f\u307e\u307e\u3067\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002\u6b21\u306e\u5b9f\u884c\u3067\u306f\u3001\u3053\u306e\u500d\u7387\u3067\u9032\u3081\u3066\u3088\u3044\u304b\u3092\u3082\u3046\u4e00\u5ea6\u304a\u805e\u304d\u3057\u307e\u3059\u3002",
                    f"The zoom filed for this project could not be taken back ({why}), so it is still filed, "
                    "and nothing was labelled. The next run asks again whether to go ahead at it."))
            if w.driver == "harness" and _taught(w):
                raise _Ended("again", w.say(
                    "\u6307\u793a\u3069\u304a\u308a\u3001\u8a18\u9332\u3055\u308c\u3066\u3044\u305f\u500d\u7387\u3092\u53d6\u308a\u6d88\u3057\u307e\u3057\u305f\u3002\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002\u6b21\u306e\u5b9f\u884c\u3067\u306f\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u500d\u7387\u3092\u4f4e\u3044\u65b9\u304b\u3089\u8a66\u3057\u76f4\u3057\u3066\u6c7a\u3081\u307e\u3059\u3002",
                    "The zoom filed for this project is taken back, as asked, and nothing was labelled. The next run "
                    "chooses it again, rehearsing the zooms on the person's own images, lowest first."))
            raise _Ended("again", w.say(
                "\u6307\u793a\u3069\u304a\u308a\u3001\u8a18\u9332\u3055\u308c\u3066\u3044\u305f\u500d\u7387\u3092\u53d6\u308a\u6d88\u3057\u307e\u3057\u305f\u3002\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002\u6b21\u306e\u5b9f\u884c\u3067\u306f\u3001\u5404\u753b\u50cf\u3092\u5168\u4f53\u306e\u307e\u307e\uff08Fit \u3067\uff09\u898b\u308b\u304b\u3069\u3046\u304b\u3092\u304a\u805e\u304d\u3057\u307e\u3059\u3002",
                "The zoom filed for this project is taken back, as asked, and nothing was labelled. The next run "
                "asks whether to look at each image whole, as Fit shows it."))
        if said != "proceed":
            quoted = " ".join(reply.split())[:120]
            raise _Ended("declined", w.say(
                f"\u9032\u3081\u306a\u3044\u3068\u3044\u3046\u7b54\u3048\uff08\u300c{quoted}\u300d\uff09\u3060\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002" if quoted else
                "\u9032\u3081\u306a\u3044\u3068\u3044\u3046\u7b54\u3048\u3060\u3063\u305f\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
                f"The answer (\"{quoted}\") was not to go ahead, so nothing was labelled." if quoted else
                "The answer was not to go ahead, so nothing was labelled."))
        w.consented_ids |= ids
    if not w.prepared:
        await _prepare(w)
        w.prepared = True


def _rehearses(w: Walk, filed: dict) -> bool:
    """Whether this call chooses the zoom before it walks: none filed, and the program walking objects the person drew.

    The model walking the parts looks at each image whole, as it did; a
    project nobody drew on has nothing to rehearse on and is labelled the old
    way (walks); specks are found on the whole picture. Once a run, whatever
    came of it.
    """
    return (bool(filed.get("default")) and w.driver == "harness" and not _specks(w) and _taught(w)
            and w.rehearsed is None)


def _scores(t: dict) -> tuple[float, float, float]:
    """(recall, precision, F1) of one rung's rehearsal: their objects found, and what was kept that is on one of theirs."""
    objects, found, false = t["objects"], t["found"], t["false"]
    recall = found / float(objects) if objects else 0.0
    precision = found / float(found + false) if found + false else 0.0
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


def _passes(t: dict) -> bool:
    """Enough of theirs found, little on nothing -- and no picture put down: a walk at it would leave images part way."""
    recall, precision, _ = _scores(t)
    return (t["objects"] > 0 and recall >= RECALL_MIN and precision >= PRECISION_MIN
            and not t.get("put_down"))


def _rung_said(w: Walk, t: dict) -> str:
    """One rung's rehearsal on the screen, with the numbers it rests on."""
    recall, precision, _ = _scores(t)
    ok, on = _passes(t), ", ".join(t["on"])
    unscored = "; ".join(t.get("unscored") or [])[:200]
    down = t.get("put_down") or []
    down_ja = "\u3001".join(f"{i} \u3067\u306f\u30dc\u30c3\u30af\u30b9{sent}\u500b\u306e\u3046\u3061{refused}\u500b" for i, refused, sent in down)
    down_en = "; ".join(f"{refused} of {sent} boxes on {i}" for i, refused, sent in down)
    verdict = "\u6e80\u305f\u3057\u3066\u3044\u307e\u3059" if ok else "\u6e80\u305f\u3057\u3066\u3044\u307e\u305b\u3093"
    return w.say(
        f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\uff08{on}\uff09\u3092 {t['rung']} \u3067\u8a66\u3057\u307e\u3057\u305f\u3002\u4eba\u304c\u4ed8\u3051\u305f{t['objects']}\u500b\u306e\u3046\u3061{t['found']}\u500b\u3092\u898b\u3064\u3051\u3001"
        f"\u4eba\u304c\u4f55\u3082\u4ed8\u3051\u3066\u3044\u306a\u3044\u6240\u306b\u4ed8\u3051\u305f\u3082\u306e\u306f{t['false']}\u500b\u3067\u3057\u305f\u3002\u518d\u73fe\u7387 {recall:.2f}\u3001\u9069\u5408\u7387 {precision:.2f} \u3067\u3001"
        f"\u57fa\u6e96\uff08\u518d\u73fe\u7387 {RECALL_MIN:g}\u4ee5\u4e0a\u3001\u9069\u5408\u7387 {PRECISION_MIN:g}\u4ee5\u4e0a\uff09\u3092{verdict}\u3002"
        + (f"{down_ja}\u304c\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u7269\u4f53\u304b\u3089\u6c42\u3081\u305f\u8a31\u5bb9\u7bc4\u56f2\u306b\u5408\u308f\u305a\u65ad\u3089\u308c\u305f\u305f\u3081\u3001\u3053\u306e\u500d\u7387\u3067\u306f\u3053\u3046\u3057\u305f\u753b\u50cf\u3092\u9014\u4e2d\u3067\u624b\u653e\u3059\u3053\u3068\u306b\u306a\u308a\u307e\u3059\u3002"
           if down else "")
        + (f"\u6b21\u306e\u753b\u50cf\u306f\u63a1\u70b9\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff08{unscored}\uff09\u3002" if unscored else ""),
        f"Rehearsed at {t['rung']} on {on}: {t['found']} of their {t['objects']} found, {t['false']} kept on none "
        f"of theirs -- recall {recall:.2f}, precision {precision:.2f}: {'enough' if ok else 'not enough'} "
        f"(recall {RECALL_MIN:g} and precision {PRECISION_MIN:g} wanted)."
        + (f" Refused by the teacher band: {down_en} -- a walk at {t['rung']} puts such images down part way."
           if down else "")
        + (f" Not scored: {unscored}." if unscored else ""))


async def _ready_to_rehearse(w: Walk) -> str:
    """The band and SAM a rehearsal stands on in this run's bridge, as a walk at Fit has them; why not, or ""."""
    if await w.gate():
        raise _Ended("stopped")
    if w.band is None:
        # Not the shrink: nothing of a rehearsal is written. The walk after it
        # reads the teachers again with it, at the zoom chosen (_prepare).
        band = await _call(w, "teacher_band", {"calibrate_shrink": False})
        if band.get("error"):
            err = " ".join(str(band["error"]).split())[:160]
            return w.say(f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u8aad\u307f\u8fbc\u3081\u307e\u305b\u3093\u3067\u3057\u305f\uff1a{err}", f"the teachers could not be read: {err}")
        w.band = band
    cal = await _call(w, "calibrate_sam", {})
    if cal.get("error"):
        w.emit({"type": "failed", "step": 0, "name": "calibrate_sam", "text": str(cal["error"])[:300]})
        if _sam_down(cal["error"]):
            raise _no_sam(w, cal["error"])
    return ""


async def _rehearse_image(w: Walk, p: dict, target: str, class_id: int | None, shown: bool) -> dict:
    """The walk's own look at one of the person's pictures, written nowhere: {"objects", "found", "false", "looks", "put_down"} or {"why"}.

    Every view the bridge picked, in order, then the whole picture for the
    large ones (_walk_views) -- on to the last, however many boxes the band
    refused: put_down, [refused, sent] where a walk would have put it down
    part way; what was kept scored against their mask on the views looked
    at, of the class written when it is known (rehearse with views_json), and
    let go of whatever came of it. At a zoom of parts with none of them
    looked at, it is not scored: the whole picture after them counts only
    the objects too large for a part, and every other of theirs was missed
    by nobody looking.
    """
    t = p["t"]
    st = _stats()
    try:
        looked, put_down = await _walk_views(w, t, p["eff"], p["views"], p["where_of"], p["large"], target,
                                             class_id, st, shown, stop=False)
        if not looked:
            return {"why": w.say(f"{t}\uff1a\u898b\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f", f"{t}: not looked at")}
        if p["eff"] != "fit" and all(v == "fit" for v in looked):
            return {"why": w.say(f"{t}\uff1a\u90e8\u5206\u30921\u3064\u3082\u898b\u3089\u308c\u307e\u305b\u3093\u3067\u3057\u305f",
                                 f"{t}: none of its parts looked at")}
        args = {"item_id": t, "views_json": json.dumps(looked)}
        if class_id is not None:
            args["class_id"] = int(class_id)
        got = await _call(w, "rehearse", args)
        objects = got.get("objects")
        if (got.get("error") or not got.get("scored") or not isinstance(objects, int)
                or isinstance(objects, bool)):
            why = " ".join(str(got.get("error") or got.get("why") or "not scored").split())[:160]
            return {"why": f"{t}: {why}"}
        return {"objects": objects, "found": int(got.get("found") or 0), "false": int(got.get("false") or 0),
                "looks": len(looked), "put_down": [st["refused"], st["sent"]] if put_down else None}
    finally:
        # Kept for the score alone: nothing of it is written, and the walk of
        # a picture after this starts from nothing.
        await _call(w, "kept_drop", {"item_id": t, "boxes_json": "all"})


async def _rehearse_rung(w: Walk, rung: str, teachers: list[str], target: str, class_id: int | None) -> dict:
    """One rung rehearsed on some of teachers: {"rung", "on", "objects", "found", "false", "looks", "own", "put_down"}, or with "skipped": what is said.

    Planned at the class the walk writes, when it is known: scored against
    every class the person painted, a walk for one of two classes found at
    best half of "theirs", and a mask on the other class was a find.
    """
    shown = {str((w.examples.get((r, class_id)) or {}).get("item_id") or "") for r in ("fit", rung)} - {""}
    others = [t for t in teachers if t not in shown]
    out: dict = {"rung": rung, "on": [], "objects": 0, "found": 0, "false": 0, "looks": 0, "own": not others,
                 "unscored": [], "put_down": []}
    plans, why = [], []
    for t in others or teachers[:1]:
        if len(plans) >= REHEARSE_TEACHERS:
            break
        if await w.gate():
            raise _Ended("stopped")
        args: dict = {"filename": t, "rung": rung, "rehearse": True}
        if class_id is not None:
            args["class_id"] = int(class_id)
        got = await _call(w, "view_plan", args)
        p = _planned(got)
        if p is None:
            why.append(f"{t}: " + " ".join(str(got.get("error") or "no views").split())[:120])
            continue
        plans.append({"t": t, **p})
    if not plans:
        said = "; ".join(why)[:300]
        return {**out, "skipped": w.say(f"{rung} \u3067\u306f\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3092\u90e8\u5206\u306b\u5206\u3051\u3089\u308c\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u8a66\u305b\u307e\u305b\u3093\u3067\u3057\u305f\uff08{said}\uff09\u3002",
                                        f"{rung} could not be rehearsed: no image of the person's could be planned "
                                        f"at it ({said})")}
    if rung != "fit" and all(p["eff"] == "fit" for p in plans):
        # A rung at or below a picture's Fit is its Fit: the same looks again.
        return {**out, "skipped": w.say(f"{rung} \u3067\u306f\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u304c Fit \u3088\u308a\u5927\u304d\u304f\u5199\u3089\u306a\u3044\u305f\u3081\u3001\u8a66\u3057\u76f4\u3057\u307e\u305b\u3093\u3002",
                                        f"{rung} shows the person's images no closer than Fit: not rehearsed again")}
    looks = sum(len(p["views"]) + (1 if p["large"] else 0) for p in plans)
    if looks > REHEARSE_LOOKS_MAX:
        return {**out, "skipped": w.say(
            f"{rung} \u3092\u8a66\u3059\u306b\u306f\u30e2\u30c7\u30eb\u304c{looks}\u56de\u898b\u308b\u5fc5\u8981\u304c\u3042\u308a\u3001\u4e0a\u9650\u306e{REHEARSE_LOOKS_MAX}\u56de\u3092\u8d85\u3048\u308b\u305f\u3081\u3001\u8a66\u3057\u307e\u305b\u3093\u3002",
            f"{rung} would take {looks} looks to rehearse, more than {REHEARSE_LOOKS_MAX}: not rehearsed")}
    for p in plans:
        got = await _rehearse_image(w, p, target, class_id, not out["own"])
        if got.get("why"):
            why.append(got["why"])
            continue
        out["on"].append(p["t"])
        for k in ("objects", "found", "false", "looks"):
            out[k] += got[k]
        if got.get("put_down"):
            out["put_down"].append((p["t"], *got["put_down"]))
    out["unscored"] = why
    if not out["on"]:
        said = "; ".join(why)[:300]
        return {**out, "skipped": w.say(f"{rung} \u3067\u8a66\u3057\u305f\u7d50\u679c\u3092\u63a1\u70b9\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff08{said}\uff09\u3002",
                                        f"what was found at {rung} could not be scored ({said})")}
    return out


async def _decide_rung(w: Walk, target: str, class_name: str, class_id: int | None) -> None:
    """The zoom to label at, where none is filed: the walk rehearsed on the person's own, lowest first, and filed.

    Each rung in turn (RUNGS), Fit first, on up to REHEARSE_TEACHERS of their
    pictures -- not the one their example is made of: a model shown its own
    answer is not measured -- at the parts the bridge picks on each (view_plan
    with rehearse=true): those holding the most of their objects whole, and
    some holding none, where a box is one on nothing. Each is looked at as
    the walk looks (_walk_views): the same briefs, the same example before
    each picture, the same order, every box to SAM and the band, the edge,
    duplicate and group rules, and the whole picture after the parts for the
    large ones. Nothing is written: what was kept is scored against their
    mask, object by object and one to one (rehearse with views_json), and let
    go of. The first rung whose recall reaches RECALL_MIN and precision
    PRECISION_MIN, with no picture of theirs put down at it -- the lowest --
    is filed (view_choose) with its numbers and said; none above it is
    tried. With none passing up to 400%, the rung with the best F1 -- the
    lower on a tie -- is filed as not passing, and the first image is
    labelled at it as a sample before the person is asked (_sample), which
    takes the filing back unless they go ahead.

    A rung at or below the Fit of every picture rehearsed on shows them as
    Fit does and is not rehearsed again; one whose plan comes to more than
    REHEARSE_LOOKS_MAX looks is not rehearsed. Each is said. Where nothing
    could be rehearsed -- the teachers unread, no picture of theirs to plan,
    nothing scored -- that is said, nothing is filed, and the images are
    looked at whole, as before this was here. A zoom of parts chosen and not
    filed ends the run ("failed"): the walk at it stands on the filing.
    """
    w.rehearsed = {"failed": True}
    why = await _ready_to_rehearse(w)
    tried: list[dict] = []
    if not why:
        teachers = [str(t) for t in (w.band or {}).get("teachers") or [] if t]
        own_said = False
        for rung in RUNGS if teachers else ():
            if await w.gate():
                raise _Ended("stopped")
            await _examples(w, rung, class_id)
            t = await _rehearse_rung(w, rung, teachers, target, class_id)
            if t["own"] and t["on"] and not own_said:
                own_said = True
                w.emit({"type": "auto", "text": w.say(
                    f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u306f\u3069\u308c\u3082\u624b\u672c\u306b\u4f7f\u3063\u3066\u3044\u308b\u305f\u3081\u3001{t['on'][0]} \u3067\u500d\u7387\u3092\u8a66\u3057\u307e\u3059\u3002\u3053\u306e\u753b\u50cf\u3067\u306f\u624b\u672c\u3092\u898b\u305b\u307e\u305b\u3093\u3002",
                    f"Every image the person drew is one their example is made of, so the zoom is rehearsed on "
                    f"{t['on'][0]}, without the example shown before it there.")})
            if t.get("skipped"):
                w.emit({"type": "auto", "text": t["skipped"]})
                continue
            tried.append(t)
            w.emit({"type": "auto", "text": _rung_said(w, t)})
            if _passes(t):
                break
        if not tried:
            why = w.say("\u3069\u306e\u500d\u7387\u3082\u3001\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u305b\u307e\u305b\u3093\u3067\u3057\u305f", "no zoom could be rehearsed on the person's images")
    if why:
        w.rehearsed = {"failed": why}
        w.emit({"type": "auto", "text": w.say(
            f"\u500d\u7387\u3092\u8a66\u305b\u306a\u304b\u3063\u305f\u305f\u3081\uff08{why}\uff09\u3001\u5404\u753b\u50cf\u3092\u5168\u4f53\u306e\u307e\u307e\uff08Fit \u3067\uff09\u898b\u307e\u3059\u3002\u500d\u7387\u306f\u8a18\u9332\u3057\u307e\u305b\u3093\u3002",
            f"The zoom could not be rehearsed ({why}), so each image is looked at whole, as Fit shows it, and "
            "nothing is filed.")})
        return
    passed = next((t for t in tried if _passes(t)), None)
    best = passed or max(tried, key=lambda t: _scores(t)[2])      # the first, the lower, on a tie
    recall, precision, _ = _scores(best)
    rung, how = best["rung"], ("rehearsal" if passed else "rehearsal, not passing")
    args: dict = {"rung": rung, "how": how, "recall": round(recall, 3), "precision": round(precision, 3),
                  "target": target[:200]}
    named = " ".join(str(class_name or "").split())[:80]
    if named:
        args["class_name"] = named
    got = await _call(w, "view_choose", args)
    record = got.get("filed") if isinstance(got.get("filed"), dict) and not got.get("error") else None
    err = "" if record else " ".join(str(got.get("error") or "nothing came back").split())[:160]
    nums = f"{recall:.2f}", f"{precision:.2f}"
    if not record and rung != "fit":
        # A walk of parts stands on its filing in the bridge: view_plan counts
        # each image's parts at the zoom filed, and teacher_band measures the
        # shrink with SAM shown parts of it (at_filed_zoom). Walked unfiled, the
        # question said one part an image and a few minutes for a walk of
        # hours, and every mask was shrunk as measured on whole frames while
        # SAM was asked on parts. Fit needs no filing: nothing filed is Fit.
        best_ja, best_en = ("", "") if passed else ("\u3001\u57fa\u6e96\u306b\u306f\u5c4a\u3044\u3066\u3044\u307e\u305b\u3093", ", the best, short of what is wanted")
        raise _Ended("failed", w.say(
            f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u3057\u3066\u500d\u7387 {rung} \u3092\u9078\u3073\u307e\u3057\u305f\u304c\uff08\u518d\u73fe\u7387 {nums[0]}\u3001\u9069\u5408\u7387 {nums[1]}{best_ja}\uff09\u3001\u8a18\u9332\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff08{err}\uff09\u3002"
            "\u90e8\u5206\u3054\u3068\u306b\u898b\u3066\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u308b\u3068\u304d\u306f\u3001\u8a18\u9332\u3057\u305f\u500d\u7387\u3067\u6642\u9593\u3092\u898b\u7a4d\u3082\u308a\u3001\u305d\u306e\u90e8\u5206\u3067 SAM \u3092\u6e2c\u308b\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002\u6b21\u306e\u5b9f\u884c\u3067\u8a66\u3057\u76f4\u3057\u307e\u3059\u3002",
            f"The rehearsal chose {rung} (recall {nums[0]}, precision {nums[1]}{best_en}), but it could not be filed "
            f"({err}). A walk of parts stands on its filing -- how long it takes is counted at it, and SAM is "
            "measured on its parts -- so nothing was labelled. The next run rehearses again."))
    # At Fit the walk goes on whether or not it could be filed: chosen for
    # this run all the same, and the next run rehearses again.
    w.plan = {**w.plan, "filed": dict(record) if record else {
        "rung": rung, "how": how, "recall": round(recall, 3), "precision": round(precision, 3), "taught": True,
        "target": target[:200] or None}}
    w.rehearsed = {"rung": rung, "recall": recall, "precision": precision, "passed": passed is not None,
                   "filed": bool(record), "on": list(best["on"])}
    tail = (w.say("\u8a18\u9332\u3057\u305f\u306e\u3067\u3001\u6b21\u56de\u4ee5\u964d\u306e\u5b9f\u884c\u3067\u3082\u4f7f\u3044\u307e\u3059\u3002", " Filed for the runs after this one.") if record else
            w.say(f"\u8a18\u9332\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\uff08{err}\uff09\u3002", f" Not filed: {err}."))
    if passed:
        w.emit({"type": "auto", "text": w.say(
            f"\u500d\u7387\u306f {rung} \u306b\u3057\u307e\u3057\u305f\u3002\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u3057\u3001\u57fa\u6e96\u3092\u6e80\u305f\u3057\u305f\u6700\u3082\u4f4e\u3044\u500d\u7387\u3067\u3059\uff08\u518d\u73fe\u7387 {nums[0]}\u3001\u9069\u5408\u7387 {nums[1]}\uff09\u3002{tail}",
            f"The zoom is {rung}: the lowest at which the model found the person's own (recall {nums[0]}, "
            f"precision {nums[1]}).{tail}")})
        return
    w.emit({"type": "auto", "text": w.say(
        f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067400%\u307e\u3067\u8a66\u3057\u307e\u3057\u305f\u304c\u3001\u57fa\u6e96\uff08\u518d\u73fe\u7387 {RECALL_MIN:g}\u4ee5\u4e0a\u3001\u9069\u5408\u7387 {PRECISION_MIN:g}\u4ee5\u4e0a\uff09\u3092\u6e80\u305f\u3059\u500d\u7387\u306f\u3042\u308a\u307e\u305b\u3093\u3067\u3057\u305f\u3002"
        f"\u6700\u3082\u826f\u304b\u3063\u305f\u306e\u306f {rung}\uff08\u518d\u73fe\u7387 {nums[0]}\u3001\u9069\u5408\u7387 {nums[1]}\uff09\u3067\u3001\u57fa\u6e96\u306b\u5c4a\u304b\u306a\u304b\u3063\u305f\u3053\u3068\u3068\u5408\u308f\u305b\u3066\u8a18\u9332\u3057\u307e\u3059\u3002"
        f"\u6700\u521d\u306e\u753b\u50cf\u3092\u3053\u306e\u500d\u7387\u3067\u8a66\u3057\u306b\u30e9\u30d9\u30eb\u4ed8\u3051\u3057\u3066\u898b\u3066\u3044\u305f\u3060\u304d\u3001\u6b8b\u308a\u3092\u9032\u3081\u308b\u304b\u3069\u3046\u304b\u3092\u304a\u805e\u304d\u3057\u307e\u3059\u3002{tail}",
        f"No zoom up to 400% found the person's own well enough (recall {RECALL_MIN:g} and precision "
        f"{PRECISION_MIN:g} wanted). The best was {rung} (recall {nums[0]}, precision {nums[1]}), filed as not "
        f"passing: the first image is labelled at it as a sample for you to see, and you are asked before the "
        f"rest.{tail}")})


async def _take_back(w: Walk, entry: dict) -> bool:
    """A sample let go of: what was written on it put back to nothing, and its flag taken off; whether it could be.

    As a person's "take it back" is done (_confirm): an all-background mask
    over what the walk wrote. The image is then as it was before the run --
    no mask of the walk's, and not handed to a person for one.
    """
    iid = str(entry["id"])
    await _call(w, "kept_drop", {"item_id": iid, "boxes_json": "all"})
    if iid in w.written:
        width, height = int(entry.get("w") or 0), int(entry.get("h") or 0)
        got = (await _call(w, "mask_put", {"item_id": iid, "mask_png_base64": _blank_png(width, height)})
               if width and height else {"error": "its size is not known"})
        if got.get("error") or got.get("status") != "written":
            return False
        w.written.remove(iid)
    if iid in w.reviewed:
        await _call(w, "mark_review", {"item_ids_json": json.dumps([iid]), "review": False,
                                       "reason": w.say("\u8a66\u3057\u306e\u30e9\u30d9\u30eb\u3092\u53d6\u308a\u6d88\u3057\u307e\u3057\u305f", "the sample was taken back")})
        w.reviewed.discard(iid)
        w.ever_flagged.discard(iid)
        w.handed_over.pop(iid, None)
    w.results[:] = [(i, "taken back" if i == iid else what, st) for i, what, st in w.results]
    w.progress.moved(f"took back {iid}")
    return True


async def _unfile(w: Walk) -> bool:
    """The zoom this run filed short of passing, taken back with its sample: whether it was.

    Filed before the sample, as a walk at it stands on the filing (_prepare),
    it stayed after the person turned the sample down: the next run found it
    filed, rehearsed nothing, showed no sample and asked only whether to go
    ahead at it -- and a yes labelled every image at a zoom they had said no
    to on seeing what it does. Taken back, the next run rehearses again and
    shows a sample again.
    """
    if not (w.rehearsed or {}).get("filed") or (w.rehearsed or {}).get("passed"):
        return False
    got = await _call(w, "view_choose", {"rung": "none"})
    return not got.get("error")


async def _consent_sample(w: Walk, rung: str, entry: dict, rest: list[dict]) -> tuple[str, str]:
    """After the sample: go ahead at the rung or stop, with how long the rest takes. Always put to the person."""
    iid, chose = str(entry["id"]), w.rehearsed or {}
    what = next((what for i, what, _ in reversed(w.results) if i == iid), "nothing")
    what_said = w.say(_SAID_JA.get(what, what), what)
    parts, minutes = _estimate(w, rest) if rest else (0, 0)
    nums = f"{float(chose.get('recall') or 0):.2f}", f"{float(chose.get('precision') or 0):.2f}"
    choices = [w.say(f"{rung} \u3067\u9032\u3081\u308b", f"Proceed at {rung}"), w.say("\u3084\u3081\u308b", "Stop")]
    if not rest:
        left_ja, left_en = "\u307b\u304b\u306b\u672a\u30e9\u30d9\u30eb\u306e\u753b\u50cf\u306f\u3042\u308a\u307e\u305b\u3093\u3002", "No other image is left to label."
    elif rung == "fit":
        left_ja, left_en = (f"\u6b8b\u308a\u306f{len(rest)}\u679a\u3067\u3001\u30e2\u30c7\u30eb\u304c1\u679a\u305a\u3064\u898b\u3066\u3044\u304f\u3068\u3001\u304a\u3088\u305d{minutes}\u5206\u304b\u304b\u308b\u898b\u8fbc\u307f\u3067\u3059\u3002",
                            f"{len(rest)} more images are left, the model looking at each in turn: about {minutes} minutes.")
    else:
        left_ja, left_en = (f"\u6b8b\u308a\u306f{len(rest)}\u679a\uff08\u90e8\u5206\u306b\u5206\u3051\u308b\u3068{parts}\u679a\uff09\u3067\u3001\u304a\u3088\u305d{minutes}\u5206\u304b\u304b\u308b\u898b\u8fbc\u307f\u3067\u3059\u3002",
                            f"{len(rest)} more images are left: {parts} parts in all, about {minutes} minutes.")
    left_ja, left_en = left_ja + _no_more_asked(w)[0], left_en + " " + _no_more_asked(w)[1]
    text = w.say(
        f"\u4eba\u304c\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f\u753b\u50cf\u3067\u8a66\u3057\u305f\u500d\u7387\u306f\u3069\u308c\u3082\u57fa\u6e96\u306b\u5c4a\u304b\u305a\u3001\u6700\u3082\u826f\u304b\u3063\u305f\u306e\u306f {rung} \u3067\u3057\u305f\uff08\u518d\u73fe\u7387 {nums[0]}\u3001\u9069\u5408\u7387 {nums[1]}\uff09\u3002"
        f"{iid} \u3092\u3053\u306e\u500d\u7387\u3067\u8a66\u3057\u306b\u30e9\u30d9\u30eb\u4ed8\u3051\u3057\u307e\u3057\u305f\uff08{what_said}\uff09\u3002\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044\u3002{left_ja}"
        f"{rung} \u3067\u9032\u3081\u307e\u3059\u304b\u3001\u3084\u3081\u307e\u3059\u304b\u3002\u3084\u3081\u308b\u3068\u3001\u8a66\u3057\u306e\u30e9\u30d9\u30eb\u306f\u53d6\u308a\u6d88\u3057\u307e\u3059\u3002",
        f"No zoom passed the rehearsal on the person's own; the best was {rung} (recall {nums[0]}, precision "
        f"{nums[1]}). {iid} is labelled at it as a sample ({what_said}): look at it. {left_en} Proceed at {rung}, "
        "or stop? Stopping takes the sample back.") + _numbered(w, choices)
    return await _ask_consent(w, rung, text, choices, ("proceed", "stop"))


async def _sample(w: Walk, todo: list[dict], rung: str, target: str, class_id: int | None) -> None:
    """No zoom passed: the first image labelled at the best one for the person to see, then go ahead or stop.

    Written, reviewed and shown as any image of the walk is (one picture on
    the screen, followed there), then asked about -- always, as the question
    before walking is, with how long the rest takes (_consent_sample). Going
    ahead keeps it and walks the rest. Anything else -- stop, a reply that
    says neither twice, nobody answering, the Stop button, a failure -- takes
    the sample back (_take_back), and the zoom filed for it (_unfile), and
    ends the run. With nobody to ask and no --yes nothing is labelled at all:
    nobody would see the sample.
    """
    entry, rest = todo[0], todo[1:]
    iid = str(entry["id"])

    def unfiled_said(unfiled: bool) -> str:
        return w.say(f"{rung} \u306f\u8a18\u9332\u304b\u3089\u5916\u3057\u305f\u305f\u3081\u3001\u6b21\u306e\u5b9f\u884c\u3067\u8a66\u3057\u76f4\u3057\u307e\u3059\u3002",
                     f" {rung} is not kept on file: the next run rehearses again.") if unfiled else ""

    if w.ask is None and not w.assume_consent:
        unfiled = await _unfile(w)
        raise _Ended("nobody", w.say(
            f"{rung} \u3067\u306e\u8a66\u3057\u306e\u30e9\u30d9\u30eb\u3092\u898b\u3066\u3001\u9032\u3081\u3066\u3088\u3044\u304b\u3092\u7b54\u3048\u308b\u4eba\u304c\u3044\u306a\u3044\u305f\u3081\u3001\u3069\u306e\u753b\u50cf\u306b\u3082\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002"
            "\u30b3\u30de\u30f3\u30c9\u30e9\u30a4\u30f3\u304b\u3089\u5b9f\u884c\u3059\u308b\u5834\u5408\u306f\u3001--yes \u3092\u4ed8\u3051\u308b\u3068\u78ba\u8a8d\u306a\u3057\u3067\u9032\u3081\u307e\u3059\u3002",
            f"Nobody is here to see a sample at {rung} and say whether to go ahead, so nothing was labelled. "
            "From the command line, --yes goes ahead without asking.") + unfiled_said(unfiled))
    if not w.prepared:
        await _prepare(w)
        w.prepared = True
    await _examples(w, rung, class_id)
    try:
        await _label_image(w, entry, 1, len(todo), rung, target, class_id, confirm=False)
        said, reply = await _consent_sample(w, rung, entry, rest)
    except BaseException:
        await _take_back(w, entry)
        await _unfile(w)
        raise
    if said == "proceed":
        w.consented_ids |= {str(e["id"]) for e in todo}
        return
    back = await _take_back(w, entry)
    unfiled = await _unfile(w)
    gone_ja, gone_en = ((f"\u8a66\u3057\u306b\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f {iid} \u306f\u53d6\u308a\u6d88\u3057\u3001", f"the sample {iid} was taken back and ") if back else
                        (f"\u8a66\u3057\u306b\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u305f {iid} \u306f\u53d6\u308a\u6d88\u305b\u307e\u305b\u3093\u3067\u3057\u305f\uff08\u66f8\u304d\u8fbc\u3093\u3060\u3082\u306e\u304c\u6b8b\u3063\u3066\u3044\u307e\u3059\uff09\u3002",
                         f"the sample {iid} could not be taken back -- what was written on it is still there -- and "))
    if said == "stopped":
        raise _Ended("stopped")
    if said in ("nobody", "unanswered"):
        raise _Ended("nobody", w.say(
            f"\u8a66\u3057\u306e\u30e9\u30d9\u30eb\u306e\u3042\u3068\u3001{rung} \u3067\u9032\u3081\u3066\u3088\u3044\u304b\u3068\u3044\u3046\u8cea\u554f\u306b\u8ab0\u3082\u7b54\u3048\u306a\u304b\u3063\u305f\u305f\u3081\u3001{gone_ja}\u307b\u304b\u306e\u753b\u50cf\u306b\u306f\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
            f"Nobody answered whether to go ahead at {rung} after the sample, so {gone_en}nothing else was labelled.")
            + unfiled_said(unfiled))
    quoted = " ".join(reply.split())[:120]
    raise _Ended("declined", w.say(
        f"\u9032\u3081\u306a\u3044\u3068\u3044\u3046\u7b54\u3048\uff08\u300c{quoted}\u300d\uff09\u3060\u3063\u305f\u305f\u3081\u3001{gone_ja}\u307b\u304b\u306e\u753b\u50cf\u306b\u306f\u30e9\u30d9\u30eb\u3092\u4ed8\u3051\u3066\u3044\u307e\u305b\u3093\u3002",
        f"The answer (\"{quoted}\") was not to go ahead, so {gone_en}nothing else was labelled.")
        + unfiled_said(unfiled))


async def label_images(w: Walk, target: str, class_name: str = "", item_ids: Any = None) -> dict:
    """Label the unlabelled images of the project at the filed zoom, or at one chosen first: {"outcome", "summary", "final"}.

    With no zoom filed, one is chosen by rehearsing the walk on the person's
    own, lowest first, and filed (_decide_rung) -- before the question about
    going ahead, whose estimate is counted at it. With none passing, the
    first image is labelled at the best as a sample, and the question comes
    after it (_sample). On a project of specks, each image whole with
    spot_detect instead (_speck_image), written as the class spot_detect
    takes from the project: a class_name is not applied there, and that is
    said.

    Every one of them, or those of item_ids, when the person named images --
    and never more than the run is for (Walk.only). outcome "done" when the
    images were gone through; anything else ends the run -- "stopped" (the
    Stop button), "declined" (the person said no), "again" (choose the zoom
    again: the filing is taken back), "nobody" (no one to say whether to go
    ahead, or nobody did) or "failed". A run that went on after one of those
    would be told to carry on, call this again, and put the same question to
    the person again.
    """
    filed = dict(w.plan.get("filed") or {})
    rung = str(filed.get("rung") or "fit")
    target = _target_of(w, target)
    specks = _specks(w)
    try:
        plan, todo = await _scope(w, item_ids)
        if not todo:
            w.covered = True
            return _out(w, "done", rung, w.say("\u30e9\u30d9\u30eb\u306e\u4ed8\u3044\u3066\u3044\u306a\u3044\u753b\u50cf\u306f\u6b8b\u3063\u3066\u3044\u307e\u305b\u3093\u3002",
                                               "No unlabelled image is left for the walk."))
        # The class the rehearsal measured with, kept for the walk after it.
        class_id = _class_of(plan, class_name, filed)
        sampled = False
        if _rehearses(w, filed):
            await _decide_rung(w, target, class_name, class_id)
            filed = dict(w.plan.get("filed") or {})
            rung = str(filed.get("rung") or "fit")
            if not filed.get("default"):
                # Each image's parts at the zoom chosen, for the estimate.
                plan, todo = await _scope(w, item_ids)
                if not todo:
                    w.covered = True
                    return _out(w, "done", rung)
                if not w.rehearsed.get("passed"):
                    await _sample(w, todo, rung, target, class_id)
                    sampled = True
        await _agree(w, filed, rung, todo)
        named = " ".join(str(class_name or "").split())[:80]
        painted = (w.band or {}).get("class_id")
        if specks and named and (class_id is None or class_id != painted):
            # spot_write writes what spot_detect staged, as the class the person
            # paints with (or the project's first): nothing takes a class here,
            # and a class asked for and not written was dropped without a word.
            w.emit({"type": "auto", "text": w.say(
                f"\u70b9\u306f\u3001spot_detect \u304c\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u304b\u3089\u9078\u3076\u30af\u30e9\u30b9\uff08\u4eba\u304c\u5857\u3063\u3066\u3044\u308b\u30af\u30e9\u30b9\u304b\u3001\u6700\u521d\u306e\u30af\u30e9\u30b9\uff09\u3067\u66f8\u304d\u8fbc\u3080\u305f\u3081\u3001"
                f"\u6307\u5b9a\u3055\u308c\u305f\u30af\u30e9\u30b9\u300c{named}\u300d\u306f\u4f7f\u3044\u307e\u305b\u3093\u3002",
                f"The class asked for, \"{named}\", is not used: specks are written as the class spot_detect takes "
                "from the project -- the one the person paints with, or its first.")})
        if not specks:
            await _examples(w, rung, class_id)
        for n, entry in enumerate(todo, 1):
            if sampled and n == 1:
                continue                                 # the sample, kept: its question had the estimate
            if await w.gate():
                raise _Ended("stopped")
            t0 = time.time()
            if specks:
                await _speck_image(w, entry, n, len(todo), target)
            else:
                await _label_image(w, entry, n, len(todo), rung, target, class_id)
            if n == 1 and len(todo) > 1:
                _reestimate(w, time.time() - t0, entry, todo[1:])
        w.covered = True
    except _Ended as end:
        return _out(w, end.outcome, rung, end.said)
    except Exception as exc:                             # noqa: BLE001 - the run says what happened and ends
        _log.exception("the walk failed on %s", w.project_id)
        return _out(w, "failed", rung, w.say(f"\u30e9\u30d9\u30eb\u4ed8\u3051\u306e\u9014\u4e2d\u3067\u5931\u6557\u3057\u307e\u3057\u305f\uff1a{exc!r}"[:300],
                                             f"The walk failed: {exc!r}"[:300]))
    return _out(w, "done", rung)


# ----------------------------------------------------------------- the model driver


#: The calls of a model that walks the parts itself (driver "model") that are
#: answered here before anything reaches the bridge: next_view is the loop's
#: own, and each of the bridge's is checked against the walk first. A flag
#: taken off (mark_review with review false) is not among them: the loop's
#: own books deal with that as they always have.
MODEL_CALLS = ("next_view", "image_view", "accept_masks", "kept_drop", "write_kept", "mark_review")


@dataclass
class _Open:
    """The image the model is walking: the views it is handed, in order, and how far it has got."""

    entry: dict
    n: int                                          # which image of the walk, from 1
    eff: str                                        # the rung it is walked at; "fit": looked at whole
    large: list | None                              # too large for a part from here, in px of the Fit view
    where_of: dict                                  # view -> where it is, in words
    parts: int                                      # its parts at the rung
    seq: list                                       # the views of this round, in the order they are handed
    st: dict = field(default_factory=_stats)
    t0: float = field(default_factory=time.time)
    i: int = 0                                      # the next of seq to hand
    shown: str | None = None                        # the view in front of the model
    size: tuple = (0, 0)                            # its width and height
    answered: set = field(default_factory=set)      # views boxes were sent off, or moved on from
    phase: str = "walk"                             # walk, write, review, rewrite
    what: str = "nothing"                           # what it is so far: "written" once written
    put_down: str = ""
    gaps: list = field(default_factory=list)        # what of it was never looked at, said when it is done
    last: dict = field(default_factory=dict)        # its last write
    rewrites: int = 0
    changed: list = field(default_factory=list)     # views changed in this round of the review
    looked: set = field(default_factory=set)        # views handed in the review so far
    final: bool = False                             # no rewrite comes after this round
    left: list = field(default_factory=lambda: [0, 0])   # what the final round still found: missed, wrong
    handed_in: int = -1                             # the loop's turn the view in front was handed in

    @property
    def iid(self) -> str:
        return str(self.entry["id"])


def _reply(res: dict, picture: str | None = None, say: str = "") -> dict:
    return {"res": res, "picture": picture, "say": say, "end": None}


def _where(o: _Open, view: str) -> str:
    return "the whole picture" if view == "fit" else (o.where_of.get(view) or view)


def _big(o: _Open, view: str) -> bool:
    """Whether a view is the whole picture looked at for what is too large to be whole in a part."""
    return view == "fit" and o.eff != "fit" and bool(o.large)


def _next_said(w: Walk, o: _Open) -> str:
    """What the model is to do next in this image, the pictures named by where they are.

    The image is named only where write_kept needs it. A refusal says this
    after its reason, inside REFUSAL_CHARS, and an image named by a camera's
    file name -- some thirty characters, said twice -- pushed where next_view
    goes past the cut.
    """
    if o.phase in ("walk", "review"):
        ahead = _where(o, o.seq[o.i]) if o.i < len(o.seq) else ""
        if o.shown is not None and o.shown not in o.answered:
            return (f"in front of you is {_where(o, o.shown)} (view '{o.shown}'): send its boxes, or next_view"
                    + (f" gives {ahead}" if ahead else ""))
        if ahead:
            return f"next_view gives {ahead}"
        if o.phase == "walk":
            return (f"every picture is answered: write_kept(item_id='{o.iid}') writes it" if o.st["kept"]
                    else "next_view moves on")
        if o.changed and o.st["kept"] and not o.final:
            return f"write_kept(item_id='{o.iid}') writes what you changed"
        return "next_view finishes the check"
    if o.phase == "write":
        return f"every picture is answered: write_kept(item_id='{o.iid}') writes it"
    return f"write_kept(item_id='{o.iid}') writes what you changed"


def _refused(w: Walk, o: _Open, why: str) -> tuple[dict, None, str]:
    """A call the walk does not take, and what to do instead, inside REFUSAL_CHARS.

    What to do goes whole, and the reason before it is cut short when the
    two do not fit: cut at the end, a refusal lost the part the model needs.
    """
    nxt = _next_said(w, o)
    room = REFUSAL_CHARS - len(nxt) - 2
    if len(why) > room:
        why = why[:max(0, room - 1)].rstrip() + "\u2026"
    return {"error": f"{why}: {nxt}"}, None, ""


def _unseen(w: Walk, o: _Open) -> bool:
    """Whether the picture in front was handed in the turn this call is from.

    It reaches the model after that turn's answers, in a message of its own.
    A turn of several calls -- next_view, next_view, ... -- was moved past
    every picture but the last, each counted as answered, and the write let
    through: what lay in the parts nobody saw was written as background.
    """
    return o.shown is not None and o.handed_in == w.turn


def _not_seen_yet(w: Walk, o: _Open) -> tuple[dict, None, str]:
    return {"error": (f"{_where(o, o.shown)} (view '{o.shown}') was handed in this same turn and reaches you "
                      "after it: look at it, then send its boxes or next_view")}, None, ""


def _through(w: Walk, o: _Open) -> bool:
    """Every picture of this round handed, and each answered -- or the last of them in front, and seen.

    The write is then the model moving on from the last picture, as
    next_view would be. Asked for next_view first, a model that had sent the
    boxes of the last part and called write_kept, as the brief says, was
    refused with "0 not answered".
    """
    if o.i < len(o.seq):
        return False
    left = [v for v in o.seq if v not in o.answered]
    return not left or (left == [o.shown] and not _unseen(w, o))


def _class_named(w: Walk, raw: Any) -> str:
    """The class the model named for a write, taken for the rest of the walk if the project has it; "" or why not.

    Taken for the walk, not for one write: the rewrite after a review is
    asked for as write_kept(item_id) alone, and went out as the class filed,
    not the one the model had written the image with. One the project does
    not have is not taken: written with it, a mask is paint the browser
    cannot colour (write_kept says so).
    """
    if raw is None or isinstance(raw, bool) or str(raw).strip() == "":
        return ""
    try:
        cid = int(str(raw).strip())
    except ValueError:
        cid = None
    if cid is not None and cid < 0:
        return ""                                        # the bridge's own default
    known = w.known_classes or frozenset()
    if cid is not None and (cid in known or (not known and cid > 0)):
        w.class_id = cid
        return ""
    now = str(w.class_id) if w.class_id is not None else "the teachers' class"
    return f"class_id {str(raw)[:12]} is not a class of this project: written as {now}"


def _not_this(w: Walk, o: _Open, args: dict) -> tuple | None:
    named = _ID_EXT.sub("", str(args.get("item_id") or "").strip())
    if named and named != o.iid:
        return _refused(w, o, f"{named[:40]} is not the image being walked")
    return None


def _handed(w: Walk, o: _Open, view: str, pic: dict) -> tuple[dict, str, str]:
    """A view handed to the model: the answer to its call, the picture, and what is said with the picture.

    The answer stays short: it is in the run's conversation for every turn
    after it, and there are dozens of parts to an image. Where the picture
    is, and what to send off it, go with the picture -- in words, with the
    view to send the boxes with.
    """
    iid, k = o.iid, o.i
    size = _one_said(pic, "One object")
    least = _large_on(o.large, pic) if _big(o, view) else None
    if o.phase == "review":
        res = {"item_id": iid, "view": view, "check": f"{k} of {len(o.seq)}"}
        say = (f"{iid}, check {k} of {len(o.seq)}: {_where(o, view)}" + ("" if view == "fit" else f", at {o.eff}")
               + ", with the mask just written on it: what is not mask is dimmed, and its edge is drawn in white."
               + (f" Check only those at least {least[0]} wide or {least[1]} tall here." if least else size))
        say += (" This is the last look at it: nothing more is written. What you still send with accept_masks or "
                "kept_drop is noted for a person with the image; then next_view." if o.final else
                f" An object with no mask: accept_masks(item_id='{iid}', view='{view}') with a box round it. A mask "
                f"on something that is not one: kept_drop(item_id='{iid}', view='{view}') with a box round that "
                "mask. Then, or if it is right, next_view.")
    elif least:
        res = {"item_id": iid, "view": view, "part": "the whole picture"}
        say = (f"{iid}, the whole picture, after its {o.parts} parts at {o.eff}. Box only those at least "
               f"{least[0]} wide or {least[1]} tall here -- smaller ones were looked for part by part -- "
               f"and send them with accept_masks(item_id='{iid}', view='fit'). None that large: next_view.")
    elif view == "fit":
        res = {"item_id": iid, "view": view, "part": "the whole picture"}
        say = (f"{iid}, the whole picture: at {o.eff} it is looked at whole.{size} Box each one you see, in "
               f"0-1000 of it, and send them with accept_masks(item_id='{iid}', view='fit'). None here: next_view.")
    else:
        k = sum(1 for v in o.seq[:o.i] if v != "fit")
        res = {"item_id": iid, "view": view, "part": f"{k} of {o.parts}"}
        # Said in full with an image's first part, and short after it: the
        # words of every part stay in the conversation, and the brief says it.
        say = (f"{iid}, part {k} of {o.parts} at {o.eff}: {_where(o, view)}.{size} Boxes of each one WHOLE in it: "
               f"accept_masks(item_id='{iid}', view='{view}'); none: next_view.")
        if k == 1:
            say += (" Give them in 0-1000 of this picture. One cut by the edge of a part is taken whole in the "
                    "neighbouring part: leave it out.")
    res["image"] = "attached in the next message"
    return res, pic["image_base64"], say


async def _hand(w: Walk, o: _Open) -> tuple[dict, str, str] | None:
    """The next view of this round, handed to the model; None when the round is over.

    A view that cannot be opened, asked twice, is a view nobody looked at:
    counted, said with the image when it is done, and the next one handed.
    """
    review = o.phase == "review"
    o.shown = None
    while o.i < len(o.seq):
        view = o.seq[o.i]
        o.i += 1
        pic = await _open(w, o.iid, view, **({"overlay": True} if review else {}))
        if pic is None:
            o.answered.add(view)
            if review:
                o.st["unreviewed"] += 1
            elif _big(o, view):
                o.st["overview_unseen"] = True
            else:
                o.st["unseen"] += 1
            continue
        o.shown, o.size = view, (int(pic.get("width") or 0), int(pic.get("height") or 0))
        o.handed_in = w.turn
        if review:
            o.st["reviewed"] += 1
        elif not _big(o, view):
            o.st["parts"] += 1
        if view == "fit" and not review:
            w.images_seen[o.iid] = pic["image_base64"]
            w.seen_crop[o.iid] = None
        # A picture not handed before is the walk going on. The same one
        # again, or a refused call, is not.
        w.progress.moved(f"{o.iid} {view}")
        return _handed(w, o, view, pic)
    return None


async def _onward(w: Walk, o: _Open, rung: str) -> tuple[dict, str | None, str]:
    handed = await _hand(w, o)
    if handed is not None:
        return handed
    return await _round_over(w, o, rung)


async def _round_over(w: Walk, o: _Open, rung: str) -> tuple[dict, str | None, str]:
    """Every view of a round handed and answered: the write, a rewrite, or the image done."""
    st = o.st
    if o.phase == "walk":
        o.gaps = _gaps(w, st, o.parts, o.eff)
        if not st["kept"]:
            await _flag(w, o.iid, _nothing_kept(w, st, o.eff, o.put_down, o.gaps))
            return await _finish(w, o, "flagged", rung)
        o.phase = "write"
        return {"item_id": o.iid, "kept_so_far": st["kept"],
                "next": f"every picture of {o.iid} is answered: write_kept(item_id='{o.iid}') writes it"}, None, ""
    if o.final:
        st["review_left"] = list(o.left) if (o.left[0] or o.left[1]) else None
    elif o.changed and st["kept"]:
        o.phase = "rewrite"
        return {"item_id": o.iid, "changed": len(set(o.changed)),
                "next": (f"write_kept(item_id='{o.iid}') writes what you changed "
                         f"(rewrite {o.rewrites + 1} of {w.review_redos})")}, None, ""
    return await _finish(w, o, await _judge(w, o.iid, st, o.last, o.put_down, o.gaps), rung)


async def _review_round(w: Walk, o: _Open, seq: list, rung: str, head: dict) -> tuple[dict, str | None, str]:
    """After a write: the views that hold mask, handed in order with the mask on them."""
    o.phase, o.seq, o.i, o.shown = "review", list(seq), 0, None
    o.answered, o.changed, o.left = set(), [], [0, 0]
    o.final = o.rewrites >= w.review_redos
    o.looked.update(seq)
    res, pic, say = await _onward(w, o, rung)
    return {**head, **res}, pic, say


async def _finish(w: Walk, o: _Open, what: str, rung: str) -> tuple[dict, str | None, str]:
    """The image done with: asked about if the person wanted that, let go of, and the next one's first view handed."""
    interrupted = False
    try:
        if what == "written" and w.confirm_each:
            what = await _confirm(w, o.iid, o.entry, o.st, what)
    except BaseException:
        interrupted = True
        raise
    finally:
        w.open = None
        await _let_go(w, o.iid, o.n, w.walk_total, what, o.st, interrupted)
    if o.n == 1 and w.walk_total > 1:
        _reestimate(w, time.time() - o.t0, o.entry, w.queue)
    done = {"finished": o.iid, "as": w.say(_SAID_JA.get(what, what), what)}
    nxt = await _take_next(w, rung)
    if nxt is None:
        return {**done, **_all_done(w, rung)}, None, ""
    res, pic, say = await _onward(w, nxt, rung)
    return {**done, **res}, pic, say


async def _take_next(w: Walk, rung: str) -> _Open | None:
    """The next image of the walk, planned and not yet shown; None when none is left."""
    while w.queue:
        entry = w.queue.pop(0)
        iid = str(entry["id"])
        if iid in w.done_with:
            continue
        n = w.walk_total - len(w.queue)
        plan = await _plan_image(w, iid, rung)
        if plan.get("flagged"):
            try:
                _unplanned(w, plan["flagged"])
            except _Ended:
                await _let_go(w, iid, n, w.walk_total, "flagged", _stats(), True)
                raise
            await _let_go(w, iid, n, w.walk_total, "flagged", _stats(), False)
            continue
        large = plan["large"]
        # Every part in the bridge's order, then the whole picture for what
        # is too large to be whole in any part -- as the harness looks, and
        # for its reason: a box round a stack kept first from the whole
        # picture had every part's box inside it skipped as kept already.
        seq = [v for v in plan["views"] if not (large and v == "fit")] + (["fit"] if large else [])
        w.open = _Open(entry=entry, n=n, eff=plan["eff"], large=large, where_of=plan["where_of"],
                       parts=len(plan["views"]), seq=seq)
        return w.open
    w.covered = True
    return None


def _all_done(w: Walk, rung: str) -> dict:
    w.covered = True
    return {"done": True, "summary": _summary(w, rung, "done"),
            "next": "nothing is left to walk: report, the images for a person first"}


async def _flags(w: Walk, args: dict, but: str = "") -> dict:
    """mark_review from the model while it walks: each image flagged as the walk flags one (_flag)."""
    raw = args.get("item_ids_json") or args.get("item_ids") or args.get("item_id")
    ids = [_ID_EXT.sub("", i) for i in _names(raw)]
    why = " ".join(str(args.get("reason") or "").split())[:200] or w.say(
        "\u30e2\u30c7\u30eb\u304c\u898b\u3066\u56de\u308b\u9014\u4e2d\u3067\u8981\u78ba\u8a8d\u306b\u3057\u307e\u3057\u305f", "flagged by the model while it walked")
    flagged, not_flagged = [], []
    for iid in ids:
        if iid == but:
            continue
        (flagged if await _flag(w, iid, why) else not_flagged).append(iid)
    out: dict = {"flagged": flagged}
    if not_flagged:
        out["not_flagged"] = not_flagged
    return out


async def _mw_next(w: Walk, o: _Open, args: dict, rung: str):
    if o.phase in ("write", "rewrite"):
        return _refused(w, o, "not yet")
    if o.shown is not None:
        o.answered.add(o.shown)                          # nothing here
    return await _onward(w, o, rung)


async def _mw_view(w: Walk, o: _Open, args: dict, rung: str):
    want = _ID_EXT.sub("", str(args.get("filename") or args.get("item_id") or "").strip())
    view = str(args.get("view") or "fit").strip()
    if want == o.iid and o.phase in ("walk", "review"):
        if view == o.shown:
            pic = await _open(w, o.iid, view, **({"overlay": True} if o.phase == "review" else {}))
            if pic is not None:
                return _handed(w, o, view, pic)          # the same picture again
            return _refused(w, o, f"{view} could not be opened again")
        if o.i < len(o.seq) and view == o.seq[o.i]:
            return await _mw_next(w, o, args, rung)
    return _refused(w, o, "not in order")


async def _mw_accept(w: Walk, o: _Open, args: dict, rung: str):
    no = _not_this(w, o, args)
    if no:
        return no
    if o.phase not in ("walk", "review") or o.shown is None:
        return _refused(w, o, "no picture to send boxes off")
    view = str(args.get("view") or o.shown).strip()
    if view != o.shown:
        return _refused(w, o, f"boxes are read off the picture in front of you, not '{view[:20]}'")
    boxes = _boxes_in(args.get("boxes_json"))
    res: dict = {"item_id": o.iid, "view": view}
    if _big(o, view):
        large = _large_only(boxes, _large_on(o.large, {"width": o.size[0], "height": o.size[1]}) or [NORM, NORM])
        if len(large) < len(boxes):
            res["left_for_the_parts"] = len(boxes) - len(large)
        boxes = large
    o.answered.add(view)
    if o.phase == "review" and o.final:
        o.left[0] += len(boxes)
        return {**res, "noted": len(boxes), "next": "the last look: noted for a person, not written; next_view"}, \
            None, ""
    st = o.st
    before = (st["refused"], st["skipped"], st["lost"])
    level = str(args.get("level") or "")
    took = (await _accept(w, o.iid, view, boxes, st, **({"level": level} if level in ("subpart", "part", "whole")
                                                          else {}))) if boxes else 0
    if o.phase == "walk" and _big(o, view):
        st["large"] += len(boxes)
    if took:
        w.progress.moved(f"kept {took} on {o.iid}")
        if o.phase == "review":
            o.changed.append(view)
    res.update(accepted=took, refused=st["refused"] - before[0], skipped=st["skipped"] - before[1],
               kept_so_far=st["kept"])
    if st["lost"] > before[2]:
        res["not_sent"] = st["lost"] - before[2]
    if o.phase == "walk" and _refused_too_many(st):
        o.put_down = _put_down_said(w, st, o.eff)
        return await _put_down(w, o, res, rung)
    res["next"] = "more in this picture: accept_masks again; done with it: next_view"
    return res, None, ""


async def _put_down(w: Walk, o: _Open, res: dict, rung: str):
    """Too many boxes refused on this image: what was kept is written, the image flagged, and the walk goes on."""
    st = o.st
    o.gaps = _gaps(w, st, o.parts, o.eff)
    if not st["kept"]:
        await _flag(w, o.iid, _nothing_kept(w, st, o.eff, o.put_down, o.gaps))
        what = "flagged"
    else:
        wrote = await _write(w, o.iid, w.class_id)
        what = await _first_write(w, o.iid, wrote, st)
        if what == "written":
            o.what = what
            what = await _judge(w, o.iid, st, wrote, o.put_down, o.gaps)
    out, pic, say = await _finish(w, o, what, rung)
    # What was said of the boxes just sent is of this image; what follows is
    # the next image's.
    return {"accepted": res.get("accepted", 0), "refused": res.get("refused", 0), "put_down": o.put_down,
            **out}, pic, say


async def _mw_drop(w: Walk, o: _Open, args: dict, rung: str):
    no = _not_this(w, o, args)
    if no:
        return no
    if o.phase not in ("walk", "review") or o.shown is None:
        return _refused(w, o, "no picture to drop masks in")
    view = str(args.get("view") or o.shown).strip()
    if view != o.shown:
        return _refused(w, o, f"masks are dropped in the picture in front of you, not '{view[:20]}'")
    spec = str(args.get("boxes_json") or "").strip()
    if spec.lower() == "all":
        return _refused(w, o, "an image's masks are let go of when it is done, not before")
    boxes = _boxes_in(spec) if spec else None
    o.answered.add(view)
    if o.phase == "review" and o.final:
        o.left[1] += len(boxes) if boxes is not None else 1
        return {"item_id": o.iid, "view": view, "noted": len(boxes) if boxes is not None else 1,
                "next": "the last look: noted for a person, not written; next_view"}, None, ""
    payload: dict = {"item_id": o.iid, "view": view, "boxes_json": json.dumps(boxes) if boxes is not None else "",
                     "units": UNITS}
    if _big(o, view):
        # On the whole picture a box round a large mask holds the middles of
        # small ones too: only the large go.
        least = _large_on(o.large, {"width": o.size[0], "height": o.size[1]}) or [NORM, NORM]
        payload.update(min_w=least[0], min_h=least[1])
    got = await _call(w, "kept_drop", payload)
    if got.get("error"):
        return {"error": got["error"]}, None, ""
    dropped = int(got.get("dropped") or 0)
    if dropped:
        o.st["kept"] = int(got.get("kept_so_far") or 0)
        w.progress.moved(f"dropped {dropped} on {o.iid}")
        if o.phase == "review":
            o.changed.append(view)
    return {"item_id": o.iid, "view": view, "dropped": dropped, "kept_so_far": o.st["kept"],
            "next": "right now: next_view"}, None, ""


async def _mw_write(w: Walk, o: _Open, args: dict, rung: str):
    no = _not_this(w, o, args)
    if no:
        return no
    if o.phase in ("walk", "review"):
        if not _through(w, o):
            if o.phase == "review":
                return _refused(w, o, "the check is not finished")
            left = sum(1 for v in o.seq if v not in o.answered)
            return _refused(w, o, f"write_kept waits for every picture ({left} not answered)")
        if o.shown is not None:
            o.answered.add(o.shown)
        was = o.phase
        res, pic, say = await _round_over(w, o, rung)
        if o.phase not in ("write", "rewrite"):
            # Nothing kept to write, or nothing changed since the write: the
            # image is done with, and the next picture handed.
            why = "nothing was kept on it" if was == "walk" else "nothing to write again"
            return {"written": False, "why": why, **res}, pic, say
    said = _class_named(w, args.get("class_id"))
    noted = {"class_id": said} if said else {}
    wrote = await _write(w, o.iid, w.class_id)
    st = o.st
    if o.phase == "write":
        what = await _first_write(w, o.iid, wrote, st)
        if what != "written":
            out, pic, say = await _finish(w, o, what, rung)
            return {"written": False, **noted, **out}, pic, say
        o.what, o.last = what, wrote
        head = {"written": True, "item_id": o.iid, "objects": st["objects"], **noted}
        return await _review_round(w, o, _review_views(wrote, o.eff, o.large, st), rung, head)
    o.rewrites += 1
    if not wrote.get("written"):
        out, pic, say = await _finish(w, o, await _judge(w, o.iid, st, o.last, o.put_down, o.gaps), rung)
        return {"written": False, **noted, **out}, pic, say
    o.last = wrote
    st["rewritten"] += 1
    st["objects"] = int(wrote.get("objects") or st["kept"])
    w.progress.moved(f"wrote {o.iid}")
    again = [v for v in (wrote.get("views_with_mask") or []) if v not in o.looked]
    head = {"written": True, "item_id": o.iid, "objects": st["objects"], "rewrite": f"{o.rewrites} of {w.review_redos}",
            **noted}
    return await _review_round(w, o, list(dict.fromkeys(o.changed + again)), rung, head)


async def _mw_flag(w: Walk, o: _Open, args: dict, rung: str):
    """mark_review while walking. The image being walked, flagged, is put down: let go of, and the next one handed."""
    raw = args.get("item_ids_json") or args.get("item_ids") or args.get("item_id")
    if o.iid not in {_ID_EXT.sub("", i) for i in _names(raw)}:
        return await _flags(w, args), None, ""
    res = await _flags(w, args, but=o.iid)
    why = " ".join(str(args.get("reason") or "").split())[:200] or w.say(
        "\u30e2\u30c7\u30eb\u304c\u898b\u3066\u56de\u308b\u9014\u4e2d\u3067\u8981\u78ba\u8a8d\u306b\u3057\u307e\u3057\u305f", "flagged by the model while it walked")
    if await _flag(w, o.iid, why):
        res["flagged"] = res["flagged"] + [o.iid]
        what = "flagged"
    else:
        res.setdefault("not_flagged", []).append(o.iid)
        what = o.what
    out, pic, say = await _finish(w, o, what, rung)
    return {**res, **out}, pic, say


_MODEL_HANDLERS = {"next_view": _mw_next, "image_view": _mw_view, "accept_masks": _mw_accept,
                   "kept_drop": _mw_drop, "write_kept": _mw_write, "mark_review": _mw_flag}
#: The calls that answer the picture in front of the model, or move it on:
#: none is taken before that picture has reached it (_unseen).
_AFTER_LOOKING = ("next_view", "image_view", "accept_masks", "kept_drop", "write_kept")


def _let_go_of_already(w: Walk, args: dict) -> dict | None:
    """kept_drop "all" for an image the walk is done with: answered as done, for it is.

    The bridge's own words for kept_drop ask for it when an image is
    finished, and the walk lets go of each image itself as it finishes it.
    Refused, a model that did as those words say put a failed line on the
    screen for every image.
    """
    if str(args.get("boxes_json") or "").strip().lower() != "all":
        return None
    named = _ID_EXT.sub("", str(args.get("item_id") or "").strip())
    if not named or (w.open is not None and w.open.iid == named) or named not in {r[0] for r in w.results}:
        return None
    return {"item_id": named, "dropped": 0, "let_go": "already, when it was done with"}


async def model_call(w: Walk, fn: str, args: dict, turn: int) -> dict:
    """One call of a model that walks the parts itself (driver "model"): {"res", "picture", "say", "end"}.

    The model calls the bridge's tools, and this is all the program does
    about it: it hands the pictures over in order -- every part at the filed
    zoom in the bridge's order, the empty ones too, then the whole picture
    for what is too large for a part; with no zoom filed, each picture once,
    whole, at Fit -- and refuses any other, naming the next one in words; it takes
    nothing about a picture before it has reached the model, which is after
    the turn that asked for it (turn, the loop's count of the model's turns);
    and it holds write_kept back until every picture of the image has been
    shown and answered (boxes sent off it, or moved on from), so no image is
    left half walked. After the write, the parts that hold mask are handed
    over the same way with the mask drawn on them; what the model changes
    there is written again, at most review_redos times, and a last look
    after that is noted for a person rather than written. The rest is the
    harness's own code: the person's go-ahead before the first picture, the
    preparation, the flags, one picture per image on the screen.

    The walk starts with next_view, which names the images when the person
    did. Started by another call, it was a walk of the whole project, and
    the image that call named was refused as out of order.

    res answers the call; picture, when there is one, goes before the model
    in a message of its own with say beside it; end, when the walk ended the
    run, is what label_images returns for that.
    """
    filed = dict(w.plan.get("filed") or {})
    rung = str(filed.get("rung") or "fit")
    w.turn = turn
    try:
        if fn == "mark_review" and w.open is None:
            return _reply(await _flags(w, args))
        if fn == "kept_drop":
            done = _let_go_of_already(w, args)
            if done is not None:
                return _reply(done)
        if w.covered and fn == "next_view" and _names(args.get("item_ids")):
            # More images named once the walk is through: walked in turn,
            # and asked about first -- a yes to one image is not a yes to more.
            w.begun, w.covered = False, False
        if not w.begun:
            if fn != "next_view":
                return _reply({"error": "the walk starts with next_view: item_ids=[...] if the person named the "
                                        "images, none for every unlabelled one"})
            plan, todo = await _scope(w, args.get("item_ids"))
            if todo:
                await _agree(w, filed, rung, todo)
                w.model_turns += _model_turns(todo)
            if w.known_classes is None:
                w.class_id = _class_of(plan, "", filed)
                w.known_classes = frozenset(int(c["id"]) for c in plan.get("classes") or []
                                            if isinstance(c, dict) and isinstance(c.get("id"), int))
            w.queue, w.walk_total, w.begun = list(todo), len(todo), True
        o = w.open or await _take_next(w, rung)
        if o is None:
            return _reply(_all_done(w, rung))
        if fn in _AFTER_LOOKING and _unseen(w, o):
            return _reply(*_not_seen_yet(w, o))
        return _reply(*await _MODEL_HANDLERS[fn](w, o, args, rung))
    except _Ended as end:
        end_out = _out(w, end.outcome, rung, end.said)
    except Exception as exc:                             # noqa: BLE001 - the run says what happened and ends
        _log.exception("the walk failed on %s", w.project_id)
        end_out = _out(w, "failed", rung, w.say(f"\u30e9\u30d9\u30eb\u4ed8\u3051\u306e\u9014\u4e2d\u3067\u5931\u6557\u3057\u307e\u3057\u305f\uff1a{exc!r}"[:300],
                                                 f"The walk failed: {exc!r}"[:300]))
    o, w.open = w.open, None
    if o is not None:
        # Stopped or failed part way through: nothing of it is written, and
        # what was kept for it is let go of.
        await _let_go(w, o.iid, o.n, w.walk_total, o.what, o.st, True)
    return {"res": {"walk": end_out["outcome"], "summary": end_out["summary"]}, "picture": None, "say": "",
            "end": end_out}


def model_left(w: Walk) -> str:
    """What a model that answered with no call while it walks is told; "" when nothing of the walk is left to it.

    An image is not left half walked: what lies in the parts nobody looked
    at would be written as background, or never written at all.
    """
    if w.covered or not w.begun:
        return ""
    o = w.open
    if o is None:
        return w.say("\u307e\u3060\u898b\u3066\u3044\u306a\u3044\u753b\u50cf\u304c\u3042\u308a\u307e\u3059\u3002next_view \u3067\u6b21\u306e\u7d75\u3092\u53d7\u3051\u53d6\u3063\u3066\u304f\u3060\u3055\u3044\u3002",
                     "Images are left to walk: next_view hands you the next picture.")
    return w.say(f"{o.iid} \u306f\u307e\u3060\u9014\u4e2d\u3067\u3059\u3002\u753b\u50cf\u3092\u9014\u4e2d\u3067\u7d42\u3048\u308b\u3053\u3068\u306f\u3067\u304d\u307e\u305b\u3093: {_next_said(w, o)}\u3002",
                 f"{o.iid} is part way through, and an image is not left half walked: {_next_said(w, o)}.")
