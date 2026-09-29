#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A local vision model labels a project through the MCP bridge, on its own.

    python scripts/examples/qwen_mcp_agent.py PROJECT_ID "part" img013 img014
    python scripts/examples/qwen_mcp_agent.py PROJECT_ID "part" --unlabelled 10
        [--model qwen3.8:27b] [--ollama http://127.0.0.1:11434] [--api http://127.0.0.1:8002]

The model gets the playbook as its system prompt and a subset of the
bridge's tools as functions. It decides what to look at and where the objects are;
teacher_band, accept_mask and write_kept do the arithmetic. Needs an Ollama
model with vision and tool calling (qwen3.8:27b, gemma4:26b) and the
fastmcp package. Nothing here is specific to Qwen but the default name.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from .backends import ANSWER_TOKENS, Backend, make_backend

# Imported when a run starts, not at import time: fastmcp is an optional
# dependency and the API has to come up without it. The names stay
# module-level so a test can put its own in their place.
Client = None
StdioTransport = None
#: The fastmcp release the bridge and this client were checked with: the one
#: THIRD_PARTY_NOTICES.md records.
FASTMCP_VERSION = "4.0.1"


def _install_line() -> str:
    """The command that installs the MCP client into the Python this runs on.

    Not a bare pip: in a new shell that is usually another interpreter's, and
    the package lands where this one never looks. A path with no space is left
    bare, which cmd, PowerShell and a POSIX shell all run; one with a space is
    quoted, and PowerShell runs a quoted path only after &, so that form is
    given beside it.
    """
    exe = sys.executable or "python"
    line = f"-m pip install fastmcp=={FASTMCP_VERSION}"
    if " " not in exe:
        return f"{exe} {line}"
    return f'"{exe}" {line}  (PowerShell: & "{exe}" {line})'


def _load_fastmcp() -> None:
    """Bind the client, or say what to install."""
    global Client, StdioTransport
    if Client is not None and StdioTransport is not None:
        return
    try:
        from fastmcp import Client as _Client
        from fastmcp.client.transports import StdioTransport as _Stdio
    except ImportError as exc:
        raise RuntimeError(
            f"labelling with a vision model needs the MCP client: {_install_line()}"
        ) from exc
    Client, StdioTransport = _Client, _Stdio


#: The variables the bridge is started with besides the token: the MCP SDK's
#: own list of those safe to inherit, on either platform.
_BRIDGE_ENV_VARS = (
    "APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
    "PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE",
    "HOME", "LOGNAME", "SHELL", "TERM", "USER",
)


def _bridge_env() -> dict[str, str]:
    """The bridge's whole environment, spelled out.

    Not this process's environment: the MCP client starts a server with a short
    list of variables it treats as safe, and the API's shared secret is added
    to it by name -- a bridge that reaches an API bound to one LAN address, off
    loopback, is answered 401 without it. In the environment rather than on
    the command line, where it would be in the process list.

    Handed the token alone, what else the bridge got was up to the installed
    MCP SDK: current releases lay it over their own list, older ones used it
    in that list's place, and the bridge would start on Windows with no PATH
    or SYSTEMROOT. So the list goes too, and the result is the same on both.
    """
    env: dict[str, str] = {}
    for key in _BRIDGE_ENV_VARS:
        value = os.environ.get(key)
        if value is not None and not value.startswith("()"):
            env[key] = value
    token = os.environ.get("SEG_API_TOKEN", "").strip()
    if token:
        env["SEG_API_TOKEN"] = token
    return env


#: What the tools do not say themselves, in as few words as the measurements
#: allow. The three recipe tools carry the arithmetic and explain it in their
#: own descriptions, so the whole playbook in the system prompt took over half
#: of a 16k window, and seconds before a turn even started, to repeat what the
#: tools already said. Pass brief=False to send the pages instead.
BRIEF = """You label images in a segmentation trainer through tools.

Say what you see, before you box it. One line in the operator's language, every
time you look at a picture: what the object is and how many of them are there --
"a red block, three of them, one lying on its side at the edge of the picture". It is the only
record of what you thought you were labelling. When a mask comes out wrong, that
line is what separates "it saw a leaf and called it the flower" from "it saw the
flower and the segmenter answered with the leaf behind it", and those have
different repairs. A run that calls tools and says nothing leaves no way to tell
them apart.

Rehearse before you start. Take an image the person already drew -- teacher_band
names them -- and work it exactly as you would an unlabelled one: look at it,
box or point at what you see and accept_mask as usual, or spot_detect on one
speck of a scattering. Then call rehearse instead of write_kept or spot_write.
It scores what you have against their own mask: how many of their objects you
found, how many of yours sit where they drew nothing, and the overlap.
Read it. If it is good, the rest of the job will look like it; if half their
objects are missed, or one of their objects came back as dozens of pieces of yours,
say so and ask -- that is the moment to ask, not after the rest of the images. Their mask
is not overwritten by this, and a rehearsal costs one image.

Order: teacher_band once for the project. Then one image at a time:
image_get_b64 to look at it, accept_masks with a box for EVERY object you can
see, then write_kept. Sending them one at a time costs a turn of thinking each
for arithmetic that needed none; read which came back refused and follow up on
those. Send about fifteen boxes per call and call again for the rest: a longer
list runs your own answer out of room and arrives as nothing at all.

Two ways of labelling, and which one fits is yours to say from what you see.
Objects you can box or point at, one mask each, go through accept_* and
write_kept, as above and below. A scattering of small alike marks -- specks,
pinholes, dust -- is not boxed one by one: spot_detect with like_item_id set to
an image the person drew takes their specks as the example and finds the rest,
and spot_write writes what it found. A point inside ONE speck works too, but a
speck is a few pixels in the copy you see, and the one that stands out is
rarely like the rest. Say which way at the second step, and use that way for
the rehearsal as for the rest.

After accept_mask, accept_masks and accept_points you are shown a picture of
SAM's answers for the box or the points, one panel per answer, smallest first:
inside the white line is the answer (the rest dimmed), the orange rectangle is
your box and orange dots are your points. KEPT marks what was
kept; 'fits' is an answer the teacher band passes, 'no: <reason>' one it refuses
whatever it is called, 'overlaps kept' one that fits but covers a mask already
kept. A panel of the whole picture is an answer that spread far past the box --
the object with what it lies in, or the background. To take another panel for a
refused box, send the same box with level set to that panel's name. What a box
KEPT stays until reset=true, which forgets all kept on that image: then send all
the image's boxes again with accept_masks, one call per level, the first with
reset=true; likewise for an image already written or finished. Never read
coordinates off this picture.

One object, one mask. Where an object is made of parts that look different --
a bright cap and a dark body, a metal head and a painted handle -- give
accept_points a GROUP for it: [[[cap_x, cap_y], [body_x, body_y]], ...].
A single point returns whichever part it sat on; that part is the size of
something the teachers drew, so the band accepts it and the object ends up in
pieces -- a handful of objects can come back as several times as many blobs
that way, the dark parts bare.
write_kept says "in_pieces" when it sees that, and the answer is to reset the
image and point again with groups. The same goes for a thin object lying across
its box, a wrench on the diagonal: the box's middle is on what it lies on, and
every answer is the floor. Point along it instead, one group, end to end.

Where you can see where an object's edge runs, you can trace it instead of
boxing it: accept_mask with outline_json=[[x, y], ...], a dozen or more points
in order around it. SAM is asked from the point deepest inside your outline,
with the box around it; the picture draws your outline as an orange line.

Measuring is once per project and it is remembered: calibrate_sam answers
"measured_before" on a project that has had it, and then you go and label. A
run that ends having proposed no mask at all has done nothing for the person
who asked -- if that is where you are, say which images you looked at and what
refused them, and mark_review one of them with the reason, rather than
reporting a measurement as if it were the job.

Once per project, after teacher_band, call teacher_view and LOOK at it: the
person's own image with their objects outlined, and single objects cut out
below. Everything else you are told about them is numbers, and numbers do not
say where one object ends. Then, for objects you box or point at, call
calibrate_sam: it tries each
segmenter and each way of pointing against objects the person already drew and
keeps the best for this project. It costs half a minute and accept_mask then
uses what it found.

How close to look at objects you box is measured, not assumed. Once per
project, call zoom_plan:
it hands back views of an image the person already annotated, widest first.
Look at each with image_get_b64 (crop_json, max_side 1280), say where you think
the objects are, and call zoom_score -- their answer is known, so it tells you
whether your eyes worked at that width. Take the widest view that still finds
most of them, and use crops that size on the images with no mask -- crop_json in
the picture's own pixels, with from_width and from_height its own size --
passing each crop as from_box_json to accept_mask. If the widest view is the whole picture,
there is nothing to crop and you work as before.

A crowded frame: your coordinates drift to the top of it. Do not point at a
frame with dozens of objects in one go. Take it in horizontal strips with
crop_json -- '[0,0,W,H/2]' then '[0,H/2,W,H]', W and H the picture's own size,
which image_get_b64 also takes as from_width and from_height -- and pass the
same crop_json to zoom_score. On a crowded frame the two strips find far more
than the whole frame does.

The reply says how big one object is in the copy you were handed, measured
from what the person drew and scaled to that copy. Your boxes should be about
that size.

Coordinates: image_get_b64 hands you a scaled copy and says its size. Box
things in THAT copy's coordinates -- the numbers you can see -- and the bridge
puts them back on the full picture itself. You need not repeat the copy's size
or crop: the last one handed over for that image is assumed -- except in
image_get_b64's crop_json, which takes the from_width and from_height of the
copy you read it off, or the picture's own size. Do not scale
boxes up yourself, and write no more than the call needs -- every word costs a
second of the answer.

When a box is the part you cannot draw -- objects lying across each other, or
a defect whose extent is not a rectangle -- point instead: accept_points takes
one point per object (or a group of points on one object, when its parts look
different) and tries boxes of the size the teachers actually are, keeping
whichever the band accepts. Coordinates and from_* work as in accept_mask.

What the numbers mean: accept_mask judges SAM's masks against the sizes and
texture the person's own masks have, so a refusal is information -- "the same
object" means that part is done, move to another; a size or texture refusal
means what SAM answered was not the object -- the picture of its answers shows
whether the box or the level was wrong. Never write a mask with nothing in it.

Before the first image of a job, call ask_user once with choices_json to settle
how to work: how much the count matters, and what to do with an image that
falls short. Offer two or three short answers, the most usual first, and follow
the answer for the rest of the job without asking again.

teacher_band says how many objects its band came from. Few teachers is the
one thing that makes the band wrong rather than strict: the biggest of eight
objects is not the biggest there is, so a box the right size for what you can
see is refused for its area. When that keeps happening on objects that are
plainly there, say so and ask for more hand-drawn images -- do not answer it by
drawing smaller boxes, which is how a project ends up labelled at half size.

expected_per_frame is what the teachers show, not a quota. If the band refuses
everything else on an image, the objects you can see are the ones that are
there: write_kept them and mark_review the shortfall. Moving a refused box a
few pixels does not change the answer.

The object word carries any judgement: "fully open flower (not a bud)", not
"flower". If a write says needs_review -- far fewer than the teachers show --
look at what was found before you flag it: a surface with fewer marks is an
answer; marks missed, or rings on something else, are a reason to mark_review
it with what you saw rather than leave it looking finished.

Every image with no mask is yours to do. annotation_status counts them and
names as many as it can; when it says the list was cut, ask again after you
have written some. Do not report completion while any are left -- if you
stop, you will be told what is still there and asked to carry on.

Finish by listing the images that need a person's eyes, first, then what you did."""

#: What to add to the system prompt so the model speaks the operator's
#: language. Tool names, arguments and JSON must stay as they are: they are
#: the wire format, not prose.
LANG_NOTE = {
    "ja": ("すべて日本語で書いてください。人に見せる文（ask_user の質問、最後の報告）は日本語にします。"
           "ツール名・引数のキー・JSON はそのまま英語で使ってください。"),
    "en": "",
}

#: Replies that mean "stop asking and finish the job". A person who has said
#: it once should not be asked again, and telling the model is not enough: it
#: asks anyway, and then the person is answering the same question per image.
GO_AHEAD = ("ぜんぶ", "全部", "すべて", "全て", "まとめて", "最後まで", "聞かないで", "確認しないで", "確認なし",
            "そのまま続け", "続けて", "all", "go ahead", "don't ask", "do not ask", "no need to ask")


def wants_no_more_questions(reply: str) -> bool:
    # An English word is matched as a word: "all" was found inside "small",
    # "call" and "really", and a reply that asked for the opposite turned the
    # questions off for the rest of the run.
    r = (reply or "").strip().lower()
    for k in (g.lower() for g in GO_AHEAD):
        if k.isascii():
            if re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", r):
                return True
        elif k in r:
            return True
    return False


def _not_asked(lang: str, said: str, to: str, choices: list[str]) -> str:
    """The answer to a question asked after the person stopped the questions.

    It was a yes. A run on a scattering of specks asked, with every image
    written, whether to redo the short ones, redo all of them, or leave
    them and report, and was answered はい -- which is
    none of the three. It took the first, found those images put down, and
    went round flagging and unflagging them until it was stopped. The only
    answer there is, is what the person did say: their own words from when
    the questions stopped, marked as their answer to that question -- whether
    it also answers this one is the model's to judge: to a per-image "may I
    write this?" it does -- and the decision left with the model, where they
    left it. Choices are named so the answer is plainly none of them; none is
    picked here.
    """
    ja = lang == "ja"
    said = " ".join(str(said).split())[:300]
    to = " ".join(str(to).split())[:120]
    if ja:
        text = (f"この質問は人に送っていません。人は前に「{to}」という質問に「{said}」と答え、"
                "それからは質問を人に送っていません。その答えが今の質問にも当てはまるかは、あなたが判断してください。"
                "どうするかは、画像と人の言葉からあなたが決めて進めてください。")
        if choices:
            text += ("、".join(f"「{c}」" for c in choices)
                     + "のどれも、人は選んでいません。どれにするか、どれでもないかはあなたが決めます。")
        return text + "決めたことと、その理由を最後の報告に書いてください。"
    text = (f"This question was not put to the person. Earlier, asked \"{to}\", they answered "
            f"\"{said}\", and since then questions have not been sent to them. Whether their answer "
            "then also answers this question is yours to judge. Decide yourself, from the images and "
            "what they said, and carry on. ")
    if choices:
        text += ("They picked none of " + ", ".join(f"\"{c}\"" for c in choices)
                 + ": which one, or none of them, is yours to decide. ")
    return text + "Say in your final report what you decided and why."


ASK_TOOL = {"type": "function", "function": {
    "name": "ask_user",
    "description": "Show the person what you have for an image and ask before writing it. Say what you kept, what you "
                   "rejected and why, in one or two lines, and end with a question, in the operator's language. "
                   "Returns their reply; act on it: a yes ('yes', 'はい') means write it (write_kept, or spot_write "
                   "for specks), anything else tells you "
                   "what to change (accept_mask more boxes, or reset and redo), then ask again. If the reply says to "
                   "carry on without asking ('ぜんぶやって', 'all of them', 'do not ask again'), stop calling this tool "
                   "for the rest of the job and write each image as you finish it.",
    "parameters": {"type": "object", "properties": {
        "item_id": {"type": "string"},
        "question": {"type": "string"},
        "choices_json": {"type": "string",
                         "description": "optional: a JSON list of 2-4 short answers, offered as buttons. "
                                        "Use it for a question with settled answers -- how to handle an "
                                        "image, how strict to be -- and put the one you would pick first."}},
                   "required": ["item_id", "question"]}}}
#: The answer to ask_user on a run with nobody to put it to: a command-line
#: run, or a caller that offers no way to reply. The brief asks for a question
#: before the first image, and answering it "no such tool" counted it as a
#: failure and left the model to guess why.
NOBODY_TO_ASK = {
    "ja": ("この run には質問に答える人がいません。画像と指示からあなたが判断して進め、"
           "何を仮定したかと、それがどの画像に関わるかを最後の報告に書いてください。"),
    "en": ("Nobody is watching this run to answer. Decide yourself, from the images and the "
           "instruction, carry on, and say in your final report what you assumed and which "
           "images it affects."),
}


#: How far the picture outside a written mask is dimmed in the review picture.
#: The picture of SAM's answers the bridge draws dims by the same amount, and
#: this is the value to change: mcp_recipe.SHEET_DIM is a copy, held equal to
#: it by a test, because the bridge runs in a process of its own.
REVIEW_DIM = 0.35


def _grow(m, px: int):
    """A boolean mask widened by px pixels, four-neighbour, without scipy."""
    out = m.copy()
    for _ in range(max(0, px)):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def _render(image_b64: str, boxes: list, mask_b64: str | None = None, max_side: int = 640) -> str:
    """The image with the model's boxes (vermilion) and, if given, the written mask, as a small JPEG.

    The mask is shown by what it is not: everything outside it is dimmed and its
    edge is drawn in white just outside it, so it reads the same on any picture
    and to any eye. It was a blue tint, which the review question called
    orange, on pictures with a blue background: the model was asked about an
    orange that was not in the picture, and a mask that spilled onto the
    background would have been blue on blue -- the one mistake the review is
    there to see.
    max_side 0 keeps the picture's own size.
    """
    import base64
    import io

    from PIL import Image, ImageDraw
    im = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    if mask_b64:
        import numpy as np
        mi = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
        if mi.size != im.size:
            # The picture here is the copy the model was shown -- enlarged when
            # the frame is smaller than the copy's side -- and the mask is the
            # frame's own pixels. Indexing one by the other raised IndexError,
            # inside the except that made the check after a write silently not
            # happen: masks written run after run, and not one of
            # them looked at. Nearest, because
            # this is a picture to look at and a smoothed edge would suggest a
            # precision the mask does not have.
            mi = mi.resize(im.size, Image.NEAREST)
        m = np.array(mi)
        m = m[..., 0] if m.ndim == 3 else m
        fg = (m != 0) & (m != 255)
        a = np.array(im).astype(np.float32)
        a[~fg] *= REVIEW_DIM
        # about three pixels in the picture the model is actually shown, which
        # is this one shrunk to max_side below; drawn thinner, the shrink and
        # the JPEG blur it into the dimmed side
        shown = min(1.0, max_side / max(im.size)) if max_side else 1.0
        a[_grow(fg, max(2, int(3.0 / shown + 0.999))) & ~fg] = 255.0
        im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle([b[0], b[1], b[2], b[3]], outline=(213, 94, 0), width=max(2, im.width // 300))
    z = min(1.0, max_side / max(im.size)) if max_side else 1.0
    if z < 1.0:
        im = im.resize((int(im.width * z), int(im.height * z)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


#: The tools whose answer carries a picture, as JPEG base64 under SHEET_KEY:
#: SAM's answers for the box, or what spot_detect found. It is taken out of the
#: answer before the model reads it as text, put in front of the model for its
#: next reply only, and never filed in images_seen: boxes are drawn on the
#: image_get_b64 copy, and a picture of another size taken for that copy is how
#: every y once came back scaled wrong. spot_detect answered with a count and
#: nothing to look at; on a run of specks that count against half the
#: teacher's was the only judgement made of an image, and on an image it
#: flagged, many of the finds sat off the surface being labelled. IMAGE_WORK_TOOLS
#: named spot_detect already, and nothing else keys on this set: what is kept,
#: tried, repeated and boxed is counted on the accept tools by name.
SHEET_TOOLS = frozenset({"accept_mask", "accept_masks", "accept_points", "spot_detect"})
SHEET_KEY = "candidates_jpeg"
#: Said with the picture. How to read it is in the brief; this names which
#: call and which box it is, since one turn can hold several.
SHEET_NOTE = {
    "ja": ("{iid} に対する SAM の答え（{what}）の絵です。答えごとに 1 枚のパネルで、小さい順に並んでいます。"
           "パネルの白い線の内側がその答え（外は暗くしています）、朱の四角があなたの箱（点で指したときは朱の点、輪郭を送ったときは朱の線）、"
           "KEPT は保持した答え、fits / no: <理由> は教師の帯の判定です"
           "（overlaps kept は、帯は通るが保持済みのマスクと重なる答え）。"
           "あなたが受けた指示は「{asked}」です。この絵から座標を読まないでください。"),
    "en": ("SAM's answers for {iid} ({what}), as a picture: one panel per answer, smallest first. "
           "Inside the white line of a panel is that answer (the rest dimmed), the orange "
           "rectangle is your box (orange dots where you pointed, an orange line where you traced), "
           "KEPT is the answer kept, and 'fits' "
           "or 'no: <reason>' is what the "
           "teacher band said ('overlaps kept': it fits, but covers a mask already kept). "
           "You were asked: \"{asked}\". Never read coordinates off this picture."),
}
#: Added when the image was written later in the same turn.
SHEET_WRITTEN = {
    "ja": "（この手番のうちに write_kept しました。KEPT が書いたものです）",
    "en": " (written later in this turn: KEPT is what went to the mask)",
}
#: What the picture becomes once the model has replied to it.
SHEET_SEEN = {
    "ja": "[{iid} の SAM の答え（{what}）の絵は見ました: {summary}]",
    "en": "[SAM's answers for {iid} ({what}), looked at: {summary}]",
}
#: What it becomes when it was never shown: no room left in the request for it,
#: a model server that had to be asked again, or a run that ended first.
SHEET_UNSHOWN = {
    "ja": "[{iid} の SAM の答え（{what}）の絵は見せられませんでした。言葉でだけ: {summary}]",
    "en": "[SAM's answers for {iid} ({what}), not shown as a picture; in words: {summary}]",
}
#: Said with the picture of what spot_detect found. A run of specks named the
#: look-alike marks beside the surface "not the target" at its second step, and
#: was never shown that the detector had found them: all it had of an image was
#: a count to hold against the teacher's.
SPOT_NOTE = {
    "ja": ("{iid} で spot_detect が見つけたもの（{what}）の絵です。上は画像全体を縮小したもので、"
           "見つけたものを一つずつ朱の丸で囲んでいます。番号の付いた白い四角は、下に並べた同じ番号の拡大"
           "（元の画像を等倍で切り出したもの）の場所で、上の段は見つけた数が多い所、下の段は最も少ない所です。"
           "黄色の線はあなたが渡した範囲が実際に置かれた位置、黄色の × はその外にあって書かれないものです。"
           "丸が、ラベルを付けている面の上にあるか、別の物の上にあるかを見てください。"
           "あなたが受けた指示は「{asked}」です。"
           "面の外に丸があるなら、この画像を image_get_b64 で見て、その写しの目盛りから面を囲む "
           "region_json を読み、spot_detect をもう一度（like_item_id はそのまま。点で指したなら、"
           "点もその同じ写しから読み直し、別の写しの from_* は付けない）。"
           "丸がラベルを付けるものなら spot_write（人が描いた画像なら rehearse）。"
           "そのうえで、拡大の中の面の上に、丸の付いたものと同じような点が丸なしで目立つなら、それは見逃しです。"
           "何を見逃したかを添えて mark_review（人が描いた画像では不要。見逃しは rehearse の点数に出ます）。"
           "どちらでもなければ、見たことを添えて mark_review。この絵から座標を読まないでください。"),
    "en": ("What spot_detect found on {iid} ({what}), as a picture. At the top is the whole picture, "
           "shrunk, each find ringed in orange; a numbered white square is where the close-up of that "
           "number below comes from, at full size -- the upper row where it found most, the lower row "
           "where it found least. A yellow line is a region you gave, where it landed; a yellow x is a "
           "find outside it, which is not written. Look at where the rings are: on the surface you are "
           "labelling, or on something else. You were asked: \"{asked}\". If some sit off that surface, "
           "image_get_b64 this image, read region_json around the surface off that copy's ruler, and "
           "spot_detect again with it -- like_item_id as before, or, if you pointed, the point read again "
           "off that same copy, with no from_* left from another. If the rings are what you are labelling, "
           "spot_write (on an image the person drew, rehearse). Then, if a close-up shows specks on that "
           "surface like the ringed ones but with no ring, those were missed: mark_review it saying what was "
           "missed (not on an image the person drew: rehearse's score says that). Otherwise mark_review it "
           "with what you saw. Never read coordinates off this picture."),
}
#: Added when spot_write wrote these later in the same turn, before the picture was seen.
SPOT_WRITTEN = {
    "ja": "（この手番のうちに、この絵を見る前に spot_write しました。丸が今この画像に書かれているものです）",
    "en": " (spot_write wrote these later in this turn, before you saw them: the rings are what is on the image now)",
}
#: Added to an earlier picture of an image when a later spot_detect on it replaced what it found.
SPOT_SUPERSEDED = {
    "ja": "（このあとの spot_detect がこの画像の結果を置き換えました。この丸は書かれるものではありません）",
    "en": " (a later spot_detect on this image replaced these: they are not what would be written)",
}
SPOT_SEEN = {
    "ja": "[{iid} で spot_detect が見つけたもの（{what}）の絵は見ました: {summary}]",
    "en": "[What spot_detect found on {iid} ({what}), looked at: {summary}]",
}
SPOT_UNSHOWN = {
    "ja": "[{iid} で spot_detect が見つけたもの（{what}）の絵は見せられませんでした。言葉でだけ: {summary}]",
    "en": "[What spot_detect found on {iid} ({what}), not shown as a picture; in words: {summary}]",
}


def _sheet_verdict(rung: dict) -> str:
    """What the band said of a rung, in the word the picture uses (mcp_recipe.sheet_verdict)."""
    if rung.get("passes"):
        return "fits"
    why = str(rung.get("why") or "").split()
    return f"no: {why[0]}" if why else "no"


def _sheet_summary(res: dict) -> str:
    """The picture in words: each rung, its share of the picture, and what became of it."""
    def one(r: dict) -> str:
        kept = r.get("level") if r.get("accepted") else None
        return ", ".join(f"{x.get('level')} {float(x.get('area_pct') or 0):.1f}% "
                         + ("KEPT" if kept and x.get("level") == kept else _sheet_verdict(x))
                         for x in (r.get("levels") or []) if isinstance(x, dict))
    if "results" in res:
        rs = res.get("results") or []
        noun = "object" if "shapes_tried" in res else "box"      # accept_points answers per object
        return "; ".join(f"{noun} {n}: {one(rs[n - 1])}" for n in (res.get("candidates_of") or [])
                         if isinstance(n, int) and 0 < n <= len(rs) and isinstance(rs[n - 1], dict))[:400]
    return one(res)[:300]


def _spot_summary(res: dict) -> str:
    """The picture of what spot_detect found, in words: how many, at what sensitivity, and what a region left out."""
    said = f"{res.get('count')} found at sensitivity {res.get('sensitivity')}"
    if res.get("outside_region"):
        said += f", {res['outside_region']} outside your region left out"
    return said[:300]


def _sheet_what(fn: str, fargs: dict, res: dict, lang: str) -> str:
    """Which call and which box a picture is of, in the model's own coordinates."""
    ja = lang == "ja"
    if fn == "spot_detect":
        # In the call's own arguments -- the example or the point, and the region
        # as it was sent -- which is what a second call sends again; after the
        # reply these words are all that is left of the picture.
        how = (f"like_item_id={fargs['like_item_id']}" if fargs.get("like_item_id") else
               f"x={fargs.get('x')}, y={fargs.get('y')}")
        if fargs.get("region_json"):
            # What a second call sends again: an outline cut short is JSON that
            # never closes, and nothing said it was cut.
            r = " ".join(str(fargs["region_json"]).split())
            how += ", region_json=" + (r if len(r) <= 240 else r[:240] + " ...(cut)")
        return f"spot_detect、{how}" if ja else f"spot_detect, {how}"
    if fn == "accept_masks":
        of = ", ".join(str(n) for n in (res.get("candidates_of") or []))
        return f"accept_masks、あなたの箱のうち {of} 番目" if ja else f"accept_masks, boxes {of} of your call"
    if fn == "accept_points":
        of = ", ".join(str(n) for n in (res.get("candidates_of") or []))
        return (f"accept_points、あなたの点のうち {of} 番目の物体" if ja else
                f"accept_points, objects {of} of your call")
    if fargs.get("outline_json"):
        return "accept_mask、あなたの輪郭" if ja else "accept_mask, your outline"
    said = " ".join(str(fargs.get("box_json") or "").split())[:60]
    if said:
        return f"accept_mask、あなたの箱 {said}" if ja else f"accept_mask, your box {said}"
    said = " ".join(str(fargs.get("points_json") or "").split())[:60]
    return f"accept_mask、あなたの点 {said}" if ja else f"accept_mask, your point {said}"


def _sheet_marked(res: dict) -> dict:
    """The answer as the model reads it: the picture's place taken by a word.

    Put before the results, so a batch cut at the tool text's limit keeps it.
    """
    out: dict = {}
    for k, v in res.items():
        if k == "results":
            out["candidates"] = "picture attached"
        out[k] = v
    out.setdefault("candidates", "picture attached")
    return out


def _sheet_words(sheet: dict, lang: str, seen: bool) -> str:
    """A picture of SAM's answers, or of what spot_detect found, as the line that stands for it in the history."""
    if sheet.get("spots"):
        said, wrote = (SPOT_SEEN if seen else SPOT_UNSHOWN), SPOT_WRITTEN
    else:
        said, wrote = (SHEET_SEEN if seen else SHEET_UNSHOWN), SHEET_WRITTEN
    return (said.get(lang, said["en"]).format(iid=sheet["iid"], what=sheet["what"], summary=sheet["summary"])
            + (wrote.get(lang, wrote["en"]) if sheet["written"] else ""))


def _sheets_without_room(sheets: list, room: int) -> list:
    """The pictures of SAM's answers that do not fit in `room` characters.

    They are sent beside the history, in what room it leaves under the budget,
    and never make it smaller: counted in the trim instead, two batches in one
    turn cut the history back past the image the boxes are read from, and left
    tool results with no call before them. A refusal's picture goes in before a
    kept box's -- a refusal is where there is a choice to make -- and otherwise
    the earlier before the later. The rest are said in words.
    """
    left, fit = room, set()
    # A picture a later spot_detect replaced goes last: it is not what would be written.
    for i in sorted(range(len(sheets)), key=lambda i: (bool(sheets[i].get("superseded")),
                                                       not sheets[i]["refused"], i)):
        size = _msg_size(sheets[i]["msg"])
        if size <= left:
            fit.add(i)
            left -= size
    return [s for i, s in enumerate(sheets) if i not in fit]


def _spot_room(sheets: list, lang: str) -> int:
    """What the pictures of what spot_detect found need beside the history, with
    the words each would be filed as: the room the trim at the top of a step
    makes for them."""
    return sum(_msg_size(s["msg"]) + len(_sheet_words(s, lang, False))
               for s in sheets if s.get("spots"))

#: The bridge, as a subprocess. It stays a separate process on purpose:
#: its writes go through the API like anybody else's, carrying the header
#: the activity feed and the "do not paint over a person's mask" guard
#: both read. The installer stages it, with the recipe and the playbook it
#: loads from beside itself, at the same place relative to the app
#: (scripts/build_installer.py, step 3); where it is missing, the route
#: answers 503 instead of a run failing when the subprocess starts.
SRV = Path(__file__).resolve().parents[5] / "scripts" / "mcp_server.py"
TOOLS = ["annotation_status", "dataset_images", "image_get_b64", "teacher_band", "accept_mask",
         "accept_masks", "accept_points",
         "zoom_plan", "zoom_score", "calibrate_sam", "teacher_view",
         "write_kept", "mark_review",
         # mask_stats is not offered. Size statistics against the person's
         # masks were read as a verdict however they were worded: objects each
         # looked at in a picture after it was written were flagged, run after
         # run, for being larger than the one on the teacher -- even under a
         # note that a difference in size is not a fault -- and images of
         # specks were flagged for their count.
         # A frame of scattered specks has no boxes to draw and nothing to
         # decide crop by crop: a run on an image of scattered specks read
         # crop after crop of the same image and wrote no mask. spot_detect measures the
         # one speck it is pointed at and finds the rest; spot_write puts that
         # on the image, because what was found is a whole-image mask rather
         # than an object for the teacher band, and write_kept would shrink it
         # by the calibrated erosion -- which is how a 3px speck disappears.
         # The mask itself never comes back to the model: it is a hundred
         # kilobytes of base64 against a 24k window.
         "spot_detect", "spot_write",
         # A rehearsal on an image the person already drew, scored against their
         # own mask. Without it a run starts on a whole set of pictures whose answer
         # nobody knows, having never once checked itself against one whose
         # answer is on file.
         "rehearse",
         # The thing write_kept is held back for. It was left out of this list
         # when the gate was added, so every write was refused with "call
         # steps(project_id)" naming a tool the model had never been given: a
         # run could spend hundreds of events, picture after picture and
         # rehearsal after rehearsal, being told to do something it could not
         # do, and write nothing.
         "steps"]
#: A job should be as long as the work is. A limit that suited a handful of
#: images ran out part-way through a slightly longer job, which is the worst
#: place to stop: masks written, the person believing the rest were coming.
#: The number is high enough now to be no one's limit but the person's.
MAX_STEPS = 2000
#: What has to end a run is not its length but its standing still. A model that
#: has neither kept a mask nor written one for this many steps is not working
#: towards anything, and the step limit was only catching that by accident.
NO_PROGRESS_STEPS = 40
#: How many things a run is asked to settle before it labels anything: look at
#: the pictures, read the teachers and name the object, measure how to cut it,
#: rehearse on an image whose answer is known. The bridge owns the list; this is
#: how many of them there are, so the harness can refuse a write before the
#: model has been through them.
STEPS_BEFORE_LABELLING = 4
#: The longest side of the copy a model is shown. A vision model resizes a
#: picture to a few hundred pixels before it looks, so a full-size camera
#: photograph is tens of megabytes of base64 spent on nothing -- and, on one
#: run, a 400 from the server that ended the job. The bridge scales the copy
#: and puts the boxes back.
MODEL_IMAGE_SIDE = 1280
#: Arguments of the bridge's tools that the model is never offered and never
#: gets to pass. overwrite lets a write replace a mask a person drew, and no
#: copy of theirs is kept: runs sent it on their own after a refusal, and the
#: person's masks are what the teacher band is measured from. Replacing one is
#: a person's decision, made on the annotation screen.
MODEL_WITHHELD_ARGS: dict[str, tuple[str, ...]] = {
    "write_kept": ("overwrite",),
    "spot_write": ("overwrite",),
}


def _schema_for_model(name: str, schema):
    """A tool's input schema as the model is offered it: without what it may not pass."""
    drop = MODEL_WITHHELD_ARGS.get(name)
    if not drop or not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if isinstance(schema.get("properties"), dict):
        out["properties"] = {k: v for k, v in schema["properties"].items() if k not in drop}
    if isinstance(schema.get("required"), list):
        out["required"] = [k for k in schema["required"] if k not in drop]
    return out


def _without_withheld(fn: str, fargs: dict) -> tuple[dict, list[str]]:
    """The model's arguments less what it may not pass, and which of those it sent."""
    sent = [k for k in MODEL_WITHHELD_ARGS.get(fn, ()) if k in fargs]
    if not sent:
        return fargs, []
    return {k: v for k, v in fargs.items() if k not in sent}, sent


def _call_args(call) -> dict:
    """A tool call's arguments as a dict, whatever the model server handed back."""
    args = call.get("args") if isinstance(call, dict) else None
    return args if isinstance(args, dict) else {}


def kept_count(res: dict) -> int:
    """How many masks a reply kept.

    accept_mask answers whether one was kept; accept_masks answers how many of
    a batch were. Reading the singular's field on the plural's reply counted
    every batch as nothing kept, and a run that was working was stopped at the
    no-progress limit with nothing written.
    """
    took = res.get("accepted") if isinstance(res, dict) else None
    if isinstance(took, bool):
        return int(took)
    if isinstance(took, int):
        return took
    return 0


def kept_growth(res: dict, high: dict, item_id: str) -> int:
    """How many masks an answer ADDS to what is already kept on this image.

    accept_masks answers "accepted: 9" for nine boxes it took, and it answers
    the same nine again for the same nine boxes sent again; kept_so_far is the
    field that tells those apart. Counting the accepted ones made a repeat look
    like work, and the repeat was the failure: a run could send the same batch
    and the same single box after it again and again, for many minutes, and
    every batch cleared the refusal counter that PROBE_REFUSALS_STOP would
    have stopped it with. write_kept answered "written: false" each of those
    times because there was nothing new to write, and the mask on disk did not
    change for most of it. Nothing was watching the one number that had stood
    still the whole time.
    """
    took = kept_count(res)
    if not took:
        return 0
    now = res.get("kept_so_far") if isinstance(res, dict) else None
    if not isinstance(now, int):
        # An answer that does not say how many it holds: trust the count, as
        # this did before kept_so_far existed.
        high[item_id] = high.get(item_id, 0) + took
        return took
    grew = now - high.get(item_id, 0)
    high[item_id] = max(now, high.get(item_id, 0))
    return max(0, grew)


#: How many times one image may be written, looked at, and written again.
#: The check after a write tells the model to try another rung when the mask
#: came out in pieces, and on an object that comes out in pieces at every rung
#: that is an instruction it can follow for ever: subpart, part, write, look,
#: subpart, part, write, look -- round after round on one image, each one a
#: success, every guard we had watching failures. The answer after this many
#: rounds is the answer; it is written, and the run moves on.
REVIEW_REDOS = 2


def _no_review_because(fresh: bool, item_id: str, images_seen: dict,
                       looked_at: int = 1, allowed: int = REVIEW_REDOS) -> str | None:
    """Why the mask just written will not be shown back, or None if it will.

    The check after a write is the only thing between a wrong mask and the
    project: it puts the mask back in front of the model -- the rest of the
    picture dimmed, the mask's edge in white -- and asks
    whether the object is inside it. Two conditions gated it and an
    ``except Exception: pass`` swallowed the rest, so when it did not happen
    there was no trace of that anywhere -- not in the log, not on the screen,
    not in the answer write_kept returned.
    """
    if not fresh:
        return None          # nothing new went to disk; there is nothing to check
    if looked_at > allowed:
        # Asking again would get the same answer and the same redo. Silent on
        # purpose: this is the run being sensible, not a failure to report.
        return None
    if item_id not in images_seen:
        return ("this run never fetched the picture, so there is nothing to draw the "
                "mask on. Call image_get_b64 for an image before writing to it")
    return None


def write_is_new(kept_high: dict, written_at: dict, item_id: str) -> bool:
    """Whether a successful write had anything new to write.

    write_kept answers "written: true" for a file it has just rewritten byte
    for byte. A run could write the same mask of one image over and over for
    many minutes -- same length, same sha256, only the timestamp moving -- and
    every one of those answers cleared the refusal counter that would have
    stopped it. The masks kept since the last write of
    this image are what say whether there was anything to write; the tool's own
    success does not.
    """
    fresh = kept_high.get(item_id, 0) > written_at.get(item_id, -1)
    written_at[item_id] = kept_high.get(item_id, 0)
    return fresh


class Progress:
    """How long since anything was achieved.

    "Achieved" is deliberately narrow: a mask accepted, an image written, a
    person answering. Looking at an image again, or being refused, is not
    progress -- being refused repeatedly is exactly the state this ends.
    """

    def __init__(self, limit: int = NO_PROGRESS_STEPS) -> None:
        self.limit = limit
        self.idle = 0
        self.last = ""
        #: How many times anything has moved, which is what tells a repeated
        #: question from a question asked again about a changed dataset.
        self.moves = 0

    def step(self) -> None:
        self.idle += 1

    def moved(self, what: str = "") -> None:
        self.idle = 0
        self.last = what
        self.moves += 1

    @property
    def stalled(self) -> bool:
        return self.idle >= self.limit
#: The same call, failing the same way, this many times in a row before the
#: answer stops being the error and starts being an instruction. A model that
#: has just been told "400 Bad Request" will send the identical request again,
#: and again: seen in the wild as a page of identical red lines, filling the
#: context with the same sentence until the step limit.
SAME_FAILURE_WARN = 3
SAME_FAILURE_ABORT = 6
#: The same tool failing over and over on the same image, however the arguments
#: are dressed. SAME_FAILURE_ABORT keys on the arguments, so a model that varies
#: them never reaches it: a run could send accept_mask on one image dozens of
#: times, inventing a segmenter name each round -- efficient_sam_b,
#: efficient_sam_l, efficient_sam, sam2_base, sam2 -- and every 400 came back
#: listing the five names that exist. Each new invention was a new signature, so
#: the counter reset every time; a rehearse call in between reset it again.
#: Counted per tool and image, and cleared only when THAT tool succeeds on THAT
#: image, so nothing else going right can hide it.
STUCK_ON_ITEM_WARN = 4
STUCK_ON_ITEM_ABORT = 12
#: A refusal is not a failure -- the band saying "that is not the object" is
#: the recipe working -- but a model that answers a refusal by moving the box
#: ten pixels will do it until the steps run out. A long string of those on
#: one image ended a run part-way through an image, with others already done.
#: After this many in a row on one image, the probe is refused here instead.
PROBE_REFUSALS_STOP = 10
#: The same accept, sent again with the same arguments. With reset=true it
#: forgets what was kept and keeps the same thing again, and every keep read as
#: progress, so nothing caught it: a run could send one accept_points dozens
#: of times on one image. On the second it is shown what it has tried on the image and
#: asked for a way it has not; on the fourth the image is left as it is,
#: flagged for a person, and the next one shown. What it kept is not written
#: for it -- on that run it was worse than what was already there.
SAME_ACCEPT_OTHER_WAY = 1
SAME_ACCEPT_PUT_DOWN = 3
#: How many of an image's attempts the list of what was tried shows.
TRIED_SHOWN = 6

#: The bridge's rehearsal verdicts, as a person reading in Japanese is told
#: them. The bridge says them to the model in English, and that stays so; the
#: line built from them is the person's, and it was half one language and half
#: the other. What this does not know is carried as the bridge said it.
_REHEARSAL_JA = (
    (re.compile(r"this is what the rest of the job will look like"),
     lambda m: "この先の画像もこの通りになる見込みです"),
    (re.compile(r"the overlap is ([\d.]+); on this image your masks and theirs are largely different pixels"),
     lambda m: f"重なりが {m[1]} で、この画像では自分と教師のマスクはほとんど別の画素です"),
    (re.compile(r"(\d+) of their (\d+) objects have nothing of yours on them"),
     lambda m: f"教師の物体 {m[2]} 個のうち {m[1]} 個に、自分のマスクがありません"),
    # The rung advice after it is for the model, and is left to the model.
    (re.compile(r"one object of theirs is (\d+) pieces of yours; the segmenter is answering with parts(?:\.[^;]*)?"),
     lambda m: f"教師の物体 1 個が自分の {m[1]} 片に分かれています（セグメンターが部品で答えています）"),
    (re.compile(r"(\d+) of their (\d+) specks have nothing of yours on them"),
     lambda m: f"教師の点 {m[2]} 個のうち {m[1]} 個に、自分の点がありません"),
    (re.compile(r"(\d+) of your (\d+) specks sit where they drew nothing"),
     lambda m: f"自分の点 {m[2]} 個のうち {m[1]} 個は、教師が何も描いていない所にあります"),
)
#: Why a rehearsal could not be scored, likewise.
_NO_REHEARSAL_JA = (
    (re.compile(r"nothing kept or found for this image yet:.*", re.S),
     lambda m: "この画像にはまだ保持したものも見つけたものもありません。"
               "先に画像を見て accept_mask（点なら spot_detect）をしてから呼んでください"),
    (re.compile(r"^(\S.*?) has no mask of the person's -- .*? The images they drew: (.*)$", re.S),
     lambda m: f"{m[1]} には人が描いたマスクがないので、比べる相手がありません。人が描いた画像: {m[2]}"),
    (re.compile(r"this image has no mask of the person's, so there is nothing to rehearse against\..*", re.S),
     lambda m: "この画像には人が描いたマスクがないので、比べる相手がありません。"
               "teacher_band が教師として挙げた画像を選んでください"),
    (re.compile(r"the person's mask on this image is empty.*", re.S),
     lambda m: "この画像の人のマスクは空なので、教師の物体を見つけられたかは分かりません"),
)
#: The bridge's names for the steps before labelling, for a person reading in
#: Japanese. As with the rehearsal, the model reads the bridge's own.
STEP_NAMES_JA = {
    "look at the pictures": "画像を見る",
    "read the teachers, and name the object": "教師を読み、対象に名前を付ける",
    "measure how to cut it": "切り出し方を測る",
    "rehearse on a known answer": "答えの分かっている画像でリハーサルする",
}


def _said_in_japanese(text: str, table) -> str:
    """The bridge's English, part by part in Japanese where the part is one this knows."""
    found = sorted((m.start(), m.end(), say(m)) for pat, say in table for m in pat.finditer(text))
    out, at = [], 0
    for start, end, ja in found:
        if start < at:
            continue                        # inside a part already said
        rest = text[at:start].strip(" ;.")
        if rest:
            out.append(rest)                # a part this does not know, as it came
        out.append(ja)
        at = end
    rest = text[at:].strip(" ;.")
    if rest:
        out.append(rest)
    return "。".join(out)


def _rehearsal_line(res: dict, lang: str) -> str:
    """The dry run's numbers as one sentence, for the person rather than the model.

    They were only ever inside the tool result, which the panel hides and then
    truncates at two hundred characters -- cutting, on a typical payload, in the
    middle of the word "verdict". The one time they reached the screen it was
    because the model happened to repeat them in a question.
    """
    if not res.get("scored"):
        why = str(res.get("why") or "")
        if lang == "ja":
            return f"リハーサルできません: {_said_in_japanese(why, _NO_REHEARSAL_JA)}"
        return f"no rehearsal: {why}"
    got, want = res.get("you_found", 0), res.get("their_objects", 0)
    # A rehearsal of specks is counted in specks, not masks and pieces: this
    # line would have said "None masks in None pieces" for every one of them.
    specks = res.get("source") == "spot_detect"
    if lang == "ja":
        mine = (f"自分の {res.get('your_objects')} 点のうち {res.get('yours_on_nothing')} 点は描かれていない所"
                if specks else f"自分の {res.get('your_masks')} マスクが {res.get('your_blobs')} 片")
        head = (f"リハーサル {res.get('item_id')}: 重なり {res.get('overlap')}、"
                f"教師の {want} 個中 {got} 個に到達、はみ出し {res.get('outside_theirs_pct')}%、{mine}")
    else:
        mine = (f"{res.get('yours_on_nothing')} of your {res.get('your_objects')} specks on nothing"
                if specks else f"{res.get('your_masks')} masks in {res.get('your_blobs')} pieces")
        head = (f"rehearsal {res.get('item_id')}: overlap {res.get('overlap')}, "
                f"{got} of the teacher's {want} objects reached, "
                f"{res.get('outside_theirs_pct')}% outside theirs, {mine}")
    verdict = str(res.get("verdict") or "").strip()
    if verdict and lang == "ja":
        verdict = _said_in_japanese(verdict, _REHEARSAL_JA)
    return f"{head} — {verdict}" if verdict else head


def _asked_for(instruction: str) -> str:
    """The person's own words, short enough to sit inside another question.

    Quoted rather than paraphrased: the whole point of asking is to check the
    mask against what was actually asked for, and a paraphrase is the model's
    own understanding, which is the thing under test.
    """
    said = " ".join(str(instruction or "").split())
    return said[:160] + ("…" if len(said) > 160 else "")


def _step_just_answered(res: dict) -> int:
    """Which step the answer was for, from the reply to answering it.

    steps replies with the step it has MOVED ON to, so the one just answered is
    the one before -- except at the end, where it replies "done" and there is
    no next step to count back from.
    """
    if res.get("done"):
        return int(res.get("steps") or len(res.get("notes") or []) or 0)
    return max(1, int(res.get("step") or 1) - 1)


def _step_name(res: dict, step_no: int) -> str:
    """The name of that step, from the record when debug is carrying one."""
    for entry in (res.get("record") or []):
        if int(entry.get("step") or 0) == step_no:
            return str(entry.get("name") or "")
    for entry in (res.get("notes") or []):
        if int(entry.get("step") or 0) == step_no:
            return str(entry.get("name") or "")
    return ""


def _step_label(res: dict, step_no: int, lang: str) -> str:
    """That step's name as the person reads it: the bridge's, or its Japanese."""
    name = _step_name(res, step_no)
    return STEP_NAMES_JA.get(name, name) if lang == "ja" else name


def _short_error(text: str) -> str:
    """The part of a failure worth carrying: the first line, no URL boilerplate."""
    first = str(text).strip().splitlines()[0] if str(text).strip() else ""
    return first.split("For more information")[0].strip()[:200]
#: The same call, answered the same way. A tool that only looks cannot say
#: anything new while nothing has moved, and a model that has decided to
#: re-check will re-check: identical zoom_score calls by the dozen, minutes
#: spent re-reading one sentence, and the no-progress guard did not catch it
#: because that guard watches masks kept and written.
SAME_READ_REPEAT = 3
SAME_READ_ABORT = 8
#: Only the tools that look. A write may repeat -- what it writes about has
#: changed -- and the refusals guard covers the one write that loops.
READ_ONLY_TOOLS = frozenset({"annotation_status", "dataset_images", "image_get_b64",
                             "teacher_band", "zoom_plan", "zoom_score", "calibrate_sam",
                             "teacher_view", "mask_stats", "mask_get_b64"})
#: The tools that put a mask on an image: write_kept writes what accept_* kept,
#: spot_write what spot_detect found. What the loop keeps about a write -- the
#: steps before labelling, progress, the images a stop names, the image put
#: down, the next image named after it -- was keyed on write_kept by name, and
#: a run on a scattering of specks wrote several images with spot_write,
#: was stopped for standing still, and said 書き込めた画像: なし.
WRITE_TOOLS = frozenset({"write_kept", "spot_write"})
#: The tools that work on one image towards a write. An image this run has put
#: down is not worked again with any of them.
IMAGE_WORK_TOOLS = SHEET_TOOLS | {"image_get_b64", "spot_detect"}
#: How much of a tool's description the model is sent. What is past it is never
#: read: rehearse said how specks are scored at character 1,050, and
#: spot_detect's "another point, not another sensitivity" sat past 700, on the
#: run they were written for. Raising it for every tool costs about 2.8k tokens
#: a request in a 16k window, so what a model must do goes first instead.
TOOL_DESC_CHARS = 700


def read_repeats(last_read: dict, fn: str, fargs: dict, moves: int) -> int:
    """How many times in a row this exact question has been asked of a tool
    that only looks, with nothing moving in between.

    Counting the moves as part of the question is what separates a loop from a
    poll: `annotation_status` asked twice around a `write_kept` is two
    different questions, and asked twice around nothing is one asked twice.
    """
    if fn not in READ_ONLY_TOOLS:
        return 0
    sig = (fn, json.dumps(fargs, sort_keys=True, ensure_ascii=False, default=str), moves)
    if sig == last_read.get("sig"):
        last_read["n"] += 1
    else:
        last_read["sig"], last_read["n"], last_read["res"] = sig, 1, None
    return last_read["n"]
#: What the model is given per request. A turn's context is the system
#: prompt plus everything since, and the pictures are the weight in it.
NUM_CTX = 16384
#: Ollama counts tokens and the budget below counts characters: three to a
#: token is close enough for English and generous for Japanese, and the cost
#: of being wrong is one more trim, not a failure.
CHARS_PER_TOKEN = 3
#: What the chat template wraps the tool schemas in -- its instructions for
#: calling them -- in tokens, about.
TOOL_TEMPLATE_TOKENS = 100
#: What the tool schemas a run offers take of the window, in tokens: the
#: eighteen tools and ask_user came to about 19,300 characters, some 6,450
#: tokens, and the template's wrapping of them. A run whose schemas come out
#: longer takes less (see _history_budget).
TOOL_SCHEMA_TOKENS = 6_450 + TOOL_TEMPLATE_TOKENS
#: Roughly how many characters of conversation fit in the window -- the
#: system prompt, the messages and the pictures -- once the tool schemas and
#: the answer have theirs. Derived from NUM_CTX, so the two cannot drift apart.
#:
#: This is the mark that triggers a trim, and TRIM_DOWN_TO is where a trim
#: stops. Trimming to the trigger meant trimming on nearly every turn -- the
#: conversation sat at the limit and shed a message or two each time -- and
#: every one of those changed the front of the prompt, so llama.cpp threw its
#: cache away and re-read the whole prompt, many times slower a turn than
#: reading it from the cache. Cutting well below the mark buys many turns of an
#: unchanged prefix, which is what the cache needs.
HISTORY_BUDGET_CHARS = (NUM_CTX - ANSWER_TOKENS - TOOL_SCHEMA_TOKENS) * CHARS_PER_TOKEN
TRIM_DOWN_TO_CHARS = 14_000
#: Images already sent are the first thing to go: the model looked at that
#: picture, said where the objects were, and the boxes it produced are in the
#: text. Keeping the pixels of an image it has finished with costs a third of
#: the window for nothing.
KEEP_IMAGES = 2


#: What a picture costs a vision model in prompt tokens: one for each 32 px
#: square of it, and a few for the markers around it. A 512 px square was
#: measured at 269 tokens and a 1280 px one at 1,613, which is that exactly.
IMAGE_TOKEN_PX = 32
IMAGE_TOKEN_EXTRA = 13
#: What a picture whose size cannot be read is charged: the flat rate every
#: picture was charged before sizes were read. A real one always has a size.
UNSIZED_IMAGE_CHARS = 1_400
#: How much of a picture's base64 is decoded to find its size first: a JPEG
#: or a PNG says it within its first few hundred bytes.
_IMAGE_HEAD_B64 = 8_192


def _image_size(b64: str) -> tuple[int, int] | None:
    """A picture's width and height, read from its header, or None."""
    from PIL import Image

    for text in (b64[:_IMAGE_HEAD_B64], b64):
        try:
            with Image.open(io.BytesIO(base64.b64decode(text))) as im:
                return im.size
        except Exception:        # noqa: BLE001 - too short to say, or not a picture
            continue
    return None


def _image_chars(b64: object) -> int:
    """What one picture in the history costs, in the budget's characters.

    It was a flat 1,400 -- a 640 px copy -- after the copies had doubled: a
    1280x960 copy costs about 3,600 and a sheet of SAM's answers about 1,200.
    With the copies undercounted, a history could pass the window with
    nothing trimmed, and the model server cut the front of the prompt -- the
    brief, the language -- instead.
    """
    size = _image_size(b64) if isinstance(b64, str) else None
    if size is None:
        return UNSIZED_IMAGE_CHARS
    w, h = size
    tokens = math.ceil(w / IMAGE_TOKEN_PX) * math.ceil(h / IMAGE_TOKEN_PX) + IMAGE_TOKEN_EXTRA
    return tokens * CHARS_PER_TOKEN


def _msg_size(m: dict) -> int:
    n = len(str(m.get("content") or ""))
    for call in (m.get("tool_calls") or []):
        n += len(json.dumps(call, ensure_ascii=False))
    for img in (m.get("images") or []):
        n += _image_chars(img)
    return n


def _history_budget(tools: list) -> int:
    """HISTORY_BUDGET_CHARS, or less when this run's tool schemas take more of
    the window than the ones it was measured with."""
    left = ((NUM_CTX - ANSWER_TOKENS - TOOL_TEMPLATE_TOKENS) * CHARS_PER_TOKEN
            - len(json.dumps(tools, ensure_ascii=False)))
    return min(HISTORY_BUDGET_CHARS, left)


def _between_results(messages: list, i: int) -> bool:
    """Whether messages[i] is a user message set between one turn's tool results.

    A turn with several calls has a picture put after the result that fetched
    it and before the next result: assistant, tool, user (the picture), tool.
    A trim that stopped after the first result left the picture and the second
    result, and the call the second one answers was gone.
    """
    if messages[i].get("role") != "user":
        return False
    j = i + 1
    while j < len(messages) and messages[j].get("role") == "user":
        j += 1
    return j < len(messages) and messages[j].get("role") == "tool"


def trim_history(messages: list, budget: int = HISTORY_BUDGET_CHARS,
                 keep_images: int = KEEP_IMAGES,
                 down_to: int | None = None) -> tuple[int, int]:
    """Keep a conversation inside the window, in place. Returns (dropped, unimaged).

    Ollama silently truncates from the front when a request will not fit --
    "truncated = 1" in its log -- and what sits at the front is the system
    prompt: the recipe, the language, the instruction to ask before writing.
    A model that has quietly lost those is worse than one that has forgotten
    an image, so the trimming is done here and from the middle.

    The system prompt and the newest exchange always stay. Older images are
    dropped first (their boxes are in the text already), then whole messages
    from the oldest end, never leaving a tool result without the assistant
    turn that asked for it -- a picture set between one turn's results goes
    with them.

    Nothing happens until the conversation passes `budget`, and then it is cut
    to `down_to` -- well below -- so the next trim is many turns away. Trimming
    on every turn keeps the prompt's front moving, and a moving front is a cold
    cache: the whole prompt re-read every turn instead of served from it.
    """
    if not messages:
        return (0, 0)
    if sum(_msg_size(m) for m in messages) <= budget:
        return (0, 0)
    target = TRIM_DOWN_TO_CHARS if down_to is None else down_to
    unimaged = 0
    with_images = [i for i, m in enumerate(messages) if m.get("images")]
    for i in with_images[:-keep_images] if keep_images else with_images:
        messages[i] = {k: v for k, v in messages[i].items() if k != "images"}
        messages[i]["content"] = str(messages[i].get("content") or "") + " [画像は省略しました]"
        unimaged += 1
    # The system prompt is kept, and so is the instruction that started the
    # job. Trimming from index 1 ate that instruction on a long run, and the
    # conversation left -- system, assistant, tool, assistant, tool -- has no
    # user turn in it at all. Ollama answers that with "no user query found in
    # messages" and a 500, which ended a run with masks kept and nothing
    # written, every time the conversation grew long enough to trim.
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    start = 1 if first_user is None else first_user + 1
    dropped = 0
    while sum(_msg_size(m) for m in messages) > target:
        # The oldest droppable turn goes whole: an assistant turn with every
        # tool result that answers it, or results already left without one.
        # Taken a message at a time, and stopped with two messages left, a
        # trim could stop between the results of one turn: the one left first
        # in line had no call before it, which an OpenAI-style server refuses
        # with a 400.
        end = start + 1
        while end < len(messages) and (messages[end].get("role") == "tool"
                                       or _between_results(messages, end)):
            end += 1
        if len(messages) - end < 2:
            break       # what would be left is the newest exchange, and it stays
        del messages[start:end]
        dropped += end - start
    return (dropped, unimaged)


def _text(res):
    j = "\n".join(c.text for c in (res.content or []) if getattr(c, "text", None))
    try:
        return json.loads(j)
    except Exception:
        return j


def _backend_for(backend, model: str, ollama: str, base_url: str | None,
                 api_key_env: str | None):
    """A model server from what the caller passed, or the one already built."""
    if isinstance(backend, Backend):
        return backend
    name = backend or "ollama"
    if name == "ollama" and base_url is None:
        base_url = ollama          # the old argument, still honoured
    return make_backend(name, model=model, base_url=base_url,
                        api_key_env=api_key_env, num_ctx=NUM_CTX)



#: What a run is told when it stops with images still unlabelled. It is not a
#: reprimand and not a question: the model has already decided it is done, so
#: what it needs is the fact it was missing. It names one image rather than
#: listing ten: a run that was handed the list took the id at the front of it
#: again and again, because the list was where it looked for what to do next
#: and the frame it had just been told to put down was still on it. One name
#: is an instruction; ten are a menu.
MORE_TO_DO = {
    "ja": ("まだ未ラベルの画像が {n} 枚残っています。次は {nxt} です。"
           "これを終わらせてから次へ進んでください。"),
    "en": ("{n} images still have no mask. Do {nxt} next, and finish it "
           "before moving on."),
}
#: The count is known and no id is: the bridge caps the list it answers with,
#: and every id on it may be one this run has already put down.
MORE_TO_DO_UNNAMED = {
    "ja": "まだ未ラベルの画像が {n} 枚残っています。続けてください。",
    "en": "{n} images still have no mask. Carry on.",
}


#: Said after a write that is not shown back as a picture, in place of the
#: review. The review asks whether what was asked for is inside the bright part
#: and repairs by SAM's rungs: a question and a repair for what accept_* kept.
#: After spot_write neither fits. A speck can be 4x4 in the 1280 px copy and
#: 2x2 in the review, and on light specks the white edge that marks a mask is
#: the colour of the specks themselves: a ring on a speck and a ring on nothing
#: were the same white square. Told nothing at all, a run invented image ids,
#: and went back to the image it had just written again and again.
WROTE_UNPICTURED = {
    "ja": "{iid} に書きました（{n}）。{go}",
    "en": "Written on {iid} ({n}). {go}",
}


def _wrote_unpictured(lang: str, iid: str, res: dict, nxt: str | None, after: str = "") -> str:
    """A write that is not shown back, in words: what went on, what it was found
    with, and what the list has next -- or, when it names nothing, `after`."""
    ja = lang == "ja"
    if res.get("spots") is not None:
        n = f"点 {res['spots']} 個" if ja else f"{res['spots']} specks"
        # The example, in the argument's own name. A trim that fell after the
        # next image_get_b64 left this sentence and that picture as all there
        # was: a run of specks then took like_item_id from the one id this
        # sentence named -- the image just written -- time after time, was
        # refused each time, and read the teacher's id off the refusal.
        if res.get("like_item_id"):
            n += (f"、like_item_id={res['like_item_id']} を例に見つけたもの" if ja
                  else f", found with like_item_id={res['like_item_id']}")
    elif res.get("objects") is not None:
        n = f"物体 {res['objects']} 個" if ja else f"{res['objects']} objects"
    else:
        n = "数は答えにありません" if ja else "no count given"
    go = ((f"次の未ラベル画像は {nxt}（画像リストの順）です。" if ja else
           f"The next unlabelled image is {nxt} (the image list's order).") if nxt else after)
    return WROTE_UNPICTURED.get(lang, WROTE_UNPICTURED["en"]).format(iid=iid, n=n, go=go).strip()


def _redo_left(lang: str, redo: set) -> str:
    """The images a clear gave back to be redone and not yet redone.

    Said where "nothing is left" would be: once every image has a mask the
    list of what is left is empty, and a run that took the flags off its
    short images to redo them was told to report and finish after the first.
    """
    first, n = sorted(redo)[0], len(redo)
    if lang == "ja":
        return (f"要確認の印を外して作り直すことにした画像が {n} 枚、まだ作り直されていません。次は {first} です。"
                "作り直さずに残すなら、理由を添えて mark_review で印を付け直してください。")
    return (f"{n} image(s) you took the review flag off to redo are not redone yet: {first} next. "
            "If you leave one as it is, mark_review it again with the reason.")


def _run_record(lang: str, written: list, reviewed: set, redo=()) -> str:
    """What the loop kept of a run that finished, set under the model's report.

    The report is written from what the trims left of the conversation. After
    a long run it can say no image needed review and a few were written, of a
    run that had written many more and flagged some of them.
    """
    if not (written or reviewed or redo):
        return ""                          # a run that wrote nothing: nothing to set right
    ja = lang == "ja"
    flagged = sorted(reviewed)
    if ja:
        text = f"（ループの記録）書き込んだ画像: {len(written)} 枚。"
        text += (f"要確認の印: {len(flagged)} 枚（{'、'.join(flagged)}）。" if flagged else "要確認の印: なし。")
        if redo:
            text += f"印を外したまま作り直していない画像: {'、'.join(sorted(redo))}。"
        return text
    text = f"(The loop's record) Written: {len(written)} image(s). "
    text += (f"Flagged for review: {len(flagged)} ({', '.join(flagged)}). " if flagged
             else "Flagged for review: none. ")
    if redo:
        text += f"Review flag taken off and not redone: {', '.join(sorted(redo))}."
    return text.strip()


def _run_memory(lang: str, answers: list, rehearsal: str | None, steps_done: bool, written: list) -> str:
    """What this run has settled, put back after a trim: the person's answers, the steps, the rehearsal.

    A trim drops the oldest turns and, with them, what the person said and what
    the rehearsal scored. A long run trims many times: without this it asked
    the person the same questions again, and called rehearse over and over --
    on images the person never drew, too -- before writing images it had
    already been told how to do.
    """
    if not (answers or rehearsal or steps_done):
        return ""
    ja = lang == "ja"
    lines = []
    for q, a in answers[-5:]:
        lines.append(f"- 人への質問「{q[:120]}」への答え: 「{a[:200]}」" if ja else
                     f'- The person, asked "{q[:120]}", answered: "{a[:200]}"')
    if steps_done:
        lines.append("- 準備の段階はすべて済んでいます。" if ja else "- The setup steps are all done.")
    if rehearsal:
        lines.append(f"- {rehearsal}")
    if written:
        lines.append(f"- 書き込んだ画像: {len(written)} 枚。" if ja else f"- Written so far: {len(written)} image(s).")
    head = ("[この run でここまでに決まったこと。古いやり取りを整理したので、ここにまとめています]" if ja else
            "[What this run has settled so far, kept here because older turns were trimmed]")
    return head + "\n" + "\n".join(lines)


def _left_behind(lang: str, written: list, reviewed: set, redo=()) -> str:
    """What a stopped run leaves on the project, the images for a person first.

    A stop is not the model finishing, so the report the brief asks it for is
    never written and this sentence is all the person gets. A run could flag
    most of its writes, and its stop named none of them.
    """
    ja = lang == "ja"
    flagged = sorted(reviewed)
    head = ((f"要確認の印: {'、'.join(flagged)}。" if ja else f"Flagged for review: {', '.join(flagged)}. ")
            if flagged else "")
    # A flag taken off to redo the image, and the redo never made: said, or the
    # person never learns the image was short.
    if redo:
        head += (f"印を外したまま作り直していない画像: {'、'.join(sorted(redo))}。" if ja else
                 f"Review flag taken off and not redone: {', '.join(sorted(redo))}. ")
    return head + (f"書き込めた画像: {'、'.join(written) if written else 'なし'}。" if ja else
                   f"Written: {', '.join(written) if written else 'nothing'}.")


#: A file's extension on an image id. image_get_b64 takes an image by its id
#: as the lists give it or by its file name, and a model that fetched one by
#: its file name sends that name on to the tools that take the id alone.
_ID_EXT = re.compile(r"\.(png|jpe?g|bmp|tiff?|webp)$", re.IGNORECASE)


def _bare_ids(fargs: dict, seen) -> tuple[dict, dict[str, str]]:
    """fargs with an image's file name given as its id put back to the id, and what was changed.

    image_get_b64 takes the id as the image lists give it, or the image's file
    name, whatever format it is stored in; every other tool takes the id
    alone. A model carries a file name from the one into the other -- one run
    sent names like "img007.png" to accept_mask again and again, and each cost
    a refusal and a trip to the image list. Put back here rather than
    in the bridge because the loop keeps its books by the id it is sent: a write
    to "img007.png" is not matched to the picture fetched as img007, and its
    review is never drawn. A name this run fetched as an id of its own (seen)
    is left as it is.
    """
    renamed: dict[str, str] = {}

    def bare(v):
        if isinstance(v, str) and _ID_EXT.search(v) and v not in seen:
            b = _ID_EXT.sub("", v)
            if b:
                renamed[v] = b
                return b
        return v

    out = dict(fargs)
    for k in ("item_id", "like_item_id"):
        if k in out:
            out[k] = bare(out[k])
    if isinstance(out.get("item_ids"), list):
        out["item_ids"] = [bare(v) for v in out["item_ids"]]
    raw = out.get("item_ids_json")
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            ids = json.loads(raw)
        except ValueError:
            ids = None
        if isinstance(ids, list):
            fixed = [bare(v) for v in ids]
            if fixed != ids:
                out["item_ids_json"] = json.dumps(fixed, ensure_ascii=False)
    return (out if renamed else fargs), renamed


def _marked_ids(fargs: dict) -> list[str]:
    """The images a mark_review call is about, however it was addressed."""
    raw = fargs.get("item_ids_json") or fargs.get("item_ids") or fargs.get("item_id") or ""
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, str):
        return [raw] if raw else []
    return [str(x) for x in raw if x] if isinstance(raw, list) else []


def _named_ids(fargs: dict) -> list[str]:
    """Every image a call names by id: item_id, like_item_id, and a list of ids."""
    def named(v) -> bool:
        return isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) != ""

    out = [str(fargs[k]) for k in ("item_id", "like_item_id") if named(fargs.get(k))]
    for key in ("item_ids", "item_ids_json"):
        raw = fargs.get(key)
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = None
        if isinstance(raw, list):
            out += [str(v) for v in raw if named(v)]
    return list(dict.fromkeys(out))


async def _project_ids(c, project_id: str) -> tuple[set[str], bool] | None:
    """The project's image ids as dataset_images lists them, and whether that is all of them.

    None when the read failed, which is worth trying again. A bridge that
    answers without the lists gives no ids and no whole list, which is not.
    """
    try:
        res = _text(await c.call_tool("dataset_images", {"project_id": project_id}))
    except Exception:                                    # noqa: BLE001 - read again next time
        return None
    if isinstance(res, dict) and res.get("error"):
        return None
    lists = res.get("ids") if isinstance(res, dict) else None
    if not isinstance(lists, dict):
        return set(), False
    ids = {str(i) for key in ("done", "todo") for i in (lists.get(key) or []) if i is not None}
    try:
        whole = not res.get("not_listed") and len(ids) >= int(res.get("total") or 0)
    except (TypeError, ValueError):
        whole = False
    return ids, bool(whole)


async def _not_in_project(c, project_id: str, names: list[str], book: dict) -> list[str]:
    """The ids among names that the run's project has no image under.

    The model's ids are text it read or made up, and they go into the bridge's
    requests. A run is held to its project (see run); this holds it to that
    project's images, and answers an id that is not one of them with a short
    refusal the model can act on, before anything is sent.

    Checked against dataset_images, read when an id is first named and again
    whenever one is not on the list -- a person may have added images since.
    A read that failed refuses nothing and is tried again when the next call
    names an id. A list the bridge cut short, or could not give, refuses
    nothing: an id missing from it may still be there, and the bridge checks
    every id it puts into a URL itself.
    """
    if not project_id or not names:
        return []
    if "ids" not in book or (book["whole"] and any(n not in book["ids"] for n in names)):
        got = await _project_ids(c, project_id)
        if got is None:
            return []              # not read: nothing is refused, and the next id asks again
        book["ids"], book["whole"] = got
    if not book["whole"]:
        return []
    return [n for n in names if n not in book["ids"]]


def _carry_on(lang: str, left: int, ids: list[str]) -> str:
    """The sentence for a model that has stopped somewhere it should not."""
    if not left:
        return ("残りはありません。報告して終えてください。"
                if lang == "ja" else "nothing is left: report what you did and finish.")
    if not ids:
        return MORE_TO_DO_UNNAMED.get(lang, MORE_TO_DO_UNNAMED["en"]).format(n=left)
    return MORE_TO_DO.get(lang, MORE_TO_DO["en"]).format(n=left, nxt=ids[0])


def _why_put_down(lang: str, iid: str, flagged_for: str | None) -> str:
    """Why an image this run has put down is not worked again, as it happened.

    All of them used to be "given up on", in English whatever the language, and
    on a run that goes well most of what is put down is what the model flagged
    for a person itself. A run on a scattering of specks flagged its short
    images; going back to one, it was told it had been given up on earlier in
    this run -- which it had not -- wrote that into every one of those flags as
    the reason, and flagged and unflagged them until it was stopped. flagged_for is
    the reason the model flagged it with, for an image put down by that flag
    alone; None for one the run gave up on, left, or put down before a flag.
    """
    ja = lang == "ja"
    if flagged_for is None:
        return (f"{iid} はこの run で見切った画像です。" if ja else
                f"{iid} was given up on earlier in this run.")
    if ja:
        why = f"（理由: {flagged_for}）" if flagged_for else ""
        return (f"{iid} はこの run で要確認の印を付けて人に渡した画像です{why}。"
                "印が付いている間は人の手元にあるので、この run では触りません。"
                "作り直すべきだと思うなら、そう最後の報告に書いてください。"
                "自分で作り直すと決めたなら、先に mark_review（review=false）で印を外してください。"
                "印を外した画像は、またこの run で扱えます。")
    why = f" (the reason given: {flagged_for})" if flagged_for else ""
    return (f"{iid} was flagged for review in this run and handed to a person{why}. While it is "
            "flagged it is with them, and this run does not work on it. If you think it should be "
            "redone, say so in your report. If you have decided to redo it yourself, take the flag "
            "off first with mark_review (review=false); an image with its flag off is this run's "
            "to work on again.")


async def _after_flag(c, project_id: str, lang: str, now: list[str], again: list[str],
                      reopened: list[str], kept_down: list[str], clearing: bool,
                      done_with, redo=frozenset()) -> str:
    """What a mark_review is answered with: what the call changed, then what is left.

    The bridge answers {"status": "ok", "updated": n} whether the flag went on,
    was on already or came off, and with nothing left to label the loop added
    nothing to it. A run that had written all of its images, flagging some as
    it went, flagged those again, cleared them, flagged them again, and kept
    sending that same flag until a person stopped it: every answer read as a
    flag just set, and none said the job was over.
    """
    ja = lang == "ja"

    def named(ids: list[str]) -> str:
        more = len(ids) - 5
        return ("、" if ja else ", ").join(ids[:5]) + (
            (f" ほか {more} 枚" if ja else f" and {more} more") if more > 0 else "")

    left, ids = await _work_left(c, project_id, done_with)
    nxt = ((f"次の未ラベル画像は {ids[0]}（画像リストの順）です。" if ja else
            f"The next unlabelled image is {ids[0]} (the image list's order).") if ids else "")
    if clearing:
        head = f"{len(now)} 枚の要確認の印を外しました。" if ja else f"The review flag is off {len(now)} image(s). "
        if reopened:
            head += (f"{named(reopened)} は、この run が印を付けたことで手を離していた画像なので、"
                     "またこの run で扱えます。" if ja else
                     f"{named(reopened)}: put down only because this run flagged them, so they are "
                     "this run's to work on again. ")
        if kept_down:
            head += (f"{named(kept_down)} は、この run が別の理由（見切り、同じ呼び出しの繰り返し、"
                     "前のマスクへの書き戻し）で手を離していた画像なので、そのままです。"
                     if ja else
                     f"{named(kept_down)}: put down by this run for another reason -- given up on, the same "
                     "call again, or a mask it had before -- and still are. ")
        # A clear is a step towards a redo, not the end of the job: nothing is
        # said of finishing here, only which image comes next.
        return (head + (nxt or (_redo_left(lang, redo) if redo else ""))).strip()
    head = ((f"{named(again)} には、この run で既に要確認の印が付いています。" if ja else
             f"{named(again)}: already flagged for review earlier in this run. ") if again else "")
    return (head + (nxt or (_redo_left(lang, redo) if redo and not left else _carry_on(lang, left, ids)))).strip()


def _first_unlabelled(status, done_with=frozenset()) -> tuple[str, str] | None:
    """The first image annotation_status lists as having no mask, as (id, name),
    leaving out what this run has put down."""
    if not isinstance(status, dict):
        return None
    for it in status.get("unannotated_images") or []:
        if isinstance(it, dict) and it.get("id") and it.get("name") and it["id"] not in done_with:
            return str(it["id"]), str(it["name"])
    return None


async def _next_in_list(c, project_id: str, done_with=frozenset()) -> str | None:
    """The next image with no mask, in the image list's order, or None."""
    if not project_id:
        return None
    try:
        after = _first_unlabelled(_text(await c.call_tool(
            "annotation_status", {"project_id": project_id})), done_with)
    except Exception:                                    # noqa: BLE001
        return None
    return after[0] if after else None


async def _show_next(c, project_id: str, done_with=frozenset()) -> tuple[str, dict] | None:
    """The next image with no mask, fetched the way the model's own reads are.

    By the file name annotation_status gives, which image_get_b64 takes as it
    takes an id: the file is found whatever format the store keeps it in.

    For a model that keeps going back to an image this run has given up on.
    Told in words where to go, a run went back again and again, the
    answer naming the next image every time; a picture in front of it is
    harder to look past than a filename in a sentence.
    """
    if not project_id:
        return None
    try:
        nxt = _first_unlabelled(_text(await c.call_tool("annotation_status", {"project_id": project_id})),
                                done_with)
        if not nxt:
            return None
        pic = _text(await c.call_tool("image_get_b64", {
            "project_id": project_id, "filename": nxt[1],
            "max_side": MODEL_IMAGE_SIDE, "min_side": MODEL_IMAGE_SIDE}))
    except Exception:
        return None
    return (nxt[0], pic) if isinstance(pic, dict) and pic.get("image_base64") else None


async def _put_down(c, project_id: str, item_id: str, done_with: set, res: dict,
                    lang: str = "ja") -> dict:
    """Put an image down when the bridge says a write would only go back.

    write_kept refuses a mask that was on the image before and was written
    over since, and says to leave the image. Said in words, that was not
    heard: a run could go round accept, write and reset on one image
    without end, and every reset and keep counted as progress. So
    the image joins the ones this run has given up on, which are not shown
    again, and the next image with no mask is shown in its place. Nothing
    left to show: the refusal is returned as it came.
    """
    done_with.add(item_id)
    instead = await _show_next(c, project_id, done_with)
    if not instead:
        return res
    nxt, pic = instead
    return {**pic, "item_id": nxt, "instead_of": item_id, "written": False, "written_before": True,
            "why": (f"{item_id} に書こうとしたマスクは、前にこの画像にあって上書きされたものと同じです。"
                    f"書き戻しても同じ答えの間を行き来するだけなので、{item_id} は今のマスクのまま置きます。"
                    "今のマスクも対象でなければ、理由を添えて mark_review してください。"
                    f"代わりに、まだマスクのない次の画像 {nxt} を添付しました。この画像で続けてください。"
                    if lang == "ja" else
                    f"The mask for {item_id} was on it before and was written over since: putting "
                    f"it back only goes round between the same answers, so {item_id} is left as it "
                    "is. If what is on it now is not the object either, mark_review it with the "
                    f"reason. The next image with no mask, {nxt}, is attached instead: carry on "
                    "with this one.")}


def _way_of(fn: str, fargs: dict) -> str:
    """Which way of asking SAM a call was: box, points or outline."""
    if fn == "accept_mask":
        if fargs.get("outline_json"):
            return "outline"
        if fargs.get("points_json") and not fargs.get("box_json"):
            return "points"
    return "points" if fn == "accept_points" else "box"


def _tried_entry(fn: str, fargs: dict, res: dict) -> dict:
    """One attempt on an image, as the list of what was tried shows it."""
    if fn in WRITE_TOOLS:
        if res.get("written"):
            out = (f"written, {res.get('spots')} specks" if fn == "spot_write" else
                   f"written, {res.get('blobs')} blobs" + (" (in pieces)" if res.get("in_pieces") else ""))
        else:
            out = "not written: " + str(res.get("why") or "")[:80]
        return {"way": "write", "asked": "", "out": out}
    way = _way_of(fn, fargs)
    if way == "outline":
        asked = f"{len(json.loads(fargs['outline_json']) or [])} points" if fargs.get("outline_json") else ""
    elif way == "points":
        asked = " ".join(str(fargs.get("points_json") or "").split())[:60]
    else:
        asked = " ".join(str(fargs.get("box_json") or fargs.get("boxes_json") or "").split())[:60]
    if fargs.get("level"):
        asked += f" level={fargs['level']}"
    rows = res.get("results") if isinstance(res.get("results"), list) else [res]
    outs = []
    for r in rows[:3]:
        if not isinstance(r, dict):
            continue
        if r.get("accepted"):
            outs.append(f"{r.get('level')} {r.get('area_pct')}% kept")
        else:
            why = r.get("why")
            why = " ".join(why) if isinstance(why, list) else str(why or "")
            outs.append("refused: " + why[:60])
    return {"way": way, "asked": asked, "out": "; ".join(outs) or "nothing kept"}


_WAY_WORDS = {
    "ja": {"box": "箱", "points": "点", "outline": "輪郭", "write": "書き込み"},
    "en": {"box": "box", "points": "points", "outline": "outline", "write": "write"},
}
_OTHER_WAYS = {
    "ja": {"box": "物体全体を囲む箱（accept_masks）",
           "points": "物体そのものの上に端から端まで置いた点のグループ（accept_points）",
           "outline": "物体の縁をなぞった輪郭（accept_mask の outline_json）"},
    "en": {"box": "a box around the whole object (accept_masks)",
           "points": "points on the object itself, end to end, as one group (accept_points)",
           "outline": "an outline traced along its edge (accept_mask with outline_json)"},
}


def _other_way(item_id: str, tried: list, lang: str) -> str:
    """What was tried on an image, and the ways that were not.

    Said when the same call comes again: the same way answers the same, and a
    model going round has lost sight of what it already did. Which way to try
    is the model's to choose; this only puts the choice in front of it.
    """
    ja = lang == "ja"
    words = _WAY_WORDS["ja" if ja else "en"]
    lines = [f"{i}. {words.get(t['way'], t['way'])} {t['asked']} -> {t['out']}".replace("  ", " ")
             for i, t in enumerate(tried[-TRIED_SHOWN:], 1)]
    used = {t["way"] for t in tried}
    fresh = [w for w in ("box", "points", "outline") if w not in used]
    menu = [_OTHER_WAYS["ja" if ja else "en"][w] for w in fresh]
    menu.append("同じ指定で別の粒度（level=subpart / part / whole、SAM の答えの絵のパネル名）" if ja else
                "the same prompt at another rung (level=subpart / part / whole, the panel names in the "
                "picture of SAM's answers)")
    if ja:
        return (f"同じ呼び出しをもう一度送りました。同じやり方は同じ答えを返します。{item_id} でこれまでに"
                "試したこと:\n" + "\n".join(lines) + "\n違うやり方を選んでください: " + " / ".join(menu)
                + "。前の答えが対象そのものだったなら、それをもう一度保持して write_kept してください。"
                "どのやり方でも対象にならなければ、mark_review で理由を残して次の画像へ進んでください。")
    return (f"the same call came again, and the same way answers the same. What was tried on {item_id}:\n"
            + "\n".join(lines) + "\nChoose a different way: " + " / ".join(menu)
            + ". If an earlier answer was the object itself, keep it again and write_kept. If no way "
            "gives the object, mark_review with the reason and go to the next image.")


async def _work_left(c, project_id: str, done_with=frozenset()) -> tuple[int, list[str]]:
    """How many images still have no mask, and a few of their ids.

    The loop used to take a turn with no tool call as the job being finished,
    and return whatever came with it -- including nothing at all. One run wrote
    a few masks of a project's worth and ended on an empty string. Nothing had
    told it what was left: dataset_images answered tens of kilobytes, of which
    the conversation keeps four thousand characters, and the annotation_status
    reply that did fit was dropped by the trimmer a few steps later. So the
    count is read here, where it cannot be trimmed, at the one moment it
    decides anything.
    """
    if not project_id:
        return 0, []
    try:
        res = _text(await c.call_tool("annotation_status", {"project_id": project_id}))
    except Exception:
        return 0, []
    if not isinstance(res, dict):
        return 0, []
    left = int(res.get("without_mask") or 0)
    ids = [i.get("id") for i in (res.get("unannotated_images") or []) if isinstance(i, dict) and i.get("id")]
    # An image this run has put down is not work left. Counting it kept the
    # loop answering "carry on" over a frame nothing was going to label, and
    # the run reached NO_PROGRESS_STEPS instead of finishing with what it had.
    # The bridge caps its list, so an id past the cap is not subtracted: the
    # count errs high, which only means the run carries on a little longer.
    put_down = [i for i in ids if i in done_with]
    ids = [i for i in ids if i not in done_with]
    return max(0, left - len(put_down)), ids[:10]


async def run(instruction: str, *, model: str = "qwen3.8:27b", ollama: str = "http://127.0.0.1:11434",
              api: str = "http://127.0.0.1:8002", policy: str = "write", on_event=None,
              history: list | None = None, ask=None, confirm_each: bool = False,
              debug_steps: bool = False,
              lang: str = "ja",
              brief: bool = True, should_stop=None, should_pause=None,
              backend: str | Backend | None = None, base_url: str | None = None,
              api_key_env: str | None = None, max_steps: int = MAX_STEPS,
              project_id: str = "", project_name: str = "") -> str:
    """Drive the model through the bridge until it answers. Returns the answer.

    on_event, if given, receives one dict per happening: {"type": "tool",
    "step", "name", "args", "result", "think_s"}, {"type": "image", "item_id",
    "caption", "jpeg_b64"}, {"type": "question", "item_id", "text"} and
    {"type": "final", "text", "think_s"}. history, if given, is the running
    message list of a conversation and is extended in place, so a chat can
    continue. ask, if given, is an async callable (question) -> reply. The
    model always has an ask_user tool -- a run has to be able to say it does not
    understand the job, and the brief tells it to ask before the first image --
    and without ask the loop answers it itself: nobody is there to reply, so it
    decides and says in its report what it assumed. The moment worth asking at
    is before the job, when nobody is usually watching, so a caller's ask should
    answer itself after a while. confirm_each additionally tells it to ask about
    every image before writing.

    debug_steps holds the run after each of the steps before labelling, and
    shows what it understood at each. It is switched on HERE rather than by
    asking the model to switch it on: a run told to call a tool it had, in a
    sentence it was given, sent something else again and again instead. A
    mode the person chose is not a request to the model.
    lang picks the language the model writes to the person in ("ja", "en").
    should_stop, if given, is called before each turn and after each tool
    call; when it returns true the loop ends there. Asking the model to stop
    does not stop it -- it only reads messages between turns, and a running
    turn has no ear -- so an interface that offers a stop needs this.
    should_pause is checked at the same points and holds the loop while it
    returns true: the work stops between steps, nothing is lost, and a stop
    still gets through. Both are checked between steps rather than inside
    one, because a turn in flight cannot be interrupted -- the wait is at
    most one model turn.
    """
    async def gate() -> bool:
        """True when the caller wants out. Holds here while it wants a pause."""
        if should_stop is not None and should_stop():
            return True
        if should_pause is None:
            return False
        announced = False
        while should_pause():
            if should_stop is not None and should_stop():
                return True
            if not announced:
                emit({"type": "paused", "text": "一時停止しました" if lang == "ja" else "paused"})
                announced = True
            await asyncio.sleep(0.2)
        if announced:
            emit({"type": "resumed", "text": "再開します" if lang == "ja" else "resumed"})
        return False

    def emit(ev):
        if not on_event:
            return
        try:
            on_event(ev)
        except Exception as exc:                        # noqa: BLE001
            # run() has nothing above it that catches, so a handler that threw
            # used to end the run here -- and end it emitting nothing, which is
            # the one failure shape with no trace of itself anywhere.
            print(f"[agent] the event handler raised, carrying on: {exc!r}",
                  file=sys.stderr, flush=True)

    _load_fastmcp()
    transport = StdioTransport(command=sys.executable, args=[str(SRV), "--policy", policy, "--api", api],
                               env=_bridge_env())
    async with Client(transport) as c:
        async def page(name):
            res = await c.read_resource(f"segstudio://playbook/{name}")
            return "\n".join(getattr(r, "text", "") for r in res)
        if brief:
            playbook = BRIEF
        else:
            playbook = (await page("00_overview")) + "\n\n" + (await page("counting_from_teacher")) + "\n\n" + (await page("prompts"))
        mcp_tools = {t.name: t for t in await c.list_tools() if t.name in TOOLS + ["mask_get_b64"]}
        mcp_tools_for_model = {k: v for k, v in mcp_tools.items() if k in TOOLS}
        tools = [{"type": "function", "function": {"name": t.name, "description": (t.description or "")[:TOOL_DESC_CHARS],
                  "parameters": _schema_for_model(t.name, t.input_schema)}} for t in mcp_tools_for_model.values()]
        # Always offered: the brief asks for it before the first image, and a
        # tool the brief names that the list does not have was answered "no
        # such tool" and counted as a failure. Without ask, the loop answers it.
        tools.append(ASK_TOOL)
        history_budget = _history_budget(tools)
        # The first thing the model reads named one way of working, and not the
        # steps every write waits on; after a trim it is all that is left of the
        # job's shape. A run on an image of specks found spot_detect in the tool
        # list alone, and nothing it was told said how that way goes.
        system = ("You label images in a segmentation trainer through tools; follow the playbook below. "
                  + ("Call steps(project_id) first and do what it asks; nothing is written until its steps are done. "
                     if "steps" in mcp_tools_for_model else "Call teacher_band once. ")
                  + "Then one image at a time, in the image list's order: image_get_b64 to look at it; for every object "
                  "you see, accept_mask with its box in the copy's coordinates -- or, for a scattering of alike marks, "
                  "spot_detect with like_item_id set to an image the person drew; "
                  + ("then ask_user with what you kept and rejected, and only after a yes write it. " if confirm_each
                     else "then write it: write_kept after accept_mask, spot_write after spot_detect. ")
                  + "If a write says needs_review, look at what was found before you flag it: flag it, with what "
                  "you saw, only if it is wrong. "
                  "Finish with the images to look at, first, then what you did. "
                  + LANG_NOTE.get(lang, "")
                  # Which project, when the caller already knows. A screen does
                  # -- it is the one the person is looking at -- and without
                  # this every instruction had to start by naming it, which is
                  # a question the person should not be asked twice.
                  + (f"\n\nThe project is {project_id}"
                     + (f", which the person sees as \"{project_name}\"" if project_name else "")
                     + f". Pass project_id={project_id} to every tool; do not ask which project."
                     if project_id else "")
                  + "\n\n" + playbook)
        # Once a person says "do them all", asking again is not caution, it is
        # the same question per image. The model is told to stop, and this
        # answers if it asks anyway -- with what the person said when they
        # stopped the questions, not with a yes: a yes to a question with three
        # answers is none of them, and the loop is not the one to pick
        # (see _not_asked).
        no_more_questions = {"flag": False, "said": "", "to": ""}

        async def ask_gate(question: str, choices: list[str] | None = None) -> str:
            if no_more_questions["flag"]:
                # Said on the screen in place of the question, which is not
                # shown as one: nothing waits on it (see the ask_user call).
                q = " ".join(str(question).split())[:200]
                emit({"type": "auto", "text": (f"人に送らず、判断をモデルに任せた質問: {q}" if lang == "ja"
                                               else f"not put to the person, the model decides: {q}")})
                return _not_asked(lang, no_more_questions["said"], no_more_questions["to"],
                                  list(choices or []))
            if ask is None:
                # Nobody to put it to. Said on the screen, as a question the
                # loop answers itself is, and answered as one.
                q = " ".join(str(question).split())[:200]
                emit({"type": "auto", "text": (f"答える人がいないため、判断をモデルに任せた質問: {q}"
                                               if lang == "ja" else
                                               f"nobody to ask, the model decides: {q}")})
                return NOBODY_TO_ASK.get(lang, NOBODY_TO_ASK["en"])
            reply = await ask(question)
            if wants_no_more_questions(reply):
                no_more_questions.update(flag=True, said=str(reply), to=str(question))
                emit({"type": "auto", "text": "以降は確認せずに最後まで進めます" if lang == "ja"
                      else "carrying on without asking"})
            return reply

        if debug_steps and "steps" in mcp_tools:
            # Straight to the bridge, before the model has said anything. The
            # model is not asked to turn this on and cannot turn it off.
            try:
                await c.call_tool("steps", {"project_id": project_id, "debug": "step"})
                emit({"type": "auto", "text": ("ステップごとに止めて確認します"
                                               if lang == "ja" else
                                               "stopping after each step to check")})
            except Exception as exc:                    # noqa: BLE001
                emit({"type": "failed", "step": 0, "name": "steps",
                      "text": f"step-by-step could not be switched on: {exc!r}"[:300]})

        bk = _backend_for(backend, model, ollama, base_url, api_key_env)
        last_failure: dict = {"sig": None, "n": 0}
        last_read: dict = {"sig": None, "n": 0, "res": None}
        seen_size: dict[str, tuple] = {}   # item_id -> the size the model was shown
        refusals: dict[str, int] = {}      # item_id -> boxes refused in a row
        # The steps before labelling, counted down as the bridge reports them.
        # Starts at "all of them": a run that never calls steps never writes.
        steps_left = STEPS_BEFORE_LABELLING
        done_with: set[str] = set()        # item_ids this run has given up on, or handed to a person
        reviewed: set[str] = set()         # item_ids flagged for a person now
        # Every image this run has ever flagged, cleared or not. It is what
        # counts a flag as progress; reviewed says what is flagged at the moment
        # and is what a stop names. A run flagged its short images, cleared
        # them and flagged them again, and the flag after the clear counted
        # each as new, because the clear had taken them out of reviewed.
        ever_flagged: set[str] = set()
        # item_id -> the reason the model flagged it with, for an image put down
        # by that flag alone: what a refusal says of it, and what a clear gives
        # back. One put down before it was flagged is not here (see mark_review).
        handed_over: dict[str, str] = {}
        # Images a clear gave back to be redone, until they are written or
        # flagged again. Once every image has a mask nothing else counts them,
        # and the run is told to finish with them half done.
        redo: set[str] = set()
        stuck_on_item: dict[tuple[str, str], int] = {}   # (tool, image) -> failures in a row
        looked_at: dict[str, int] = {}     # item_id -> times its write has been reviewed
        kept_high: dict[str, int] = {}     # item_id -> the most it has ever held
        same_accept: dict[str, dict] = {}  # item_id -> each accept sent on it since its last write, and repeats
        # What the run stands on that a trim would take: the person's answers,
        # the last scored rehearsal, and whether the steps are done.
        answers: list[tuple[str, str]] = []
        rehearsal_said: str | None = None
        steps_finished = False
        tried: dict[str, list] = {}        # item_id -> what was tried on it, and what came back
        written: list[str] = []            # images this run has finished
        progress = Progress()
        images_seen: dict[str, str] = {}   # item_id -> image b64, for the renderings
        boxes_seen: dict[str, list] = {}   # item_id -> boxes passed to accept_mask
        # The project's image ids, read when a call first names one (_not_in_project).
        project_images: dict = {}
        # Pictures of SAM's answers from this turn's accept calls. They go to the
        # model beside the history, for its next reply only, and are filed in
        # the history as words once it has replied: a picture of candidates is
        # for the choice made right after it. Held here and not marked in the
        # messages, so nothing a model server has not been told about is sent,
        # and no trim counts them. One that is never shown -- no room, a retry,
        # the run ending first -- is filed as words all the same: every way out
        # of the loop below calls file_sheets before it returns.
        live_sheets: list[dict] = []
        messages = history if history is not None else []
        if not messages:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})

        def file_sheets(sheets: list, seen: bool) -> None:
            for s in list(sheets):
                messages.append({"role": "user", "content": _sheet_words(s, lang, seen)})
                live_sheets[:] = [x for x in live_sheets if x is not s]

        for step in range(max_steps):
            progress.step()
            if progress.stalled:
                file_sheets(live_sheets, False)
                stuck = max(refusals, key=refusals.get) if refusals else None
                text = ((f"{progress.limit} 手のあいだ何も進まなかったので止めました。"
                         + _left_behind(lang, written, reviewed, redo)
                         + (f" {stuck} で弾かれ続けています。" if stuck else ""))
                        if lang == "ja" else
                        (f"stopped: nothing kept or written for {progress.limit} steps. "
                         + _left_behind(lang, written, reviewed, redo)
                         + (f" Stuck on {stuck}." if stuck else "")))
                emit({"type": "final", "text": text, "think_s": 0})
                return text
            live = sum(_msg_size(s["msg"]) for s in live_sheets)
            used = sum(_msg_size(m) for m in messages) + live
            emit({"type": "context", "used": used, "budget": history_budget,
                  "pct": round(100 * used / history_budget)})
            # A picture of what spot_detect found is the one look a run of specks
            # takes at an image: one such run stopped opening copies early on,
            # and many of its spot_detect answers came with the history already
            # near the budget, where the picture and its words
            # do not fit beside it and would have gone as words alone. The trim the
            # budget would have called for a turn or two later is made now, down to
            # the same TRIM_DOWN_TO_CHARS; with no such picture waiting it is as before.
            dropped, unimaged = trim_history(messages, budget=history_budget - _spot_room(live_sheets, lang))
            if dropped or unimaged:
                emit({"type": "trimmed", "dropped": dropped, "unimaged": unimaged,
                      "text": (f"古いやり取りを整理しました（{dropped} 件を削除、画像 {unimaged} 枚を省略）"
                               if lang == "ja" else
                               f"trimmed the conversation ({dropped} messages, {unimaged} images)")})
            if dropped:
                # What the trim took that the run still stands on goes back in,
                # right after the instruction every trim keeps. Only when turns
                # were dropped: the front of the prompt moves then anyway, so the
                # cache loses nothing it had.
                note = _run_memory(lang, answers, rehearsal_said, steps_finished, written)
                first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
                if note and first_user is not None:
                    messages.insert(first_user + 1, {"role": "user", "content": note})
            if await gate():
                file_sheets(live_sheets, False)
                emit({"type": "stopped", "text": "中止しました" if lang == "ja" else "stopped"})
                return "stopped"
            # The pictures get the room the history leaves under the budget,
            # with the words each of them will be filed as set aside first.
            room = (history_budget - sum(_msg_size(m) for m in messages)
                    - sum(len(_sheet_words(s, lang, False)) for s in live_sheets))
            file_sheets(_sheets_without_room(live_sheets, room), False)
            t0 = time.time()
            try:
                reply = await asyncio.to_thread(bk.chat, messages + [s["msg"] for s in live_sheets], tools)
            except Exception as exc:
                # A model server that refuses one request has ended runs that
                # had masks kept and nothing written. Cut the conversation
                # back and ask once more; only then give up, and say what was
                # in hand when it happened.
                short = _short_error(str(exc))
                emit({"type": "trimmed", "dropped": 0, "unimaged": 0,
                      "text": (f"モデルが応答しませんでした（{short}）。やり取りを短くして、もう一度試します"
                               if lang == "ja" else
                               f"the model server refused that request ({short}); shortening and retrying")})
                # Asked again without the pictures: they are the first thing
                # to go, and their words are enough to carry on with.
                file_sheets(live_sheets, False)
                trim_history(messages, budget=TRIM_DOWN_TO_CHARS, down_to=TRIM_DOWN_TO_CHARS // 2)
                try:
                    reply = await asyncio.to_thread(bk.chat, messages, tools)
                except Exception as exc2:
                    text = ((f"モデルが二度応答しませんでした（{_short_error(str(exc2))}）。"
                             + _left_behind(lang, written, reviewed, redo)
                             + (f" {list(refusals)[0]} は保持したまま書けていません。" if refusals else ""))
                            if lang == "ja" else
                            f"the model server refused twice ({_short_error(str(exc2))}). "
                            + _left_behind(lang, written, reviewed, redo))
                    emit({"type": "final", "text": text, "think_s": round(time.time() - t0, 1)})
                    return text
            think = time.time() - t0
            # Replied to: each picture of SAM's answers is filed as its words.
            file_sheets(live_sheets, True)
            messages.append({"role": "assistant", "content": reply.text,
                             "tool_calls": reply.tool_calls})
            calls = reply.tool_calls
            if calls and (reply.text or "").strip():
                # What it said on the way to calling something. Only the last
                # turn's words were kept, and those are a summary written after
                # the fact; the reading that matters -- "the round part in the
                # middle", said of a picture it is about to box -- was thrown
                # away every turn but the last. Without it there is no way to
                # tell a mask that is wrong because the model saw the wrong
                # thing from one that is wrong because SAM answered the wrong
                # way, and those have different repairs.
                emit({"type": "say", "step": step, "text": reply.text[:2000],
                      "about": [_call_args(c).get("item_id") for c in calls
                                if _call_args(c).get("item_id")],
                      "think_s": round(think, 1)})
            if not calls:
                left, ids = await _work_left(c, project_id, done_with)
                if left or redo:
                    # Not finished: say what is still there and let it carry on.
                    # The no-progress guard is what ends a model that will not.
                    note = _carry_on(lang, left, ids) if left else _redo_left(lang, redo)
                    emit({"type": "trimmed", "dropped": 0, "unimaged": 0, "text": note})
                    messages.append({"role": "user", "content": note})
                    continue
                # The model's report is written from what the trims left of the
                # conversation, and after a long run it can say no image needs
                # review and only a few were written, of a run that had written
                # and flagged many more. What the loop kept goes under it.
                record = _run_record(lang, written, reviewed, redo)
                text = (reply.text or "").rstrip() + "\n\n" + record if record else reply.text
                emit({"type": "final", "text": text, "think_s": round(think, 1)})
                return text
            for tc in calls:
                fn = tc["name"]
                fargs, renamed = _bare_ids(_call_args(tc), images_seen)
                if project_id and fn != "ask_user" and fargs.get("project_id") != project_id:
                    # A run is this project's, whatever a call names. The
                    # bridge takes a project by its id or any part of its name,
                    # so a guess from the model -- or text it read somewhere --
                    # could write into a project nobody is watching.
                    fargs = {**fargs, "project_id": project_id}
                # Not the model's to pass: see MODEL_WITHHELD_ARGS.
                fargs, withheld = _without_withheld(fn, fargs)
                # Only images the run's project has (see _not_in_project).
                unknown = ([] if fn not in mcp_tools or (isinstance(tc, dict) and tc.get("malformed"))
                           else await _not_in_project(c, project_id, _named_ids(fargs), project_images))
                if isinstance(tc, dict) and tc.get("malformed"):
                    # Answered as a fault the model can read and repair, not sent
                    # on: arguments that were not an object ended the whole run
                    # where they were first read as one.
                    res = {"error": (f"the arguments of this {fn} call were not a JSON object "
                                     f"({tc['malformed']}): send them as one, "
                                     "{\"name\": value, ...}")}
                elif unknown:
                    res = {"error": (f"{', '.join(unknown[:5])}: this project has no image with that id. "
                                     "Take ids from dataset_images or annotation_status, as listed")}
                elif fn == "ask_user":
                    choices = fargs.get("choices_json") or fargs.get("choices") or []
                    if isinstance(choices, str):
                        try:
                            choices = json.loads(choices)
                        except ValueError:
                            choices = [c.strip() for c in choices.split("|") if c.strip()]
                    choices = [str(c)[:80] for c in choices][:5] if isinstance(choices, list) else []
                    # Known before it is sent: a reply that turns asking off is
                    # still the person's reply, and one the loop writes is not --
                    # nor is the one it writes when there is nobody to ask.
                    not_asked = no_more_questions["flag"] or ask is None
                    if not not_asked:
                        # Shown as a question only when it is one. One the loop
                        # answers itself went out with its three choices while
                        # nothing waited on it: agent_reply answers a reply to it
                        # with 409. ask_gate says it on the screen instead.
                        emit({"type": "question", "item_id": fargs.get("item_id"),
                              "text": str(fargs.get("question", "")), "choices": choices})
                        iid = fargs.get("item_id")
                        if iid in images_seen:
                            emit({"type": "image", "item_id": iid, "caption": f"{iid}: the boxes so far",
                                  "jpeg_b64": _render(images_seen[iid], boxes_seen.get(iid, []))})
                    reply = await ask_gate(str(fargs.get("question", "")), choices)
                    if not_asked:
                        # Nobody answered, so nothing moved. Counted as a person
                        # answering, a model that asks on after being told not to
                        # never meets the no-progress limit.
                        res = {"not_asked": reply}
                    else:
                        progress.moved("the person answered")
                        res = {"reply": reply}
                        answers.append((str(fargs.get("question", "")), str(reply)))
                elif (fn in IMAGE_WORK_TOOLS
                      and (fargs.get("item_id")
                           or str(fargs.get("filename") or "").rsplit(".", 1)[0]) in done_with):
                    # An image the run has given up on does not come back. The
                    # cutoff below already says "go to the next image", and a
                    # run answered that by fetching the same picture over and
                    # over until the repeated-read guard ended the whole job
                    # over a single frame. There is nothing to be gained by
                    # showing it again, so it is not shown.
                    iid = (fargs.get("item_id")
                           or str(fargs.get("filename") or "").rsplit(".", 1)[0])
                    left, ids = await _work_left(c, project_id, done_with)
                    # Said as what happened to it: a flagged image is with a
                    # person (see _why_put_down).
                    gone = _why_put_down(lang, iid, handed_over.get(iid))
                    res = {"item_id": iid, "finished": True, "why": gone,
                           "next": _carry_on(lang, left, ids)}
                    # Said in words, that was not heard: a run was given it
                    # again and again, the next image named each time, and
                    # went back every time. So the next image is shown rather
                    # than named -- attached the way any picture it asks for is.
                    instead = await _show_next(c, project_id, done_with)
                    if instead:
                        nxt, pic = instead
                        res = {**pic, "item_id": nxt, "instead_of": iid,
                               "why": gone + (f"代わりに、まだマスクのない次の画像 {nxt} を添付しました。"
                                              "この画像で続けてください。"
                                              if lang == "ja" else
                                              f" The next image with no mask, {nxt}, is attached instead: "
                                              "carry on with this one.")}
                elif fn in WRITE_TOOLS and steps_left and "steps" in mcp_tools_for_model:
                    # The steps before labelling are not advice. A run that skips
                    # them opens on pictures nobody has drawn, having looked at
                    # none of them and read nothing of what the person drew, and
                    # the first sign that it misunderstood the job is a project
                    # full of wrong masks. The bridge reports how far they have
                    # got and refuses nothing -- other clients drive those tools
                    # too -- so refusing is the harness's business, the harness
                    # being what skips them.
                    res = {"written": False, "item_id": fargs.get("item_id"),
                           "why": f"{steps_left} of the {STEPS_BEFORE_LABELLING} steps before "
                                  f"labelling are not done",
                           "next": "call steps(project_id), do what it asks, and call it "
                                   "again with what you found. If a step cannot be "
                                   "answered, ask_user rather than guessing. Nothing is "
                                   "written until they are done."}
                elif fn in mcp_tools:
                    # Not while the steps are open: the images then are the
                    # teachers, and one written on the model's behalf and given
                    # up on can never be rehearsed, so the steps never close.
                    stuck = (fn in ("accept_mask", "accept_masks")
                             and refusals.get(fargs.get("item_id"), 0) >= PROBE_REFUSALS_STOP
                             and not (steps_left and "steps" in mcp_tools_for_model))
                    if stuck:
                        # Not sent to the bridge at all: the band has said no
                        # this many times on this image, and it will not say
                        # anything else to a box a few pixels along. Telling the
                        # model to write and move on was not enough -- it went
                        # on asking, again and again, and a run ended with many
                        # masks kept and nothing written. So write them here, and
                        # tell it what was done.
                        iid = fargs.get("item_id")
                        n = refusals[iid]
                        try:
                            wrote = _text(await c.call_tool(
                                "write_kept", {"project_id": fargs.get("project_id"), "item_id": iid}))
                        except Exception as exc:
                            wrote = {"error": str(exc)[:200]}
                        res = {"accepted": False, "item_id": iid,
                               "why": f"refused here: {n} boxes in a row on this image were rejected",
                               "wrote_what_was_kept": wrote,
                               "next": ("この画像は打ち切って、残していた分を書き込みました。次の画像へ進んでください。"
                                        if lang == "ja" else
                                        "this image is done: what was kept has been written. "
                                        "Go to the next image.")}
                        # Giving up on an image is a decision about it, and a run
                        # that decides is not standing still. It used to count
                        # only if something got written, so an image that could
                        # not be labelled at all charged its whole cost to the
                        # no-progress limit: a string of calls on one hard image,
                        # correctly cut off, and the run died soon after with
                        # most of its images never opened.
                        done_with.add(iid)
                        progress.moved(f"gave up on {iid}")
                        if isinstance(wrote, dict) and wrote.get("written"):
                            refusals.pop(iid, None)
                            progress.moved(f"wrote {iid}")
                            if iid not in written:
                                written.append(iid)
                            reviewed.discard(iid)      # the save clears the trainer's flag
                            kept_high.pop(iid, None)   # and the bridge let go of what was kept
                        emit({"type": "trimmed", "dropped": 0, "unimaged": 0,
                              "text": ((f"{iid}: {n} 回続けて弾かれたので打ち切り、"
                                        + ("保持していた分を書き込みました" if isinstance(wrote, dict) and wrote.get("written")
                                           else "書き込むものはありませんでした"))
                                       if lang == "ja" else
                                       f"{iid}: stopped probing after {n} refusals; "
                                       + ("wrote what was kept" if isinstance(wrote, dict) and wrote.get("written")
                                          else "there was nothing to write"))})
                    else:
                        if fn == "image_get_b64" and not fargs.get("max_side"):
                            # A photograph off a camera can be tens of megabytes,
                            # a third more again as base64 in the request; one of
                            # those came back from the model server as a 400 and
                            # ended a run.
                            fargs = {**fargs, "max_side": MODEL_IMAGE_SIDE}
                        if fn == "image_get_b64" and not fargs.get("min_side"):
                            # max_side only ever shrinks. On a project whose
                            # pictures are already smaller than it, everything
                            # the brief says about looking closer is a no-op: a
                            # crop hands the model the same pixels with less
                            # around them, and max_side does nothing at all.
                            # Measured on small pictures with a person's masks
                            # to score against, asking for one point per object,
                            # the model found several times more of the objects
                            # at 1280 than at the pictures' own size. The
                            # picture gains no detail from being enlarged; the
                            # object simply spans enough of the encoder's
                            # patches to be seen. It is not free -- a 512 square
                            # costs 269 prompt tokens and a 1280 one 1613 -- and
                            # the history, charged each picture by its size,
                            # trims sooner.
                            fargs = {**fargs, "min_side": MODEL_IMAGE_SIDE}
                        if fn == "accept_mask" and fargs.get("item_id") in seen_size \
                                and not fargs.get("from_width"):
                            w, h = seen_size[fargs["item_id"]]
                            fargs = {**fargs, "from_width": w, "from_height": h}
                        # Asking a read tool the same thing while nothing has
                        # moved is a loop the model cannot see: its own earlier
                        # answer is the one it is about to receive again.
                        repeats = read_repeats(last_read, fn, fargs, progress.moves)
                        pushed_on = ""
                        if repeats >= SAME_READ_ABORT:
                            # A read that repeats is a loop over one question,
                            # not over the job. Ending the run here once cost
                            # the rest of a project, because one picture could not be
                            # labelled and the model would not put it down. If
                            # work is left, say what it is and let it carry on;
                            # NO_PROGRESS_STEPS is what ends a model that will
                            # not, and it says so in its final.
                            left, ids = await _work_left(c, project_id, done_with)
                            if not left:
                                file_sheets(live_sheets, False)
                                emit({"type": "stopped",
                                      "text": (f"同じ読み取り（{fn}）を同じ引数で {repeats} 回繰り返し、"
                                               "その間に何も進まなかったため中止しました。"
                                               if lang == "ja" else
                                               f"stopped: {fn} was called {repeats} times with the same "
                                               "arguments and nothing moved in between")})
                                return "stopped"
                            pushed_on = _carry_on(lang, left, ids)
                            emit({"type": "trimmed", "dropped": 0, "unimaged": 0,
                                  "text": pushed_on})
                            last_read["sig"], last_read["n"], last_read["res"] = None, 0, None
                        cached = last_read.get("res") if repeats >= SAME_READ_REPEAT else None
                        if cached is not None:
                            # Not sent to the bridge at all: it answered this
                            # exact question already and nothing has happened
                            # since, so the answer is the one it already has.
                            res = dict(cached)
                        else:
                            try:
                                res = _text(await c.call_tool(fn, fargs))
                            except Exception as exc:
                                res = {"error": str(exc)[:300]}
                            if repeats and isinstance(res, dict) and "error" not in res \
                                    and not res.get("image_base64"):
                                # An answer carrying a picture is not served
                                # from here: a trim may have taken the picture
                                # out of the history, and then asking again is
                                # the model repairing itself, not looping.
                                last_read["res"] = dict(res)
                        if pushed_on and isinstance(res, dict):
                            res["next"] = pushed_on
                        elif repeats >= SAME_READ_REPEAT and isinstance(res, dict):
                            res["next"] = (f"この呼び出しは同じ引数で {repeats} 回目で、答えは前回と同じです。"
                                           "同じものを送らないでください。引数を変えるか、次の画像へ進むか、"
                                           "今分かっていることを報告して終えてください。"
                                           if lang == "ja" else
                                           f"this is the {repeats}th time you have asked this with the same "
                                           "arguments, and the answer is the one you already have. Do not send "
                                           "it again: change the arguments, move on, or report what you know.")
                            if repeats == SAME_READ_REPEAT:
                                emit({"type": "trimmed", "dropped": 0, "unimaged": 0,
                                      "text": (f"{fn} を同じ引数で {repeats} 回呼んだので、前回の答えを返しました"
                                               if lang == "ja" else
                                               f"{fn} asked {repeats} times with the same arguments: "
                                               "answered from what it said before")})
                else:
                    res = {"error": f"no such tool {fn}"}
                # A repeated identical failure is a loop, and the loop is ours to
                # break: the model cannot see that it has asked this before.
                failed = isinstance(res, dict) and ("error" in res)
                sig = (fn, json.dumps(fargs, sort_keys=True, ensure_ascii=False)) if failed else None
                if sig and sig == last_failure["sig"]:
                    last_failure["n"] += 1
                else:
                    last_failure["sig"], last_failure["n"] = sig, 1 if sig else 0
                stuck_key = (fn, str(fargs.get("item_id") or fargs.get("filename") or ""))
                if failed:
                    stuck_on_item[stuck_key] = stuck_on_item.get(stuck_key, 0) + 1
                else:
                    stuck_on_item.pop(stuck_key, None)
                if failed:
                    res = {"error": _short_error(res.get("error", ""))}
                    # Until now a failure was an ordinary "tool" event, which the
                    # panel hides: a string of identical refusals in a minute or
                    # two looked, from the outside, like a run thinking quietly.
                    emit({"type": "failed", "step": step, "name": fn,
                          "text": res["error"], "n": last_failure["n"]})
                    over = stuck_on_item.get(stuck_key, 0)
                    if over >= STUCK_ON_ITEM_ABORT:
                        file_sheets(live_sheets, False)
                        emit({"type": "stopped",
                              "text": (f"{fn} \u304c {stuck_key[1]} \u3067 {over} \u56de\u9023\u7d9a\u3067\u5931\u6557"
                                       f"\u3057\u305f\u305f\u3081\u4e2d\u6b62\u3057\u307e\u3057\u305f: {res['error']}"
                                       if lang == "ja" else
                                       f"stopped: {fn} failed {over} times in a row on {stuck_key[1]}, "
                                       f"whatever the arguments: {res['error']}")})
                        return "stopped"
                    if over >= STUCK_ON_ITEM_WARN:
                        res["next"] = (
                            f"{fn} \u306f\u3053\u306e\u753b\u50cf\u3067 {over} \u56de\u9023\u7d9a\u3067\u5931\u6557"
                            "\u3057\u3066\u3044\u307e\u3059\u3002\u5f15\u6570\u3092\u5c11\u3057\u5909\u3048\u308b"
                            "\u306e\u3067\u306f\u306a\u304f\u3001\u4e0a\u306e\u7406\u7531\u3092\u8aad\u3093\u3067"
                            "\u304b\u3089\u76f4\u3057\u3066\u304f\u3060\u3055\u3044\u3002\u76f4\u305b\u306a\u3051"
                            "\u308c\u3070\u5225\u306e\u753b\u50cf\u3078\u9032\u3080\u304b ask_user \u3057\u3066"
                            "\u304f\u3060\u3055\u3044\u3002"
                            if lang == "ja" else
                            f"{fn} has now failed {over} times in a row on this image. Guessing another "
                            "value is not working: read the reason above and do what it says. If you "
                            "cannot, move to another image or ask_user.")
                    if last_failure["n"] >= SAME_FAILURE_ABORT:
                        file_sheets(live_sheets, False)
                        emit({"type": "stopped",
                              "text": (f"同じ呼び出し（{fn}）が {last_failure['n']} 回続けて失敗したため中止しました: "
                                       f"{res['error']}" if lang == "ja" else
                                       f"stopped: {fn} failed {last_failure['n']} times with the same arguments: {res['error']}")})
                        return "stopped"
                    if last_failure["n"] >= SAME_FAILURE_WARN:
                        res["next"] = (f"この呼び出しは同じ引数で {last_failure['n']} 回失敗しています。"
                                       "同じものを送らないでください。引数を変えるか、別の画像へ進むか、"
                                       "できないことを報告して終えてください。"
                                       if lang == "ja" else
                                       f"this exact call has failed {last_failure['n']} times. Do not send it again: "
                                       "change the arguments, move to another image, or report that it cannot be done.")
                if fn in SHEET_TOOLS and isinstance(res, dict):
                    # Never as text: base64 read as words is a hundred kilobytes
                    # of nothing. It is shown as a picture after the turn's results.
                    pic = res.pop(SHEET_KEY, None)
                    iid = str(fargs.get("item_id") or res.get("item_id") or "")
                    if fn == "spot_detect" and "error" not in res:
                        # Each spot_detect lets go of what the one before it staged,
                        # picture or no picture: an earlier picture of this image is
                        # not what a later spot_write writes -- and was claimed as
                        # written when the later call's drawing failed. One already
                        # written stays so: those rings are on the image.
                        for s in live_sheets:
                            if s["iid"] == iid and s.get("spots") and not s.get("superseded"):
                                s["superseded"] = True
                                if not s["written"]:
                                    s["msg"]["content"] += SPOT_SUPERSEDED.get(lang, SPOT_SUPERSEDED["en"])
                    if pic and "error" not in res:
                        spots = fn == "spot_detect"
                        what = _sheet_what(fn, fargs, res, lang)
                        summary = _spot_summary(res) if spots else _sheet_summary(res)
                        note = SPOT_NOTE if spots else SHEET_NOTE
                        live_sheets.append({"iid": iid, "what": what, "summary": summary, "written": False,
                                            "spots": spots,
                                            # What spot_detect found always has a choice after it --
                                            # write it, a region, a flag -- as a refused box does, so
                                            # its picture goes in first.
                                            "refused": (True if spots else
                                                        bool(res.get("refused")) if fn in ("accept_masks", "accept_points")
                                                        else not res.get("accepted")),
                                            "msg": {"role": "user", "images": [pic],
                                                    "content": note.get(lang, note["en"]).format(
                                                        iid=iid, what=what, asked=_asked_for(instruction))}})
                        emit({"type": "image", "item_id": iid,
                              "caption": (f"{iid}: what spot_detect found -- {summary}" if spots else
                                          f"{iid}: SAM's answers -- {summary}")[:300], "jpeg_b64": pic})
                        res = _sheet_marked(res)
                    if res.get("candidates_error"):
                        # Said to the person, as a review that could not be drawn
                        # is: a run choosing rungs without the picture should not
                        # look like one that was shown it. The model reads it too.
                        emit({"type": "failed", "step": step, "name": "candidates",
                              "text": f"{iid}: {res['candidates_error']}"[:300]})
                if (fn in WRITE_TOOLS and isinstance(res, dict) and res.get("unchanged")
                        and str(fargs.get("item_id") or "") in redo):
                    # The redo came out as the mask already there. The flag it
                    # was taken off for is gone, and nothing says the image was
                    # short any more unless it is put back.
                    rid = str(fargs.get("item_id"))
                    res = {**res, "next": (
                        f"{rid} は作り直すために要確認の印を外しましたが、作り直しても同じマスクでした。"
                        "理由を添えて mark_review で印を付け直すか、別のやり方で作り直してください。"
                        if lang == "ja" else
                        f"{rid}: you took its review flag off to redo it, and the redo came out the same. "
                        "mark_review it again with the reason, or redo it another way.")}
                if fn in WRITE_TOOLS and isinstance(res, dict) and fargs.get("item_id") and "error" not in res:
                    tried.setdefault(str(fargs["item_id"]), []).append(_tried_entry(fn, fargs, res))
                    # A write moves the image on: the same accept after it is a
                    # new question about a new state, not the same one again.
                    same_accept.pop(str(fargs["item_id"]), None)
                    if fn == "write_kept" and any(res.get(k) for k in (
                            "written", "unchanged", "written_before", "already_there")):
                        # The bridge lets go of what was kept whenever it answers
                        # one of these, so the next keep on this image counts from
                        # one again. Held at the old high here, a keep after a
                        # write read as no growth, and no growth is counted as a
                        # refusal: enough of them and the image is given up on
                        # over masks that were being kept.
                        kept_high.pop(str(fargs["item_id"]), None)
                if fn in WRITE_TOOLS and isinstance(res, dict) and res.get("written_before"):
                    # The bridge says to leave the image; the loop is what makes
                    # that stick (see _put_down). Putting it down is a decision
                    # about it, counted once, as giving up on an image is.
                    gone = str(fargs.get("item_id") or "")
                    if gone and gone not in done_with:
                        progress.moved(f"put down {gone}")
                        emit({"type": "trimmed", "dropped": 0, "unimaged": 0,
                              "text": (f"{gone}: 前に上書きされたのと同じマスクへ戻そうとしたので、"
                                       "今のマスクのまま置いて次の画像へ進みます"
                                       if lang == "ja" else
                                       f"{gone}: the write would only put back a mask it had "
                                       "before; left as it is, on to the next image")})
                    if gone:
                        # Down for the write it had before, not for a flag: a
                        # clear must not give it back (see handed_over).
                        handed_over.pop(gone, None)
                        was_redo = gone in redo
                        redo.discard(gone)
                        res = await _put_down(c, project_id or str(fargs.get("project_id") or ""),
                                              gone, done_with, res, lang)
                        if was_redo:
                            res = {**res, "flag_again": (
                                f"{gone} は作り直すために要確認の印を外した画像です。印を付け直してください（mark_review）。"
                                if lang == "ja" else
                                f"{gone} is one you took the review flag off to redo: mark_review it again.")}
                accept_repeat = 0
                if (fn in ("accept_mask", "accept_masks", "accept_points") and isinstance(res, dict)
                        and "error" not in res
                        # The loop's own answer for an image it has put down: its
                        # picture of the next image, or its carry-on, is the answer.
                        # "Try another way on this image" beside it sent the model
                        # back to the image it had just been moved off.
                        and not (res.get("instead_of") or res.get("finished"))):
                    aid = str(fargs.get("item_id") or "")
                    sig = (fn, json.dumps(fargs, sort_keys=True, ensure_ascii=False))
                    # Counted per call since the image's last write, not only
                    # against the call just before: a model can alternate
                    # level=part and level=whole on one box again and again,
                    # each answered the same, and a guard that saw only "not
                    # the same as last time" counted none.
                    sent = same_accept.setdefault(aid, {})
                    accept_repeat = sent.get(sig, -1) + 1
                    sent[sig] = accept_repeat
                    if not accept_repeat:
                        tried.setdefault(aid, []).append(_tried_entry(fn, fargs, res))
                    if (accept_repeat >= SAME_ACCEPT_PUT_DOWN and aid and aid not in done_with
                            # Not while the steps are open: the image then is a
                            # teacher, and one put down cannot be rehearsed, so
                            # the steps -- and every write -- would never open.
                            and not (steps_left and "steps" in mcp_tools_for_model)):
                        why = (f"同じ呼び出しを {accept_repeat + 1} 回送り、毎回同じ答えでした"
                               if lang == "ja" else
                               f"the same call was sent {accept_repeat + 1} times and answered the same")
                        try:
                            flag = _text(await c.call_tool("mark_review", {
                                "project_id": fargs.get("project_id"),
                                "item_ids_json": json.dumps([aid], ensure_ascii=False),
                                "reason": why, "review": True}))
                        except Exception:                        # noqa: BLE001
                            flag = None
                        # Said flagged only when it was: the bridge holds back a
                        # person's mask, and a stop that named it among the
                        # flagged would send the person to a flag that is not there.
                        flagged = (isinstance(flag, dict) and bool(flag.get("updated"))
                                   and aid not in (flag.get("not_flagged") or flag.get("teachers") or []))
                        done_with.add(aid)
                        if flagged:
                            reviewed.add(aid)
                            # Not handed_over: it was put down for the same call
                            # sent again, and a clear that gave it back would be
                            # answered by the same call and the same put-down,
                            # counted as progress every other step.
                            ever_flagged.add(aid)
                        progress.moved(f"put down {aid}")
                        left_ja = "今のマスクのまま要確認の印を付けて" if flagged else "今のマスクのまま置いて"
                        done_ja = "今のマスクのまま要確認の印を付けました" if flagged else "今のマスクのまま置きました"
                        left_en = "left as it is, flagged for review" if flagged else "left as it is"
                        emit({"type": "trimmed", "dropped": 0, "unimaged": 0,
                              "text": (f"{aid}: {why}。{left_ja}次の画像へ進みます"
                                       if lang == "ja" else
                                       f"{aid}: {why}; {left_en}, on to the next image")})
                        instead = await _show_next(c, project_id or str(fargs.get("project_id") or ""),
                                                   done_with)
                        if instead:
                            nxt, pic = instead
                            res = {**pic, "item_id": nxt, "instead_of": aid, "why": (
                                f"{aid} には{why}。{aid} は{done_ja}。"
                                f"代わりに、まだマスクのない次の画像 {nxt} を添付しました。この画像で続けてください。"
                                if lang == "ja" else
                                f"{aid}: {why}. It is {left_en}. The next image "
                                f"with no mask, {nxt}, is attached instead: carry on with this one.")}
                        else:
                            res = {**res, "next": (
                                f"{aid} には{why}。{done_ja}。報告して終えてください。"
                                if lang == "ja" else
                                f"{aid}: {why}. It is {left_en}; report and finish.")}
                    elif accept_repeat >= SAME_ACCEPT_OTHER_WAY:
                        res = {**res, "next": _other_way(aid, tried.get(aid) or [], lang)}
                        emit({"type": "auto",
                              "text": (f"{aid}: 同じ呼び出しが続いたので、試したことの一覧と別のやり方を見せました"
                                       if lang == "ja" else
                                       f"{aid}: the same call again; shown what was tried and the other ways")})
                images = []
                review_shot: str | None = None
                last_look: tuple | None = None      # (image, writes) once past REVIEW_REDOS
                if isinstance(res, dict) and res.get("image_base64"):
                    images = [res["image_base64"]]
                    # a picture shown in place of the one asked for is filed under its own id
                    iid = (str(res["item_id"]) if res.get("instead_of") else
                           str(fargs.get("filename", "") or res.get("item_id", "")).rsplit(".", 1)[0])
                    images_seen[iid] = res["image_base64"]
                    if res.get("width") and res.get("height"):
                        # What the model is looking at, so its boxes can be put
                        # back where they belong on the real picture.
                        seen_size[iid] = (int(res["width"]), int(res["height"]))
                    boxes_seen.setdefault(iid, [])
                    emit({"type": "image", "item_id": iid, "caption": iid, "jpeg_b64": _render(res["image_base64"], [])})
                if (fn in ("accept_mask", "accept_masks", "accept_points") and isinstance(res, dict)
                        and not res.get("instead_of")):
                    iid = fargs.get("item_id")
                    if fargs.get("reset"):
                        boxes_seen[iid] = []
                        kept_high.pop(iid, None)
                    # accept_masks answers for a batch: "accepted" is how many
                    # of them were kept, not whether one was. Matching only the
                    # singular name left every batch counted as no progress,
                    # and a run that was keeping masks was stopped at the
                    # no-progress limit with nothing written.
                    took = kept_count(res)
                    grew = kept_growth(res, kept_high, iid)
                    if grew and accept_repeat:
                        pass                     # the same keep again is not progress
                    elif grew:
                        refusals.pop(iid, None)
                        progress.moved(f"kept {grew} on {iid}")
                    elif "error" not in res:
                        refusals[iid] = refusals.get(iid, 0) + max(1, int(res.get("refused") or 1))
                    if took and fargs.get("box_json"):
                        try:
                            boxes_seen.setdefault(iid, []).append([float(v) for v in json.loads(fargs["box_json"])])
                        except Exception:
                            pass
                    elif took and fargs.get("boxes_json"):
                        try:
                            for b in json.loads(fargs["boxes_json"]):
                                boxes_seen.setdefault(iid, []).append([float(v) for v in b])
                        except Exception:
                            pass
                if fn == "mark_review" and isinstance(res, dict) and not res.get("error") and res.get("updated"):
                    clearing = str(fargs.get("review", True)).strip().lower() in ("false", "0", "no")
                    held = set(res.get("not_flagged") or res.get("teachers") or [])
                    now = [m for m in _marked_ids(fargs) if m not in held]
                    # Read before anything moves: flagged by this run, and not
                    # cleared or written over since.
                    again = [m for m in now if m in reviewed]
                    reopened: list[str] = []
                    if clearing:
                        # A flag taken off is not a flag: it was counted as one,
                        # and a stop named it among the images to look at. An
                        # image put down by its flag alone is no longer with a
                        # person, so it is the run's again: left put down, the
                        # cleared images were refused as "given up on", which they
                        # never were. One put down before it was flagged -- stuck,
                        # the same accept again, a write it had before -- stays
                        # down: a clear is not how that guard is undone.
                        for marked in now:
                            reviewed.discard(marked)
                            if handed_over.pop(marked, None) is not None:
                                done_with.discard(marked)
                                reopened.append(marked)
                                redo.add(marked)
                    else:
                        # Counted once per image and run, however often it is
                        # cleared in between (see ever_flagged). Nor by a flag
                        # that flagged nothing -- the bridge holds back a
                        # person's mask.
                        for marked in now:
                            if marked not in done_with or marked in handed_over:
                                handed_over[marked] = " ".join(str(fargs.get("reason") or "").split())[:160]
                            reviewed.add(marked)
                            redo.discard(marked)
                            if marked not in ever_flagged:
                                ever_flagged.add(marked)
                                progress.moved(f"flagged {marked}")
                            # Handed to a person: not worked again this run. An
                            # image flagged and never written -- a clean screen --
                            # stayed first in the list of what is left, and every
                            # write named it as the next image.
                            done_with.add(marked)
                    kept_down = [m for m in now if clearing and m in done_with]
                    res = {**res, "next": await _after_flag(
                        c, project_id or str(fargs.get("project_id") or ""), lang,
                        now, again, reopened, kept_down, clearing, done_with, redo)}
                if fn in WRITE_TOOLS and isinstance(res, dict) and res.get("written"):
                    iid = fargs.get("item_id")
                    if fn == "write_kept":
                        # KEPT is what went onto the image only when write_kept
                        # wrote it: after spot_write, a picture of SAM's answers
                        # marked written would name a mask that is not there.
                        for s in live_sheets:
                            if s["iid"] == iid and not s["written"] and not s.get("spots"):
                                s["written"] = True
                                s["msg"]["content"] += SHEET_WRITTEN.get(lang, SHEET_WRITTEN["en"])
                    else:
                        # The rings are what went on only when spot_write wrote
                        # them, and only the last spot_detect's: each one lets go
                        # of what the one before it staged, so an earlier picture
                        # marked written would name specks that are not there.
                        for s in [s for s in live_sheets if s["iid"] == iid and s.get("spots")
                                  and not s.get("superseded")][-1:]:
                            if not s["written"]:
                                s["written"] = True
                                s["msg"]["content"] += SPOT_WRITTEN.get(lang, SPOT_WRITTEN["en"])
                    # "written" is a new mask on the image: both write tools answer
                    # unchanged or written_before rather than put back one the
                    # image has, or had. The count of kept masks that decided it
                    # before had nothing to count for spot_write, which keeps
                    # nothing, and called a redo after reset=true -- the redo the
                    # review asks for -- nothing new whenever it kept as many.
                    refusals.pop(iid, None)
                    progress.moved(f"wrote {iid}")
                    if iid not in written:
                        written.append(iid)
                    # A save takes the image out of review on the trainer, so it
                    # is no longer one this run can say it flagged -- or one it
                    # handed to a person: left in handed_over, a later visit was
                    # told the image was flagged and to clear a flag not there.
                    reviewed.discard(iid)
                    redo.discard(iid)
                    if handed_over.pop(iid, None) is not None:
                        done_with.discard(iid)
                    looked_at[iid] = looked_at.get(iid, 0) + 1
                    skip = None
                    if fn == "write_kept":
                        # Every write is shown back, past the cap too: "written" is a
                        # mask this image never had (write_kept answers unchanged or
                        # written_before otherwise). Past the cap it was hidden and
                        # called "the same result" -- once spot_write was counted as a
                        # write, every write counted -- and on an object that came
                        # out in pieces the last of the different masks it wrote was
                        # the one that took in the floor under it, and was flagged
                        # for it.
                        skip = _no_review_because(True, iid, images_seen)
                        if skip is None:
                            try:
                                mres = _text(await c.call_tool("mask_get_b64", {"project_id": fargs.get("project_id"), "item_id": iid}))
                                # At the copy's own size, ruler and all. Shrunk to
                                # 640 px it was half the copy the model boxes on, and
                                # a box read off it for the redo it asks for landed
                                # half size in the top left: objects on one
                                # image were written as the empty floor beside
                                # them, and looked at, and let stand.
                                shot = _render(images_seen[iid], [], mres.get("mask_png_base64"), max_side=0)
                                emit({"type": "image", "item_id": iid,
                                      "caption": f"{iid}: written ({res.get('objects')} objects)",
                                      "jpeg_b64": shot})
                                review_shot = shot
                            except Exception as exc:                # noqa: BLE001
                                skip = f"the picture could not be made: {exc!r}"
                        if looked_at[iid] > REVIEW_REDOS:
                            # What the cap does: no redo on offer, and the image is
                            # not worked again this run. A flag still reaches it.
                            last_look = (iid, looked_at[iid])
                            done_with.add(iid)
                    if last_look:
                        res = {**res, "next": (
                            f"{iid} は {looked_at[iid]} 回書きました。これ以上は作り直しません。"
                            "言われているものがマスクに入っていなければ mark_review で理由を残し、"
                            "入っていれば次の未ラベル画像へ進んでください。"
                            if lang == "ja" else
                            f"{iid} has been written {looked_at[iid]} times and is not redone again. If "
                            "the thing you were asked for is not inside the mask, mark_review it with the "
                            "reason; otherwise go on to the next unlabelled image.")}
                    elif review_shot is None:
                        # No picture of this write: spot_write's are never shown
                        # back (WROTE_UNPICTURED says why), and a review that could
                        # not be drawn has none. What went on, and which image the
                        # list has next, go in words.
                        pid = project_id or str(fargs.get("project_id") or "")
                        nxt = await _next_in_list(c, pid, done_with)
                        after = ""
                        if not nxt:
                            # Nothing named was nothing said. One run's
                            # last write was answered with its count and no more; it
                            # went through mask_stats, the steps and the teachers,
                            # lost its own writes to a trim on the way, and asked the
                            # person what to do about "drafts made earlier" that were
                            # its own. What is left is said instead, as it is to a
                            # model that stops.
                            left, ids = await _work_left(c, pid, done_with)
                            after = _redo_left(lang, redo) if redo and not left else _carry_on(lang, left, ids)
                        res = {**res, "next": _wrote_unpictured(lang, iid, res, nxt, after)}
                    if skip:
                        # A mask written and never looked at is the worst of the
                        # failures to keep quiet about, and this was the quietest:
                        # masks were written run after run and the review picture
                        # was produced for none of them, with nothing anywhere
                        # saying so.
                        emit({"type": "failed", "step": step, "name": "review",
                              "text": (f"{iid}: written without being checked -- {skip}")})
                    boxes_seen[iid] = []
                if fn == "rehearse" and isinstance(res, dict) and not failed:
                    emit({"type": "rehearsal", "step": step, "item_id": res.get("item_id"),
                          "ok": bool(res.get("ok")), "text": _rehearsal_line(res, lang)})
                    if res.get("scored"):
                        rehearsal_said = _rehearsal_line(res, lang)
                if fn == "steps" and isinstance(res, dict):
                    if res.get("done"):
                        steps_left = 0
                        steps_finished = True
                    elif res.get("of"):
                        steps_left = int(res["of"]) - int(res.get("step", 1)) + 1
                    if res.get("accepted") and fargs.get("said"):
                        # What it understood, where its other words are. The
                        # bridge has kept these from the day the steps existed;
                        # they were inside a tool result, which this log folds
                        # away and then cuts at two hundred characters.
                        done_no = _step_just_answered(res)
                        # A step passed is a step achieved: it is checked against
                        # tools that ran. Not counted, a run of specks -- none of
                        # whose calls before labelling keep a mask -- spends its
                        # no-progress allowance getting to the first write.
                        progress.moved(f"step {done_no}")
                        emit({"type": "step", "step": step, "step_no": done_no,
                              "of": int(res.get("of") or res.get("steps") or 0),
                              "name": _step_label(res, done_no, lang),
                              "text": str(fargs.get("said"))[:600]})
                        if res.get("stop_each"):
                            # The bridge asked for the run to be held. It cannot
                            # hold it itself -- it answers a call and returns --
                            # and it cannot reach the person either.
                            held_back = no_more_questions["flag"] or ask is None
                            reply = await ask_gate(
                                f"STEP {done_no}/{res.get('of') or res.get('steps')} "
                                 f"{_step_label(res, done_no, lang)}\n{fargs.get('said')}\n\n"
                                 "ここまでで合っていますか。続けるなら「進めて」、"
                                 "違うなら何が違うかを書いてください。"
                                 if lang == "ja" else
                                 f"STEP {done_no}/{res.get('of') or res.get('steps')} "
                                 f"{_step_label(res, done_no, lang)}\n{fargs.get('said')}\n\n"
                                 "Is that right so far? Say carry on, or say what is wrong.")
                            # What the loop wrote is not what the person said, and
                            # is not cut: _not_asked bounds it where it is made.
                            res = ({**res, "not_asked": reply} if held_back
                                   else {**res, "person_said": str(reply)[:400]})
                # The answer as it stands now. It was taken before the blocks
                # above added to it, so the "that is its answer" after
                # REVIEW_REDOS writes, and what a person said at a held step,
                # were built and never sent.
                if withheld and isinstance(res, dict):
                    # Said, so a model that sent it knows it was not used.
                    res = {**res, "not_passed": (
                        f"{'、'.join(withheld)} はこの run では使えません。人が描いたマスクを置き換えるのは人です"
                        if lang == "ja" else
                        f"{', '.join(withheld)} is not available to this run: a mask a person drew is "
                        "replaced only by a person")}
                if renamed and isinstance(res, dict):
                    # Said in the answer, so the next call sends the id as it is.
                    res = {**res, "item_id_read_as": (
                        "、".join(f"「{k}」を「{v}」として読みました" for k, v in renamed.items())
                        + "。item_id は画像の ID で、拡張子は付けません"
                        if lang == "ja" else
                        "; ".join(f'"{k}" was read as "{v}"' for k, v in renamed.items())
                        + ": item_id is the image's id, without the file extension")}
                shown = res
                if isinstance(res, dict) and res.get("image_base64"):
                    shown = {k: v for k, v in res.items() if k != "image_base64"} | {"image": "attached in the next message"}
                text = json.dumps(shown, ensure_ascii=False)
                emit({"type": "tool", "step": step, "name": fn, "args": fargs, "result": text[:600], "think_s": round(think, 1)})
                messages.append({"role": "tool", "content": text[:4000], "tool_name": fn,
                                 "tool_call_id": tc.get("id")})
                if await gate():
                    file_sheets(live_sheets, False)
                    emit({"type": "stopped", "text": "中止しました" if lang == "ja" else "stopped"})
                    return "stopped"
                if images:
                    messages.append({"role": "user", "content": "The image from the last image_get_b64 call; give boxes in its pixel coordinates.",
                                     "images": images})
                if review_shot:
                    # The picture the person is shown, shown to the model as well,
                    # and the question asked in the person's own words rather than
                    # in ours. "Is the object you are labelling inside it" makes the
                    # model supply the object from whatever it currently believes;
                    # the instruction is the one statement of the job that cannot
                    # have drifted, and it is in scope here. Whether what the person
                    # asked for is in the mask is a question it can answer -- and answering it costs
                    # no per-project threshold, which is what every rule of the form
                    # "refuse anything whose colour or area is unlike the teachers"
                    # does cost. Asked of marks whose answers were known, it
                    # called every wrong one wrong and most of the right ones
                    # right: the errors fall on the cautious side, which
                    # is the side to be wrong on when the answer is "look again".
                    # The next image named, in the order of the image list. Told
                    # only "the next unlabelled image", a run went from img010
                    # to img011 by the look of the names, past what the list
                    # -- the order the person sees -- had next.
                    after = await _next_in_list(c, project_id, done_with)
                    go_ja = f"次の未ラベル画像 {after}（画像リストの順）へ" if after else "次の未ラベル画像へ"
                    go_en = (f"the next unlabelled image, {after} (the image list's order)" if after
                             else "the next unlabelled image")
                    if not after:
                        # The last image: "go to the next unlabelled image" when
                        # there is none sent the run looking for one.
                        left, _ids = await _work_left(c, project_id, done_with)
                        if redo and not left:
                            first = sorted(redo)[0]
                            go_ja = f"作り直すために印を外した {first} へ"
                            go_en = f"{first}, which you took the review flag off to redo"
                        elif not left:
                            go_ja = "最後の報告へ"
                            go_en = "your final report -- nothing else is left"
                    content = (
                        "いま書いたマスクの外側を暗くし、マスクの縁に白い線を引いています。"
                        "明るく残っているところが、いま書いたマスクです。"
                        f"あなたが受けた指示は「{_asked_for(instruction)}」です。"
                        "そこで言われているものは、明るく残っているところの中にありますか。"
                        "中にあって、明るいところがほぼそれだけなら、この画像はこれで完了です。"
                        f"この画像にはもう何も呼ばず、{go_ja}進んでください。"
                        "SAM の答えの絵で同じに見える別の粒度に替えても、良くはなりません。"
                        "中心だけ、あるいは周りまで入っているなら、"
                        "箱は合っていて粒度が違うだけです。"
                        "この画像の箱をすべて accept_masks で送り直してください。level ごとに呼び出しを分け、"
                        "最初の呼び出しに reset=true を付けます。粒度が違った箱には別の SAM の答えの名前"
                        "（accept の答えの levels にある名前）を、ほかの箱にはいま保持した名前を level にします。"
                        "その答えにも SAM の答えの絵が付きます。KEPT が対象そのものになっていれば、"
                        "write_kept でもう一度書いてください。対象自体がいなければ mark_review で理由を残してください。"
                        if lang == "ja" else
                        f"The mask just written is the part left bright: the rest of the picture "
                        f"is dimmed and the mask's edge is drawn in white. You were asked: "
                        f"\"{_asked_for(instruction)}\". Is the thing it names inside the bright part? "
                        "If it is, and the bright part is that and little else, this image is done: "
                        f"call nothing more on it and go to {go_en} -- another rung "
                        "that looks the same in the picture of SAM's answers is not a better answer. "
                        "If it took the middle of it only, or took it and what it sits on, the box was "
                        "right and the rung was not: send every box of this image again with accept_masks, "
                        "one call per level, the first with reset=true -- for the boxes that took the wrong "
                        "rung, the name of another of SAM's answers (the levels in the accept reply); for "
                        "the rest, the level they kept. That reply comes with the picture of SAM's answers "
                        "again; when KEPT is the object itself, write_kept again. If the object is not "
                        "there at all, mark_review with the reason.")
                    if last_look:
                        # Past REVIEW_REDOS: the mask is shown with no redo on offer.
                        lid, times = last_look
                        content = (
                            "いま書いたマスクの外側を暗くし、マスクの縁に白い線を引いています。"
                            "明るく残っているところが、いま書いたマスクです。"
                            f"あなたが受けた指示は「{_asked_for(instruction)}」です。"
                            f"{lid} は {times} 回書いたので、これ以上は作り直しません。"
                            "言われているものが明るいところに入っていなければ、見たことを添えて mark_review。"
                            f"入っていれば、この画像にはもう何も呼ばず、{go_ja}進んでください。"
                            if lang == "ja" else
                            "The mask just written is the part left bright: the rest of the picture is "
                            f"dimmed and the mask's edge is drawn in white. You were asked: "
                            f"\"{_asked_for(instruction)}\". {lid} has been written {times} times and is not "
                            "redone again. If the thing it names is not inside the bright part, mark_review "
                            f"it with what you saw; if it is, call nothing more on it and go to {go_en}.")
                    messages.append({"role": "user", "images": [review_shot], "content": content})
                    review_shot = None
        file_sheets(live_sheets, False)
        stuck = max(refusals, key=refusals.get) if refusals else None
        if lang == "ja":
            text = f"手数の上限（{max_steps}）に達しました。" + _left_behind(lang, written, reviewed, redo)
            if stuck:
                text += f" {stuck} で {refusals[stuck]} 回続けて弾かれ、そこで止まっています。"
            text += " 続けるなら、そのまま「続き」と指示してください。"
        else:
            text = "reached the step limit. " + _left_behind(lang, written, reviewed, redo)
            if stuck:
                text += f" Stuck on {stuck}, refused {refusals[stuck]} times in a row."
        emit({"type": "final", "text": text, "think_s": 0})
        return text
