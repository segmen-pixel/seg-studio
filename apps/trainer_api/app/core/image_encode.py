# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The only place that produces bytes for a project's ``images/`` directory.

Once ``images/`` holds the single copy of the pixels, whatever writes into it
decides what every later stage gets to see. There are five import routes
(upload, ZIP import, video frames, resize-clone, synthetic generation) and they
used to encode independently; the JPEG chroma knowledge in particular lived in
exactly one function and one test, so any sixth route would have been 4:2:0
without anyone noticing.

The measurement behind that: at quality 95, a one-pixel colour-only line loses
its contrast 16.25 -> 5.21 under libjpeg's default 4:2:0, versus 16.25 -> 16.21
under 4:4:4. A defect whose signal is a tint rather than a brightness step is
attenuated about threefold, and no reported score can reveal it, because
training and evaluation read the same degraded copy.

So ``subsampling`` is not a parameter here, and not a setting anywhere. It is a
literal, in one call, verified against the bytes that were actually produced. A
value that cannot be configured cannot be configured wrongly.

The JPEG encoder is PIL, never ``cv2.imencode``. OpenCV's
``IMWRITE_JPEG_SAMPLING_FACTOR`` needs 4.7+ and can be ignored outright
depending on which libjpeg the wheel was built against, and a silently ignored
flag here is an unobservable failure: train and eval both read the degraded
copy, so the score never moves.

Pixels are stored the way a viewer shows them. A phone or camera held upright
writes the sensor's rows and records the turn in the EXIF orientation tag;
browsers and OpenCV apply the tag, PIL does not. Re-encoding through PIL used
to drop it, so a portrait photo landed on its side and a mask drawn on the
upright photo -- what every other tool produces -- missed it. Kept as a tag
(raw), the size recorded for the image was the sensor's, the other way round
from what the browser draws. So the turn is applied here, once, and the tag is
never kept.
"""
from __future__ import annotations

import io
import logging
from pathlib import PurePath

from PIL import Image, ImageOps, JpegImagePlugin

_logger = logging.getLogger(__name__)

#: What a project may be configured to store.
#:
#: ``raw``  keep the uploaded bytes (see RAW_PASSTHROUGH_CONTAINERS); a photo
#:          its EXIF orientation turns is stored upright as PNG instead.
#: ``png``  re-encode losslessly to PNG. The default.
#: ``jpg``  re-encode to JPEG q95 4:4:4. Irreversible.
IMAGE_FORMATS = ("raw", "png", "jpg")

#: The format a project gets when nothing says otherwise, and the one an
#: unrecognised value falls back to.
#:
#: Never ``jpg``. dataset_prep made this call already, for the copies: a typo
#: in a format name must not quietly turn every image lossy. It matters more
#: here, because here the lossy copy is the only copy.
DEFAULT_IMAGE_FORMAT = "png"

#: Quality for the JPEG branch. Not a per-project setting: the reason to pick
#: jpg is disk, and a quality dial invites tuning it down until the defects go.
JPEG_QUALITY = 95

#: Containers that pass through untouched in ``raw`` mode.
#:
#: Limited to what a browser draws in an <img>, because the annotator serves
#: images/ straight off disk and lets the browser decode. A 16-bit or CMYK TIFF
#: kept verbatim would simply fail to open, and the only reason that does not
#: happen today is the server-side normalisation this module replaces.
RAW_PASSTHROUGH_CONTAINERS = ("png", "jpeg", "webp")

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png", ".png"),
    (b"\xff\xd8\xff", "jpeg", ".jpg"),
    (b"GIF87a", "gif", ".gif"),
    (b"GIF89a", "gif", ".gif"),
    (b"BM", "bmp", ".bmp"),
    (b"II*\x00", "tiff", ".tif"),
    (b"MM\x00*", "tiff", ".tif"),
)


class EncodeError(RuntimeError):
    """The bytes could not be turned into something safe to store."""


def normalize_format(value: object) -> str:
    """*value* as one of IMAGE_FORMATS, falling back to the safe one."""
    text = str(value or "").strip().lower()
    if text in IMAGE_FORMATS:
        return text
    if text:
        _logger.warning("unknown image format %r; using %r", value, DEFAULT_IMAGE_FORMAT)
    return DEFAULT_IMAGE_FORMAT


def sniff_container(data: bytes) -> str:
    """The real container of *data* by magic bytes, or "" when unrecognised.

    Filenames are not evidence. An installation can hold hundreds of files
    named ``.png`` that hold JPEG bytes, because the upload route decided
    PNG-ness from the suffix and stored the payload verbatim. Anything that reasons about
    format -- the chroma census, for one -- has to open the file, not read
    its name.
    """
    head = data[:16]
    for magic, name, _ext in _MAGIC:
        if head.startswith(magic):
            return name
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return ""


def jpeg_sampling(data: bytes) -> int | None:
    """Chroma subsampling of *data*: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0.

    -1 is PIL's answer for a single-component (greyscale) JPEG. None means the
    bytes are not JPEG at all, which is a different question from "JPEG with
    unknown sampling" and must not be conflated with it by the caller.
    """
    if sniff_container(data) != "jpeg":
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return JpegImagePlugin.get_sampling(im)
    except Exception:
        return None


def inspect_file(path) -> tuple[str, int | None]:
    """The container of the file at *path*, and its chroma if it is JPEG.

    Reads the header, not the pixels. The census this exists for can cover
    tens of thousands of files and tens of gigabytes, and the answer
    is in the first few hundred bytes of each: PIL parses the JPEG frame
    header during open(), so no decode is needed to learn the sampling.

    Returns ``("", None)`` for anything unreadable, because a census is a
    survey -- one unreadable file must not end it.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(32)
    except OSError:
        return "", None
    container = sniff_container(head)
    if container != "jpeg":
        return container, None
    try:
        with Image.open(path) as im:
            return container, JpegImagePlugin.get_sampling(im)
    except Exception:
        return container, None


#: EXIF orientation: 1 is "as stored", 2-8 a mirror and/or a quarter turn.
_ORIENTATION_TAG = 0x0112


def _orientation(im: Image.Image) -> int:
    """*im*'s EXIF orientation, 1 when it has none or an unusable one."""
    try:
        value = im.getexif().get(_ORIENTATION_TAG, 1)
    except Exception:
        return 1
    return value if isinstance(value, int) and 2 <= value <= 8 else 1


def _upright(im: Image.Image) -> Image.Image:
    """*im* turned the way a viewer shows it; *im* itself when it needs no turn."""
    if _orientation(im) == 1:
        return im
    return ImageOps.exif_transpose(im)


def _png_has_exif(data: bytes) -> bool:
    """Whether a PNG carries an eXIf chunk, from the chunk headers alone.

    Most PNGs have none, and finding that out must not cost a decode: an
    existing PNG is otherwise kept without ever being opened.
    """
    pos = 8
    while pos + 8 <= len(data):
        kind = data[pos + 4:pos + 8]
        if kind == b"eXIf":
            return True
        if kind == b"IEND":
            return False
        pos += 12 + int.from_bytes(data[pos:pos + 4], "big")
    return False


def _open_rgb_ready(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception as err:
        raise EncodeError(f"not a decodable image: {err}") from err


def _encode_png(data: bytes) -> bytes:
    """*data* as PNG, verbatim when it already is one that needs no turn.

    Re-encoding a PNG is lossless but not free, and it rewrites the pixels'
    provenance for no gain -- ``images/`` is supposed to be the original.
    """
    if sniff_container(data) == "png":
        if not _png_has_exif(data):
            return data
        with _open_rgb_ready(data) as im:
            if _orientation(im) == 1:
                return data
    with _open_rgb_ready(data) as im:
        # Alpha survives here. dataset_prep drops it when it builds the
        # training copy, which is the right place for that: this file is the
        # user's image, not the trainer's input.
        out = io.BytesIO()
        _upright(im).save(out, format="PNG", compress_level=1)
    return out.getvalue()


def _encode_jpeg(data: bytes) -> bytes:
    """*data* as JPEG q95 4:4:4, verified.

    The verification is not belt-and-braces. It converts a library or build
    regression -- a PIL that stops honouring ``subsampling``, a wheel with a
    different libjpeg -- from an invisible, permanent quality loss into a
    failure on the very first image imported.
    """
    with _open_rgb_ready(data) as im:
        rgb = _upright(im).convert("RGB")
        out = io.BytesIO()
        rgb.save(out, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=False)
    encoded = out.getvalue()
    sampling = jpeg_sampling(encoded)
    if sampling != 0:
        raise EncodeError(
            "JPEG encoder produced chroma subsampling "
            f"{sampling!r} instead of 0 (4:4:4); refusing to store the image"
        )
    return encoded


def encode_for_store(data: bytes, *, source_name: str, image_format: str) -> tuple[str, bytes]:
    """The suffix and bytes to write into ``images/`` for an uploaded file.

    The suffix comes from what was actually encoded, never from *source_name*.
    Naming a JPEG ``.png`` is how the existing mislabelled files got there, and
    a stem-to-file map cannot repair a file whose name lies about its contents.

    *source_name* is used only to decide the passthrough suffix in raw mode,
    and even then only after the container has been confirmed by magic bytes.

    Every branch stores the picture upright (see the module docstring), raw
    included: a photo its EXIF orientation turns cannot be kept as it came,
    so it becomes PNG, lossless from the decoded pixels.
    """
    fmt = normalize_format(image_format)
    container = sniff_container(data)

    if fmt == "jpg":
        return ".jpg", _encode_jpeg(data)

    if fmt == "raw" and container in RAW_PASSTHROUGH_CONTAINERS:
        # Confirm the browser can actually decode it before we commit to
        # keeping bytes we never parsed.
        with _open_rgb_ready(data) as im:
            turned = _orientation(im) != 1
        if not turned:
            suffix = PurePath(source_name or "").suffix.lower()
            canonical = {"png": (".png",), "jpeg": (".jpg", ".jpeg"), "webp": (".webp",)}[container]
            return (suffix if suffix in canonical else canonical[0]), data

    # png, raw's non-renderable containers (tif/bmp/gif), which become PNG so
    # the annotator can display them at all, and raw's turned photos.
    return ".png", _encode_png(data)
