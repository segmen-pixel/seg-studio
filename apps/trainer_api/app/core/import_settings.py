# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Which format a project stores its imported images in, and who decides.

Two levels, and the relationship between them is the whole point:

* ``runtime_settings.json`` holds a **default**, edited in Settings. It is the
  value a *new* project is stamped with.
* ``project.json`` holds the project's own ``image_store``. This is the
  effective value, always.

The default is copied, never referenced. If effective format were read from
the global setting, an administrator changing it would retroactively change
what 100 existing projects store, and the next upload into each of them would
be written in the new format -- irreversibly, for ``jpg``. Freezing does not
help there, because freezing only rejects explicit per-project changes.

That design only holds if every existing project actually carries the stamp,
which is what the layout-4 migration is for. Until a project has been stamped,
this module falls back to the global default and then to PNG, so the
un-migrated case is defined rather than accidental.
"""
from __future__ import annotations

import json
import logging

from fastapi import HTTPException

from segcore.dataset_layout import CANONICAL_IMAGE_EXTS

from .image_encode import (
    DEFAULT_IMAGE_FORMAT,
    IMAGE_FORMATS,
    JPEG_QUALITY,
    normalize_format,
)
from .paths import (
    IMAGE_STORE_KEY,
    IMAGE_STORE_SCHEMA,
    get_project_lock,
    project_dir,
    write_json,
)
from .torch_device import merge_runtime_settings, read_runtime_settings

_logger = logging.getLogger(__name__)

#: Key in runtime_settings.json.
DEFAULT_FORMAT_KEY = "default_image_format"

#: Re-exported: project.json's schema belongs to paths, which writes it, but
#: callers of this module think in terms of the image store.
__all__ = [
    "IMAGE_STORE_KEY",
    "IMAGE_STORE_SCHEMA",
    "assert_format_changeable",
    "effective_image_format",
    "effective_image_store",
    "has_stored_images",
    "image_store_for_new_project",
    "is_frozen",
    "read_default_image_format",
    "read_image_store",
    "save_default_image_format",
    "save_image_store",
]


def read_default_image_format() -> str:
    """The global default, normalised."""
    return normalize_format(read_runtime_settings().get(DEFAULT_FORMAT_KEY))


def save_default_image_format(value: object) -> str:
    """Persist the global default (merge-safe) and return what was stored.

    Changing this affects projects created afterwards. It deliberately does not
    touch a single existing project.
    """
    fmt = normalize_format(value)
    merge_runtime_settings({DEFAULT_FORMAT_KEY: fmt})
    return fmt


def _read_project_json(project_id: str) -> dict:
    path = project_dir(project_id) / "project.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


#: Rebuilt on every read; everything else in the block is carried through.
_OWNED_KEYS = ("schema", "format", "jpeg_quality", "locked")

#: Where the chroma census lands. It is a measurement of the bytes, so it
#: belongs next to the format rather than in a file of its own -- and it
#: must survive a format change, which is why the coercion below preserves
#: unknown keys instead of rebuilding the block from scratch.
MEASURED_KEY = "measured"


def _coerce_store(raw: object, *, fallback_format: str) -> dict:
    block = raw if isinstance(raw, dict) else {}
    carried = {k: v for k, v in block.items() if k not in _OWNED_KEYS}
    return {
        **carried,
        "schema": IMAGE_STORE_SCHEMA,
        "format": normalize_format(block.get("format", fallback_format)),
        "jpeg_quality": JPEG_QUALITY,
        "locked": bool(block.get("locked", False)),
    }


def image_store_for_new_project(*, locked: bool = False) -> dict:
    """The ``image_store`` block to stamp on a project being created."""
    return _coerce_store({"locked": locked}, fallback_format=read_default_image_format())


def read_image_store(project_id: str) -> dict | None:
    """The project's own block, or None when it has never been stamped."""
    raw = _read_project_json(project_id).get(IMAGE_STORE_KEY)
    if not isinstance(raw, dict):
        return None
    return _coerce_store(raw, fallback_format=DEFAULT_IMAGE_FORMAT)


def effective_image_store(project_id: str) -> dict:
    """The block in force for *project_id*. Always a dict.

    Resolution order: the project's stamp, then the global default, then PNG.
    The second step only ever applies to a project that predates the layout-4
    migration; once stamped, the global default stops mattering to it.
    """
    stored = read_image_store(project_id)
    if stored is not None:
        return stored
    return _coerce_store({}, fallback_format=read_default_image_format())


def effective_image_format(project_id: str) -> str:
    """Shorthand for the format alone."""
    return effective_image_store(project_id)["format"]


def save_image_store(project_id: str, value: object) -> dict:
    """Persist the project's own format. Rejects an unknown name outright.

    normalize_format() falls back to png so that a stale config file cannot
    stop a project loading. An explicit request is different: silently
    storing png when the caller asked for something else would report
    success for a setting that did not take.
    """
    text = str(value or "").strip().lower()
    if text not in IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown image format {value!r}; expected one of {list(IMAGE_FORMATS)}",
        )
    assert_format_changeable(project_id)
    path = project_dir(project_id) / "project.json"
    with get_project_lock(project_id):
        data = _read_project_json(project_id)
        block = data.get(IMAGE_STORE_KEY)
        # Merged, not replaced: replacing it dropped the census, and the
        # colour gate would then read an unmeasured project as clean.
        block = dict(block) if isinstance(block, dict) else {}
        block.update({
            "schema": IMAGE_STORE_SCHEMA,
            "format": text,
            "jpeg_quality": JPEG_QUALITY,
            "locked": has_stored_images(project_id),
        })
        data[IMAGE_STORE_KEY] = block
        write_json(path, data)
    return effective_image_store(project_id)


def read_measured_census(project_id: str) -> dict | None:
    """The stored chroma census, or None when nobody has measured yet.

    Absence is a distinct answer from a clean census and callers must
    treat it as one: the whole reason for measuring is that a project can
    look clean in every declaration it carries.
    """
    block = effective_image_store(project_id).get(MEASURED_KEY)
    return block if isinstance(block, dict) else None


def save_measured_census(project_id: str, census: dict) -> dict:
    """Store *census* on the project, leaving the rest of the block alone."""
    path = project_dir(project_id) / "project.json"
    with get_project_lock(project_id):
        data = _read_project_json(project_id)
        block = data.get(IMAGE_STORE_KEY)
        block = dict(block) if isinstance(block, dict) else {}
        block[MEASURED_KEY] = census
        data[IMAGE_STORE_KEY] = block
        write_json(path, data)
    return census


def has_stored_images(project_id: str) -> bool:
    """Whether ``images/`` already holds at least one image.

    Uses the canonical extension tuple rather than a private list: the
    project-listing code had its own four-entry version, so a project holding
    only ``.webp`` or ``.tiff`` counted as empty -- and an empty project is
    exactly the one whose format may still be changed.
    """
    images = project_dir(project_id) / "images"
    if not images.is_dir():
        return False
    exts = set(CANONICAL_IMAGE_EXTS)
    with_suffix = (p for p in images.iterdir() if p.is_file())
    return any(p.suffix.lower() in exts for p in with_suffix)


def is_frozen(project_id: str) -> bool:
    """Whether the format may no longer be changed.

    Frozen once pixels exist, regardless of the stored ``locked`` flag: the
    flag records intent at stamp time, the directory records the fact. A mixed
    ``images/`` destabilises statistics, stem resolution and reproducibility
    all at once, and no UI state can be trusted to have kept up -- ZIP import
    and direct API calls add images without the settings screen ever loading.
    """
    return has_stored_images(project_id)


def assert_format_changeable(project_id: str) -> None:
    """Raise 409 when the project's format is frozen."""
    if is_frozen(project_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "image format is frozen: this project already holds images. "
                "Convert the existing images instead of changing the setting."
            ),
        )
