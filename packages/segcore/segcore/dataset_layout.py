# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What a prepared dataset directory holds, and where its images really are.

Splits record stems. Turning a stem back into a file has been guesswork
repeated in eight places, each with its own extension tuple and no two of them
the same -- ``.tif`` missing in one, ``.webp`` missing in another. That was
survivable only because ``prepared/images`` re-encoded every training image to
``.png`` or ``.jpg``, so the first guess almost always hit.

Once training reads the annotation images directly, the guessing has to stop:
that directory holds whatever the user imported. So a prepared directory now
carries a descriptor -- ``dataset.json`` -- that names the image directory and
maps every stem in the splits to its actual filename. Consumers ask this
module rather than the filesystem.

This lives in segcore because both sides need one definition and segcore
cannot import the Trainer API. ``training/layout.py`` exists for the same
reason, and says so.

A prepared directory with no descriptor is a *legacy* one: it predates this
module, its images are the ``prepared/images`` copies, and it is resolved by
the old probe. That branch is not dead scaffolding -- it is the rollback route
for the migration, and the existing golden-run tests build their datasets that
way, so it is covered by tests deliberately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Name of the descriptor, inside the prepared directory.
DESCRIPTOR_NAME = "dataset.json"

#: Descriptor version. A reader that does not recognise the version falls back
#: to legacy rather than guessing at a layout it does not understand: a wrong
#: guess here is a silently empty training set, which no reported score shows.
DESCRIPTOR_SCHEMA = 1

#: Version of the population rule behind ``dataset_stats.json``.
#:
#: Deliberately a separate number from DESCRIPTOR_SCHEMA. The freshness gate
#: has to be able to say "this project has a descriptor but its statistics were
#: computed under the old rule", and it cannot say that if the two share a
#: field -- writing the descriptor would mark the statistics fresh in the same
#: instant, and the population change would never fire.
STATS_SCHEMA = 1

#: Where a prepared directory's images live.
#:
#: ``images`` is the project's own annotation images -- one copy of the pixels,
#: the direction this is going. ``prepared_images`` is the second, re-encoded
#: copy under ``prepared/images``, which is what every existing project still
#: uses. The switch itself is stored in ``project.json``, not here; this field
#: records what was in force when the descriptor was written.
LAYOUT_IMAGES = "images"
LAYOUT_PREPARED_IMAGES = "prepared_images"
KNOWN_LAYOUTS = (LAYOUT_IMAGES, LAYOUT_PREPARED_IMAGES)

#: Every image container the app accepts, in one tuple.
#:
#: The eight private copies of this list disagreed with each other, and each
#: disagreement is a class of image that one consumer silently skips. Adding a
#: container means adding it here and nowhere else.
CANONICAL_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: Probe order for legacy directories, kept exactly as ``split_utils`` had it.
#:
#: NOT the same order as CANONICAL_IMAGE_EXTS, and not an oversight: when a
#: stem matches more than one file the winner decides which pixels get trained
#: on, and re-ordering the probe would silently change that for directories
#: written before this module existed. New directories carry a descriptor and
#: never probe, so the canonical order is the one that matters going forward.
_LEGACY_PROBE_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def descriptor_path(prepared_dir: Path) -> Path:
    """Where *prepared_dir* keeps its descriptor."""
    return Path(prepared_dir) / DESCRIPTOR_NAME


def _probe_legacy(root: Path, stem: str) -> Path | None:
    """The pre-descriptor stem probe, returning None instead of raising.

    Mirrors ``segcore.training.split_utils._find_by_stem`` step for step,
    including the whitespace fallbacks -- a project can have hundreds of split
    ids whose stems begin with a space, matched by files whose names begin
    with a space, so the exact-match branch carries them and the fallbacks
    stay unused.
    They are kept because a directory that needed them once may still exist in
    somebody's backup.
    """
    for ext in _LEGACY_PROBE_EXTS:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(root.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    stripped = stem.strip()
    if stripped != stem:
        return _probe_legacy(root, stripped)
    for ext in _LEGACY_PROBE_EXTS:
        for candidate in sorted(root.glob(f"*{stem}*{ext}")):
            if candidate.stem.strip() == stem:
                return candidate
    return None


@dataclass(frozen=True)
class ImageSource:
    """Resolves a split stem to the file holding its pixels.

    Two modes, and the difference is deliberate rather than incidental:

    * **Descriptor mode** (``items`` is set) knows every filename. A stem that
      is not in ``items`` is unresolved, full stop -- no probe, no glob. A glob
      against a 20,254-file directory once per image is not a fallback, it is a
      performance bug that hides a data bug.
    * **Legacy mode** (``items`` is None) probes, exactly as the code did
      before descriptors existed.

    Whether an unresolved stem is fatal is the *caller's* decision, not this
    class's -- see ``resolve`` versus ``resolve_or_raise``. Making it always
    fatal breaks the auto-configuration path, whose broad ``except Exception``
    turns the raise into a silently downgraded recommendation.
    """

    images_dir: Path
    items: dict[str, str] | None = None
    #: What the descriptor said the layout was. Informational: consumers that
    #: only resolve stems never need it, but the freshness gate compares it
    #: against project.json to notice a descriptor written before a flip.
    images_layout: str = LAYOUT_PREPARED_IMAGES
    #: Image format in force when the descriptor was written, for provenance.
    image_format: str = ""
    stats_schema: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def is_legacy(self) -> bool:
        return self.items is None

    def resolve(self, stem: str) -> Path | None:
        """The file for *stem*, or None when it cannot be resolved."""
        if self.items is None:
            return _probe_legacy(self.images_dir, stem)
        name = self.items.get(stem)
        if name is None:
            # One step, and only this one. Split ids are read through
            # load_split_ids, which strips every line, so a descriptor
            # written from the raw index ids is keyed with whitespace the
            # caller has already removed. A project can have hundreds of such
            # ids; the legacy probe carried them on its last fallback and
            # nothing noticed until the descriptor stopped probing.
            name = self.items.get(stem.strip())
        if not name:
            return None
        candidate = self.images_dir / name
        return candidate if candidate.exists() else None

    def resolve_or_raise(self, stem: str) -> Path:
        """The file for *stem*, or FileNotFoundError.

        For the callers where a missing image means the training set is wrong
        and continuing would produce a number nobody can trust.
        """
        found = self.resolve(stem)
        if found is None:
            raise FileNotFoundError(f"missing image for {stem!r} under {self.images_dir}")
        return found

    def unresolved(self, stems) -> list[str]:
        """Which of *stems* cannot be resolved, in the order given."""
        return [s for s in stems if self.resolve(s) is None]

    def resolved_paths(self, stems) -> list[Path]:
        """The files for *stems*, skipping the ones that do not resolve."""
        out: list[Path] = []
        for stem in stems:
            found = self.resolve(stem)
            if found is not None:
                out.append(found)
        return out


def as_image_source(value) -> ImageSource:
    """An ImageSource for *value*, which may already be one or a directory.

    The leaf evaluators take an images directory positionally and are
    reached through several call chains. Accepting both lets those chains
    be upgraded one at a time, instead of every caller having to change on
    the same commit -- and, more to the point, lets the leaves drop their
    private extension lists now rather than after the last chain lands.
    """
    if isinstance(value, ImageSource):
        return value
    return ImageSource(images_dir=Path(value), items=None)


def read_descriptor(prepared_dir: Path) -> dict | None:
    """The parsed descriptor, or None when there isn't a usable one.

    Unreadable, malformed and unknown-version descriptors all return None, so
    the caller lands on the legacy path. Raising here would take down a project
    that worked yesterday because a file got truncated; the legacy path still
    finds its images.
    """
    path = descriptor_path(prepared_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") != DESCRIPTOR_SCHEMA:
        return None
    return raw


def load_layout(prepared_dir: Path) -> ImageSource:
    """How to resolve stems for the dataset in *prepared_dir*.

    Always returns an ImageSource. A caller that wants to know whether the
    images are actually there asks the ImageSource, rather than testing for a
    directory: ``(prepared_dir / "images").exists()`` is exactly the check that
    silently sends training to the wrong pixels once the copies are gone.
    """
    prepared_dir = Path(prepared_dir)
    raw = read_descriptor(prepared_dir)
    if raw is None:
        return ImageSource(images_dir=prepared_dir / "images", items=None)

    rel = str(raw.get("images_dir") or "images")
    # POSIX separators in the descriptor, resolved against the descriptor's own
    # directory: the documented backup procedure is to copy a project folder
    # back into projects/, so an absolute path baked in here would point at the
    # machine that wrote it.
    images_dir = (prepared_dir / Path(rel)).resolve()

    items_raw = raw.get("items")
    items: dict[str, str] | None = None
    if isinstance(items_raw, dict):
        items = {}
        for stem, entry in items_raw.items():
            if isinstance(entry, dict):
                name = entry.get("file")
            else:
                name = entry
            if isinstance(name, str) and name:
                key = str(stem)
                items[key] = name
                # Split ids arrive stripped -- load_split_ids strips every
                # line it reads -- while an id from the annotation index can
                # carry surrounding whitespace, and a project can have hundreds
                # of them. A descriptor keyed by the raw id would be
                # unreachable for exactly those stems, and the miss shows up
                # as a shorter split rather than an error. setdefault, so a
                # genuine entry for the stripped form always wins.
                stripped = key.strip()
                if stripped and stripped != key:
                    items.setdefault(stripped, name)

    layout = raw.get("images_layout")
    if layout not in KNOWN_LAYOUTS:
        layout = LAYOUT_PREPARED_IMAGES
    return ImageSource(
        images_dir=images_dir,
        items=items,
        images_layout=layout,
        image_format=str(raw.get("image_format") or ""),
        stats_schema=int(raw.get("stats_schema") or 0),
        extra={k: v for k, v in raw.items() if k != "items"},
    )


def build_descriptor(
    *,
    images_layout: str,
    images_dir_rel: str,
    items: dict[str, dict],
    image_format: str,
    generated_at: str,
    masks_dir: str = "masks",
    splits_dir: str = "splits",
    source: str | None = None,
) -> dict:
    """The descriptor payload for a freshly prepared dataset.

    *items* maps stem to ``{"file": ..., "size": ..., "mtime": ...}``. The size
    and mtime are provenance, not a cache key on their own: they let a run say
    which bytes it read, which ``report.json`` never could because prepare
    overwrites it.
    """
    if images_layout not in KNOWN_LAYOUTS:
        raise ValueError(f"unknown images_layout: {images_layout!r}")
    payload = {
        "schema": DESCRIPTOR_SCHEMA,
        "images_layout": images_layout,
        "images_dir": images_dir_rel,
        "masks_dir": masks_dir,
        "splits_dir": splits_dir,
        "image_format": image_format,
        "stats_schema": STATS_SCHEMA,
        "generated_at": generated_at,
        "items": items,
    }
    if source:
        payload["source"] = source
    return payload
