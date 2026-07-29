# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Streaming dataset statistics.

Returns the bytes-on-disk and pixel-count distribution of a prepared dataset
without loading every image into memory at once. Used by the planner to
decide whether the dataset fits in a bytes-mode warm cache, a decoded cache,
or needs to be streamed.

We sample a bounded subset (default 32 files) and extrapolate, because for
multi-thousand-image datasets a full scan would itself become an I/O cost we
want to amortise.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


_DEFAULT_SAMPLE_SIZE = 32
_PROFILE_CACHE_TTL_SEC = 24 * 3600


@dataclass
class DatasetProfile:
    images_dir: str
    masks_dir: str

    num_train: int
    num_val: int

    # Sampled statistics (bytes on disk, post-compression).
    image_bytes_mean: int
    image_bytes_max: int
    mask_bytes_mean: int
    mask_bytes_max: int

    # Decoded pixel counts (H × W). The decoded RGB byte-count is
    # 3 × pixels for uint8, 12 × pixels for float32.
    pixels_mean: int
    pixels_max: int

    # Number of files actually sampled (for diagnostics).
    sample_size: int

    # ---- derived helpers ---------------------------------------------------

    def total_image_bytes_estimate(self) -> int:
        """Disk bytes for all training images (extrapolated from sample)."""
        return self.image_bytes_mean * self.num_train

    def total_mask_bytes_estimate(self) -> int:
        return self.mask_bytes_mean * self.num_train

    def max_decoded_image_bytes(self, channels: int = 3, dtype_bytes: int = 1) -> int:
        return self.pixels_max * channels * dtype_bytes

    def total_decoded_image_bytes_estimate(self, channels: int = 3, dtype_bytes: int = 1) -> int:
        return self.pixels_mean * channels * dtype_bytes * self.num_train


# ---------------------------------------------------------------------------

def probe_dataset(prepared_dir: Path,
                  sample_size: int = _DEFAULT_SAMPLE_SIZE,
                  cache_path: Path | None = None) -> DatasetProfile:
    """Profile a prepared dataset directory.

    Expects ``prepared_dir/{images, masks, splits/{train.txt, val.txt}}``.

    Result is memoised at ``cache_path`` (default
    ``prepared_dir / "_dataset_profile.json"``) and invalidated when any of
    the inputs' mtimes change.
    """
    prepared_dir = Path(prepared_dir)
    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"
    splits_dir = prepared_dir / "splits"

    if cache_path is None:
        cache_path = prepared_dir / "_dataset_profile.json"

    train_ids = _load_split(splits_dir / "train.txt")
    val_ids = _load_split(splits_dir / "val.txt")

    fingerprint = _fingerprint(images_dir, masks_dir, train_ids, val_ids, sample_size)

    cached = _read_cache(cache_path)
    if cached is not None and cached.get("_fingerprint") == fingerprint:
        try:
            data = dict(cached)
            data.pop("_fingerprint", None)
            data.pop("_ts", None)
            return DatasetProfile(**data)
        except TypeError:
            pass  # cache schema mismatch — fall through

    if not train_ids:
        return DatasetProfile(
            images_dir=str(images_dir), masks_dir=str(masks_dir),
            num_train=0, num_val=len(val_ids),
            image_bytes_mean=0, image_bytes_max=0,
            mask_bytes_mean=0, mask_bytes_max=0,
            pixels_mean=0, pixels_max=0, sample_size=0,
        )

    # Pick a deterministic spread of stems so re-running gives stable numbers.
    n = min(sample_size, len(train_ids))
    if n >= len(train_ids):
        sample_stems = list(train_ids)
    else:
        step = len(train_ids) / n
        sample_stems = [train_ids[int(i * step)] for i in range(n)]

    img_bytes: list[int] = []
    msk_bytes: list[int] = []
    pixel_counts: list[int] = []

    for stem in sample_stems:
        ip = _find_by_stem(images_dir, stem)
        mp = _find_by_stem(masks_dir, stem)
        if ip is not None:
            try:
                img_bytes.append(ip.stat().st_size)
            except OSError:
                pass
            px = _read_image_dimensions(ip)
            if px is not None:
                pixel_counts.append(px)
        if mp is not None:
            try:
                msk_bytes.append(mp.stat().st_size)
            except OSError:
                pass

    profile = DatasetProfile(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        num_train=len(train_ids),
        num_val=len(val_ids),
        image_bytes_mean=int(_mean(img_bytes)),
        image_bytes_max=int(max(img_bytes) if img_bytes else 0),
        mask_bytes_mean=int(_mean(msk_bytes)),
        mask_bytes_max=int(max(msk_bytes) if msk_bytes else 0),
        pixels_mean=int(_mean(pixel_counts)),
        pixels_max=int(max(pixel_counts) if pixel_counts else 0),
        sample_size=len(sample_stems),
    )

    _write_cache(cache_path, profile, fingerprint)
    return profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: list[int]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def _load_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return []


def _find_by_stem(d: Path, stem: str) -> Path | None:
    if not d.is_dir():
        return None
    # Cheap: try common extensions first; fall back to glob.
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    matches = list(d.glob(f"{stem}.*"))
    return matches[0] if matches else None


def _read_image_dimensions(path: Path) -> int | None:
    """Return H × W without decoding pixels.

    PIL.Image.open is lazy — it only reads the header, so this is cheap even
    for large images.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            return int(w) * int(h)
    except Exception:
        return None


def _fingerprint(images_dir: Path, masks_dir: Path,
                 train_ids: list[str], val_ids: list[str],
                 sample_size: int) -> str:
    parts = [
        str(images_dir),
        str(masks_dir),
        str(len(train_ids)),
        str(len(val_ids)),
        str(sample_size),
    ]
    for d in (images_dir, masks_dir):
        try:
            parts.append(str(d.stat().st_mtime))
        except OSError:
            parts.append("0")
    return "|".join(parts)


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data.get("_ts", 0) > _PROFILE_CACHE_TTL_SEC:
            return None
        return data
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, profile: DatasetProfile, fingerprint: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(profile)
        data["_fingerprint"] = fingerprint
        data["_ts"] = time.time()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        logger.debug("dataset profile cache write failed: %s", e)
