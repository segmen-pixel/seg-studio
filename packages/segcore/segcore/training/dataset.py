# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..dataset_layout import ImageSource
from ..tiling_geometry import default_patch_stride
from .split_utils import _find_by_stem as _find_by_stem_standalone

logger = logging.getLogger(__name__)


# Cache modes. "decoded" stores PIL.Image objects in memory (zero per-step
# cost, but pickles poorly across spawn boundaries — main-process / workers=0
# only). "bytes" stores the raw file contents (tiny per-step decode cost, but
# safe under fork-COW). "none" reads from disk every step (workers > 0 only,
# typically). The DataLoader planner picks the right one for the host.
_CACHE_MODES = {"decoded", "bytes", "none"}


def _reflect_indices(start: int, count: int, n: int) -> np.ndarray:
    """Fold window indices [start, start+count) into [0, n) by even reflection.

    Reproduces the extension ``np.pad(mode="reflect")`` computes about 0 and
    n-1 (including repeated reflections when the overshoot exceeds the axis
    length), without materializing the padded array.
    """
    idx = np.arange(start, start + count)
    if n <= 1:
        return np.zeros(count, dtype=idx.dtype)
    period = 2 * n - 2
    idx = np.mod(idx, period)
    return np.where(idx < n, idx, period - idx)


class SegDataset(Dataset):
    """PyTorch ``Dataset`` for paired image + segmentation-mask training.

    Wraps a flat-on-disk dataset and produces normalized image / class-index
    mask pairs suitable for cross-entropy training. Supports random patch
    sampling biased toward foreground, foreground-centered cropping,
    geometric + photometric augmentation, hard-negative mining injection,
    and pseudo-labeled sample down-weighting.

    Input directory layout:
        Images and masks live in two parallel directories, one entry per
        stem. When an ``image_source`` is attached, stems resolve through
        its descriptor; otherwise the image extension is probed via
        ``_find_by_stem`` (``.png``, ``.jpg``, ``.webp``, ``.tif``,
        ``.tiff``, ``.bmp``). Mask files are read as single-channel "L"
        PNG/JPG whose pixel values are class indices. Pixel value ``255``
        is relabeled to background (``0``) only while
        ``relabel_ignore_as_bg`` is True (the default); otherwise it is
        kept as an ignore label.

        ``split_ids`` selects which stems this dataset instance exposes.

    Key options:
        - ``input_size`` + ``output_stride``: post-resize image / mask
          resolution. ``input_size`` must divide evenly by ``output_stride``.
        - ``patch_size`` + ``patches_per_image`` + ``fg_patch_prob``:
          enable patch-based sampling. ``__len__`` becomes
          ``len(split_ids) * patches_per_image`` (oversampled to 64 min).
        - ``annotation_patches_only`` + ``context_expand``: restrict patch
          centers to a precomputed grid around annotated regions.
        - ``crop_foreground`` + ``crop_scale``: full-image foreground-biased
          crop applied before patch sampling.
        - ``augment_*``: probabilities / strengths for h/v flip, rot90,
          brightness, contrast, gaussian noise.
        - ``pseudo_ids`` + ``pseudo_weight``: items in ``pseudo_ids`` are
          flagged as pseudo-labeled and emit a reduced sample weight.
        - ``cache_mode``: ``"decoded"`` keeps PIL.Image in memory (fastest,
          workers=0 only), ``"bytes"`` keeps raw file bytes (fork-safe),
          ``"none"`` re-reads disk each step.
        - ``return_meta``: switch the 3rd tuple element from a scalar
          weight tensor to a dict containing augmentation state.
        - ``relabel_ignore_as_bg``: when True (default), mask pixels equal
          to ``255`` are relabeled to ``0`` before use. Set False when
          ``255`` marks real ignore regions that must stay out of the loss.
        - ``image_source``: optional ``ImageSource`` descriptor. When set,
          ``_resolve_image`` asks it for the file and does not probe
          extensions under ``images_dir``.

    Returns:
        Each ``__getitem__(idx)`` returns a 3-tuple ``(image, mask, extra)``:
            - ``image`` (``torch.FloatTensor`` of shape ``[3, H, W]``):
              normalized RGB at ``input_size``.
            - ``mask`` (``torch.LongTensor`` of shape ``[H/output_stride,
              W/output_stride]``): per-pixel class index. ``255`` is
              relabeled to 0 only when ``relabel_ignore_as_bg`` is True.
            - ``extra``: if ``return_meta`` is False, a scalar
              ``FloatTensor`` sample weight (``pseudo_weight`` for pseudo
              items, else ``1.0``). If True, a dict with keys
              ``{"stem_idx", "hflip", "vflip", "rot90_k", "sample_weight"}``.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        split_ids: list[str],
        input_size: list[int],
        normalize: dict,
        output_stride: int = 4,
        crop_foreground: bool = False,
        crop_scale: float = 1.0,
        patch_size: int = 0,
        patches_per_image: int = 1,
        fg_patch_prob: float = 0.0,
        annotation_patches_only: bool = False,
        context_expand: float = 0.0,
        augment_enabled: bool = False,
        augment_hflip_prob: float = 0.0,
        augment_vflip_prob: float = 0.0,
        augment_rotate90_prob: float = 0.0,
        augment_brightness: float = 0.0,
        augment_contrast: float = 0.0,
        augment_noise_std: float = 0.0,
        augment_translate: float = 0.0,
        return_meta: bool = False,
        allow_missing_mask: bool = False,
        pseudo_ids: set[str] | None = None,
        pseudo_weight: float = 0.5,
        hard_ids: set[str] | None = None,
        hard_weight_boost: float = 3.0,
        cache_mode: str = "decoded",
        relabel_ignore_as_bg: bool = True,
        image_source: ImageSource | None = None,
    ):
        self.images_dir = images_dir
        # When the prepared directory describes itself, stems resolve
        # through that description instead of by probing extensions. The
        # directory is still kept, because callers log it and because a
        # legacy dataset has nothing else.
        self.image_source = image_source
        self.relabel_ignore_as_bg = bool(relabel_ignore_as_bg)
        self.masks_dir = masks_dir
        self.split_ids = split_ids
        self.input_size = input_size
        self.normalize = normalize
        self.output_stride = output_stride
        self.crop_foreground = crop_foreground
        self.crop_scale = crop_scale
        self.patch_size = max(0, int(patch_size))
        self.patches_per_image = max(1, int(patches_per_image))
        self.fg_patch_prob = float(np.clip(fg_patch_prob, 0.0, 1.0))
        self.annotation_patches_only = bool(annotation_patches_only)
        self.context_expand = float(max(0.0, context_expand))
        self.augment_enabled = bool(augment_enabled)
        self.augment_hflip_prob = float(np.clip(augment_hflip_prob, 0.0, 1.0))
        self.augment_vflip_prob = float(np.clip(augment_vflip_prob, 0.0, 1.0))
        self.augment_rotate90_prob = float(np.clip(augment_rotate90_prob, 0.0, 1.0))
        self.augment_brightness = float(np.clip(augment_brightness, 0.0, 1.0))
        self.augment_contrast = float(np.clip(augment_contrast, 0.0, 1.0))
        self.augment_noise_std = float(np.clip(augment_noise_std, 0.0, 0.5))
        # Fraction of the shorter side; 0.5 would shift the content clean out
        # of frame, so the useful range is small.
        self.augment_translate = float(np.clip(augment_translate, 0.0, 0.5))
        self.return_meta = bool(return_meta)
        # False: a mask that is absent is a data fault, not an empty label.
        self.allow_missing_mask = bool(allow_missing_mask)
        self.pseudo_ids: set[str] = pseudo_ids or set()
        self.pseudo_weight = float(np.clip(pseudo_weight, 0.0, 1.0))
        self.hard_ids: set[str] = set(hard_ids) if hard_ids else set()
        self.hard_weight_boost = float(np.clip(hard_weight_boost, 1.0, 10.0))
        if cache_mode not in _CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {_CACHE_MODES}, got {cache_mode!r}")
        self.cache_mode = cache_mode
        if input_size[0] % output_stride != 0 or input_size[1] % output_stride != 0:
            raise ValueError("input_size must be divisible by output_stride")

        # Pre-compute normalization tensors (shape [3,1,1] for broadcast)
        self._norm_mean = torch.tensor(normalize["mean"], dtype=torch.float32).view(3, 1, 1)
        self._norm_std = torch.tensor(normalize["std"], dtype=torch.float32).view(3, 1, 1)

        # Pre-compute annotation region grid for annotation_patches_only mode
        self._annotation_grid: dict[str, np.ndarray] = {}
        if self.annotation_patches_only and self.patch_size > 0:
            self._precompute_annotation_grid()

        # Cache foreground pixel coordinates per image for fast patch sampling
        self._fg_coords_cache: dict[str, np.ndarray] = {}
        # In-memory image/mask cache. Element type depends on cache_mode:
        #   "decoded" -> PIL.Image.Image
        #   "bytes"   -> bytes (raw file contents, decoded per __getitem__)
        #   "none"    -> never populated
        self._img_cache: dict[str, object] = {}
        self._mask_cache: dict[str, object] = {}
        self._cache_warmed = False

        # Hard negative mining: FP patch centers injected during training
        self._hn_centers: dict[str, np.ndarray] = {}  # stem -> Nx2 (cx, cy)

        # Per-instance RNG to avoid polluting / depending on global np.random state.
        # Re-seeded per worker on first use: DataLoader forks/spawns a copy of
        # this object into each worker, so a seed fixed here gives every worker
        # the identical stream -- the same patch positions, flips, rotations and
        # noise, drawn in lockstep. That is less augmentation diversity than the
        # single-worker case, silently, and only when num_workers > 1.
        self._rng_base_seed = 42
        self._rng = np.random.default_rng(self._rng_base_seed)
        self._rng_worker_id: int | None = None

    def _check_masks_present(self) -> None:
        """Pre-flight the whole split for missing masks."""
        missing = []
        for stem in self.split_ids:
            try:
                self._find_by_stem(self.masks_dir, stem)
            except FileNotFoundError:
                missing.append(stem)
        self._report_missing_masks(missing)

    def _report_missing_masks(self, missing: list[str]) -> None:
        if not missing:
            return
        if not self.allow_missing_mask:
            # Report the whole gap at once, before training starts, rather than
            # failing on whichever batch happens to reach the first bad item.
            sample = ", ".join(missing[:5])
            more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.split_ids)} items have no mask in "
                f"{self.masks_dir}: {sample}{more}. Missing masks would be trained and "
                f"scored as all-background. Fix the dataset, or pass "
                f"allow_missing_mask=True if this split is deliberately unlabelled."
            )
        logger.warning(
            "%d items have no mask and will be treated as all-background "
            "(allow_missing_mask=True)", len(missing))

    def __len__(self) -> int:
        base = len(self.split_ids) * self.patches_per_image
        # Oversample small datasets so each epoch has enough gradient steps.
        # Round up to a whole multiple of `base` rather than to exactly 64:
        # __getitem__ maps back with `idx % base`, so a length that is not a
        # multiple gave the first (64 % base) items one extra appearance per
        # epoch -- a quiet, order-dependent reweighting of the dataset.
        min_samples = 64
        if base <= 0:
            return 0
        if base < min_samples:
            reps = -(-min_samples // base)  # ceil
            return base * reps
        return base

    def _crop_around_foreground(self, image: Image.Image, mask: Image.Image, stem: str = "") -> tuple[Image.Image, Image.Image]:
        img_w, img_h = image.size
        crop_w = max(32, int(img_w * self.crop_scale))
        crop_h = max(32, int(img_h * self.crop_scale))
        if img_w < crop_w or img_h < crop_h:
            return image, mask
        fg_coords = self._get_fg_coords(stem) if stem else np.empty((0, 2), dtype=np.int32)
        if fg_coords.shape[0] == 0:
            mask_np = np.array(mask)
            fg_coords = np.argwhere((mask_np > 0) & (mask_np != 255))
        if fg_coords.shape[0] == 0:
            return image, mask
        idx = self._rng.integers(0, fg_coords.shape[0])
        cy, cx = int(fg_coords[idx, 0]), int(fg_coords[idx, 1])
        jitter_x = self._rng.integers(-crop_w // 4, crop_w // 4 + 1)
        jitter_y = self._rng.integers(-crop_h // 4, crop_h // 4 + 1)
        cx = max(crop_w // 2, min(img_w - crop_w // 2, cx + jitter_x))
        cy = max(crop_h // 2, min(img_h - crop_h // 2, cy + jitter_y))
        left = cx - crop_w // 2
        top = cy - crop_h // 2
        right = left + crop_w
        bottom = top + crop_h
        return image.crop((left, top, right, bottom)), mask.crop((left, top, right, bottom))

    def warm_cache(self) -> None:
        """Pre-load images and masks into memory according to ``cache_mode``.

        - "decoded": stores PIL.Image instances. Cheapest per-step but cannot
          survive a worker spawn (PIL Images don't pickle compactly).
        - "bytes": stores raw file contents. Decode cost moves into
          __getitem__, but the cache survives fork/spawn cleanly.
        - "none": no-op; __getitem__ reads from disk every time.
        """
        if self._cache_warmed:
            return
        if self.cache_mode == "none":
            # Nothing to cache, but the pre-flights still have to run: they
            # are the only thing that reports a gap up front instead of
            # failing on whichever batch happens to reach the bad item first.
            #
            # The image half used to be skipped here, and "none" is chosen
            # for the largest datasets -- so the one configuration where a
            # missing image costs the most was the one that never looked.
            self._check_images_present()
            self._check_masks_present()
            self._cache_warmed = True
            return
        skipped: list[str] = []
        missing_masks: list[str] = []
        for stem in self.split_ids:
            if stem not in self._img_cache:
                try:
                    image_path = self._resolve_image(stem)
                except FileNotFoundError:
                    logger.warning("Skipping %s: image file not found in %s", stem, self.images_dir)
                    skipped.append(stem)
                    continue
                if self.cache_mode == "decoded":
                    self._img_cache[stem] = Image.open(image_path).convert("RGB")
                else:  # bytes
                    self._img_cache[stem] = Path(image_path).read_bytes()
            if stem not in self._mask_cache:
                try:
                    mask_path = self._find_by_stem(self.masks_dir, stem)
                    if self.cache_mode == "decoded":
                        self._mask_cache[stem] = Image.open(mask_path).convert("L")
                    else:  # bytes
                        self._mask_cache[stem] = Path(mask_path).read_bytes()
                except FileNotFoundError:
                    missing_masks.append(stem)
        self._report_missing_masks(missing_masks)
        if skipped:
            for s in skipped:
                self.split_ids.remove(s)
            logger.warning("Removed %d missing items from split (remaining: %d)", len(skipped), len(self.split_ids))
        self._cache_warmed = True

    def _seed_rng_for_worker(self) -> None:
        """Give this worker its own augmentation stream, once."""
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
        except Exception:
            info = None
        wid = 0 if info is None else int(info.id)
        if self._rng_worker_id == wid:
            return
        self._rng_worker_id = wid
        # Derived from the base seed, so a run stays reproducible for a given
        # worker count while the workers no longer mirror each other.
        self._rng = np.random.default_rng([self._rng_base_seed, wid])

    def __getitem__(self, idx: int):
        if not self.split_ids:
            raise IndexError("empty dataset")
        self._seed_rng_for_worker()
        # Wrap index for oversampled small datasets
        base = len(self.split_ids) * self.patches_per_image
        idx = idx % base
        stem_idx = idx // self.patches_per_image
        stem = self.split_ids[stem_idx]
        # Resolve image / mask. Cache entries may be PIL Images ("decoded"
        # mode) or raw bytes ("bytes" mode); fall through to disk on miss.
        # Do NOT .copy() decoded cached images: _sample_patch uses .crop() which
        # already returns a new Image, so copying the full image is wasted
        # memcpy.
        _ic = getattr(self, '_img_cache', {})
        _mc = getattr(self, '_mask_cache', {})
        if stem in _ic:
            entry = _ic[stem]
            if isinstance(entry, (bytes, bytearray)):
                image = Image.open(io.BytesIO(entry)).convert("RGB")
            else:
                image = entry
        else:
            image_path = self._resolve_image(stem)
            image = Image.open(image_path).convert("RGB")
        if stem in _mc:
            entry = _mc[stem]
            if isinstance(entry, (bytes, bytearray)):
                mask = Image.open(io.BytesIO(entry)).convert("L")
            else:
                mask = entry
        else:
            try:
                mask_path = self._find_by_stem(self.masks_dir, stem)
                mask = Image.open(mask_path).convert("L")
            except FileNotFoundError:
                mask = None
        if mask is None:
            if not self.allow_missing_mask:
                # An all-background stand-in makes a labelling gap look like a
                # clean part: the image trains as "nothing here", and a correct
                # prediction on it scores as a false positive. Every cause is a
                # data fault worth stopping for -- masks not copied, a filename
                # that does not match its image, a broken split, a wrong
                # extension. Opt in explicitly for genuinely unlabelled data.
                raise FileNotFoundError(
                    f"No mask for '{stem}' in {self.masks_dir}. A missing mask would be "
                    f"trained and scored as all-background. Fix the dataset, or pass "
                    f"allow_missing_mask=True if this split is deliberately unlabelled."
                )
            mask = Image.new("L", image.size, 0)

        # The frame the coordinate channels are normalised against, read
        # before anything crops: a window's position is only meaningful
        # against the whole picture it was cut from.
        frame_w, frame_h = image.size
        window = (0, 0, frame_w, frame_h)
        if self.crop_foreground:
            image, mask = self._crop_around_foreground(image, mask, stem=stem)
        if self.patch_size > 0:
            image, mask, window = self._sample_patch(image, mask, stem=stem)

        # Masks written before ignore existed use 255 for "unpainted", and
        # the pipeline has always trained those as background; dataset_prep
        # converts most of them and this catches the rest. A run whose masks
        # use 255 for an actual ignore must keep it: converting THAT to
        # background teaches the model that a masked-out surface is
        # background. Which meaning is in force comes from the config, never
        # from the pixels.
        mask_np = np.array(mask)
        if self.relabel_ignore_as_bg and np.any(mask_np == 255):
            if not getattr(self, '_warned_255', False):
                _n = int((mask_np == 255).sum())
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "mask %s has %d ignore(255) pixels -> converting to BG", stem, _n)
                self._warned_255 = True
            mask_np[mask_np == 255] = 0
            mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")

        aug_state = {"hflip": 0, "vflip": 0, "rot90_k": 0}
        if self.augment_enabled:
            image, mask, aug_state = self._apply_augment(image, mask)

        image = image.resize((self.input_size[0], self.input_size[1]), resample=Image.BILINEAR)
        out_w = self.input_size[0] // self.output_stride
        out_h = self.input_size[1] // self.output_stride
        mask = mask.resize((out_w, out_h), resample=Image.NEAREST)
        # Convert to tensor and normalize in-place.
        # np.array (not asarray) to force a writable copy — avoids torch's
        # non-writable-tensor warning that floods logs.
        img_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255.0)
        img_t.sub_(self._norm_mean).div_(self._norm_std)
        mask_t = torch.from_numpy(np.array(mask).astype("int64"))
        # Sample weight priority: hard mining boost > pseudo down-weight > 1.0
        if stem in self.hard_ids:
            sample_weight = self.hard_weight_boost
        elif stem in self.pseudo_ids:
            sample_weight = self.pseudo_weight
        else:
            sample_weight = 1.0
        if self.return_meta:
            meta = {
                "stem_idx": stem_idx,
                "hflip": aug_state["hflip"],
                "vflip": aug_state["vflip"],
                "rot90_k": aug_state["rot90_k"],
                "sample_weight": sample_weight,
            }
            return img_t, mask_t, meta
        return img_t, mask_t, torch.tensor(sample_weight, dtype=torch.float32)

    def _precompute_annotation_grid(self) -> None:
        """Pre-compute patch centers from annotation regions.

        When context_expand > 0 (e.g. 3.0):
          - Expands each annotation bbox by the factor (3x = 1x padding each side)
          - Tiles the expanded region with fixed patch_size patches at 75% stride
          - Combined with ignore→BG relabeling, this trains the model on both
            foreground AND surrounding background at native resolution

        When context_expand == 0:
          - Original centroid-based single patch per compact annotation
          - Large/elongated regions get tiled at 50% overlap
        """
        from scipy import ndimage as ndi

        tile_stride = default_patch_stride(self.patch_size)  # 25% overlap for context tiling

        for stem in self.split_ids:
            try:
                mask_path = self._find_by_stem(self.masks_dir, stem)
            except FileNotFoundError:
                self._annotation_grid[stem] = np.empty((0, 2), dtype=np.int32)
                continue
            mask_np = np.array(Image.open(mask_path).convert("L"))
            img_h, img_w = mask_np.shape
            fg_mask = (mask_np > 0) & (mask_np != 255)
            if not fg_mask.any():
                self._annotation_grid[stem] = np.empty((0, 2), dtype=np.int32)
                continue

            labeled, num_cc = ndi.label(fg_mask)
            # Filter out noise: ignore components smaller than 4 pixels
            cc_areas = ndi.sum(fg_mask, labeled, range(1, num_cc + 1))
            min_area = 4
            centers = []
            for cc_id in range(1, num_cc + 1):
                if cc_areas[cc_id - 1] < min_area:
                    continue
                cc_ys, cc_xs = np.where(labeled == cc_id)
                y_min, y_max = int(cc_ys.min()), int(cc_ys.max())
                x_min, x_max = int(cc_xs.min()), int(cc_xs.max())
                cc_h = y_max - y_min + 1
                cc_w = x_max - x_min + 1

                if self.context_expand > 0:
                    # Expand bbox by context_expand factor, then tile at native res
                    pad_h = int(cc_h * (self.context_expand - 1) / 2)
                    pad_w = int(cc_w * (self.context_expand - 1) / 2)
                    exp_y_min = max(0, y_min - pad_h)
                    exp_y_max = min(img_h - 1, y_max + pad_h)
                    exp_x_min = max(0, x_min - pad_w)
                    exp_x_max = min(img_w - 1, x_max + pad_w)
                    exp_h = exp_y_max - exp_y_min + 1
                    exp_w = exp_x_max - exp_x_min + 1

                    if exp_h <= self.patch_size and exp_w <= self.patch_size:
                        # Expanded region fits in one patch
                        cx = (exp_x_min + exp_x_max) // 2
                        cy = (exp_y_min + exp_y_max) // 2
                        centers.append((cx, cy))
                    else:
                        # Tile expanded region with fixed-size patches
                        for gy in range(exp_y_min, max(exp_y_min + 1, exp_y_max - self.patch_size + 2), tile_stride):
                            for gx in range(exp_x_min, max(exp_x_min + 1, exp_x_max - self.patch_size + 2), tile_stride):
                                cx = gx + self.patch_size // 2
                                cy = gy + self.patch_size // 2
                                centers.append((cx, cy))
                else:
                    # Original mode: centroid-based
                    cy = int(cc_ys.mean())
                    cx = int(cc_xs.mean())
                    fg_area = len(cc_ys)
                    bbox_area = cc_h * cc_w
                    fill_ratio = fg_area / max(bbox_area, 1)
                    step = self.patch_size // 2

                    if (cc_h <= self.patch_size and cc_w <= self.patch_size
                            and fill_ratio > 0.5):
                        centers.append((cx, cy))
                    else:
                        for gy in range(y_min, y_max + 1, step):
                            for gx in range(x_min, x_max + 1, step):
                                centers.append((gx + step // 2, gy + step // 2))

            self._annotation_grid[stem] = np.array(centers, dtype=np.int32) if centers else np.empty((0, 2), dtype=np.int32)

    def _get_fg_coords(self, stem: str) -> np.ndarray:
        """Return cached Nx2 array of (y, x) foreground pixel coordinates."""
        if stem not in self._fg_coords_cache:
            try:
                mask_path = self._find_by_stem(self.masks_dir, stem)
                mask_np = np.array(Image.open(mask_path).convert("L"))
                fg = (mask_np > 0) & (mask_np != 255)
                coords = np.argwhere(fg)  # Nx2 (y, x)
            except FileNotFoundError:
                coords = np.empty((0, 2), dtype=np.int32)
            self._fg_coords_cache[stem] = coords
        return self._fg_coords_cache[stem]

    def set_hard_negatives(self, hn_map: dict[str, np.ndarray]) -> None:
        """Inject hard negative patch centers from FP mining."""
        self._hn_centers = hn_map

    def _sample_patch(
        self, image: Image.Image, mask: Image.Image, stem: str = ""
    ) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
        """Cut one patch, and say where in the frame it was cut from.

        The window ``(x0, y0, w, h)`` is in FRAME pixels and x0/y0 go
        negative inside the virtual reflect margin. It is the only record of
        where a patch came from, and nothing else can reconstruct it after.
        """
        img_w, img_h = image.size
        crop_w = min(self.patch_size, img_w)
        crop_h = min(self.patch_size, img_h)
        if crop_w <= 0 or crop_h <= 0:
            return image, mask, (0, 0, img_w, img_h)
        if crop_w == img_w and crop_h == img_h:
            return image, mask, (0, 0, img_w, img_h)

        # Reflect-pad half a patch on every side, matching the margin
        # sliding-window inference adds (sliding_window.py). Without it,
        # patch centers clamp >= patch/2 away from the border, so border
        # pixels only ever train at patch EDGES — while at inference they
        # sit mid-patch surrounded by mirrored context the model has never
        # seen. That train/infer mismatch showed up as border-hugging FPs
        # (most of the residual FP pixels sat right at the border).
        # The margin stays virtual: padding the full image here cost
        # O(image) per sample (tens of MB of allocation on a 5 MP frame,
        # thousands of times per epoch), which dominated epoch time. The crop at the
        # end of this function maps the window through the same reflection
        # instead, touching only O(patch) pixels.
        _pad = crop_w // 2 if crop_w == crop_h else 0

        # Annotation-only mode: 3-way split (NG / context / random)
        # fg_patch_prob controls NG ratio; remainder split evenly between
        # context (annotation grid) and random (anywhere in image).
        # e.g. fg_patch_prob=0.50 → 50% NG / 25% context / 25% random
        if self.annotation_patches_only and stem in self._annotation_grid:
            grid = self._annotation_grid[stem]
            ng_prob = self.fg_patch_prob
            ctx_prob = (1.0 - ng_prob) / 2.0
            roll = float(self._rng.random())

            if roll < ng_prob and grid.shape[0] > 0:
                # NG patch: center on a foreground pixel (cached coords)
                fg_coords = self._get_fg_coords(stem)
                if fg_coords.shape[0] > 0:
                    pick = int(self._rng.integers(0, fg_coords.shape[0]))
                    cy, cx = int(fg_coords[pick, 0]), int(fg_coords[pick, 1])
                else:
                    # No FG pixels, fallback to grid
                    pick = int(self._rng.integers(0, grid.shape[0]))
                    cx, cy = int(grid[pick, 0]), int(grid[pick, 1])
            elif roll < ng_prob + ctx_prob and grid.shape[0] > 0:
                # Context patch: from annotation grid (expanded NG area)
                pick = int(self._rng.integers(0, grid.shape[0]))
                cx, cy = int(grid[pick, 0]), int(grid[pick, 1])
                jitter = self.patch_size // 4
                cx += int(self._rng.integers(-jitter, jitter + 1))
                cy += int(self._rng.integers(-jitter, jitter + 1))
                cx = int(np.clip(cx, 0, img_w - 1))
                cy = int(np.clip(cy, 0, img_h - 1))
            else:
                # Random patch — prefer hard negative centers if available
                hn = self._hn_centers.get(stem)
                if hn is not None and len(hn) > 0 and self._rng.random() < 0.7:
                    idx = self._rng.integers(len(hn))
                    cx, cy = int(hn[idx, 0]), int(hn[idx, 1])
                    # Add jitter of ±patch_size/4
                    jitter = self.patch_size // 4
                    cx += int(self._rng.integers(-jitter, jitter + 1))
                    cy += int(self._rng.integers(-jitter, jitter + 1))
                    cx = np.clip(cx, 0, img_w - 1)
                    cy = np.clip(cy, 0, img_h - 1)
                else:
                    cx = int(self._rng.integers(0, img_w))
                    cy = int(self._rng.integers(0, img_h))
        else:
            # Original mode: random or fg_patch_prob
            use_fg = self._rng.random() < self.fg_patch_prob
            cx = int(self._rng.integers(0, img_w))
            cy = int(self._rng.integers(0, img_h))
            if use_fg:
                fg_coords = self._get_fg_coords(stem)
                if fg_coords.shape[0] > 0:
                    pick = int(self._rng.integers(0, fg_coords.shape[0]))
                    cy = int(fg_coords[pick, 0])
                    cx = int(fg_coords[pick, 1])

        # Centers (cx, cy) are in ORIGINAL image coordinates; with the
        # reflect margin the crop window always fits, so an edge-centered
        # patch really is centered on the edge pixel.
        max_left = img_w + 2 * _pad - crop_w
        max_top = img_h + 2 * _pad - crop_h
        left = int(np.clip(cx + _pad - crop_w // 2, 0, max_left))
        top = int(np.clip(cy + _pad - crop_h // 2, 0, max_top))
        x0 = left - _pad
        y0 = top - _pad
        if 0 <= x0 and 0 <= y0 and x0 + crop_w <= img_w and y0 + crop_h <= img_h:
            # The window lies fully inside the image: plain crop, no
            # reflected pixels involved.
            box = (x0, y0, x0 + crop_w, y0 + crop_h)
            return image.crop(box), mask.crop(box), (x0, y0, crop_w, crop_h)
        # The window overlaps the virtual margin: gather the reflected
        # pixels from a just-large-enough crop of the original.
        idx_x = _reflect_indices(x0, crop_w, img_w)
        idx_y = _reflect_indices(y0, crop_h, img_h)
        lo_x, hi_x = int(idx_x.min()), int(idx_x.max()) + 1
        lo_y, hi_y = int(idx_y.min()), int(idx_y.max()) + 1
        img_np = np.asarray(image.crop((lo_x, lo_y, hi_x, hi_y)))
        mask_np = np.asarray(mask.crop((lo_x, lo_y, hi_x, hi_y)))
        ix = idx_x - lo_x
        iy = idx_y - lo_y
        return (
            Image.fromarray(img_np[iy[:, None], ix[None, :]]),
            Image.fromarray(mask_np[iy[:, None], ix[None, :]]),
            (x0, y0, crop_w, crop_h),
        )

    def _shift_pair(
        self, image: Image.Image, mask: Image.Image, dx: int, dy: int
    ) -> tuple[Image.Image, Image.Image]:
        """Shift image and mask by the same integer offset.

        The image is mirror-padded (the same even reflection the patch sampler
        uses) so the vacated band keeps plausible texture; the mask is padded
        with the ignore label (255) so those pixels contribute no loss. Both are
        plain array slices, so no label is ever interpolated and the two stay
        aligned exactly.
        """
        img_np = np.asarray(image)
        mask_np = np.asarray(mask)
        h, w = mask_np.shape[:2]
        pad_x, pad_y = abs(dx), abs(dy)
        # np.pad's "reflect" needs the pad to be smaller than the axis;
        # "edge" covers the degenerate remainder deterministically.
        img_mode = "reflect" if (pad_x < w and pad_y < h and w > 1 and h > 1) else "edge"
        img_pad_width = [(pad_y, pad_y), (pad_x, pad_x)] + [(0, 0)] * (img_np.ndim - 2)
        img_np = np.pad(img_np, img_pad_width, mode=img_mode)
        mask_np = np.pad(
            mask_np, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=255
        )
        y0, x0 = pad_y - dy, pad_x - dx
        img_np = img_np[y0 : y0 + h, x0 : x0 + w]
        mask_np = mask_np[y0 : y0 + h, x0 : x0 + w]
        return (
            Image.fromarray(np.ascontiguousarray(img_np), mode=image.mode),
            Image.fromarray(np.ascontiguousarray(mask_np), mode=mask.mode),
        )

    def _apply_augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
        state = {"hflip": 0, "vflip": 0, "rot90_k": 0}
        may_move = True
        hflip_prob = self.augment_hflip_prob if may_move else 0.0
        vflip_prob = self.augment_vflip_prob if may_move else 0.0
        rot90_prob = self.augment_rotate90_prob if may_move else 0.0
        translate = self.augment_translate if may_move else 0.0
        if self._rng.random() < hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            state["hflip"] = 1
        if self._rng.random() < vflip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            state["vflip"] = 1
        if self._rng.random() < rot90_prob:
            k = int(self._rng.integers(1, 4))
            if k == 1:
                method = Image.ROTATE_90
            elif k == 2:
                method = Image.ROTATE_180
            else:
                method = Image.ROTATE_270
            image = image.transpose(method)
            mask = mask.transpose(method)
            state["rot90_k"] = k

        if translate > 0.0:
            w, h = image.size
            # The mask is resized to input_size/output_stride while the image
            # goes to input_size, so a shift that is not a whole number of
            # mask cells would land the two on different lattices.
            step = max(1, int(self.output_stride))
            max_dx = (int(w * translate) // step) * step
            max_dy = (int(h * translate) // step) * step
            if max_dx > 0 or max_dy > 0:
                dx = int(self._rng.integers(-max_dx // step, max_dx // step + 1)) * step
                dy = int(self._rng.integers(-max_dy // step, max_dy // step + 1)) * step
                if dx or dy:
                    image, mask = self._shift_pair(image, mask, dx, dy)

        if self.augment_brightness > 0.0 or self.augment_contrast > 0.0 or self.augment_noise_std > 0.0:
            img = np.asarray(image).astype("float32") / 255.0
            if self.augment_brightness > 0.0:
                brightness_factor = 1.0 + float(self._rng.uniform(-self.augment_brightness, self.augment_brightness))
                img = img * brightness_factor
            if self.augment_contrast > 0.0:
                contrast_factor = 1.0 + float(self._rng.uniform(-self.augment_contrast, self.augment_contrast))
                mean = img.mean(axis=(0, 1), keepdims=True)
                img = (img - mean) * contrast_factor + mean
            if self.augment_noise_std > 0.0:
                noise = self._rng.normal(0.0, self.augment_noise_std, img.shape).astype("float32")
                img = img + noise
            img = np.clip(img, 0.0, 1.0)
            image = Image.fromarray((img * 255.0).astype("uint8"), mode="RGB")
        return image, mask, state

    @staticmethod
    def _find_by_stem(root: Path, stem: str) -> Path:
        return _find_by_stem_standalone(root, stem)

    def _resolve_image(self, stem: str) -> Path:
        """The file holding *stem*'s pixels, or FileNotFoundError."""
        if self.image_source is not None:
            return self.image_source.resolve_or_raise(stem)
        return self._find_by_stem(self.images_dir, stem)

    def _check_images_present(self) -> None:
        """Drop split entries whose image cannot be resolved, and say so.

        Mirrors what the caching branch already did on the way past. It
        removes rather than raises: this is Phase 0 of the layout change
        and the tolerant behaviour is what every existing project has been
        running under. Turning a gap into a hard failure is a separate,
        deliberate step.
        """
        missing = []
        for stem in self.split_ids:
            try:
                self._resolve_image(stem)
            except FileNotFoundError:
                missing.append(stem)
        if not missing:
            return
        for stem in missing:
            self.split_ids.remove(stem)
        logger.warning(
            "Removed %d item(s) with no image from split (remaining: %d); first: %s",
            len(missing), len(self.split_ids), missing[:3],
        )

