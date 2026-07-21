# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)


def _copy_image_as_rgb_jpeg(src: Path, dest_dir: Path, item_id: str, *, quality: int = 95) -> Path:
    """Copy *src* into *dest_dir* as ``<item_id>.jpg`` (RGB, JPEG q95).

    Source images are commonly RGBA PNGs from annotation tools; the
    training pipeline always discards alpha and JPEG q=95 cuts disk size
    by ~85% and decode time by ~2x, lifting the I/O bottleneck out of
    the GPU step (verified on a real-world inspection dataset).
    Splits track stems only so the extension change is transparent
    downstream.
    """
    dest = dest_dir / f"{item_id}.jpg"
    with Image.open(src) as im:
        im.convert("RGB").save(dest, format="JPEG", quality=quality, optimize=False)
    return dest


from .annotate_index import load_annotate_index
from .classes import auto_inactivate_zero_mask_classes, collect_mask_class_presence
from .config import (
    AUTO_BG_WEIGHT_BOOST_MAX,
    AUTO_CLASS_WEIGHT_FG_RATIO_HIGH,
    AUTO_CLASS_WEIGHT_FG_RATIO_LOW,
    AUTO_CLASS_WEIGHT_STRENGTH_SCALE,
    AUTO_VAL_MIN_COUNT,
    AUTO_VAL_TARGET_RATIO,
    FIXED_INPUT_SIZE,  # noqa: F401
    IGNORE_INDEX,
    MODELS_DIR,  # noqa: F401
    NUM_CLASSES,  # noqa: F401
    OUTPUT_STRIDE,
    REGISTRY_DIR,
)
from .paths import annotate_images_dir, annotate_masks_dir, prepared_dir, project_dir, write_json


def discover_pairs(root: Path) -> tuple[list[Path], list[Path]]:
    image_candidates = []
    mask_candidates = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        lower = path.name.lower()
        parent = path.parent.name.lower()
        if parent in {"images", "jpegimages", "imgs"} and lower.endswith((".jpg", ".jpeg", ".png")):
            image_candidates.append(path)
        if parent in {"masks", "segmentationclass", "labels", "annotations"} and lower.endswith(".png"):
            mask_candidates.append(path)
    if not image_candidates or not mask_candidates:
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            lower = path.name.lower()
            if lower.endswith((".jpg", ".jpeg", ".png")) and "mask" not in lower:
                image_candidates.append(path)
            if lower.endswith(".png") and ("mask" in lower or "seg" in lower or "label" in lower):
                mask_candidates.append(path)
    if not image_candidates or not mask_candidates:
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_candidates.append(path)
            if path.name.lower().endswith(".png"):
                mask_candidates.append(path)
    return image_candidates, mask_candidates


def match_pairs(images: list[Path], masks: list[Path]) -> dict[str, tuple[Path, Path]]:
    masks_by_stem = {m.stem: m for m in masks}
    pairs = {}
    for img in images:
        mask = masks_by_stem.get(img.stem)
        if mask:
            pairs[img.stem] = (img, mask)
    return pairs


def auto_holdout_val_ids(train_ids: list[str], ratio: float = 0.2) -> tuple[list[str], list[str]]:
    if len(train_ids) <= 1:
        return list(train_ids), []
    holdout = int(round(len(train_ids) * ratio))
    holdout = max(1, min(holdout, len(train_ids) - 1))
    # Stable hash-based split keeps the same IDs in val across repeated prepare runs.
    ranked = sorted(train_ids, key=lambda item_id: hashlib.sha1(item_id.encode("utf-8")).hexdigest())
    val_set = set(ranked[:holdout])
    kept_train = [item_id for item_id in train_ids if item_id not in val_set]
    held_val = [item_id for item_id in train_ids if item_id in val_set]
    return kept_train, held_val


SAFETY_MIN_VAL = 2  # absolute minimum val items when val is empty or near-empty


def rebalance_train_val_ids(
    train_ids: list[str],
    val_ids: list[str],
    target_ratio: float = AUTO_VAL_TARGET_RATIO,
    min_val_count: int = AUTO_VAL_MIN_COUNT,
) -> tuple[list[str], list[str], int]:
    """Balance train/val split (safety net only).

    Policy: the user's val split (produced by prepare_annotate_dataset from
    val_ratio) is authoritative. We only step in when val is effectively
    missing (< SAFETY_MIN_VAL items) and rescue by moving a minimal number
    of training items to val so training has something to evaluate against.

    This replaces the older behaviour that proactively re-grew val to 25%
    of total / min 6 items — that stole training signal from small/medium
    datasets. Whatever split prepare_annotate_dataset produced from the
    user's val_ratio, we keep.

    *target_ratio* and *min_val_count* are kept as parameters for
    compatibility with existing call sites but are now unused.

    Returns (train_ids, val_ids, auto_moved_count).
    """
    del target_ratio, min_val_count  # legacy kwargs, intentionally unused
    # Keep insertion order while removing duplicates and overlap.
    unique_train = list(dict.fromkeys(train_ids))
    unique_val = [item_id for item_id in dict.fromkeys(val_ids) if item_id not in set(unique_train)]
    train_set = set(unique_train)
    unique_val = [item_id for item_id in unique_val if item_id not in train_set]

    total = len(unique_train) + len(unique_val)
    if total <= 1 or not unique_train:
        return unique_train, unique_val, 0

    # Safety net only: if the user produced a usable val split, leave it alone.
    if len(unique_val) >= SAFETY_MIN_VAL:
        return unique_train, unique_val, 0

    # Rescue path: top up to SAFETY_MIN_VAL (or whatever fits).
    target_val = min(SAFETY_MIN_VAL, max(1, total // 4))
    target_val = max(1, min(target_val, total - 1))
    # Ensure train keeps at least half of total (don't starve training)
    max_val = total // 2
    target_val = min(target_val, max_val)
    need = target_val - len(unique_val)
    if need <= 0:
        return unique_train, unique_val, 0

    can_take = max(0, len(unique_train) - 1)
    move_count = min(need, can_take)
    if move_count <= 0:
        return unique_train, unique_val, 0

    ranked = sorted(unique_train, key=lambda item_id: hashlib.sha1(item_id.encode("utf-8")).hexdigest())
    move_set = set(ranked[:move_count])
    moved_ids = [item_id for item_id in unique_train if item_id in move_set]
    kept_train = [item_id for item_id in unique_train if item_id not in move_set]
    next_val = unique_val + moved_ids
    return kept_train, next_val, len(moved_ids)


def estimate_foreground_ratio(
    masks_dir: Path,
    sample_ids: list[str],
    ignore_index: int,
    max_samples: int = 64,
) -> tuple[float, int]:
    if not sample_ids:
        return 0.0, 0
    chosen = list(sample_ids)
    if len(chosen) > max_samples:
        chosen = random.sample(chosen, max_samples)
    fg_pixels = 0
    valid_pixels = 0
    used = 0
    for item_id in chosen:
        mask_path = masks_dir / f"{item_id}.png"
        if not mask_path.exists():
            continue
        try:
            with Image.open(mask_path) as img:
                arr = np.array(img.convert("L"))
        except Exception:
            continue
        # Treat legacy 255 (unpainted) as background (0)
        arr[arr == 255] = 0
        valid = arr != ignore_index
        valid_count = int(valid.sum())
        if valid_count <= 0:
            continue
        fg_count = int(((arr > 0) & valid).sum())
        fg_pixels += fg_count
        valid_pixels += valid_count
        used += 1
    if valid_pixels <= 0:
        return 0.0, used
    return float(fg_pixels / valid_pixels), used


def adaptive_class_weight_strength(foreground_ratio: float) -> float:
    ratio = float(np.clip(foreground_ratio, 0.0, 1.0))
    low = float(AUTO_CLASS_WEIGHT_FG_RATIO_LOW)
    high = float(AUTO_CLASS_WEIGHT_FG_RATIO_HIGH)
    if high <= low:
        return 0.0
    if ratio <= low:
        return float(AUTO_CLASS_WEIGHT_STRENGTH_SCALE)
    if ratio >= high:
        return 0.0
    base = float((high - ratio) / (high - low))
    return float(np.clip(base * AUTO_CLASS_WEIGHT_STRENGTH_SCALE, 0.0, 1.0))


def adaptive_background_weight_boost(foreground_ratio: float) -> float:
    # Increase background penalty when foreground is very sparse to reduce false positives.
    s = adaptive_class_weight_strength(foreground_ratio)
    return float(1.0 + (AUTO_BG_WEIGHT_BOOST_MAX - 1.0) * s)


def prepare_dataset(project_id: str, export_dir: Path) -> dict:
    prepared = prepared_dir(project_id)
    images_dir = prepared / "images"
    masks_dir = prepared / "masks"
    splits_dir = prepared / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    images, masks = discover_pairs(export_dir)
    pairs = match_pairs(images, masks)
    copied = []
    missing_masks = []
    for img in images:
        if img.stem not in pairs:
            missing_masks.append(img.name)
            continue
    for stem, (img, mask) in pairs.items():
        _copy_image_as_rgb_jpeg(img, images_dir, stem)
        dest_mask = masks_dir / mask.name
        shutil.copy2(mask, dest_mask)
        copied.append(stem)

    copied.sort(key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    split_index = int(len(copied) * 0.8)
    train_ids = copied[:split_index]
    val_ids = copied[split_index:]
    (splits_dir / "train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_ids), encoding="utf-8")

    report = {
        "export_dir": str(export_dir),
        "total_images": len(images),
        "total_masks": len(masks),
        "paired": len(copied),
        "missing_masks": missing_masks,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
    }
    write_json(prepared / "report.json", report)
    return report


def _stable_split_pool(
    pool: list[str],
    val_ratio: float,
    test_ratio: float,
    pinned_train: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split a pool of IDs into train/val/test using hash-based stable sort.

    If ``pinned_train`` is set, those IDs bypass the ranking and land in
    train unconditionally — used by the iterative hard-mining loop to
    guarantee that images flagged as hard on the previous iteration are
    seen during training on the next one. Val / test counts scale down
    to what's left of the pool.

    If the *free* pool has fewer than 3 items, everything (free + pinned)
    goes to train.
    """
    pinned = set(pinned_train) if pinned_train else set()
    free_pool = [x for x in pool if x not in pinned]
    if len(free_pool) < 3:
        return list(pool), [], []
    val_count = max(1, round(len(free_pool) * val_ratio))
    test_count = max(1, round(len(free_pool) * test_ratio))
    # Ensure at least one free-pool id stays in the derived-train remainder
    if val_count + test_count >= len(free_pool):
        total_holdout = len(free_pool) - 1
        vr = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
        val_count = max(1, round(total_holdout * vr))
        test_count = max(1, total_holdout - val_count)
        if val_count + test_count >= len(free_pool):
            test_count = max(0, len(free_pool) - 1 - val_count)

    ranked = sorted(free_pool, key=lambda item_id: hashlib.sha1(item_id.encode("utf-8")).hexdigest())
    val_set = set(ranked[:val_count])
    test_set = set(ranked[val_count : val_count + test_count])
    train_ids = [item_id for item_id in pool if item_id in pinned or (item_id not in val_set and item_id not in test_set)]
    val_ids = [item_id for item_id in pool if item_id in val_set]
    test_ids = [item_id for item_id in pool if item_id in test_set]
    return train_ids, val_ids, test_ids


def _stable_kfold_split_pool(
    pool: list[str],
    k_folds: int,
    fold_index: int,
    test_ratio: float,
    pinned_train: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train, val, test) for one fold of a k-fold cross-validation.

    Deterministic: SHA1(id) rank picks a stable ordering. The top
    ``round(len(pool) * test_ratio)`` IDs are reserved as a **fixed** test
    hold-out that is identical across every fold — so k-fold rotates val
    inside the remaining train+val pool without leaking into test.

    The remaining IDs are then chunked into ``k_folds`` contiguous buckets
    by rank; bucket ``fold_index`` becomes val, the others go to train.
    Bucket sizes are balanced (first ``n % k_folds`` buckets are 1 larger).
    """
    pinned = set(pinned_train) if pinned_train else set()
    free_pool = [x for x in pool if x not in pinned]
    if len(free_pool) < 3 or k_folds < 2:
        return _stable_split_pool(pool, 1.0 / max(2, k_folds), test_ratio, pinned_train=pinned)
    if not 0 <= fold_index < k_folds:
        raise ValueError(f"fold_index {fold_index} out of range for k_folds={k_folds}")
    ranked = sorted(free_pool, key=lambda item_id: hashlib.sha1(item_id.encode("utf-8")).hexdigest())
    test_count = max(1, round(len(free_pool) * test_ratio)) if test_ratio > 0 else 0
    if len(free_pool) - test_count < k_folds:
        test_count = max(0, len(free_pool) - k_folds)
    test_set = set(ranked[:test_count]) if test_count > 0 else set()
    trainval_ranked = ranked[test_count:]
    n = len(trainval_ranked)
    base, extra = divmod(n, k_folds)
    fold_sizes = [base + (1 if i < extra else 0) for i in range(k_folds)]
    start = sum(fold_sizes[:fold_index])
    end = start + fold_sizes[fold_index]
    val_set = set(trainval_ranked[start:end])
    train_ids = [x for x in pool if x in pinned or (x not in test_set and x not in val_set)]
    val_ids = [x for x in pool if x in val_set]
    test_ids = [x for x in pool if x in test_set]
    return train_ids, val_ids, test_ids


def _compute_dinov2_embeddings(
    image_paths: dict[str, Path],
    device: str = "cuda:0",
    cache_path: Path | None = None,
    log_fn=None,
) -> dict[str, np.ndarray]:
    """Compute DINOv2 CLS-token embeddings per image, cached to disk.

    Loads the bundled dinov2_vitb14 via the existing distill.load_dinov2_teacher
    path (same weight file, same offline fallback). Preprocess is the standard
    ImageNet-normalized 224x224 center-crop. Returns id -> (768,) ndarray.

    Cache format (np.savez): {"ids": object array, "embeddings": (N, 768)}.
    Cache is invalidated when any requested id is missing.

    Any failure raises — callers should catch to fall back to hash split.
    """
    _log = log_fn if log_fn is not None else (lambda _msg: None)

    # Try cache first
    if cache_path is not None and cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_ids = data["ids"].tolist()
            cached_embs = data["embeddings"]
            cache_dict = dict(zip(cached_ids, cached_embs))
            if all(i in cache_dict for i in image_paths):
                _log(f"[embedding] Reusing {len(cache_dict)} cached embeddings\n")
                return {i: cache_dict[i] for i in image_paths}
        except Exception as _e:
            _log(f"[embedding] Cache load failed ({_e}), recomputing\n")

    # Load DINOv2 via existing helper
    import torch
    from packages.segcore.training.distill import load_dinov2_teacher

    wrapped_model, _ = load_dinov2_teacher(variant="dinov2_vitb14", device=device)
    # _DINOv2Wrapper stores the raw DINOv2 model as .dino; fall back to .model, then the wrapper itself.
    # ``or`` chain would trip nn.Module truthiness (Tensor bool ambiguity), so branch on ``is not None`` explicitly.
    _raw = getattr(wrapped_model, "dino", None)
    if _raw is None:
        _raw = getattr(wrapped_model, "model", None)
    model = _raw if _raw is not None else wrapped_model
    model.eval()

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings: dict[str, np.ndarray] = {}
    _log(f"[embedding] Computing DINOv2 embeddings for {len(image_paths)} images\n")
    with torch.no_grad():
        for img_id, img_path in image_paths.items():
            try:
                img = Image.open(img_path).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)
                out = model.forward_features(x)
                if isinstance(out, dict):
                    # ``a or b`` would evaluate Tensor truthiness (ambiguous on multi-value tensors),
                    # so branch on key presence instead.
                    if "x_norm_clstoken" in out:
                        emb_t = out["x_norm_clstoken"]
                    elif "cls_token" in out:
                        emb_t = out["cls_token"]
                    else:
                        emb_t = next(iter(out.values()))
                else:
                    emb_t = out
                embeddings[img_id] = emb_t.cpu().numpy().squeeze().astype("float32")
            except Exception as _e:
                _log(f"[embedding] Failed for {img_id}: {_e}\n")

    if cache_path is not None and embeddings:
        try:
            ids = list(embeddings.keys())
            embs_arr = np.stack([embeddings[i] for i in ids])
            np.savez(cache_path, ids=np.array(ids, dtype=object), embeddings=embs_arr)
            _log(f"[embedding] Cached {len(ids)} embeddings to {cache_path.name}\n")
        except Exception as _e:
            _log(f"[embedding] Cache save failed: {_e}\n")

    return embeddings


def _stratified_split_by_embedding(
    pool: list[str],
    val_ratio: float,
    test_ratio: float,
    embeddings_by_id: dict[str, np.ndarray],
    pinned_train: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split a pool into train/val/test via KMeans on precomputed embeddings.

    Each cluster contributes ``round(cluster_size * test_ratio)`` medoids
    (closest to the centroid) to test, then ``round(cluster_size * val_ratio)``
    next-closest to val. The rest, especially outliers far from the centroid,
    go to train — the intent is to keep test as "representative typicals"
    and let unique / hard cases enter the training set.

    Falls back to ``_stable_split_pool`` if the pool is too small (< 3),
    if embeddings are missing for any id, or if KMeans fails.
    """
    pinned = set(pinned_train) if pinned_train else set()
    free_pool = [x for x in pool if x not in pinned]
    if len(free_pool) < 3:
        return list(pool), [], []
    if not all(i in embeddings_by_id for i in free_pool):
        return _stable_split_pool(pool, val_ratio, test_ratio, pinned_train=pinned)

    embs = np.stack([embeddings_by_id[i] for i in free_pool])
    N = len(free_pool)
    K = max(2, min(int(np.sqrt(N)), 10))
    K = min(K, N)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(embs)
        labels = km.labels_
        centers = km.cluster_centers_
    except Exception:
        return _stable_split_pool(pool, val_ratio, test_ratio)

    # Group by cluster, sorted closest-first to the centroid
    per_cluster: dict[int, list[tuple[str, float]]] = {c: [] for c in range(K)}
    for idx, item_id in enumerate(free_pool):
        c = int(labels[idx])
        dist = float(np.linalg.norm(embs[idx] - centers[c]))
        per_cluster[c].append((item_id, dist))
    for c in per_cluster:
        per_cluster[c].sort(key=lambda x: x[1])

    val_ids: list[str] = []
    test_ids: list[str] = []
    for c, members in per_cluster.items():
        n = len(members)
        # Proportional allocation with at-least-1 when cluster is big enough.
        c_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        c_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        if c_test + c_val >= n:
            # Ensure at least 1 stays in train per cluster
            c_test = max(0, min(c_test, n - 2))
            c_val = max(0, min(c_val, n - 1 - c_test))
        for member_id, _ in members[:c_test]:
            test_ids.append(member_id)
        for member_id, _ in members[c_test:c_test + c_val]:
            val_ids.append(member_id)

    test_set = set(test_ids)
    val_set = set(val_ids)
    train_ids = [x for x in pool if x in pinned or (x not in test_set and x not in val_set)]
    val_ids = [x for x in pool if x in val_set]
    test_ids = [x for x in pool if x in test_set]
    return train_ids, val_ids, test_ids


def _compute_dataset_stats(
    images_dir: Path,
    masks_dir: Path,
    train_count: int,
    val_count: int,
    test_count: int,
) -> dict:
    """Compute basic scalar dataset stats for the ML combo predictor.

    Mirrors the fields the ablation library was built with: num_train/val/total,
    mean_width/height, fg_ratio, mean_fg_area_px, std_fg_area_px,
    mean_fg_ratio_per_image, fg_area_frac, num_active_classes,
    class_imbalance_ratio, log_num_train, log_img_pixels.

    Kept in-sync with training_runner._compute_basic_stats_fallback.
    """
    import math as _math

    img_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".bmp")
    ) if images_dir.exists() else []
    mask_files = sorted(
        f for f in masks_dir.iterdir()
        if f.suffix.lower() in (".png", ".bmp", ".tif")
    ) if masks_dir.exists() else []
    if not img_files or not mask_files:
        return {}

    widths: list[int] = []
    heights: list[int] = []
    for p in img_files[:30]:
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue

    fg_areas: list[int] = []
    fg_ratios: list[float] = []
    class_pixel_counts: dict[int, int] = {}
    for mf in mask_files[:60]:
        try:
            arr = np.array(Image.open(mf).convert("L"))
            total_px = int(arr.shape[0] * arr.shape[1])
            fg_mask = (arr > 0) & (arr != IGNORE_INDEX)
            fg_px = int(fg_mask.sum())
            if fg_px > 0:
                fg_areas.append(fg_px)
                fg_ratios.append(fg_px / total_px)
            for cls_id in np.unique(arr):
                if int(cls_id) == IGNORE_INDEX:
                    continue
                class_pixel_counts[int(cls_id)] = class_pixel_counts.get(int(cls_id), 0) + int((arr == cls_id).sum())
        except Exception:
            continue

    mean_w = float(np.mean(widths)) if widths else 0.0
    mean_h = float(np.mean(heights)) if heights else 0.0
    img_px = mean_w * mean_h
    mean_fg_area = float(np.mean(fg_areas)) if fg_areas else 0.0
    std_fg_area = float(np.std(fg_areas, ddof=0)) if len(fg_areas) > 1 else 0.0
    fg_ratio = float(np.mean(fg_ratios)) if fg_ratios else 0.0

    # class_imbalance_ratio matches the library definition: max/min over all
    # class pixel counts (including background).
    all_counts = [c for c in class_pixel_counts.values() if c > 0]
    num_active_classes = float(len(class_pixel_counts))
    class_imbalance = 0.0
    if len(all_counts) >= 2:
        class_imbalance = max(all_counts) / max(1.0, min(all_counts))

    num_total = train_count + val_count + test_count
    stats: dict[str, float] = {
        "num_train": float(train_count),
        "num_val": float(val_count),
        "num_test": float(test_count),
        "num_total": float(num_total),
        "mean_width": mean_w,
        "mean_height": mean_h,
        "fg_ratio": fg_ratio,
        "mean_fg_area_px": mean_fg_area,
        "std_fg_area_px": std_fg_area,
        "mean_fg_ratio_per_image": fg_ratio,
        "num_active_classes": num_active_classes,
        "class_imbalance_ratio": class_imbalance,
    }
    if train_count > 0:
        stats["log_num_train"] = _math.log1p(train_count)
    if img_px > 0:
        stats["log_img_pixels"] = _math.log(img_px)
        if mean_fg_area > 0:
            stats["fg_area_frac"] = mean_fg_area / img_px
    return stats


def prepare_annotate_dataset(
    project_id: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    include_pseudo: bool = False,
    pseudo_weight: float = 0.5,
    include_unmasked: bool = False,
    k_folds: int = 1,
    fold_index: int = 0,
    split_method: str = "hash",
    log_fn=None,
    pinned_train_ids: list[str] | set[str] | None = None,
) -> dict:
    prepared = prepared_dir(project_id)
    images_dir = prepared / "images"
    masks_dir = prepared / "masks"
    splits_dir = prepared / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    index = load_annotate_index(project_id)
    items = index.get("items", [])
    annotate_images = annotate_images_dir(project_id)
    annotate_masks = annotate_masks_dir(project_id)

    total = len(items)
    with_mask = 0
    without_mask = 0
    masked_ids: list[str] = []
    # Sub-pools used so the fg/clean balance in the raw dataset is preserved
    # inside the val/test splits. Without this, projects that rely heavily
    # on "Mark Clean" (e.g. film inspection with very rare defects) end up
    # with val sets that are 80%+ clean, driving val F1 to 0 regardless of
    # actual model quality.
    fg_ids: list[str] = []
    clean_ids: list[str] = []
    unmasked_ids: list[str] = []
    copied_mask_paths: list[Path] = []

    # Build filename lookup for all items
    item_filenames: dict[str, str] = {}
    for item in items:
        item_id = item.get("id")
        filename = item.get("filename")
        if item_id and filename:
            item_filenames[item_id] = filename

    for item in items:
        item_id = item.get("id")
        filename = item.get("filename")
        annotation = item.get("annotation") or {}
        has_mask = bool(annotation.get("hasMask"))
        marked_clean = bool(annotation.get("markedClean"))
        # hasForeground is the authoritative per-image fg indicator once the
        # index has been refreshed; fall back to "not markedClean" when it's
        # missing so legacy indexes still classify correctly.
        has_fg = annotation.get("hasForeground")
        if has_fg is None:
            has_fg = not marked_clean
        if not item_id or not filename:
            continue
        image_path = annotate_images / filename
        if not image_path.exists():
            continue
        mask_path = annotate_masks / f"{item_id}.png"
        zarr_path = annotate_masks / f"{item_id}.zarr"
        if has_mask and (mask_path.exists() or zarr_path.is_dir()):
            with_mask += 1
            dest_img = _copy_image_as_rgb_jpeg(image_path, images_dir, item_id)
            dest_mask = masks_dir / f"{item_id}.png"
            # Convert unpainted pixels (255/ignore) to background (0)
            # so the model always learns proper background.
            if zarr_path.is_dir():
                import zarr as _zarr
                _z = _zarr.open_array(str(zarr_path), mode="r")
                _mask_arr = _z[:]
            else:
                _mask_img = Image.open(mask_path)
                if _mask_img.mode in ("RGBA", "RGB", "PA"):
                    _mask_arr = np.array(_mask_img)[:, :, 0]
                else:
                    _mask_arr = np.array(_mask_img.convert("L"))
            _n255 = int((_mask_arr == IGNORE_INDEX).sum())
            if _n255 > 0:
                _mask_arr[_mask_arr == IGNORE_INDEX] = 0
                _pct = _n255 / _mask_arr.size * 100
                _logger.info("%s: %d ignore pixels -> BG (%.1f%%)", item_id[:8], _n255, _pct)
            Image.fromarray(_mask_arr, mode="L").save(dest_mask)
            copied_mask_paths.append(dest_mask)
            masked_ids.append(item_id)
            # Verify the on-disk mask actually has fg: markedClean flags and
            # hasForeground can lag behind the current PNG contents. Use the
            # mask we just wrote as the final source of truth.
            if bool((_mask_arr > 0).any()):
                fg_ids.append(item_id)
            else:
                clean_ids.append(item_id)
        else:
            without_mask += 1
            _copy_image_as_rgb_jpeg(image_path, images_dir, item_id)
            unmasked_ids.append(item_id)

    # Only manual-masked images go into train/val/test splits. We split the
    # fg-bearing pool and the clean (mark-clean / empty) pool independently
    # so every split keeps the same fg:clean ratio as the overall dataset.
    # Splitting the union causes projects with rare defects to end up with
    # val/test sets that are almost entirely clean, hiding real fg accuracy.
    _emb_by_id: dict[str, np.ndarray] = {}
    _used_split_method = split_method
    _pinned = set(pinned_train_ids) if pinned_train_ids else set()
    if _pinned and log_fn is not None:
        log_fn(f"Iterative: pinning {len(_pinned)} hard IDs into the training set.\n")
    if k_folds > 1:
        _used_split_method = "hash"  # k-fold reuses the SHA1 rank; embedding is single-split only.
        fg_train, fg_val, fg_test = _stable_kfold_split_pool(fg_ids, k_folds, fold_index, test_ratio, pinned_train=_pinned)
        clean_train, clean_val, clean_test = _stable_kfold_split_pool(clean_ids, k_folds, fold_index, test_ratio, pinned_train=_pinned)
    elif split_method == "embedding_stratified" and masked_ids:
        try:
            _img_paths = {mid: (images_dir / f"{mid}.jpg") for mid in masked_ids}
            _cache = prepared / "embeddings.npz"
            _emb_by_id = _compute_dinov2_embeddings(
                _img_paths, device="cuda:0", cache_path=_cache, log_fn=log_fn,
            )
            if not _emb_by_id:
                raise RuntimeError("no embeddings computed")
            fg_train, fg_val, fg_test = _stratified_split_by_embedding(fg_ids, val_ratio, test_ratio, _emb_by_id, pinned_train=_pinned)
            clean_train, clean_val, clean_test = _stratified_split_by_embedding(clean_ids, val_ratio, test_ratio, _emb_by_id, pinned_train=_pinned)
            if log_fn is not None:
                log_fn(f"[embedding] Stratified split OK: fg=({len(fg_train)}/{len(fg_val)}/{len(fg_test)}), clean=({len(clean_train)}/{len(clean_val)}/{len(clean_test)})\n")
        except Exception as _emb_err:
            _logger.warning("Embedding split failed (%s); falling back to hash.", _emb_err)
            if log_fn is not None:
                log_fn(f"[embedding] FAILED ({_emb_err}); falling back to hash split.\n")
            _used_split_method = "hash"
            fg_train, fg_val, fg_test = _stable_split_pool(fg_ids, val_ratio, test_ratio, pinned_train=_pinned)
            clean_train, clean_val, clean_test = _stable_split_pool(clean_ids, val_ratio, test_ratio, pinned_train=_pinned)
    else:
        _used_split_method = "hash"
        fg_train, fg_val, fg_test = _stable_split_pool(fg_ids, val_ratio, test_ratio, pinned_train=_pinned)
        clean_train, clean_val, clean_test = _stable_split_pool(clean_ids, val_ratio, test_ratio, pinned_train=_pinned)

    # Rebuild splits in the original masked_ids order so downstream code that
    # scans directories alphabetically sees a consistent ordering.
    fg_val_set = set(fg_val)
    fg_test_set = set(fg_test)
    clean_val_set = set(clean_val)
    clean_test_set = set(clean_test)
    val_set = fg_val_set | clean_val_set
    test_set = fg_test_set | clean_test_set
    train_ids: list[str] = [mid for mid in masked_ids if mid not in val_set and mid not in test_set]
    val_ids: list[str] = [mid for mid in masked_ids if mid in val_set]
    test_ids: list[str] = [mid for mid in masked_ids if mid in test_set]
    _logger.info(
        "Balanced split: fg=%d (train=%d/val=%d/test=%d), clean=%d (train=%d/val=%d/test=%d)",
        len(fg_ids), len(fg_train), len(fg_val), len(fg_test),
        len(clean_ids), len(clean_train), len(clean_val), len(clean_test),
    )

    # Include unmasked images in train set as all-background (no FG)
    unmasked_count = 0
    if include_unmasked and unmasked_ids:
        for uid in unmasked_ids:
            # Create all-zero mask (pure background)
            filename = item_filenames.get(uid)
            if not filename:
                continue
            src_img = annotate_images / filename
            if not src_img.exists():
                continue
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(src_img) as img:
                    w, h = img.size
                zero_mask = Image.new("L", (w, h), 0)
                dest_mask = masks_dir / f"{uid}.png"
                zero_mask.save(dest_mask)
                copied_mask_paths.append(dest_mask)
            except Exception:
                continue
            train_ids.append(uid)
            unmasked_count += 1
        if unmasked_count > 0:
            _logger.info("Added %d unmasked images to train set (all-background)", unmasked_count)

    # --- Pseudo-label integration (train set only) ---
    pseudo_count = 0
    pseudo_ids: list[str] = []
    if include_pseudo:
        pseudo_dir = project_dir(project_id) / "pseudo_masks"
        if pseudo_dir.is_dir():
            manual_set = set(masked_ids)
            for pseudo_mask_path in pseudo_dir.glob("*.png"):
                pid = pseudo_mask_path.stem
                if pid in manual_set:
                    continue  # Manual masks take priority
                # Need the image to exist
                filename = item_filenames.get(pid)
                if not filename:
                    continue
                src_img = annotate_images / filename
                if not src_img.exists():
                    continue
                # Copy image as RGB JPEG (alpha is unused downstream)
                dest_img = images_dir / f"{pid}.jpg"
                if not dest_img.exists():
                    _copy_image_as_rgb_jpeg(src_img, images_dir, pid)
                # Copy pseudo mask (already clean, no 255)
                dest_mask = masks_dir / f"{pid}.png"
                shutil.copy2(pseudo_mask_path, dest_mask)
                copied_mask_paths.append(dest_mask)
                pseudo_ids.append(pid)
                pseudo_count += 1

            # Add pseudo IDs to train set ONLY (never val/test)
            train_ids.extend(pseudo_ids)
            if pseudo_count > 0:
                _logger.info("Added %d pseudo-labeled images to train set (weight=%s)", pseudo_count, pseudo_weight)

    # Write pseudo weight marker so train.py can apply lower loss weight
    pseudo_weight_path = prepared / "pseudo_ids.json"
    if pseudo_ids:
        pseudo_weight_path.write_text(
            json.dumps({"ids": pseudo_ids, "weight": pseudo_weight}, ensure_ascii=False),
            encoding="utf-8",
        )
    elif pseudo_weight_path.exists():
        pseudo_weight_path.unlink()

    (splits_dir / "train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_ids), encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(test_ids), encoding="utf-8")

    class_presence = collect_mask_class_presence(copied_mask_paths)
    auto_inactive_class_ids = auto_inactivate_zero_mask_classes(project_id, class_presence)

    report = {
        "source": "annotate",
        "total_items": total,
        "with_mask": with_mask,
        "without_mask": without_mask,
        "unmasked_in_train": unmasked_count if include_unmasked else 0,
        "pseudo_count": pseudo_count,
        "pseudo_weight": pseudo_weight if pseudo_count > 0 else None,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "auto_inactive_class_ids": auto_inactive_class_ids,
    }
    write_json(prepared / "report.json", report)

    # dataset_stats.json: basic scalar features consumed by the ML combo
    # predictor (and other auto-* paths). Previously missing, which caused
    # the predictor to see NaN for log_img_pixels / fg_ratio / num_train etc.
    try:
        _stats = _compute_dataset_stats(
            images_dir, masks_dir,
            train_count=len(train_ids),
            val_count=len(val_ids),
            test_count=len(test_ids),
        )
        if _stats:
            write_json(prepared / "dataset_stats.json", _stats)
    except Exception as _ds_err:
        _logger.warning("dataset_stats.json generation failed: %s", _ds_err)
    return report


def build_dummy_onnx(model_path: Path, num_classes: int, input_size: list[int], output_stride: int) -> None:
    import onnx
    from onnx import TensorProto, helper
    in_w = int(input_size[0])
    in_h = int(input_size[1])
    out_stride = int(output_stride) if int(output_stride) > 0 else int(OUTPUT_STRIDE)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, in_h, in_w])
    output_tensor = helper.make_tensor_value_info(
        "logits",
        TensorProto.FLOAT,
        [1, num_classes, in_h // out_stride, in_w // out_stride],
    )
    weight = helper.make_tensor("W", TensorProto.FLOAT, [num_classes, 3, 1, 1], [0.0] * (num_classes * 3))
    bias = helper.make_tensor("B", TensorProto.FLOAT, [num_classes], [0.0] * num_classes)
    conv_node = helper.make_node("Conv", inputs=["input", "W", "B"], outputs=["logits"], kernel_shape=[1, 1])
    graph = helper.make_graph([conv_node], "seg-studio-dummy", [input_tensor], [output_tensor], [weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path)


def scan_registry() -> list[str]:
    if not REGISTRY_DIR.exists():
        return []
    return sorted([p.name for p in REGISTRY_DIR.iterdir() if p.is_dir() and (p / "model.onnx").exists()])
