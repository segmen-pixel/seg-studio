# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""One training epoch: batch loop, loss combination, distillation, AMP step.

Extracted verbatim from train() during the pre-OSS refactor. Distillation
state arrives packed as the DistillState built by setup_distillation and
is unpacked locally so the loop body reads exactly as it did inline.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from .checkpoint_adapter import _suppress_inactive_logits
from .losses import (
    _ohem_topk,
    compute_boundary_weights,
    deep_supervision_loss,
    dice_loss,
    focal_loss,
    lovasz_softmax_loss,
    tversky_loss,
)
from .train_config import TrainConfig
from .train_phase_utils import _CudaPrefetcher


def run_train_epoch(
    model,
    epoch: int,
    train_loader,
    optimizer,
    scaler,
    device: torch.device,
    is_cuda: bool,
    is_mps: bool,
    use_amp: bool,
    config: TrainConfig,
    num_classes: int,
    train_ids: list[str],
    inactive_ids: list[int],
    resolved_active_ids: list[int],
    class_weights_t,
    loss_type: str,
    ohem_ratio: float,
    dice_weight: float,
    accum_steps: int,
    max_grad_norm: float,
    distill_state,
    log_fn: Callable[[str], None],
    stop_flag: Callable[[], bool] | None,
) -> tuple[float, float]:
    """Run one epoch over *train_loader*; returns (avg_loss, avg_distill_loss)."""
    _is_cuda = is_cuda
    _is_mps = is_mps
    distill_projector = distill_state.distill_projector
    channel_projector = distill_state.channel_projector
    teacher_cache = distill_state.teacher_cache
    teacher_gap_cache = distill_state.teacher_gap_cache
    teacher_model_online = distill_state.teacher_model_online
    ensemble_logits_cache = distill_state.ensemble_logits_cache
    distill_on = distill_state.distill_on
    distill_spatial = distill_state.distill_spatial
    distill_channel = distill_state.distill_channel
    distill_ensemble = distill_state.distill_ensemble
    distill_online = distill_state.distill_online
    teacher_model_online2 = distill_state.teacher_model_online2
    distill_projector2 = distill_state.distill_projector2

    # Distillation helpers (cached in sys.modules after the first epoch)
    if distill_spatial:
        from .distill import apply_augmentation_to_features, feature_distillation_loss, get_teacher_batch
    if distill_channel:
        from .distill import channel_distillation_loss, get_teacher_batch_vec

    model.train()
    if distill_projector is not None:
        distill_projector.train()
    if channel_projector is not None:
        channel_projector.train()
    losses = []
    distill_losses: list[float] = []
    optimizer.zero_grad()
    # Wrap the DataLoader with a CUDA prefetcher when on GPU. The
    # prefetcher overlaps the next batch's H2D transfer with the
    # current step's compute on a secondary stream, eliminating the
    # per-step idle window where the GPU would otherwise wait for
    # pin_memory transfer to finish on the default stream.
    _epoch_loader = _CudaPrefetcher(train_loader, device) if _is_cuda else train_loader
    for step_idx, batch in enumerate(_epoch_loader):
        if stop_flag and stop_flag():
            break
        if distill_on:
            images, masks, meta = batch
            sample_weights = meta.get("sample_weight") if isinstance(meta, dict) else None
        else:
            images, masks, third = batch
            if isinstance(third, dict):
                # return_meta=True but distill fell back to off
                meta = third
                sw_val = meta.get("sample_weight")
                sample_weights = torch.tensor(sw_val, dtype=torch.float32) if sw_val is not None else None
            else:
                sample_weights = third
                meta = None
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        # Safety: clamp out-of-range mask values to ignore_index
        bad = (masks >= num_classes) & (masks != config.ignore_index)
        if bad.any():
            if step_idx == 0 and epoch == 1:
                bad_vals = masks[bad].unique().tolist()
                log_fn(
                    f"WARNING: mask has values {bad_vals} >= num_classes={num_classes}, "
                    f"clamping to ignore_index={config.ignore_index}\n"
                )
            masks = masks.clone()
            masks[bad] = config.ignore_index
        if inactive_ids:
            for cls_id in inactive_ids:
                masks = torch.where(masks == cls_id, torch.zeros_like(masks), masks)
        deep_sup_on = getattr(config, "deep_supervision", False)
        with torch.amp.autocast(device.type if _is_cuda else "cpu", enabled=use_amp):
            aux_logits: list[torch.Tensor] = []
            need_extras = distill_on or deep_sup_on
            if need_extras:
                result = model(images, return_features=True)
                if isinstance(result, tuple):
                    logits, extras = result
                    features = extras if distill_on else None
                    aux_logits = extras.get("aux_logits", []) if isinstance(extras, dict) else []
                else:
                    logits = result
                    features = None
                logits = _suppress_inactive_logits(logits, resolved_active_ids)
            else:
                logits = _suppress_inactive_logits(model(images), resolved_active_ids)
            bw = compute_boundary_weights(masks, ignore_index=config.ignore_index, boundary_weight=config.boundary_weight)
            if loss_type == "lovasz":
                loss_main = lovasz_softmax_loss(
                    logits, masks, num_classes, config.ignore_index,
                )
            elif loss_type == "focal":
                loss_main = focal_loss(
                    logits, masks,
                    weight=class_weights_t,
                    gamma=2.0,
                    ignore_index=config.ignore_index,
                    pixel_weights=bw,
                    ohem_ratio=ohem_ratio,
                )
            else:
                ce_per_pixel = nn.functional.cross_entropy(
                    logits, masks,
                    weight=class_weights_t,
                    ignore_index=config.ignore_index,
                    reduction="none",
                )
                ce_weighted = ce_per_pixel * bw
                valid = masks != config.ignore_index
                if ohem_ratio > 0.0:
                    loss_main = _ohem_topk(ce_weighted, valid, ohem_ratio)
                else:
                    loss_main = ce_weighted[valid].mean() if valid.any() else ce_weighted.mean()
            loss_dice = dice_loss(logits, masks, num_classes, config.ignore_index)
            loss = (loss_main + dice_weight * loss_dice) / accum_steps
            # Optional Tversky loss for FN-biased learning (micro-defect focus)
            if config.tversky_weight > 0:
                loss_tv = tversky_loss(
                    logits, masks, num_classes, config.ignore_index,
                    alpha=config.tversky_alpha, beta=config.tversky_beta,
                    gamma=config.tversky_gamma,
                )
                loss = loss + config.tversky_weight * loss_tv / accum_steps
            # Deep supervision: auxiliary loss from intermediate decoder stages
            if aux_logits:
                loss_ds = deep_supervision_loss(
                    aux_logits, masks, num_classes, config.ignore_index,
                )
                loss = loss + loss_ds / accum_steps
            # Pseudo-label sample weighting (lower loss for auto-generated labels)
            if sample_weights is not None:
                sw = sample_weights
                if _is_mps:
                    sw = sw.float()  # MPS doesn't support float64
                sw = sw.to(device, non_blocking=True)
                if sw.ndim == 0:
                    sw = sw.unsqueeze(0)
                loss = loss * sw.mean()

            # Feature distillation loss (spatial mode)
            if distill_spatial and distill_online and teacher_model_online is not None:
                # Online mode: run teacher on same input patches
                from .distill import online_teacher_features
                teacher_feat = online_teacher_features(
                    teacher_model_online, images, config.distill_feature_tap,
                )
                feat_loss = feature_distillation_loss(
                    features["e3"], teacher_feat,
                    distill_projector, config.distill_feature_loss,
                )
                loss = loss + config.distill_feature_weight * feat_loss / accum_steps
                distill_losses.append(feat_loss.item())
                # Dual-teacher: 2nd online teacher feature loss
                if teacher_model_online2 is not None and distill_projector2 is not None:
                    teacher_feat2 = online_teacher_features(
                        teacher_model_online2, images, config.distill_feature_tap,
                    )
                    feat_loss2 = feature_distillation_loss(
                        features["e3"], teacher_feat2,
                        distill_projector2, config.distill_feature_loss,
                    )
                    loss = loss + config.distill_teacher2_weight * feat_loss2 / accum_steps
            elif distill_spatial and meta is not None and teacher_cache is not None:
                # Cached mode: look up precomputed features
                stems = [train_ids[idx.item()] for idx in meta["stem_idx"]]
                teacher_feat = get_teacher_batch(teacher_cache, stems, device)
                if teacher_feat is not None:
                    teacher_feat = apply_augmentation_to_features(
                        teacher_feat, meta["hflip"], meta["vflip"], meta["rot90_k"],
                    )
                    feat_loss = feature_distillation_loss(
                        features["e3"], teacher_feat,
                        distill_projector, config.distill_feature_loss,
                    )
                    loss = loss + config.distill_feature_weight * feat_loss / accum_steps
                    distill_losses.append(feat_loss.item())

            # Channel distillation loss (GAP-based, spatially invariant)
            if distill_channel and meta is not None and teacher_gap_cache is not None:
                stems = [train_ids[idx.item()] for idx in meta["stem_idx"]]
                teacher_vec = get_teacher_batch_vec(teacher_gap_cache, stems, device)
                if teacher_vec is not None:
                    feat_loss = channel_distillation_loss(
                        features["e3"], teacher_vec,
                        channel_projector, config.distill_feature_loss,
                    )
                    loss = loss + config.distill_feature_weight * feat_loss / accum_steps
                    distill_losses.append(feat_loss.item())
                    # Debug log: shapes at first step of first epoch
                    if epoch == 1 and step_idx == 0:
                        student_gap_dbg = features["e3"].mean(dim=(2, 3))
                        log_fn(
                            f"Channel distill: student_e3={list(features['e3'].shape)}, "
                            f"gap={list(student_gap_dbg.shape)}, "
                            f"teacher_vec={list(teacher_vec.shape)}\n"
                        )
                else:
                    log_fn(f"WARNING: teacher cache miss for stems={stems}\n")

            # Ensemble logits distillation (KL divergence on cached multi-teacher logits)
            if distill_ensemble and meta is not None and ensemble_logits_cache is not None:
                from .distill import ensemble_logits_loss, get_ensemble_logits_batch
                stems = [train_ids[idx.item()] for idx in meta["stem_idx"]]
                # Get output spatial size for target matching
                out_h, out_w = logits.shape[2], logits.shape[3]
                teacher_logits = get_ensemble_logits_batch(
                    ensemble_logits_cache, stems,
                    crop_boxes=None,  # full-image mode; patches handled by resize
                    target_size=(out_h, out_w),
                    device=device,
                )
                if teacher_logits is not None:
                    # Apply geometric augmentations to cached logits
                    from .distill import apply_augmentation_to_features
                    teacher_logits = apply_augmentation_to_features(
                        teacher_logits, meta["hflip"], meta["vflip"], meta["rot90_k"],
                    )
                    ens_loss = ensemble_logits_loss(
                        logits, teacher_logits.float(),
                        temperature=config.distill_ensemble_temperature,
                        ignore_index=config.ignore_index,
                    )
                    loss = loss + config.distill_ensemble_weight * ens_loss / accum_steps
                    distill_losses.append(ens_loss.item())

        scaler.scale(loss).backward()
        losses.append(loss.item() * accum_steps)
        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
            clip_params = list(model.parameters())
            if distill_projector is not None:
                clip_params += list(distill_projector.parameters())
            if channel_projector is not None:
                clip_params += list(channel_projector.parameters())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_distill_loss = float(np.mean(distill_losses)) if distill_losses else 0.0
    return avg_loss, avg_distill_loss
