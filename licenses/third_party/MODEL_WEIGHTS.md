# Redistributed Model Weights — License & Attribution

The Seg-Studio installer and the HuggingFace mirror at
[`segmen-pixel/seg-studio`](https://huggingface.co/segmen-pixel/seg-studio)
redistribute the following pre-trained model weights from third-party upstream
repositories. Each weight is licensed under **Apache License 2.0** (see the root
`LICENSE` file for the full text) and is reproduced with the upstream copyright
notice preserved as required by §4(a)–(c).

---

## SAM (Segment Anything Model)

| File | `sam_vit_*.pth` (not currently bundled) |
|---|---|
| Upstream | https://github.com/facebookresearch/segment-anything |
| Copyright | Copyright (c) Meta Platforms, Inc. and affiliates |
| License | Apache-2.0 |
| LICENSE source | https://github.com/facebookresearch/segment-anything/blob/main/LICENSE |

## SAM 2

| Files | `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt` |
|---|---|
| Upstream | https://github.com/facebookresearch/sam2 |
| Copyright | Copyright (c) Meta Platforms, Inc. and affiliates |
| License | Apache-2.0 |
| LICENSE source | https://github.com/facebookresearch/sam2/blob/main/LICENSE |

## MobileSAM

| File | `mobile_sam.pt` |
|---|---|
| Upstream | https://github.com/ChaoningZhang/MobileSAM |
| Copyright | Copyright (c) 2023 Chaoning Zhang and contributors |
| License | Apache-2.0 |
| LICENSE source | https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE |

## TinySAM

| File | `tinysam.pth` |
|---|---|
| Upstream | https://github.com/xinghaochen/TinySAM |
| Copyright | Copyright (c) 2024 Huawei Noah's Ark Lab |
| License | Apache-2.0 |
| LICENSE source | https://github.com/xinghaochen/TinySAM/blob/main/LICENSE |

## EfficientSAM

| File | `efficient_sam_vitt.pt` |
|---|---|
| Upstream | https://github.com/yformer/EfficientSAM |
| Copyright | Copyright (c) Yunyang Xiong et al. |
| License | Apache-2.0 |
| LICENSE source | https://github.com/yformer/EfficientSAM/blob/main/LICENSE |

## DINOv2 (feature distillation teacher)

| Files | `dinov2_vitb14_pretrain.pth` (weights only) |
|---|---|
| Upstream | https://github.com/facebookresearch/dinov2 |
| Copyright | Copyright (c) Meta Platforms, Inc. and affiliates |
| Weights license | Apache-2.0 |
| LICENSE source | https://github.com/facebookresearch/dinov2/blob/main/LICENSE |
| Pretrained on | ImageNet-22k + LVD-142M (curated by Meta) |

> ⚠️ **Source tree note.** Only the pretrained weight file is redistributed
> with Seg-Studio. The DINOv2 torch-hub source tree
> (`facebookresearch_dinov2_main/`) is **not** bundled because recent
> versions mix Apache-2.0 with non-commercial fragments
> (`LICENSE_CELL_DINO_CODE`: CC-BY-NC-4.0 and `LICENSE_XRAY_DINO_MODEL`:
> FAIR Noncommercial), which cannot be re-shipped under Apache-2.0.
> The model-definition Python files are fetched at runtime via
> `torch.hub.load('facebookresearch/dinov2', ...)` on the user's machine,
> staying outside Seg-Studio's redistribution surface.

---

## License compliance summary

All redistributed weights above are governed by the same Apache License 2.0 that
applies to Seg-Studio itself. The required obligations are met as follows:

- **§4(a) Recipients receive a copy of the license** — satisfied by the root
  `LICENSE` file shipped alongside every distribution.
- **§4(c) Copyright/attribution notices preserved** — listed in this file and
  in `THIRD_PARTY_NOTICES.md`.
- **§4(d) NOTICE attributions preserved** — none of the upstream repositories
  publish a separate `NOTICE` file (verified 2026-04-28). If any upstream adds
  one in future, this project will incorporate it on the next refresh.

The model weights themselves are reproduced verbatim as binary "Object form"
artifacts; no internal modification has been performed.
