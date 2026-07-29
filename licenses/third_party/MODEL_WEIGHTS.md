# Model Weights — Provenance, License & Attribution

Model weights and the source code that produces them are separate works, and a
repository's `LICENSE` does not automatically cover the checkpoints published
alongside it. This file therefore records, per checkpoint, where the exact file
comes from, how it is pinned, and what the licence claim actually rests on.

## What this project redistributes, and what it does not

**Redistributed by this project** — mirrored at
[`segmen-pixel/seg-studio`](https://huggingface.co/segmen-pixel/seg-studio) on
HuggingFace and bundled by the Windows installer. These are the files for which
we need redistribution rights, and they are the subject of the table below.

**Fetched at run time from the vendor, not redistributed by us** — the
RF-DETR-Seg checkpoints used by instance-segmentation training. `rfdetr`
downloads them from Roboflow's own storage onto the user's machine on first
use; this project neither mirrors nor bundles them. Roboflow distributes those
files directly to the user under its own terms. We record them here for
transparency, not because we redistribute them.

Roboflow states the licence for these checkpoints directly: "All RF-DETR-Seg
checkpoints released in this update are available under the Apache 2.0
license", and separately that "The XLarge and 2XLarge **detection** models are
based on DINOv3 and are released under the Platform Model License 1.0"
(<https://blog.roboflow.com/rf-detr-segmentation/>, retrieved 2026-07-24). The
restricted licence therefore covers detection-only variants; the segmentation
checkpoints are Apache-2.0 at every size. This project uses the segmentation
family and offers Small, Medium and Large.

## Verification status

| Checked | Status |
|---|---|
| Upstream `LICENSE` retrieved and read for every redistributed checkpoint | ✅ 2026-07-24 — all standard, unmodified Apache-2.0, no non-commercial or field-of-use terms |
| SHA-256 pinned for every redistributed checkpoint | ✅ `SAM_CHECKPOINTS` in `scripts/build_installer.py`, `_SAM_SHA256` in `apps/trainer_api/app/core/sam_assist.py` — identical values, verified on every download from any mirror |
| Source URLs pinned to an immutable reference | ✅ MobileSAM to a commit SHA, SAM 2 to a dated release path, others to fixed HF paths |
| Upstream licence text explicitly names *weights* | ⚠️ **No.** None of these upstream licences mention checkpoints or weights specifically — they are repository licences under which the authors also publish the checkpoints. This is the normal state of the field, and it is the basis on which the claims below rest. It is stated plainly rather than presented as certainty. |
| Training-data terms reviewed per checkpoint | ⚠️ **Not done.** Upstream does not publish per-checkpoint training-data terms for these models. Anyone with a stricter requirement — for example a downstream user who must clear dataset provenance — should treat this as an open item. |

Each redistributed checkpoint is reproduced with its upstream copyright notice
preserved as required by Apache-2.0 §4(a)–(c). The full licence text is in the
root `LICENSE`.

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
| Pinned source | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` (dated release path) |
| SHA-256 | tiny `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`<br>small `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` |
| Licence basis | Upstream repository LICENSE, retrieved 2026-07-24: standard Apache-2.0. Does not name weights explicitly. |

## MobileSAM

| File | `mobile_sam.pt` |
|---|---|
| Upstream | https://github.com/ChaoningZhang/MobileSAM |
| Copyright | Copyright (c) 2023 Chaoning Zhang and contributors |
| License | Apache-2.0 |
| LICENSE source | https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE |
| Pinned source | `https://github.com/ChaoningZhang/MobileSAM/raw/b01a9ccef3b9e10b099b544efe004d0871802c3b/weights/mobile_sam.pt` (commit-pinned) |
| SHA-256 | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |
| Licence basis | Upstream repository LICENSE, retrieved 2026-07-24: standard Apache-2.0. Does not name weights explicitly. |

## TinySAM

| File | `tinysam.pth` |
|---|---|
| Upstream | https://github.com/xinghaochen/TinySAM |
| Copyright | Copyright (c) 2024 Huawei Noah's Ark Lab |
| License | Apache-2.0 |
| LICENSE source | https://github.com/xinghaochen/TinySAM/blob/main/LICENSE |
| Pinned source | `https://huggingface.co/xinghaochen/tinysam/resolve/main/tinysam.pth` (authors' own HF repo) |
| SHA-256 | `4b8edcf93af46e2a658ae455574de62873778a5cc3fd8e8adf094dcdfa957cf2` |
| Licence basis | Upstream repository LICENSE, retrieved 2026-07-24: standard Apache-2.0. Does not name weights explicitly. |

## EfficientSAM

| File | `efficient_sam_vitt.pt` |
|---|---|
| Upstream | https://github.com/yformer/EfficientSAM |
| Copyright | Copyright (c) Yunyang Xiong et al. |
| License | Apache-2.0 |
| LICENSE source | https://github.com/yformer/EfficientSAM/blob/main/LICENSE |
| Pinned source | `https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt` |
| SHA-256 | `dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a` |
| Licence basis | Upstream repository LICENSE, retrieved 2026-07-24: standard Apache-2.0. Does not name weights explicitly. |

## DINOv2 (feature distillation teacher)

| Files | `dinov2_vitb14_pretrain.pth` (weights only) |
|---|---|
| Upstream | https://github.com/facebookresearch/dinov2 |
| Copyright | Copyright (c) Meta Platforms, Inc. and affiliates |
| Weights license | Apache-2.0 |
| LICENSE source | https://github.com/facebookresearch/dinov2/blob/main/LICENSE |
| Pretrained on | ImageNet-22k + LVD-142M (curated by Meta) |
| Pinned source revision | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (audited 2026-07-22) |

> ⚠️ **Source tree note.** Only the pretrained weight file is redistributed
> with Seg-Studio. The DINOv2 torch-hub source tree is **not** bundled
> because recent versions mix Apache-2.0 with non-commercial fragments
> (`LICENSE_CELL_DINO_CODE`: CC-BY-NC-4.0 and `LICENSE_XRAY_DINO_MODEL`:
> FAIR Noncommercial), which cannot be re-shipped under Apache-2.0.
> The model-definition Python files are fetched at runtime via
> `torch.hub.load('facebookresearch/dinov2:<pinned SHA>', ...)` on the
> user's machine, staying outside Seg-Studio's redistribution surface.
> The revision is pinned in code (`_DINOV2_HUB_REF`) so the executed
> source cannot drift from the audited state; bumps require re-audit.

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
