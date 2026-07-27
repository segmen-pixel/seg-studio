# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Plan DataLoader sizing from a measured host + dataset profile.

Goal: maximise GPU utilisation while staying inside the host's memory budget.

The planner is a pure function: same inputs → same outputs, no I/O.

Decision surface:

  cache_mode      one of "decoded", "bytes", "none"
                  "decoded" — full PIL Images held in main process. Cheapest
                              per-step (no decode), but memory-hungry. Only
                              works with workers=0 (PIL Image isn't picklable
                              cheaply across worker boundaries).
                  "bytes"   — raw file bytes held in memory, decoded per step
                              in workers (or main process). Good middle
                              ground: avoids disk I/O without paying the
                              decoded-pixel cost.
                  "none"    — stream from disk every step. Required when the
                              dataset is too big to fit in RAM.

  num_workers     0 or higher. On Windows, spawn cost is real (each worker
                  pickles dataset state and opens its own CUDA context if
                  pin_memory is set), so we only spend workers when there is
                  measurable benefit. We approximate "benefit" by the ratio
                  of estimated decode-time to estimated GPU-step-time: if
                  decode would stall the GPU, we add workers.

  prefetch_factor Number of batches each worker pre-stages. Buys latency hide
                  at the cost of more shared-memory commitment.

  persistent_workers  Whether workers stay alive between epochs. Worth it
                      when worker startup cost is non-trivial relative to
                      epoch length (which is essentially always, except very
                      short epochs).

Constants in this file are *fractions* (names ending in ``_FRAC``) or pure
physical sizes. There are no thresholds expressed in absolute bytes or
worker counts.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from .dataset_profile import DatasetProfile
from .host_probe import HostProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable fractions. Keep this list short — every entry is justified inline.
# ---------------------------------------------------------------------------

# Fraction of the per-process memory budget we are willing to commit to
# DataLoader machinery (workers + prefetch + cache combined). The rest is
# left for model state, gradients, optimizer, and OS overhead. 0.5 is a
# defensible split: training on a 16 GB box, half goes to data plumbing.
_DATALOADER_BUDGET_FRAC = 0.5

# Within the DataLoader budget, fraction we are willing to spend on the
# warm cache (vs prefetch / shared-memory queues). The cache benefit is
# upper-bounded by hit rate (= 1.0 for a fitted cache), so spending most of
# the budget here pays off when it fits.
_CACHE_BUDGET_FRAC = 0.7

# Safety margin against probe error and OS/driver overhead. We never plan
# right up to the measured budget. 0.85 gives us room for the OS file cache
# to grow, antivirus scans, etc., without OOMing.
_SAFETY_HEADROOM = 0.85

# Minimum prefetch_factor when workers > 0. PyTorch's default is 2; below
# this the workers can't double-buffer.
_MIN_PREFETCH = 2

# Maximum prefetch_factor — beyond this, additional buffers don't help
# (decode-bound steady state) but they do consume shared memory commitment.
_MAX_PREFETCH = 8


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

@dataclass
class ModelMeta:
    """What the planner needs to know about the training workload."""
    batch_size: int
    patch_size: int                # 0 means "use full image"
    input_h: int                   # full-image H (used when patch_size == 0)
    input_w: int                   # full-image W (used when patch_size == 0)
    channels: int = 3
    # Patches per image (annotation_patches mode). When > 1, each
    # __getitem__ call produces multiple patches from one stem (one
    # decoded image), so the worker producer rate should be measured in
    # stems/step = bs / patches_per_image, not in raw samples/step.
    # Without this correction, the planner over-provisions workers when
    # ppi >> 1.
    patches_per_image: int = 1
    # Estimate of GPU compute time per step in milliseconds. May be None on
    # the first call; later calls (re-plan) populate from observed steps.
    gpu_step_ms: float | None = None
    # Estimate of decode time per sample in ms. Same caveat.
    decode_ms_per_sample: float | None = None
    # Total samples per epoch (= num_train * patches_per_image, with the
    # min-samples floor from SegDataset.__len__). When provided, the
    # planner uses it to enforce step_cap = samples_per_epoch // bs —
    # never spawn more workers than steps per epoch (a worker that
    # never gets a turn is pure spawn overhead).
    samples_per_epoch: int | None = None


@dataclass
class DataLoaderPlan:
    num_workers: int
    prefetch_factor: int | None        # None when workers == 0
    persistent_workers: bool
    pin_memory: bool

    cache_mode: str                    # "decoded" | "bytes" | "none"
    cache_budget_bytes: int            # ceiling for the cache

    # The planner's RAM commitment for this trainer. Caller registers this
    # with ProcessRegistry so peers see it.
    claimed_ram_bytes: int

    # Reasoning trace — one human-readable line per decision. Logged but
    # not parsed.
    reasoning: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_dataloader(host: HostProfile,
                    dataset: DatasetProfile,
                    model: ModelMeta,
                    peers_ram_bytes: int = 0,
                    is_cuda: bool = True) -> DataLoaderPlan:
    """Compute a DataLoaderPlan from measured inputs.

    Args:
        host: result of host_probe.probe_host()
        dataset: result of dataset_profile.probe_dataset()
        model: workload description
        peers_ram_bytes: total RAM already claimed by other training processes
            on this machine (from ProcessRegistry.others_total_ram())
        is_cuda: whether the trainer is using a CUDA device. Pin-memory and
            CUDA-context costs only apply when this is True.
    """
    reasoning: list[str] = []

    # --- 1. Memory budget --------------------------------------------------
    raw_budget = max(0, host.memory_budget - peers_ram_bytes)
    budget = int(raw_budget * _SAFETY_HEADROOM)
    dataloader_budget = int(budget * _DATALOADER_BUDGET_FRAC)
    cache_budget = int(dataloader_budget * _CACHE_BUDGET_FRAC)
    # runtime_budget (for workers + prefetch) is computed below after we
    # know the actual cache_used — it's `dataloader_budget - cache_used`,
    # not `- cache_budget` (which is the cache *cap*, not its usage).

    reasoning.append(
        f"budget: host_memory={_h(host.memory_budget)} peers={_h(peers_ram_bytes)} "
        f"safety x{_SAFETY_HEADROOM:.2f} -> {_h(budget)} "
        f"(dataloader={_h(dataloader_budget)}, cache_cap={_h(cache_budget)})"
    )

    # --- 2. Cache mode -----------------------------------------------------
    # We try the cheapest cache that fits: decoded > bytes > none, in terms
    # of per-step performance. We pick the most expensive (in memory) that
    # fits into cache_budget.
    bytes_required_for_decoded = dataset.total_decoded_image_bytes_estimate(
        channels=model.channels, dtype_bytes=1
    ) + dataset.total_mask_bytes_estimate()  # masks are 1 byte/px after decode but stored compressed; rough
    bytes_required_for_bytes = dataset.total_image_bytes_estimate() + dataset.total_mask_bytes_estimate()

    # Fast path: only safe when a single CPU thread can keep up with the
    # GPU's data appetite. The dimensionally-correct measure of "thread
    # parallelism budget" is the number of physical cores: with bs >>
    # cores, single-thread batch construction (PIL crop + aug + tensorise
    # per sample, sequential) cannot match a multi-core CPU's potential
    # throughput, so workers=0 strands the GPU waiting on the main
    # thread.
    #
    # Threshold: bs <= cpu_cores. This is a workload-vs-hardware
    # comparison, not a tuned constant — the same condition holds whether
    # the user is on a 4-core laptop (workers > 0 from bs=5) or a 32-core
    # workstation (workers > 0 from bs=33).
    #
    # Empirically observed: bs=332 with workers=0 on 16 cores gave GPU
    # util 0-21% because the main thread couldn't construct batches fast
    # enough. Switching to workers>0 + cache=none (Windows-safe) lets the
    # planner spend the saved cache space on prefetch buffers instead.
    # stems_per_step is the right unit for DECODE work (one decode per
    # stem). But augmentation + tensor conversion is per-PATCH and
    # happens regardless of cache mode (decoded cache only avoids the
    # decode step). For cache=decoded, per-step CPU = bs × per_patch_aug,
    # which is the dominant cost when ppi >> 1.
    #
    # Empirically (A5000 x8 box, bs=192, ppi=16, workers=0,
    # cache=decoded): stems_per_step=12 << 48 cores tempted the planner
    # into fast-path, but GPU util stuck at 0-5% because main-thread
    # 192 × ~5ms aug = ~1s/step >> ~300ms GPU step.
    #
    # Use bs as the workload measure for the gate: it represents the
    # actual per-thread workload regardless of cache mode.
    _ppi = max(1, getattr(model, "patches_per_image", 1))
    _stems_per_step = max(1, model.batch_size // _ppi)
    fast_path_ok = (bytes_required_for_decoded > 0
                    and bytes_required_for_decoded <= cache_budget
                    and model.batch_size <= host.cpu_cores_physical)
    if not fast_path_ok and bytes_required_for_decoded > 0 and bytes_required_for_decoded <= cache_budget:
        reasoning.append(
            f"fast-path REJECTED: bs={model.batch_size} > cpu_cores={host.cpu_cores_physical} "
            f"-> single-thread CPU (per-patch aug) cannot feed GPU; "
            f"falling through to workers>0 path"
        )

    if fast_path_ok:
        reasoning.append(
            f"fast-path: decoded cache fits ({_h(bytes_required_for_decoded)} <= "
            f"{_h(cache_budget)}) AND bs={model.batch_size} <= cpu_cores="
            f"{host.cpu_cores_physical} -> workers=0, cache=decoded"
        )
        return DataLoaderPlan(
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False,
            pin_memory=is_cuda,
            cache_mode="decoded",
            cache_budget_bytes=bytes_required_for_decoded,
            claimed_ram_bytes=int(bytes_required_for_decoded),
            reasoning=reasoning,
        )

    cache_mode: str
    cache_used: int
    if bytes_required_for_decoded > 0 and bytes_required_for_decoded <= cache_budget:
        cache_mode = "decoded"
        cache_used = bytes_required_for_decoded
        reasoning.append(
            f"cache: decoded ({_h(cache_used)} fits in {_h(cache_budget)})"
        )
    elif bytes_required_for_bytes > 0 and bytes_required_for_bytes <= cache_budget:
        cache_mode = "bytes"
        cache_used = bytes_required_for_bytes
        reasoning.append(
            f"cache: bytes ({_h(cache_used)} fits in {_h(cache_budget)}; "
            f"decoded would need {_h(bytes_required_for_decoded)})"
        )
    else:
        cache_mode = "none"
        cache_used = 0
        reasoning.append(
            f"cache: none (dataset is {_h(bytes_required_for_bytes)} on disk, "
            f"{_h(bytes_required_for_decoded)} decoded; cache_cap={_h(cache_budget)})"
        )

    # --- 3. Per-sample cost estimates -------------------------------------
    # The relevant cost for budget-capping workers is the *steady-state*
    # memory each in-flight sample occupies in the prefetch queue, not the
    # transient decode buffer. Each worker's __getitem__ briefly allocates
    # a decoded RGB array (~pixels × 3 bytes) but converts it to a tensor
    # and releases the array immediately; only the tensor enters the
    # IPC/shared-memory queue. So the queue cost per sample is the
    # tensor footprint, which is patch_size² × channels × 4 (float32).
    #
    # The transient decode buffer still matters for *peak* worker memory
    # but it's amortised across the worker's processing rate, not
    # multiplied by prefetch_factor × batch_size like the queue is.
    if model.patch_size > 0:
        sample_tensor_bytes = model.patch_size * model.patch_size * model.channels * 4
    else:
        sample_tensor_bytes = model.input_h * model.input_w * model.channels * 4
    decode_bytes = dataset.max_decoded_image_bytes(channels=model.channels, dtype_bytes=1)
    # Per-worker peak (transient): decode + one tensor in flight at any time
    per_worker_peak_transient = decode_bytes + sample_tensor_bytes
    # Per-queue-slot cost (steady): just the tensor
    per_sample_cost = sample_tensor_bytes

    # --- 4. Workers --------------------------------------------------------
    # Each worker carries: (a) IPC queue holding `prefetch_factor × bs`
    # tensors; (b) one transient decode buffer in flight; (c) a flat
    # spawn overhead (Python interp, torch, dataset state pickled in).
    # We bundle (b) and (c) together as `spawn_per_worker_overhead`.
    spawn_per_worker_overhead = per_worker_peak_transient

    # runtime_budget = whatever we don't spend on the cache. Use the
    # actual cache_used (which may be 0 if cache_mode=='none' or much
    # smaller than cache_budget if the dataset is small).
    runtime_budget = max(0, dataloader_budget - cache_used)

    # Decide the desired worker count from the speed standpoint, then clamp
    # by the budget.
    desired_workers = _desired_workers_from_throughput(host, dataset, model, reasoning)

    # Budget-driven cap.
    # Assume prefetch_factor=2 for the budget calculation (we'll refine
    # below). We must fit:
    #   workers × (prefetch × batch_size × per_sample_cost + overhead)
    # within runtime_budget.
    if desired_workers == 0:
        budget_cap_workers = 0
    else:
        cost_per_worker = (
            _MIN_PREFETCH * model.batch_size * per_sample_cost
            + spawn_per_worker_overhead
        )
        if cost_per_worker <= 0:
            budget_cap_workers = desired_workers
        else:
            budget_cap_workers = max(0, runtime_budget // cost_per_worker)

    # CPU cap. We never want workers to outnumber physical cores: oversub
    # just trashes the cache.
    cpu_cap = max(0, host.cpu_cores_physical - 1) if desired_workers > 0 else 0

    # Step cap: never spawn more workers than steps per epoch. A worker
    # that never gets a __getitem__ call is pure spawn-time overhead.
    # samples_per_epoch may be None on the first call (legacy callers),
    # in which case we skip this cap.
    if model.samples_per_epoch is not None and model.samples_per_epoch > 0:
        step_cap = max(1, model.samples_per_epoch // max(1, model.batch_size))
    else:
        step_cap = desired_workers if desired_workers > 0 else 1

    num_workers = int(min(desired_workers, budget_cap_workers, cpu_cap, step_cap))
    num_workers = max(0, num_workers)

    reasoning.append(
        f"workers: desired={desired_workers} budget_cap={budget_cap_workers} "
        f"cpu_cap={cpu_cap} step_cap={step_cap} -> {num_workers} "
        f"(per_sample={_h(per_sample_cost)}, runtime_budget={_h(runtime_budget)})"
    )

    # When workers > 0, the cache interacts with worker startup very
    # differently per OS:
    #   * Linux (fork): the main-process cache is shared copy-on-write.
    #     bytes-mode is essentially free; decoded-mode would still be free
    #     in principle but PIL.Image lazy decoders break under fork, so we
    #     downgrade to bytes.
    #   * Windows / macOS (spawn): each worker re-pickles the dataset
    #     including the cache, so we'd pay (1 + num_workers) x cache_size.
    #     This is what blew past the Windows commit limit in the previous
    #     attempt. Force cache = none in that case.
    if num_workers > 0:
        if host.os_family == "linux":
            if cache_mode == "decoded":
                if bytes_required_for_bytes > 0 and bytes_required_for_bytes <= cache_budget:
                    cache_mode = "bytes"
                    cache_used = bytes_required_for_bytes
                    reasoning.append(
                        "cache: decoded -> bytes (workers>0 on Linux uses COW bytes cache)"
                    )
                else:
                    cache_mode = "none"
                    cache_used = 0
                    reasoning.append(
                        "cache: -> none (workers>0 + decoded incompatible; bytes too large)"
                    )
        else:
            # Windows / macOS: spawn duplicates the cache per worker.
            if cache_mode != "none":
                old = cache_mode
                cache_mode = "none"
                cache_used = 0
                reasoning.append(
                    f"cache: {old} -> none (workers>0 on {host.os_family} would duplicate "
                    f"the cache per worker via spawn)"
                )

    # --- 5. Prefetch factor ------------------------------------------------
    # Pick the smallest prefetch that hides decode latency.
    if num_workers == 0:
        prefetch_factor: int | None = None
    else:
        prefetch_factor = _pick_prefetch(host, dataset, model, num_workers, reasoning)
        # Final budget check now that we know prefetch.
        worker_cost = (
            num_workers * (
                prefetch_factor * model.batch_size * per_sample_cost
                + spawn_per_worker_overhead
            )
        )
        if worker_cost > runtime_budget and prefetch_factor > _MIN_PREFETCH:
            old_pf = prefetch_factor
            prefetch_factor = _MIN_PREFETCH
            reasoning.append(
                f"prefetch: trimmed {old_pf} -> {prefetch_factor} to fit runtime_budget"
            )

    # --- 6. persistent_workers --------------------------------------------
    # Only meaningful if workers > 0. We turn it off on Windows historically
    # because spawn-pickled state has caused deadlocks; here we keep it on
    # if workers > 0 since the spawn cost is what we are amortising.
    persistent_workers = num_workers > 0

    # --- 7. Final RAM claim ------------------------------------------------
    if num_workers == 0:
        claimed = cache_used  # main-process cache only
    else:
        claimed = cache_used + num_workers * (
            (prefetch_factor or _MIN_PREFETCH) * model.batch_size * per_sample_cost
            + spawn_per_worker_overhead
        )

    reasoning.append(
        f"claim: cache={_h(cache_used)} + workers={num_workers}x"
        f"(pf={prefetch_factor}xbs={model.batch_size}x{_h(per_sample_cost)}) "
        f"= {_h(claimed)}"
    )

    return DataLoaderPlan(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=is_cuda,
        cache_mode=cache_mode,
        cache_budget_bytes=cache_used,
        claimed_ram_bytes=int(claimed),
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Sub-decisions
# ---------------------------------------------------------------------------

def _desired_workers_from_throughput(host: HostProfile,
                                     dataset: DatasetProfile,
                                     model: ModelMeta,
                                     reasoning: list[str]) -> int:
    """How many workers do we *want* before we apply the budget cap?

    Logic:
      - If we have measured both decode-time and gpu-step-time (re-plan),
        workers = ceil(decode_total / gpu_step_total).
      - Otherwise, estimate decode-time from storage_read_bps and image
        bytes; if storage is unmeasured, fall back to one worker per
        decoded GiB-per-second of bandwidth × physical core.
      - On Windows, we apply an additional minimum of 0 because the spawn
        cost is high enough that we'd rather use main-process caching when
        the dataset is small.
    """
    # Path A: measured timings.
    if model.gpu_step_ms is not None and model.decode_ms_per_sample is not None:
        decode_per_step = model.decode_ms_per_sample * model.batch_size
        if model.gpu_step_ms <= 0:
            stems_per_step = 1.0
        else:
            stems_per_step = max(1.0, decode_per_step / model.gpu_step_ms)
        desired = int(math.ceil(stems_per_step))
        reasoning.append(
            f"workers/desired: measured decode={model.decode_ms_per_sample:.1f}ms/sample "
            f"gpu_step={model.gpu_step_ms:.1f}ms -> ceil({stems_per_step:.2f})={desired}"
        )
        return desired

    # Path B: estimated from storage + image size.
    #   decode_time ≈ image_bytes / storage_bps + decode_cpu_time
    # We don't have decode_cpu_time without measuring; assume it's ~50% of
    # I/O time (rule of thumb for JPEG: throughput is bounded by both, and
    # typical ratio for SSDs vs CPU JPEG decode is roughly 1:1).
    if host.storage_read_bps is not None and host.storage_read_bps > 0:
        # Time to read one image:
        sec_per_image_io = dataset.image_bytes_mean / host.storage_read_bps
        # Total CPU time per image: ~1.5× I/O is a rough estimate.
        sec_per_image = sec_per_image_io * 1.5
        # Per-step decode time (sequential):
        sec_per_step_decode = sec_per_image * model.batch_size
        # GPU step: we don't know yet; assume image-size-proportional.
        # Use a baseline that makes the worker count come out to ~CPU count
        # for typical data, then let the budget cap take over.
        # We treat "we want workers s.t. decode ≤ small fraction of step" as
        # equivalent to "we want at most one worker per physical core minus
        # one (reserved for main / pin-memory)".
        max_useful = max(1, host.cpu_cores_physical - 1)
        # Bias by storage class: NVMe (>1 GB/s) → fewer workers needed.
        if host.storage_read_bps > 1e9:
            desired = max(1, max_useful // 2)
        elif host.storage_read_bps > 2e8:  # ~200 MB/s, SSD
            desired = max(1, int(max_useful * 0.75))
        else:  # HDD
            desired = max_useful
        reasoning.append(
            f"workers/desired: storage_bps={_h(host.storage_read_bps)}/s "
            f"image_mean={_h(dataset.image_bytes_mean)} → {desired} "
            f"(io_per_step≈{sec_per_step_decode*1000:.0f}ms)"
        )
        return desired

    # Path C: no timings, no storage benchmark. Heuristic from workload
    # vs hardware: per-step CPU work = bs × per_patch_aug (always done,
    # cache only avoids decode). If bs > cores, parallelism is needed.
    #
    # Workers needed: ceil(bs / cores) capped at cores-1.
    if model.batch_size > host.cpu_cores_physical:
        needed = (model.batch_size + host.cpu_cores_physical - 1) // host.cpu_cores_physical
        desired = max(1, min(host.cpu_cores_physical - 1, needed))
        reasoning.append(
            f"workers/desired: {desired} (bs={model.batch_size} > cpu_cores="
            f"{host.cpu_cores_physical}, need ceil(bs/cores)={needed} workers, "
            f"cap cores-1)"
        )
        return desired
    # Small workload: spawn overhead may exceed parallelism gain.
    if host.os_family == "windows":
        reasoning.append(
            f"workers/desired: 0 (Windows, bs={model.batch_size} <= cores)"
        )
        return 0
    desired = max(1, min(host.cpu_cores_physical - 1, 4))
    reasoning.append(f"workers/desired: {desired} (no timings, default)")
    return desired


def _pick_prefetch(host: HostProfile,
                   dataset: DatasetProfile,
                   model: ModelMeta,
                   num_workers: int,
                   reasoning: list[str]) -> int:
    """Choose prefetch_factor.

    A larger prefetch hides decode-step variance (some images are slower to
    decode than others). The benefit saturates at the point where the queue
    smooths over the slowest expected decode. Cost: shared memory commit.

    Heuristic: we look at the ratio max_image_bytes / mean_image_bytes. If
    it's large (skewed distribution), we want more prefetch buffers. If
    everything's the same size, _MIN_PREFETCH is fine.
    """
    if dataset.image_bytes_mean <= 0:
        return _MIN_PREFETCH
    skew = dataset.image_bytes_max / dataset.image_bytes_mean
    if skew <= 1.5:
        pf = _MIN_PREFETCH
    elif skew <= 3.0:
        pf = 3
    elif skew <= 6.0:
        pf = 4
    else:
        pf = min(_MAX_PREFETCH, int(math.ceil(skew / 2)))
    reasoning.append(
        f"prefetch: image_size_skew={skew:.2f}× → {pf}"
    )
    return pf


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _h(n: float | int) -> str:
    """Human-readable byte count."""
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(n)}B"
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"
