# Blackwell (RTX 5090 / sm_120) Migration Guide

_Drafted 2026-05-11. **Superseded 2026-07-23 — the migration is complete and
this document is kept for historical reference only.**_

## Status: no migration needed

Blackwell is supported by the default install path. Nothing on this page needs
to be followed any more.

- `apps/trainer_api/requirements.txt` pins the `torch==2.13.*` family, which
  ships Blackwell (sm_120) kernels.
- The Windows installer and `install_windows.bat` add the CUDA 12.8 wheel index
  at install time (`--extra-index-url https://download.pytorch.org/whl/cu128`),
  so CUDA-specific wheels are selected without a separate pin set. Pass
  `install_windows.bat cuda124` for pre-Turing GPUs (Maxwell / Pascal / Volta).
- One lockfile now covers Pascal sm_60 through Blackwell sm_120. There is no
  parallel venv and no parallel lockfile.

## What was here before

Between 2026-05 and 2026-07 the main lockfile pinned `torch==2.6.0` (CUDA 12.4,
`arch_list` up to sm_90), which fails on a Blackwell device with:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

The workaround was a parallel virtual environment (`.venv-windows-cu128`) fed by
a dedicated lockfile `apps/trainer_api/requirements-cu128.in` / `.txt` that
pinned `torch==2.11.0+cu128`. The main `.venv-windows` stayed on cu124 for
production stability.

## Why the parallel lockfile was removed (2026-07-23)

Once the main lockfile moved to a torch build carrying sm_120 kernels, the
parallel lockfile was redundant — and it had become actively harmful:

- **It could not be verified by CI.** `lockfile-drift.yml` excluded it by
  design, because it was compiled on Windows against the CUDA extra index with a
  hand-added `pywin32` marker and was not reproducible on the Linux runner.
- **It silently fell behind on security patches.** By 2026-07-23 an
  `osv-scanner` sweep attributed 1 Critical and 19 High advisories to this file
  alone — `onnx` 1.20.1 (CVSS 9.1) against 1.22.0 in the main lockfile, plus
  stale `transformers`, `torch`, `setuptools` and `idna` pins. The main lockfile
  was clean.

**Do not reintroduce a parallel lockfile.** To move the CUDA target, bump the
torch pin in `requirements.in` and recompile, so the single lockfile stays under
drift CI and the license gate.
