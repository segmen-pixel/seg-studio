# Contributing to Seg-Studio

Thank you for your interest in contributing! This guide covers everything you
need to get started — from environment setup to submitting your first PR.

---

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style](#code-style)
- [Architecture Decisions](#architecture-decisions)
- [License](#license)

---

## Reporting Bugs

Open a [GitHub Issue](../../issues/new?template=bug_report.md) with:

- A clear title and description
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, Node version, GPU model)

Check existing issues first to avoid duplicates.

## Suggesting Features

Open a [Feature Request](../../issues/new?template=feature_request.md) describing
the use case, proposed solution, and any alternatives you considered.

---

## Development Setup

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ (3.11 recommended) | `from __future__ import annotations` required |
| Node.js | 18+ | For the React UI |
| npm | 9+ | |
| Git | 2.30+ | |
| NVIDIA GPU | Optional | CUDA 12.8 (or 12.4 on older GPUs) for GPU training; CPU-only works fine for UI/API dev |
| Apple Silicon | Optional | MPS acceleration on macOS (M1/M2/M3/M4) |

### Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/segmen-pixel/seg-studio.git
cd seg-studio

# 2. Python environment
# Windows:
python -m venv .venv-windows
.venv-windows\Scripts\activate
# macOS:
python3 -m venv .venv-macos
source .venv-macos/bin/activate
# Linux:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies (pinned via lockfile)
# `requirements.txt` is a lockfile generated from `requirements.in` by
# `uv pip compile`. See "Dependency Policy" below before adding/upgrading
# packages.
pip install -r apps/trainer_api/requirements.txt

# (Optional) Editable install of the training core (nested package layout)
# pip install -e packages/segcore

# (Optional, contributors) Install dev/test dependencies
# pip install -r apps/trainer_api/requirements-dev.txt

# (Optional) GPU support — overrides the CPU torch wheel from the lockfile.
# This is dev-only; the lockfile still records the canonical CPU pin.
# Windows/Linux — install PyTorch with CUDA (cu124 for Maxwell/Pascal/Volta):
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
# macOS — default PyPI wheels include MPS support:
# pip install torch torchvision

# 4. Install UI dependencies and build
cd apps/trainer_ui
npm install
npm run build
cd ../..

# 5. Start the API server
python -m uvicorn apps.trainer_api.app.main:app --host 127.0.0.1 --port 8002

# 6. Open http://localhost:8002/ui/ in your browser
```

### Windows Shortcut

```batch
scripts\windows\install_windows.bat cuda
scripts\windows\start_local_windows.bat
```

### macOS Shortcut

```bash
bash scripts/macos/install_macos.sh
bash scripts/macos/start_local_macos.sh
```

### Verify Your Setup

```bash
# Run the test suite (venv must be activated)
bash scripts/test.sh          # Linux/macOS/WSL
scripts\test.bat              # Windows (cmd)
```

### Pre-commit Hook (recommended)

The repo ships a pre-commit hook that blocks staged changes which would
re-introduce non-commercial-licensed vendor references. Enable it once per
clone:

```bash
git config core.hooksPath scripts/dev-hooks
```

(If you work from a superproject that contains this repo as a `seg-studio/`
subdirectory, use `git config core.hooksPath seg-studio/scripts/dev-hooks`
instead — see the header of `scripts/dev-hooks/pre-commit`.)

---

## Project Structure

```
seg-studio/
├── apps/
│   ├── trainer_api/        # FastAPI backend (Python)
│   │   ├── app/
│   │   │   ├── main.py     # Entry point, lazy-loading startup
│   │   │   ├── core/       # Config, DB, utilities
│   │   │   └── routers/    # API endpoints
│   │   └── tests/          # API unit tests (pytest)
│   ├── trainer_ui/         # React frontend (TypeScript)
│   │   ├── src/
│   │   │   ├── annotate/   # Annotation tools (brush, SAM, wand, etc.)
│   │   │   ├── training/   # Training tab
│   │   │   ├── results/    # Results & heatmap tab
│   │   │   └── store.ts    # Zustand state management
│   │   └── e2e/            # Playwright E2E tests
│   └── serving_api/        # ONNX inference server
├── packages/
│   ├── segcore/            # Training core library (installable: pip install -e packages/segcore)
│   │   └── segcore/        # Nested package directory
│   │       └── training/
│   │           ├── model.py    # SimpleUNet + model registry
│   │           ├── train.py    # Training loop, auto-tuning
│   │           └── dataset.py  # Patch sampling, augmentation
│   └── seg-sdk/            # Client SDK
├── scripts/
│   ├── cli_train.py        # CLI training interface
│   ├── mcp_server.py       # MCP tool bridge (37 tools)
│   ├── test.sh / test.bat  # Unified test runners
│   ├── windows/            # Windows setup/start scripts
│   └── macos/              # macOS setup/start scripts
├── docs/                   # Documentation (EN + ja/)
├── tests/                  # segcore unit tests (pytest)
├── pyproject.toml          # Python project config
└── README.md
```

### Which directory should I edit?

| I want to... | Edit |
|--------------|------|
| Fix a backend API endpoint | `apps/trainer_api/app/routers/` |
| Change the training loop | `packages/segcore/segcore/training/train.py` |
| Add a new model architecture | `packages/segcore/segcore/training/model_*.py` |
| Modify the annotation UI | `apps/trainer_ui/src/annotate/` |
| Update training/results UI | `apps/trainer_ui/src/training/` or `results/` |
| Add/fix an E2E test | `apps/trainer_ui/e2e/` |
| Add a Python unit test | `tests/`, `apps/trainer_api/tests/`, `apps/serving_api/tests/` or `packages/segcore/tests/` |

---

## Running Tests

### Full Test Suite

```bash
# Activate venv first!
bash scripts/test.sh
```

This runs (in order):
1. **TypeScript type check** — `tsc --noEmit`
2. **ESLint** — UI linting
3. **Ruff** — Python linting (`packages/`, `apps/trainer_api/app/`, `scripts/`)
4. **Python import verification** — ensures core modules load correctly
5. **Pytest** — unit tests (`tests/`, `apps/trainer_api/tests/`,
   `apps/serving_api/tests/`, `packages/segcore/tests/`)
6. **UI build** — `vite build` (catches compile errors)
7. **E2E tests** — Playwright (requires API running on port 8002)

### Running Tests Individually

```bash
# Python unit tests only
pytest tests/ -v
pytest apps/trainer_api/tests/ -v

# TypeScript check only
cd apps/trainer_ui && npx tsc --noEmit

# E2E tests (API must be running)
cd apps/trainer_ui && npx playwright test

# Single E2E test file
npx playwright test e2e/specs/smoke.spec.ts
```

> **E2E notes:**
> - Run `npx playwright test` **without** a `--reporter` flag — the CLI flag
>   would override the reporter list in `playwright.config.ts` and silently
>   drop the skip-budget gate (`e2e/skip-budget-reporter.ts`).
> - The suite seeds its own fixture projects (`zz-e2e-seed-1/2`) on the
>   running API via `e2e/global-setup.ts`; no manual data setup is needed.

### UI Development Cycle

After editing frontend code, you **must** rebuild before testing on port 8002:

```bash
cd apps/trainer_ui
npm run build
# Then check http://localhost:8002/ui/
```

> **Note:** Vite dev server (port 5173) is not used in production.
> Always test against the built version on port 8002.

---

## Submitting Pull Requests

1. **Fork** the repository and clone your fork.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/short-description
   ```
3. **Make your changes** in small, focused commits.
4. **Run the test suite** — all checks must pass.
5. **Push** your branch and open a Pull Request against `main`.
6. Fill in the PR template. Link any related issues.

### Branch Naming

| Type     | Prefix      | Example                  |
|----------|-------------|--------------------------|
| Bug fix  | `fix/`      | `fix/mask-save-race`     |
| Feature  | `feat/`     | `feat/onnx-quantization` |
| Refactor | `refactor/` | `refactor/train-loop`    |
| Docs     | `docs/`     | `docs/deployment-guide`  |

### Commit Messages

Use a short imperative summary (50 chars or less), then a blank line and
optional details:

```
Fix mask save race condition on slow disks

The beforeunload handler now waits for the keepalive fetch to complete
before allowing the page to unload.
```

### What Makes a Good PR

- **Focused**: one logical change per PR
- **Tested**: include relevant test updates
- **Documented**: update docs if behavior changes
- **Small**: prefer multiple small PRs over one large one

---

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/). Enforced by [Ruff](https://docs.astral.sh/ruff/).
- `from __future__ import annotations` in all modules.
- `logging` instead of `print()` for server output.
- `torch.load(..., weights_only=True)` for security.
- `encoding="utf-8"` when opening files (Windows compatibility).
- GroupNorm only — never use BatchNorm (single-image inference requirement).

### TypeScript / React

- Follow existing conventions in `apps/trainer_ui/src/`.
- No unused imports or variables (ESLint enforced).
- Use Zustand selectors — never subscribe to the whole store
  (e.g., `useMaskStore((s) => s.setMask)`, not `useMaskStore()`).
- Keep components focused; extract large sections into separate files.

### CSS

- Use CSS custom properties (`--sidebar-w`, `--accent`, etc.).
- `overflow: hidden` + `text-overflow: ellipsis` for any text that might overflow.
- `box-shadow: inset 0 0 0 2px` for selection borders (not border/outline).
- No `!important` unless overriding third-party styles.

---

## Dependency Policy

We bundle a lot of ML wheels and ship under Apache-2.0, so a careless
dependency add can quietly violate the project's license guarantee. The
flow below makes that violation mechanically detectable.

### File layout

| File | Role |
|------|------|
| `apps/trainer_api/requirements.in`     | Human-edited dependency source (loose ranges OK). |
| `apps/trainer_api/requirements.txt`    | **Lockfile** — fully pinned, auto-generated from `.in`. |
| `apps/trainer_api/requirements-dev.in` / `.txt` | Same pair for dev/test deps. |
| `apps/trainer_api/requirements-openvino.in` / `.txt` | Same pair for the optional OpenVINO export deps (`--with-openvino`). |
| `apps/serving_api/requirements.in` / `.txt`     | Same pair for the inference server. |
| `apps/trainer_api/overrides.txt`       | uv override file for the trainer compile (`--override`). Currently disables the GUI `opencv-python` build pulled in by supervision. |

`requirements.txt` is **never edited by hand**. CI verifies it matches
`requirements.in` (`.github/workflows/lockfile-drift.yml`).

### Adding or upgrading a Python dependency

1. **Edit `requirements.in`** (the relevant one — trainer / serving / dev).
   Keep ranges loose for utilities, pin to `==X.Y.*` for ML/server core.
2. **Confirm the license is OSS-compatible** at *all three* sources:
   - GitHub repo root `LICENSE`
   - Any sub-tree `LICENSE_*` / `LICENSE.<component>` files
   - HuggingFace model card `license:` field (if it's a model)

   Acceptable: `Apache-2.0` / `MIT` / `BSD-2-Clause` / `BSD-3-Clause` /
   `MPL-2.0` / `LGPL` (linked, not modified) / `Python-2.0`.

   **Blocked**: any non-commercial license — `NVIDIA Source Code License-NC`,
   `CC-BY-NC*`, `Research-only` — and any GPL flavour for runtime deps.
   AGPL-3.0 is also blocked because of its redistribution clause.

   `.github/workflows/license-check.yml` greps source files for known NC
   vendor strings. `.github/workflows/sbom.yml` re-checks the
   machine-readable license expressions in the SBOM.
   `scripts/ci/check_dep_licenses.py` (PyPI metadata, allowlisted in
   `scripts/ci/dep-license-allowlist.txt`) and
   `scripts/ci/check_npm_licenses.py` (npm, via osv-scanner) gate the
   resolved dependency trees.
3. **Recompile the lockfile** with [uv](https://docs.astral.sh/uv/).
   Use exactly the command (and working directory) documented in each
   `requirements*.in` header — uv embeds the command line into the lockfile,
   so any deviation shows up as drift in CI
   (`.github/workflows/lockfile-drift.yml`):

   ```bash
   # Install uv once: pip install uv  (or: pipx install uv)

   # Trainer API (run inside apps/trainer_api)
   (cd apps/trainer_api && uv pip compile requirements.in \
     -o requirements.txt --python-version 3.11 --universal \
     --override overrides.txt)

   # Serving API (run from the repo root)
   uv pip compile apps/serving_api/requirements.in \
     -o apps/serving_api/requirements.txt \
     --python-version 3.11 --no-emit-index-url --universal

   # Dev deps (repo root; constrained by the trainer lockfile)
   uv pip compile apps/trainer_api/requirements-dev.in \
     -o apps/trainer_api/requirements-dev.txt \
     --python-version 3.11 --no-emit-index-url --universal \
     --constraint apps/trainer_api/requirements.txt

   # OpenVINO export deps (run inside apps/trainer_api)
   (cd apps/trainer_api && uv pip compile requirements-openvino.in \
     -o requirements-openvino.txt --python-version 3.11 --universal)
   ```

4. **Run the test suite** to confirm the new resolution actually works:

   ```bash
   bash scripts/test.sh
   ```

5. **Commit `.in` and `.txt` together**, with a license-confirmation trail
   in the commit body — one line per package added or bumped:

   ```
   deps(trainer): add foo for X feature

   LICENSE: foo 1.2.3 Apache-2.0 confirmed at https://github.com/example/foo/blob/main/LICENSE
   ```

   This trail is what makes a half-yearly re-audit (`git log --grep
   "LICENSE:"`) tractable.

### Pinning git+URL dependencies

`mobile-sam` and `sam-2` are not on PyPI; the `.in` file pins them to a
specific commit SHA (not a branch ref). Bumping a SHA is a dependency
upgrade — re-confirm the upstream `LICENSE` has not changed AND re-run the
distill / inference smoke tests before committing the new SHA.

`.github/workflows/sca-git-deps.yml` runs
`scripts/ci/check_git_dep_licenses.py` on every change to
`apps/**/requirements*.txt`; it clones each pinned git dep and fails if
copyleft code would actually ship.

### Vulnerability response

`pip-audit` runs in CI against the trainer, serving and dev lockfiles (OSV
vulnerability database); the optional `requirements-openvino.txt` is not
yet covered. If a transitive dep gets a CVE, bump it in `requirements.in`
and recompile per the steps above — don't pin a patched transitive in the
lockfile by hand, or drift detection will block the next PR.

### SBOM

Every release tag (`v*`) triggers `.github/workflows/sbom.yml`, which
attaches CycloneDX 1.6 + SPDX 2.3 SBOMs to the GitHub Release. The
workflow also re-verifies that no NC license expression slipped through
transitive deps before the SBOM goes public.

### GPU / CUDA matrix

`requirements.txt` is the **single** lockfile for the trainer. It pins the
`torch==2.13.*` family, which ships Blackwell (sm_120) kernels, so one
lockfile now covers Pascal sm_60 through Blackwell sm_120. CI installs the
CPU wheels; the Windows installer and `install_windows.bat` add the CUDA
wheel index at install time (`--index-url .../whl/cu128`, or
`.../whl/cu124` with the `cuda124` argument), so the CUDA-specific
wheels are selected without a separate pin set.

| Lockfile | torch | CUDA wheel index | arch_list covers | Status |
|---|---|---|---|---|
| `requirements.txt` | 2.13.* | cu128 (cu124 via `install_windows.bat cuda124`) | sm_60 – sm_120 | Production |

A parallel `requirements-cu128.txt` existed between 2026-05 and 2026-07,
back when the main lockfile was still on `torch==2.6.0` / CUDA 12.4 and
could not run on Blackwell. It was removed once the main lockfile moved to
a torch build carrying sm_120 kernels: keeping it added a second pin set
that the drift CI could not verify, and it silently fell behind on security
patches. Do not reintroduce a parallel lockfile — bump the torch pin in
`requirements.in` instead.

When upgrading the torch family in the lockfile, follow the same
license-confirmation trail as for any other dependency change. The
lockfile records canonical PyPI versions (no `+cu128` local-version
suffix); the CUDA-specific wheels are selected at install time via the
PyTorch index URL, so CI can install CPU wheels from the same lockfile.

---

## Getting Help

- **Questions**: [GitHub Discussions](../../discussions)
- **Bugs**: [GitHub Issues](../../issues)
- **Security**: See [SECURITY.md](SECURITY.md)

---

## License & Developer Certificate of Origin

By contributing, you agree that your contributions will be licensed under the
same terms as the project — [Apache License 2.0](LICENSE).

We use the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) to confirm that contributors have the right to submit their changes.
Every commit on a pull request must be signed off by adding a `Signed-off-by`
line to the commit message:

```
Fix mask save race on slow disks

Signed-off-by: Your Name <your.email@example.com>
```

The easiest way to add this is the `-s` flag:

```bash
git commit -s -m "Fix mask save race on slow disks"
```

The name and email **must** match the identity in `git config user.name` /
`user.email`; pull requests with unsigned commits will be asked to amend.

By signing off, you certify the four points listed at
https://developercertificate.org/ — in short, that you wrote the code (or have
the right to submit it) and that it is provided under the project's license.
