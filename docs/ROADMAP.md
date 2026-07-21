# Seg-Studio OSS Roadmap

> **Vision:** Bring deep learning out of the hands of specialists and into the fields, clinics, and factories where it's needed most.
> Our goal: "Build your own DL model from your own data — without ever opening a terminal."

---

## Phase 0: OSS Release Preparation (v1.0) — Current

**Status:** Nearly complete. A fully functional tool that runs locally.

- [x] Semantic segmentation training (SimpleUNet / STDC / DeepLabV3+)
- [x] Rich annotation tools (brush, SAM, wand, superpixel, etc.)
- [x] Patch-based training + sliding window evaluation
- [x] CoreML / ONNX export
- [x] Knowledge distillation pipeline
- [x] MCP bridge (LLM integration)
- [x] Windows installer (.bat)
- [x] Documentation in English and Japanese
- [x] Apache 2.0 license

---

## Phase 1: Reach Developers (v1.1)

**Goal:** Make it easy for ML developers and researchers to try immediately.
**Estimated timeline:** 1-2 months

### Package Distribution
- [x] Create `pyproject.toml` (hatchling build)
- [ ] `pip install seg-studio` installs the API + CLI
- [ ] Entry point: `seg-studio start` launches the server
- [ ] Publish to PyPI (TestPyPI first, then production)
- [ ] GitHub Actions with Trusted Publisher for automated releases

### Model & Weight Distribution
- [ ] Create model repository on Hugging Face Hub
- [ ] Auto-download teacher models on first launch
- [ ] Confirm `.gitignore` includes `*.pth`, `*.onnx`, `*.mlmodel`
- [ ] Download progress display (CLI and UI)

### Developer Experience
- [ ] `seg-studio --demo` for instant launch with demo data
- [ ] Comprehensive API documentation (Swagger)
- [x] Contributing guide (`CONTRIBUTING.md`)
- [x] GitHub Issue / PR templates (`.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`)

---

## Phase 2: Show It to the World (v1.2)

**Goal:** Publish a browser-based demo. Make people think "I want to use this."
**Estimated timeline:** 1-2 months (can run in parallel with Phase 1)

### Online Demo
- [ ] Deploy demo on Hugging Face Spaces
  - Reuse the existing FastAPI + React UI almost as-is
  - Free tier (CPU) for the annotation experience
  - GPU Spaces ($0.60+/h) for training demos
- [ ] Bundle sample datasets (a few images from agriculture/manufacturing)
- [ ] Landing page that communicates value in 30 seconds

### Community Building
- [ ] Enable GitHub Discussions
- [ ] Announce on Hugging Face / Reddit / X

---

## Phase 3: Reach Non-Engineers (v2.0)

**Goal:** A terminal-free experience. Farmers, clinicians, and factory workers can use it on their own.
**Estimated timeline:** 3-6 months

### Desktop Application
- [ ] One-click installer via Electron or Tauri
  - Windows: `.exe` installer (NSIS or WiX)
  - macOS: `.dmg`
  - Bundled Python runtime (no Python installation required)
- [ ] GPU setup wizard
  - Auto-detect CUDA availability
  - Fall back to CPU mode with a clear explanation of performance limitations
- [ ] Auto-update mechanism

### UX Improvements (for Non-Engineers)
- [ ] Onboarding wizard
  - "What do you want to detect?" -> template suggestions
  - Hands-on experience with a sample project
- [ ] Automatic parameter recommendations (simplify the existing auto-tune)
  - Three steps: "Upload images -> Train -> Done"
  - Advanced settings hidden under an "Expert" toggle
- [ ] Localization beyond English and Japanese (i18n)

### Cloud Training Option
- ~~Cloud GPU fallback when no local GPU is available~~ — shipped as RunPod
  Serverless training in the 0.9.x line, then moved out of the open-source
  distribution in 2026-07 to keep the public scope local-first (the
  implementation is parked internally and may return)
  - "You can train even without a GPU" messaging
- [ ] Training completion notifications (email or browser push)

---

## Phase 4: Expand the Ecosystem (v2.x+)

**Goal:** Extend beyond segmentation. Industry-specific templates.
**Estimated timeline:** 6+ months

### Task Expansion
- [ ] Object Detection support
- [ ] Classification support
- ~~Anomaly Detection support~~ — retired 2026-07: the mode was down to a single statistical method; anomaly detection moves to the companion AnomaLens project

### Industry Templates
- [ ] Agriculture pack: disease detection, weed detection, harvest timing
- [ ] Manufacturing pack: visual inspection, defect classification, dimensional measurement
- [ ] Medical pack: cell segmentation, pathology image analysis
- [ ] Pre-trained models + sample data + tutorials per pack

### Integrations
- [ ] Enhanced iPad / iPhone integration (direct CoreML deployment)
- [ ] Edge device support (Jetson, Raspberry Pi)
- [x] REST API / Python SDK for programmatic access (`/v2` streaming inference API + `packages/seg-sdk`)

---

## Decision Criteria: What to Prioritize in Each Phase

| Criterion | Priority |
|-----------|----------|
| **Can users feel the value in 30 seconds?** | Highest |
| **Does it lower the installation barrier?** | Highest |
| **Does it minimize changes to existing code?** | High |
| **Is it achievable solo?** | High |
| **Does it avoid introducing new tech stacks?** | Medium |

## Success Metrics

| Phase | KPI |
|-------|-----|
| Phase 1 | PyPI downloads, GitHub Stars |
| Phase 2 | HF Spaces demo traffic, quality of GitHub Issues |
| Phase 3 | Feedback from non-engineer users |
| Phase 4 | Industry-specific adoption case studies |

---

> **Principle:** "A working demo > perfect code." Getting it into people's hands is the top priority.
> Invest in lowering the barrier to try, not in adding features.
