# LGPL / MPL / GPL upstream notices

Seg-Studio is licensed under Apache License 2.0. The Windows installer
bundles dynamically-linked components governed by the GNU Lesser General
Public License (LGPL), the Mozilla Public License (MPL), or — only as a
referenced clause inside multi-license headers — the GNU General Public
License (GPL). The full text of those licenses is reproduced verbatim
in this directory:

| File | Used by |
|---|---|
| `LGPL-2.0.txt` | Pango |
| `LGPL-2.1.txt` | libvips, Cairo, FFmpeg (LGPL build), pyphen (we elect LGPL) |
| `MPL-1.1.txt` | Cairo (alternative MPL track), pyphen (alternative MPL track) |
| `GPL-2.0.txt` | Reference text only — pyphen's tri-license cites GPL-2.0; we elect LGPL-2.1 |

## How obligations are satisfied

The LGPL grants users the right to **replace** the bundled libraries
with their own builds. To make that practical we:

1. Ship each library as a *dynamically-linked* DLL (no static linking),
   so the user can drop in a compatible build without rebuilding
   Seg-Studio.
2. Reproduce the full LGPL text alongside the binaries (this directory).
3. List the upstream source URL for each library in the table below so
   users can fetch the matching source.

| Component | Version source | LGPL track | Upstream source |
|---|---|---|---|
| libvips | https://github.com/libvips/libvips/releases | LGPL-2.1+ | https://github.com/libvips/libvips |
| Cairo | https://gitlab.freedesktop.org/cairo/cairo/-/releases | LGPL-2.1 (we elect) | https://gitlab.freedesktop.org/cairo/cairo |
| Pango | https://gitlab.gnome.org/GNOME/pango/-/releases | LGPL-2.0+ | https://gitlab.gnome.org/GNOME/pango |
| HarfBuzz | https://github.com/harfbuzz/harfbuzz/releases | MIT (no LGPL obligation) | https://github.com/harfbuzz/harfbuzz |
| FFmpeg (LGPL build, via OpenCV) | https://ffmpeg.org/download.html | LGPL-2.1+ | https://github.com/FFmpeg/FFmpeg |
| pyphen | https://github.com/Kozea/Pyphen/releases | LGPL-2.1+ (we elect) | https://github.com/Kozea/Pyphen |

## Exact corresponding source (binary releases)

Every **binary** release (installer) publishes, as a release asset named
`lgpl-sources-v<version>.zip`:

- the exact upstream source archive (with SHA-256) for every shipped
  LGPL/MPL DLL, as recorded in `lgpl_sources_manifest.json`,
- any patches applied (currently none — we ship unmodified upstream
  builds), and
- the build configuration reference for bundled builds (e.g. the OpenCV
  3rdparty FFmpeg build config).

The bundle is produced by `scripts/release/collect_lgpl_sources.py`
against the installer's `release_manifest.json` and is a release gate:
a binary release must not go out without it. If a published release is
ever missing the asset, open an issue citing the release tag — providing
the matching source is our obligation, not a favour.

## Replacing a bundled library (LGPL §6)

The DLLs are loaded dynamically by file name from the installed
application directory (no static linking, no signature pinning). To use
your own build of e.g. libvips: build a binary-compatible DLL of the
same soname/interface version, replace the file in the installation
directory, and restart Seg-Studio. Nothing in the Seg-Studio license
restricts reverse engineering of the application for the purpose of
debugging such replacement-library modifications.

## OpenCV-bundled FFmpeg note

`opencv_videoio_ffmpegXXXX_64.dll` is the LGPL build that ships inside
the `opencv-python-headless` PyPI wheel. We do not modify, statically
link, or otherwise consume FFmpeg internals; OpenCV loads it at runtime
through its standard pluggable backend. The full LGPL-2.1 text in
`LGPL-2.1.txt` covers this redistribution.

## HarfBuzz

HarfBuzz is licensed under the "Old MIT" license (functionally MIT) and
does not impose LGPL obligations. It is listed here only because it
typically ships in the same DLL bundle as Cairo / Pango.
