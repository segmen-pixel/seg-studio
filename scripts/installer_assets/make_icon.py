#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Render every Seg-Studio icon from one drawing: a white S on an indigo tile.

Geometric shapes only (no fonts), so the output is byte-stable across
machines with the same Pillow major version. Re-run after editing and commit
what it writes:

    python scripts/installer_assets/make_icon.py

  scripts/installer_assets/seg-studio.ico          shortcuts, the Windows tray, the packaged launcher
  scripts/installer_assets/seg-studio-menubar.png  the macOS menu bar item (a template image)
  apps/trainer_ui/public/favicon.ico               the browser tab

The macOS app's icon is drawn when the app is made, at every size it needs
(scripts/macos/create_app_macos.sh runs this with the checkout's venv):

    python scripts/installer_assets/make_icon.py --iconset <dir>.iconset
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

INDIGO = (79, 70, 229, 255)      # #4F46E5, the browser tab's colour
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
CLEAR = (0, 0, 0, 0)
SUPERSAMPLE = 4                  # drawn 4x, then downscaled: smooth edges without fonts

ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
FAVICON_SIZES = [16, 32, 48]
MENUBAR_PX = 36                  # 18 pt at 2x; the menu bar shows it at 18 x 18
MENUBAR_INSET = 2 / 36           # 1 pt clear on each side, as the system's own items have
#: Apple's icon grid: the tile is 824 of 1024, centred.
MACOS_INSET = 100 / 1024
ICONSET = [16, 32, 128, 256, 512]


def _s(d: ImageDraw.ImageDraw, k: float, colour) -> None:
    """The S, in 256-unit coordinates scaled by k: two bowls that share the
    spine, each a thick arc, ends cut square to the stroke."""
    cx, top, height, width, stroke, end = 128, 58, 140, 104, 26, 34
    bowl = (height + stroke) / 2
    x0, x1 = cx - width / 2, cx + width / 2
    upper = [x0 * k, top * k, x1 * k, (top + bowl) * k]
    lower = [x0 * k, (top + height - bowl) * k, x1 * k, (top + height) * k]
    d.arc(upper, 90, 360 - end, fill=colour, width=round(stroke * k))
    d.arc(lower, 270, 360 + 180 - end, fill=colour, width=round(stroke * k))


def render(size: int, template: bool = False, inset: float = 0.0) -> Image.Image:
    """The icon at size x size pixels. template: a black tile with the S cut
    out, for the macOS menu bar. inset: empty margin on each side, as a
    fraction of size."""
    big = size * SUPERSAMPLE
    img = Image.new("RGBA", (big, big), CLEAR)
    tile = round(big * (1 - 2 * inset))
    art = Image.new("RGBA", (tile, tile), CLEAR)
    d = ImageDraw.Draw(art)
    k = tile / 256
    d.rounded_rectangle([0, 0, tile - 1, tile - 1], radius=64 * k, fill=BLACK if template else INDIGO)
    _s(d, k, CLEAR if template else WHITE)
    offset = (big - tile) // 2
    img.paste(art, (offset, offset))
    return img.resize((size, size), Image.LANCZOS)


def _ico(path: Path, sizes: list[int]) -> None:
    largest, *rest = sorted(sizes, reverse=True)
    render(largest).save(path, format="ICO", sizes=[(n, n) for n in sizes],
                         append_images=[render(n) for n in rest])


def write_committed() -> list[Path]:
    out = [HERE / "seg-studio.ico", HERE / "seg-studio-menubar.png",
           REPO / "apps" / "trainer_ui" / "public" / "favicon.ico"]
    _ico(out[0], ICO_SIZES)
    render(MENUBAR_PX, template=True, inset=MENUBAR_INSET).save(out[1], format="PNG")
    _ico(out[2], FAVICON_SIZES)
    return out


def write_iconset(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in ICONSET:
        render(n, inset=MACOS_INSET).save(folder / f"icon_{n}x{n}.png", format="PNG")
        render(2 * n, inset=MACOS_INSET).save(folder / f"icon_{n}x{n}@2x.png", format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iconset", type=Path, help="write a macOS .iconset folder here instead")
    args = parser.parse_args()
    if args.iconset:
        write_iconset(args.iconset)
        print(f"wrote {args.iconset}")
        return
    for path in write_committed():
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
