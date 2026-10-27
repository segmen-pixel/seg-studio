// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Tiny deterministic assets for the e2e seed projects (see global-setup.ts).
 * 64x64 RGB PNGs, gray background with one colored rectangle each so that
 * per-image region labels differ.
 */

export const SEED_IMAGES: Record<string, string> = {
  "seed-01.png":
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAsklEQVR4nO3PsQ2AQBDEQBdGcZRHWVRAggPrpR1ddomX+3DUARZ1gEUdYFEHWNQBFnWARR1g8fV4rkveBmzABmzABmzABmzABmzABmzABmzABmzAnwGnoA6wqAMs6gCLOsCiDrCoAyzqAIs6wKIOsKgDLOoAizrAog6wqAMs6gCLOsCiDrCoAyzqAIs6wKIOsKgDLOoAizrAog6wqAMs6gCLOsCiDrCoAyzqAIs6wKIOsF7CSohaDt70hwAAAABJRU5ErkJggg==",
  "seed-02.png":
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAqUlEQVR4nO3PwQmAABDEwBRmcZZnWRYRJQo73PvIcv4cdYBFHWBRB1jUARZ1gEUdYFEHWNQBFnWARR1gUQdY1AEWdYBFHWBRB1jUARZ1gEUdYFEHWNQBFnWARR1gUQdY1AEWdYBFHWBRB1jUARZ1gMV7r4/rePA2YAM2YAM2YAM2YAM2YAM2YAM2YAM2YAM2YAM24EuoAyzqAIs6wKIOsKgDLOoAizrAugGvrVhac1yQggAAAABJRU5ErkJggg==",
  "seed-03.png":
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAtElEQVR4nO3PwQmAUBDE0BRmcZZnWRYRIX6Yx14XMtyHow6wqAMs6gAL83xdzye3ARuwARuwARuwARuwARuwARuwARtw2oA/oA6wqAMs6gCLOsCiDrCoAyzqAIs6wKIOsKgDLOoAizrAog6wqAMs6gCLOsCiDrCoAyzqAIs6wKIOsKgDLOoAizrAog6wqAMs6gCLOsCiDrCoAyzqAIs6wKIOsKgDLOoAizrAog6wqAMs6gDrBZ+MiVo5J+vaAAAAAElFTkSuQmCC",
};

/** 64x64 raw uint8 mask: 255 = unpainted, a 32x32 square of class 1. */
export function seedMask(): Uint8Array {
  const w = 64;
  const mask = new Uint8Array(w * w).fill(255);
  for (let y = 16; y < 48; y++) {
    for (let x = 16; x < 48; x++) {
      mask[y * w + x] = 1;
    }
  }
  return mask;
}
