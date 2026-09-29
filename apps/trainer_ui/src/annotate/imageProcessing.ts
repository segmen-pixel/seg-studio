// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
export function buildLut(classes: { id: number; color: [number, number, number] }[], alpha: number) {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i += 1) {
    lut[i * 4 + 3] = 0;
  }
  classes.forEach((cls) => {
    const base = cls.id * 4;
    lut[base] = cls.color[0];
    lut[base + 1] = cls.color[1];
    lut[base + 2] = cls.color[2];
    lut[base + 3] = cls.id === 0 ? 0 : alpha;
  });
  return lut;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  // sRGB → linear
  let rl = r / 255, gl = g / 255, bl = b / 255;
  rl = rl > 0.04045 ? ((rl + 0.055) / 1.055) ** 2.4 : rl / 12.92;
  gl = gl > 0.04045 ? ((gl + 0.055) / 1.055) ** 2.4 : gl / 12.92;
  bl = bl > 0.04045 ? ((bl + 0.055) / 1.055) ** 2.4 : bl / 12.92;
  // linear RGB → XYZ (D65)
  const x = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / 0.95047;
  const y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750);
  const z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / 1.08883;
  // XYZ → Lab
  const f = (t: number) => t > 0.008856 ? t ** (1 / 3) : 7.787 * t + 16 / 116;
  const fx = f(x), fy = f(y), fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function morphClose(mask: Uint8Array, w: number, h: number): Uint8Array {
  // 3×3 binary dilate then erode
  const dilated = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let found = false;
      for (let dy = -1; dy <= 1 && !found; dy++) {
        for (let dx = -1; dx <= 1 && !found; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx] > 0) found = true;
        }
      }
      if (found) dilated[y * w + x] = 1;
    }
  }
  const result = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let allSet = true;
      for (let dy = -1; dy <= 1 && allSet; dy++) {
        for (let dx = -1; dx <= 1 && allSet; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h || dilated[ny * w + nx] === 0) allSet = false;
        }
      }
      if (allSet) result[y * w + x] = 1;
    }
  }
  return result;
}

/** Sample average Lab color from a 5×5 patch around (cx, cy). */
export function sampleClickLab(
  pixels: Uint8ClampedArray, w: number, h: number, cx: number, cy: number,
): [number, number, number] {
  let sumL = 0, sumA = 0, sumB = 0, count = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const px = cx + dx, py = cy + dy;
      if (px >= 0 && px < w && py >= 0 && py < h) {
        const bi = (py * w + px) * 4;
        const lab = rgbToLab(pixels[bi], pixels[bi + 1], pixels[bi + 2]);
        sumL += lab[0]; sumA += lab[1]; sumB += lab[2]; count++;
      }
    }
  }
  return [sumL / count, sumA / count, sumB / count];
}

/**
 * Connected component sizes in a binary mask.
 */
export function base64ToBlob(b64: string, mime = "image/png"): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export async function decodeMaskBlob(blob: Blob, w: number, h: number): Promise<Uint8Array> {
  const img = new Image();
  const url = URL.createObjectURL(blob);
  try {
    const data = await new Promise<ImageData>((resolve, reject) => {
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        const ctx = c.getContext("2d")!;
        ctx.drawImage(img, 0, 0, w, h);
        resolve(ctx.getImageData(0, 0, w, h));
      };
      img.onerror = () => reject(new Error("decode failed"));
      img.src = url;
    });
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < mask.length; i++) mask[i] = data.data[i * 4];
    return mask;
  } finally {
    URL.revokeObjectURL(url);
  }
}
