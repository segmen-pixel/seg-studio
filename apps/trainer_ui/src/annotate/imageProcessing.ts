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

export function gaussianBlur(gray: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const kSize = (Math.ceil(sigma * 6) | 1);
  const half = kSize >> 1;
  const kernel = new Float32Array(kSize);
  let sum = 0;
  for (let i = 0; i < kSize; i++) {
    const x = i - half;
    kernel[i] = Math.exp(-0.5 * (x * x) / (sigma * sigma));
    sum += kernel[i];
  }
  for (let i = 0; i < kSize; i++) kernel[i] /= sum;
  // Horizontal pass
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let k = -half; k <= half; k++) {
        const sx = Math.min(w - 1, Math.max(0, x + k));
        v += gray[y * w + sx] * kernel[k + half];
      }
      tmp[y * w + x] = v;
    }
  }
  // Vertical pass
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let k = -half; k <= half; k++) {
        const sy = Math.min(h - 1, Math.max(0, y + k));
        v += tmp[sy * w + x] * kernel[k + half];
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

export function laplacianOfGaussian(gray: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const blurred = gaussianBlur(gray, w, h, sigma);
  const log = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      log[idx] = blurred[idx - 1] + blurred[idx + 1] + blurred[idx - w] + blurred[idx + w] - 4 * blurred[idx];
    }
  }
  return log;
}

/** CIE76 delta-E between two Lab colors. */
export function deltaE76(lab1: [number, number, number], lab2: [number, number, number]): number {
  const dL = lab1[0] - lab2[0], da = lab1[1] - lab2[1], db = lab1[2] - lab2[2];
  return Math.sqrt(dL * dL + da * da + db * db);
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
function _connectedComponentSizes(mask: Uint8Array, w: number, h: number): number[] {
  const n = w * h;
  const visited = new Uint8Array(n);
  const sizes: number[] = [];
  for (let i = 0; i < n; i++) {
    if (mask[i] === 0 || visited[i]) continue;
    let size = 0;
    const queue = [i];
    visited[i] = 1;
    let head = 0;
    while (head < queue.length) {
      const ci = queue[head++];
      size++;
      const cx = ci % w, cy = (ci - cx) / w;
      const tryAdd = (ni: number) => { if (!visited[ni] && mask[ni] > 0) { visited[ni] = 1; queue.push(ni); } };
      if (cx > 0) tryAdd(ci - 1);
      if (cx < w - 1) tryAdd(ci + 1);
      if (cy > 0) tryAdd(ci - w);
      if (cy < h - 1) tryAdd(ci + w);
    }
    sizes.push(size);
  }
  return sizes;
}

/**
 * Analyze painted mask pixels to determine:
 * - Best detection channel (gray/L/a/b)
 * - Target Lab color + auto color tolerance
 * - Spot size range (±50% of painted spots median)
 */
export function analyzeMaskForSpotDetect(
  pixels: Uint8ClampedArray, w: number, h: number,
  maskIndex: Uint8Array,
): {
  channel: "gray" | "L" | "a" | "b";
  targetLab: [number, number, number];
  autoColorTolerance: number;
  sizeRange: [number, number];
  /** Best channel SNR — high (≥3) = color mode recommended, low = DoG mode */
  bestSnr: number;
} | null {
  const n = w * h;
  // Collect RGB stats of painted pixels using fast approximation
  // Use R-G ≈ a*, (R+G)/2-B ≈ b*, gray ≈ L to avoid costly rgbToLab per pixel
  let fgCount = 0;
  let fgSumGray = 0, fgSumA = 0, fgSumB = 0;
  let fgSumR = 0, fgSumG = 0, fgSumBl = 0;
  for (let i = 0; i < n; i++) {
    if (maskIndex[i] > 0) {
      const bi = i * 4;
      const r = pixels[bi], g = pixels[bi + 1], b = pixels[bi + 2];
      fgSumGray += 0.299 * r + 0.587 * g + 0.114 * b;
      fgSumA += r - g;
      fgSumB += (r + g) * 0.5 - b;
      fgSumR += r; fgSumG += g; fgSumBl += b;
      fgCount++;
    }
  }
  if (fgCount < 5) return null;

  const fgMeanGray = fgSumGray / fgCount;
  const fgMeanA = fgSumA / fgCount;
  const fgMeanB = fgSumB / fgCount;

  // FG std (fast channels)
  let fgVarGray = 0, fgVarA = 0, fgVarB = 0;
  for (let i = 0; i < n; i++) {
    if (maskIndex[i] > 0) {
      const bi = i * 4;
      const r = pixels[bi], g = pixels[bi + 1], b = pixels[bi + 2];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      const a = r - g;
      const bv = (r + g) * 0.5 - b;
      fgVarGray += (gray - fgMeanGray) ** 2;
      fgVarA += (a - fgMeanA) ** 2;
      fgVarB += (bv - fgMeanB) ** 2;
    }
  }
  const fgStdGray = Math.sqrt(fgVarGray / fgCount);
  const fgStdA = Math.sqrt(fgVarA / fgCount);
  const fgStdB = Math.sqrt(fgVarB / fgCount);

  // BG mean — sample up to 3000 pixels
  const step = Math.max(1, Math.floor(n / 3000));
  let bgGray = 0, bgA = 0, bgB = 0, bgCount = 0;
  for (let i = 0; i < n; i += step) {
    if (maskIndex[i] === 0) {
      const bi = i * 4;
      const r = pixels[bi], g = pixels[bi + 1], b = pixels[bi + 2];
      bgGray += 0.299 * r + 0.587 * g + 0.114 * b;
      bgA += r - g;
      bgB += (r + g) * 0.5 - b;
      bgCount++;
    }
  }
  if (bgCount < 10) return null;
  bgGray /= bgCount; bgA /= bgCount; bgB /= bgCount;

  // SNR per channel
  const snr: [number, number, number] = [
    Math.abs(fgMeanGray - bgGray) / Math.max(fgStdGray, 1),
    Math.abs(fgMeanA - bgA) / Math.max(fgStdA, 1),
    Math.abs(fgMeanB - bgB) / Math.max(fgStdB, 1),
  ];
  const channels: Array<"L" | "a" | "b"> = ["L", "a", "b"];
  let bestIdx = 0;
  if (snr[1] > snr[bestIdx]) bestIdx = 1;
  if (snr[2] > snr[bestIdx]) bestIdx = 2;
  const bestChannel: "gray" | "L" | "a" | "b" = snr[bestIdx] < 1.5 ? "gray" : channels[bestIdx];

  // targetLab: compute only for the FG mean (single call, not per-pixel)
  const fgMeanR = fgSumR / fgCount, fgMeanG = fgSumG / fgCount, fgMeanBl = fgSumBl / fgCount;
  const targetLabFull = rgbToLab(fgMeanR, fgMeanG, fgMeanBl);

  // Auto color tolerance from fast std → approximate ΔE
  const autoColorTolerance = Math.max(8, Math.min(40,
    2 * Math.sqrt(fgStdGray * 0.3 + fgStdA * 0.5 + fgStdB * 0.3)
  ));

  // Size filter: connected component sizes of painted spots → median ±50%
  const spotSizes = _connectedComponentSizes(maskIndex, w, h);
  let sizeRange: [number, number] = [1, 500]; // fallback
  if (spotSizes.length > 0) {
    spotSizes.sort((a, b) => a - b);
    const median = spotSizes[Math.floor(spotSizes.length / 2)];
    sizeRange = [Math.max(1, Math.round(median * 0.5)), Math.round(median * 1.5)];
  }

  return { channel: bestChannel, targetLab: targetLabFull, autoColorTolerance, sizeRange, bestSnr: snr[bestIdx] };
}

/**
 * Auto-optimize sensitivity by maximizing IoU against sample mask.
 * Tries 20 sensitivity values and returns the best.
 */
export function autoOptimizeSensitivity(
  scores: Float32Array, std: number, extrema: Uint32Array,
  w: number, h: number, classId: number,
  sampleMask: Uint8Array,
  pixels?: Uint8ClampedArray,
  targetLab?: [number, number, number] | null,
  colorTolerance?: number,
  sizeRange?: [number, number],
): { bestSensitivity: number; bestIoU: number } {
  let bestSens = 15;
  let bestIoU = -1;
  // Binary ground truth from sample mask
  const n = w * h;
  let gtCount = 0;
  for (let i = 0; i < n; i++) { if (sampleMask[i] > 0) gtCount++; }
  if (gtCount === 0) return { bestSensitivity: 15, bestIoU: 0 };

  // Try sensitivities 2, 6, 10, ..., 38 (10 steps)
  for (let sens = 2; sens <= 38; sens += 4) {
    const { mask } = thresholdSpots(scores, std, extrema, w, h, sens, classId, pixels, targetLab, colorTolerance, undefined, sizeRange);
    let intersection = 0, union = 0;
    for (let i = 0; i < n; i++) {
      const gt = sampleMask[i] > 0;
      const pred = mask[i] > 0;
      if (gt && pred) intersection++;
      if (gt || pred) union++;
    }
    const iou = union > 0 ? intersection / union : 0;
    if (iou > bestIoU) { bestIoU = iou; bestSens = sens; }
  }
  return { bestSensitivity: bestSens, bestIoU };
}

/** Compute multi-scale LoG spot score map + local extrema map (expensive). */
export function computeSpotScores(
  pixels: Uint8ClampedArray, w: number, h: number,
  channel: "gray" | "L" | "a" | "b" = "gray",
): { scores: Float32Array; std: number; extrema: Uint32Array } {
  const n = w * h;
  const gray = new Float32Array(n);
  if (channel === "gray" || channel === "L") {
    // Grayscale ≈ L (fast, no Lab conversion needed)
    for (let i = 0; i < n; i++) {
      const bi = i * 4;
      gray[i] = 0.299 * pixels[bi] + 0.587 * pixels[bi + 1] + 0.114 * pixels[bi + 2];
    }
  } else if (channel === "a") {
    // Fast approximation of Lab a* (green-red axis): R - G
    for (let i = 0; i < n; i++) {
      const bi = i * 4;
      gray[i] = pixels[bi] - pixels[bi + 1];
    }
  } else {
    // Fast approximation of Lab b* (blue-yellow axis): (R+G)/2 - B
    for (let i = 0; i < n; i++) {
      const bi = i * 4;
      gray[i] = (pixels[bi] + pixels[bi + 1]) * 0.5 - pixels[bi + 2];
    }
  }
  const log1 = laplacianOfGaussian(gray, w, h, 1.5);
  const log2 = laplacianOfGaussian(gray, w, h, 3.0);
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    scores[i] = Math.max(Math.abs(log1[i]), Math.abs(log2[i]));
  }
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += scores[i] * scores[i];
  const std = Math.sqrt(sumSq / n);
  // Precompute local maxima (5×5 window) — indices only
  const extremaList: number[] = [];
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const idx = y * w + x;
      const val = scores[idx];
      let isMax = true;
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        for (let dx = -2; dx <= 2 && isMax; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (scores[(y + dy) * w + (x + dx)] > val) isMax = false;
        }
      }
      if (isMax) extremaList.push(idx);
    }
  }
  const extrema = new Uint32Array(extremaList);
  return { scores, std, extrema };
}

/** Flood fill a single seed and write to mask. Returns region size (0 if skipped). */
function _floodFillSeed(
  idx: number, scores: Float32Array, threshold: number,
  w: number, h: number, classId: number,
  mask: Uint8Array, visited: Uint8Array,
  pixels?: Uint8ClampedArray, targetLab?: [number, number, number] | null, colorTolerance?: number,
  skipColorFilter?: boolean,
  sizeRange?: [number, number],
): number {
  if (visited[idx]) return 0;
  // For the click seed, accept any positive score; for normal seeds require > threshold
  if (!skipColorFilter && scores[idx] <= threshold) return 0;
  const queue = [idx];
  visited[idx] = 1;
  let regionSize = 1;
  let head = 0;
  let sumR = 0, sumG = 0, sumB = 0;
  if (pixels && targetLab) {
    const bi = idx * 4;
    sumR = pixels[bi]; sumG = pixels[bi + 1]; sumB = pixels[bi + 2];
  }
  // Use a relaxed threshold for click-seeded fills (half of normal)
  const fillThreshold = skipColorFilter ? threshold * 0.5 : threshold;
  while (head < queue.length) {
    const ci = queue[head++];
    const cx = ci % w;
    const cy = (ci - cx) / w;
    const tryExpand = (ni: number) => {
      if (!visited[ni] && scores[ni] > fillThreshold) {
        visited[ni] = 1; queue.push(ni); regionSize++;
        if (pixels && targetLab) {
          const bi = ni * 4;
          sumR += pixels[bi]; sumG += pixels[bi + 1]; sumB += pixels[bi + 2];
        }
      }
    };
    if (cx > 0) tryExpand(ci - 1);
    if (cx < w - 1) tryExpand(ci + 1);
    if (cy > 0) tryExpand(ci - w);
    if (cy < h - 1) tryExpand(ci + w);
  }
  const minSize = sizeRange ? sizeRange[0] : 1;
  const maxSize = sizeRange ? sizeRange[1] : 200;
  if (regionSize < minSize || regionSize > maxSize) return 0;
  if (!skipColorFilter && pixels && targetLab && colorTolerance !== undefined) {
    const avgLab = rgbToLab(sumR / regionSize, sumG / regionSize, sumB / regionSize);
    if (deltaE76(avgLab, targetLab) > colorTolerance) return 0;
  }
  for (let qi = 0; qi < queue.length; qi++) mask[queue[qi]] = classId;
  return regionSize;
}

/** Cheap re-threshold using precomputed scores + extrema indices. */
export function thresholdSpots(
  scores: Float32Array, std: number, extrema: Uint32Array,
  w: number, h: number,
  sensitivity: number, classId: number,
  pixels?: Uint8ClampedArray, targetLab?: [number, number, number] | null, colorTolerance?: number,
  clickIdx?: number,
  sizeRange?: [number, number],
): { mask: Uint8Array; count: number } {
  const n = w * h;
  const threshold = std * (sensitivity / 5);
  const mask = new Uint8Array(n);
  const visited = new Uint8Array(n);
  let count = 0;
  // Process all extrema seeds
  for (let mi = 0; mi < extrema.length; mi++) {
    if (_floodFillSeed(extrema[mi], scores, threshold, w, h, classId, mask, visited, pixels, targetLab, colorTolerance, false, sizeRange)) count++;
  }
  // Ensure the click position blob is always included (user explicitly clicked it)
  if (clickIdx !== undefined && clickIdx >= 0 && clickIdx < n && !visited[clickIdx]) {
    if (_floodFillSeed(clickIdx, scores, threshold, w, h, classId, mask, visited, pixels, targetLab, colorTolerance, true, sizeRange)) count++;
  }
  return { mask, count };
}

export function detectSpots(
  pixels: Uint8ClampedArray,
  w: number,
  h: number,
  sensitivity: number,
  classId: number,
  targetLab?: [number, number, number] | null,
  colorTolerance?: number,
  clickIdx?: number,
  channel: "gray" | "L" | "a" | "b" = "gray",
  sizeRange?: [number, number],
): { mask: Uint8Array; count: number; scores: Float32Array; std: number; extrema: Uint32Array } {
  const { scores, std, extrema } = computeSpotScores(pixels, w, h, channel);
  const { mask, count } = thresholdSpots(scores, std, extrema, w, h, sensitivity, classId, pixels, targetLab, colorTolerance, clickIdx, sizeRange);
  return { mask, count, scores, std, extrema };
}

export function wandFlood(
  startPos: [number, number],
  distMap: Float32Array,
  tolerance: number,
  classId: number,
  width: number,
  height: number,
): Uint8Array {
  const n = width * height;
  const raw = new Uint8Array(n);
  const visited = new Uint8Array(n);
  const startIdx = startPos[1] * width + startPos[0];
  const tolSq = tolerance * tolerance;
  if (distMap[startIdx] > tolSq) return new Uint8Array(n);
  const queue = [startIdx];
  visited[startIdx] = 1;
  let head = 0;
  while (head < queue.length) {
    const idx = queue[head++];
    raw[idx] = 1;
    const x = idx % width;
    const y = (idx - x) / width;
    const nb = [];
    if (x > 0) nb.push(idx - 1);
    if (x < width - 1) nb.push(idx + 1);
    if (y > 0) nb.push(idx - width);
    if (y < height - 1) nb.push(idx + width);
    for (let i = 0; i < nb.length; i++) {
      const ni = nb[i];
      if (visited[ni]) continue;
      visited[ni] = 1;
      if (distMap[ni] > tolSq) continue;
      queue.push(ni);
    }
  }
  // morphological close to fill small holes
  const closed = morphClose(raw, width, height);
  // write classId into result
  const preview = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (closed[i]) preview[i] = classId;
  }
  return preview;
}



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
