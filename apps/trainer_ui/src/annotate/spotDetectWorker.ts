// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
// Web Worker: Difference-of-Gaussians (DoG) spot score computation
// Uses IIR Gaussian blur (O(n), sigma-independent) for maximum speed.
// Runs on downscaled image (~1280px) for consistent sub-50ms performance.

// ---- Types ----

type ComputeRequest = {
  kind: "compute";
  pixels: Uint8ClampedArray; // RGBA of downscaled image
  width: number;
  height: number;
  channel: "gray" | "L" | "a" | "b";
  reqId: number;
};

type ThresholdRequest = {
  kind: "threshold";
  sensitivity: number;
  classId: number;
  colorTolerance?: number;
  targetLab?: [number, number, number] | null;
  sizeRange?: [number, number];
  reqId: number;
};

type ColorDistRequest = {
  kind: "colorDist";
  pixels: Uint8ClampedArray;
  width: number;
  height: number;
  targetLab: [number, number, number];
  reqId: number;
};

type ColorThresholdRequest = {
  kind: "colorThreshold";
  tolerance: number; // ΔE threshold (sensitivity maps to this)
  classId: number;
  sizeRange?: [number, number];
  reqId: number;
};

type SpotWorkerRequest = ComputeRequest | ThresholdRequest | ColorDistRequest | ColorThresholdRequest;

// ---- IIR Gaussian blur (glur-style, O(n) regardless of sigma) ----

/**
 * Fast IIR Gaussian approximation based on:
 * "Recursive implementation of the Gaussian filter" (Young & van Vliet, 1995)
 * O(n) complexity independent of sigma.
 */
function iirCoeffs(sigma: number): { b0: number; b1: number; b2: number; b3: number; B: number } {
  // Compute filter coefficients for given sigma
  const q = sigma < 2.5
    ? 3.97156 - 4.14554 * Math.sqrt(1 - 0.26891 * sigma)
    : 0.98711 * sigma - 0.96330;
  const q2 = q * q;
  const q3 = q2 * q;
  const b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
  const b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
  const b2 = -(1.4281 * q2 + 1.26661 * q3);
  const b3 = 0.422205 * q3;
  const B = 1 - (b1 + b2 + b3) / b0;
  return { b0, b1, b2, b3, B };
}

function iirBlur1D(data: Float32Array, len: number, stride: number, count: number, c: ReturnType<typeof iirCoeffs>) {
  const { b0, b1, b2, b3, B } = c;
  for (let line = 0; line < count; line++) {
    const off = line * stride;
    // Forward pass
    let w1 = data[off], w2 = w1, w3 = w1;
    for (let i = 0; i < len; i++) {
      const idx = off + i;
      const w = B * data[idx] + (b1 * w1 + b2 * w2 + b3 * w3) / b0;
      data[idx] = w;
      w3 = w2; w2 = w1; w1 = w;
    }
    // Backward pass
    w1 = data[off + len - 1]; w2 = w1; w3 = w1;
    for (let i = len - 1; i >= 0; i--) {
      const idx = off + i;
      const w = B * data[idx] + (b1 * w1 + b2 * w2 + b3 * w3) / b0;
      data[idx] = w;
      w3 = w2; w2 = w1; w1 = w;
    }
  }
}

function iirGaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (sigma < 0.5) return new Float32Array(src); // no-op for tiny sigma
  const c = iirCoeffs(sigma);
  const out = new Float32Array(src);

  // Horizontal pass (rows are contiguous)
  iirBlur1D(out, w, w, h, c);

  // Vertical pass — transpose, blur rows, transpose back (cache-friendly)
  const transposed = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      transposed[x * h + y] = out[y * w + x];
    }
  }
  iirBlur1D(transposed, h, h, w, c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = transposed[x * h + y];
    }
  }
  return out;
}

// ---- Channel extraction ----

function extractChannel(pixels: Uint8ClampedArray, n: number, channel: string): Float32Array {
  const gray = new Float32Array(n);
  if (channel === "gray" || channel === "L") {
    for (let i = 0; i < n; i++) {
      const bi = i * 4;
      gray[i] = 0.299 * pixels[bi] + 0.587 * pixels[bi + 1] + 0.114 * pixels[bi + 2];
    }
  } else if (channel === "a") {
    for (let i = 0; i < n; i++) {
      const bi = i * 4;
      gray[i] = pixels[bi] - pixels[bi + 1]; // R - G
    }
  } else {
    for (let i = 0; i < n; i++) {
      const bi = i * 4;
      gray[i] = (pixels[bi] + pixels[bi + 1]) * 0.5 - pixels[bi + 2]; // (R+G)/2 - B
    }
  }
  return gray;
}

// ---- DoG (Difference of Gaussians) ----

function computeDoG(gray: Float32Array, w: number, h: number): { scores: Float32Array; std: number } {
  const n = w * h;
  // Two-scale DoG: σ1=1.5, σ2=2.4 (ratio 1.6) and σ3=3.0, σ4=4.8
  const blur1 = iirGaussianBlur(gray, w, h, 1.5);
  const blur2 = iirGaussianBlur(gray, w, h, 2.4);
  const blur3 = iirGaussianBlur(gray, w, h, 3.0);
  const blur4 = iirGaussianBlur(gray, w, h, 4.8);

  const scores = new Float32Array(n);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const dog1 = Math.abs(blur1[i] - blur2[i]);
    const dog2 = Math.abs(blur3[i] - blur4[i]);
    const v = Math.max(dog1, dog2);
    scores[i] = v;
    sumSq += v * v;
  }
  const std = Math.sqrt(sumSq / n);
  return { scores, std };
}

// ---- Local extrema (3x3 window — faster than 5x5, sufficient after DoG) ----

function findExtrema(scores: Float32Array, w: number, h: number, minScore: number): Uint32Array {
  const list: number[] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const val = scores[idx];
      if (val <= minScore) continue;
      // 3x3 neighborhood check
      if (val >= scores[idx - 1] && val >= scores[idx + 1] &&
          val >= scores[idx - w] && val >= scores[idx + w] &&
          val >= scores[idx - w - 1] && val >= scores[idx - w + 1] &&
          val >= scores[idx + w - 1] && val >= scores[idx + w + 1]) {
        list.push(idx);
      }
    }
  }
  return new Uint32Array(list);
}

// ---- Flood fill + filtering ----

function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  let rl = r / 255, gl = g / 255, bl = b / 255;
  rl = rl > 0.04045 ? ((rl + 0.055) / 1.055) ** 2.4 : rl / 12.92;
  gl = gl > 0.04045 ? ((gl + 0.055) / 1.055) ** 2.4 : gl / 12.92;
  bl = bl > 0.04045 ? ((bl + 0.055) / 1.055) ** 2.4 : bl / 12.92;
  const x = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / 0.95047;
  const y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750);
  const z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / 1.08883;
  const f = (t: number) => t > 0.008856 ? t ** (1 / 3) : 7.787 * t + 16 / 116;
  const fx = f(x), fy = f(y), fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function thresholdAndFill(
  scores: Float32Array, std: number, extrema: Uint32Array,
  w: number, h: number,
  sensitivity: number, classId: number,
  pixels: Uint8ClampedArray | null,
  targetLab: [number, number, number] | null | undefined,
  colorTolerance: number | undefined,
  sizeRange: [number, number] | undefined,
): { mask: Uint8Array; count: number } {
  const n = w * h;
  const threshold = std * (sensitivity / 5);
  const fillThreshold = threshold * 0.5;
  const mask = new Uint8Array(n);
  const visited = new Uint8Array(n);
  let count = 0;

  const minSize = sizeRange ? sizeRange[0] : 1;
  const maxSize = sizeRange ? sizeRange[1] : 500;
  const checkColor = !!pixels && !!targetLab && colorTolerance !== undefined;

  for (let ei = 0; ei < extrema.length; ei++) {
    const seed = extrema[ei];
    if (visited[seed] || scores[seed] <= threshold) continue;

    // Flood fill
    const queue = [seed];
    visited[seed] = 1;
    let head = 0;
    let sumR = 0, sumG = 0, sumB = 0;
    if (checkColor) {
      const bi = seed * 4;
      sumR = pixels![bi]; sumG = pixels![bi + 1]; sumB = pixels![bi + 2];
    }

    while (head < queue.length) {
      const ci = queue[head++];
      const cx = ci % w;
      const cy = (ci - cx) / w;
      const neighbors = [];
      if (cx > 0) neighbors.push(ci - 1);
      if (cx < w - 1) neighbors.push(ci + 1);
      if (cy > 0) neighbors.push(ci - w);
      if (cy < h - 1) neighbors.push(ci + w);
      for (let ni = 0; ni < neighbors.length; ni++) {
        const nIdx = neighbors[ni];
        if (!visited[nIdx] && scores[nIdx] > fillThreshold) {
          visited[nIdx] = 1;
          queue.push(nIdx);
          if (checkColor) {
            const bi = nIdx * 4;
            sumR += pixels![bi]; sumG += pixels![bi + 1]; sumB += pixels![bi + 2];
          }
        }
      }
    }

    const regionSize = queue.length;
    if (regionSize < minSize || regionSize > maxSize) continue;

    // Color filter
    if (checkColor && regionSize > 0) {
      const avgLab = rgbToLab(sumR / regionSize, sumG / regionSize, sumB / regionSize);
      const dL = avgLab[0] - targetLab![0], da = avgLab[1] - targetLab![1], db = avgLab[2] - targetLab![2];
      if (Math.sqrt(dL * dL + da * da + db * db) > colorTolerance!) continue;
    }

    for (let qi = 0; qi < queue.length; qi++) mask[queue[qi]] = classId;
    count++;
  }
  return { mask, count };
}

// ---- Color distance map ----

function computeColorDistMap(
  pixels: Uint8ClampedArray, w: number, h: number,
  targetLab: [number, number, number],
): Float32Array {
  const n = w * h;
  const dist = new Float32Array(n);
  const tL = targetLab[0], ta = targetLab[1], tb = targetLab[2];
  for (let i = 0; i < n; i++) {
    const bi = i * 4;
    const lab = rgbToLab(pixels[bi], pixels[bi + 1], pixels[bi + 2]);
    const dL = lab[0] - tL, da = lab[1] - ta, db = lab[2] - tb;
    dist[i] = Math.sqrt(dL * dL + da * da + db * db);
  }
  return dist;
}

function colorThresholdAndLabel(
  distMap: Float32Array, w: number, h: number,
  tolerance: number, classId: number,
  sizeRange: [number, number] | undefined,
): { mask: Uint8Array; count: number } {
  const n = w * h;
  const mask = new Uint8Array(n);
  const visited = new Uint8Array(n);
  const minSize = sizeRange ? sizeRange[0] : 1;
  const maxSize = sizeRange ? sizeRange[1] : 2000;
  let count = 0;

  for (let i = 0; i < n; i++) {
    if (visited[i] || distMap[i] > tolerance) continue;
    // BFS connected component
    const queue = [i];
    visited[i] = 1;
    let head = 0;
    while (head < queue.length) {
      const ci = queue[head++];
      const cx = ci % w;
      const cy = (ci - cx) / w;
      if (cx > 0 && !visited[ci - 1] && distMap[ci - 1] <= tolerance) { visited[ci - 1] = 1; queue.push(ci - 1); }
      if (cx < w - 1 && !visited[ci + 1] && distMap[ci + 1] <= tolerance) { visited[ci + 1] = 1; queue.push(ci + 1); }
      if (cy > 0 && !visited[ci - w] && distMap[ci - w] <= tolerance) { visited[ci - w] = 1; queue.push(ci - w); }
      if (cy < h - 1 && !visited[ci + w] && distMap[ci + w] <= tolerance) { visited[ci + w] = 1; queue.push(ci + w); }
    }
    const regionSize = queue.length;
    if (regionSize < minSize || regionSize > maxSize) continue;
    for (let qi = 0; qi < queue.length; qi++) mask[queue[qi]] = classId;
    count++;
  }
  return { mask, count };
}

// ---- Cached state ----

let cachedScores: Float32Array | null = null;
let cachedStd = 0;
let cachedExtrema: Uint32Array | null = null;
let cachedPixels: Uint8ClampedArray | null = null;
let cachedWidth = 0;
let cachedHeight = 0;
let cachedColorDist: Float32Array | null = null;
let cachedColorWidth = 0;
let cachedColorHeight = 0;

// ---- Message handler ----

self.onmessage = (e: MessageEvent<SpotWorkerRequest>) => {
  const msg = e.data;

  if (msg.kind === "compute") {
    const { pixels, width, height, channel, reqId } = msg;
    const n = width * height;
    const t0 = performance.now();

    const gray = extractChannel(pixels, n, channel);
    const { scores, std } = computeDoG(gray, width, height);
    // Use a very low threshold for extrema pre-filtering (std * 0.1)
    const extrema = findExtrema(scores, width, height, std * 0.1);

    const elapsed = performance.now() - t0;

    // Cache for subsequent threshold requests
    cachedScores = scores;
    cachedStd = std;
    cachedExtrema = extrema;
    cachedPixels = pixels;
    cachedWidth = width;
    cachedHeight = height;

    // Transfer scores buffer (zero-copy)
    const scoresCopy = new Float32Array(scores);
    self.postMessage({
      kind: "compute",
      reqId,
      scores: scoresCopy,
      std,
      extrema: new Uint32Array(extrema),
      width,
      height,
      elapsed,
    }, { transfer: [scoresCopy.buffer, ] } as unknown as StructuredSerializeOptions);
    return;
  }

  if (msg.kind === "threshold") {
    if (!cachedScores || !cachedExtrema) {
      self.postMessage({ kind: "threshold", reqId: msg.reqId, error: "no cached scores" });
      return;
    }
    const t0 = performance.now();
    const { mask, count } = thresholdAndFill(
      cachedScores, cachedStd, cachedExtrema,
      cachedWidth, cachedHeight,
      msg.sensitivity, msg.classId,
      cachedPixels, msg.targetLab, msg.colorTolerance,
      msg.sizeRange,
    );
    const elapsed = performance.now() - t0;

    self.postMessage({
      kind: "threshold",
      reqId: msg.reqId,
      mask,
      count,
      elapsed,
    }, { transfer: [mask.buffer] });
    return;
  }

  // ---- Color distance mode ----

  if (msg.kind === "colorDist") {
    const { pixels, width, height, targetLab, reqId } = msg;
    const t0 = performance.now();
    const distMap = computeColorDistMap(pixels, width, height, targetLab);
    const elapsed = performance.now() - t0;

    cachedColorDist = distMap;
    cachedColorWidth = width;
    cachedColorHeight = height;

    // Compute stats for UI: mean and std of the distance map
    let sum = 0;
    const n = width * height;
    for (let i = 0; i < n; i++) sum += distMap[i];
    const mean = sum / n;

    const distCopy = new Float32Array(distMap);
    self.postMessage({
      kind: "colorDist",
      reqId,
      distMap: distCopy,
      mean,
      width,
      height,
      elapsed,
    }, { transfer: [distCopy.buffer] });
    return;
  }

  if (msg.kind === "colorThreshold") {
    if (!cachedColorDist) {
      self.postMessage({ kind: "colorThreshold", reqId: msg.reqId, error: "no cached color dist" });
      return;
    }
    const t0 = performance.now();
    const { mask, count } = colorThresholdAndLabel(
      cachedColorDist, cachedColorWidth, cachedColorHeight,
      msg.tolerance, msg.classId, msg.sizeRange,
    );
    const elapsed = performance.now() - t0;

    self.postMessage({
      kind: "colorThreshold",
      reqId: msg.reqId,
      mask,
      count,
      elapsed,
    }, { transfer: [mask.buffer] });
    return;
  }
};
