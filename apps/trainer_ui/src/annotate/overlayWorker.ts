// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
// Web Worker: generates RGBA pixel buffer from mask + LUT + preview
// Supports ROI (region of interest) for viewport-aware rendering.

type ROI = { x: number; y: number; w: number; h: number };

type WorkerInput = {
  maskIndex: Uint8Array;
  lut: Uint8ClampedArray;
  width: number;
  height: number;
  preview: Uint8Array | null;
  superpixelBoundary: Uint8Array | null;
  previewStyle: number;   // 0=hatch 1=blink 2=checker
  blinkPhase: number;     // 0 or 1 (for blink mode)
  roi?: ROI;              // visible region (optional, falls back to full)
  reqId?: number;         // request ID for stale-response detection
};

self.onmessage = (e: MessageEvent<WorkerInput>) => {
  const { maskIndex, lut, width, height, preview, superpixelBoundary, previewStyle, blinkPhase, roi, reqId } = e.data;

  // Determine render region
  const rx = roi?.x ?? 0;
  const ry = roi?.y ?? 0;
  const rw = roi?.w ?? width;
  const rh = roi?.h ?? height;

  const buffer = new ArrayBuffer(rw * rh * 4);
  const buf32 = new Uint32Array(buffer);

  // Pre-build 32-bit LUT (RGBA packed as Uint32 little-endian)
  const lut32 = new Uint32Array(256);
  for (let c = 0; c < 256; c++) {
    const b = c * 4;
    lut32[c] = lut[b] | (lut[b + 1] << 8) | (lut[b + 2] << 16) | (lut[b + 3] << 24);
  }

  if (!preview) {
    for (let row = 0; row < rh; row++) {
      const srcRow = (ry + row) * width + rx;
      const dstRow = row * rw;
      for (let col = 0; col < rw; col++) {
        buf32[dstRow + col] = lut32[maskIndex[srcRow + col]];
      }
    }
  } else {
    const pStyle = previewStyle ?? 0;
    const bPhase = blinkPhase ?? 0;
    for (let row = 0; row < rh; row++) {
      const py = ry + row;
      const srcRow = py * width + rx;
      const dstRow = row * rw;
      for (let col = 0; col < rw; col++) {
        const px = rx + col;
        const i = srcRow + col;
        if (preview[i] === 254 || (preview[i] > 0 && preview[i] !== maskIndex[i])) {
          const isCandidate = preview[i] === 254;
          const color = isCandidate ? (0 | (200 << 8) | (255 << 16)) : (lut32[preview[i]] & 0x00FFFFFF);
          if (pStyle === 0) {
            if (((px + py) % 8) < 3) {
              const a = isCandidate ? 180 : 200;
              buf32[dstRow + col] = 255 | (255 << 8) | (255 << 16) | (a << 24);
            } else {
              const a = isCandidate ? 140 : 140;
              buf32[dstRow + col] = color | (a << 24);
            }
          } else if (pStyle === 1) {
            const alpha = isCandidate ? (bPhase ? 60 : 160) : (bPhase ? 30 : 180);
            buf32[dstRow + col] = color | (alpha << 24);
          } else {
            const alpha = isCandidate
              ? ((((px >> 2) + (py >> 2)) & 1) ? 160 : 80)
              : ((((px >> 2) + (py >> 2)) & 1) ? 180 : 80);
            buf32[dstRow + col] = color | (alpha << 24);
          }
        } else {
          buf32[dstRow + col] = lut32[maskIndex[i]];
        }
      }
    }
  }

  // Superpixel boundary overlay: white semi-transparent lines
  if (superpixelBoundary) {
    const boundaryColor = 255 | (255 << 8) | (255 << 16) | (80 << 24);
    for (let row = 0; row < rh; row++) {
      const srcRow = (ry + row) * width + rx;
      const dstRow = row * rw;
      for (let col = 0; col < rw; col++) {
        if (superpixelBoundary[srcRow + col]) buf32[dstRow + col] = boundaryColor;
      }
    }
  }

  // Transfer buffer (zero-copy back to main thread)
  self.postMessage({ buffer, width: rw, height: rh, roi: roi ?? null, reqId: reqId ?? 0 }, { transfer: [buffer] });
};
