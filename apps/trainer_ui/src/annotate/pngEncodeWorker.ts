// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
// Web Worker: encode mask Uint8Array → PNG Blob using OffscreenCanvas
// Offloads canvas.toBlob() from the main thread during autoSave.

type EncodeInput = {
  mask: Uint8Array;
  touched: Uint8Array | null;
  width: number;
  height: number;
  id: number; // request ID for matching response
};

self.onmessage = async (e: MessageEvent<EncodeInput>) => {
  const { mask, touched, width, height, id } = e.data;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) { self.postMessage({ id, blob: null }); return; }
  const imageData = ctx.createImageData(width, height);
  const buf32 = new Uint32Array(imageData.data.buffer);
  for (let i = 0; i < mask.length; i++) {
    const v = touched && !touched[i] ? 255 : mask[i];
    buf32[i] = v | (v << 8) | (v << 16) | 0xFF000000;
  }
  ctx.putImageData(imageData, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  self.postMessage({ id, blob });
};
