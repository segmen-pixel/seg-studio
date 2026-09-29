// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
// Web Worker: connected-component region labels via Union-Find.
// Offloaded from main thread to avoid blocking UI during brush strokes.

type RegionLabelInput = {
  maskIndex: Uint8Array;
  width: number;
  height: number;
  activeClassIds: number[];
  classNames: Record<number, string>;
  classColors: Record<number, [number, number, number]>;
  reqId: number;
  // Optional area filters in pixels. 0 or undefined means "no cutoff".
  // Mirrors the Min/Max Area controls in the Results toolbar so labels
  // stay consistent with the heatmap.
  minArea?: number;
  maxArea?: number;
};

type RegionLabel = {
  classId: number;
  cx: number;
  topY: number;
  count: number;
  name: string;
  color: [number, number, number];
};

self.onmessage = (e: MessageEvent<RegionLabelInput>) => {
  const { maskIndex, width, height, activeClassIds, classNames, classColors, reqId, minArea, maxArea } = e.data;
  const minA = minArea && minArea > 0 ? minArea : 0;
  const maxA = maxArea && maxArea > 0 ? maxArea : 0;
  const n = maskIndex.length;
  if (n === 0 || width === 0 || height === 0) {
    self.postMessage({ labels: [], reqId });
    return;
  }

  const activeSet = new Set(activeClassIds);

  // Union-Find with path compression + union by rank
  const par = new Int32Array(n).fill(-1);
  const rnk = new Uint8Array(n);

  function find(x: number): number {
    while (par[x] !== x) {
      par[x] = par[par[x]!]!;
      x = par[x]!;
    }
    return x;
  }

  function unite(a: number, b: number) {
    const ra = find(a), rb = find(b);
    if (ra === rb) return;
    if (rnk[ra]! < rnk[rb]!) par[ra] = rb;
    else if (rnk[ra]! > rnk[rb]!) par[rb] = ra;
    else { par[rb] = ra; rnk[ra] = rnk[ra]! + 1; }
  }

  // Pass 1: build Union-Find (skip background pixels)
  for (let i = 0; i < n; i++) {
    if (maskIndex[i] === 0) continue;
    par[i] = i;
    const x = i % width;
    if (x > 0 && maskIndex[i - 1] === maskIndex[i]) unite(i, i - 1);
    if (i >= width && maskIndex[i - width] === maskIndex[i]) unite(i, i - width);
  }

  // Pass 2: collect regions
  const regions = new Map<number, {
    classId: number; minX: number; minY: number;
    maxX: number; maxY: number; count: number;
  }>();

  for (let i = 0; i < n; i++) {
    if (maskIndex[i] === 0) continue;
    const root = find(i);
    const x = i % width, y = (i - x) / width;
    let r = regions.get(root);
    if (!r) {
      r = { classId: maskIndex[i]!, minX: x, minY: y, maxX: x, maxY: y, count: 0 };
      regions.set(root, r);
    }
    if (x < r.minX) r.minX = x;
    if (x > r.maxX) r.maxX = x;
    if (y < r.minY) r.minY = y;
    if (y > r.maxY) r.maxY = y;
    r.count++;
  }

  // Build labels — keep the 5px floor as a noise guard, then honor the
  // user's explicit Min/Max Area cutoffs from the toolbar so labels match
  // the area-filtered heatmap.
  const labels: RegionLabel[] = [];
  regions.forEach((r) => {
    if (r.count < 5) return;
    if (minA > 0 && r.count < minA) return;
    if (maxA > 0 && r.count > maxA) return;
    if (!activeSet.has(r.classId)) return;
    labels.push({
      classId: r.classId,
      cx: (r.minX + r.maxX) / 2,
      topY: r.minY,
      count: r.count,
      name: classNames[r.classId] ?? `Class ${r.classId}`,
      color: classColors[r.classId] ?? [255, 255, 255],
    });
  });

  self.postMessage({ labels, reqId });
};
