// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

type MeasureMode = "none" | "count" | "area";
type AreaUnit = "px" | "m" | "cm" | "mm";
type Calibration = { pixelDist: number; realDist: number; unit: AreaUnit };

type ClassItem = {
  id: number;
  name: string;
  color: [number, number, number];
  active: boolean;
};

export type { MeasureMode, AreaUnit, Calibration };

/** Count connected regions per class using union-find */
export function countRegionsPerClass(mask: Uint8Array, w: number, h: number): Map<number, number> {
  const n = w * h;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  function union(a: number, b: number) {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = mask[i];
      if (v === 0) continue;
      if (x + 1 < w && mask[i + 1] === v) union(i, i + 1);
      if (y + 1 < h && mask[i + w] === v) union(i, i + w);
    }
  }
  const roots = new Set<string>();
  for (let i = 0; i < n; i++) {
    if (mask[i] === 0) continue;
    roots.add(`${mask[i]}:${find(i)}`);
  }
  const counts = new Map<number, number>();
  for (const key of roots) {
    const classId = parseInt(key.split(":")[0], 10);
    counts.set(classId, (counts.get(classId) ?? 0) + 1);
  }
  return counts;
}

/** Count area (pixel count) per class */
export function areaPerClass(mask: Uint8Array): Map<number, number> {
  const areas = new Map<number, number>();
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i];
    if (v === 0) continue;
    areas.set(v, (areas.get(v) ?? 0) + 1);
  }
  return areas;
}

export function formatArea(px: number, cal: Calibration | null): string {
  if (!cal) return `${px.toLocaleString()} px`;
  const scale = cal.realDist / cal.pixelDist; // real units per pixel
  const realArea = px * scale * scale;
  const u = cal.unit;
  if (realArea >= 1_000_000) return `${(realArea / 1_000_000).toFixed(2)} ${u}\u00B2 (M)`;
  if (realArea >= 1_000) return `${(realArea / 1_000).toFixed(1)} k${u}\u00B2`;
  return `${realArea.toFixed(2)} ${u}\u00B2`;
}

type MeasurementPanelProps = {
  measureMode: MeasureMode;
  showCount: boolean;
  showArea: boolean;
  effectiveClasses: ClassItem[];
  regionCounts: Map<number, number>;
  classAreas: Map<number, number>;
  calibration: Calibration | null;
};

export default React.memo(function MeasurementPanel({
  showCount,
  showArea,
  effectiveClasses,
  regionCounts,
  classAreas,
  calibration,
}: MeasurementPanelProps) {
  if (!showCount && !showArea) return null;

  const title = showCount && showArea
    ? `Count + Area${calibration ? ` (${calibration.unit}\u00B2)` : ""}`
    : showCount
      ? "Region Count"
      : `Area${calibration ? ` (${calibration.unit}\u00B2)` : " (px)"}`;

  return (
    <div className="measure-panel">
      <div className="measure-panel-title">{title}</div>
      {effectiveClasses.filter((c) => c.id > 0).map((cls) => {
        const count = regionCounts.get(cls.id) ?? 0;
        const area = classAreas.get(cls.id) ?? 0;
        if (count === 0 && area === 0) return null;
        return (
          <div key={cls.id} className="measure-row">
            <span className="measure-color" style={{ background: `rgb(${cls.color.join(",")})` }} />
            <span className="measure-name">{cls.name}</span>
            <span className="measure-value">
              {showCount && <span>{count} rgn</span>}
              {showCount && showArea && <span style={{ opacity: 0.4, margin: "0 3px" }}>|</span>}
              {showArea && <span>{formatArea(area, calibration)}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
});
