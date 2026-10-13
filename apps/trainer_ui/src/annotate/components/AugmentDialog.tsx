// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useI18n } from "../../i18n";
import type { ClassItem } from "../../store";

// ---------------------------------------------------------------------------
// Client-side preview canvas: draws a stylised "background + circular defect"
// then applies a Perlin warp + color jitter so the user can get a feel for
// what the slider values produce. This is illustrative only — the real
// synthesis on the server uses the project's own images.
// ---------------------------------------------------------------------------

const PREVIEW_W = 480;
const PREVIEW_H = 240;

// Tiny deterministic pseudo-Perlin value noise generator (good enough for
// preview purposes; not the full Perlin).
function makeNoiseField(seed: number) {
  const rand = (i: number, j: number) => {
    const n = Math.sin(i * 12.9898 + j * 78.233 + seed * 43.123) * 43758.5453;
    return n - Math.floor(n); // [0, 1)
  };
  const smooth = (x: number) => x * x * (3 - 2 * x);
  return (x: number, y: number, freq: number) => {
    const fx = x * freq;
    const fy = y * freq;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = smooth(fx - x0);
    const ty = smooth(fy - y0);
    const v00 = rand(x0, y0);
    const v10 = rand(x0 + 1, y0);
    const v01 = rand(x0, y0 + 1);
    const v11 = rand(x0 + 1, y0 + 1);
    const a = v00 + (v10 - v00) * tx;
    const b = v01 + (v11 - v01) * tx;
    return (a + (b - a) * ty) * 2 - 1; // [-1, 1]
  };
}

function drawBaseImage(ctx: CanvasRenderingContext2D, classColor: [number, number, number]) {
  // Gradient background
  const grad = ctx.createLinearGradient(0, 0, PREVIEW_W, PREVIEW_H);
  grad.addColorStop(0, "#b8b6a5");
  grad.addColorStop(1, "#5a5848");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, PREVIEW_W, PREVIEW_H);
  // A few fake fibres to make the "background" visually noisy
  ctx.strokeStyle = "rgba(0,0,0,0.15)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 36; i++) {
    const seed = (i * 73) % 256;
    const x0 = (seed * 37) % PREVIEW_W;
    const y0 = (seed * 91) % PREVIEW_H;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + 20, y0 + 6);
    ctx.stroke();
  }
  // Three "defect" blobs (circles) that will get warped
  const [cr, cg, cb] = classColor;
  const defectStyle = `rgba(${cr},${cg},${cb},0.85)`;
  ctx.fillStyle = defectStyle;
  ctx.beginPath();
  ctx.arc(150, 110, 38, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(310, 150, 28, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(400, 80, 22, 0, Math.PI * 2);
  ctx.fill();
}

function renderPreview(
  canvas: HTMLCanvasElement,
  perlinStrength: number,
  colorJitter: number,
  classColor: [number, number, number],
  seed: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  // Step 1: draw the base image onto an offscreen buffer
  const off = document.createElement("canvas");
  off.width = PREVIEW_W;
  off.height = PREVIEW_H;
  const octx = off.getContext("2d");
  if (!octx) return;
  drawBaseImage(octx, classColor);
  const src = octx.getImageData(0, 0, PREVIEW_W, PREVIEW_H);

  // Step 2: build the warped image by sampling src through a Perlin-ish field
  const dst = ctx.createImageData(PREVIEW_W, PREVIEW_H);
  const noiseX = makeNoiseField(seed);
  const noiseY = makeNoiseField(seed + 97);
  // Frequency controls feature size — lower freq = larger warps
  const freq = 0.03;
  // Scale perlinStrength from slider [0..20] → displacement pixels
  const amp = perlinStrength;
  // Color jitter — simple per-channel multiplicative offset, same across preview
  const jr = 1 + (noiseX(7, 11, 0.1) * 0.5) * colorJitter;
  const jg = 1 + (noiseX(13, 17, 0.1) * 0.5) * colorJitter;
  const jb = 1 + (noiseX(19, 23, 0.1) * 0.5) * colorJitter;
  for (let y = 0; y < PREVIEW_H; y++) {
    for (let x = 0; x < PREVIEW_W; x++) {
      const dx = noiseX(x, y, freq) * amp;
      const dy = noiseY(x, y, freq) * amp;
      const sx = Math.min(PREVIEW_W - 1, Math.max(0, Math.round(x + dx)));
      const sy = Math.min(PREVIEW_H - 1, Math.max(0, Math.round(y + dy)));
      const si = (sy * PREVIEW_W + sx) * 4;
      const di = (y * PREVIEW_W + x) * 4;
      dst.data[di] = Math.min(255, Math.max(0, src.data[si] * jr));
      dst.data[di + 1] = Math.min(255, Math.max(0, src.data[si + 1] * jg));
      dst.data[di + 2] = Math.min(255, Math.max(0, src.data[si + 2] * jb));
      dst.data[di + 3] = 255;
    }
  }
  ctx.putImageData(dst, 0, 0);
}

// ---------------------------------------------------------------------------
// Dialog component
// ---------------------------------------------------------------------------

export type AugmentLightingVariant = "daytime" | "evening" | "night";

export type AugmentDialogProps = {
  open: boolean;
  classes: ClassItem[];
  onClose: () => void;
  onConfirm: (params: {
    count: number;
    classId: number;
    perlinStrength: number;
    colorJitter: number;
    defectsPerImage: [number, number];
    modePerlin: boolean;
    modeLighting: boolean;
    lightingVariants: AugmentLightingVariant[];
    useCleanHosts: boolean;
  }) => void;
  running: boolean;
  lastError: string | null;
};

export function AugmentDialog({
  open,
  classes,
  onClose,
  onConfirm,
  running,
  lastError,
}: AugmentDialogProps) {
  const { t } = useI18n();
  const [count, setCount] = useState(10);
  // Non-background classes available for synthesis.
  const fgClasses = useMemo(() => classes.filter((c) => c.id !== 0), [classes]);
  const hasMultipleClasses = fgClasses.length >= 2;
  // Default: "all classes" (id=0) when the project has 2+ classes,
  // otherwise the only available fg class.
  const [classId, setClassId] = useState<number>(
    hasMultipleClasses ? 0 : (fgClasses[0]?.id ?? 1),
  );
  const [perlinStrength, setPerlinStrength] = useState(6);
  const [colorJitter, setColorJitter] = useState(0.15);
  const [defectsMin, setDefectsMin] = useState(1);
  const [defectsMax, setDefectsMax] = useState(4);
  const [previewSeed, setPreviewSeed] = useState(0);
  // Mode toggles — at least one must be on for Generate to enable.
  const [modePerlin, setModePerlin] = useState(true);
  const [modeLighting, setModeLighting] = useState(false);
  const [lightingDaytime, setLightingDaytime] = useState(true);
  const [lightingEvening, setLightingEvening] = useState(true);
  const [lightingNight, setLightingNight] = useState(true);
  // Mix verified-OK ("Mark Clean") images into the Perlin paste-target pool.
  // Off by default to preserve legacy behaviour.
  const [useCleanHosts, setUseCleanHosts] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const activeClass = useMemo(
    () => classes.find((c) => c.id === classId) ?? classes.find((c) => c.id !== 0),
    [classes, classId],
  );
  // Preview colour: specific class → its colour, "all classes" (0) → first
  // fg class so the preview still shows something representative.
  const classColor: [number, number, number] = useMemo(() => {
    const src = classId === 0 ? fgClasses[0] : activeClass;
    if (!src) return [220, 60, 60];
    const c = src.color;
    return [c[0], c[1], c[2]];
  }, [classId, fgClasses, activeClass]);

  // Re-render preview whenever any input changes
  useEffect(() => {
    if (!open) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    renderPreview(canvas, perlinStrength, colorJitter, classColor, previewSeed);
  }, [open, perlinStrength, colorJitter, classColor, previewSeed]);

  const selectedLightingVariants = useMemo<AugmentLightingVariant[]>(() => {
    const vs: AugmentLightingVariant[] = [];
    if (lightingDaytime) vs.push("daytime");
    if (lightingEvening) vs.push("evening");
    if (lightingNight) vs.push("night");
    return vs;
  }, [lightingDaytime, lightingEvening, lightingNight]);

  const canGenerate =
    (modePerlin || (modeLighting && selectedLightingVariants.length > 0)) && !running;

  // Server processes each enabled mode for `count` samples, so enabling both
  // doubles the total. Reflect that in the running button so the user knows
  // how many they asked for.
  const totalCount =
    (modePerlin ? count : 0) +
    (modeLighting && selectedLightingVariants.length > 0 ? count : 0);

  const handleConfirm = useCallback(() => {
    if (!canGenerate) return;
    const lo = Math.max(1, Math.min(defectsMin, defectsMax));
    const hi = Math.max(lo, defectsMax);
    onConfirm({
      count,
      classId,
      perlinStrength,
      colorJitter,
      defectsPerImage: [lo, hi],
      modePerlin,
      modeLighting,
      lightingVariants: selectedLightingVariants,
      useCleanHosts,
    });
  }, [canGenerate, count, classId, perlinStrength, colorJitter, defectsMin, defectsMax, modePerlin, modeLighting, selectedLightingVariants, useCleanHosts, onConfirm]);

  if (!open) return null;

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-content augment-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>{t("augment.title")}</h3>
        <div className="augment-body">
          <div className="augment-preview-row">
            <canvas
              ref={canvasRef}
              width={PREVIEW_W}
              height={PREVIEW_H}
              className="augment-preview-canvas"
              title={t("augment.previewHint")}
            />
            <button
              className="ghost compact"
              onClick={() => setPreviewSeed((s) => s + 1)}
              title={t("augment.reroll")}
            >
              ↻
            </button>
          </div>
          <div className="augment-form">
            <label>{t("augment.modes")}</label>
            <div className="augment-modes-wrap" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div className="augment-mode-row" style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <input
                    type="checkbox"
                    checked={modePerlin}
                    onChange={(e) => setModePerlin(e.target.checked)}
                  />
                  {t("augment.mode.perlin")}
                </label>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <input
                    type="checkbox"
                    checked={modeLighting}
                    onChange={(e) => setModeLighting(e.target.checked)}
                  />
                  {t("augment.mode.lighting")}
                </label>
              </div>
              <div
                className="augment-perlin-options"
                style={{
                  display: "flex",
                  gap: 10,
                  flexWrap: "wrap",
                  opacity: modePerlin ? 1 : 0.4,
                  pointerEvents: modePerlin ? "auto" : "none",
                }}
                aria-disabled={!modePerlin}
              >
                <label
                  style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12 }}
                  title={t("augment.useCleanHostsHint")}
                >
                  <input
                    type="checkbox"
                    checked={useCleanHosts}
                    disabled={!modePerlin}
                    onChange={(e) => setUseCleanHosts(e.target.checked)}
                  />
                  {t("augment.useCleanHosts")}
                </label>
              </div>
              <div
                className="augment-lighting-variants"
                style={{
                  display: "flex",
                  gap: 10,
                  flexWrap: "wrap",
                  opacity: modeLighting ? 1 : 0.4,
                  pointerEvents: modeLighting ? "auto" : "none",
                }}
                aria-disabled={!modeLighting}
              >
                <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12 }}>
                  <input
                    type="checkbox"
                    checked={lightingDaytime}
                    disabled={!modeLighting}
                    onChange={(e) => setLightingDaytime(e.target.checked)}
                  />
                  {t("augment.lighting.daytime")}
                </label>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12 }}>
                  <input
                    type="checkbox"
                    checked={lightingEvening}
                    disabled={!modeLighting}
                    onChange={(e) => setLightingEvening(e.target.checked)}
                  />
                  {t("augment.lighting.evening")}
                </label>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 12 }}>
                  <input
                    type="checkbox"
                    checked={lightingNight}
                    disabled={!modeLighting}
                    onChange={(e) => setLightingNight(e.target.checked)}
                  />
                  {t("augment.lighting.night")}
                </label>
              </div>
            </div>

            <label htmlFor="aug-count">{t("augment.count")}</label>
            <input
              id="aug-count"
              type="number"
              min={1}
              max={500}
              value={count}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (!isNaN(v)) setCount(Math.max(1, Math.min(500, v)));
              }}
            />

            <label htmlFor="aug-class">{t("augment.classLabel")}</label>
            <select
              id="aug-class"
              value={classId}
              onChange={(e) => setClassId(parseInt(e.target.value, 10))}
            >
              {hasMultipleClasses && (
                <option value={0}>{t("augment.allClasses")}</option>
              )}
              {fgClasses.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} (#{c.id})
                </option>
              ))}
            </select>

            <label htmlFor="aug-perlin">
              {t("augment.perlinStrength")}: <strong>{perlinStrength.toFixed(1)}</strong>
            </label>
            <input
              id="aug-perlin"
              type="range"
              min={0}
              max={20}
              step={0.5}
              value={perlinStrength}
              onChange={(e) => setPerlinStrength(parseFloat(e.target.value))}
            />

            <label htmlFor="aug-jitter">
              {t("augment.colorJitter")}: <strong>{colorJitter.toFixed(2)}</strong>
            </label>
            <input
              id="aug-jitter"
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={colorJitter}
              onChange={(e) => setColorJitter(parseFloat(e.target.value))}
            />

            <label htmlFor="aug-defects-min">{t("augment.defectsPerImage")}</label>
            <div className="augment-range-pair">
              <input
                id="aug-defects-min"
                type="number"
                min={1}
                max={20}
                value={defectsMin}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v)) setDefectsMin(Math.max(1, Math.min(20, v)));
                }}
              />
              <span>〜</span>
              <input
                type="number"
                min={1}
                max={20}
                value={defectsMax}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v)) setDefectsMax(Math.max(1, Math.min(20, v)));
                }}
              />
            </div>
          </div>
          {lastError ? <div className="augment-error">{lastError}</div> : null}
        </div>
        <div className="augment-footer">
          <button className="ghost" onClick={onClose} disabled={running}>
            {t("common.cancel")}
          </button>
          <button className="primary" onClick={handleConfirm} disabled={!canGenerate}>
            {running
              ? t("augment.generating").replace("{count}", String(totalCount))
              : t("augment.generate")}
          </button>
        </div>
      </div>
    </div>
  );
}
