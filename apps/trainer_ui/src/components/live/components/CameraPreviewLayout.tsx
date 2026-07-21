// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useRef } from "react";
import { useI18n } from "../../../i18n";
import type { CameraConfig, CameraState, AreaUnit, InferenceResult, InferenceStats, SessionInfo } from "../types";

/** Match backend _PALETTE for consistent class colors */
const CLASS_COLORS = [
  "rgba(0,0,0,0)",        // 0: background
  "rgba(255,60,60,0.85)",  // 1: red
  "rgba(60,120,255,0.85)", // 2: blue
  "rgba(60,220,60,0.85)",  // 3: green
  "rgba(255,180,0,0.85)",  // 4: orange
  "rgba(213,94,0,0.85)",   // 5: vermilion
  "rgba(0,220,220,0.85)",  // 6: cyan
  "rgba(255,255,60,0.85)", // 7: yellow
  "rgba(255,100,180,0.85)",// 8: pink
];

type CameraPreviewLayoutProps = {
  cameraState: CameraState;
  cameraConfig: CameraConfig;
  previewFps: number;
  previewCanvasRef: React.RefObject<HTMLCanvasElement>;
  session: SessionInfo | null;
  results: InferenceResult[];
  stats: InferenceStats;
  showRegionCount: boolean;
  showArea: boolean;
  pxToArea: (px: number) => string;
  minConfidence: number;
  setMinConfidence: (v: number) => void;
  minSize: number;
  setMinSize: (v: number) => void;
  maxSize: number;
  setMaxSize: (v: number) => void;
  sizeSliderMax: number;
};

export function CameraPreviewLayout({
  cameraState,
  cameraConfig,
  previewFps,
  previewCanvasRef,
  session,
  results,
  stats,
  showRegionCount,
  showArea,
  pxToArea,
  minConfidence,
  setMinConfidence,
  minSize,
  setMinSize,
  maxSize,
  setMaxSize,
  sizeSliderMax,
}: CameraPreviewLayoutProps) {
  const { t } = useI18n();

  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  // Draw bbox + class labels on overlay canvas
  useEffect(() => {
    const overlay = overlayCanvasRef.current;
    const preview = previewCanvasRef.current;
    if (!overlay || !preview) return;
    const lastR = results.length > 0 ? results[results.length - 1]! : null;

    // Match overlay size to preview canvas internal size
    overlay.width = preview.width;
    overlay.height = preview.height;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!lastR || lastR.regions.length === 0) return;

    for (const reg of lastR.regions) {
      const [bx, by, bw, bh] = reg.bbox;
      const color = CLASS_COLORS[reg.class_id % CLASS_COLORS.length] ?? "rgba(255,60,60,0.85)";

      // BBox
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, Math.round(overlay.width / 400));
      ctx.strokeRect(bx, by, bw, bh);

      // Label
      const fontSize = Math.max(11, Math.round(overlay.width / 60));
      ctx.font = `bold ${fontSize}px sans-serif`;
      const label = `${reg.class} ${(reg.confidence * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width;
      const labelY = by > fontSize + 6 ? by - 4 : by + bh + fontSize + 2;
      const bgY = labelY - fontSize;
      ctx.fillStyle = color;
      ctx.fillRect(bx, bgY - 2, tw + 8, fontSize + 6);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 4, labelY);
    }
  }, [results, previewCanvasRef]);

  if (cameraState === "IDLE") return null;

  const lastResult = results.length > 0 ? results[results.length - 1]! : null;

  return (
    <div className="inspect-preview-layout">
      {/* Left: Camera info + Status */}
      <div className="inspect-side-panel inspect-side-left">
        <div className="inspect-side-section">
          <div className="inspect-side-title">{t("live.cameraInfo")}</div>
          <div className="inspect-side-row"><span className="muted">Device</span><span>{cameraConfig.device_id}</span></div>
          <div className="inspect-side-row"><span className="muted">Resolution</span><span>{cameraConfig.width}×{cameraConfig.height}</span></div>
          <div className="inspect-side-row"><span className="muted">FPS</span><span>{cameraConfig.fps}</span></div>
          <div className="inspect-side-row"><span className="muted">Preview</span><span>{previewFps} fps</span></div>
        </div>
        <div className="inspect-side-section">
          <div className="inspect-side-title">{t("live.status")}</div>
          <div className="inspect-side-row">
            <span className="muted">State</span>
            <span className={`inspect-side-badge inspect-side-badge-${cameraState.toLowerCase()}`}>{cameraState}</span>
          </div>
          <div className="inspect-side-row"><span className="muted">Session</span><span>{session ? "Active" : "—"}</span></div>
          {session && <div className="inspect-side-row"><span className="muted">Device</span><span>{session.device}</span></div>}
          {session && <div className="inspect-side-row"><span className="muted">Model</span><span style={{ fontSize: 10 }}>{session.model_id.slice(0, 8)}</span></div>}
        </div>
      </div>

      {/* Center: Camera preview */}
      <div className="inspect-preview">
        <canvas
          ref={previewCanvasRef}
          style={{ width: "100%", maxHeight: "70vh", objectFit: "contain", background: "#111" }}
        />
        {lastResult?.maskUrl && (
          <img
            src={lastResult.maskUrl}
            className="inspect-preview-mask"
            alt=""
          />
        )}
        <canvas
          ref={overlayCanvasRef}
          className="inspect-preview-mask"
          style={{ pointerEvents: "none" }}
        />
        <div className="inspect-preview-badge">
          <span className={`inspect-badge ${cameraState === "INSPECT" ? "inspect-badge-ng" : "inspect-badge-ok"}`}>
            {cameraState === "INSPECT" ? "INSPECT" : "PREVIEW"}
          </span>
        </div>
      </div>

      {/* Right: Confidence bar + Latency + Stats + Display + Detected classes */}
      <div className="inspect-side-panel inspect-side-right">
        <div className="inspect-side-section">
          <div className="inspect-side-title">Confidence</div>
          <div className="inspect-side-row">
            <span style={{ fontSize: 10, minWidth: 32 }}>{minConfidence}%</span>
            <input
              type="range" min={0} max={100} value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="inspect-size-slider"
            />
          </div>
        </div>
        <div className="inspect-side-section">
          <div className="inspect-side-title">{t("live.latency")}</div>
          {lastResult ? (() => {
            const last = lastResult.latency_ms;
            return (
              <>
                <div className="inspect-side-row"><span className="muted">Total</span><span>{last.total?.toFixed(0) ?? "—"}ms</span></div>
                <div className="inspect-side-row"><span className="muted">Infer</span><span>{last.inference?.toFixed(0) ?? "—"}ms</span></div>
                <div className="inspect-side-row"><span className="muted">Pre</span><span>{last.preprocess?.toFixed(0) ?? "—"}ms</span></div>
                <div className="inspect-side-row"><span className="muted">Post</span><span>{last.postprocess?.toFixed(0) ?? "—"}ms</span></div>
              </>
            );
          })() : <div className="muted" style={{ fontSize: 11 }}>—</div>}
        </div>
        <div className="inspect-side-section">
          <div className="inspect-side-title">{t("live.throughput")}</div>
          <div className="inspect-side-row"><span className="muted">Total</span><span>{stats.total}</span></div>
          <div className="inspect-side-row"><span className="muted">OK</span><span className="inspect-ok">{stats.ok}</span></div>
          <div className="inspect-side-row"><span className="muted">NG</span><span className="inspect-ng">{stats.ng}</span></div>
          <div className="inspect-side-row"><span className="muted">Avg</span><span>{stats.avgMs.toFixed(1)}ms</span></div>
        </div>
        <div className="inspect-side-section">
          <div className="inspect-side-title">Size Filter</div>
          <div className="inspect-side-row">
            <span className="muted" style={{ minWidth: 28 }}>Min</span>
            <span style={{ fontSize: 10, minWidth: 32 }}>{minSize}</span>
            <input
              type="range" min={0} max={sizeSliderMax} value={minSize}
              onChange={(e) => setMinSize(Number(e.target.value))}
              className="inspect-size-slider"
            />
          </div>
          <div className="inspect-side-row">
            <span className="muted" style={{ minWidth: 28 }}>Max</span>
            <span style={{ fontSize: 10, minWidth: 32 }}>{maxSize === 0 ? "∞" : maxSize}</span>
            <input
              type="range" min={0} max={sizeSliderMax} value={maxSize}
              onChange={(e) => setMaxSize(Number(e.target.value))}
              className="inspect-size-slider"
            />
          </div>
        </div>
        <div className="inspect-side-section">
          <div className="inspect-side-title">{t("live.detectedClasses")}</div>
          {lastResult && lastResult.regions.length > 0 ? (
            lastResult.regions.map((r, i) => (
              <div key={i} className="inspect-side-row">
                <span>{r.class}</span>
                <span>
                  <span className="muted">{(r.confidence * 100).toFixed(0)}%</span>
                  {showArea && <span className="muted" style={{ marginLeft: 4 }}>{pxToArea(r.area_px)}</span>}
                </span>
              </div>
            ))
          ) : (
            <div className="muted" style={{ fontSize: 11 }}>—</div>
          )}
          {lastResult && showRegionCount && (
            <div className="inspect-side-row" style={{ marginTop: 2, borderTop: "1px solid var(--border)", paddingTop: 2 }}>
              <span className="muted">Defects</span><span>{lastResult.summary.num_defects}</span>
            </div>
          )}
          {lastResult && showArea && (
            <div className="inspect-side-row">
              <span className="muted">FG ratio</span><span>{(lastResult.summary.fg_ratio * 100).toFixed(2)}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
