// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useMemo, useState } from "react";
import { useI18n } from "../i18n";
import type { AreaUnit, InferenceResult } from "./live/types";
import { useLiveState } from "./live/hooks/useLiveState";
import { useInferenceSocket } from "./live/hooks/useInferenceSocket";
import { useWebcam } from "./live/hooks/useWebcam";
import { CameraControls } from "./live/components/CameraControls";
import { CameraPreviewLayout } from "./live/components/CameraPreviewLayout";
import { ResultsOverlay } from "./live/components/ResultsOverlay";

export default function LiveInspection({
  projectId: parentProjectId,
  projectName: parentProjectName,
  active,
  targetRunId,
  showToast,
}: {
  projectId: string;
  projectName?: string;
  active: boolean;
  targetRunId?: string;
  showToast?: (msg: string) => void;
}) {
  const toast = showToast ?? (() => {});
  const { t } = useI18n();

  // Display toggles
  const [showDetail, setShowDetail] = useState(false);
  const [showRegionCount, setShowRegionCount] = useState(false);
  const [showArea, setShowArea] = useState(false);

  // Region filters
  const [minConfidence, setMinConfidence] = useState(0); // 0–100
  const [minSize, setMinSize] = useState(0);
  const [maxSize, setMaxSize] = useState(0); // 0 = no limit

  // Calibration (px -> real unit)
  const [calPxDist, setCalPxDist] = useState(1);
  const [calRealDist, setCalRealDist] = useState(1);
  const [calUnit, setCalUnit] = useState<AreaUnit>("px");
  const pxToArea = useCallback((px: number) => {
    if (calUnit === "px" || calPxDist <= 0) return `${px.toLocaleString()} px`;
    const scale = calRealDist / calPxDist;
    const area = px * scale * scale;
    return `${area.toFixed(2)} ${calUnit}\u00B2`;
  }, [calUnit, calPxDist, calRealDist]);

  // Core state
  const state = useLiveState(parentProjectId, parentProjectName, active, targetRunId, toast);

  // Inference socket & file handling
  const socket = useInferenceSocket(state.addResult, state.session, toast);

  // Camera
  const cam = useWebcam(state.addResult, state.session, state.selectedRun, state.effectiveProjectId, toast);

  // Slider max = longer edge / 100
  const sizeSliderMax = useMemo(() => {
    const longer = Math.max(cam.cameraConfig.width, cam.cameraConfig.height);
    return Math.ceil(longer / 100);
  }, [cam.cameraConfig.width, cam.cameraConfig.height]);

  // Filter regions by size
  const filterRegions = useCallback((r: InferenceResult): InferenceResult => {
    if (minConfidence === 0 && minSize === 0 && maxSize === 0) return r;
    const minConf = minConfidence / 100;
    const filtered = r.regions.filter((reg) => {
      if (reg.confidence < minConf) return false;
      if (minSize > 0 && reg.area_px < minSize) return false;
      if (maxSize > 0 && reg.area_px > maxSize) return false;
      return true;
    });
    if (filtered.length === r.regions.length) return r;
    return {
      ...r,
      regions: filtered,
      judgement: filtered.length > 0 ? "NG" : "OK",
      defect_found: filtered.length > 0,
      summary: { ...r.summary, num_defects: filtered.length },
    };
  }, [minConfidence, minSize, maxSize]);

  const filteredResults = useMemo(
    () => state.results.map(filterRegions),
    [state.results, filterRegions],
  );

  // Wrapped start/stop that bridge camera state
  const handleStart = useCallback(() => {
    state.startSession(cam.cameraState, cam.setCameraState);
  }, [state.startSession, cam.cameraState, cam.setCameraState]);

  const handleStop = useCallback(() => {
    state.stopSession(cam.cameraState, cam.setCameraState);
  }, [state.stopSession, cam.cameraState, cam.setCameraState]);

  return (
    <div
      className={`inspect-container${socket.dragOver ? " inspect-dragover" : ""}`}
      onDragOver={socket.handleDragOver}
      onDragLeave={socket.handleDragLeave}
      onDrop={socket.handleDrop}
    >
      {/* Top bar: selectors | controls | image load */}
      <div className="inspect-topbar">
        {/* Left: project + model selectors (max 1/3) */}
        <div className="inspect-topbar-left">
          <select
            value={state.effectiveProjectId}
            onChange={(e) => state.handleProjectChange(e.target.value)}
            disabled={!!state.session}
            style={{ flex: 1, minWidth: 0 }}
          >
            <option value="">{t("live.projectPlaceholder")}</option>
            {state.allProjects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
          <select
            value={state.selectedRun}
            onChange={(e) => state.setSelectedRun(e.target.value)}
            disabled={!!state.session || !state.effectiveProjectId}
            style={{ flex: 1, minWidth: 0 }}
          >
            <option value="">{t("live.modelPlaceholder")}</option>
            {state.runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>{r.label}</option>
            ))}
          </select>
        </div>

        {/* Center: model load + camera + display toggles */}
        <div className="inspect-topbar-center">
          {!state.session ? (
            <button className="primary" onClick={handleStart} disabled={state.connecting || !state.selectedRun} style={{ whiteSpace: "nowrap", fontSize: 12 }}>
              {state.connecting ? t("live.loadingModel") : t("live.loadModel")}
            </button>
          ) : (
            <button className="danger" onClick={handleStop} style={{ whiteSpace: "nowrap", fontSize: 12 }}>{t("live.stop")}</button>
          )}
          {state.session && (
            <span className="inspect-badge inspect-badge-ok">READY</span>
          )}

          <span className="bottom-separator" />

          {/* Camera */}
          {cam.cameraState === "IDLE" ? (
            <button
              className="inspect-mode-btn"
              onClick={cam.connectCamera}
              disabled={cam.cameraConnecting}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17 10.5V7a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h12a1 1 0 001-1v-3.5l4 4v-11l-4 4z"/></svg>
              {cam.cameraConnecting ? t("live.cameraConnecting") : t("live.camera")}
            </button>
          ) : (
            <button
              className="inspect-mode-btn inspect-mode-btn-active"
              onClick={cam.disconnectCamera}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17 10.5V7a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h12a1 1 0 001-1v-3.5l4 4v-11l-4 4z"/></svg>
              {t("live.cameraStop")}
            </button>
          )}
          <button
            className="bottom-icon"
            onClick={() => cam.setShowCameraDialog(true)}
            disabled={cam.cameraState !== "IDLE"}
            title={t("live.camera.title")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.49.49 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.49.49 0 00-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.62-.07.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6A3.6 3.6 0 1112 8.4a3.6 3.6 0 010 7.2z" fill="currentColor"/></svg>
          </button>
          {cam.cameraState !== "IDLE" && (
            <button
              className="bottom-icon"
              onClick={cam.captureSnapshot}
              title={t("live.capture")}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.2" fill="currentColor"/><path d="M9 2L7.17 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V6a2 2 0 00-2-2h-3.17L15 2H9zm3 15a5 5 0 110-10 5 5 0 010 10z" fill="currentColor"/></svg>
            </button>
          )}
          {cam.cameraState !== "IDLE" && (
            <span className="muted" style={{ fontSize: 10 }}>{cam.previewFps}fps</span>
          )}

          <span className="bottom-separator" />

          {/* Display toggles */}
          <button
            className={`bottom-icon ${showDetail ? "active" : ""}`}
            onClick={() => setShowDetail((v) => !v)}
            disabled={!state.session}
            title={t("live.detailToggle.title")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z"/><circle cx="12" cy="12" r="5"/></svg>
          </button>
          <button
            className={`bottom-icon ${showRegionCount ? "active" : ""}`}
            onClick={() => setShowRegionCount((v) => !v)}
            disabled={!state.session}
            title={t("live.regionCount.title")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3h2v18H5zm6 6h2v12h-2zm6-3h2v15h-2z" fill="currentColor"/></svg>
          </button>
          <button
            className={`bottom-icon ${showArea ? "active" : ""}`}
            onClick={() => setShowArea((v) => !v)}
            disabled={!state.session}
            title={t("live.detailAreaPx")}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 3v18h18V3H3zm16 16H5V5h14v14z" fill="none" stroke="currentColor" strokeWidth="2"/><path d="M8 8h8v8H8z" fill="currentColor" opacity=".3"/></svg>
          </button>
        </div>

        {/* Right: image load (separate section) */}
        <div className="inspect-topbar-right">
          <button
            className="inspect-mode-btn"
            onClick={() => socket.fileInputRef.current?.click()}
            disabled={!state.session || socket.fileInferring}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 19V5a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2zM8.5 13.5l2.5 3 3.5-4.5 4.5 6H5l3.5-5z"/></svg>
            {socket.fileInferring ? t("live.inferring") : t("live.loadImages")}
          </button>
          <input
            ref={socket.fileInputRef}
            type="file"
            accept="image/*"
            multiple
            style={{ display: "none" }}
            onChange={socket.handleFileInfer}
          />
        </div>
      </div>
      {state.switchHint && (
        <div className="inspect-switch-hint">{state.switchHint}</div>
      )}

      {/* Camera settings dialog */}
      <CameraControls
        showCameraDialog={cam.showCameraDialog}
        setShowCameraDialog={cam.setShowCameraDialog}
        cameraConfig={cam.cameraConfig}
        saveCameraConfig={cam.saveCameraConfig}
      />

      {/* Camera preview with side panels */}
      <CameraPreviewLayout
        cameraState={cam.cameraState}
        cameraConfig={cam.cameraConfig}
        previewFps={cam.previewFps}
        previewCanvasRef={cam.previewCanvasRef}
        session={state.session}
        results={filteredResults}
        stats={state.stats}
        minConfidence={minConfidence}
        setMinConfidence={setMinConfidence}
        minSize={minSize}
        setMinSize={setMinSize}
        maxSize={maxSize}
        setMaxSize={setMaxSize}
        sizeSliderMax={sizeSliderMax}
        showRegionCount={showRegionCount}
        showArea={showArea}
        pxToArea={pxToArea}
      />

      {/* Results + stats + lightbox */}
      <ResultsOverlay
        results={filteredResults}
        stats={state.stats}
        session={state.session}
        cameraStateIsIdle={cam.cameraState === "IDLE"}
        effectiveProjectId={state.effectiveProjectId}
        runsEmpty={state.runs.length === 0}
        showDetail={showDetail}
        showRegionCount={showRegionCount}
        showArea={showArea}
        pxToArea={pxToArea}
        clearResults={state.clearResults}
        lightboxIdx={socket.lightboxIdx}
        setLightboxIdx={socket.setLightboxIdx}
      />
    </div>
  );
}
