// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect } from "react";
import { type Project } from "../api";
import { useI18n } from "../i18n";
import ImageListPanel from "./ImageListPanel";
import MeasurementPanel from "./MeasurementPanel";
import type { AreaUnit } from "./MeasurementPanel";
import { useResultsState } from "./hooks/useResultsState";
import { useInferenceEngine } from "./hooks/useInferenceEngine";
import { useCanvasRendering } from "./hooks/useCanvasRendering";
import ResultsToolbar from "./components/ResultsToolbar";
import ActionBar from "./components/ActionBar";
import ExportDialog from "./components/ExportDialog";
import ReportModal from "../reports/ReportModal";
import MetricsSection from "./components/MetricsSection";
import PredictionDetail from "./components/PredictionDetail";

export default React.memo(function Results({
  projectId,
  projects: _projects,
  onProjectChange: _onProjectChange,
  active,
  showToast,
  onInferStatus,
  runId,
  onClose: _onClose,
  onGoInspect,
  onOpenReport,
}: {
  projectId: string | null;
  projects: Project[];
  onProjectChange: (id: string) => void;
  active?: boolean;
  showToast?: (msg: string) => void;
  onInferStatus?: (msg: string) => void;
  runId?: string;
  onClose?: () => void;
  onGoInspect?: (runId: string) => void;
  onOpenReport?: (reportId: string, runId: string, label: string) => void;
}) {
  const { t, lang } = useI18n();
  const s = useResultsState(projectId, runId);
  const [exportOpen, setExportOpen] = React.useState(false);
  const [reportOpen, setReportOpen] = React.useState(false);
  const reportRun = s.runs.find((r) => r.run_id === s.activeRunId);

  const engine = useInferenceEngine(
    s, projectId, active, showToast, onInferStatus, runId, t, onGoInspect,
  );

  const canvas = useCanvasRendering(s);

  // Consolidate prediction-loading status into the global toast (single status location)
  useEffect(() => {
    if (!active || s.isInferring) return;
    onInferStatus?.(s.isPredictionLoading ? t("results.predictionLoading") : "");
  }, [active, s.isInferring, s.isPredictionLoading, onInferStatus, t]);

  useEffect(() => {
    if (!active) return;
    function handleKey(event: KeyboardEvent) {
      if (event.defaultPrevented) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const el = document.activeElement as HTMLElement | null;
      if (el) {
        const tag = el.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable) return;
      }
      const key = event.key.toLowerCase();
      switch (key) {
        case "p":
          s.setShowGtOutline((v) => !v); break;
        case "l":
          s.setShowRegionLabels((v) => !v); break;
        case "c":
          if (s.isInstanceRun) return;
          s.setHeatmapMode((m) => m === "confidence" ? "none" : "confidence"); break;
        case "h":
          if (s.isInstanceRun) return;
          s.setHeatmapMode((m) => m === "class" ? "none" : "class"); break;
        case "e":
          if (s.isInstanceRun) return;
          s.setHeatmapMode((m) => m === "error" ? "none" : "error"); break;
        case "n":
          s.setShowCount((v) => !v); break;
        case "a":
          s.setShowArea((v) => !v); break;
        default:
          return;
      }
      event.preventDefault();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [active, s]);

  return (
    <div className="results-root">
      <div className="results-layout results-layout-single" style={{ marginTop: 0 }}>
        <ImageListPanel
          images={s.images}
          filterSet={s.filterSet}
          onFilterChange={s.setFilterSet}
          keyboardActive={active}
          activeImageId={s.activeImageId}
          selectedIds={s.selectedIds}
          onSelectedIdsChange={s.setSelectedIds}
          onSelectImage={engine.selectImage}
          onApplyPredToLabel={engine.applyPredToLabel}
          onBulkApplyPredToLabel={engine.bulkApplyPredToLabel}
          onClearOkLabels={engine.clearOkLabels}
          onMoveSelection={engine.handleMoveImageSelection}
          onRefresh={() => projectId && engine.loadDataset(projectId)}
          perImageClassIds={s.perImageClassIds}
          classColorMap={s.classColorMap}
          visibleCount={s.filteredImages.length}
          totalCount={s.images.length}
        />
        <div className="preview-panel">
          <div className="section preview-frame">
            <div className="results-preview-meta">
              <div style={{ display: "flex", alignItems: "flex-start", gap: 8, flexWrap: "nowrap" }}>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="results-preview-title">{s.activeImage?.name ?? t("results.imageNotSelected")}</div>
                  <div className="results-preview-subtitle">
                    {s.isDatasetLoading
                      ? t("results.loadingImages")
                      : s.activeImage
                        ? `${s.activeImage.width}x${s.activeImage.height} px`
                        : t("results.selectImageHint")}
                    {s.activeImage ? ` · ${s.activeImage.set}` : ""}
                    {s.isInstanceRun && s.instanceData && (() => {
                      const data = s.instanceData;
                      const nameOf = (cid: number) =>
                        s.effectiveClasses.find((c) => c.id === cid)?.name
                        ?? data.class_names?.[String(cid)]
                        ?? `class${cid}`;
                      // Multi-class runs report a count per class; fall back
                      // to the single total for older single-class results.
                      const entries = Object.entries(data.counts_by_class ?? {})
                        .map(([cid, n]) => [Number(cid), n] as const)
                        .sort((a, b) => a[0] - b[0]);
                      const shown = entries.length > 0
                        ? entries
                        : [[s.presentClassIds[0]
                            ?? s.effectiveClasses.find((c) => c.id > 0)?.id
                            ?? 1, data.count] as const];
                      return shown.map(([cid, n]) => (
                        <span className="instance-count-chip" key={cid}
                          data-testid="instance-count-chip">
                          {t("results.instanceCountClass")
                            .replace("{cls}", nameOf(cid))
                            .replace("{n}", String(n))}
                        </span>
                      ));
                    })()}
                  </div>
                </div>
                {projectId && s.activeRunId && (
                  <div className="results-preview-actions" style={{ display: "flex", gap: 6, marginLeft: "auto", alignItems: "flex-start" }}>
                    <button className="ghost results-header-btn" onClick={() => setExportOpen(true)} disabled={!s.activeRunId} title={t("results.exportModelTitle")} data-desc={t("results.exportModelDescLong")}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                      {t("results.exportModel")}
                    </button>
                    <button className="ghost results-header-btn" onClick={() => setReportOpen(true)} disabled={!reportRun?.has_model || s.isInstanceRun} title={s.isInstanceRun ? t("results.instanceReportSoon") : t("training.generateReport")} data-desc={s.isInstanceRun ? t("results.instanceReportSoon") : t("training.generateReportDesc")}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                      {t("results.reportShort")}
                    </button>
                    {s.isInferring ? (
                      <button className="ghost results-header-btn results-stop-btn" onClick={() => engine.handleStopInference()} title={t("results.stopInference")}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>
                        {t("results.stopShort")}
                      </button>
                    ) : (
                      <button className="primary results-header-btn" onClick={() => engine.handleRunInference()} disabled={!s.activeRunId} title={t("results.runInference")} data-desc={t("results.runInferenceDesc")}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3c.3 3.6 2.4 5.7 6 6-3.6.3-5.7 2.4-6 6-.3-3.6-2.4-5.7-6-6 3.6-.3 5.7-2.4 6-6z"/></svg>
                        {t("results.runInference")}
                      </button>
                    )}
                    {!s.isInferring && s.activeRunId && (s.inferredRuns.get(s.activeRunId)?.size ?? 0) > 0 && (
                      <button className="ghost results-header-btn" onClick={() => engine.handleRestoreCache()} title={t("results.restoreCache")} data-desc={t("results.restoreCacheDesc")}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><polyline points="3 3 3 8 8 8"/></svg>
                        {t("results.restoreCache")}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="canvas-stack" ref={s.previewRef}
              onWheel={canvas.handlePreviewWheel}
              onPointerDown={canvas.handlePreviewPointerDown}
              onPointerMove={canvas.handlePreviewPointerMove}
              onPointerUp={canvas.handlePreviewPointerUp}
              onContextMenu={(e) => {
                if (s.measureMode === "area") { e.preventDefault(); s.setUnitMenuPos({ x: e.clientX, y: e.clientY }); }
              }}
              style={{ cursor: s.panRef.current ? "grabbing" : "grab" }}
            >
              {s.activeImage?.set && s.activeImage.set !== "none" && (
                <div className={`preview-set-badge preview-set-${s.activeImage.set}`}>
                  {s.activeImage.set.toUpperCase()}
                </div>
              )}
              <div
                className="canvas-inner"
                style={{
                  width: s.width || 0,
                  height: s.height || 0,
                  transform: `translate(${s.previewOffset.x}px, ${s.previewOffset.y}px) scale(${s.previewScale})`,
                  transformOrigin: "top left"
                }}
              >
                <canvas ref={s.imageCanvasRef} style={{ filter: (s.imageBrightness !== 100 || s.imageContrast !== 100) ? `brightness(${s.imageBrightness}%) contrast(${s.imageContrast}%)` : undefined }} />
                <canvas ref={s.overlayCanvasRef} style={{ opacity: s.overlayVisible ? 1 : 0, pointerEvents: "none" }} />
                <canvas ref={s.gtOutlineCanvasRef} style={{ opacity: s.showGtOutline && !s.isInstanceRun ? 1 : 0, pointerEvents: "none" }} />
                {s.isInstanceRun && s.showGtOutline && s.instanceOverlayUrl && (
                  <img
                    src={s.instanceOverlayUrl}
                    className="heatmap-overlay-img"
                    style={{ width: s.width || 0, height: s.height || 0 }}
                    alt="instances"
                    data-testid="instance-overlay-img"
                  />
                )}
                {s.heatmapUrl && (
                  <img
                    src={s.heatmapUrl}
                    className="heatmap-overlay-img"
                    style={{ width: s.width || 0, height: s.height || 0 }}
                    alt="heatmap"
                    onError={() => { s.setHeatmapMode("none"); engine.setStatus("Heatmap not available — run prediction first"); }}
                  />
                )}
                {/* Calibration line SVG overlay */}
                {s.calibrating && (
                  <svg
                    className="calibration-overlay"
                    onPointerDown={s.handleCalibrationClick}
                    onPointerMove={(e) => {
                      if (!s.calibrating?.p1 || !s.calibLineRef.current || !s.previewRef.current) return;
                      const rect = s.previewRef.current.getBoundingClientRect();
                      const x = (e.clientX - rect.left - s.previewOffset.x) / s.previewScale;
                      const y = (e.clientY - rect.top - s.previewOffset.y) / s.previewScale;
                      s.calibLineRef.current.setAttribute("x2", String(x));
                      s.calibLineRef.current.setAttribute("y2", String(y));
                    }}
                    style={{ position: "absolute", top: 0, left: 0, width: s.width || 1, height: s.height || 1, zIndex: 20, cursor: "crosshair" }}
                    viewBox={`0 0 ${s.width || 1} ${s.height || 1}`}
                  >
                    {s.calibrating.p1 && (
                      <line
                        ref={s.calibLineRef}
                        x1={s.calibrating.p1[0]} y1={s.calibrating.p1[1]}
                        x2={s.calibrating.p1[0]} y2={s.calibrating.p1[1]}
                        stroke="#0af" strokeWidth={2 / s.previewScale} strokeDasharray={`${4 / s.previewScale}`}
                      />
                    )}
                  </svg>
                )}
              </div>
              {/* Region labels on overlay — semantic runs only: the instance
                  overlay image already carries numbered badges, so drawing
                  the per-region area pills on top would collide with them. */}
              {!s.isInstanceRun && s.showGtOutline && s.showRegionLabels && s.regionLabels.map((lbl, i) => {
                const rawX = s.previewOffset.x + lbl.cx * s.previewScale;
                const rawY = s.previewOffset.y + lbl.topY * s.previewScale;
                if (s.hiddenClassIds.has(lbl.classId)) return null;
                return (
                  <div
                    key={`${lbl.classId}-${i}`}
                    style={{
                      position: "absolute", left: rawX, top: rawY,
                      transform: "translate(-50%, -100%) translateY(-2px)",
                      fontSize: 11, fontWeight: 600, color: "#fff",
                      background: `rgba(${lbl.color[0]}, ${lbl.color[1]}, ${lbl.color[2]}, 0.85)`,
                      padding: "1px 5px", borderRadius: 3, pointerEvents: "none",
                      whiteSpace: "nowrap", lineHeight: "16px",
                      textShadow: "0 1px 2px rgba(0,0,0,0.6)",
                      zIndex: 10,
                    }}
                  >
                    {lbl.name} ({lbl.count.toLocaleString()}px)
                  </div>
                );
              })}
              {/* Left view toolbar */}
              <ResultsToolbar
                overlayVisible={s.overlayVisible}
                setOverlayVisible={s.setOverlayVisible}
                showRegionLabels={s.showRegionLabels}
                setShowRegionLabels={s.setShowRegionLabels}
                imageBrightness={s.imageBrightness}
                setImageBrightness={s.setImageBrightness}
                imageContrast={s.imageContrast}
                setImageContrast={s.setImageContrast}
                overlayAlpha={s.overlayAlpha}
                setOverlayAlpha={s.setOverlayAlpha}
                heatmapMode={s.heatmapMode}
                setHeatmapMode={s.setHeatmapMode}
                heatmapClassId={s.heatmapClassId}
                setHeatmapClassId={s.setHeatmapClassId}
                isInstanceRun={s.isInstanceRun}
                effectiveClasses={s.effectiveClasses}
                showCount={s.showCount}
                setShowCount={s.setShowCount}
                showArea={s.showArea}
                setShowArea={s.setShowArea}
                calibration={s.calibration}
                setCalibration={s.setCalibration}
                previewScale={s.previewScale}
                width={s.width}
                height={s.height}
                hiddenClassIds={s.hiddenClassIds}
                setHiddenClassIds={s.setHiddenClassIds}
                showGtOutline={s.showGtOutline}
                setShowGtOutline={s.setShowGtOutline}
                instanceHighlight={s.instanceHighlight}
                setInstanceHighlight={s.setInstanceHighlight}
                predOverlayPattern={s.predOverlayPattern}
                setPredOverlayPattern={s.setPredOverlayPattern}
                fitPreview={canvas.fitPreview}
              />
              {/* Measure results panel */}
              {s.maskIndex.length > 0 && (
                <MeasurementPanel
                  measureMode={s.measureMode}
                  showCount={s.showCount}
                  showArea={s.showArea}
                  effectiveClasses={s.effectiveClasses}
                  regionCounts={s.regionCounts}
                  classAreas={s.classAreas}
                  calibration={s.calibration}
                />
              )}
            </div>
          </div>
        </div>
        <div className="results-right">
          <ActionBar
            onOpenExport={() => setExportOpen(true)}
            activeRunId={s.activeRunId}
            isInferring={s.isInferring}
            currentImageNotInferred={s.currentImageNotInferred}
            inferredRuns={s.inferredRuns}
            confidenceThreshold={s.confidenceThreshold}
            setConfidenceThreshold={s.setConfidenceThreshold}
            ppMinArea={s.ppMinArea}
            setPpMinArea={s.setPpMinArea}
            ppMaxArea={s.ppMaxArea}
            setPpMaxArea={s.setPpMaxArea}
            handleRunInference={() => engine.handleRunInference()}
            handleStopInference={engine.handleStopInference}
            handleApplyPostprocessAll={engine.handleApplyPostprocessAll}
            handleClearPostprocessAll={engine.handleClearPostprocessAll}
            ppApplyAll={s.ppApplyAll}
            handleRestoreCache={engine.handleRestoreCache}
            t={t}
          />
          <MetricsSection
            metrics={s.metrics}
            isInstance={s.isInstanceRun}
            activeRunId={s.activeRunId}
            predictBackend={s.predictBackend}
            scoreCacheRef={s.scoreCacheRef}
            images={s.images}
            confidenceThreshold={s.confidenceThreshold}
            effectiveClasses={s.effectiveClasses}
            pixelHist={s.pixelHist}
            cacheVersion={s.cacheVersion}
          />
          <PredictionDetail
            visiblePredictionClasses={s.visiblePredictionClasses}
            effectiveClasses={s.effectiveClasses}
            confidenceThreshold={s.confidenceThreshold}
            confidenceIndex={s.confidenceIndex}
            maskIndex={s.maskIndex}
            gtMetrics={s.gtMetrics}
            liveStats={s.liveStats}
            isPredictionLoading={s.isPredictionLoading}
            predictionLoadingLabel={s.predictionLoadingLabel}
            notInferred={s.currentImageNotInferred}
            metrics={s.metrics}
            predictionScore={s.predictionScore}
            t={t}
          />
        </div>
      </div>
      {/* Unit conversion context menu */}
      {s.unitMenuPos && <div className="unit-menu-backdrop" onClick={() => s.setUnitMenuPos(null)} />}
      {s.unitMenuPos && (
        <div
          className="unit-context-menu"
          style={{ position: "fixed", left: s.unitMenuPos.x, top: s.unitMenuPos.y, zIndex: 9999 }}
        >
          <div className="unit-menu-header">{t("results.unitConvert")}</div>
          {(["mm", "cm", "m"] as AreaUnit[]).map((u) => (
            <button key={u} onClick={() => {
              s.setCalibrating({ unit: u, p1: null });
              s.setUnitMenuPos(null);
              engine.setStatus(t("results.calibrate").replace("{unit}", u));
            }}>
              {u}
            </button>
          ))}
          {s.calibration && (
            <button onClick={() => { s.setCalibration(null); s.setUnitMenuPos(null); }}>
              {t("results.pxReset")}
            </button>
          )}
        </div>
      )}
      <ExportDialog
        open={exportOpen}
        projectId={projectId ?? ""}
        runId={s.activeRunId ?? ""}
        onClose={() => setExportOpen(false)}
        showToast={showToast}
        isInstance={s.isInstanceRun}
      />
      <ReportModal
        open={reportOpen}
        onClose={() => setReportOpen(false)}
        projectId={projectId ?? ""}
        runId={s.activeRunId ?? ""}
        runLabel={reportRun?.model_name ?? (s.activeRunId ? `Run ${s.activeRunId.slice(0, 8)}` : "")}
        onGenerated={onOpenReport}
        t={t}
        lang={lang}
      />
    </div>
  );
});
