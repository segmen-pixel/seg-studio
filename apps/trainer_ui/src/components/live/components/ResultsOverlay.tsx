// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useRef } from "react";
import { useI18n } from "../../../i18n";
import type { InferenceResult, InferenceStats, Region } from "../types";

type ResultsOverlayProps = {
  results: InferenceResult[];
  stats: InferenceStats;
  session: { session_id: string } | null;
  cameraStateIsIdle: boolean;
  effectiveProjectId: string;
  runsEmpty: boolean;
  showDetail: boolean;
  showRegionCount: boolean;
  showArea: boolean;
  pxToArea: (px: number) => string;
  clearResults: () => void;
  lightboxIdx: number | null;
  setLightboxIdx: (idx: number | null) => void;
};

export function ResultsOverlay({
  results,
  stats,
  session,
  cameraStateIsIdle,
  effectiveProjectId,
  runsEmpty,
  showDetail,
  showRegionCount,
  showArea,
  pxToArea,
  clearResults,
  lightboxIdx,
  setLightboxIdx,
}: ResultsOverlayProps) {
  const { t } = useI18n();
  const resultsEndRef = useRef<HTMLDivElement>(null);

  // Lightbox keyboard navigation
  useEffect(() => {
    if (lightboxIdx === null) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === "ArrowRight") {
        e.preventDefault();
        setLightboxIdx(lightboxIdx < results.length - 1 ? lightboxIdx + 1 : lightboxIdx);
      } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        setLightboxIdx(lightboxIdx > 0 ? lightboxIdx - 1 : lightboxIdx);
      } else if (e.key === "Escape") {
        setLightboxIdx(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [lightboxIdx, results.length, setLightboxIdx]);

  return (
    <>
      {/* Stats bar */}
      {stats.total > 0 && (
        <div className="inspect-stats">
          <div className="inspect-stat">
            <span className="inspect-stat-label">Total</span>
            <span className="inspect-stat-value">{stats.total}</span>
          </div>
          <div className="inspect-stat">
            <span className="inspect-stat-label">OK</span>
            <span className="inspect-stat-value inspect-ok">{stats.ok}</span>
          </div>
          <div className="inspect-stat">
            <span className="inspect-stat-label">NG</span>
            <span className="inspect-stat-value inspect-ng">{stats.ng}</span>
          </div>
          <div className="inspect-stat">
            <span className="inspect-stat-label">Avg</span>
            <span className="inspect-stat-value">{stats.avgMs.toFixed(1)}ms</span>
          </div>
          <div className="inspect-stat">
            <span className="inspect-stat-label">{t("live.ngRate")}</span>
            <span className="inspect-stat-value">
              {stats.total > 0 ? ((stats.ng / stats.total) * 100).toFixed(1) : "0.0"}%
            </span>
          </div>
          <button
            className="ghost inspect-clear-btn"
            onClick={clearResults}
            title={t("live.clearResults")}
          >{t("live.clearShort")}</button>
        </div>
      )}

      {/* Results feed */}
      <div className="inspect-results">
        {results.length === 0 && session && cameraStateIsIdle && (
          <div className="inspect-empty">{t("live.startInferenceHint")}</div>
        )}
        {results.length === 0 && !cameraStateIsIdle && !session && (
          <div className="inspect-empty">{t("live.cameraPreviewHint")}</div>
        )}
        {!session && cameraStateIsIdle && !effectiveProjectId && (
          <div className="inspect-empty">{t("live.selectProject")}</div>
        )}
        {!session && cameraStateIsIdle && effectiveProjectId && runsEmpty && (
          <div className="inspect-empty">{t("live.noCompletedModel")}</div>
        )}
        {results.map((r, i) => (
          <ResultCard
            key={`${r.frame_id}-${i}`}
            result={r}
            index={i}
            showDetail={showDetail}
            showRegionCount={showRegionCount}
            showArea={showArea}
            pxToArea={pxToArea}
            onClickThumb={() => setLightboxIdx(i)}
          />
        ))}
        <div ref={resultsEndRef} />
      </div>

      {/* Lightbox with bbox overlay */}
      {lightboxIdx !== null && results[lightboxIdx]?.imageUrl && (
        <Lightbox
          result={results[lightboxIdx]}
          index={lightboxIdx}
          total={results.length}
          onClose={() => setLightboxIdx(null)}
        />
      )}
    </>
  );
}

// --- Sub-components ---

function ResultCard({
  result: r,
  index: i,
  showDetail,
  showRegionCount,
  showArea,
  pxToArea,
  onClickThumb,
}: {
  result: InferenceResult;
  index: number;
  showDetail: boolean;
  showRegionCount: boolean;
  showArea: boolean;
  pxToArea: (px: number) => string;
  onClickThumb: () => void;
}) {
  const { t } = useI18n();

  return (
    <div className={`inspect-result-card ${r.judgement === "NG" ? "inspect-result-ng" : "inspect-result-ok"}`}>
      {r.imageUrl && (
        <div className="inspect-result-thumb-wrap" onClick={(e) => { e.stopPropagation(); onClickThumb(); }}>
          <img src={r.imageUrl} className="inspect-result-thumb" alt={r.frame_id} />
          {r.maskUrl && <img src={r.maskUrl} className="inspect-result-thumb inspect-mask-overlay" alt="" />}
        </div>
      )}
      <div className="inspect-result-body">
        <div className="inspect-result-header">
          <span className={`inspect-badge ${r.judgement === "NG" ? "inspect-badge-ng" : "inspect-badge-ok"}`}>
            {r.judgement}
          </span>
          <span className="inspect-result-frame">{r.frame_id}</span>
          <span className="inspect-result-time">{r.latency_ms.total?.toFixed(1)}ms</span>
        </div>
        <div className="inspect-regions">
          {r.regions.length > 0 ? r.regions.map((reg, ri) => (
            <span key={ri} className="inspect-region-tag inspect-region-ng">
              {reg.class} {(reg.confidence * 100).toFixed(0)}%
              {showArea && ` ${pxToArea(reg.area_px)}`}
            </span>
          )) : (
            <span className="inspect-region-tag inspect-region-ok">{t("live.noDetection")}</span>
          )}
        </div>
        {showDetail && (
          <div className="inspect-result-detail-full">
            <table className="inspect-detail-table">
              <thead>
                <tr>
                  <th>{t("live.class")}</th>
                  <th>{t("live.detailAreaPx")}</th>
                  <th>{t("live.detailConfidence")}</th>
                  <th>BBox</th>
                </tr>
              </thead>
              <tbody>
                {r.regions.length > 0 ? r.regions.map((reg, ri) => (
                  <tr key={ri}>
                    <td>{reg.class}</td>
                    <td>{reg.area_px.toLocaleString()}</td>
                    <td>{(reg.confidence * 100).toFixed(1)}%</td>
                    <td className="muted" style={{ fontSize: 11 }}>{reg.bbox.join(", ")}</td>
                  </tr>
                )) : (
                  <tr><td colSpan={4} className="muted">{t("live.noDetection")}</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        <div className="inspect-result-detail muted">
          infer: {r.latency_ms.inference?.toFixed(1)}ms
          {showRegionCount && ` | defects: ${r.summary.num_defects}`}
          {showArea && ` | fg: ${(r.summary.fg_ratio * 100).toFixed(2)}%`}
          {showDetail && ` | max conf: ${(r.summary.max_confidence * 100).toFixed(1)}%`}
        </div>
      </div>
    </div>
  );
}

function Lightbox({
  result: lr,
  index,
  total,
  onClose,
}: {
  result: InferenceResult;
  index: number;
  total: number;
  onClose: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="inspect-lightbox" onClick={onClose}>
      <div className="inspect-lightbox-inner" onClick={(e) => e.stopPropagation()}>
        <div className="inspect-lightbox-img-wrap">
          <img
            src={lr.imageUrl}
            alt={lr.frame_id}
            onLoad={(e) => {
              const img = e.currentTarget;
              const canvas = img.parentElement?.querySelector("canvas");
              if (!canvas) return;
              canvas.width = img.naturalWidth;
              canvas.height = img.naturalHeight;
              const ctx = canvas.getContext("2d");
              if (!ctx) return;
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              for (const reg of lr.regions) {
                const [bx, by, bw, bh] = reg.bbox;
                const color = reg.class_id === 0 ? "rgba(0,200,0,0.7)" : "rgba(255,60,60,0.8)";
                ctx.strokeStyle = color;
                ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
                ctx.strokeRect(bx, by, bw, bh);
                ctx.fillStyle = color;
                const fontSize = Math.max(12, Math.round(canvas.width / 50));
                ctx.font = `bold ${fontSize}px sans-serif`;
                const label = `${reg.class} ${(reg.confidence * 100).toFixed(0)}%`;
                const tw = ctx.measureText(label).width;
                ctx.fillRect(bx, by - fontSize - 4, tw + 8, fontSize + 4);
                ctx.fillStyle = "#fff";
                ctx.fillText(label, bx + 4, by - 4);
              }
            }}
          />
          {lr.maskUrl && (
            <img src={lr.maskUrl} className="inspect-lightbox-mask" alt="" />
          )}
          <canvas className="inspect-lightbox-canvas" />
        </div>
        <button className="inspect-lightbox-close" onClick={onClose} title={t("live.lightboxClose")}>✕</button>
        <div className="inspect-lightbox-info">
          <span className={`inspect-badge ${lr.judgement === "NG" ? "inspect-badge-ng" : "inspect-badge-ok"}`}>
            {lr.judgement}
          </span>
          <span>{lr.frame_id}</span>
          <span className="muted">{lr.latency_ms.total?.toFixed(1)}ms</span>
          {lr.regions.length > 0 && <span className="muted">defects: {lr.regions.length}</span>}
          <span className="muted" style={{ marginLeft: "auto" }}>
            {index + 1} / {total}  {t("live.lightboxNavHint")}
          </span>
        </div>
      </div>
    </div>
  );
}
