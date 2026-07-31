// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * OpenSeadragon-based tiled image viewer for large images (>4096px).
 *
 * Architecture:
 *   - OpenSeadragon handles image tile loading + zoom/pan
 *   - A transparent overlay canvas synced with OSD viewport renders the mask
 *   - Mask tiles (256x256 uint8 chunks) are fetched per-viewport from the server
 *   - Brush strokes update local tile cache + debounced save to server
 */
import React, { useEffect, useRef, useCallback } from "react";

const TILE_SIZE = 256;

export type TiledViewerProps = {
  /** DZI URL, e.g. /api/v1/projects/{pid}/tiles/{imageId}.dzi */
  dziUrl: string | null;
  /** Base URL for mask tile API, e.g. /api/v1/projects/{pid}/tiles/{imageId}/mask */
  maskTileBaseUrl: string | null;
  /** Image dimensions (full resolution) */
  width: number;
  height: number;
  /** Class color LUT (256 entries × 4 bytes RGBA) */
  lut: Uint8ClampedArray;
  /** Active class ID for brush drawing */
  activeClassId: number;
  /** Current tool */
  tool: string;
  /** Brush size in pixels */
  brushSize: number;
};

export const TiledViewer = React.memo(function TiledViewer({
  dziUrl,
  maskTileBaseUrl,
  width,
  height,
  lut,
  activeClassId,
  tool,
  brushSize,
}: TiledViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const osdRef = useRef<any>(null);

  // Mask tile cache: Map<"tx,ty" → Uint8Array(256*256)>
  const maskCacheRef = useRef<Map<string, Uint8Array>>(new Map());
  const dirtyTilesRef = useRef<Set<string>>(new Set());
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track drawing state
  const drawingRef = useRef(false);

  // ---- Mask tile fetch ----
  const fetchMaskTile = useCallback(async (tx: number, ty: number): Promise<Uint8Array> => {
    const key = `${tx},${ty}`;
    const cached = maskCacheRef.current.get(key);
    if (cached) return cached;

    if (!maskTileBaseUrl) return new Uint8Array(TILE_SIZE * TILE_SIZE);

    try {
      const res = await fetch(`${maskTileBaseUrl}/${tx}/${ty}`);
      if (!res.ok) return new Uint8Array(TILE_SIZE * TILE_SIZE);
      const buf = await res.arrayBuffer();
      const tile = new Uint8Array(buf);
      maskCacheRef.current.set(key, tile);
      return tile;
    } catch {
      return new Uint8Array(TILE_SIZE * TILE_SIZE);
    }
  }, [maskTileBaseUrl]);

  // ---- Save dirty tiles ----
  const saveDirtyTiles = useCallback(async () => {
    if (!maskTileBaseUrl || dirtyTilesRef.current.size === 0) return;
    const toSave = new Set(dirtyTilesRef.current);
    dirtyTilesRef.current.clear();

    for (const key of toSave) {
      const tile = maskCacheRef.current.get(key);
      if (!tile) continue;
      const [txStr, tyStr] = key.split(",");
      try {
        await fetch(`${maskTileBaseUrl}/${txStr}/${tyStr}`, {
          method: "PUT",
          headers: { "Content-Type": "application/octet-stream" },
          body: tile,
        });
      } catch (e) {
        console.warn("[TiledViewer] save tile failed:", key, e);
        dirtyTilesRef.current.add(key); // retry later
      }
    }
  }, [maskTileBaseUrl]);

  const scheduleSave = useCallback(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => saveDirtyTiles(), 1000);
  }, [saveDirtyTiles]);

  // ---- Render visible mask tiles to overlay canvas ----
  const renderOverlay = useCallback(() => {
    const viewer = viewerRef.current;
    const canvas = overlayRef.current;
    if (!viewer || !canvas || !viewer.viewport) return;

    const vp = viewer.viewport;
    const containerSize = viewer.viewport.getContainerSize();
    canvas.width = containerSize.x;
    canvas.height = containerSize.y;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Get visible image bounds in image coordinates
    const topLeft = vp.viewportToImageCoordinates(vp.getBounds(true).getTopLeft());
    const bottomRight = vp.viewportToImageCoordinates(vp.getBounds(true).getBottomRight());

    const imgX0 = Math.max(0, Math.floor(topLeft.x));
    const imgY0 = Math.max(0, Math.floor(topLeft.y));
    const imgX1 = Math.min(width, Math.ceil(bottomRight.x));
    const imgY1 = Math.min(height, Math.ceil(bottomRight.y));

    // Find visible tiles
    const tx0 = Math.floor(imgX0 / TILE_SIZE);
    const ty0 = Math.floor(imgY0 / TILE_SIZE);
    const tx1 = Math.ceil(imgX1 / TILE_SIZE);
    const ty1 = Math.ceil(imgY1 / TILE_SIZE);

    // Build 32-bit LUT
    const lut32 = new Uint32Array(256);
    for (let c = 0; c < 256; c++) {
      const b = c * 4;
      lut32[c] = lut[b] | (lut[b + 1] << 8) | (lut[b + 2] << 16) | (lut[b + 3] << 24);
    }

    // Render each visible tile
    for (let ty = ty0; ty < ty1; ty++) {
      for (let tx = tx0; tx < tx1; tx++) {
        const key = `${tx},${ty}`;
        const tile = maskCacheRef.current.get(key);
        if (!tile) {
          // Fetch async, will render on next viewport change
          fetchMaskTile(tx, ty).then(() => renderOverlay());
          continue;
        }

        // Convert tile image coords to screen coords
        const OSD = osdRef.current;
        if (!OSD) continue;
        const tileImgX = tx * TILE_SIZE;
        const tileImgY = ty * TILE_SIZE;
        const vpTopLeft = vp.imageToViewportCoordinates(tileImgX, tileImgY);
        const vpBottomRight = vp.imageToViewportCoordinates(
          tileImgX + TILE_SIZE, tileImgY + TILE_SIZE
        );
        const screenTL = vp.pixelFromPoint(vpTopLeft, true);
        const screenBR = vp.pixelFromPoint(vpBottomRight, true);

        const sw = Math.ceil(screenBR.x - screenTL.x);
        const sh = Math.ceil(screenBR.y - screenTL.y);
        if (sw <= 0 || sh <= 0) continue;

        // Render tile to a small ImageData, then draw scaled
        const imgData = new ImageData(TILE_SIZE, TILE_SIZE);
        const buf32 = new Uint32Array(imgData.data.buffer);
        for (let i = 0; i < TILE_SIZE * TILE_SIZE; i++) {
          buf32[i] = lut32[tile[i]!];
        }

        // Use offscreen canvas for scaling
        const offscreen = new OffscreenCanvas(TILE_SIZE, TILE_SIZE);
        const offCtx = offscreen.getContext("2d");
        if (!offCtx) continue;
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, screenTL.x, screenTL.y, sw, sh);
      }
    }
  }, [width, height, lut, fetchMaskTile]);

  // ---- Brush drawing ----
  const paintAt = useCallback((imgX: number, imgY: number) => {
    const r = Math.floor(brushSize / 2);
    const classId = tool === "eraser" ? 0 : activeClassId;

    // Find affected tiles
    const x0 = Math.max(0, imgX - r);
    const y0 = Math.max(0, imgY - r);
    const x1 = Math.min(width, imgX + r);
    const y1 = Math.min(height, imgY + r);

    const tx0 = Math.floor(x0 / TILE_SIZE);
    const ty0 = Math.floor(y0 / TILE_SIZE);
    const tx1 = Math.floor(x1 / TILE_SIZE);
    const ty1 = Math.floor(y1 / TILE_SIZE);

    const r2 = r * r;

    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) {
        const key = `${tx},${ty}`;
        let tile = maskCacheRef.current.get(key);
        if (!tile) {
          tile = new Uint8Array(TILE_SIZE * TILE_SIZE);
          maskCacheRef.current.set(key, tile);
        }

        const tileX0 = tx * TILE_SIZE;
        const tileY0 = ty * TILE_SIZE;

        // Paint circle within this tile
        const localX0 = Math.max(0, x0 - tileX0);
        const localY0 = Math.max(0, y0 - tileY0);
        const localX1 = Math.min(TILE_SIZE, x1 - tileX0);
        const localY1 = Math.min(TILE_SIZE, y1 - tileY0);

        for (let ly = localY0; ly < localY1; ly++) {
          for (let lx = localX0; lx < localX1; lx++) {
            const gx = tileX0 + lx;
            const gy = tileY0 + ly;
            const dx = gx - imgX;
            const dy = gy - imgY;
            if (dx * dx + dy * dy <= r2) {
              tile[ly * TILE_SIZE + lx] = classId;
            }
          }
        }
        dirtyTilesRef.current.add(key);
      }
    }

    renderOverlay();
    scheduleSave();
  }, [width, height, brushSize, activeClassId, tool, renderOverlay, scheduleSave]);

  // ---- OpenSeadragon setup ----
  useEffect(() => {
    if (!containerRef.current || !dziUrl) return;

    let destroyed = false;

    import("openseadragon").then((OSD) => {
      if (destroyed || !containerRef.current) return;
      const OpenSeadragon = OSD.default;
      osdRef.current = OpenSeadragon;

      const viewer = OpenSeadragon({
        element: containerRef.current,
        tileSources: dziUrl,
        prefixUrl: "",
        showNavigationControl: false,
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        navigatorSizeRatio: 0.15,
        minZoomLevel: 0.05,
        maxZoomLevel: 80,
        visibilityRatio: 0.2,
        constrainDuringPan: false,
        animationTime: 0.2,
        blendTime: 0.1,
        immediateRender: true,
        gestureSettingsMouse: {
          clickToZoom: false,
          dblClickToZoom: true,
          scrollToZoom: true,
        },
      });

      viewer.addHandler("open", () => {
        renderOverlay();
      });
      viewer.addHandler("open-failed", (e: any) => {
        console.error("[DBG][OSD] open-failed:", e.message || e);
      });
      viewer.addHandler("tile-load-failed", (e: any) => {
        console.warn("[DBG][OSD] tile-load-failed:", e.tile?.url);
      });

      // Re-render overlay on viewport change
      viewer.addHandler("animation", () => renderOverlay());
      viewer.addHandler("animation-finish", () => renderOverlay());

      // Pointer events for drawing
      const canvasEl = viewer.canvas as HTMLElement;

      const toImageCoords = (event: PointerEvent): [number, number] | null => {
        if (!viewer.viewport) return null;
        const rect = canvasEl.getBoundingClientRect();
        const webPoint = new OpenSeadragon.Point(
          event.clientX - rect.left,
          event.clientY - rect.top
        );
        const viewportPoint = viewer.viewport.pointFromPixel(webPoint);
        const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
        return [Math.round(imagePoint.x), Math.round(imagePoint.y)];
      };

      canvasEl.addEventListener("pointerdown", (e: PointerEvent) => {
        if (tool !== "brush" && tool !== "eraser") return;
        if (e.button !== 0) return;
        drawingRef.current = true;
        const coords = toImageCoords(e);
        if (coords) paintAt(coords[0], coords[1]);
        // Prevent OSD panning while drawing
        viewer.setMouseNavEnabled(false);
      });

      canvasEl.addEventListener("pointermove", (e: PointerEvent) => {
        if (!drawingRef.current) return;
        const coords = toImageCoords(e);
        if (coords) paintAt(coords[0], coords[1]);
      });

      const handlePointerUp = () => {
        if (drawingRef.current) {
          drawingRef.current = false;
          viewer.setMouseNavEnabled(true);
          saveDirtyTiles(); // immediate save on stroke end
        }
      };
      canvasEl.addEventListener("pointerup", handlePointerUp);
      canvasEl.addEventListener("pointerleave", handlePointerUp);

      viewerRef.current = viewer;
    });

    return () => {
      destroyed = true;
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
      maskCacheRef.current.clear();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dziUrl]);

  // Re-render overlay when LUT changes (class color changes)
  useEffect(() => {
    renderOverlay();
  }, [lut, renderOverlay]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div
        ref={containerRef}
        className="tiled-viewer"
        style={{
          width: "100%",
          height: "100%",
          background: "#1a1a1a",
        }}
      />
      <canvas
        ref={overlayRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />
    </div>
  );
});

export default TiledViewer;
