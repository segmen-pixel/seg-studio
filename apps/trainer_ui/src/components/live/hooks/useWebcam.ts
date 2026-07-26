// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "../../../api";
import type { CameraConfig, CameraState, InferenceResult, SessionInfo } from "../types";
import { DEFAULT_CAMERA_CONFIG } from "../types";

export function useWebcam(
  addResult: (result: InferenceResult) => void,
  session: SessionInfo | null,
  selectedRun: string,
  effectiveProjectId: string,
  toast: (msg: string) => void,
) {
  const [cameraState, setCameraState] = useState<CameraState>("IDLE");
  const [cameraConnecting, setCameraConnecting] = useState(false);
  const [showCameraDialog, setShowCameraDialog] = useState(false);
  const [cameraConfig, setCameraConfig] = useState<CameraConfig>(() => {
    try {
      const saved = localStorage.getItem("seg-camera-config");
      return saved ? { ...DEFAULT_CAMERA_CONFIG, ...JSON.parse(saved) } : DEFAULT_CAMERA_CONFIG;
    } catch (err) { console.warn("LiveInspection: parse camera config from localStorage failed:", err); return DEFAULT_CAMERA_CONFIG; }
  });
  const cameraWsRef = useRef<WebSocket | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const [previewFps, setPreviewFps] = useState(0);
  const previewFpsRef = useRef({ count: 0, lastTime: 0 });

  const connectCamera = useCallback(async () => {
    setCameraConnecting(true);
    try {
      // Configure
      await fetch(`${API_BASE}/v2/camera/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cameraConfig),
      });
      // Start
      const res = await fetch(`${API_BASE}/v2/camera/start`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setCameraState("PREVIEW");

      // Open camera WS for preview + results
      const wsUrl = `${API_BASE.replace(/^http/, "ws")}/ws/v2/camera`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      cameraWsRef.current = ws;

      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          // Binary = JPEG preview frame
          const blob = new Blob([ev.data], { type: "image/jpeg" });
          const url = URL.createObjectURL(blob);
          const img = new Image();
          img.onload = () => {
            const canvas = previewCanvasRef.current;
            if (canvas) {
              canvas.width = img.width;
              canvas.height = img.height;
              const ctx = canvas.getContext("2d");
              ctx?.drawImage(img, 0, 0);
            }
            URL.revokeObjectURL(url);
            // FPS counter
            const now = performance.now();
            const fps = previewFpsRef.current;
            fps.count++;
            if (now - fps.lastTime >= 1000) {
              setPreviewFps(fps.count);
              fps.count = 0;
              fps.lastTime = now;
            }
          };
          img.onerror = () => {
            URL.revokeObjectURL(url);
            console.warn("LiveInspection: failed to decode preview frame blob");
          };
          img.src = url;
        } else {
          // Text = JSON (inference result or control)
          try {
            const data = JSON.parse(ev.data);
            if (data.judgement) {
              const r = data as InferenceResult;
              if (r.mask_png_b64) {
                const bin = atob(r.mask_png_b64);
                const buf = new Uint8Array(bin.length);
                for (let j = 0; j < bin.length; j++) buf[j] = bin.charCodeAt(j);
                r.maskUrl = URL.createObjectURL(new Blob([buf], { type: "image/png" }));
                delete r.mask_png_b64;
              }
              addResult(r);
            }
          } catch (err) { console.warn("LiveInspection: camera WS message parse error:", err); }
        }
      };

      ws.onclose = () => {
        cameraWsRef.current = null;
      };

      // If session already active, attach inference
      if (session && selectedRun) {
        await fetch(`${API_BASE}/v2/camera/attach`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project_id: effectiveProjectId, run_id: selectedRun }),
        });
        setCameraState("INSPECT");
      }
    } catch (e) {
      console.error("Camera connect failed:", e);
      toast(`Camera connect failed: ${(e as Error).message}`);
    } finally {
      setCameraConnecting(false);
    }
  }, [cameraConfig, session, selectedRun, effectiveProjectId, addResult, toast]);

  const disconnectCamera = useCallback(async () => {
    if (cameraWsRef.current) {
      cameraWsRef.current.close();
      cameraWsRef.current = null;
    }
    try {
      await fetch(`${API_BASE}/v2/camera/stop`, { method: "POST" });
    } catch (err) { console.warn("LiveInspection: camera stop failed:", err); }
    setCameraState("IDLE");
    setPreviewFps(0);
  }, []);

  const saveCameraConfig = useCallback((cfg: CameraConfig) => {
    setCameraConfig(cfg);
    localStorage.setItem("seg-camera-config", JSON.stringify(cfg));
  }, []);

  const captureSnapshot = useCallback(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || canvas.width === 0) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const ts = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 15);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `picture_${ts}.jpg`;
      a.click();
      URL.revokeObjectURL(url);
    }, "image/jpeg", 0.95);
  }, []);

  // Cleanup on unmount + page reload
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (cameraWsRef.current) cameraWsRef.current.close();
      // Fire-and-forget: stop camera + session on backend before page unloads
      // Use sendBeacon for reliability during unload (fetch may be cancelled)
      const base = API_BASE;
      navigator.sendBeacon(`${base}/v2/camera/stop`, "");
      navigator.sendBeacon(`${base}/v2/session/stop`, "");
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      if (cameraWsRef.current) cameraWsRef.current.close();
    };
  }, []);

  return {
    cameraState,
    setCameraState,
    cameraConnecting,
    showCameraDialog,
    setShowCameraDialog,
    cameraConfig,
    previewCanvasRef,
    previewFps,
    connectCamera,
    disconnectCamera,
    saveCameraConfig,
    captureSnapshot,
  };
}
