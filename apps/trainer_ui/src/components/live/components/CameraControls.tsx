// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../../i18n";
import type { CameraConfig } from "../types";

type CameraControlsProps = {
  showCameraDialog: boolean;
  setShowCameraDialog: (v: boolean) => void;
  cameraConfig: CameraConfig;
  saveCameraConfig: (cfg: CameraConfig) => void;
};

export function CameraControls({ showCameraDialog, setShowCameraDialog, cameraConfig, saveCameraConfig }: CameraControlsProps) {
  const { t } = useI18n();
  if (!showCameraDialog) return null;

  return (
    <div className="modal-overlay" onClick={() => setShowCameraDialog(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 360 }}>
        <h3 style={{ margin: "0 0 12px" }}>{t("live.camera.title")}</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{t("live.camera.deviceId")}</span>
            <input
              type="number"
              value={String(cameraConfig.device_id)}
              onChange={(e) => saveCameraConfig({ ...cameraConfig, device_id: parseInt(e.target.value) || 0 })}
              style={{ width: 80, textAlign: "right" }}
            />
          </label>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{t("live.camera.width")}</span>
            <input
              type="number"
              value={cameraConfig.width}
              onChange={(e) => saveCameraConfig({ ...cameraConfig, width: parseInt(e.target.value) || 640 })}
              style={{ width: 80, textAlign: "right" }}
            />
          </label>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{t("live.camera.height")}</span>
            <input
              type="number"
              value={cameraConfig.height}
              onChange={(e) => saveCameraConfig({ ...cameraConfig, height: parseInt(e.target.value) || 480 })}
              style={{ width: 80, textAlign: "right" }}
            />
          </label>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>FPS</span>
            <input
              type="number"
              value={cameraConfig.fps}
              onChange={(e) => saveCameraConfig({ ...cameraConfig, fps: parseInt(e.target.value) || 30 })}
              style={{ width: 80, textAlign: "right" }}
            />
          </label>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{t("live.camera.previewMaxWidth")}</span>
            <input
              type="number"
              value={cameraConfig.preview_max_width}
              onChange={(e) => saveCameraConfig({ ...cameraConfig, preview_max_width: parseInt(e.target.value) || 640 })}
              style={{ width: 80, textAlign: "right" }}
            />
          </label>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>{t("live.camera.previewFpsCap")}</span>
            <input
              type="number"
              value={cameraConfig.preview_fps}
              onChange={(e) => saveCameraConfig({ ...cameraConfig, preview_fps: parseInt(e.target.value) || 15 })}
              style={{ width: 80, textAlign: "right" }}
            />
          </label>
        </div>
        <div style={{ marginTop: 16, display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button onClick={() => setShowCameraDialog(false)}>{t("common.close")}</button>
        </div>
      </div>
    </div>
  );
}
