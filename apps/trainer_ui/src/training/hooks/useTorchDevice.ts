// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useEffect, useState } from "react";
import {
  fetchGlobalTrainingStatus,
  fetchTorchDevices,
  setTorchDevice,
  type TorchDeviceState,
} from "../../api";
import type { TranslationKey } from "../../i18n";

// Extracted verbatim from Training.tsx (pre-OSS refactor): torch device
// selection state, the 5s device refresh poll, the cross-project GPU-busy
// poll, and the human-readable device summary line.
export function useTorchDevice(
  active: boolean | undefined,
  setStatus: (msg: string) => void,
  t: (key: TranslationKey) => string,
) {
  const [torchState, setTorchState] = useState<TorchDeviceState | null>(null);
  const [updatingTorchDevice, setUpdatingTorchDevice] = useState(false);
  const [gpuBusy, setGpuBusy] = useState(false);

  async function refreshTorchState() {
    try {
      const state = await fetchTorchDevices();
      setTorchState(state);
    } catch (err) {
      console.warn("Training: fetch torch devices failed:", err);
    }
  }

  async function handleTorchDeviceChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const deviceId = event.target.value;
    if (!deviceId) return;
    setUpdatingTorchDevice(true);
    try {
      const state = await setTorchDevice(deviceId);
      setTorchState(state);
      setStatus(`Device set: ${state.selected_device}`);
    } catch (err) {
      setStatus(`GPU select failed: ${(err as Error).message}`);
    } finally {
      setUpdatingTorchDevice(false);
    }
  }

  useEffect(() => {
    void refreshTorchState();
  }, []);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      void refreshTorchState();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [active]);

  // Global GPU busy polling (cross-project)
  useEffect(() => {
    if (!active) return;
    const check = async () => {
      try {
        const st = await fetchGlobalTrainingStatus(torchState?.configured_device ?? "auto");
        setGpuBusy(st.gpu_busy);
      } catch (err) { console.warn("Training: GPU busy poll failed:", err); }
    };
    void check();
    const timer = window.setInterval(check, 5000);
    return () => window.clearInterval(timer);
  }, [active, torchState?.configured_device]);

  const deviceSummary = torchState
    ? torchState.configured_device === "auto"
      ? `${t("training.deviceAuto")} (${torchState.selected_device})`
      : torchState.selected_device
    : "Loading devices...";

  return { torchState, updatingTorchDevice, handleTorchDeviceChange, gpuBusy, deviceSummary };
}
