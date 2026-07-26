// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

export type TorchDevice = {
  id: string;
  label: string;
  kind: string;
  available: boolean;
  selected: boolean;
  busy?: boolean;
  busy_owner_kind?: string;
  busy_owner_id?: string;
  index?: number;
  memory_mb?: number | null;
  allocated_mb?: number | null;
  reserved_mb?: number | null;
};

export type TorchDeviceState = {
  configured_device: string;
  selected_device: string;
  devices: TorchDevice[];
};

// ---------------------------------------------------------------------------
// Health API
// ---------------------------------------------------------------------------

export type HealthInfo = {
  status: string;
  version: string;
  build_date: string;
  disk: { total_gb: number; free_gb: number; used_pct: number } | null;
  ram: { total_gb: number; available_gb: number; used_pct: number } | null;
  gpu: { name: string; vram_total_mb: number; vram_allocated_mb: number } | null;
};

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchTorchDevices(): Promise<TorchDeviceState> {
  const res = await fetch(`${API_BASE}/hardware/torch/devices`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function setTorchDevice(device: string): Promise<TorchDeviceState> {
  const res = await fetch(`${API_BASE}/hardware/torch/device`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device })
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}
