// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

export interface NetworkSettings {
  lan_access: boolean;
  current_bind_host: string;
  expected_bind_host: string;
  restart_required: boolean;
  lan_addresses: string[];
  api_token_configured: boolean;
  cvat_proxy_configured: boolean;
  annotation_proxy_configured: boolean;
}

export async function fetchNetworkSettings(): Promise<NetworkSettings> {
  const res = await fetch(`${API_BASE}/system/network`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function updateNetworkSettings(lan_access: boolean): Promise<NetworkSettings> {
  const res = await fetch(`${API_BASE}/system/network`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lan_access }),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}
