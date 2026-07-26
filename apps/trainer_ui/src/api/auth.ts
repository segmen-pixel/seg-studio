// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE } from "./shared";

export interface AuthStatus {
  token_required: boolean;
  authenticated: boolean;
}

/** Whether this server wants a token, and whether this browser already has a session. */
export async function getAuthStatus(): Promise<AuthStatus> {
  const res = await fetch(`${API_BASE}/auth/status`);
  if (!res.ok) throw new Error(`auth/status failed: ${res.status}`);
  return res.json();
}

/**
 * Exchange the shared secret for a session cookie.
 *
 * The cookie is HttpOnly, so nothing here ever holds the credential: the
 * browser attaches it to subsequent requests — including `<img>` and download
 * URLs, which cannot carry an `X-API-Token` header.
 */
export async function signIn(token: string): Promise<AuthStatus> {
  const res = await fetch(`${API_BASE}/auth/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  return res.json();
}

export async function signOut(): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, { method: "POST" });
}

/**
 * The server's own token, for typing into another device.
 *
 * Only answered for a request from the machine running the server — a caller
 * on the network gets 403, so this returns null there rather than throwing.
 */
export async function fetchApiToken(): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/auth/token`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.token || null;
  } catch {
    return null;
  }
}
