// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useEffect, useState } from "react";
import { getAuthStatus, signIn } from "../../api/auth";
import { useI18n } from "../../i18n";

/**
 * Sign-in gate for servers started with SEG_API_TOKEN (the LAN configuration).
 *
 * Without a token configured — the loopback default — this renders its children
 * immediately and the user never sees it. With one, the app cannot make a
 * single API call until the browser holds a session cookie, so asking up front
 * beats letting every panel fail with 401.
 */
export default function TokenGate({ children }: { children: React.ReactNode }) {
  const { t } = useI18n();
  const [state, setState] = useState<"checking" | "locked" | "open">("checking");
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getAuthStatus()
      .then((s) => {
        if (!cancelled) setState(!s.token_required || s.authenticated ? "open" : "locked");
      })
      // A server too old to have /auth/status, or one still booting, is not a
      // reason to block the UI: fall through and let the app's own error
      // handling speak.
      .catch(() => { if (!cancelled) setState("open"); });
    return () => { cancelled = true; };
  }, []);

  const submit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const s = await signIn(token.trim());
      if (s.authenticated) setState("open");
      else setError(t("auth.invalid"));
    } catch {
      setError(t("auth.failed"));
    } finally {
      setBusy(false);
    }
  }, [token, busy, t]);

  if (state === "checking") return null;
  if (state === "open") return <>{children}</>;

  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      minHeight: "100vh", padding: 24,
    }}>
      <form onSubmit={submit} style={{ width: "min(420px, 100%)" }}>
        <h1 style={{ fontSize: 20, marginBottom: 8 }}>{t("auth.title")}</h1>
        <p style={{ fontSize: 13, opacity: 0.75, marginBottom: 16, lineHeight: 1.6 }}>
          {t("auth.description")}
        </p>
        <input
          type="password"
          value={token}
          autoFocus
          autoComplete="off"
          onChange={(e) => setToken(e.target.value)}
          placeholder={t("auth.placeholder")}
          aria-label={t("auth.placeholder")}
          style={{ width: "100%", padding: "10px 12px", fontSize: 14, marginBottom: 12 }}
        />
        {error && (
          <p role="alert" style={{ color: "#D55E00", fontSize: 13, marginBottom: 12 }}>{error}</p>
        )}
        <button type="submit" disabled={busy || !token.trim()} style={{ width: "100%", padding: "10px 12px" }}>
          {busy ? t("auth.signingIn") : t("auth.signIn")}
        </button>
      </form>
    </div>
  );
}
