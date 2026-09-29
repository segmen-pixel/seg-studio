// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState, useEffect } from "react";
import { useI18n } from "../i18n";
import { fetchNetworkSettings, updateNetworkSettings } from "../api";
import { fetchApiToken } from "../api/auth";
import type { NetworkSettings } from "../api";

export interface SettingsValues {
  valRatio: number;
  testRatio: number;
  exportFormat: "coreml" | "onnx";
  showInspectTab: boolean;
  previewStyle: number;
}

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
  values: SettingsValues;
  onChange: <K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => void;
  showToast?: (msg: string) => void;
}

function NetworkSettingsSection() {
  const { t } = useI18n();
  const [state, setState] = useState<NetworkSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  // Only ever non-null on the machine running the server; the endpoint refuses
  // to answer across the network.
  const [token, setToken] = useState<string | null>(null);
  const [tokenShown, setTokenShown] = useState(false);

  useEffect(() => {
    fetchNetworkSettings().then(setState).catch(() => {});
    fetchApiToken().then(setToken).catch(() => {});
  }, []);

  const handleToggle = async (lan_access: boolean) => {
    setSaving(true);
    setSavedMsg(null);
    try {
      const next = await updateNetworkSettings(lan_access);
      setState(next);
      setSavedMsg(t("settings.network.restartRequired"));
      setTimeout(() => setSavedMsg(null), 6000);
    } catch (err) {
      setSavedMsg(`Failed: ${(err as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  if (!state) return null;
  const port = 8002;
  const showTokenWarn = state.lan_access && !state.api_token_configured;
  const showProxyWarn = state.lan_access && (state.cvat_proxy_configured || state.annotation_proxy_configured);
  const currentLabel = state.current_bind_host === "0.0.0.0"
    ? t("settings.network.currentBind.lan")
    : t("settings.network.currentBind.loopback");

  return (
    <section className="settings-section-divider">
      <h3 style={{ color: "#4fc3f7" }}>🌐 {t("settings.network")}</h3>
      <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <input
          type="checkbox"
          checked={state.lan_access}
          disabled={saving}
          onChange={(e) => handleToggle(e.target.checked)}
        />
        <span>{t("settings.network.lanAccess")}</span>
      </label>
      <p className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
        {t("settings.network.lanAccess.desc")}
      </p>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>
        {currentLabel}
        {state.restart_required && (
          <span style={{ color: "#ffb74d", marginLeft: 8 }}>
            ⟳ {t("settings.network.restartRequired")}
          </span>
        )}
      </div>
      {state.lan_access && (
        <div style={{ fontSize: 12, marginBottom: 6 }}>
          <span style={{ color: "var(--muted)" }}>{t("settings.network.lanUrls")}: </span>
          {state.lan_addresses.length > 0 ? (
            state.lan_addresses.map((ip) => (
              <code key={ip} style={{ marginRight: 8, color: "#a5d6a7" }}>
                http://{ip}:{port}/ui/
              </code>
            ))
          ) : (
            <span style={{ color: "#ef9a9a" }}>{t("settings.network.lanUrls.none")}</span>
          )}
        </div>
      )}
      {state.lan_access && token && (
        <div style={{ fontSize: 12, marginBottom: 6, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span style={{ color: "var(--muted)" }}>{t("settings.network.token")}: </span>
          <code data-testid="lan-token-value" style={{ color: "#a5d6a7", userSelect: "all" }}>
            {tokenShown ? token : "••••••••••••"}
          </code>
          <button className="ghost" type="button" style={{ fontSize: 11, padding: "2px 8px" }}
                  onClick={() => setTokenShown(!tokenShown)}>
            {tokenShown ? t("settings.network.token.hide") : t("settings.network.token.show")}
          </button>
          <button className="ghost" type="button" style={{ fontSize: 11, padding: "2px 8px" }}
                  onClick={() => { navigator.clipboard?.writeText(token); setSavedMsg(t("settings.network.token.copied")); setTimeout(() => setSavedMsg(null), 3000); }}>
            {t("settings.network.token.copy")}
          </button>
          <span className="muted" style={{ flexBasis: "100%", fontSize: 11 }}>
            {t("settings.network.token.desc")}
          </span>
        </div>
      )}
      {savedMsg && (
        <div style={{ fontSize: 12, color: savedMsg.startsWith("Failed") ? "#ef9a9a" : "#a5d6a7", marginBottom: 6 }}>
          {savedMsg}
        </div>
      )}
      {(showTokenWarn || showProxyWarn) && (
        <div style={{ background: "rgba(255,152,0,0.08)", border: "1px solid rgba(255,152,0,0.3)", borderRadius: 6, padding: "8px 10px", fontSize: 12, lineHeight: 1.5 }}>
          <div style={{ fontWeight: 600, color: "#ff9800", marginBottom: 4 }}>
            ⚠ {t("settings.network.warnings")}
          </div>
          {showTokenWarn && <div>• {t("settings.network.warnTokenMissing")}</div>}
          {showProxyWarn && <div>• {t("settings.network.warnProxyEnabled")}</div>}
          <div>• {t("settings.network.firewallHint")}</div>
        </div>
      )}
    </section>
  );
}

const SettingsDialog: React.FC<SettingsDialogProps> = ({ open, onClose, values, onChange }) => {
  const { t } = useI18n();
  if (!open) return null;

  const {
    valRatio, testRatio,
    showInspectTab, previewStyle,
  } = values;

  return (
    <div className="settings-overlay" onClick={onClose} onTouchEnd={(e) => { if (e.target === e.currentTarget) onClose(); }} onTouchMove={(e) => e.stopPropagation()}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()} onTouchEnd={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t("settings.title")}</h2>
          <button className="ghost" onClick={onClose} data-desc={t("common.close")} data-desc-pos="bottom">×</button>
        </div>
        {/* 3-column grid for basic settings */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px 0", marginBottom: 12 }}>
          <section style={{ paddingRight: 10, borderRight: "1px solid var(--border)" }}>
            <h3>{t("settings.imageSplit")}</h3>
            <p className="muted" style={{ fontSize: 11, marginBottom: 6 }}>
              {t("settings.imageSplit.desc")}
            </p>
            <label style={{ fontSize: 12 }}>{t("settings.validationRatio")}: {(valRatio * 100).toFixed(0)}%</label>
            <input
              type="range" min={0.05} max={0.40} step={0.05}
              value={valRatio}
              onChange={(e) => onChange("valRatio", Number(e.target.value))}
            />
            <label style={{ fontSize: 12 }}>{t("settings.testRatio")}: {(testRatio * 100).toFixed(0)}%</label>
            <input
              type="range" min={0.05} max={0.40} step={0.05}
              value={testRatio}
              onChange={(e) => onChange("testRatio", Number(e.target.value))}
            />
            <p style={{ marginTop: 4, fontSize: 12 }}>
              Train: {((1 - valRatio - testRatio) * 100).toFixed(0)}%
            </p>
          </section>
          <section style={{ padding: "0 10px", borderRight: "1px solid var(--border)" }}>
            <h3>{t("settings.previewStyle")}</h3>
            <div style={{ display: "flex", gap: 4 }}>
              {([[t("settings.hatch"), 0], [t("settings.blink"), 1], [t("settings.checker"), 2]] as [string, 0 | 1 | 2][]).map(([label, v]) => (
                <button key={v} className={previewStyle === v ? "primary" : "ghost"} style={{ flex: 1, fontSize: 12 }} onClick={() => onChange("previewStyle", v)}>{label}</button>
              ))}
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 12 }}>
              <input type="checkbox" checked={showInspectTab} onChange={(e) => onChange("showInspectTab", e.target.checked)} />
              {t("settings.showInspectTab")}
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 12, fontSize: 12 }}>
              {t("settings.zipImportLimit")}
              <input type="number" min={1} max={64} defaultValue={parseInt(localStorage.getItem("seg_max_zip_import_gb") || "4", 10)} onChange={(e) => localStorage.setItem("seg_max_zip_import_gb", e.target.value)} style={{ width: 56, fontSize: 12, padding: "2px 6px" }} />
              GB
            </label>
          </section>
        </div>
        {/* Network access */}
        <NetworkSettingsSection />
      </div>
    </div>
  );
};

export default React.memo(SettingsDialog);
