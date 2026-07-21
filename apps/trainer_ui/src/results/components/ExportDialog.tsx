// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import { exportOnnx, exportCoreML, exportCoreMLUpdatable } from "../../api";
import { useI18n, type TranslationKey } from "../../i18n";

type Fmt = "onnx" | "coreml" | "coreml_updatable";

type Props = {
  open: boolean;
  projectId: string;
  runId: string;
  onClose: () => void;
  showToast?: (msg: string) => void;
};

const FORMATS: { id: Fmt; label: string; descKey: TranslationKey; fn: (projectId: string, runId: string) => Promise<unknown> }[] = [
  { id: "onnx", label: "ONNX", descKey: "results.export.desc.onnx", fn: exportOnnx },
  { id: "coreml", label: "CoreML", descKey: "results.export.desc.coreml", fn: exportCoreML },
  { id: "coreml_updatable", label: "CoreML (Updatable)", descKey: "results.export.desc.coremlUpdatable", fn: exportCoreMLUpdatable },
];

export default function ExportDialog({ open, projectId, runId, onClose, showToast }: Props) {
  const { t } = useI18n();
  const [fmt, setFmt] = useState<Fmt>("onnx");
  const [busy, setBusy] = useState(false);
  if (!open) return null;
  const sel = FORMATS.find((f) => f.id === fmt)!;
  const run = async () => {
    if (!projectId || !runId) return;
    setBusy(true);
    try {
      showToast?.(t("results.export.exportingToast").replace("{label}", sel.label));
      await sel.fn(projectId, runId);
      showToast?.(t("results.export.doneToast").replace("{label}", sel.label));
      onClose();
    } catch (e) {
      showToast?.(`${sel.label} export failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div
      onClick={onClose}
      style={{ position: "fixed", inset: 0, zIndex: 1000, background: "rgba(0,0,0,0.45)", display: "flex", alignItems: "center", justifyContent: "center" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: "18px 20px", width: "min(92vw, 460px)", boxShadow: "0 10px 40px rgba(0,0,0,0.35)" }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
          <h3 style={{ margin: 0, fontSize: 16 }}>{t("results.exportModel")}</h3>
          <button className="ghost" onClick={onClose} aria-label={t("common.close")} style={{ fontSize: 18, lineHeight: 1, padding: "2px 8px" }}>×</button>
        </div>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>{t("results.export.choosePrompt")}</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {FORMATS.map((f) => {
            const on = fmt === f.id;
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => setFmt(f.id)}
                aria-pressed={on}
                style={{
                  textAlign: "left", cursor: "pointer", padding: "10px 12px", borderRadius: 10,
                  border: on ? "2px solid var(--accent)" : "1px solid var(--border)",
                  background: on ? "color-mix(in srgb, var(--accent) 12%, transparent)" : "var(--panel-2)",
                  display: "flex", flexDirection: "column", gap: 3,
                }}
              >
                <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 14, fontWeight: 700, color: "var(--ink)" }}>
                  <span style={{ width: 14, display: "inline-flex", color: "var(--accent)" }}>{on ? "✓" : ""}</span>{f.label}
                </span>
                <span style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.45 }}>{t(f.descKey)}</span>
              </button>
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 16 }}>
          <button className="ghost" onClick={onClose} disabled={busy}>{t("common.cancel")}</button>
          <button className="primary" onClick={run} disabled={busy}>{busy ? t("results.export.exporting") : t("results.export.exportFmt").replace("{label}", sel.label)}</button>
        </div>
      </div>
    </div>
  );
}
