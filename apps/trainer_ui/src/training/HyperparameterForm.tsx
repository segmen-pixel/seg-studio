// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../i18n";
import {
  type TorchDeviceState,
} from "../api";
import {
  type TrainRunItem,
} from "../utils";
import { useTrainForm } from "./hooks/useTrainForm";
import ModelSection from "./components/ModelSection";
import DatasetSection from "./components/DatasetSection";
import AugmentSection from "./components/AugmentSection";
import ModeSelectButtons from "./components/ModeSelectButtons";
import NumberField from "./components/NumberField";

type HyperparameterFormProps = {
  projectId: string;
  isStartingTrain: boolean;
  hasRunningRun: boolean;
  onStartTrain: (payload: Record<string, unknown>) => void;
  torchState: TorchDeviceState | null;
  onTorchDeviceChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  updatingTorchDevice: boolean;
  showToast: (msg: string) => void;
  runs: TrainRunItem[];
  prepareReport: {
    train_count: number;
    val_count: number;
    with_mask: number;
    auto_val_from_train_count?: number;
  } | null;
  onModelSearch?: () => void;
  onCancelModelSearch?: () => void;
  isSearching?: boolean;
  libraryStats?: { total_profiles: number; total_projects: number } | null;
  trainingMode?: "standard" | "quick" | "transfer" | null;
  onTrainingModeChange?: (mode: "standard" | "quick" | "transfer") => void;
};

function deviceOptionLabel(item: {
  kind: string;
  label: string;
  allocated_mb?: number | null;
  memory_mb?: number | null;
  busy?: boolean;
}): string {
  if (item.kind !== "cuda") return item.label;
  const parts = [item.label];
  if (typeof item.allocated_mb === "number" && typeof item.memory_mb === "number") {
    parts.push(`- ${item.allocated_mb}/${item.memory_mb}MB used`);
  }
  if (item.busy) parts.push("(busy)");
  return parts.join(" ");
}

export default React.memo(function HyperparameterForm({
  projectId,
  isStartingTrain,
  hasRunningRun,
  onStartTrain,
  torchState,
  onTorchDeviceChange,
  updatingTorchDevice,
  showToast,
  runs,
  prepareReport,
  onModelSearch,
  onCancelModelSearch,
  isSearching,
  libraryStats,
  trainingMode = null,
  onTrainingModeChange,
}: HyperparameterFormProps) {
  const { t } = useI18n();
  const setStatus = showToast;
  const hasExistingRuns = runs.length > 0;

  const form = useTrainForm();

  function handleSubmit() {
    const payload = form.buildPayload(runs, isStartingTrain, trainingMode);
    if (payload) onStartTrain(payload);
  }

  return (
    <div className="section training-config-section" style={{ marginBottom: 0 }}>
      {!trainingMode && (
        <div className="training-mode-hint">
          {t("training.form.modeHint")}
        </div>
      )}
      <div className="training-start-row">
        {/* Left: Train buttons + model name + settings (3 rows) */}
        <div className="training-left-controls">
          <div className="training-start-buttons">
            <button
              className="primary training-start-btn"
              onClick={handleSubmit}
              disabled={isStartingTrain || !trainingMode}
              data-desc={!trainingMode ? t("training.form.selectModeRight") : hasRunningRun ? t("training.reserveTrain.desc") : t("training.startTrain.desc")}
              data-desc-pos="bottom"
              data-tutorial-step="training-start"
            >
              {isStartingTrain ? t("training.starting") : hasRunningRun ? t("training.reserveTrain") : t("training.startTrain")}
            </button>
          </div>
          <input
            type="text"
            className="training-name-input"
            placeholder={t("training.modelName")}
            aria-label="Model name"
            value={form.trainModelName}
            onChange={(e) => form.setTrainModelName(e.target.value)}
          />
          <select
            className="training-name-input training-device-select"
            aria-label={t("training.device")}
            title={t("training.device")}
            value={torchState?.configured_device ?? "auto"}
            onChange={onTorchDeviceChange}
            disabled={updatingTorchDevice || !torchState}
          >
            <option value="auto">
              {t("training.deviceAuto")}
              {torchState?.selected_device ? ` (${torchState.selected_device})` : ""}
            </option>
            {(torchState?.devices ?? []).map((d) => (
              <option key={d.id} value={d.id}>{deviceOptionLabel(d)}</option>
            ))}
          </select>
          <button
            className="ghost training-hyper-toggle"
            onClick={() => form.setHyperParamsOpen(!form.hyperParamsOpen)}
            type="button"
            aria-expanded={form.hyperParamsOpen}
            aria-controls="hyperparameter-panel"
            data-desc={t("training.detailedSettings.desc")}
          >
            <span>{t("training.detailedSettings")}</span>
            <span className={`tl-arrow${form.hyperParamsOpen ? " open" : ""}`}>&#9654;</span>
          </button>
        </div>

        {/* Right: Mode selection (large with SVG) */}
        <ModeSelectButtons
          trainingMode={trainingMode}
          onTrainingModeChange={onTrainingModeChange}
          libraryStats={libraryStats}
        />

      </div>
      <div className="transfer-learning-fieldset">
        {form.hyperParamsOpen && (
      <div className="training-flat-params" id="hyperparameter-panel">
        {/* Master switch: single "Auto" knob (ADR-005 Phase D step 2b).
            The 3 options collapse the pre-Phase-D pair of toggles
            (Auto-select + Auto-config) into one UX. Legacy state is still
            2 booleans internally; only the wire payload (auto_mode) and
            this UI are unified for now. Independent Auto-select toggle in
            ModelSection has been retired. */}
        <div className="hp-master-switch" data-desc={t("training.autoMode.desc")} data-tutorial-step="auto-config-master">
          <span className="hp-master-switch-label">{t("training.autoMode")}</span>
          <select
            className="hp-master-switch-select"
            data-testid="select-auto-mode"
            value={form.useAutoConfig ? "on" : "off"}
            onChange={(e) => form.setUseAutoConfig(e.target.value === "on")}
          >
            <option value="on">{t("training.autoMode.on")}</option>
            <option value="off">{t("training.autoMode.off")}</option>
          </select>
          <span className="hp-master-switch-hint">{t("training.autoMode.desc")}</span>
        </div>
        <div className="hp-cat-columns">
          <div className="hp-cat-col">
            <div className="hp-cat-col-head hp-cat-h-model"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>{t("training.form.catModel")}</div>
            <div className="training-form-grid hp-grid-2col"><ModelSection form={form} /></div>
          </div>
          <div className="hp-cat-col">
            <div className="hp-cat-col-head hp-cat-h-data"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>{t("training.form.catData")}</div>
            <div className="training-form-grid hp-grid-2col"><DatasetSection form={form} hasCuda={(torchState?.devices ?? []).some((d) => d.kind === "cuda" && d.available)} /></div>
          </div>
          <div className="hp-cat-col">
            <div className="hp-cat-col-head hp-cat-h-aug"><svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M12 3c.3 3.6 2.4 5.7 6 6-3.6.3-5.7 2.4-6 6-.3-3.6-2.4-5.7-6-6 3.6-.3 5.7-2.4 6-6z"/></svg>{t("training.form.catAugment")}</div>
            <div className="training-form-grid hp-grid-2col"><AugmentSection form={form} part="aug" /></div>
          </div>
          <div className="hp-cat-col">
            <div className="hp-cat-col-head hp-cat-h-weight"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v18"/><path d="M6 6h12"/><path d="M6 6l-3 6a3 3 0 0 0 6 0z"/><path d="M18 6l-3 6a3 3 0 0 0 6 0z"/><path d="M9 21h6"/></svg>{t("training.form.catWeight")}</div>
            <div className="training-form-grid hp-grid-2col"><AugmentSection form={form} part="weight" /></div>
          </div>
        </div>
      </div>
        )}
      </div>
    </div>
  );
});
