// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import type { TrainFormReturn } from "../hooks/useTrainForm";
import NumberField from "./NumberField";

type ModelSectionProps = {
  form: TrainFormReturn;
};

/** Key model settings (left column): arch, base channels, epochs, lr, etc. */
export default React.memo(function ModelSection({ form }: ModelSectionProps) {
  const { t } = useI18n();

  // Auto Config (v2) is rendered one level up by HyperparameterForm as a
  // banner above the 4 category columns — it acts as a master switch that
  // affects every column, not just model settings, so it belongs outside
  // this section. arch / base_channels are the fields the server overrides
  // when Auto Config is on (see training_runner.py: arch / base_channels /
  // patch_size). We dim them while it's on and flip the switch off the
  // moment the user picks a manual value here.
  const autoOn = form.useAutoConfig;
  const autoClass = autoOn ? "hp-auto-overridden" : "";
  const autoBadge = autoOn ? <span className="hp-auto-badge">{t("training.autoConfig.badge")}</span> : null;
  return (
    <>
        <label className="hp-cat-basic" htmlFor="hp-arch" data-desc={t("training.architecture.desc")}>{t("training.architecture")}{autoBadge}</label>
        <select
          id="hp-arch"
          className={autoClass}
          value={form.trainArch}
          onChange={(e) => {
            const v = e.target.value as "simpleunet" | "stdc" | "deeplabv3plus";
            if (autoOn) form.setUseAutoConfig(false);
            form.setTrainArch(v);
          }}
        >
          <option value="simpleunet">SimpleUNet</option>
          <option value="stdc">STDC</option>
          <option value="deeplabv3plus">DeepLabV3+</option>
        </select>
        <label className="hp-cat-basic" htmlFor="hp-base-ch" data-desc={t("training.modelSize.desc")}>{t("training.modelSize")}{autoBadge}</label>
        <select
          id="hp-base-ch"
          className={autoClass}
          value={form.trainBaseChannels}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            if (isNaN(v)) return;
            if (autoOn) form.setUseAutoConfig(false);
            form.setTrainBaseChannels(v);
          }}
        >
          <option value={32}>{t("training.compact")}</option>
          <option value={64}>{t("training.standard")}</option>
          <option value={128}>{t("training.large")}</option>
        </select>
        <label className="hp-cat-basic" htmlFor="hp-epochs" data-desc={t("training.epochs.desc")}>{t("training.epochs")}</label>
        <NumberField id="hp-epochs" integer min={1} data-testid="input-epochs" value={form.trainEpochs} onCommit={form.setTrainEpochs} />
        <label className="hp-cat-basic" data-desc={t("training.autoEpochs.desc")}>{t("training.autoEpochs")}</label>
        <select data-testid="select-auto-epochs" value={form.trainAutoEpochs ? "on" : "off"} onChange={(e) => form.setTrainAutoEpochs(e.target.value === "on")}>
          <option value="on">{t("common.on")}</option>
          <option value="off">{t("common.off")}</option>
        </select>
        <label className="hp-cat-basic" htmlFor="hp-lr" data-desc={t("training.lr.desc")}>{t("training.lr")}</label>
        <NumberField id="hp-lr" min={0.000001} step={0.0001} value={form.trainLearningRate} onCommit={form.setTrainLearningRate} />
        {/* Auto-select toggle removed in ADR-005 Phase D step 2b; the
            master switch above expresses the single Auto (recipe) knob. */}
        <label className="hp-cat-transfer" data-desc={t("training.dinov2.desc")}>{t("training.dinov2")}{autoBadge}</label>
        <select
          className={autoClass}
          value={form.useDinov2 ? "on" : "off"}
          onChange={(e) => {
            if (autoOn) form.setUseAutoConfig(false);
            form.setUseDinov2(e.target.value === "on");
          }}
        >
          <option value="on">{t("common.on")}</option>
          <option value="off">{t("common.off")}</option>
        </select>
        <label className="hp-cat-basic" data-desc={t("training.deepSupervision.desc")}>{t("training.deepSupervision")}</label>
        <select value={form.trainDeepSupervision ? "on" : "off"} onChange={(e) => form.setTrainDeepSupervision(e.target.value === "on")}>
          <option value="off">{t("common.off")}</option>
          <option value="on">{t("common.on")}</option>
        </select>
        <label className="hp-cat-basic" data-desc={t("training.frequencyMap.desc")}>{t("training.frequencyMap")}</label>
        <select value={form.trainFrequencyMap ? "on" : "off"} onChange={(e) => form.setTrainFrequencyMap(e.target.value === "on")}>
          <option value="off">{t("common.off")}</option>
          <option value="on">{t("common.on")}</option>
        </select>
    </>
  );
});
