// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import type { TrainFormReturn } from "../hooks/useTrainForm";
import NumberField from "./NumberField";

type AugmentSectionProps = {
  form: TrainFormReturn;
  part?: "aug" | "weight";
};

/** Augmentation & class-weight settings (right column, lower). */
export default React.memo(function AugmentSection({ form, part = "aug" }: AugmentSectionProps) {
  const { t } = useI18n();

  return (
    <>
      {part !== "weight" && (<>
      <label className="hp-cat-aug" data-desc={t("training.augment.desc")}>{t("training.augment")}</label>
      <select value={form.trainAugmentEnabled ? "on" : "off"} onChange={(e) => form.setTrainAugmentEnabled(e.target.value === "on")}>
        <option value="on">{t("common.on")}</option>
        <option value="off">{t("common.off")}</option>
      </select>
      <label className="hp-cat-aug" data-desc={t("training.hflip.desc")}>{t("training.hflip")}</label>
      <NumberField min={0} max={1} step={0.05} value={form.trainAugmentHFlipProb} onCommit={form.setTrainAugmentHFlipProb} disabled={!form.trainAugmentEnabled} />
      <label className="hp-cat-aug" data-desc={t("training.vflip.desc")}>{t("training.vflip")}</label>
      <NumberField min={0} max={1} step={0.05} value={form.trainAugmentVFlipProb} onCommit={form.setTrainAugmentVFlipProb} disabled={!form.trainAugmentEnabled} />
      <label className="hp-cat-aug" data-desc={t("training.rot90.desc")}>{t("training.rot90")}</label>
      <NumberField min={0} max={1} step={0.05} value={form.trainAugmentRotate90Prob} onCommit={form.setTrainAugmentRotate90Prob} disabled={!form.trainAugmentEnabled} />
      <label className="hp-cat-aug" data-desc={t("training.brightness.desc")}>{t("training.brightness")}</label>
      <NumberField min={0} max={1} step={0.05} value={form.trainAugmentBrightness} onCommit={form.setTrainAugmentBrightness} disabled={!form.trainAugmentEnabled} />
      <label className="hp-cat-aug" data-desc={t("training.contrast.desc")}>{t("training.contrast")}</label>
      <NumberField min={0} max={1} step={0.05} value={form.trainAugmentContrast} onCommit={form.setTrainAugmentContrast} disabled={!form.trainAugmentEnabled} />
      <label className="hp-cat-aug" data-desc={t("training.noiseStd.desc")}>{t("training.noiseStd")}</label>
      <NumberField min={0} max={0.5} step={0.01} value={form.trainAugmentNoiseStd} onCommit={form.setTrainAugmentNoiseStd} disabled={!form.trainAugmentEnabled} />
      </>)}
      {part === "weight" && (<>
      <label className="hp-cat-weight" htmlFor="hp-loss" data-desc={t("training.lossType.desc")}>{t("training.lossType")}</label>
      <select id="hp-loss" value={form.trainLossType} onChange={(e) => form.setTrainLossType(e.target.value as "auto" | "ce" | "focal" | "lovasz")}>
        <option value="auto">{t("training.lossType.auto")}</option>
        <option value="ce">CE</option>
        <option value="focal">Focal</option>
        <option value="lovasz">Lovász</option>
      </select>
      <label className="hp-cat-weight" htmlFor="hp-cw" data-desc={t("training.classWeights.desc")}>{t("training.classWeights")}</label>
      <select id="hp-cw" value={form.trainUseClassWeights ? "on" : "off"} onChange={(e) => form.setTrainUseClassWeights(e.target.value === "on")}>
        <option value="on">{t("common.on")}</option>
        <option value="off">{t("common.off")}</option>
      </select>
      <label className="hp-cat-weight" data-desc={t("training.cwStrength.desc")}>{t("training.cwStrength")}</label>
      <div className="row training-inline-controls">
        <select aria-label="Class weight strength mode" value={form.trainUseAutoClassWeightStrength ? "auto" : "manual"} onChange={(e) => form.setTrainUseAutoClassWeightStrength(e.target.value === "auto")} disabled={!form.trainUseClassWeights}>
          <option value="auto">{t("common.auto")}</option>
          <option value="manual">{t("common.manual")}</option>
        </select>
        <NumberField aria-label="Class weight strength value" min={0} max={1} step={0.05} value={form.trainClassWeightStrength} onCommit={form.setTrainClassWeightStrength} disabled={!form.trainUseClassWeights || form.trainUseAutoClassWeightStrength} />
      </div>
      <label className="hp-cat-weight" data-desc={t("training.bgBoost.desc")}>{t("training.bgBoost")}</label>
      <div className="row training-inline-controls">
        <select aria-label="Background boost mode" value={form.trainUseAutoBackgroundBoost ? "auto" : "manual"} onChange={(e) => form.setTrainUseAutoBackgroundBoost(e.target.value === "auto")} disabled={!form.trainUseClassWeights}>
          <option value="auto">{t("common.auto")}</option>
          <option value="manual">{t("common.manual")}</option>
        </select>
        <NumberField aria-label="Background weight boost value" min={1} max={3} step={0.05} value={form.trainBackgroundWeightBoost} onCommit={form.setTrainBackgroundWeightBoost} disabled={!form.trainUseClassWeights || form.trainUseAutoBackgroundBoost} />
      </div>
      </>)}
    </>
  );
});
