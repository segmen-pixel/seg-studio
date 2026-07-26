// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { useI18n } from "../../i18n";
import type { TrainFormReturn } from "../hooks/useTrainForm";
import { PATCH_SIZE_PRESETS } from "../hooks/useTrainForm";
import NumberField from "./NumberField";

type DatasetSectionProps = {
  form: TrainFormReturn;
  hasCuda: boolean;
};

/** Dataset settings (right column, upper): patch size, fg ratio, split ratio, etc. */
export default React.memo(function DatasetSection({ form, hasCuda }: DatasetSectionProps) {
  const { t } = useI18n();

  // DINOv2 feature split needs a CUDA GPU to compute embeddings; without one
  // it would silently degrade to hash at prepare time, so fall back here and
  // disable the option in the dropdown.
  React.useEffect(() => {
    if (!hasCuda && form.trainSplitMethod === "embedding_stratified") {
      form.setTrainSplitMethod("hash");
    }
  }, [hasCuda, form.trainSplitMethod, form.setTrainSplitMethod]);

  const iterOff = !form.trainIterativeMode;
  const iterOffCls = iterOff ? " iterative-off" : "";

  return (
    <>
      <label className="hp-cat-basic" htmlFor="hp-es-patience" data-desc={t("training.esPatience.desc")}>{t("training.esPatience")}</label>
      <NumberField id="hp-es-patience" integer min={0} data-testid="input-es-patience" value={form.trainEarlyStoppingPatience} onCommit={form.setTrainEarlyStoppingPatience} />
      <label className="hp-cat-basic" htmlFor="hp-min-epochs" data-desc={t("training.minEpochs.desc")}>{t("training.minEpochs")}</label>
      <NumberField id="hp-min-epochs" integer min={1} data-testid="input-min-epochs" value={form.trainMinEpochs} onCommit={form.setTrainMinEpochs} />
      <label className="hp-cat-input" htmlFor="hp-stride" data-desc={t("training.outputStride.desc")}>{t("training.outputStride")}</label>
      <select id="hp-stride" value={form.trainOutputStride} onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) form.setTrainOutputStride(v); }}>
        <option value={1}>1</option>
        <option value={2}>2</option>
        <option value={4}>4</option>
      </select>
      <label className="hp-cat-input" data-desc={t("training.patch.desc")}>
        {t("training.patch")}
        {form.useAutoConfig && <span className="hp-auto-badge">{t("training.autoConfig.badge")}</span>}
      </label>
      <div className="row training-inline-controls">
        <select
          className={form.useAutoConfig ? "hp-auto-overridden" : ""}
          value={(PATCH_SIZE_PRESETS as readonly number[]).includes(form.trainPatchSize) ? String(form.trainPatchSize) : "custom"}
          onChange={(e) => {
            if (e.target.value === "custom") return;
            const v = parseInt(e.target.value, 10);
            if (isNaN(v)) return;
            if (form.useAutoConfig) form.setUseAutoConfig(false);
            form.setTrainPatchSize(v);
          }}
        >
          {PATCH_SIZE_PRESETS.map((v) => <option key={v} value={v}>{v}</option>)}
          {!(PATCH_SIZE_PRESETS as readonly number[]).includes(form.trainPatchSize) && <option value="custom">{t("common.custom")}</option>}
        </select>
        <NumberField
          integer
          min={0}
          className={form.useAutoConfig ? "hp-auto-overridden" : ""}
          value={form.trainPatchSize}
          onBeforeCommit={() => { if (form.useAutoConfig) form.setUseAutoConfig(false); }}
          onCommit={form.setTrainPatchSize}
        />
      </div>
      <label className="hp-cat-input" data-desc={t("training.fgRatio.desc")}>{t("training.fgRatio")}</label>
      <span className="inline-controls" style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <NumberField integer min={0} max={100} step={5} style={{ width: 48 }} value={form.trainFgRatioMin} transform={(v) => Math.min(v, form.trainFgRatioMax)} onCommit={form.setTrainFgRatioMin} />
        <span className="muted">{"〜"}</span>
        <NumberField integer min={0} max={100} step={5} style={{ width: 48 }} value={form.trainFgRatioMax} transform={(v) => Math.max(v, form.trainFgRatioMin)} onCommit={form.setTrainFgRatioMax} />
        <span className="muted">%</span>
      </span>
      <label className="hp-cat-input" data-desc={t("training.splitRatio.desc")}>{t("training.splitRatio")}</label>
      <span className="inline-controls" style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span className="muted" style={{ fontSize: 10 }}>val</span>
        <NumberField integer min={0} max={50} step={5} style={{ width: 42 }} value={form.trainValPct} onCommit={form.setTrainValPct} disabled={form.trainKFolds > 1} />
        <span className="muted" style={{ fontSize: 10 }}>test</span>
        <NumberField integer min={0} max={50} step={5} style={{ width: 42 }} value={form.trainTestPct} onCommit={form.setTrainTestPct} />
        <span className="muted">%</span>
      </span>
      <label className="hp-cat-input" data-desc={t("training.kFolds.desc")}>{t("training.kFolds")}</label>
      <span className="inline-controls" style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <NumberField
          integer
          min={1}
          max={20}
          step={1}
          style={{ width: 52 }}
          value={form.trainKFolds}
          onCommit={form.setTrainKFolds}
        />
        <span className="muted" style={{ fontSize: 10 }}>{form.trainKFolds > 1 ? t("training.kFolds.on") : t("training.kFolds.off")}</span>
      </span>
      <label className="hp-cat-input" data-desc={t("training.splitMethod.desc")}>{t("training.splitMethod")}</label>
      <select
        value={form.trainSplitMethod}
        onChange={(e) => form.setTrainSplitMethod(e.target.value as "hash" | "embedding_stratified")}
        disabled={form.trainKFolds > 1}
      >
        <option value="hash">{t("training.splitMethod.hash")}</option>
        <option value="embedding_stratified" disabled={!hasCuda}>
          {t("training.splitMethod.embeddingStratified")}{hasCuda ? "" : t("training.splitMethod.cudaOnly")}
        </option>
      </select>
      <label className="hp-cat-input" data-desc={t("training.iterative.desc")}>{t("training.iterative")}</label>
      <select value={form.trainIterativeMode ? "on" : "off"} onChange={(e) => form.setTrainIterativeMode(e.target.value === "on")}>
        <option value="off">{t("common.off")}</option>
        <option value="on">{t("common.on")}</option>
      </select>
      <label className={"hp-cat-input" + iterOffCls} data-desc={t("training.iterativeTarget.desc")}>{t("training.iterativeTarget")}</label>
      <span className={"inline-controls" + iterOffCls} style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span className="muted" style={{ fontSize: 10 }}>Prec</span>
        <NumberField
          min={0.5} max={1.0} step={0.05} style={{ width: 52 }}
          value={form.trainTargetPrecision}
          onCommit={form.setTrainTargetPrecision}
          disabled={iterOff}
        />
        <span className="muted" style={{ fontSize: 10 }}>Rec</span>
        <NumberField
          min={0.5} max={1.0} step={0.05} style={{ width: 52 }}
          value={form.trainTargetRecall}
          onCommit={form.setTrainTargetRecall}
          disabled={iterOff}
        />
        <span className="muted" style={{ fontSize: 10 }}>Conf</span>
        <NumberField
          min={0} max={1.0} step={0.05} style={{ width: 52 }}
          value={form.trainTargetConfidence}
          onCommit={form.setTrainTargetConfidence}
          disabled={iterOff}
          title={t("training.iterativeConf.title")}
        />
      </span>
      <label className={"hp-cat-input" + iterOffCls} data-desc={t("training.iterativeParams.desc")}>{t("training.iterativeParams")}</label>
      <span className={"inline-controls" + iterOffCls} style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span className="muted" style={{ fontSize: 10 }}>max</span>
        <NumberField
          integer min={1} max={10} step={1} style={{ width: 46 }}
          value={form.trainIterMax}
          onCommit={form.setTrainIterMax}
          disabled={iterOff}
        />
        <span className="muted" style={{ fontSize: 10 }}>boost</span>
        <NumberField
          min={1.0} max={10.0} step={0.5} style={{ width: 52 }}
          value={form.trainHardWeightBoost}
          onCommit={form.setTrainHardWeightBoost}
          disabled={iterOff}
        />
      </span>
    </>
  );
});
