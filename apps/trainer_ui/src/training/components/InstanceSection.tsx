// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import { useI18n } from "../../i18n";
import { fetchInstancePreview, type InstancePreviewResponse } from "../../api";
import { INSTANCE_MODEL_SIZES, INSTANCE_PATCH_SIZE_PRESETS, type InstanceModelSize, type TrainFormReturn } from "../hooks/useTrainForm";
import NumberField from "./NumberField";

type InstanceSectionProps = {
  projectId: string;
  form: TrainFormReturn;
  showToast: (msg: string) => void;
};

/** Instance-mode synthesis settings (flat fields) + composed-sample preview strip. */
export default React.memo(function InstanceSection({ projectId, form, showToast }: InstanceSectionProps) {
  const { t } = useI18n();
  const [preview, setPreview] = useState<InstancePreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  async function handlePreview() {
    if (previewLoading) return;
    setPreviewLoading(true);
    try {
      const params: Record<string, unknown> = {
        instance_objects_min: form.trainInstanceObjectsMin,
        instance_objects_max: form.trainInstanceObjectsMax,
        instance_stack_pair_prob: form.trainInstanceStackPairProb,
        instance_seed: form.trainInstanceSeed,
        n_samples: 3,
      };
      if (form.trainInstanceAreaBandMin > 0 && form.trainInstanceAreaBandMax > 0) {
        params.instance_area_band_min = form.trainInstanceAreaBandMin;
        params.instance_area_band_max = form.trainInstanceAreaBandMax;
      }
      setPreview(await fetchInstancePreview(projectId, params));
    } catch (err) {
      setPreview(null);
      showToast(`${t("training.instance.previewFailed")}: ${(err as Error).message}`);
    } finally {
      setPreviewLoading(false);
    }
  }

  return (
    <div className="training-flat-params" id="instance-panel" data-testid="instance-section">
      <div className="hp-cat-columns">
        <div className="hp-cat-col">
          <div className="hp-cat-col-head hp-cat-h-model">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="8" y="14" width="7" height="7" rx="1.5"/></svg>
            {t("training.instance.catModel")}
          </div>
          <div className="training-form-grid hp-grid-2col">
            <label className="hp-cat-input" htmlFor="inst-model-size" data-desc={t("training.instance.modelSize.desc")}>{t("training.instance.modelSize")}</label>
            <select
              id="inst-model-size"
              data-testid="select-instance-model-size"
              value={form.trainInstanceModelSize}
              onChange={(e) => {
                const v = e.target.value as InstanceModelSize;
                form.setTrainInstanceModelSize(INSTANCE_MODEL_SIZES.includes(v) ? v : "small");
              }}
            >
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="large">Large</option>
            </select>
            <label className="hp-cat-input" data-desc={t("training.instance.patch.desc")}>{t("training.instance.patch")}</label>
            <div className="row training-inline-controls instance-patch-controls">
              <select
                data-testid="select-instance-patch-size"
                value={(INSTANCE_PATCH_SIZE_PRESETS as readonly number[]).includes(form.trainInstancePatchSize) ? String(form.trainInstancePatchSize) : "custom"}
                onChange={(e) => {
                  if (e.target.value === "custom") return;
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v)) form.setTrainInstancePatchSize(v);
                }}
              >
                {INSTANCE_PATCH_SIZE_PRESETS.map((v) => (
                  <option key={v} value={v}>{v === 0 ? t("training.instance.patch.off") : v}</option>
                ))}
                {!(INSTANCE_PATCH_SIZE_PRESETS as readonly number[]).includes(form.trainInstancePatchSize) && <option value="custom">{t("common.custom")}</option>}
              </select>
              <NumberField
                integer
                min={0}
                step={64}
                data-testid="input-instance-patch-size"
                value={form.trainInstancePatchSize}
                onCommit={form.setTrainInstancePatchSize}
              />
            </div>
            <label className="hp-cat-input" htmlFor="inst-epochs" data-desc={t("training.epochs.desc")}>{t("training.epochs")}</label>
            <NumberField id="inst-epochs" integer min={1} data-testid="input-instance-epochs" value={form.trainEpochs} onCommit={form.setTrainEpochs} />
          </div>
        </div>
        <div className="hp-cat-col">
          <div className="hp-cat-col-head hp-cat-h-data">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>
            {t("training.instance.catSynthesis")}
          </div>
          <div className="training-form-grid hp-grid-2col">
            <label className="hp-cat-input" htmlFor="inst-n-train" data-desc={t("training.instance.nTrain.desc")}>{t("training.instance.nTrain")}</label>
            <NumberField id="inst-n-train" integer min={8} max={5000} step={50} data-testid="input-instance-n-train" value={form.trainInstanceNTrain} onCommit={form.setTrainInstanceNTrain} />
            <label className="hp-cat-input" htmlFor="inst-n-val" data-desc={t("training.instance.nVal.desc")}>{t("training.instance.nVal")}</label>
            <NumberField id="inst-n-val" integer min={2} max={1000} step={10} data-testid="input-instance-n-val" value={form.trainInstanceNVal} onCommit={form.setTrainInstanceNVal} />
            <label className="hp-cat-input" data-desc={t("training.instance.objects.desc")}>{t("training.instance.objects")}</label>
            <span className="inline-controls" style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <NumberField integer min={1} max={64} style={{ width: 48 }} data-testid="input-instance-objects-min" value={form.trainInstanceObjectsMin} transform={(v) => Math.min(v, form.trainInstanceObjectsMax)} onCommit={form.setTrainInstanceObjectsMin} />
              <span className="muted">{"〜"}</span>
              <NumberField integer min={1} max={64} style={{ width: 48 }} data-testid="input-instance-objects-max" value={form.trainInstanceObjectsMax} transform={(v) => Math.max(v, form.trainInstanceObjectsMin)} onCommit={form.setTrainInstanceObjectsMax} />
            </span>
            <label className="hp-cat-input" htmlFor="inst-stack-prob" data-desc={t("training.instance.stackPairProb.desc")}>{t("training.instance.stackPairProb")}</label>
            <NumberField id="inst-stack-prob" min={0} max={1} step={0.05} data-testid="input-instance-stack-prob" value={form.trainInstanceStackPairProb} onCommit={form.setTrainInstanceStackPairProb} />
            <label className="hp-cat-input" htmlFor="inst-seed" data-desc={t("training.instance.seed.desc")}>{t("training.instance.seed")}</label>
            <NumberField id="inst-seed" integer min={0} data-testid="input-instance-seed" value={form.trainInstanceSeed} onCommit={form.setTrainInstanceSeed} />
            <label className="hp-cat-input" data-desc={t("training.instance.areaBand.desc")}>{t("training.instance.areaBand")}</label>
            <span className="inline-controls" style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <NumberField integer min={0} step={100} style={{ width: 62 }} data-testid="input-instance-band-min" value={form.trainInstanceAreaBandMin} onCommit={form.setTrainInstanceAreaBandMin} />
              <span className="muted">{"〜"}</span>
              <NumberField integer min={0} step={100} style={{ width: 62 }} data-testid="input-instance-band-max" value={form.trainInstanceAreaBandMax} onCommit={form.setTrainInstanceAreaBandMax} />
              <span className="muted">px²</span>
            </span>
          </div>
        </div>
        <div className="hp-cat-col instance-preview-col">
          <div className="hp-cat-col-head hp-cat-h-aug">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z"/><circle cx="12" cy="12" r="3"/></svg>
            {t("training.instance.preview")}
            <button
              className="ghost instance-preview-btn"
              type="button"
              onClick={handlePreview}
              disabled={previewLoading}
              data-testid="btn-instance-preview"
              data-desc={t("training.instance.preview.desc")}
            >
              {previewLoading ? t("training.instance.previewLoading") : t("training.instance.previewRun")}
            </button>
          </div>
          {preview ? (
            <>
              <div className="instance-preview-strip" data-testid="instance-preview-strip">
                {preview.samples.map((sample, i) => (
                  <figure className="instance-preview-item" key={i}>
                    <img src={sample.image} alt={`preview ${i + 1}`} />
                    <figcaption>{t("training.instance.previewCount").replace("{n}", String(sample.n_instances))}</figcaption>
                  </figure>
                ))}
              </div>
              <div className="muted instance-preview-meta">
                {t("training.instance.previewMeta")
                  .replace("{cutouts}", String(preview.n_cutouts))
                  .replace("{sources}", String(preview.n_sources))
                  .replace("{band}", `${preview.area_band[0]}〜${preview.area_band[1]}`)}
              </div>
            </>
          ) : (
            <div className="muted instance-preview-empty">{t("training.instance.previewEmpty")}</div>
          )}
        </div>
      </div>
    </div>
  );
});
