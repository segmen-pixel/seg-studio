// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useState } from "react";
import {
  snapToStride,
  DEFAULT_OUTPUT_STRIDE,
  VALID_OUTPUT_STRIDES,
  type TrainRunItem,
} from "../../utils";

export const INPUT_SIZE_PRESETS = [128, 192, 256, 320, 384, 512] as const;
export const PATCH_SIZE_PRESETS = [128, 256, 512] as const;

export type TrainFormState = {
  // Basic
  trainEpochs: number;
  trainAutoEpochs: boolean;
  trainPatchSize: number;
  trainFgRatioMin: number;
  trainFgRatioMax: number;
  trainValPct: number;
  trainTestPct: number;
  trainKFolds: number;
  trainSplitMethod: "hash" | "embedding_stratified";
  trainIterativeMode: boolean;
  trainTargetPrecision: number;
  trainTargetRecall: number;
  trainTargetConfidence: number;
  trainIterMax: number;
  trainHardWeightBoost: number;
  trainContextExpand: number;
  // Augmentation
  trainAugmentEnabled: boolean;
  trainAugmentHFlipProb: number;
  trainAugmentVFlipProb: number;
  trainAugmentRotate90Prob: number;
  trainAugmentBrightness: number;
  trainAugmentContrast: number;
  trainAugmentNoiseStd: number;
  // Model
  trainOutputStride: number;
  trainLearningRate: number;
  trainUseClassWeights: boolean;
  trainUseAutoClassWeightStrength: boolean;
  trainClassWeightStrength: number;
  trainUseAutoBackgroundBoost: boolean;
  trainBackgroundWeightBoost: number;
  trainEarlyStoppingPatience: number;
  trainMinEpochs: number;
  trainModelName: string;
  trainMemo: string;
  trainBaseChannels: number;
  trainArch: "simpleunet" | "stdc" | "deeplabv3plus";
  trainLossType: "auto" | "ce" | "focal" | "lovasz";
  trainDeepSupervision: boolean;
  trainFrequencyMap: boolean;
  // UI
  hyperParamsOpen: boolean;
  trainAllImages: boolean;
  // DINOv2
  useDinov2: boolean;
  useAutoConfig: boolean;
};

export type TrainFormSetters = {
  [K in keyof TrainFormState as `set${Capitalize<string & K>}`]: React.Dispatch<React.SetStateAction<TrainFormState[K]>>;
};

export type TrainFormReturn = TrainFormState & TrainFormSetters & {
  buildPayload: (
    runs: TrainRunItem[],
    isStartingTrain: boolean,
    trainingMode: string | null,
  ) => Record<string, unknown> | null;
};

export function useTrainForm(): TrainFormReturn {
  const [trainEpochs, setTrainEpochs] = useState(80);
  const [trainAutoEpochs, setTrainAutoEpochs] = useState(true);
  // Fixed training params — not exposed in the simplified form. They still
  // flow into the payload; backend auto-config may override them per run.
  const trainBatch = 8;
  const trainWidth = 256;
  const trainHeight = 256;
  const trainCropForeground = false;
  const trainCropScale = 0.7;
  const trainPatchesPerImage = 8;
  const [trainPatchSize, setTrainPatchSize] = useState(256);
  const [trainFgRatioMin, setTrainFgRatioMin] = useState(60);
  const [trainFgRatioMax, setTrainFgRatioMax] = useState(80);
  const [trainValPct, setTrainValPct] = useState(15);
  const [trainTestPct, setTrainTestPct] = useState(10);
  const [trainKFolds, setTrainKFolds] = useState(1);
  const [trainSplitMethod, setTrainSplitMethod] = useState<"hash" | "embedding_stratified">("hash");
  const [trainIterativeMode, setTrainIterativeMode] = useState(false);
  const [trainTargetPrecision, setTrainTargetPrecision] = useState(0.80);
  const [trainTargetRecall, setTrainTargetRecall] = useState(0.90);
  const [trainTargetConfidence, setTrainTargetConfidence] = useState(0.70);
  const [trainIterMax, setTrainIterMax] = useState(3);
  const [trainHardWeightBoost, setTrainHardWeightBoost] = useState(3.0);
  const [trainContextExpand, setTrainContextExpand] = useState(3.0);
  const [trainAugmentEnabled, setTrainAugmentEnabled] = useState(true);
  const [trainAugmentHFlipProb, setTrainAugmentHFlipProb] = useState(0.5);
  const [trainAugmentVFlipProb, setTrainAugmentVFlipProb] = useState(0.0);
  const [trainAugmentRotate90Prob, setTrainAugmentRotate90Prob] = useState(0.25);
  const [trainAugmentBrightness, setTrainAugmentBrightness] = useState(0.15);
  const [trainAugmentContrast, setTrainAugmentContrast] = useState(0.15);
  const [trainAugmentNoiseStd, setTrainAugmentNoiseStd] = useState(0.02);
  const [trainOutputStride, setTrainOutputStride] = useState(DEFAULT_OUTPUT_STRIDE);
  const [trainLearningRate, setTrainLearningRate] = useState(0.0005);
  const [trainUseClassWeights, setTrainUseClassWeights] = useState(true);
  const [trainUseAutoClassWeightStrength, setTrainUseAutoClassWeightStrength] = useState(true);
  const [trainClassWeightStrength, setTrainClassWeightStrength] = useState(0.5);
  const [trainUseAutoBackgroundBoost, setTrainUseAutoBackgroundBoost] = useState(true);
  const [trainBackgroundWeightBoost, setTrainBackgroundWeightBoost] = useState(2.0);
  const [trainEarlyStoppingPatience, setTrainEarlyStoppingPatience] = useState(15);
  const [trainMinEpochs, setTrainMinEpochs] = useState(5);
  const [trainModelName, setTrainModelName] = useState("");
  const [trainMemo, setTrainMemo] = useState("");
  const [trainBaseChannels, setTrainBaseChannels] = useState(128);
  const [trainArch, setTrainArch] = useState<"simpleunet" | "stdc" | "deeplabv3plus">("deeplabv3plus");
  const [trainLossType, setTrainLossType] = useState<"auto" | "ce" | "focal" | "lovasz">("auto");
  const [trainDeepSupervision, setTrainDeepSupervision] = useState(true);
  const [trainFrequencyMap, setTrainFrequencyMap] = useState(true);


  // UI
  const [hyperParamsOpen, setHyperParamsOpen] = useState(false);
  const [trainAllImages, setTrainAllImages] = useState(true);

  // DINOv2
  const [useDinov2, setUseDinov2] = useState(false);

  const [useAutoConfig, setUseAutoConfig] = useState(true);

  function buildPayload(
    runs: TrainRunItem[],
    isStartingTrain: boolean,
    trainingMode: string | null,
  ): Record<string, unknown> | null {
    if (isStartingTrain) return null;

    let trimmedName = trainModelName.trim();
    if (!trimmedName) {
      const today = new Date().toISOString().slice(0, 10);
      const todayRuns = runs.filter((r) => r.model_name?.startsWith(today));
      const seq = String(todayRuns.length + 1).padStart(3, "0");
      trimmedName = `${today}_${seq}`;
    }

    const epochs = Number.isFinite(trainEpochs) && trainEpochs > 0 ? trainEpochs : 30;
    const batch = Number.isFinite(trainBatch) && trainBatch > 0 ? trainBatch : 1;
    const outputStride = VALID_OUTPUT_STRIDES.includes(trainOutputStride as 1 | 2 | 4)
      ? trainOutputStride
      : DEFAULT_OUTPUT_STRIDE;
    if (outputStride !== trainOutputStride) setTrainOutputStride(outputStride);
    const inputW = snapToStride(trainWidth, outputStride);
    const inputH = snapToStride(trainHeight, outputStride);
    const cropScale = Number.isFinite(trainCropScale)
      ? Math.max(0.2, Math.min(1.0, trainCropScale))
      : 0.7;
    const lr = Number.isFinite(trainLearningRate) && trainLearningRate > 0 ? trainLearningRate : 0.0003;
    const patchSize = Number.isFinite(trainPatchSize) && trainPatchSize >= 0 ? trainPatchSize : 256;
    const patchesPerImage = Number.isFinite(trainPatchesPerImage) && trainPatchesPerImage > 0
      ? trainPatchesPerImage
      : 4;
    const fgPatchProb = Math.max(0, Math.min(1, (trainFgRatioMin + trainFgRatioMax) / 200));
    const augmentHFlipProb = Number.isFinite(trainAugmentHFlipProb)
      ? Math.max(0, Math.min(1, trainAugmentHFlipProb))
      : 0.5;
    const augmentVFlipProb = Number.isFinite(trainAugmentVFlipProb)
      ? Math.max(0, Math.min(1, trainAugmentVFlipProb))
      : 0.0;
    const augmentRotate90Prob = Number.isFinite(trainAugmentRotate90Prob)
      ? Math.max(0, Math.min(1, trainAugmentRotate90Prob))
      : 0.25;
    const augmentBrightness = Number.isFinite(trainAugmentBrightness)
      ? Math.max(0, Math.min(1, trainAugmentBrightness))
      : 0.15;
    const augmentContrast = Number.isFinite(trainAugmentContrast)
      ? Math.max(0, Math.min(1, trainAugmentContrast))
      : 0.15;
    const augmentNoiseStd = Number.isFinite(trainAugmentNoiseStd)
      ? Math.max(0, Math.min(0.5, trainAugmentNoiseStd))
      : 0.02;
    const useClassWeights = !!trainUseClassWeights;
    const classWeightStrength = Number.isFinite(trainClassWeightStrength)
      ? Math.max(0, Math.min(1, trainClassWeightStrength))
      : 0.5;
    const bgWeightBoost = Number.isFinite(trainBackgroundWeightBoost)
      ? Math.max(1, Math.min(3, trainBackgroundWeightBoost))
      : 2.0;
    const earlyStoppingPatience = Number.isFinite(trainEarlyStoppingPatience) && trainEarlyStoppingPatience >= 0
      ? trainEarlyStoppingPatience
      : 12;
    const minEpochsRaw = Number.isFinite(trainMinEpochs) && trainMinEpochs > 0 ? trainMinEpochs : 5;
    const minEpochs = Math.max(1, Math.min(minEpochsRaw, epochs));
    if (minEpochs !== trainMinEpochs) setTrainMinEpochs(minEpochs);

    const payload: Record<string, unknown> = {
      model_name: trimmedName.length ? trimmedName : undefined,
      memo: trainMemo.trim().length ? trainMemo.trim() : undefined,
      epochs,
      auto_epochs: trainAutoEpochs,
      batch_size: batch,
      lr,
      input_size: [inputW, inputH],
      crop_foreground: !!trainCropForeground,
      crop_scale: cropScale,
      output_stride: outputStride,
      patch_size: patchSize,
      patches_per_image: patchesPerImage,
      fg_patch_prob: fgPatchProb,
      augment_enabled: !!trainAugmentEnabled,
      augment_hflip_prob: augmentHFlipProb,
      augment_vflip_prob: augmentVFlipProb,
      augment_rotate90_prob: augmentRotate90Prob,
      augment_brightness: augmentBrightness,
      augment_contrast: augmentContrast,
      augment_noise_std: augmentNoiseStd,
      use_class_weights: useClassWeights,
      early_stopping_patience: earlyStoppingPatience,
      min_epochs: minEpochs,
      distill_mode: useDinov2 ? "feature" : "off",
      distill_feature_weight: 1.0,
      distill_feature_loss: "smooth_l1",
      distill_teacher_model_dir: useDinov2 ? "dinov2_vitb14" : undefined,
      base_channels: trainBaseChannels,
      arch: trainArch,
      postprocess_min_area: 0,
      deep_supervision: trainDeepSupervision,
      frequency_map: trainFrequencyMap,
      annotation_patches_only: true,
      context_expand: trainContextExpand,
      val_ratio: trainValPct / 100,
      test_ratio: trainTestPct / 100,
      k_folds: trainKFolds,
      split_method: trainSplitMethod,
      iterative_mode: trainIterativeMode,
      target_precision: trainTargetPrecision,
      target_recall: trainTargetRecall,
      target_confidence: trainTargetConfidence,
      iter_max: trainIterMax,
      hard_weight_boost: trainHardWeightBoost,
      include_unmasked: trainAllImages,
      // Single Auto knob = recipe/config recommendation only. Weight
      // transfer is opt-in via the explicit transfer-learning mode; the
      // backend's automatic donor path was removed (ADR-005 addendum).
      auto_mode: useAutoConfig ? "recipe_only" : "off",
      training_mode: trainingMode ?? "standard",
    };
    if (useClassWeights && !trainUseAutoClassWeightStrength) {
      payload.class_weight_strength = classWeightStrength;
    }
    if (useClassWeights && !trainUseAutoBackgroundBoost) {
      payload.background_weight_boost = bgWeightBoost;
    }
    // "auto" → omit loss_type so the backend resolves the data-driven recipe.
    if (trainLossType !== "auto") {
      payload.loss_type = trainLossType;
    }
    return payload;
  }

  return {
    trainEpochs, setTrainEpochs,
    trainAutoEpochs, setTrainAutoEpochs,
    trainPatchSize, setTrainPatchSize,
    trainFgRatioMin, setTrainFgRatioMin,
    trainFgRatioMax, setTrainFgRatioMax,
    trainValPct, setTrainValPct,
    trainTestPct, setTrainTestPct,
    trainKFolds, setTrainKFolds,
    trainSplitMethod, setTrainSplitMethod,
    trainIterativeMode, setTrainIterativeMode,
    trainTargetPrecision, setTrainTargetPrecision,
    trainTargetRecall, setTrainTargetRecall,
    trainTargetConfidence, setTrainTargetConfidence,
    trainIterMax, setTrainIterMax,
    trainHardWeightBoost, setTrainHardWeightBoost,
    trainContextExpand, setTrainContextExpand,
    trainAugmentEnabled, setTrainAugmentEnabled,
    trainAugmentHFlipProb, setTrainAugmentHFlipProb,
    trainAugmentVFlipProb, setTrainAugmentVFlipProb,
    trainAugmentRotate90Prob, setTrainAugmentRotate90Prob,
    trainAugmentBrightness, setTrainAugmentBrightness,
    trainAugmentContrast, setTrainAugmentContrast,
    trainAugmentNoiseStd, setTrainAugmentNoiseStd,
    trainOutputStride, setTrainOutputStride,
    trainLearningRate, setTrainLearningRate,
    trainUseClassWeights, setTrainUseClassWeights,
    trainUseAutoClassWeightStrength, setTrainUseAutoClassWeightStrength,
    trainClassWeightStrength, setTrainClassWeightStrength,
    trainUseAutoBackgroundBoost, setTrainUseAutoBackgroundBoost,
    trainBackgroundWeightBoost, setTrainBackgroundWeightBoost,
    trainEarlyStoppingPatience, setTrainEarlyStoppingPatience,
    trainMinEpochs, setTrainMinEpochs,
    trainModelName, setTrainModelName,
    trainMemo, setTrainMemo,
    trainBaseChannels, setTrainBaseChannels,
    trainArch, setTrainArch,
    trainLossType, setTrainLossType,
    trainDeepSupervision, setTrainDeepSupervision,
    trainFrequencyMap, setTrainFrequencyMap,
    hyperParamsOpen, setHyperParamsOpen,
    trainAllImages, setTrainAllImages,
    useDinov2, setUseDinov2,
    useAutoConfig, setUseAutoConfig,
    buildPayload,
  };
}
