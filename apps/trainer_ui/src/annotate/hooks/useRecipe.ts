// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useRef, useState, type RefObject } from "react";
import { useMaskStore } from "../../store";
import {
  uploadRecipe,
  fetchActiveRecipe,
  previewRecipe,
  applyRecipe,
  autoLabel,
  fetchAnnotateItems,
  type Recipe,
} from "../../api";
import { base64ToBlob, decodeMaskBlob } from "../imageProcessing";
import { useI18n } from "../../i18n";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RecipeOps = {
  markDirty: () => void;
  markTouched: (indices: Uint32Array) => void;
  autoSave: (imageId: string) => Promise<void>;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRecipe(
  projectId: string | null,
  activeImageId: string | null,
  activeImageIdRef: RefObject<string | null>,
  width: number,
  height: number,
  getOps: () => RecipeOps,
  setStatus: (msg: string) => void,
  setBusyMessage: (msg: string | null) => void,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setImages: (updater: (prev: any[]) => any[]) => void
) {
  const { t } = useI18n();
  // ---- local state ----
  const [activeRecipe, setActiveRecipe] = useState<Recipe | null>(null);
  const [recipePreview, setRecipePreview] = useState<Uint8Array | null>(null);
  const [isRecipeRunning, setIsRecipeRunning] = useState(false);
  const recipeInputRef = useRef<HTMLInputElement | null>(null);

  // ---- store ----
  const applyDelta = useMaskStore((s) => s.applyDelta);

  // ---------------------------------------------------------------------------
  // handleRecipeImport
  // ---------------------------------------------------------------------------
  async function handleRecipeImport(file: File) {
    if (!projectId) return;
    setIsRecipeRunning(true);
    setStatus(t("recipe.importing"));
    try {
      const result = await uploadRecipe(projectId, file);
      setActiveRecipe(result.recipe);
      setStatus(t("recipe.imported").replace("{name}", result.recipe.name));
    } catch (err) {
      let msg = (err as Error).message;
      try {
        msg = JSON.parse(msg).detail ?? msg;
      } catch { /* ignore */ }
      setStatus(`Recipe: ${msg}`);
    } finally {
      setIsRecipeRunning(false);
    }
  }

  // ---------------------------------------------------------------------------
  // loadActiveRecipe
  // ---------------------------------------------------------------------------
  async function loadActiveRecipe() {
    if (!projectId) return;
    try {
      const result = await fetchActiveRecipe(projectId);
      setActiveRecipe(result.recipe);
    } catch {
      // ignore
    }
  }

  // ---------------------------------------------------------------------------
  // handleRecipePreview
  // ---------------------------------------------------------------------------
  async function handleRecipePreview() {
    if (!projectId || !activeImageId || !width || !height) return;
    const targetId = activeImageId;
    setIsRecipeRunning(true);
    setRecipePreview(null);
    setStatus(t("recipe.previewing"));
    try {
      const result = await previewRecipe(projectId, targetId);
      if (activeImageIdRef.current !== targetId) return;
      const mask = await decodeMaskBlob(
        base64ToBlob(result.mask_base64),
        width,
        height
      );
      if (activeImageIdRef.current !== targetId) return;
      setRecipePreview(mask);
      const fgCount = mask.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
      setStatus(
        `Recipe: ${fgCount}px detected (${(result.fg_ratio * 100).toFixed(2)}%) -- Confirm / Cancel`
      );
    } catch (err) {
      let msg = (err as Error).message;
      try {
        msg = JSON.parse(msg).detail ?? msg;
      } catch { /* ignore */ }
      setStatus(`Recipe Preview: ${msg}`);
    } finally {
      setIsRecipeRunning(false);
    }
  }

  // ---------------------------------------------------------------------------
  // handleAutoLabelPreview — propose labels for the active image from the
  // colors/shapes of the images annotated so far (backend auto_label).
  // Reuses recipePreview so the existing Confirm / Cancel buttons apply.
  // ---------------------------------------------------------------------------
  async function handleAutoLabelPreview(classId: number) {
    if (!projectId || !activeImageId || !width || !height || classId <= 0) return;
    const targetId = activeImageId;
    setIsRecipeRunning(true);
    setRecipePreview(null);
    setStatus(t("aiAssist.autoLabel.running"));
    try {
      const blob = await autoLabel(projectId, targetId, classId);
      if (activeImageIdRef.current !== targetId) return;
      const mask = await decodeMaskBlob(blob, width, height);
      if (activeImageIdRef.current !== targetId) return;
      setRecipePreview(mask);
      const fgCount = mask.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
      setStatus(t("aiAssist.autoLabel.preview").replace("{count}", String(fgCount)));
    } catch (err) {
      let msg = (err as Error).message;
      try {
        msg = JSON.parse(msg).detail ?? msg;
      } catch { /* ignore */ }
      setStatus(`Auto-label: ${msg}`);
    } finally {
      setIsRecipeRunning(false);
    }
  }

  // ---------------------------------------------------------------------------
  // handleRecipeConfirm
  // ---------------------------------------------------------------------------
  function handleRecipeConfirm() {
    if (!recipePreview) return;
    const ops = getOps();
    const curMask = useMaskStore.getState().maskIndex;
    const indices: number[] = [],
      prev: number[] = [],
      next: number[] = [];
    for (let i = 0; i < recipePreview.length; i++) {
      if (recipePreview[i] > 0 && recipePreview[i] !== curMask[i]) {
        indices.push(i);
        prev.push(curMask[i]);
        next.push(recipePreview[i]);
      }
    }
    if (indices.length > 0) {
      const idxArr = new Uint32Array(indices);
      ops.markTouched(idxArr);
      applyDelta(idxArr, new Uint8Array(prev), new Uint8Array(next));
      ops.markDirty();
    }
    setRecipePreview(null);
    setStatus(`Recipe confirmed: ${indices.length}px applied`);
  }

  // ---------------------------------------------------------------------------
  // handleRecipeCancel
  // ---------------------------------------------------------------------------
  function handleRecipeCancel() {
    setRecipePreview(null);
    setStatus("");
  }

  // ---------------------------------------------------------------------------
  // handleRecipeApplyAll
  // ---------------------------------------------------------------------------
  async function handleRecipeApplyAll() {
    if (!projectId) return;
    const ops = getOps();
    setIsRecipeRunning(true);
    setBusyMessage(t("annotate.recipe.applying"));
    try {
      if (activeImageId) {
        await ops.autoSave(activeImageId);
      }
      const result = await applyRecipe(projectId);
      setStatus(
        t("annotate.recipe.applied").replace("{applied}", String(result.applied)).replace("{skipped}", String(result.skipped))
      );
      const data = await fetchAnnotateItems(projectId);
      setImages(() => data.items || []);
    } catch (err) {
      let msg = (err as Error).message;
      try {
        msg = JSON.parse(msg).detail ?? msg;
      } catch { /* ignore */ }
      setStatus(`Recipe Apply: ${msg}`);
    } finally {
      setIsRecipeRunning(false);
      setBusyMessage(null);
    }
  }

  // ---------------------------------------------------------------------------
  // Return
  // ---------------------------------------------------------------------------
  return {
    // state
    activeRecipe,
    setActiveRecipe,
    recipePreview,
    setRecipePreview,
    isRecipeRunning,
    setIsRecipeRunning,
    recipeInputRef,
    // functions
    handleRecipeImport,
    loadActiveRecipe,
    handleRecipePreview,
    handleAutoLabelPreview,
    handleRecipeConfirm,
    handleRecipeCancel,
    handleRecipeApplyAll,
  };
}
