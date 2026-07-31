// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useEffect } from "react";
import { useMaskStore, type ClassItem } from "../../store";
import {
  fetchClasses,
  saveClasses,
  purgeClass,
  fetchClassReconcile,
  postClassReconcile,
  API_BASE,
} from "../../api";
import { DEFAULT_CLASS_COLORS, nextPaletteColor } from "../annotatorTypes";
import type { CacheEntry } from "../annotatorTypes";
import { useI18n } from "../../i18n";
import type { RegionLabel } from "./usePixelStats";

/**
 * Class CRUD (add / update / deactivate / delete) with debounced autosave.
 */
export function useClassManager(
  projectId: string | null,
  projectRef: React.MutableRefObject<string | null>,
  classesDraft: ClassItem[],
  setClassesDraft: React.Dispatch<React.SetStateAction<ClassItem[]>>,
  classesDraftRef: React.MutableRefObject<ClassItem[]>,
  classIdCounterRef: React.MutableRefObject<number>,
  classSaveTimerRef: React.MutableRefObject<number | null>,
  skipClassAutoSaveRef: React.MutableRefObject<boolean>,
  cacheRef: React.MutableRefObject<Map<string, CacheEntry>>,
  activeImageId: string | null,
  autoSaveTimerRef: React.MutableRefObject<number | null>,
  saveMask: (
    imageId: string,
    mask: Uint8Array,
    w: number,
    h: number,
    projectIdOverride?: string | null
  ) => Promise<void>,
  scheduleDrawOverlay: () => void,
  setRegionLabels: (labels: RegionLabel[]) => void,
  setStatus: (msg: string) => void
) {
  const { t } = useI18n();
  const _setClasses = useMaskStore((s) => s.setClasses);
  const setActiveClass = useMaskStore((s) => s.setActiveClass);
  const activeClassId = useMaskStore((s) => s.activeClassId);

  // ---------------------------------------------------------------------------
  // loadClasses
  // ---------------------------------------------------------------------------
  async function loadClasses(id: string) {
    try {
      const payload = await fetchClasses(id);
      if (projectRef.current !== id) return;
      const classes: ClassItem[] = (payload.classes || []).map(
        (c: ClassItem) => {
          const fallback = DEFAULT_CLASS_COLORS[c.id];
          const color =
            Array.isArray(c.color) && c.color.length === 3
              ? (c.color as [number, number, number])
              : fallback ?? ([200, 200, 200] as [number, number, number]);
          return { ...c, color, active: c.id === 0 ? true : c.active ?? true };
        }
      );
      if (!classes.find((c) => c.id === 0)) {
        classes.unshift({
          id: 0,
          name: "background",
          color: [0, 0, 0],
          active: true,
        });
      }
      const savedCounter =
        typeof payload.next_class_id === "number" ? payload.next_class_id : 0;
      (classIdCounterRef as React.MutableRefObject<number>).current = Math.max(
        savedCounter,
        ...classes.map((c) => c.id)
      );
      (skipClassAutoSaveRef as React.MutableRefObject<boolean>).current = true;
      setClassesDraft(classes);
      useMaskStore.getState().setClasses(classes);
      const preferredActive =
        classes.find((c) => c.id !== 0 && c.active !== false) ??
        classes.find((c) => c.id !== 0) ??
        classes.find((c) => c.id === 0);
      setActiveClass(preferredActive?.id ?? 0);
      setStatus("");
      // Check for orphan class IDs in masks after loading classes
      checkOrphanClasses(id);
    } catch (err) {
      // Fallback: ensure at least background class exists so UI isn't empty
      if (projectRef.current === id) {
        const fallback: ClassItem[] = [
          { id: 0, name: "background", color: [0, 0, 0], active: true },
        ];
        setClassesDraft(fallback);
        useMaskStore.getState().setClasses(fallback);
        setActiveClass(0);
      }
      setStatus(`Load failed: ${(err as Error).message}`);
    }
  }

  // ---------------------------------------------------------------------------
  // checkOrphanClasses – warn about mask pixels with unknown class IDs
  // ---------------------------------------------------------------------------
  async function checkOrphanClasses(id: string) {
    try {
      const result = await fetchClassReconcile(id);
      if (projectRef.current !== id) return;
      if (result.orphan_ids.length > 0) {
        // Auto-reconcile instead of just warning
        const reconciled = await postClassReconcile(id);
        if (projectRef.current !== id) return;
        if (reconciled.added.length > 0) {
          // Reload classes to pick up recovered entries
          await loadClasses(id);
          setStatus(
            t("annotator.classReconciled").replace("{names}", reconciled.added.map((c) => c.name).join(", "))
          );
        }
      }
    } catch {
      // Non-critical – skip silently
    }
  }

  // ---------------------------------------------------------------------------
  // saveClassList
  // ---------------------------------------------------------------------------
  async function saveClassList() {
    const pid = projectRef.current;
    const draft = classesDraftRef.current;
    if (!pid || draft.length === 0) return;
    const payload = {
      version: 1,
      ignore_index: 255,
      classes: draft,
      next_class_id: classIdCounterRef.current,
    };
    try {
      const saved = await saveClasses(pid, payload);
      // Only apply response if we're still on the same project
      if (projectRef.current === pid && saved?.classes) {
        useMaskStore.getState().setClasses(saved.classes);
      }
      setStatus("Classes saved.");
    } catch (err) {
      setStatus(`Save failed: ${(err as Error).message}`);
    }
  }

  // ---------------------------------------------------------------------------
  // addClass
  // ---------------------------------------------------------------------------
  function addClass() {
    let createdId: number | null = null;
    setClassesDraft((prev) => {
      const activeClasses = prev.filter((c) => c.id !== 0);
      if (activeClasses.length >= 10) return prev;
      const nextId = classIdCounterRef.current + 1;
      if (nextId > 254) return prev; // Uint8Array max (255 = ignore_index)
      (classIdCounterRef as React.MutableRefObject<number>).current = nextId;
      const usedColors = activeClasses.map((c) => c.color);
      const color: [number, number, number] = nextPaletteColor(usedColors);
      createdId = nextId;
      const usedNames = new Set(prev.map((c) => c.name));
      let dn = 1;
      while (usedNames.has(`class${dn}`)) dn++;
      return [
        ...prev,
        { id: nextId, name: `class${dn}`, color, active: true },
      ];
    });
    if (createdId !== null) {
      setActiveClass(createdId);
    }
  }

  // ---------------------------------------------------------------------------
  // updateClass
  // ---------------------------------------------------------------------------
  function updateClass(idx: number, patch: Partial<ClassItem>) {
    setClassesDraft((prev) =>
      prev.map((c, i) => (i === idx ? { ...c, ...patch } : c))
    );
  }

  // ---------------------------------------------------------------------------
  // deactivateClass
  // ---------------------------------------------------------------------------
  function deactivateClass(idx: number) {
    const cls = classesDraft[idx];
    if (!cls || cls.id === 0) return;
    setClassesDraft((prev) => {
      const updated = prev.map((c, i) =>
        i === idx ? { ...c, active: false } : c
      );
      if (activeClassId === cls.id) {
        const nextActive = updated.find(
          (c) => c.id !== 0 && c.id !== cls.id && c.active
        );
        if (nextActive) setActiveClass(nextActive.id);
      }
      return updated;
    });
  }

  // ---------------------------------------------------------------------------
  // handleDeleteClass – the full delete handler (purge server, clear masks, ...)
  // ---------------------------------------------------------------------------
  async function handleDeleteClass() {
    if (!projectId || activeClassId === 0) return;
    const cls = classesDraft.find((c) => c.id === activeClassId);
    const name = cls ? cls.name : `class ${activeClassId}`;
    if (!window.confirm(`Delete "${name}" and erase from all masks?`)) return;
    const removedId = activeClassId;
    try {
      await purgeClass(projectId, removedId);
    } catch {
      // Class may not exist on server yet - that's OK, just remove locally
    }
    // Cancel pending autosave (both mask and class timers)
    if (autoSaveTimerRef.current !== null) {
      window.clearTimeout(autoSaveTimerRef.current);
      (autoSaveTimerRef as React.MutableRefObject<number | null>).current =
        null;
    }
    if (classSaveTimerRef.current !== null) {
      window.clearTimeout(classSaveTimerRef.current);
      (classSaveTimerRef as React.MutableRefObject<number | null>).current =
        null;
    }
    // Clean removed class pixels from live mask (preserve other classes)
    const curMask = useMaskStore.getState().maskIndex;
    if (curMask) {
      for (let i = 0; i < curMask.length; i++) {
        if (curMask[i] === removedId) curMask[i] = 0;
      }
    }
    // Purge removed class from undo/redo stacks
    const st = useMaskStore.getState();
    for (const entry of [...st.undoStack, ...st.redoStack]) {
      for (let i = 0; i < entry.prevValues.length; i++) {
        if (entry.prevValues[i] === removedId) entry.prevValues[i] = 0;
      }
    }
    // Bump maskVersion so regionLabels & overlay react
    useMaskStore.setState({ maskVersion: st.maskVersion + 1 });
    // Clean removed class pixels from ALL cached masks (preserve other classes)
    for (const [, entry] of cacheRef.current.entries()) {
      if (!entry.maskIndex) continue;
      let changed = false;
      for (let i = 0; i < entry.maskIndex.length; i++) {
        if (entry.maskIndex[i] === removedId) {
          entry.maskIndex[i] = 0;
          changed = true;
        }
      }
      if (changed && entry.touched) {
        for (let i = 0; i < entry.touched.length; i++) entry.touched[i] = 1;
      }
    }
    scheduleDrawOverlay();
    // Save current image mask
    const { width, height } = useMaskStore.getState();
    if (activeImageId && curMask && width && height) {
      try {
        await saveMask(activeImageId, curMask, width, height);
      } catch { /* ignore */ }
    }
    setRegionLabels([]);
    // Update class list - immediately update ref to prevent
    // beforeunload from saving stale draft with deleted class
    const remaining = classesDraft.filter((item) => item.id !== removedId);
    (classesDraftRef as React.MutableRefObject<ClassItem[]>).current =
      remaining;
    (skipClassAutoSaveRef as React.MutableRefObject<boolean>).current = true;
    setClassesDraft(remaining);
    useMaskStore.getState().setClasses(remaining);
    const nextActive =
      remaining.find((c) => c.id !== 0 && c.active !== false) ??
      remaining.find((c) => c.id !== 0);
    setActiveClass(nextActive?.id ?? 0);
    // Save updated class list to server (with counter)
    try {
      const saved = await saveClasses(projectId, {
        version: 1,
        ignore_index: 255,
        classes: remaining,
        next_class_id: classIdCounterRef.current,
      });
      if (saved?.classes) {
        (skipClassAutoSaveRef as React.MutableRefObject<boolean>).current =
          true;
        setClassesDraft(saved.classes);
        useMaskStore.getState().setClasses(saved.classes);
      }
    } catch { /* ignore */ }
    setStatus(`Deleted class "${name}"`);
  }

  // ---------------------------------------------------------------------------
  // Autosave classes (debounced 800ms)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!projectId || classesDraft.length === 0) return;
    if (skipClassAutoSaveRef.current) {
      (skipClassAutoSaveRef as React.MutableRefObject<boolean>).current = false;
      return;
    }
    if (classSaveTimerRef.current !== null)
      window.clearTimeout(classSaveTimerRef.current);
    (classSaveTimerRef as React.MutableRefObject<number | null>).current =
      window.setTimeout(() => {
        (classSaveTimerRef as React.MutableRefObject<number | null>).current =
          null;
        saveClassList();
      }, 800);
    const timerRef = classSaveTimerRef;
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        (timerRef as React.MutableRefObject<number | null>).current =
          null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classesDraft, projectId]);

  // ---------------------------------------------------------------------------
  // Flush pending class save on browser close / reload
  // ---------------------------------------------------------------------------
  useEffect(() => {
    function onBeforeUnload() {
      if (classSaveTimerRef.current !== null) {
        window.clearTimeout(classSaveTimerRef.current);
        (classSaveTimerRef as React.MutableRefObject<number | null>).current =
          null;
      }
      const draft = classesDraftRef.current;
      const pid = projectRef.current;
      if (pid && draft.length > 0) {
        fetch(`${API_BASE}/projects/${pid}/classes`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            version: 1,
            ignore_index: 255,
            classes: draft,
            next_class_id: classIdCounterRef.current,
          }),
          keepalive: true,
        });
      }
    }
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    loadClasses,
    saveClassList,
    addClass,
    updateClass,
    deactivateClass,
    handleDeleteClass,
  };
}
