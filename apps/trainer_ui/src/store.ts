// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { TrainRunItem } from "./utils";

/** Replacer for devtools serialize: show TypedArray length instead of full data */
const typedArrayReplacer = (_key: string, value: unknown) => {
  if (value instanceof Uint8Array) return `Uint8Array(${value.length})`;
  if (value instanceof Uint32Array) return `Uint32Array(${value.length})`;
  return value;
};

export type Tool = "brush" | "eraser" | "bucket" | "colorpick" | "wand" | "sam" | "sambox" | "roi" | "measure" | "spotdetect" | "superpixel" | "cracktrace" | "move";

export type ClassItem = {
  id: number;
  name: string;
  color: [number, number, number];
  active: boolean;
};

export type MaskChange = {
  indices: Uint32Array;
  prevValues: Uint8Array;
};

export type MaskState = {
  width: number;
  height: number;
  maskIndex: Uint8Array;
  maskVersion: number;
  clipboard: Uint8Array | null;
  imageUrl: string | null;
  activeClassId: number;
  tool: Tool;
  classes: ClassItem[];
  undoStack: MaskChange[];
  redoStack: MaskChange[];
  scale: number;
  offsetX: number;
  offsetY: number;
  setImage: (url: string, width: number, height: number) => void;
  setMask: (mask: Uint8Array, width: number, height: number) => void;
  setHistory: (undoStack: MaskChange[], redoStack: MaskChange[]) => void;
  setClasses: (classes: ClassItem[]) => void;
  setActiveClass: (id: number) => void;
  setTool: (tool: Tool) => void;
  applyDelta: (indices: Uint32Array, prevValues: Uint8Array, nextValues: Uint8Array) => void;
  applyDeltaSilent: (indices: Uint32Array, prevValues: Uint8Array, nextValues: Uint8Array) => void;
  pushUndo: (indices: Uint32Array, prevValues: Uint8Array) => void;
  undo: () => void;
  redo: () => void;
  clearAll: () => void;
  copyAll: () => void;
  cutAll: () => void;
  pasteAll: () => void;
  setView: (scale: number, offsetX: number, offsetY: number) => void;
};

export const useMaskStore = create<MaskState>()(
  devtools(
  immer((set, get) => ({
    width: 0,
    height: 0,
    maskIndex: new Uint8Array(),
    maskVersion: 0,
    clipboard: null,
    imageUrl: null,
    activeClassId: 1,
    tool: "brush",
    classes: [
      { id: 0, name: "background", color: [0, 0, 0], active: true }
    ],
    undoStack: [],
    redoStack: [],
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    setImage: (url, width, height) =>
      set((state) => {
        state.imageUrl = url;
        // Always allocate new array — immer cannot detect Uint8Array.fill()
        state.maskIndex = new Uint8Array(width * height);
        state.width = width;
        state.height = height;
        state.undoStack = [];
        state.redoStack = [];
        state.scale = 1;
        state.offsetX = 0;
        state.offsetY = 0;
      }),
    setMask: (mask, width, height) =>
      set((state) => {
        const isSame = mask === state.maskIndex;
        // Always deep-copy to prevent Uint8Array aliasing between cache and store
        state.maskIndex = isSame ? mask : new Uint8Array(mask);
        state.width = width;
        state.height = height;
        state.undoStack = [];
        state.redoStack = [];
        state.maskVersion += 1;
      }),
    setHistory: (undoStack, redoStack) =>
      set((state) => {
        state.undoStack = undoStack;
        state.redoStack = redoStack;
      }),
    setClasses: (classes) =>
      set((state) => {
        state.classes = classes;
        if (!classes.find((c) => c.id === state.activeClassId)) {
          state.activeClassId =
            classes.find((c) => c.id !== 0 && c.active !== false)?.id ??
            classes.find((c) => c.id !== 0)?.id ??
            0;
        }
      }),
    setActiveClass: (id) => set((state) => { state.activeClassId = id; }),
    setTool: (tool) => set((state) => { state.tool = tool; }),
    applyDelta: (indices, prevValues, nextValues) =>
      set((state) => {
        if (indices.length === 0) return;
        for (let i = 0; i < indices.length; i += 1) {
          state.maskIndex[indices[i]] = nextValues[i];
        }
        state.undoStack.push({ indices, prevValues });
        if (state.undoStack.length > 30) state.undoStack.splice(0, state.undoStack.length - 30);
        state.redoStack = [];
        state.maskVersion += 1;
      }),
    applyDeltaSilent: (indices, _prevValues, nextValues) =>
      set((state) => {
        for (let i = 0; i < indices.length; i += 1) {
          state.maskIndex[indices[i]] = nextValues[i];
        }
        state.maskVersion += 1;
      }),
    pushUndo: (indices, prevValues) =>
      set((state) => {
        if (indices.length === 0) return;
        state.undoStack.push({ indices, prevValues });
        if (state.undoStack.length > 30) state.undoStack.splice(0, state.undoStack.length - 30);
        state.redoStack = [];
      }),
    undo: () =>
      set((state) => {
        const last = state.undoStack.pop();
        if (!last) return;
        const mask = state.maskIndex;
        const indices = last.indices;
        const prev = last.prevValues;
        const current = new Uint8Array(indices.length);
        for (let i = 0; i < indices.length; i += 1) {
          const idx = indices[i];
          current[i] = mask[idx];
          mask[idx] = prev[i];
        }
        state.redoStack.push({ indices, prevValues: current });
        state.maskVersion += 1;
      }),
    redo: () =>
      set((state) => {
        const last = state.redoStack.pop();
        if (!last) return;
        const mask = state.maskIndex;
        const indices = last.indices;
        const prev = last.prevValues;
        const current = new Uint8Array(indices.length);
        for (let i = 0; i < indices.length; i += 1) {
          const idx = indices[i];
          current[i] = mask[idx];
          mask[idx] = prev[i];
        }
        state.undoStack.push({ indices, prevValues: current });
        state.maskVersion += 1;
      }),
    clearAll: () =>
      set((state) => {
        if (state.maskIndex.length === 0) return;
        const indices = new Uint32Array(state.maskIndex.length);
        const prev = new Uint8Array(state.maskIndex.length);
        for (let i = 0; i < state.maskIndex.length; i += 1) {
          indices[i] = i;
          prev[i] = state.maskIndex[i];
          state.maskIndex[i] = 0;
        }
        state.undoStack.push({ indices, prevValues: prev });
        state.redoStack = [];
        state.maskVersion += 1;
      }),
    copyAll: () =>
      set((state) => {
        state.clipboard = new Uint8Array(state.maskIndex);
      }),
    cutAll: () => {
      get().copyAll();
      get().clearAll();
    },
    pasteAll: () =>
      set((state) => {
        if (!state.clipboard || state.clipboard.length !== state.maskIndex.length) return;
        const indices = new Uint32Array(state.maskIndex.length);
        const prev = new Uint8Array(state.maskIndex.length);
        for (let i = 0; i < state.maskIndex.length; i += 1) {
          indices[i] = i;
          prev[i] = state.maskIndex[i];
          state.maskIndex[i] = state.clipboard[i];
        }
        state.undoStack.push({ indices, prevValues: prev });
        state.redoStack = [];
        state.maskVersion += 1;
      }),
    setView: (scale, offsetX, offsetY) =>
      set((state) => {
        state.scale = scale;
        state.offsetX = offsetX;
        state.offsetY = offsetY;
      })
  })),
  {
    name: "MaskStore",
    serialize: { replacer: typedArrayReplacer },
  }
  )
);

/* ------------------------------------------------------------------ */
/*  Shared training store — single source of truth for runs & config  */
/* ------------------------------------------------------------------ */

export type TrainingState = {
  /** Sorted runs array (newest first). Shared across all tabs. */
  runs: TrainRunItem[];
  /** Project ID that the current runs belong to (stale-guard). */
  runsProjectId: string | null;
  setRuns: (projectId: string, runs: TrainRunItem[]) => void;
  clearRuns: () => void;
};

export const useTrainingStore = create<TrainingState>()(
  devtools(
  immer((set) => ({
    runs: [],
    runsProjectId: null,
    setRuns: (projectId, runs) =>
      set((state) => {
        state.runsProjectId = projectId;
        state.runs = runs as TrainRunItem[];
      }),
    clearRuns: () =>
      set((state) => {
        state.runs = [];
        state.runsProjectId = null;
      }),
  })),
  { name: "TrainingStore" }
  )
);
