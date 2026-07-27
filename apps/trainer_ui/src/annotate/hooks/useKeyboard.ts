// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";
import { useEffect } from "react";
import { useMaskStore, type ClassItem } from "../../store";
import type { ToolId } from "../annotatorTypes";
import type { SamRefValue, SpotDetectRefValue, SuperpixelRefValue, CrackTraceRefValue } from "./useDrawingEvents";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type KeyboardOps = {
  handleUndo: () => void;
  handleRedo: () => void;
  handleClear: () => void;
  handleSamConfirm: () => void;
  handleSamCancel: () => void;
  handleSpotConfirm: () => void;
  handleSpotCancel: () => void;
  handleSuperpixelConfirm: () => void;
  handleSuperpixelCancel: () => void;
  handleCrackConfirm: () => void;
  handleCrackCancel: () => void;
  handleMarkClean?: () => void;
  handleSaveAll: () => Promise<void>;
  handleArrowNav: (direction: "up" | "down", shiftKey: boolean) => void;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useKeyboard(
  getOps: () => KeyboardOps,
  samRef: React.MutableRefObject<SamRefValue>,
  spotDetectRef: React.MutableRefObject<SpotDetectRefValue>,
  superpixelRef: React.MutableRefObject<SuperpixelRefValue>,
  crackTraceRef: React.MutableRefObject<CrackTraceRefValue>,
  classesDraftRef: React.MutableRefObject<ClassItem[]>,
  setBrushSize: React.Dispatch<React.SetStateAction<number>>,
  gpuBusyRef?: React.MutableRefObject<boolean>
) {
  const tool = useMaskStore((s) => s.tool);
  const setTool = useMaskStore((s) => s.setTool);
  const setActiveClass = useMaskStore((s) => s.setActiveClass);

  // ---------------------------------------------------------------------------
  // Main keydown / keyup effect
  // ---------------------------------------------------------------------------
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA")
      )
        return;
      const key = event.key.toLowerCase();
      const ops = getOps();

      // Ctrl+S  save
      if ((event.metaKey || event.ctrlKey) && key === "s") {
        event.preventDefault();
        void ops.handleSaveAll();
        return;
      }

      // Ctrl+Z  undo
      if ((event.metaKey || event.ctrlKey) && key === "z") {
        event.preventDefault();
        ops.handleUndo();
        return;
      }

      // Ctrl+Y  redo
      if ((event.metaKey || event.ctrlKey) && key === "y") {
        event.preventDefault();
        ops.handleRedo();
        return;
      }

      // Delete / Backspace  clear
      if (key === "delete" || key === "backspace") {
        event.preventDefault();
        ops.handleClear();
        return;
      }

      // Shift+C  Mark Clean (no defects)
      if (key === "c" && event.shiftKey && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        ops.handleMarkClean?.();
        return;
      }

      // Arrow Up/Down: image navigation (Shift extends selection)
      // Guard: if the focused listbox already handled this key (and called preventDefault),
      // do not fire a second time — that would cause a double-step.
      if (key === "arrowup" || key === "arrowdown") {
        if (event.defaultPrevented) return;
        event.preventDefault();
        ops.handleArrowNav(key === "arrowup" ? "up" : "down", event.shiftKey);
        return;
      }

      // SAM: Enter to confirm, Escape to cancel
      if ((tool === "sam" || tool === "sambox") && samRef.current && key === "enter") {
        event.preventDefault();
        ops.handleSamConfirm();
        return;
      }
      if ((tool === "sam" || tool === "sambox") && samRef.current && key === "escape") {
        event.preventDefault();
        ops.handleSamCancel();
        return;
      }

      // Spot Detect: Enter to confirm, Escape to cancel
      if (tool === "spotdetect" && spotDetectRef.current && key === "enter") {
        event.preventDefault();
        ops.handleSpotConfirm();
        return;
      }
      if (tool === "spotdetect" && spotDetectRef.current && key === "escape") {
        event.preventDefault();
        ops.handleSpotCancel();
        return;
      }

      // Superpixel: Enter to confirm, Escape to cancel
      if (tool === "superpixel" && superpixelRef.current && key === "enter") {
        event.preventDefault();
        ops.handleSuperpixelConfirm();
        return;
      }
      if (tool === "superpixel" && superpixelRef.current && key === "escape") {
        event.preventDefault();
        ops.handleSuperpixelCancel();
        return;
      }

      // Crack Trace: Enter to confirm, Escape to cancel
      if (tool === "cracktrace" && crackTraceRef.current && key === "enter") {
        event.preventDefault();
        ops.handleCrackConfirm();
        return;
      }
      if (tool === "cracktrace" && crackTraceRef.current && key === "escape") {
        event.preventDefault();
        ops.handleCrackCancel();
        return;
      }

      // Tool shortcuts (no modifiers)
      if (!event.metaKey && !event.ctrlKey && !event.altKey) {
        const toolMap: Record<string, ToolId> = {
          b: "brush",
          e: "eraser",
          v: "move",
          g: "bucket",
          w: "wand",
          s: "sam",
          x: "sambox",
          m: "measure",
          d: "spotdetect",
          p: "superpixel",
          c: "cracktrace",
        };
        if (toolMap[key]) {
          event.preventDefault();
          // Block GPU-dependent tools when GPU is busy (training/inference)
          const gpuTools: ToolId[] = ["sam", "sambox", "cracktrace", "superpixel"];
          if (gpuTools.includes(toolMap[key]!) && gpuBusyRef?.current) return;
          if (samRef.current && toolMap[key] !== "sam" && toolMap[key] !== "sambox") ops.handleSamCancel();
          if (spotDetectRef.current && toolMap[key] !== "spotdetect")
            ops.handleSpotCancel();
          if (superpixelRef.current && toolMap[key] !== "superpixel")
            ops.handleSuperpixelCancel();
          if (crackTraceRef.current && toolMap[key] !== "cracktrace")
            ops.handleCrackCancel();
          setTool(toolMap[key]!);
          return;
        }
        // Brush size: [ shrink, ] grow
        if (key === "[") {
          event.preventDefault();
          setBrushSize((prev) => Math.max(2, prev - (prev > 20 ? 5 : 2)));
          return;
        }
        if (key === "]") {
          event.preventDefault();
          setBrushSize((prev) => Math.min(200, prev + (prev >= 20 ? 5 : 2)));
          return;
        }
        // Class switching: 1-9
        const digit = parseInt(key, 10);
        if (digit >= 1 && digit <= 9) {
          event.preventDefault();
          const draft = classesDraftRef.current;
          const fgClasses = draft.filter((c) => c.id !== 0);
          if (digit <= fgClasses.length) {
            setActiveClass(fgClasses[digit - 1]!.id);
          }
          return;
        }
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool]);
}
