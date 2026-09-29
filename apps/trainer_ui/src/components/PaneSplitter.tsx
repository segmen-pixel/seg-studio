// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * A draggable divider, and the size it controls.
 *
 * The training monitor grew one of these inline. Annotate and results then
 * wanted four more -- either side of the canvas, on both tabs -- and four
 * copies of a drag handler is four chances for one of them to feel different
 * from the others. So the behaviour lives here once: drag, double-click to put
 * it back, arrow keys for the widths a mouse cannot land on, and the size
 * remembered per key.
 *
 * What the number *means* is the caller's business. A side pane wants pixels
 * and a split grid wants a percentage, so the caller supplies `measure` and
 * decides how to spend the result; this file owns only the dragging, the
 * clamp, and the remembering.
 */
import { useCallback, useRef, useState } from "react";

export type PaneSizeOptions = {
  /** localStorage key. Two splitters sharing one key move together. */
  storageKey: string;
  /** Used when nothing is stored, and restored by double-click or Home. */
  fallback: number;
  min: number;
  max: number;
  /** The element the pointer is measured against. */
  containerRef: React.RefObject<HTMLElement | null>;
  /** Pointer position -> the number this splitter controls. */
  measure: (event: MouseEvent, rect: DOMRect) => number;
  /** One arrow-key press. */
  step?: number;
  /**
   * Which way ArrowRight moves the number. A pane anchored to the right edge
   * gets *smaller* as its divider moves right, and a keyboard that disagrees
   * with the mouse about that is worse than no keyboard at all.
   */
  arrowSign?: 1 | -1;
};

export type PaneHandle = {
  onMouseDown: (event: React.MouseEvent) => void;
  onKeyDown: (event: React.KeyboardEvent) => void;
  onDoubleClick: () => void;
  now: number;
  min: number;
  max: number;
};

export function usePaneSize(opts: PaneSizeOptions): { size: number; reset: () => void; handle: PaneHandle } {
  const { storageKey, fallback, min, max, containerRef, measure, step = 16, arrowSign = 1 } = opts;

  const clamp = useCallback(
    (value: number) => Math.max(min, Math.min(max, value)), [min, max]);

  const [size, setSize] = useState<number>(() => {
    try {
      const stored = parseFloat(localStorage.getItem(storageKey) ?? "");
      return Number.isFinite(stored) ? Math.max(min, Math.min(max, stored)) : fallback;
    } catch {
      // Private browsing, or storage disabled. A layout preference is not
      // worth failing a render over.
      return fallback;
    }
  });

  // The drag reads the newest value from a listener that closed over the
  // first render, so state alone cannot answer "what is it now".
  const sizeRef = useRef(size);
  sizeRef.current = size;

  const remember = useCallback((value: number) => {
    try {
      localStorage.setItem(storageKey, String(Math.round(value)));
    } catch { /* see above */ }
  }, [storageKey]);

  const apply = useCallback((value: number) => {
    const next = clamp(value);
    sizeRef.current = next;
    setSize(next);
    return next;
  }, [clamp]);

  const onMouseDown = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    const onMove = (ev: MouseEvent) => {
      apply(measure(ev, container.getBoundingClientRect()));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      remember(sizeRef.current);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    // On the body rather than the divider: the pointer leaves a 12px strip on
    // the first move, and a cursor flicking back to the arrow mid-drag reads
    // as the drag having been dropped.
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [apply, containerRef, measure, remember]);

  const reset = useCallback(() => {
    remember(apply(fallback));
  }, [apply, fallback, remember]);

  const onKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      remember(apply(sizeRef.current - step * arrowSign));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      remember(apply(sizeRef.current + step * arrowSign));
    } else if (event.key === "Home") {
      event.preventDefault();
      reset();
    }
  }, [apply, arrowSign, remember, reset, step]);

  return {
    size,
    reset,
    handle: { onMouseDown, onKeyDown, onDoubleClick: reset, now: size, min, max },
  };
}

export function PaneSplitter({ handle, title }: { handle: PaneHandle; title: string }) {
  return (
    <div
      className="pane-splitter"
      role="separator"
      aria-orientation="vertical"
      aria-valuenow={Math.round(handle.now)}
      aria-valuemin={handle.min}
      aria-valuemax={handle.max}
      tabIndex={0}
      title={title}
      onMouseDown={handle.onMouseDown}
      onKeyDown={handle.onKeyDown}
      onDoubleClick={handle.onDoubleClick}
    />
  );
}
