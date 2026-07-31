// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef } from "react";

/**
 * Like setInterval, but automatically pauses while the tab is hidden
 * (document.visibilityState !== "visible") and resumes on focus. Use this
 * inside a useEffect cleanup pair to replace raw `setInterval(cb, ms)`:
 *
 *   useEffect(() => {
 *     poll();
 *     return visibleInterval(poll, 3_000);
 *   }, []);
 *
 * The first tick fires synchronously at mount (only if the tab is visible)
 * and on every visibilitychange back to "visible".
 */
export function visibleInterval(cb: () => void, intervalMs: number): () => void {
  let timer: number | null = null;
  const start = () => {
    if (timer != null) return;
    timer = window.setInterval(cb, intervalMs);
  };
  const stop = () => {
    if (timer != null) { window.clearInterval(timer); timer = null; }
  };
  const onVisibilityChange = () => {
    if (document.visibilityState === "visible") start();
    else stop();
  };
  if (document.visibilityState === "visible") start();
  document.addEventListener("visibilitychange", onVisibilityChange);
  return () => {
    document.removeEventListener("visibilitychange", onVisibilityChange);
    stop();
  };
}

/**
 * React hook wrapper around visibleInterval.
 * Usage: useVisibleInterval(() => { void fetchSomething(); }, 3000);
 */
export function useVisibleInterval(
  callback: () => void | Promise<void>,
  intervalMs: number,
  enabled: boolean = true,
): void {
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    if (!enabled || intervalMs <= 0) return;
    const tick = () => { void cbRef.current(); };
    if (document.visibilityState === "visible") tick();
    return visibleInterval(tick, intervalMs);
  }, [intervalMs, enabled]);
}
