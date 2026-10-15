// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type React from "react";

/**
 * An Enter that only finishes typing in an input method (Japanese, Chinese,
 * Korean): it confirms the conversion, and must not also send the message or
 * commit the field. A reply typed in Japanese went off at the Enter that
 * confirmed its first word. While a word is being converted the event says
 * isComposing; Safari on macOS sends the Enter that ends the conversion after
 * compositionend, with isComposing already false and keyCode 229, so both
 * are read.
 */
export function isImeEnter(ev: React.KeyboardEvent): boolean {
  const native = ev.nativeEvent as KeyboardEvent;
  return native.isComposing || native.keyCode === 229;
}
