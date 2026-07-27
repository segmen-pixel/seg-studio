// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type { ToolId } from "./annotatorTypes";

export function ToolIcon({ id }: { id: ToolId }) {
  switch (id) {
    case "brush":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 21c3 0 5-2 6-4" />
          <path d="M9 17l9-9 3 3-9 9" />
          <path d="M16 5l3 3" />
        </svg>
      );
    case "move":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3v18M3 12h18" />
          <path d="M12 3l-3 3M12 3l3 3" />
          <path d="M12 21l-3-3M12 21l3-3" />
          <path d="M3 12l3-3M3 12l3 3" />
          <path d="M21 12l-3-3M21 12l-3 3" />
        </svg>
      );
    case "eraser":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 16l7-7 5 5-7 7H4z" />
          <path d="M14 20h6" />
        </svg>
      );
    case "bucket":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 11l6-6 8 8-6 6z" />
          <path d="M3 19h18" />
          <path d="M16 13c1 1 1 2 0 3-1 1-2 1-3 0-1-1-1-2 0-3" />
        </svg>
      );
    case "wand":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 21L15 9" />
          <path d="M15 9l2-2 4 4-2 2z" />
          <path d="M17 3v2M21 7h-2M17 11v-2M13 7h2" />
          <circle cx="17" cy="7" r="0.5" />
        </svg>
      );
    case "sam":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="9" strokeDasharray="4 2" />
          <circle cx="12" cy="12" r="2" fill="currentColor" />
          <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
        </svg>
      );
    case "sambox":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="4" y="4" width="16" height="16" rx="1" strokeDasharray="4 2" />
          <circle cx="12" cy="12" r="2" fill="currentColor" />
        </svg>
      );
    case "measure":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 20l12-12 4 4-12 12z" />
          <path d="M8 16l2-2" />
          <path d="M10 18l2-2" />
          <path d="M12 20l2-2" />
        </svg>
      );
    case "spotdetect":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="6" cy="6" r="2" fill="currentColor" />
          <circle cx="17" cy="5" r="1.5" fill="currentColor" />
          <circle cx="10" cy="13" r="2.5" fill="currentColor" />
          <circle cx="19" cy="14" r="1.8" fill="currentColor" />
          <circle cx="7" cy="19" r="1.5" fill="currentColor" />
          <circle cx="15" cy="20" r="2" fill="currentColor" />
        </svg>
      );
    case "superpixel":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2l5 3v6l-5 3-5-3V5z" />
          <path d="M2 9l5 3v6l-5 3V9z" />
          <path d="M17 9l5 3v6l-5 3V9z" />
        </svg>
      );
    case "cracktrace":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 4 L8 9 L6 14 L10 17 L8 22" fill="none" strokeWidth="2.5" />
          <path d="M14 2 L17 7 L15 12 L19 16 L16 21" fill="none" strokeWidth="1.5" strokeDasharray="3 2" />
        </svg>
      );
    default:
      return null;
  }
}
