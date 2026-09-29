// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

/** Inline SVG badge for training mode (20x20 pill button style). */
export default function TrainingModeBadge({ mode }: { mode?: string | null }) {
  if (!mode) return null;
  switch (mode) {
    case "standard":
      return (
        <svg className="run-mode-badge" width="28" height="28" viewBox="0 0 16 16" fill="none" aria-label="standard">
          <rect x="1" y="1" width="14" height="14" rx="4" fill="#42A5F5"/>
          <rect x="1.5" y="1.5" width="13" height="13" rx="3.5" stroke="white" strokeOpacity="0.18"/>
          <path d="M4 5.5L8 4L12 5.5L8 7L4 5.5Z" fill="white" fillOpacity="0.95"/>
          <path d="M5 8L8 6.8L11 8L8 9.2L5 8Z" fill="white" fillOpacity="0.72"/>
          <path d="M6 10.3L8 9.5L10 10.3L8 11.1L6 10.3Z" fill="white" fillOpacity="0.5"/>
        </svg>
      );
    case "quick":
      return (
        <svg className="run-mode-badge" width="28" height="28" viewBox="0 0 16 16" fill="none" aria-label="quick">
          <rect x="1" y="1" width="14" height="14" rx="4" fill="#FF9800"/>
          <rect x="1.5" y="1.5" width="13" height="13" rx="3.5" stroke="white" strokeOpacity="0.18"/>
          <path d="M8.9 3.8L5.6 8.1H7.7L7 12.2L10.4 7.8H8.4L8.9 3.8Z" fill="white"/>
        </svg>
      );
    case "instance":
      return (
        <svg className="run-mode-badge" width="28" height="28" viewBox="0 0 16 16" fill="none" aria-label="instance">
          <rect x="1" y="1" width="14" height="14" rx="4" fill="#009E73"/>
          <rect x="1.5" y="1.5" width="13" height="13" rx="3.5" stroke="white" strokeOpacity="0.18"/>
          <circle cx="5.4" cy="5.4" r="2.1" stroke="white" strokeWidth="1.3" fill="none"/>
          <circle cx="10.6" cy="5.4" r="2.1" stroke="white" strokeWidth="1.3" fill="none"/>
          <circle cx="8" cy="10.6" r="2.1" stroke="white" strokeWidth="1.3" fill="none"/>
          <circle cx="5.4" cy="5.4" r="0.8" fill="white"/>
          <circle cx="10.6" cy="5.4" r="0.8" fill="white"/>
          <circle cx="8" cy="10.6" r="0.8" fill="white"/>
        </svg>
      );
    case "transfer":
      return (
        <svg className="run-mode-badge" width="28" height="28" viewBox="0 0 16 16" fill="none" aria-label="transfer">
          <rect x="1" y="1" width="14" height="14" rx="4" fill="#D55E00"/>
          <rect x="1.5" y="1.5" width="13" height="13" rx="3.5" stroke="white" strokeOpacity="0.18"/>
          <path d="M4.5 6H9.5" stroke="white" strokeWidth="1.6" strokeLinecap="round"/>
          <path d="M8.1 4.7L10 6L8.1 7.3" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M11.5 10H6.5" stroke="white" strokeWidth="1.6" strokeLinecap="round"/>
          <path d="M7.9 8.7L6 10L7.9 11.3" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      );
    default:
      return null;
  }
}
