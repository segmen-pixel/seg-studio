// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

export type AugmentIconProps = {
  size?: number;
  className?: string;
};

/** Augment / synthesis icon: a stack of image cards with a sparkle to
 *  suggest "new variations generated from existing samples". Uses
 *  currentColor so it inherits the button text color. */
export function AugmentIcon({ size = 16, className }: AugmentIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {/* back card (faded, rotated) */}
      <rect
        x={5}
        y={9}
        width={13}
        height={13}
        rx={2}
        transform="rotate(-12 11.5 15.5)"
        opacity={0.45}
      />
      {/* front card */}
      <rect x={6} y={6} width={13} height={13} rx={2} />
      {/* tiny mountain scene inside the front card for "image" cue */}
      <path d="M9 15 L12 12 L14 14 L17 11" strokeWidth={1.4} />
      <circle cx={10} cy={10} r={1} strokeWidth={1.4} />
      {/* 4-point sparkle top-right */}
      <path
        d="M21 2.5 L21.65 4.35 L23.5 5 L21.65 5.65 L21 7.5 L20.35 5.65 L18.5 5 L20.35 4.35 Z"
        fill="currentColor"
        stroke="none"
      />
    </svg>
  );
}
