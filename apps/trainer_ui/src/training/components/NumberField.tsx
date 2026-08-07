// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";

type NumberFieldProps = {
  value: number;
  onCommit: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  /** parseInt instead of parseFloat */
  integer?: boolean;
  /** extra clamp applied after min/max (e.g. couple a range pair) */
  transform?: (v: number) => number;
  /** side effect fired before any commit (e.g. switch Auto Config off) */
  onBeforeCommit?: () => void;
  disabled?: boolean;
  id?: string;
  style?: React.CSSProperties;
  className?: string;
  title?: string;
  "data-testid"?: string;
  "aria-label"?: string;
};

/**
 * Controlled numeric input that stays typeable.
 *
 * A plain `<input type="number" value={n} onChange={if (!isNaN) set(n)}>`
 * swallows the transient empty string when the user Backspaces the field
 * (the display pins to the last valid number and typing appears dead), and
 * echoing committed values straight back through props races multi-char
 * typing ("0.95" becomes "0.595"). This keeps the raw text in local state,
 * pushes normalized numbers to the form, and skips exactly the prop echo
 * it caused itself; blur restores the last valid value when the text is
 * unparsable.
 */
export default function NumberField({
  value,
  onCommit,
  min,
  max,
  step,
  integer,
  transform,
  onBeforeCommit,
  ...rest
}: NumberFieldProps) {
  const skipSync = React.useRef(false);
  const [text, setText] = React.useState<string>(String(value));
  React.useEffect(() => {
    if (skipSync.current) { skipSync.current = false; return; }
    setText(String(value));
  }, [value]);

  const clamp = (v: number): number => {
    let out = v;
    if (min !== undefined) out = Math.max(min, out);
    if (max !== undefined) out = Math.min(max, out);
    if (transform) out = transform(out);
    return out;
  };
  const parse = (raw: string): number => (integer ? parseInt(raw, 10) : parseFloat(raw));

  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={text}
      onChange={(e) => {
        const raw = e.target.value;
        setText(raw);
        const v = parse(raw);
        if (!isNaN(v)) {
          skipSync.current = true;
          onBeforeCommit?.();
          onCommit(clamp(v));
        }
      }}
      onBlur={() => {
        const v = parse(text);
        const resolved = isNaN(v) ? clamp(value) : clamp(v);
        onCommit(resolved);
        setText(String(resolved));
      }}
      {...rest}
    />
  );
}
