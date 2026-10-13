// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useState } from "react";
import { useI18n } from "../../i18n";
import type { ClassItem } from "../../store";

export type ClassMergeDialogProps = {
  open: boolean;
  classes: ClassItem[];
  /** the class the panel had selected, offered as the one to merge away */
  initialFromId: number;
  onClose: () => void;
  /** move `fromId`'s paint into `toId`; rename the survivor when `rename` differs */
  onMerge: (fromId: number, toId: number, rename?: string) => void;
};

/** Merge one class into another, and name what is left.
 *
 * The case this exists for: a class deleted and remade takes a new id, so a
 * project ends up with two classes that mean the same thing -- often the same
 * name twice -- and the paint split between them. Deleting one throws its
 * paint away. Here the paint moves and the survivor can be renamed in the
 * same step, because "class1 (1)" and "class1 (3)" is exactly the state that
 * needs a better name once it is one class again.
 */
export default function ClassMergeDialog({ open, classes, initialFromId, onClose, onMerge }: ClassMergeDialogProps) {
  const { t } = useI18n();
  const usable = classes.filter((c) => c.id !== 0);
  const [fromId, setFromId] = useState(initialFromId);
  const [toId, setToId] = useState(0);
  const [name, setName] = useState("");

  // Reopening with another class selected must not keep the last choice.
  useEffect(() => {
    if (!open) return;
    const from = initialFromId !== 0 ? initialFromId : (usable[0]?.id ?? 0);
    setFromId(from);
    const to = usable.find((c) => c.id !== from)?.id ?? 0;
    setToId(to);
    setName(usable.find((c) => c.id === to)?.name ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialFromId]);

  if (!open) return null;

  const from = usable.find((c) => c.id === fromId);
  const to = usable.find((c) => c.id === toId);
  const ready = !!from && !!to && fromId !== toId;
  const swatch = (c?: ClassItem) => (
    <span
      className="color-swatch"
      style={{ width: 14, height: 14, display: "inline-block", verticalAlign: "-2px",
               background: c ? `rgb(${c.color[0]},${c.color[1]},${c.color[2]})` : "transparent",
               border: "1px solid var(--text-muted)", borderRadius: 3, marginRight: 6 }}
    />
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content class-merge-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>{t("classMerge.title")}</h3>
        <p className="class-merge-lead">{t("classMerge.lead")}</p>

        <div className="class-merge-fields">
          <label>
            <span>{t("classMerge.from")}</span>
            <select value={fromId} onChange={(e) => setFromId(Number(e.target.value))}>
              {usable.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.id})</option>)}
            </select>
          </label>
          <div className="class-merge-arrow" aria-hidden="true">→</div>
          <label>
            <span>{t("classMerge.to")}</span>
            <select
              value={toId}
              onChange={(e) => {
                const id = Number(e.target.value);
                setToId(id);
                setName(usable.find((c) => c.id === id)?.name ?? "");
              }}
            >
              {usable.filter((c) => c.id !== fromId).map((c) => (
                <option key={c.id} value={c.id}>{c.name} ({c.id})</option>
              ))}
            </select>
          </label>
        </div>

        <label className="class-merge-rename">
          <span>{t("classMerge.rename")}</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder={to?.name ?? ""} />
        </label>

        <p className="class-merge-summary">
          {swatch(from)}<b>{from ? `${from.name} (${from.id})` : "-"}</b>
          {" → "}
          {swatch(to)}<b>{name.trim() || (to ? to.name : "-")}{to ? ` (${to.id})` : ""}</b>
        </p>
        <p className="class-merge-warn">{t("classMerge.warn")}</p>

        <div className="modal-actions">
          <button className="ghost" onClick={onClose}>{t("common.cancel")}</button>
          <button
            className="primary"
            disabled={!ready}
            onClick={() => { if (ready) onMerge(fromId, toId, name.trim() || undefined); }}
          >
            {t("classMerge.run")}
          </button>
        </div>
      </div>
    </div>
  );
}
