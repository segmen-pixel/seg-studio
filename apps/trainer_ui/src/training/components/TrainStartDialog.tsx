// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useEffect, useState } from "react";
import { useI18n, type TranslationKey } from "../../i18n";

export type StartPhase = "preparing" | "claiming" | "queued" | "failed";

type TrainStartDialogProps = {
  phase: StartPhase;
  /** Server-supplied line for the current phase (dataset counts, error text). */
  detail?: string;
  /** Epoch ms when the user pressed start, for the elapsed counter. */
  startedAt: number;
  onClose: () => void;
};

const ORDER: StartPhase[] = ["preparing", "claiming", "queued"];

/**
 * What the app is doing between pressing start and the first training log.
 *
 * That gap is dataset preparation, and on a project with thousands of images it
 * runs for a long time. The only feedback used to be one line of text in the
 * run list, which reads as the app having ignored the click — so people press
 * start again, or conclude it is broken. This says which step is running and
 * how long it has been going, and stays up to deliver a failure rather than
 * letting the run quietly end up in the list marked failed.
 */
export default React.memo(function TrainStartDialog({ phase, detail, startedAt, onClose }: TrainStartDialogProps) {
  const { t } = useI18n();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const tick = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  const failed = phase === "failed";
  const activeIndex = failed ? -1 : ORDER.indexOf(phase);
  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;

  return (
    <div className="training-mode-help-overlay" onClick={onClose}>
      <div
        className="training-mode-help-popup"
        role="dialog"
        aria-live="polite"
        aria-busy={!failed}
        data-testid="train-start-dialog"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 420 }}
      >
        <h3 style={{ marginTop: 0, marginBottom: 12 }}>
          {failed ? t("training.startDialog.failedTitle") : t("training.startDialog.title")}
        </h3>

        {!failed && (
          <ol style={{ listStyle: "none", padding: 0, margin: "0 0 12px", fontSize: 13, lineHeight: 1.9 }}>
            {ORDER.map((step, i) => {
              const done = i < activeIndex;
              const active = i === activeIndex;
              return (
                <li key={step} style={{ opacity: done || active ? 1 : 0.45, fontWeight: active ? 600 : 400 }}>
                  {/* Shape, not colour alone: readable without colour vision. */}
                  <span style={{ display: "inline-block", width: 20 }}>
                    {done ? "✓" : active ? "▶" : "・"}
                  </span>
                  {t(`training.startDialog.${step}` as TranslationKey)}
                </li>
              );
            })}
          </ol>
        )}

        {detail && (
          <p className="muted" style={{ fontSize: 12, margin: "0 0 12px", wordBreak: "break-word" }}>
            {detail}
          </p>
        )}

        {!failed && (
          <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
            {t("training.startDialog.elapsed").replace("{time}", mmss)}
            <div style={{ marginTop: 4 }}>{t("training.startDialog.keepOpen")}</div>
          </div>
        )}
        {/* Always dismissible. A queued run keeps this open indefinitely -- it
            waits on another job -- and with no way out it would block the whole
            app, including the run list you would want to look at meanwhile. */}
        <button className="ghost" type="button" onClick={onClose} data-testid="train-start-dialog-close">
          {t("training.startDialog.close")}
        </button>
      </div>
    </div>
  );
});
