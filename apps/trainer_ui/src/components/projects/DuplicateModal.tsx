// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useState } from "react";
import { duplicateProject, fetchProjectsSummary, type Project, type ProjectSummary } from "../../api";
import { useI18n } from "../../i18n";

type Props = {
  project: Project;
  onClose: () => void;
  applyProjectsSummary: (summaries: ProjectSummary[]) => void;
  setSelectedProjectId: (id: string) => void;
  showToast: (msg: string) => void;
};

/**
 * Another project on the same pictures.
 *
 * Two things had to be asked, so there is a dialog rather than a confirm: what
 * the copy is called, and whether the labels come with it. The second is the
 * real question -- duplicating to label the same images a different way is why
 * this exists at all -- and neither answer has a default that is right often
 * enough to guess.
 *
 * The cost is not asked about, because there is no longer one worth asking
 * about: the images are hard-linked, so the copy is the labels and the index
 * and nothing else.
 */
const DuplicateModal: React.FC<Props> = ({
  project, onClose, applyProjectsSummary, setSelectedProjectId, showToast,
}) => {
  const { t } = useI18n();
  const [name, setName] = useState(
    t("projects.duplicate.defaultName").replace("{name}", project.name),
  );
  const [includeMasks, setIncludeMasks] = useState(true);
  const [busy, setBusy] = useState(false);

  async function run() {
    if (busy) return;
    setBusy(true);
    try {
      const made = await duplicateProject(project.id, { name: name.trim(), includeMasks });
      const summaries = await fetchProjectsSummary();
      applyProjectsSummary(summaries);
      setSelectedProjectId(made.id);
      showToast(t("projects.duplicate.done").replace("{name}", made.name));
      onClose();
    } catch (err) {
      showToast(t("projects.duplicate.failed").replace("{msg}", (err as Error).message));
      setBusy(false);
    }
  }

  return (
    <div className="settings-overlay" onClick={busy ? undefined : onClose}>
      <div
        className="settings-panel"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 420, minHeight: "auto" }}
      >
        <div className="settings-header">
          <h2>{t("projects.duplicate.title")}</h2>
          <button className="settings-close-btn" onClick={onClose} disabled={busy}>&times;</button>
        </div>
        <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 12, opacity: 0.75 }}>{t("projects.duplicate.nameLabel")}</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={busy}
              autoFocus
              onKeyDown={(e) => { if (e.key === "Enter" && name.trim()) void run(); }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
            <input
              type="checkbox"
              checked={includeMasks}
              disabled={busy}
              onChange={(e) => setIncludeMasks(e.target.checked)}
            />
            {t("projects.duplicate.includeMasks")}
          </label>
          <p style={{ margin: 0, fontSize: 12, opacity: 0.7, lineHeight: 1.6 }}>
            {t("projects.duplicate.note")}
          </p>
          <button
            className="models-action-btn"
            style={{ width: "100%", padding: "9px 0", borderRadius: 6, fontWeight: 600,
                     background: "var(--accent)", color: "#fff" }}
            disabled={busy || !name.trim()}
            onClick={() => void run()}
          >
            {busy ? t("projects.duplicate.working") : t("projects.duplicate.go")}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DuplicateModal;
