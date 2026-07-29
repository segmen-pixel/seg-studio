// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import { reorderProjects, updateProject, type Project } from "../../api";
import { useI18n } from "../../i18n";

type ProjectPreview = { thumbUrl: string | null; imageCount: number; maskCount: number };
type SortKey = "newest" | "oldest" | "name_asc" | "name_desc" | "custom";

type ProjectTileProps = {
  project: Project;
  preview: ProjectPreview | undefined;
  sortedProjects: Project[];
  sortKey: SortKey;
  selectedProjectId: string | null;
  setSelectedProjectId: React.Dispatch<React.SetStateAction<string | null>>;
  checkedIds: Set<string>;
  setCheckedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  lastCheckedId: string | null;
  setLastCheckedId: React.Dispatch<React.SetStateAction<string | null>>;
  dragId: string | null;
  setDragId: React.Dispatch<React.SetStateAction<string | null>>;
  dragOverId: string | null;
  setDragOverId: React.Dispatch<React.SetStateAction<string | null>>;
  editingNameId: string | null;
  setEditingNameId: React.Dispatch<React.SetStateAction<string | null>>;
  editingMemoId: string | null;
  setEditingMemoId: React.Dispatch<React.SetStateAction<string | null>>;
  editingTagsId: string | null;
  setEditingTagsId: React.Dispatch<React.SetStateAction<string | null>>;
  setProjects: React.Dispatch<React.SetStateAction<Project[]>>;
  projectsSummaryReady: boolean;
  projectsLoading: boolean;
  exportBusy: boolean;
  setExportModalProject: React.Dispatch<React.SetStateAction<Project | null>>;
  toggleChecked: (id: string, e: React.MouseEvent | React.ChangeEvent) => void;
  handleDelete: (id: string) => void;
  openProjectWorkspace: (projectId: string, tab: "annotate" | "training") => void;
  showToast: (msg: string) => void;
};

// Extracted verbatim from ProjectsPanel.tsx (pre-OSS refactor): one project
// card in the grid — selection / shift range-select, custom-sort drag reorder,
// inline name / memo / tags editing, and the per-tile action buttons.
const ProjectTile: React.FC<ProjectTileProps> = ({
  project,
  preview,
  sortedProjects,
  sortKey,
  selectedProjectId,
  setSelectedProjectId,
  checkedIds,
  setCheckedIds,
  lastCheckedId,
  setLastCheckedId,
  dragId,
  setDragId,
  dragOverId,
  setDragOverId,
  editingNameId,
  setEditingNameId,
  editingMemoId,
  setEditingMemoId,
  editingTagsId,
  setEditingTagsId,
  setProjects,
  projectsSummaryReady,
  projectsLoading,
  exportBusy,
  setExportModalProject,
  toggleChecked,
  handleDelete,
  openProjectWorkspace,
  showToast,
}) => {
  const { t } = useI18n();
  return (
    <div
      className={`project-tile ${selectedProjectId === project.id ? "active" : ""}${dragId === project.id ? " dragging" : ""}${dragOverId === project.id ? " drag-over" : ""}`}
      onClick={(e) => {
        // Shift+click: range-select from the last clicked tile to here
        // (inclusive), using the visible sort order. Falls back to a
        // single toggle if no anchor exists yet.
        if (e.shiftKey) {
          setCheckedIds((prev) => {
            const next = new Set(prev);
            if (lastCheckedId && lastCheckedId !== project.id) {
              const a = sortedProjects.findIndex((p) => p.id === lastCheckedId);
              const b = sortedProjects.findIndex((p) => p.id === project.id);
              if (a >= 0 && b >= 0) {
                const [lo, hi] = a < b ? [a, b] : [b, a];
                for (let i = lo; i <= hi; i++) next.add(sortedProjects[i].id);
              } else {
                next.add(project.id);
              }
            } else {
              next.add(project.id);
            }
            return next;
          });
          setLastCheckedId(project.id);
          return;
        }
        // Multi-select mode: any check toggles the bulk selection,
        // not the active-project highlight.
        if (checkedIds.size > 0) {
          setCheckedIds((prev) => {
            const next = new Set(prev);
            if (next.has(project.id)) next.delete(project.id); else next.add(project.id);
            return next;
          });
          setLastCheckedId(project.id);
        } else {
          setSelectedProjectId(project.id);
          // Quietly remember the anchor so a future shift+click works
          // even when the user has not opened multi-select via checkbox.
          setLastCheckedId(project.id);
        }
      }}
      onDoubleClick={() => openProjectWorkspace(project.id, "annotate")}
      role="button"
      tabIndex={0}
      aria-pressed={selectedProjectId === project.id}
      draggable={sortKey === "custom"}
      onDragStart={(e) => {
        if (sortKey !== "custom") return;
        setDragId(project.id);
        e.dataTransfer.effectAllowed = "move";
      }}
      onDragOver={(e) => {
        if (sortKey !== "custom" || !dragId) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        if (project.id !== dragId) setDragOverId(project.id);
      }}
      onDragLeave={() => {
        if (dragOverId === project.id) setDragOverId(null);
      }}
      onDrop={(e) => {
        e.preventDefault();
        if (sortKey !== "custom" || !dragId || dragId === project.id) {
          setDragId(null);
          setDragOverId(null);
          return;
        }
        const fromIdx = sortedProjects.findIndex((p) => p.id === dragId);
        const toIdx = sortedProjects.findIndex((p) => p.id === project.id);
        if (fromIdx < 0 || toIdx < 0) { setDragId(null); setDragOverId(null); return; }
        const reordered = [...sortedProjects];
        const [moved] = reordered.splice(fromIdx, 1);
        reordered.splice(toIdx, 0, moved);
        const newOrder = reordered.map((p) => p.id);
        setProjects(reordered.map((p, i) => ({ ...p, sort_order: i })));
        setDragId(null);
        setDragOverId(null);
        reorderProjects(newOrder).catch(() => showToast("Reorder failed"));
      }}
      onDragEnd={() => { setDragId(null); setDragOverId(null); }}
      onKeyDown={(event) => {
        if (event.key === " " || event.key === "Spacebar") {
          event.preventDefault();
          setSelectedProjectId(project.id);
        } else if (event.key === "Enter") {
          event.preventDefault();
          if (selectedProjectId === project.id) openProjectWorkspace(project.id, "annotate");
          else setSelectedProjectId(project.id);
        }
      }}
    >
      <div className="project-tile-thumb">
        <input
          type="checkbox"
          className="project-tile-checkbox"
          checked={checkedIds.has(project.id)}
          onChange={(e) => toggleChecked(project.id, e)}
          onClick={(e) => e.stopPropagation()}
        />
        {!projectsSummaryReady || projectsLoading ? (
          <div className="project-tile-empty">{t("projects.syncing")}</div>
        ) : preview?.thumbUrl ? (
          <img src={preview.thumbUrl} alt="" loading="lazy" />
        ) : (
          <div className="project-tile-empty">{t("projects.noImages")}</div>
        )}
      </div>
      <div className="project-tile-info">
        {editingNameId === project.id ? (
          <input
            className="project-tile-name-edit"
            defaultValue={project.name}
            autoFocus
            onClick={(e) => e.stopPropagation()}
            onBlur={async (e) => {
              const value = e.target.value.trim();
              setEditingNameId(null);
              if (value && value !== project.name) {
                try {
                  await updateProject(project.id, { name: value });
                  setProjects((prev) => prev.map((p) => p.id === project.id ? { ...p, name: value } : p));
                  showToast(`Renamed: ${value}`);
                } catch { showToast("Rename failed"); }
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              if (e.key === "Escape") setEditingNameId(null);
            }}
          />
        ) : (
          <div className="project-tile-name">{project.name}</div>
        )}
        {project.description && (
          <div className="project-tile-desc">{project.description}</div>
        )}
        <div className="project-tile-meta">
          {preview ? `${preview.imageCount} ${t("projects.imageCount")} · ${preview.maskCount} ${t("projects.maskCount")}` : "..."} · {project.id.slice(0, 8)}
        </div>
        {editingMemoId === project.id ? (
          <textarea
            className="project-tile-memo-edit"
            defaultValue={project.memo ?? ""}
            autoFocus
            rows={2}
            placeholder="Add a memo..."
            onClick={(e) => e.stopPropagation()}
            onBlur={async (e) => {
              const value = e.target.value.trim();
              setEditingMemoId(null);
              if (value !== (project.memo ?? "")) {
                try {
                  await updateProject(project.id, { memo: value || "" });
                  setProjects((prev) => prev.map((p) => p.id === project.id ? { ...p, memo: value || null } : p));
                } catch { /* ignore */ }
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setEditingMemoId(null);
              }
            }}
          />
        ) : (
          <div
            className="project-tile-memo"
            onClick={(e) => { e.stopPropagation(); setEditingMemoId(project.id); }}
            title={project.memo || "Click to add memo"}
          >
            {project.memo || "memo..."}
          </div>
        )}
        {editingTagsId === project.id ? (
          <input
            className="project-tile-tags-edit"
            defaultValue={(project.tags || []).join(", ")}
            autoFocus
            placeholder={t("projects.tags.placeholder")}
            onClick={(e) => e.stopPropagation()}
            onBlur={async (e) => {
              const raw = e.target.value;
              setEditingTagsId(null);
              const nextTags = Array.from(new Set(raw.split(",").map((s) => s.trim()).filter(Boolean)));
              const prevTags = project.tags || [];
              const same = nextTags.length === prevTags.length && nextTags.every((t, i) => t === prevTags[i]);
              if (same) return;
              try {
                await updateProject(project.id, { tags: nextTags });
                setProjects((prev) => prev.map((p) => p.id === project.id ? { ...p, tags: nextTags } : p));
              } catch { /* ignore */ }
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") setEditingTagsId(null);
              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
            }}
          />
        ) : (
          <div
            className="project-tile-tags"
            onClick={(e) => { e.stopPropagation(); setEditingTagsId(project.id); }}
            title={t("projects.tags.editHint")}
          >
            {(project.tags && project.tags.length > 0) ? (
              project.tags.map((tag) => (
                <span key={tag} className="project-tile-tag-chip">{tag}</span>
              ))
            ) : (
              <span className="project-tile-tag-empty">{t("projects.tags.empty")}</span>
            )}
          </div>
        )}
      </div>
      <div className="project-tile-actions">
        <button
          className="models-action-btn"
          onClick={(e) => { e.stopPropagation(); setEditingNameId(project.id); }}
          title="Rename"
          aria-label="Rename"
          data-desc={t("projects.renameDesc")}
          data-desc-pos="bottom"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M17 3a2.83 2.83 0 0 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>
          </svg>
        </button>
        <button
          className="models-action-btn"
          onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(project.id); showToast(`Copied: ${project.id.slice(0, 8)}`); }}
          title="Copy Project"
          aria-label="Copy Project"
          data-desc={t("projects.copyIdDesc")}
          data-desc-pos="bottom"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
        </button>
        <button
          className="models-action-btn"
          onClick={(e) => {
            e.stopPropagation();
            setExportModalProject(project);
          }}
          disabled={exportBusy}
          title="Export Project"
          aria-label="Export Project"
          data-desc={t("projects.exportDesc")}
          data-desc-pos="bottom"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
        </button>
        <button
          className="models-action-btn models-action-btn-danger"
          onClick={(e) => { e.stopPropagation(); handleDelete(project.id); }}
          title="Delete Project"
          aria-label="Delete Project"
          data-desc={t("projects.deleteDesc")}
          data-desc-pos="bottom"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default React.memo(ProjectTile);
