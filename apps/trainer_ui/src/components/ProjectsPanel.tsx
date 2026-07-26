// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useRef, useState } from "react";
import {
  createProject,
  deleteProject,
  updateProject,
  fetchProjectsSummary,
  thumbnailUrl,
  exportDatasetUrl,
  importAnnotateZip,
  type Project,
  type ProjectSummary,
} from "../api";
import { useI18n, formatError } from "../i18n";
import ExportModal from "./projects/ExportModal";
import ProjectTile from "./projects/ProjectTile";
import CurrentProjectTrainingSummary from "./projects/CurrentProjectTrainingSummary";

interface ProjectsPanelProps {
  projects: Project[];
  setProjects: React.Dispatch<React.SetStateAction<Project[]>>;
  selectedProjectId: string | null;
  setSelectedProjectId: React.Dispatch<React.SetStateAction<string | null>>;
  projectPreviews: Record<string, { thumbUrl: string | null; imageCount: number; maskCount: number }>;
  setProjectPreviews: React.Dispatch<React.SetStateAction<Record<string, { thumbUrl: string | null; imageCount: number; maskCount: number }>>>;
  projectsLoading: boolean;
  projectsSummaryReady: boolean;
  apiStatus: "connecting" | "connected" | "error";
  currentProject: Project | null;
  currentProjectPreview: { thumbUrl: string | null; imageCount: number; maskCount: number } | null;
  openProjectWorkspace: (projectId: string, tab: "annotate" | "training") => void;
  showToast: (msg: string) => void;
}

const ProjectsPanel: React.FC<ProjectsPanelProps> = ({
  projects, setProjects,
  selectedProjectId, setSelectedProjectId,
  projectPreviews, setProjectPreviews,
  projectsLoading, projectsSummaryReady, apiStatus,
  currentProject, currentProjectPreview,
  openProjectWorkspace, showToast,
}) => {
  const { lang, t } = useI18n();

  const [projectName, setProjectName] = useState("");
  const [projectDesc, setProjectDesc] = useState("");
  const [editingMemoId, setEditingMemoId] = useState<string | null>(null);
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [editingTagsId, setEditingTagsId] = useState<string | null>(null);
  // Tag filter is a 3-state cycle per chip: none -> include -> exclude -> none.
  // Multiple includes are OR'd; the project is then narrowed by removing any
  // project that carries any of the exclude tags. include / exclude can mix.
  type TagFilterMode = "include" | "exclude";
  const [tagFilter, setTagFilter] = useState<Map<string, TagFilterMode>>(new Map());
  const [bulkTagInput, setBulkTagInput] = useState<string | null>(null);
  const [bulkTagBusy, setBulkTagBusy] = useState(false);
  const [lastCheckedId, setLastCheckedId] = useState<string | null>(null);
  const [importBusy, setImportBusy] = useState(false);
  const [importProgress, setImportProgress] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportModalProject, setExportModalProject] = useState<Project | null>(null);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const importZipInputRef = useRef<HTMLInputElement>(null);

  type SortKey = "newest" | "oldest" | "name_asc" | "name_desc" | "custom";
  const [sortKey, setSortKey] = useState<SortKey>("newest");
  const [dragId, setDragId] = useState<string | null>(null);
  const [dragOverId, setDragOverId] = useState<string | null>(null);

  const allTags = React.useMemo(() => {
    const s = new Set<string>();
    for (const p of projects) for (const t of p.tags || []) s.add(t);
    return Array.from(s).sort();
  }, [projects]);

  // Prune filter selections that no longer exist in any project.
  // Without this, deleting a tag (or the last project owning it) leaves
  // an orphaned chip selected and the list silently goes empty.
  React.useEffect(() => {
    setTagFilter((prev) => {
      if (prev.size === 0) return prev;
      const known = new Set(allTags);
      const next = new Map<string, TagFilterMode>();
      for (const [tag, mode] of prev) if (known.has(tag)) next.set(tag, mode);
      return next.size === prev.size ? prev : next;
    });
  }, [allTags]);

  const filteredProjects = React.useMemo(() => {
    if (tagFilter.size === 0) return projects;
    const includes: string[] = [];
    const excludes: string[] = [];
    for (const [tag, mode] of tagFilter) {
      (mode === "include" ? includes : excludes).push(tag);
    }
    return projects.filter((p) => {
      const projectTags = p.tags || [];
      // Any include selected → project must have at least one matching tag.
      if (includes.length > 0 && !projectTags.some((t) => includes.includes(t))) return false;
      // Any exclude selected → project must not carry any of them.
      if (excludes.length > 0 && projectTags.some((t) => excludes.includes(t))) return false;
      return true;
    });
  }, [projects, tagFilter]);

  const sortedProjects = React.useMemo(() => {
    const sorted = [...filteredProjects];
    switch (sortKey) {
      case "custom":
        sorted.sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0));
        break;
      case "newest":
        sorted.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
        break;
      case "oldest":
        sorted.sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));
        break;
      case "name_asc":
        sorted.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case "name_desc":
        sorted.sort((a, b) => b.name.localeCompare(a.name));
        break;
    }
    return sorted;
  }, [filteredProjects, sortKey]);

  function applyProjectsSummary(summaries: ProjectSummary[]) {
    setProjects(summaries);
    const previews: Record<string, { thumbUrl: string | null; imageCount: number; maskCount: number }> = {};
    for (const s of summaries) {
      previews[s.id] = {
        thumbUrl: s.first_filename ? thumbnailUrl(s.id, s.first_filename) : null,
        imageCount: s.image_count,
        maskCount: s.mask_count,
      };
    }
    setProjectPreviews(previews);
  }

  async function handleCreate() {
    if (!projectName.trim()) return;
    try {
      const created: Project = await createProject({ name: projectName.trim(), description: projectDesc.trim() });
      setProjectName("");
      setProjectDesc("");
      showToast("Created.");
      const optimistic: ProjectSummary = {
        ...created,
        image_count: 0,
        mask_count: 0,
        first_filename: null,
      };
      setProjects((prev) => {
        if (prev.some((p) => p.id === created.id)) return prev;
        return [optimistic, ...prev];
      });
      setSelectedProjectId(created.id);
      fetchProjectsSummary().then(applyProjectsSummary).catch(() => {/* ignore */});
    } catch (err) {
      showToast(`Create failed: ${formatError(err, lang)}`);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this project?")) return;
    try {
      await deleteProject(id);
      const summaries = await fetchProjectsSummary();
      applyProjectsSummary(summaries);
      const first = summaries[0] ?? null;
      if (selectedProjectId === id) {
        setSelectedProjectId(first?.id ?? null);
      }
      showToast("Deleted.");
    } catch (err) {
      showToast(`Delete failed: ${formatError(err, lang)}`);
    }
  }

  async function handleImportZipFiles(e: React.ChangeEvent<HTMLInputElement>) {
    const fileInput = e.target;
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (!file) return;

    setImportBusy(true);
    const zipName = file.name.replace(/\.zip$/i, "") || "Imported";
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    setImportProgress(`Creating project "${zipName}"...`);
    try {
      const created = await createProject({ name: zipName });
      const newProjectId: string = created.id;

      setImportProgress(`${t("projects.importing")}... ${file.name} (${sizeMB}MB)`);
      const result = await importAnnotateZip(newProjectId, file);

      setImportProgress(t("projects.importFinalizing"));
      await new Promise((r) => setTimeout(r, 500));
      const summaries = await fetchProjectsSummary();
      applyProjectsSummary(summaries);
      setSelectedProjectId(newProjectId);
      setImportProgress(null);
      showToast(`Imported: ${result.image_count} images, ${result.mask_count} masks${result.classes_imported ? ", classes" : ""}`);
    } catch (err) {
      setImportProgress(null);
      showToast(`Import failed: ${formatError(err, lang)}`);
    } finally {
      setImportBusy(false);
    }
  }

  function handleImportProject() {
    importZipInputRef.current?.click();
  }

  function toggleChecked(id: string, e: React.MouseEvent | React.ChangeEvent) {
    e.stopPropagation();
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  async function handleBatchExport() {
    if (checkedIds.size === 0) return;
    setExportBusy(true);
    const ids = [...checkedIds];
    try {
      for (let i = 0; i < ids.length; i++) {
        const proj = projects.find((p) => p.id === ids[i]);
        const name = proj?.name || ids[i].slice(0, 8);
        showToast(t("projects.batchExport.progress").replace("{name}", name).replace("{i}", String(i + 1)).replace("{total}", String(ids.length)));
        const res = await fetch(exportDatasetUrl(ids[i]));
        if (!res.ok) { showToast(t("projects.batchExport.failedOne").replace("{name}", name).replace("{status}", String(res.status))); continue; }
        const blob = await res.blob();
        const cd = res.headers.get("content-disposition") || "";
        const match = cd.match(/filename="?([^";]+)"?/);
        const filename = match?.[1] || `${name}.zip`;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = filename; a.click();
        URL.revokeObjectURL(url);
      }
      showToast(t("projects.batchExport.done").replace("{n}", String(ids.length)));
      setCheckedIds(new Set());
    } catch (err) {
      showToast(t("projects.batchExport.failed").replace("{msg}", (err as Error).message));
    } finally {
      setExportBusy(false);
    }
  }

  return (
    <>
      <div>
        <div className="row projects-create-row">
          <input value={projectName} onChange={(e) => setProjectName(e.target.value)} placeholder={t("projects.projectName")} />
          <input value={projectDesc} onChange={(e) => setProjectDesc(e.target.value)} placeholder={t("projects.description")} />
          <button className="primary" onClick={handleCreate} data-desc={t("projects.create.desc")} data-desc-pos="bottom" data-tutorial-step="create-project-btn">{t("projects.create")}</button>
          <button className="ghost" onClick={handleImportProject} disabled={importBusy} data-desc={t("projects.import.desc")} data-desc-pos="bottom">
            {importBusy ? t("projects.importing") : t("projects.import")}
          </button>
          {checkedIds.size > 0 && (
            <>
              <button
                className="primary"
                onClick={handleBatchExport}
                disabled={exportBusy}
                style={{ fontSize: 13, padding: "6px 12px" }}
                data-desc={t("projects.batchExport.desc")}
              >
                {exportBusy ? t("projects.exporting") : t("projects.batchExport.button").replace("{n}", String(checkedIds.size))}
              </button>
              {bulkTagInput === null ? (
                <button
                  className="ghost"
                  onClick={() => setBulkTagInput("")}
                  style={{ fontSize: 13, padding: "6px 10px" }}
                  data-desc={t("projects.bulkAddTags.desc")}
                >
                  {t("projects.bulkAddTags").replace("{n}", String(checkedIds.size))}
                </button>
              ) : (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <input
                    autoFocus
                    value={bulkTagInput}
                    onChange={(e) => setBulkTagInput(e.target.value)}
                    placeholder={t("projects.bulkAddTags.placeholder")}
                    disabled={bulkTagBusy}
                    style={{ fontSize: 13, padding: "4px 8px", width: 180 }}
                    onKeyDown={async (e) => {
                      if (e.key === "Escape") { setBulkTagInput(null); return; }
                      if (e.key !== "Enter") return;
                      const raw = bulkTagInput ?? "";
                      const newTags = Array.from(new Set(raw.split(",").map((s) => s.trim()).filter(Boolean)));
                      if (newTags.length === 0) { setBulkTagInput(null); return; }
                      setBulkTagBusy(true);
                      try {
                        const ids = [...checkedIds];
                        for (const id of ids) {
                          const proj = projects.find((p) => p.id === id);
                          if (!proj) continue;
                          const merged = Array.from(new Set([...(proj.tags || []), ...newTags]));
                          await updateProject(id, { tags: merged });
                          setProjects((prev) => prev.map((p) => p.id === id ? { ...p, tags: merged } : p));
                        }
                        showToast(t("projects.bulkAddTags.done").replace("{n}", String(ids.length)));
                        setBulkTagInput(null);
                      } catch (err) {
                        showToast(formatError(err, lang));
                      } finally {
                        setBulkTagBusy(false);
                      }
                    }}
                  />
                  <button
                    className="ghost"
                    onClick={() => setBulkTagInput(null)}
                    disabled={bulkTagBusy}
                    style={{ fontSize: 13, padding: "4px 8px" }}
                  >
                    {t("projects.bulkAddTags.cancel")}
                  </button>
                </span>
              )}
              <button
                className="ghost"
                onClick={() => setCheckedIds(new Set())}
                style={{ fontSize: 13, padding: "6px 10px" }}
                data-desc={t("projects.clearSelection.desc")}
              >
                {t("projects.clearSelection")}
              </button>
            </>
          )}
          {apiStatus === "connected" && projects.length > 0 && (
            <select
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as SortKey)}
              className="ghost"
              style={{ marginLeft: "auto", fontSize: 13, padding: "6px 10px", cursor: "pointer" }}
              data-desc={t("projects.sort.desc")}
            >
              <option value="custom">{t("projects.sortCustom")}</option>
              <option value="newest">{t("projects.sortNewest")}</option>
              <option value="oldest">{t("projects.sortOldest")}</option>
              <option value="name_asc">{t("projects.sortNameAsc")}</option>
              <option value="name_desc">{t("projects.sortNameDesc")}</option>
            </select>
          )}
        </div>
        {apiStatus === "connected" && allTags.length > 0 && (
          <div className="project-tag-filter">
            <span className="project-tag-filter-label">{t("projects.tags.filter")}:</span>
            <button
              type="button"
              className={tagFilter.size === 0 ? "tag-filter-chip active" : "tag-filter-chip"}
              onClick={() => setTagFilter(new Map())}
            >
              {t("projects.tags.all")}
            </button>
            {allTags.map((tag) => {
              const mode = tagFilter.get(tag);
              const cls =
                mode === "include" ? "tag-filter-chip include"
                : mode === "exclude" ? "tag-filter-chip exclude"
                : "tag-filter-chip";
              return (
                <button
                  key={tag}
                  type="button"
                  className={cls}
                  title={t("projects.tags.cycleHint")}
                  aria-pressed={mode === "include"}
                  onClick={() => {
                    setTagFilter((prev) => {
                      const next = new Map(prev);
                      const cur = next.get(tag);
                      if (cur === undefined) next.set(tag, "include");
                      else if (cur === "include") next.set(tag, "exclude");
                      else next.delete(tag);
                      return next;
                    });
                  }}
                >
                  {mode === "exclude" ? "− " : ""}{tag}
                </button>
              );
            })}
          </div>
        )}
        <div className="projects-helper">
          {t("projects.helper")}
        </div>
        {apiStatus !== "connected" && (
          <div className="api-connecting-banner">
            <div className="api-connecting-spinner" />
            <div>
              <div style={{ fontWeight: 600 }}>{t("projects.connecting")}</div>
              <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                {t("projects.connectingDesc")}
              </div>
            </div>
          </div>
        )}
        <input
          ref={importZipInputRef}
          type="file"
          accept=".zip"
          style={{ display: "none" }}
          onChange={handleImportZipFiles}
        />
        {apiStatus === "connected" && currentProject && (
          <div className="projects-current-card">
            <div>
              <div className="projects-current-label">{t("projects.selectedProject")}</div>
              <div className="projects-current-title">{currentProject.name}</div>
              <div className="projects-current-meta">
                {!projectsSummaryReady || projectsLoading
                  ? t("projects.syncingImages")
                  : currentProjectPreview
                    ? `${currentProjectPreview.imageCount} ${t("projects.imageCount")} | ${currentProjectPreview.maskCount} ${t("projects.maskCount")} | ${currentProject.id.slice(0, 8)}`
                    : currentProject.id.slice(0, 8)}
              </div>
              <CurrentProjectTrainingSummary projectId={currentProject.id} />
              {currentProject.description && (
                <div className="muted" style={{ marginTop: 4 }}>{currentProject.description}</div>
              )}
            </div>
            <div className="projects-current-actions">
              <button
                className="ghost"
                onClick={() => openProjectWorkspace(currentProject.id, "annotate")}
                data-desc={t("projects.openAnnotate.desc")}
              >
                {t("projects.openAnnotate")}
              </button>
              <button
                className="ghost"
                onClick={() => openProjectWorkspace(currentProject.id, "training")}
                data-desc={t("projects.openTraining.desc")}
              >
                {t("projects.openTraining")}
              </button>
            </div>
          </div>
        )}
        {apiStatus === "connected" && projectsLoading && projects.length === 0 && (
          <div className="projects-loading-card" aria-live="polite">
            <div className="api-connecting-spinner" />
            <div>
              <div className="projects-empty-title">{t("projects.loadingProjects")}</div>
              <div className="muted">{t("projects.loadingProjectsDesc")}</div>
            </div>
          </div>
        )}
        {apiStatus === "connected" && !projectsLoading && projects.length === 0 && (
          <div className="projects-empty-state">
            <div className="projects-empty-title">{t("projects.noProjectsYet")}</div>
            <div className="muted">
              {t("projects.emptyDesc")}
            </div>
          </div>
        )}
        {apiStatus === "connected" && <div className="project-grid">
          {sortedProjects.map((project) => (
            <ProjectTile
              key={project.id}
              project={project}
              preview={projectPreviews[project.id]}
              sortedProjects={sortedProjects}
              sortKey={sortKey}
              selectedProjectId={selectedProjectId}
              setSelectedProjectId={setSelectedProjectId}
              checkedIds={checkedIds}
              setCheckedIds={setCheckedIds}
              lastCheckedId={lastCheckedId}
              setLastCheckedId={setLastCheckedId}
              dragId={dragId}
              setDragId={setDragId}
              dragOverId={dragOverId}
              setDragOverId={setDragOverId}
              editingNameId={editingNameId}
              setEditingNameId={setEditingNameId}
              editingMemoId={editingMemoId}
              setEditingMemoId={setEditingMemoId}
              editingTagsId={editingTagsId}
              setEditingTagsId={setEditingTagsId}
              setProjects={setProjects}
              projectsSummaryReady={projectsSummaryReady}
              projectsLoading={projectsLoading}
              exportBusy={exportBusy}
              setExportModalProject={setExportModalProject}
              toggleChecked={toggleChecked}
              handleDelete={handleDelete}
              openProjectWorkspace={openProjectWorkspace}
              showToast={showToast}
            />
          ))}
        </div>}
      </div>
      {importProgress && (
        <div className="import-overlay">
          <div className="import-dialog">
            <div className="import-dialog-spinner" />
            <div className="import-dialog-text">{importProgress}</div>
          </div>
        </div>
      )}
      {exportBusy && (
        <div className="import-overlay">
          <div className="import-dialog">
            <div className="import-dialog-spinner" />
            <div className="import-dialog-text">{t("projects.exporting")}</div>
          </div>
        </div>
      )}
      {exportModalProject && (
        <ExportModal
          project={exportModalProject}
          onClose={() => setExportModalProject(null)}
          exportBusy={exportBusy}
          setExportBusy={setExportBusy}
          applyProjectsSummary={applyProjectsSummary}
          showToast={showToast}
        />
      )}
    </>
  );
};

export default React.memo(ProjectsPanel);
