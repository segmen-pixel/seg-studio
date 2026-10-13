// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { useCallback, useRef, useState } from "react";
import { useI18n } from "../../i18n";

export type RenameImportDialogProps = {
  open: boolean;
  onClose: () => void;
  onImport: (files: File[]) => void;
};

export function RenameImportDialog({ open, onClose, onImport }: RenameImportDialogProps) {
  const { t } = useI18n();
  const [prefix, setPrefix] = useState("");
  const [suffix, setSuffix] = useState("");
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const imageFilter = useCallback((f: File) =>
    f.type.startsWith("image/") || /\.(png|jpe?g|bmp|tiff?|webp|gif|zip|mp4|avi|mov|mkv|wmv|flv|webm|m4v|mpg|mpeg)$/i.test(f.name)
  , []);

  const renameFile = useCallback((file: File): File => {
    const dot = file.name.lastIndexOf(".");
    const base = dot > 0 ? file.name.slice(0, dot) : file.name;
    const ext = dot > 0 ? file.name.slice(dot) : "";
    const newName = `${prefix}${base}${suffix}${ext}`;
    return new File([file], newName, { type: file.type });
  }, [prefix, suffix]);

  const addFiles = useCallback((fileList: FileList | File[], folderName?: string) => {
    const arr = Array.from(fileList).filter(imageFilter);
    if (arr.length === 0) return;
    if (folderName) {
      // Embed folder name into each filename directly
      const prefixed = arr.map(f => new File([f], `${folderName}_${f.name}`, { type: f.type }));
      setPendingFiles((prev) => [...prev, ...prefixed]);
    } else {
      setPendingFiles((prev) => [...prev, ...arr]);
    }
  }, [imageFilter]);

  // Recursively read files from a dropped directory entry.
  // pathPrefix accumulates subfolder names so nested files get unique names,
  // e.g. NG/crack/001.png → "crack_001.png" (top-level folder is used as prefix separately).
  const readDirectoryEntries = useCallback(async (entry: FileSystemDirectoryEntry, pathPrefix = ""): Promise<File[]> => {
    const reader = entry.createReader();
    const files: File[] = [];
    const readBatch = (): Promise<FileSystemEntry[]> =>
      new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    let batch: FileSystemEntry[];
    do {
      batch = await readBatch();
      for (const child of batch) {
        if (child.isFile) {
          const file = await new Promise<File>((resolve, reject) =>
            (child as FileSystemFileEntry).file(resolve, reject)
          );
          if (!imageFilter(file)) continue;
          // Prepend subfolder path to filename to avoid collisions
          if (pathPrefix) {
            const renamed = new File([file], `${pathPrefix}${file.name}`, { type: file.type });
            files.push(renamed);
          } else {
            files.push(file);
          }
        } else if (child.isDirectory) {
          const subPrefix = `${pathPrefix}${child.name}_`;
          const subFiles = await readDirectoryEntries(child as FileSystemDirectoryEntry, subPrefix);
          files.push(...subFiles);
        }
      }
    } while (batch.length > 0);
    return files;
  }, [imageFilter]);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);

    // Check for folder drop via DataTransferItem entries
    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }
      // If any entry is a directory, handle as folder drop
      const dirEntries = entries.filter((e) => e.isDirectory) as FileSystemDirectoryEntry[];
      if (dirEntries.length > 0) {
        const allFiles: File[] = [];
        for (const dir of dirEntries) {
          const dirFiles = await readDirectoryEntries(dir);
          if (dirEntries.length > 1) {
            // Multiple folders: prepend each folder name to avoid collisions
            for (const f of dirFiles) {
              allFiles.push(new File([f], `${dir.name}_${f.name}`, { type: f.type }));
            }
          } else {
            allFiles.push(...dirFiles);
          }
        }
        // Also include any loose files
        const fileEntries = entries.filter((e) => e.isFile) as FileSystemFileEntry[];
        for (const fe of fileEntries) {
          const file = await new Promise<File>((resolve, reject) => fe.file(resolve, reject));
          if (imageFilter(file)) allFiles.push(file);
        }
        // Auto-set prefix from folder name only when single folder
        const folderName = dirEntries.length === 1 ? dirEntries[0].name : undefined;
        addFiles(allFiles, folderName);
        return;
      }
    }

    // Fallback: plain file drop
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
  }, [addFiles, readDirectoryEntries, imageFilter]);

  const handleSubmit = useCallback(() => {
    if (pendingFiles.length === 0) return;
    const renamed = pendingFiles.map(renameFile);
    onImport(renamed);
    setPendingFiles([]);
    setPrefix("");
    setSuffix("");
    onClose();
  }, [pendingFiles, renameFile, onImport, onClose]);

  const handleClose = useCallback(() => {
    setPendingFiles([]);
    setPrefix("");
    setSuffix("");
    onClose();
  }, [onClose]);

  const removeFile = useCallback((idx: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  if (!open) return null;

  const previewName = pendingFiles.length > 0
    ? renameFile(pendingFiles[0]!).name
    : `${prefix}example${suffix}.png`;

  return (
    <div className="modal-overlay" onClick={handleClose}
      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); }}
    >
      <div className="modal-content rename-import-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>{t("imageList.renameImport")}</h3>

        <div className="rename-import-fields">
          <label>
            <span>{t("imageList.renamePrefix")}</span>
            <input type="text" value={prefix} onChange={(e) => setPrefix(e.target.value)} placeholder="prefix_" />
          </label>
          <label>
            <span>{t("imageList.renameSuffix")}</span>
            <input type="text" value={suffix} onChange={(e) => setSuffix(e.target.value)} placeholder="_suffix" />
          </label>
        </div>

        <div className="rename-import-preview">
          {t("imageList.renamePreview")}: <code>{previewName}</code>
        </div>

        <div
          className={`rename-import-dropzone ${dragOver ? "drag-active" : ""}`}
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setDragOver(true); }}
          onDragEnter={(e) => { e.stopPropagation(); setDragOver(true); }}
          onDragLeave={(e) => { e.stopPropagation(); setDragOver(false); }}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.zip,video/*,.mp4,.avi,.mov,.mkv,.webm"
            multiple
            style={{ display: "none" }}
            onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }}
          />
          <input
            ref={folderInputRef}
            type="file"
            // @ts-expect-error webkitdirectory is not in React's type defs
            webkitdirectory=""
            style={{ display: "none" }}
            onChange={(e) => {
              if (!e.target.files || e.target.files.length === 0) return;
              // Extract folder name + rename files with subfolder path
              const first = e.target.files[0];
              const relPath = first.webkitRelativePath || "";
              const folderName = relPath.split("/")[0] || undefined;
              // Rename files: NG/crack/001.png → crack_001.png (strip top-level folder, kept as prefix)
              const renamed: File[] = [];
              for (const f of Array.from(e.target.files)) {
                if (!imageFilter(f)) continue;
                const parts = (f.webkitRelativePath || f.name).split("/");
                // parts: ["NG", "crack", "001.png"] → skip [0] (top folder = prefix), join rest with _
                const subPath = parts.length > 2 ? parts.slice(1, -1).join("_") + "_" : "";
                const name = parts[parts.length - 1];
                renamed.push(subPath ? new File([f], `${subPath}${name}`, { type: f.type }) : f);
              }
              addFiles(renamed, folderName);
              e.target.value = "";
            }}
          />
          {pendingFiles.length === 0
            ? t("imageList.renameDropHint")
            : `${pendingFiles.length} ${t("imageList.renameFilesReady")}`
          }
        </div>
        <div className="rename-import-folder-btn-row">
          <button className="ghost compact" onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click(); }}>
            {t("imageList.selectFolder")}
          </button>
        </div>

        {pendingFiles.length > 0 && (
          <div className="rename-import-file-list" style={{ maxHeight: 240, overflowY: "auto" }}>
            {pendingFiles.map((f, i) => (
              <div key={i} className="rename-import-file-row">
                <span className="rename-import-file-original">{f.name}</span>
                <span className="rename-import-file-arrow">→</span>
                <span className="rename-import-file-renamed">{renameFile(f).name}</span>
                <button className="rename-import-file-remove" onClick={() => removeFile(i)}>×</button>
              </div>
            ))}
          </div>
        )}

        <div className="rename-import-actions">
          <button onClick={handleClose}>{t("imageList.renameCancel")}</button>
          <button className="primary" onClick={handleSubmit} disabled={pendingFiles.length === 0}>
            {t("imageList.renameImportBtn")} ({pendingFiles.length})
          </button>
        </div>
      </div>
    </div>
  );
}
