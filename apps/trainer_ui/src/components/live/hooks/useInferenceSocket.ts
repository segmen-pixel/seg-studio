// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useRef, useState } from "react";
import { API_BASE, API_ORIGIN } from "../../../api";
import type { InferenceResult } from "../types";

export function useInferenceSocket(
  addResult: (result: InferenceResult) => void,
  session: { session_id: string } | null,
  toast: (msg: string) => void,
) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileInferring, setFileInferring] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // Lightbox preview
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);

  // File inference (single or multiple) — shared by file input and drag-and-drop
  const inferFiles = useCallback(async (files: FileList | File[]) => {
    if (!files || files.length === 0) return;
    setFileInferring(true);
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file.type.startsWith("image/")) continue;
      const formData = new FormData();
      formData.append("file", file);
      formData.append("frame_id", file.name);
      try {
        const imageUrl = URL.createObjectURL(file);
        const res = await fetch(`${API_ORIGIN}/v2/infer`, { method: "POST", body: formData });
        if (!res.ok) { URL.revokeObjectURL(imageUrl); throw new Error(await res.text()); }
        const result: InferenceResult = await res.json();
        result.imageUrl = imageUrl;
        if (result.mask_png_b64) {
          const bin = atob(result.mask_png_b64);
          const buf = new Uint8Array(bin.length);
          for (let j = 0; j < bin.length; j++) buf[j] = bin.charCodeAt(j);
          result.maskUrl = URL.createObjectURL(new Blob([buf], { type: "image/png" }));
          delete result.mask_png_b64;
        }
        addResult(result);
      } catch (err) {
        console.error("Inference error:", err);
        toast(`Inference error: ${(err as Error).message}`);
      }
    }
    setFileInferring(false);
  }, [addResult, toast]);

  const handleFileInfer = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) inferFiles(Array.from(e.target.files));
    e.target.value = "";
  }, [inferFiles]);

  // Drag-and-drop support
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (session && !fileInferring) setDragOver(true);
  }, [session, fileInferring]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (!session || fileInferring) return;
    const files = e.dataTransfer.files;
    if (files.length > 0) inferFiles(files);
  }, [session, fileInferring, inferFiles]);

  return {
    fileInputRef,
    fileInferring,
    dragOver,
    lightboxIdx,
    setLightboxIdx,
    inferFiles,
    handleFileInfer,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  };
}
