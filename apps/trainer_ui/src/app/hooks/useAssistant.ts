// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useState } from "react";
import {
  fetchAssistantContext,
  fetchAssistantThread,
  postAssistantMessage,
  runAssistantCommand,
  saveAssistantContext,
  type AssistantMessage,
} from "../../api";

/**
 * Manages assistant chat state per project.
 */
export function useAssistant(selectedProjectId: string | null) {
  const [assistantContext, setAssistantContext] = useState("");
  const [assistantThread, setAssistantThread] = useState<AssistantMessage[]>([]);
  const [assistantDraftByProject, setAssistantDraftByProject] = useState<Record<string, string>>({});
  const [assistantBusy, setAssistantBusy] = useState(false);

  const assistantDraft = selectedProjectId ? (assistantDraftByProject[selectedProjectId] ?? "") : "";

  function setAssistantDraft(nextValue: string) {
    if (!selectedProjectId) return;
    setAssistantDraftByProject((prev) => ({ ...prev, [selectedProjectId]: nextValue }));
  }

  async function loadAssistant(projectId: string) {
    try {
      const [contextRes, threadRes] = await Promise.all([
        fetchAssistantContext(projectId),
        fetchAssistantThread(projectId, 200),
      ]);
      setAssistantContext(contextRes.markdown ?? "");
      setAssistantThread(threadRes.messages ?? []);
    } catch {
      setAssistantContext("");
      setAssistantThread([]);
    }
  }

  async function handleSaveAssistantContext() {
    if (!selectedProjectId || assistantBusy) return;
    setAssistantBusy(true);
    try {
      await saveAssistantContext(selectedProjectId, assistantContext);
    } finally {
      setAssistantBusy(false);
    }
  }

  async function handleAssistantSubmit() {
    if (!selectedProjectId || assistantBusy) return;
    const raw = assistantDraft.trim();
    if (!raw) return;
    setAssistantBusy(true);
    try {
      if (raw.startsWith("/")) {
        const res = await runAssistantCommand(selectedProjectId, raw);
        setAssistantThread((prev) => [
          ...(prev ?? []),
          res.user_message as AssistantMessage,
          res.assistant_message as AssistantMessage,
        ]);
      } else {
        const sent = await postAssistantMessage(selectedProjectId, raw, "user");
        setAssistantThread((prev) => [
          ...(prev ?? []),
          sent.message as AssistantMessage,
        ]);
      }
      setAssistantDraft("");
    } finally {
      setAssistantBusy(false);
    }
  }

  useEffect(() => {
    if (!selectedProjectId) {
      setAssistantContext("");
      setAssistantThread([]);
      return;
    }
    void loadAssistant(selectedProjectId);
  }, [selectedProjectId]);

  return {
    assistantContext, setAssistantContext,
    assistantThread, assistantDraft, setAssistantDraft,
    assistantBusy, handleSaveAssistantContext, handleAssistantSubmit,
  } as const;
}
