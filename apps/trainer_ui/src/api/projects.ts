// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

export type Project = {
  id: string;
  name: string;
  description?: string | null;
  memo?: string | null;
  sort_order?: number;
  tags?: string[];
  created_at: string;
  updated_at: string;
};

export type AssistantMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  kind: string;
  content: string;
  created_at: string;
};

export type ProjectSummary = Project & {
  image_count: number;
  mask_count: number;
  first_filename: string | null;
};

export async function fetchProjects(): Promise<Project[]> {
  const res = await fetch(`${API_BASE}/projects`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchProjectsSummary(): Promise<ProjectSummary[]> {
  const res = await fetch(`${API_BASE}/projects/summary`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function createProject(payload: { name: string; description?: string; tags?: string[] }) {
  const res = await fetch(`${API_BASE}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function updateProject(id: string, payload: { name?: string; description?: string; memo?: string; tags?: string[] }): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function deleteProject(id: string) {
  const res = await fetch(`${API_BASE}/projects/${id}`, { method: "DELETE" });
  if (!res.ok) throw await parseApiError(res);
}

export async function reorderProjects(order: string[]): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/reorder`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order }),
  });
  if (!res.ok) throw await parseApiError(res);
}

export async function fetchAssistantContext(projectId: string): Promise<{ project_id: string; markdown: string }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/assistant/context`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function saveAssistantContext(projectId: string, markdown: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/assistant/context`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ markdown })
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchAssistantThread(projectId: string, limit: number = 200): Promise<{ project_id: string; messages: AssistantMessage[] }> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/assistant/thread?limit=${encodeURIComponent(String(limit))}`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function postAssistantMessage(projectId: string, content: string, role: "user" | "assistant" | "system" = "user") {
  const res = await fetch(`${API_BASE}/projects/${projectId}/assistant/thread/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content, kind: "message" })
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function runAssistantCommand(projectId: string, command: string) {
  const res = await fetch(`${API_BASE}/projects/${projectId}/assistant/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command })
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}
