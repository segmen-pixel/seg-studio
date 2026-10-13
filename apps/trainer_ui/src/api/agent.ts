// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { API_BASE, parseApiError } from "./shared";

/** One thing an agent did to a project, as the server recorded it. */
export type AgentEvent = {
  seq: number;
  at: number;
  agent: string;
  tool: string;
  action: string;
  project_id: string;
  item_id: string | null;
  count: number | null;
};

export type AgentActivity = {
  seq: number;
  active: boolean;
  active_window_s: number;
  last: AgentEvent | null;
  events: AgentEvent[];
};

export async function fetchAgentActivity(since: number, projectId?: string | null): Promise<AgentActivity> {
  const q = new URLSearchParams({ since: String(since) });
  if (projectId) q.set("project_id", projectId);
  const res = await fetch(`${API_BASE}/agent/activity?${q.toString()}`);
  if (!res.ok) throw new Error(`agent activity: HTTP ${res.status}`);
  return res.json();
}

/** A labelling run the server has going, for a screen that is looking elsewhere. */
export type AgentRun = {
  project_id: string;
  project_name: string | null;
  paused: boolean;
  waiting_for_reply: boolean;
  turns: number;
  item_id: string | null;
};

/** Which projects are being labelled right now. One request, whatever the count. */
export async function fetchAgentRuns(): Promise<{ items: AgentRun[] }> {
  const res = await fetch(`${API_BASE}/agent/runs`);
  if (!res.ok) throw new Error(`agent runs: HTTP ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Labelling with a vision model, from the Assist tab. The loop runs on the
// server -- a browser cannot spawn the bridge -- and reports back one JSON
// object per line, the same shape the example chat page used to render.
// ---------------------------------------------------------------------------

export type AgentRunEvent = {
  type: "tool" | "image" | "question" | "final" | "stopped" | "error"
      | "paused" | "resumed" | "context" | "trimmed" | "started"
      // a tool that refused, and the dry run against a teacher image. Both used
      // to exist only inside a "tool" event, which the log hides: a run could
      // fail the same call again and again and look, from here, like one
      // thinking quietly.
      | "failed" | "rehearsal"
      // one of the steps before labelling, answered: what it understood, which
      // is the thing worth reading before the rest of the images are labelled on it
      | "step"
      // it was told to stop asking, and is carrying on by itself
      | "auto"
      // the two that are somebody speaking: the model on its way to a tool
      // call, and the person -- the instruction that started the run, and any
      // answer to a question it asked
      | "say" | "you";
  /** tool events */
  step?: number;
  name?: string;
  args?: Record<string, unknown>;
  result?: string;
  think_s?: number;
  /** image events: what the model was shown */
  item_id?: string;
  caption?: string;
  jpeg_b64?: string;
  image?: string;
  /** say events: which images the model's tool calls in that turn were about */
  about?: string[];
  /** failed events: how many times this same call has now failed */
  n?: number;
  /** rehearsal events: whether the dry run came out clean */
  ok?: boolean;
  /** step events: which of how many, and its name */
  step_no?: number;
  of?: number;
  /** everything else */
  text?: string;
  choices?: string[];
};

export type AgentConnection = {
  configured: boolean;
  backend: string;
  base_url: string;
  model: string;
  /** where to change it: this is configuration, not a field in a request */
  where: string;
};

export type AgentRunState = {
  running: boolean;
  paused: boolean;
  waiting_for_reply: boolean;
  question: string | null;
  turns: number;
  connection: AgentConnection;
};

export type AgentModels = {
  models: string[];
  selected: string;
  /** why the list is empty, when it is */
  why: string;
  backend: string;
  base_url: string;
};

export async function fetchAgentModels(projectId: string): Promise<AgentModels> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/models`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchAgentRunState(projectId: string): Promise<AgentRunState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/state`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function fetchAgentThread(projectId: string): Promise<{
  entries: AgentRunEvent[]; state: AgentRunState;
}> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/thread`);
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

/**
 * Empty the conversation the panel shows, keeping the record of it.
 *
 * The append-only log the server writes as a run goes is not touched: a clean
 * screen is not the same wish as losing what was said, and the one thing asked
 * for before this was that the conversation stop being losable.
 */
export async function clearAgentThread(projectId: string): Promise<AgentRunState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/clear`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function stopAgentRun(projectId: string): Promise<AgentRunState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/stop`, { method: "POST" });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function pauseAgentRun(projectId: string, paused: boolean): Promise<AgentRunState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/pause`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused }),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

export async function replyToAgentRun(projectId: string, text: string): Promise<AgentRunState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/reply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw await parseApiError(res);
  return res.json();
}

/**
 * Start a run and hand each event over as it arrives.
 *
 * There is deliberately no model or address here: where the prompts go is
 * configuration on the server, because a URL taken from the browser would
 * make the server fetch whatever the caller named.
 */
export async function startAgentRun(
  projectId: string,
  instruction: string,
  opts: { confirm?: boolean; lang?: string; model?: string; debugSteps?: boolean } = {},
  onEvent: (event: AgentRunEvent) => void = () => {},
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/agent/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      instruction,
      confirm: opts.confirm ?? false,
      lang: opts.lang ?? "ja",
      // a name, not an address: see core/vlm_agent/settings.py
      model: opts.model ?? "",
      debug_steps: opts.debugSteps ?? false,
    }),
    signal,
  });
  if (!res.ok) throw await parseApiError(res);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed));
      } catch {
        // a half-written line; the next chunk completes it
      }
    }
  }
  if (buffer.trim()) {
    try {
      onEvent(JSON.parse(buffer.trim()));
    } catch {
      // ignore a trailing fragment
    }
  }
}
