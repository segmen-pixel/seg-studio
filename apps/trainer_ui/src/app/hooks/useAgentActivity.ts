// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useRef, useState } from "react";
import { fetchAgentActivity, type AgentEvent } from "../../api/agent";
import { visibleInterval } from "./useVisibleInterval";

// Once a second, for Follow and the chip: Follow moves the screen to the image
// a run has started on and shows each mask as it is written, and the chip
// names the tool at work. Polled every two seconds, both could trail the run
// by that much. The endpoint answers from memory and the poll stops while the
// tab is hidden, so the faster rate costs next to nothing. The timings in
// e2e/specs/agent-live-follow.spec.ts were set against this rate.
const POLL_MS = 1_000;

export type AgentActivityState = {
  /** an agent wrote something within the server's active window */
  active: boolean;
  /** the newest event, kept a little after `active` drops so the chip can fade */
  last: AgentEvent | null;
  /** feed position; bumps whenever new events arrive */
  seq: number;
  /** events that arrived on the latest poll (not cumulative) */
  events: AgentEvent[];
};

/** Polls the agent activity feed and hands back what arrived since last time.

    The MCP bridge labels through the same routes the browser uses, so
    without this the only sign of it was red dots appearing on the next
    project switch. How often, and why, is at POLL_MS above. */
export function useAgentActivity(projectId: string | null): AgentActivityState {
  const [state, setState] = useState<AgentActivityState>({ active: false, last: null, seq: 0, events: [] });
  const sinceRef = useRef(0);
  const projectRef = useRef(projectId);
  projectRef.current = projectId;

  useEffect(() => {
    // a fresh feed position per project so the first poll shows what is going on now
    sinceRef.current = 0;
    setState((s) => ({ ...s, events: [] }));
    const poll = async () => {
      try {
        const d = await fetchAgentActivity(sinceRef.current, projectRef.current);
        const fresh = d.events ?? [];
        if (fresh.length) sinceRef.current = fresh[fresh.length - 1].seq;
        else if (d.seq > sinceRef.current) sinceRef.current = d.seq;
        setState((s) => {
          const changed = fresh.length > 0 || s.active !== d.active
            || (d.last?.seq ?? -1) !== (s.last?.seq ?? -1);
          if (!changed) return s;
          return { active: d.active, last: d.last, seq: d.seq, events: fresh };
        });
      } catch { /* the feed is decoration; a miss is not an error */ }
    };
    poll();
    return visibleInterval(poll, POLL_MS);
  }, [projectId]);

  return state;
}
