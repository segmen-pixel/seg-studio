# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What an agent did to a project, kept long enough for a screen to notice.

The MCP bridge writes masks through the same routes the browser uses, so the
UI had no way to tell that a project was being labelled by something other
than the person looking at it -- the red dots just appeared on the next
project switch. The bridge now names itself in a header; the annotation
routes drop a record here when they see it; the UI polls the feed and shows
who is working and on what, and refreshes the images that changed.

In-memory and per process on purpose: the audience is a browser tab open on
the same server, and an event older than a minute is of no use to it. There
is no actor field on the annotation index to extend, and a database row per
mask write would outlive its usefulness by months.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

#: Header the bridge sends on every request, e.g. ``mcp/write``.
AGENT_HEADER = "X-Seg-Agent"
#: Header naming the tool behind the request, e.g. ``mask_put``.
TOOL_HEADER = "X-Seg-Agent-Tool"
#: How long after the last event the agent still counts as active.
#:
#: Eight seconds fitted a Python driver writing a mask every five. A vision
#: model driving the same tools thinks for far longer than that between
#: calls, and the chip -- with the Follow toggle on it -- blinked out for
#: most of every image. Ninety seconds covers the thinking and still goes
#: quiet a minute and a half after the last write.
ACTIVE_WINDOW_S = 90.0
_MAX_EVENTS = 500


@dataclass(frozen=True)
class AgentEvent:
    seq: int
    at: float
    agent: str
    tool: str
    action: str
    project_id: str
    item_id: str | None = None
    count: int | None = None


class AgentActivity:
    def __init__(self, max_events: int = _MAX_EVENTS) -> None:
        self._events: deque[AgentEvent] = deque(maxlen=max_events)
        self._seq = 0
        self._lock = threading.Lock()

    def record(self, *, agent: str, tool: str, action: str, project_id: str,
               item_id: str | None = None, count: int | None = None) -> AgentEvent:
        with self._lock:
            self._seq += 1
            ev = AgentEvent(seq=self._seq, at=time.time(), agent=agent, tool=tool,
                            action=action, project_id=project_id, item_id=item_id,
                            count=count)
            self._events.append(ev)
            return ev

    def since(self, seq: int, project_id: str | None = None, limit: int = 200) -> list[AgentEvent]:
        with self._lock:
            out = [e for e in self._events
                   if e.seq > seq and (project_id is None or e.project_id == project_id)]
        return out[-limit:]

    def snapshot(self, project_id: str | None = None, now: float | None = None) -> dict:
        """The feed's head: whether something is working right now, and on what."""
        now = time.time() if now is None else now
        with self._lock:
            last = None
            for e in reversed(self._events):
                if project_id is None or e.project_id == project_id:
                    last = e
                    break
            seq = self._seq
        active = last is not None and (now - last.at) <= ACTIVE_WINDOW_S
        return {"seq": seq, "active": active,
                "last": asdict(last) if last else None}

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._seq = 0


ACTIVITY = AgentActivity()


def agent_of(headers) -> tuple[str, str] | None:
    """(agent, tool) from a request's headers, or None for a browser."""
    agent = (headers.get(AGENT_HEADER) or "").strip()
    if not agent:
        return None
    tool = (headers.get(TOOL_HEADER) or "").strip() or "?"
    return agent[:64], tool[:64]


def note(headers, *, action: str, project_id: str, item_id: str | None = None,
         count: int | None = None) -> None:
    """Record the request if an agent made it. A browser leaves no trace."""
    who = agent_of(headers)
    if who is None:
        return
    agent, tool = who
    ACTIVITY.record(agent=agent, tool=tool, action=action, project_id=project_id,
                    item_id=item_id, count=count)
