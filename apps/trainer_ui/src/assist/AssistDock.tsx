// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useRef, useState } from "react";

import { useI18n } from "../i18n";
import { useAssistStore } from "../store";
import Assist from "./Assist";

type Props = {
  projectId: string | null;
  /** the labelling screen is the one it belongs to */
  active: boolean;
  lang: string;
};

type Size = { w: number; h: number };
/** Distance from the bottom-right corner, which is what the panel is anchored to. */
type Spot = { right: number; bottom: number };

const MIN: Size = { w: 320, h: 260 };
// Big enough to read a run in without dragging it out first: the conversation
// is what the panel is for.
const DEFAULT: Size = { w: 520, h: 720 };
/** Where it has always sat: clear of the .fab-stack column and the toast row. */
const HOME: Spot = { right: 84, bottom: 62 };

interface Win { w: number; h: number }

function windowSize(): Win {
  return { w: window.innerWidth, h: window.innerHeight };
}

/** Far enough on screen to grab again. A panel dragged off an edge, or left
 *  behind by a window that got smaller, is a panel you cannot get back.
 *
 *  A window with no size -- a tab opened in the background, a pane not shown
 *  yet -- says nothing about where the panel fits: the spot is left as it is.
 *  Measured against one, the panel was pushed 120px past the right edge and
 *  80px below the bottom, and stayed there once the window had a size. */
function onScreen(spot: Spot, size: Size, win: Win = windowSize()): Spot {
  if (win.w <= 0 || win.h <= 0) return spot;
  const right = Math.max(8 - size.w + 120, Math.min(win.w - 120, spot.right));
  const bottom = Math.max(8 - size.h + 80, Math.min(win.h - 80, spot.bottom));
  return { right, bottom };
}

function read(projectId: string): Size {
  try {
    const raw = localStorage.getItem(`seg.assist.size.${projectId}`);
    const got = raw ? (JSON.parse(raw) as Partial<Size>) : null;
    if (got && typeof got.w === "number" && typeof got.h === "number") {
      return { w: got.w, h: got.h };
    }
  } catch {
    // a stale or hand-edited value is not worth a broken panel
  }
  return DEFAULT;
}

function readSpot(projectId: string): Spot {
  try {
    const raw = localStorage.getItem(`seg.assist.spot.${projectId}`);
    const got = raw ? (JSON.parse(raw) as Partial<Spot>) : null;
    if (got && typeof got.right === "number" && typeof got.bottom === "number") {
      return { right: got.right, bottom: got.bottom };
    }
  } catch {
    // as above
  }
  return HOME;
}

/**
 * The vision-model conversation, in the corner of the labelling screen.
 *
 * It was a tab of its own and that pushed the tab strip onto a second row --
 * and put the conversation a tab away from the masks it is writing, which is
 * the wrong place for it. Each project remembers whether the panel is open
 * and how big it was, so a project labelled through the conversation can keep
 * a tall panel and one that never uses it here never grows one.
 *
 * The panel is fixed rather than in the flow: the annotator measures its own
 * space and fills it, so a panel appearing beside it would move the canvas.
 *
 * Bottom-right, clear of .fab-stack's own 52px column. What the toasts used to
 * cover was the pill that opened this, sitting at bottom:18 among them; that is
 * gone -- the button is in the tool column now -- and the panel itself starts at
 * bottom:62, above the row a message occupies.
 */
export default function AssistDock({ projectId, active, lang }: Props) {
  const { t } = useI18n();
  // Held in the store, not here: the button that opens this sits at the foot
  // of the tool column, which is several components away and knows nothing
  // about this file.
  const open = useAssistStore((s) => s.open);
  const setOpen = useAssistStore((s) => s.setOpen);
  const setAvailable = useAssistStore((s) => s.setAvailable);
  const [size, setSize] = useState<Size>(DEFAULT);
  // Where the person put it. Where it is drawn is this, held on screen for
  // the window as it is now (shown, below): a window that shrinks moves the
  // panel in, and one that grows again lets it back.
  const [spot, setSpot] = useState<Spot>(HOME);
  const [win, setWin] = useState<Win>(windowSize);
  const drag = useRef<{ x: number; y: number; w: number; h: number } | null>(null);
  const move = useRef<{ x: number; y: number; right: number; bottom: number } | null>(null);

  // Rendered at all means there is an assistant to open.
  useEffect(() => {
    setAvailable(true);
    return () => setAvailable(false);
  }, [setAvailable]);

  useEffect(() => {
    if (!projectId) { setOpen(false); return; }
    setOpen(localStorage.getItem(`seg.assist.open.${projectId}`) === "1");
    setSize(read(projectId));
    setSpot(readSpot(projectId));
    setWin(windowSize());
  }, [projectId, setOpen]);

  // A window that gets smaller must not take the panel with it.
  useEffect(() => {
    const measure = () => setWin(windowSize());
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);
  const shown = onScreen(spot, size, win);

  // Remembered per project, as before, but written where the state settles
  // rather than inside the one handler that used to change it.
  useEffect(() => {
    if (projectId) localStorage.setItem(`seg.assist.open.${projectId}`, open ? "1" : "0");
  }, [projectId, open]);

  const toggle = useCallback(() => setOpen(!open), [open, setOpen]);

  // Dragged from the top-left corner, because the panel is anchored to the
  // bottom-right: pulling up and left is what makes it bigger.
  const onGripDown = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    ev.preventDefault();
    drag.current = { x: ev.clientX, y: ev.clientY, w: size.w, h: size.h };
    (ev.target as HTMLElement).setPointerCapture(ev.pointerId);
  }, [size.w, size.h]);

  const onGripMove = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    const from = drag.current;
    if (!from) return;
    const w = Math.max(MIN.w, Math.min(window.innerWidth - 36, from.w - (ev.clientX - from.x)));
    const h = Math.max(MIN.h, Math.min(window.innerHeight - 100, from.h - (ev.clientY - from.y)));
    setSize({ w, h });
  }, []);

  const onGripUp = useCallback(() => {
    drag.current = null;
    if (projectId) localStorage.setItem(`seg.assist.size.${projectId}`, JSON.stringify(size));
  }, [projectId, size]);

  // Dragged by its bar, the way a window is. The panel is anchored to the
  // bottom-right corner, so moving it right or down is moving those numbers
  // down: the sign is the anchor's, not the mouse's.
  const onBarDown = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    if ((ev.target as HTMLElement).closest("button")) return;   // the close button is not a handle
    ev.preventDefault();
    // From where it is drawn, which is where the person took hold of it.
    move.current = { x: ev.clientX, y: ev.clientY, right: shown.right, bottom: shown.bottom };
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
  }, [shown.right, shown.bottom]);

  const onBarMove = useCallback((ev: React.PointerEvent<HTMLDivElement>) => {
    const from = move.current;
    if (!from) return;
    setSpot(onScreen({ right: from.right - (ev.clientX - from.x),
                       bottom: from.bottom - (ev.clientY - from.y) }, size));
  }, [size]);

  const onBarUp = useCallback(() => {
    if (!move.current) return;
    move.current = null;
    if (projectId) localStorage.setItem(`seg.assist.spot.${projectId}`, JSON.stringify(spot));
  }, [projectId, spot]);

  // Somewhere to put it back, for a panel dragged into a corner it is awkward in.
  const onBarDouble = useCallback(() => {
    setSpot(HOME);
    if (projectId) localStorage.setItem(`seg.assist.spot.${projectId}`, JSON.stringify(HOME));
  }, [projectId]);

  if (!active || !projectId) return null;
  return (
    <>
      {open && (
        <aside className="assist-dock"
               style={{ width: size.w, height: size.h, right: shown.right, bottom: shown.bottom }}>
          <div
            className="assist-dock-grip"
            onPointerDown={onGripDown}
            onPointerMove={onGripMove}
            onPointerUp={onGripUp}
            title={t("assist.resize")}
          />
          <div className="assist-dock-bar"
               onPointerDown={onBarDown}
               onPointerMove={onBarMove}
               onPointerUp={onBarUp}
               onPointerCancel={onBarUp}
               onDoubleClick={onBarDouble}
               title={t("assist.move")}>
            <span>{t("assist.open")}</span>
            <button className="assist-dock-close" onClick={toggle} aria-label={t("assist.close")}>
              x
            </button>
          </div>
          <Assist projectId={projectId} active={open} lang={lang} />
        </aside>
      )}
    </>
  );
}
