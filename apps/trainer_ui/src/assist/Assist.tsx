// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useRef, useState } from "react";

import {
  type AgentRunEvent,
  type AgentRunState,
  clearAgentThread,
  fetchAgentModels,
  fetchAgentThread,
  pauseAgentRun,
  replyToAgentRun,
  startAgentRun,
  stopAgentRun,
} from "../api/agent";
import { useI18n } from "../i18n";

type Props = {
  projectId: string | null;
  active: boolean;
  lang: string;
};

/**
 * Labelling with a vision model, in the product rather than beside it.
 *
 * The loop runs on the server -- a browser cannot spawn the bridge -- so this
 * is a conversation and four buttons. Where the prompts go is configuration:
 * the tab shows what is set and where to change it, and cannot set it, because
 * an address from the browser would make the server fetch whatever it named.
 */
/**
 * Which side of the conversation an event belongs on.
 *
 * Speech goes left and right the way a chat does -- the model's on the left,
 * the person's on the right. Everything else is the machinery of the run: tool
 * calls, the pictures it was shown, the notes about trimming and pausing. Those
 * are not anybody's speech, and putting them on a side reads as though they
 * were said; they run down the middle, small.
 */
function sideOf(type: string): "mine" | "theirs" | "note" {
  if (type === "you") return "mine";
  if (type === "say" || type === "final" || type === "question") return "theirs";
  // The rehearsal is the model's own dry run against an image somebody already
  // labelled -- its answer to "am I about to get this right", which is the
  // thing worth reading before it labels the rest. It belongs with its words.
  if (type === "rehearsal") return "theirs";
  // A step's answer is the model saying what it understands, which is speech.
  if (type === "step") return "theirs";
  return "note";
}

export default function Assist({ projectId, active, lang }: Props) {
  const { t } = useI18n();
  const [entries, setEntries] = useState<AgentRunEvent[]>([]);
  // Remembered per project: a person who is stepping through one project is
  // usually still stepping through it after a reload.
  const [debugSteps, setDebugSteps] = useState(false);
  const [statusLine, setStatusLine] = useState("");
  const [state, setState] = useState<AgentRunState | null>(null);
  const [instruction, setInstruction] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [showSteps, setShowSteps] = useState(false);
  const [error, setError] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  // which picture is open: a run shows dozens, each 1280 px wide
  const [expanded, setExpanded] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  // Rejoin whatever is already going, and show what the last run said. The
  // server keeps the conversation whether or not a screen was listening, so a
  // reload is not a lost run, and a run that has ended -- the case where why
  // it stopped matters most -- can still be read. One request when the panel
  // opens: while a run is going the server's copy is the whole of it; after
  // one, what this screen already shows is kept, and an empty screen takes
  // the stored thread.
  useEffect(() => {
    if (!active || !projectId) return;
    let cancelled = false;
    fetchAgentThread(projectId)
      .then((got) => {
        if (cancelled) return;
        setState(got.state);
        setEntries((have) => (got.state?.running || !have.length ? got.entries : have));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [active, projectId]);

  // What the configured server holds. A model is a name, so picking one is
  // safe from here; where the server sends prompts is not offered, because an
  // address from a browser would make it fetch whatever was named.
  useEffect(() => {
    if (!projectId) return;
    setDebugSteps(!!localStorage.getItem(`seg.assist.debugSteps.${projectId}`));
    setStatusLine("");
    // another project's conversation is not this one's, even for a moment
    setEntries([]);
  }, [projectId]);

  useEffect(() => {
    if (!active || !projectId) return;
    let cancelled = false;
    fetchAgentModels(projectId)
      .then((got) => {
        if (cancelled) return;
        setModels(got.models);
        const remembered = localStorage.getItem(`seg.assist.model.${projectId}`) ?? "";
        const usable = got.models.includes(remembered) ? remembered : got.selected;
        setModel(usable);
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [active, projectId]);

  // While something is running, the server is the authority on pause and on a
  // question waiting: another screen -- or a reload of this one -- can be the
  // thing that answered it.
  // The conversation, not only the state. A run is driven by the request that
  // started it, and its events reach the browser down that request's own
  // stream -- so a tab that reloaded, or one opened later, has no stream and
  // was left with the snapshot it fetched on mount. The state poll kept
  // "waiting for a reply" true, so the box appeared, over a panel that never
  // showed the question. It looked like a run that had stopped talking.
  useEffect(() => {
    if (!active || !projectId || !state?.running) return;
    const id = window.setInterval(() => {
      fetchAgentThread(projectId)
        .then((got) => {
          setState(got.state);
          if (got.state?.running) setEntries(got.entries);
        })
        .catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(id);
  }, [active, projectId, state?.running]);

  useEffect(() => {
    const box = logRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [entries.length]);

  /**
   * Empties the log and the thread the next run would carry on from. The
   * server keeps its own record of every run. A button beside Pause -- asked
   * for by the person using it -- as well as /clear in the box: a command
   * nobody is shown is one nobody finds. Not during a run: the log is its own.
   */
  const clearLog = useCallback(async () => {
    if (!projectId) return;
    setEntries([]);
    try {
      setState(await clearAgentThread(projectId));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, [projectId]);

  /**
   * Stepping: the next run stops after each step before labelling, with what
   * it found. The STEP button beside Clear, or /debug-step and /debug-off in
   * the box. It goes with a run when it starts, so not while one is going.
   */
  const setStepMode = useCallback((on: boolean) => {
    if (!projectId) return;
    setDebugSteps(on);
    localStorage.setItem(`seg.assist.debugSteps.${projectId}`, on ? "1" : "");
    setStatusLine(on ? t("assist.debugStepOn") : t("assist.debugStepOff"));
  }, [projectId, t]);

  /**
   * The commands you can type instead of saying something.
   *
   * Returns true when the text was a command and has been dealt with.
   */
  const command = useCallback(async (said: string): Promise<boolean> => {
    if (!projectId) return false;
    const word = said.trim().toLowerCase();
    if (word === "/debug-step" || word === "/debug-off") {
      setStepMode(word === "/debug-step");
      setInstruction("");
      return true;
    }
    if (word !== "/clear" && word !== "/reset") return false;
    setInstruction("");
    await clearLog();
    return true;
  }, [projectId, clearLog, setStepMode]);

  const start = useCallback(async () => {
    if (!projectId || !instruction.trim()) return;
    if (await command(instruction)) return;
    // Emptied as it is sent, as a reply is. It used to stay in the box until
    // the run ended -- for the whole of a run, the same words sitting there as
    // if they had not gone.
    const said = instruction.trim();
    setInstruction("");
    setError("");
    // A new run starts on a clean sheet rather than under the previous one.
    // Cleared, not seeded. The run puts the instruction into the stream itself
    // -- the person's side of the conversation is the server's to record -- and
    // adding it here as well showed everything you typed twice.
    setEntries([]);
    setState((s) => (s ? { ...s, running: true, paused: false, turns: 0 } : s));
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await startAgentRun(
        projectId,
        said,
        { confirm, lang, model, debugSteps },
        (event) => {
          setEntries((was) => [...was, event]);
          setState((s) => {
            if (!s) return s;
            if (event.type === "question") {
              return { ...s, question: event.text ?? null, waiting_for_reply: true };
            }
            if (event.type === "tool") return { ...s, turns: s.turns + 1 };
            if (event.type === "paused") return { ...s, paused: true };
            if (event.type === "resumed") return { ...s, paused: false };
            if (event.type === "final" || event.type === "stopped" || event.type === "error") {
              return { ...s, running: false, waiting_for_reply: false, question: null };
            }
            return s;
          });
        },
        controller.signal,
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      // not taken: give the words back, unless something new was typed since
      setInstruction((now) => now || said);
    } finally {
      abortRef.current = null;
      setState((s) => (s ? { ...s, running: false } : s));
      if (projectId) fetchAgentThread(projectId).then((g) => setState(g.state)).catch(() => undefined);
    }
  }, [projectId, instruction, confirm, lang, model, debugSteps, command]);

  const stop = useCallback(async () => {
    if (!projectId) return;
    setState(await stopAgentRun(projectId));
  }, [projectId]);

  const togglePause = useCallback(async () => {
    if (!projectId) return;
    setState(await pauseAgentRun(projectId, !(state?.paused ?? false)));
  }, [projectId, state?.paused]);

  const send = useCallback(async () => {
    if (!projectId || !instruction.trim()) return;
    if (await command(instruction)) return;
    const said = instruction.trim();
    setInstruction("");
    setState(await replyToAgentRun(projectId, said));
  }, [projectId, instruction, command]);

  // What it did, against what it looked at. Every call is logged, and in a
  // run the looking outnumbers the doing several times over: a frame is
  // fetched, the status asked, the band read, and one box is kept. Naming the
  // few that change something keeps the list readable and keeps it honest when
  // a new read is added later -- an unknown tool is a read until it says
  // otherwise, and the count of what was folded away is always shown.
  // Every tool call and every picture, behind the toggle. The writing ones --
  // accept_mask, write_kept -- used to be left in view on the grounds that they
  // are what the run is for, and a hundred images of them buried the two or
  // three lines anybody reads. What is left is the conversation; the count says
  // how much is folded, and one click brings it all back.
  const quiet = (e: { type: string; name?: string }) =>
    e.type === "tool" || e.type === "image";
  // context carries no text -- it is the token budget, once per step -- and had
  // no branch below, so a long run laid down one empty div per step and pushed
  // the conversation up out of view.
  const legible = entries.filter((e) => e.type !== "context");
  const shown = showSteps ? legible : legible.filter((e) => !quiet(e));
  const folded = legible.length - shown.length;

  if (!projectId) return <p className="muted">{t("assist.noProject")}</p>;

  const conn = state?.connection;
  return (
    <div className="assist">
      {/* The model and the run's state up here, beside what they are about,
          so the conversation below gets the height: asked for by the person
          using it. Where the server is goes on the picker's tooltip. */}
      <div className="assist-head">
        <span className="assist-role">{t("assist.aiOutput")}</span>
        {state && state.turns > 0 ? <span className="muted">({state.turns})</span> : null}
        {models.length > 0 ? (
          <select
            className="assist-model"
            value={model}
            disabled={state?.running}
            title={conn ? `${conn.backend} ${conn.base_url}` : undefined}
            onChange={(ev) => {
              setModel(ev.target.value);
              if (projectId) localStorage.setItem(`seg.assist.model.${projectId}`, ev.target.value);
            }}
          >
            {models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        ) : conn ? (
          <span className="muted assist-conn">
            {conn.backend} {conn.base_url}
          </span>
        ) : null}
        <span className={state?.running ? "badge badge-live" : "badge"}>
          {state?.running ? t("assist.running") : t("assist.idle")}
        </span>
      </div>

      {conn && !conn.configured && (
        <p className="notice">
          {t("assist.notConfigured")} <br />
          <span className="muted">{t("assist.where")}: {conn.where}</span>
        </p>
      )}

      {(folded > 0 || showSteps) && (
        <button className="assist-steps-toggle" onClick={() => setShowSteps(!showSteps)}>
          {showSteps
            ? t("assist.hideSteps")
            : t("assist.showSteps").replace("{n}", String(folded))}
        </button>
      )}

      <div className="assist-log" ref={logRef}>
        {shown.length === 0 && <p className="muted">{t("assist.empty")}</p>}
        {shown.map((e, i) => (
          <div key={i} className={`assist-entry assist-${e.type} assist-${sideOf(e.type)}`}>
            {e.type === "tool" && (
              <div>
                <code>{e.name}</code>
                <span className="muted"> {String(e.result ?? "").slice(0, 200)}</span>
              </div>
            )}
            {e.type === "image" && (
              e.jpeg_b64 ? (
                <figure className={expanded === i ? "wide" : ""}>
                  <img
                    src={`data:image/jpeg;base64,${e.jpeg_b64}`}
                    alt={e.caption ?? ""}
                    onClick={() => setExpanded(expanded === i ? null : i)}
                  />
                  <figcaption className="muted">{e.caption} - {t("assist.shown")}</figcaption>
                </figure>
              ) : (
                <span className="muted">{e.caption} - {t("assist.shown")}</span>
              )
            )}
            {e.type === "started" && (
              <p className="muted">{e.text}</p>
            )}
            {(e.type === "say" || e.type === "you") && <p>{e.text}</p>}
            {e.type === "step" && (
              /* No class of its own: the wrapper above already carries
                 assist-step from the event type, and naming the inner one the
                 same made every selector match two elements. */
              <div>
                <div className="assist-step-head">
                  {e.step_no && e.of ? `STEP ${e.step_no}/${e.of}` : "STEP"}
                  {e.name ? <span className="muted"> {e.name}</span> : null}
                </div>
                <p>{e.text}</p>
              </div>
            )}
            {e.type === "rehearsal" && (
              <p className={e.ok ? "assist-rehearsal" : "assist-rehearsal assist-rehearsal-bad"}>
                <span aria-hidden="true">{e.ok ? "\u25CF " : "\u25B2 "}</span>
                {e.text}
              </p>
            )}
            {e.type === "failed" && (
              <p className="assist-failed">
                <span aria-hidden="true">{"\u25B2 "}</span>
                <code>{e.name}</code> {e.text}
                {typeof e.n === "number" && e.n > 1 ? (
                  <span className="muted"> &times;{e.n}</span>
                ) : null}
              </p>
            )}
            {e.type === "auto" && <p className="muted">{e.text}</p>}
            {(e.type === "final" || e.type === "stopped" || e.type === "error"
              || e.type === "paused" || e.type === "resumed" || e.type === "trimmed") && (
              <p>{e.text}</p>
            )}
            {e.type === "question" && <p className="assist-question">{e.text}</p>}
          </div>
        ))}
      </div>

      {/* One box. Answering a question and giving an instruction are the same
          act -- typing to it -- and two fields meant the question had its own
          little input above the one you were already looking at. */}
      <textarea id="assist-instruction" rows={2} value={instruction}
                aria-label={state?.waiting_for_reply ? t("assist.answering") : t("assist.userInput")}
                placeholder={state?.waiting_for_reply
                  ? t("assist.answerPlaceholder") : t("assist.placeholder")}
                onChange={(ev) => setInstruction(ev.target.value)}
                onKeyDown={(ev) => {
                  if (state?.waiting_for_reply && ev.key === "Enter" && !ev.shiftKey) {
                    ev.preventDefault();
                    void send();
                  }
                }} />
      {debugSteps && (
        <p className="assist-mode">{t("assist.debugStepOn")}</p>
      )}
      {statusLine && <p className="muted">{statusLine}</p>}
      <div className="assist-controls">
        {state?.waiting_for_reply ? (
          <button className="primary" onClick={() => void send()}
                  disabled={!instruction.trim()}>
            {t("assist.reply")}
          </button>
        ) : (
          <button onClick={() => void start()} disabled={!instruction.trim() || state?.running}>
            {t("assist.run")}
          </button>
        )}
        <button onClick={() => void stop()} disabled={!state?.running}>{t("assist.stop")}</button>
        <button onClick={() => void togglePause()} disabled={!state?.running}>
          {state?.paused ? t("assist.resume") : t("assist.pause")}
        </button>
        <button onClick={() => void clearLog()} disabled={state?.running}
                title={t("assist.clearTitle")}>
          {t("assist.clear")}
        </button>
        <button onClick={() => setStepMode(!debugSteps)} disabled={state?.running}
                aria-pressed={debugSteps} className={debugSteps ? "primary" : undefined}
                title={t("assist.stepViewTitle")}>
          {t("assist.stepView")}
        </button>
        <label>
          <input type="checkbox" checked={confirm} onChange={(ev) => setConfirm(ev.target.checked)} />
          {t("assist.confirm")}
        </label>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
