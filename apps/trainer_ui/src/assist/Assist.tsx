// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useEffect, useId, useRef, useState } from "react";

import {
  AGENT_DRIVERS,
  type AgentDriver,
  type AgentRunEvent,
  type AgentRunState,
  clearAgentThread,
  fetchAgentModels,
  fetchAgentThread,
  pauseAgentRun,
  replyToAgentRun,
  saveAgentConnection,
  startAgentRun,
  stopAgentRun,
} from "../api/agent";
import { isImeEnter } from "../components/imeEnter";
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
 * is a conversation and four buttons. The tab chooses two names, the kind of
 * model server on the server's machine and the model. It shows an address and
 * cannot set one, because an address from the browser would make the server
 * fetch whatever it named.
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

/**
 * Who walks the parts, remembered for this viewer across projects: it is a way
 * of working with the model, not a fact about one project. The browser may
 * refuse storage (a private window, blocked site data); the choice then lasts
 * as long as the tab.
 */
const DRIVER_KEY = "seg.assist.driver";

/**
 * The names people know the model servers by. The keys are the kinds the
 * server offers; a kind it adds later is shown under its own key.
 */
const SERVER_NAMES: Record<string, string> = {
  ollama: "Ollama",
  lmstudio: "LM Studio",
  vllm: "vLLM",
  mlx: "MLX",
  llamacpp: "llama.cpp",
  openai: "OpenAI",
};

function rememberedDriver(): AgentDriver {
  try {
    const got = localStorage.getItem(DRIVER_KEY);
    return AGENT_DRIVERS.find((d) => d === got) ?? "harness";
  } catch {
    return "harness";
  }
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
  // Why the list of models is empty, when it is; and a counter that asks for
  // the list again once the kind of server has been changed.
  const [modelsWhy, setModelsWhy] = useState("");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [serverPick, setServerPick] = useState(0);
  const [driver, setDriver] = useState<AgentDriver>(rememberedDriver);
  // Each panel its own group: the dock and the tab can both be open.
  const driverGroup = useId();
  // The question whose answer was just pressed, so a second press does not
  // send a second reply to a question that no longer waits.
  const [pressed, setPressed] = useState<string | null>(null);
  // which picture is open: a run shows dozens, each up to 1344x896
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

  // What the model server holds. A model is a name, and so is the kind of
  // server on this machine, so both are picked from here; an address is not
  // offered, because one from a browser would make the server fetch whatever
  // was named.
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
        setModelsWhy(got.why);
        setModelsLoaded(true);
        const remembered = localStorage.getItem(`seg.assist.model.${projectId}`) ?? "";
        // A name the server lists wins: after a change of server the first
        // model it holds stands in until one is picked. A server that lists
        // nothing keeps the model the machine named, which is sent as it is.
        const usable = got.models.includes(remembered) ? remembered
          : got.models.includes(got.selected) ? got.selected
          : got.models.length > 0 ? got.models[0] : got.selected;
        setModel(usable);
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [active, projectId, serverPick]);

  /**
   * Another kind of model server on this machine. The server stores the name
   * and answers with the address that kind stands for; the list of models is
   * then asked for again. Not during a run, which keeps the server it began with.
   */
  const chooseServer = useCallback(async (backend: string) => {
    setError("");
    try {
      const connection = await saveAgentConnection({ backend });
      setState((s) => (s ? { ...s, connection } : s));
      setModels([]);
      setModel("");
      setModelsLoaded(false);
      setServerPick((n) => n + 1);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, []);

  /** The model for this project on this browser, and the one a new browser starts with. */
  const chooseModel = useCallback((name: string) => {
    setModel(name);
    setError("");
    try {
      if (projectId) localStorage.setItem(`seg.assist.model.${projectId}`, name);
    } catch { /* storage refused: the choice lasts as long as the tab */ }
    // Kept on the server too so the choice outlives this browser. A machine
    // that fixes the model itself refuses, and the per-run choice still holds.
    void saveAgentConnection({ model: name }).catch(() => undefined);
  }, [projectId]);

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
    if (!model && modelsLoaded) {
      // Said here: the instruction stays in the box for when there is one.
      // The list is asked for again, so a model server started since the
      // panel opened is found on the next press.
      setError(models.length > 0 ? t("assist.pickModel") : t("assist.noServer"));
      setServerPick((n) => n + 1);
      return;
    }
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
        { confirm, lang, model, debugSteps, driver },
        (event) => {
          setEntries((was) => [...was, event]);
          // A new question: its answers can be pressed, whatever was pressed before.
          if (event.type === "question") setPressed(null);
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
  }, [projectId, instruction, confirm, lang, model, models, modelsLoaded, debugSteps, driver, command, t]);

  const chooseDriver = useCallback((next: AgentDriver) => {
    setDriver(next);
    try {
      localStorage.setItem(DRIVER_KEY, next);
    } catch {
      // not stored: the choice holds for this tab
    }
  }, []);

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

  /**
   * A question's answer chosen with its button: the label goes back as the
   * reply, the same way a typed one does, and whatever was typed stays.
   */
  const choose = useCallback(async (question: string, label: string) => {
    if (!projectId) return;
    setPressed(question);
    try {
      setState(await replyToAgentRun(projectId, label));
    } catch (exc) {
      setPressed(null);
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, [projectId]);

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
  // The question a reply is waiting on: the last one asked, with nothing said
  // to it since. Its answers are buttons. They were only words at the end of
  // its text, and a person had to type "1" or the label back; an older
  // question's answers stay words, since nothing waits on them any more.
  const lastQuestion = shown.reduce((at, e, i) => (e.type === "question" ? i : at), -1);
  const standing = state?.waiting_for_reply && lastQuestion >= 0
    && !shown.slice(lastQuestion + 1).some((e) => e.type === "you") ? lastQuestion : -1;

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
        {conn && conn.selectable && (conn.kinds?.length ?? 0) > 0 ? (
          <select
            className="assist-kind"
            value={conn.backend}
            disabled={state?.running}
            aria-label={t("assist.server")}
            title={`${t("assist.serverTitle")} ${conn.base_url}`}
            onChange={(ev) => void chooseServer(ev.target.value)}
          >
            {(conn.kinds ?? []).map((k) => (
              <option key={k.backend} value={k.backend}>{SERVER_NAMES[k.backend] ?? k.backend}</option>
            ))}
          </select>
        ) : conn ? (
          <span className="muted assist-conn" title={`${t("assist.setOnMachine")} ${conn.where}`}>
            {SERVER_NAMES[conn.backend] ?? conn.backend}
          </span>
        ) : null}
        {models.length > 0 ? (
          <select
            className="assist-model"
            value={model}
            disabled={state?.running}
            aria-label={t("assist.model")}
            title={conn ? `${conn.backend} ${conn.base_url}` : undefined}
            onChange={(ev) => chooseModel(ev.target.value)}
          >
            {models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        ) : conn && !conn.selectable ? (
          <span className="muted assist-conn">{conn.base_url}</span>
        ) : null}
        <span className={state?.running ? "badge badge-live" : "badge"}>
          {state?.running ? t("assist.running") : t("assist.idle")}
        </span>
      </div>

      {conn && modelsLoaded && models.length === 0 && !state?.running && (
        <p className="notice">
          {t("assist.noServer")} <br />
          <span className="muted">
            {SERVER_NAMES[conn.backend] ?? conn.backend} {conn.base_url}
            {modelsWhy ? ` (${modelsWhy})` : ""}
          </span>
          <br />
          <span className="muted">
            {conn.selectable ? t("assist.noServerHint") : `${t("assist.setOnMachine")} ${conn.where}`}
          </span>
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
            {e.type === "question" && i === standing && e.choices && e.choices.length > 0 && (
              <div className="assist-choices" role="group" aria-label={t("assist.choices")}>
                {e.choices.map((c) => (
                  <button key={c} type="button" disabled={pressed === e.text}
                          onClick={() => void choose(e.text ?? "", c)}>{c}</button>
                ))}
              </div>
            )}
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
                  // The Enter that confirms a Japanese conversion is not a send.
                  if (state?.waiting_for_reply && ev.key === "Enter" && !ev.shiftKey && !isImeEnter(ev)) {
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
        <label title={t("assist.confirmTitle")}>
          <input type="checkbox" checked={confirm} onChange={(ev) => setConfirm(ev.target.checked)} />
          {t("assist.confirm")}
        </label>
        {/* Who walks an image part by part at a filed zoom: two answers, so
            two radio buttons rather than a list to open -- the lists of the
            panel stay the server's and the model's, in the head. Goes with the
            run when it starts, so not while one is going. */}
        <span className="assist-driver" role="radiogroup" aria-label={t("assist.driver")}
              title={t("assist.driverTitle")}>
          {t("assist.driver")}
          {AGENT_DRIVERS.map((d) => (
            <label key={d}>
              <input type="radio" name={driverGroup} value={d} checked={driver === d}
                     disabled={state?.running} onChange={() => chooseDriver(d)} />
              {d === "model" ? t("assist.driverModel") : t("assist.driverHarness")}
            </label>
          ))}
        </span>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
