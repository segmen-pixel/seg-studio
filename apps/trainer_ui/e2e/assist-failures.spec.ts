// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

import { selectFirstProject, waitForApi } from "./helpers";

/**
 * A run that is failing has to look like one.
 *
 * A refused tool call was an ordinary "tool" event, which this log folds away
 * behind the steps toggle, and the rehearsal's numbers were inside that same
 * hidden event -- cut at two hundred characters when it was unfolded, in the
 * middle of the word "verdict". So the same refusal, again and again, read
 * from here as a run thinking quietly.
 *
 * The thread is the one a Japanese screen is sent, since that is the language
 * these specs run in: the run words a step's name and the rehearsal line in
 * the screen's language, and what the model itself said comes as it said it.
 */
const THREAD = {
  state: { running: false, paused: false, turns: 3, connection: null },
  entries: [
    { type: "started", text: "labelling zz-e2e" },
    // one per step, carries no text: it used to leave an empty div behind
    { type: "context", used: 1200, budget: 8000, pct: 15 },
    { type: "say", text: "gears on a table" },
    {
      type: "step", step: 3, step_no: 2, of: 4,
      name: "人がラベルを付けた画像を読み、対象に名前を付ける",
      text: "the person labels each whole gear, one mask per gear; the table is not it",
    },
    { type: "context", used: 1600, budget: 8000, pct: 20 },
    {
      type: "rehearsal", step: 4, item_id: "img003", ok: false,
      text: "リハーサル（img003）：重なり 0.912、人が付けた1個のうち1個に到達、はみ出し 2.4%、"
        + "モデルのマスク1枚（3片）。人がラベルを付けた物体1個が、モデルのマスクでは3片に分かれています"
        + "（セグメンターが部品ごとに答えています）。",
    },
    {
      type: "failed", step: 5, name: "accept_mask", n: 3,
      text: "400 from /projects/p/datasets/annotate/img003/sam-segment: "
        + "points/labels or box required",
    },
  ],
};

/**
 * The model server is somebody's machine, not part of these specs: without
 * this a run could only be started where one happened to be running, and
 * picking a model wrote the choice into the settings of the API under test.
 * A spec that is about the list or the connection routes them again itself;
 * the route registered last answers.
 */
const LOCAL_KINDS = [
  { backend: "ollama", base_url: "http://127.0.0.1:11434" },
  { backend: "lmstudio", base_url: "http://127.0.0.1:1234/v1" },
];
const CONNECTION = {
  configured: false, backend: "ollama", base_url: "http://127.0.0.1:11434", model: "",
  where: "the vlm block of projects/runtime_settings.json", source: "default",
  selectable: true, model_from_environment: false, kinds: LOCAL_KINDS,
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/projects/*/agent/models", (route) =>
    route.fulfill({ json: { models: ["a-vision-model"], selected: "a-vision-model", why: "",
                            backend: "ollama", base_url: "http://127.0.0.1:11434" } }));
  await page.route("**/api/v1/agent/connection", (route) =>
    route.fulfill({ json: { connection: CONNECTION } }));
});

async function toLabelTab(page: import("@playwright/test").Page): Promise<void> {
  await page.locator(".tabs-fixed button").nth(1).click();
  await page.locator(".tool-icon[aria-pressed]").waitFor({ state: "visible", timeout: 30_000 });
}

test.describe("A failing run says so", () => {
  test("shows the refused call and the rehearsal, and leaves no empty rows",
    async ({ page }) => {
      await page.route("**/api/v1/projects/*/agent/thread", (route) =>
        route.fulfill({ json: THREAD }));
      await page.goto("/");
      await waitForApi(page);
      await selectFirstProject(page);
      await toLabelTab(page);

      const open = page.locator(".tool-icon[aria-pressed]");
      if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();
      const log = page.locator(".assist-log");
      await expect(log).toBeVisible();

      // The refused call, with the server's own reason -- not "400 Bad Request".
      const failed = log.locator("p.assist-failed");
      await expect(failed).toHaveCount(1);
      await expect(failed).toContainText("points/labels or box required");
      await expect(failed).toContainText("accept_mask");
      await expect(failed).toContainText("\u00d73");          // the third time

      // The dry run, in the model's own column rather than folded away.
      const rehearsal = log.locator("p.assist-rehearsal");
      await expect(rehearsal).toHaveCount(1);
      await expect(rehearsal).toContainText("0.912");
      // the verdict as a Japanese screen is told it, not in the bridge's English
      await expect(rehearsal).toContainText("部品ごとに答えています");
      await expect(rehearsal).toHaveClass(/assist-rehearsal-bad/);
      await expect(log.locator(".assist-entry.assist-theirs.assist-rehearsal")).toHaveCount(1);

      // What it understood, numbered, on the model's own side of the log.
      const stepped = log.locator(".assist-step");
      await expect(stepped).toHaveCount(1);
      await expect(stepped).toContainText("STEP 2/4");
      await expect(stepped).toContainText("人がラベルを付けた画像を読み、対象に名前を付ける");
      await expect(stepped).toContainText("the table is not it");
      await expect(log.locator(".assist-entry.assist-theirs.assist-step")).toHaveCount(1);

      // And nothing rendered for the events that carry no words.
      await expect(log.locator(".assist-context")).toHaveCount(0);
      const empties = await log.locator(".assist-entry:empty").count();
      expect(empties).toBe(0);
    });
});

test.describe("The log can be emptied", () => {
  test("/clear typed in the box empties it, and says the record is kept", async ({ page }) => {
    const cleared: string[] = [];
    await page.route("**/api/v1/projects/*/agent/thread", (route) =>
      route.fulfill({ json: THREAD }));
    await page.route("**/api/v1/projects/*/agent/clear", (route) => {
      cleared.push(route.request().url());
      return route.fulfill({ json: THREAD.state });
    });
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await toLabelTab(page);
    const open = page.locator(".tool-icon[aria-pressed]");
    if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

    const log = page.locator(".assist-log");
    await expect(log.locator("p.assist-failed")).toHaveCount(1);
    // It has to be findable without being told about it: a button now.
    await expect(page.locator(".assist-controls").getByRole("button", { name: /^(clear|削除)$/i }))
      .toBeVisible();
    await expect(page.locator(".assist-hint")).toHaveCount(0);

    await page.locator("#assist-instruction").fill("/clear");
    await page.getByRole("button", { name: /run|実行/i }).first().click();

    await expect(log.locator("p.assist-failed")).toHaveCount(0);
    await expect(log.locator(".assist-entry")).toHaveCount(0);
    expect(cleared.length).toBe(1);

    // and it is a command, not an instruction: nothing was sent to the model
    await expect(page.locator("#assist-instruction")).toHaveValue("");
  });
});

test.describe("Stepping through a run", () => {
  test("/debug-step turns it on, says so, and sends it with the run", async ({ page }) => {
    let sent: Record<string, unknown> | null = null;
    await page.route("**/api/v1/projects/*/agent/thread", (route) =>
      route.fulfill({ json: { state: THREAD.state, entries: [] } }));
    await page.route("**/api/v1/projects/*/agent/run", (route) => {
      sent = JSON.parse(route.request().postData() || "{}");
      return route.fulfill({ body: "", headers: { "content-type": "application/x-ndjson" } });
    });
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await toLabelTab(page);
    const open = page.locator(".tool-icon[aria-pressed]");
    if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

    const box = page.locator("#assist-instruction");
    // It has to be findable without being told it exists: a button now.
    const stepButton = page.locator(".assist-controls")
      .getByRole("button", { name: /^(step view|STEP 表示)$/i });
    await expect(stepButton).toBeVisible();
    await expect(stepButton).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator(".assist-mode")).toHaveCount(0);

    await box.fill("/debug-step");
    await page.getByRole("button", { name: /run|実行/i }).first().click();
    await expect(page.locator(".assist-mode")).toHaveCount(1);
    await expect(box).toHaveValue("");
    expect(sent).toBeNull();          // a command is not an instruction

    await box.fill("机の上の歯車をラベル付けして");
    await page.getByRole("button", { name: /run|実行/i }).first().click();
    await expect.poll(() => sent && (sent as Record<string, unknown>).debug_steps)
      .toBe(true);

    // and it can be turned off again
    await box.fill("/debug-off");
    await page.getByRole("button", { name: /run|実行/i }).first().click();
    await expect(page.locator(".assist-mode")).toHaveCount(0);

    // the button does the same, and shows which way it is
    await stepButton.click();
    await expect(stepButton).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator(".assist-mode")).toHaveCount(1);
    sent = null;
    await box.fill("机の上の歯車をラベル付けして");
    await page.getByRole("button", { name: /run|実行/i }).first().click();
    await expect.poll(() => sent && (sent as Record<string, unknown>).debug_steps).toBe(true);
    await stepButton.click();
    await expect(stepButton).toHaveAttribute("aria-pressed", "false");
    await expect(page.locator(".assist-mode")).toHaveCount(0);
  });
});

test.describe("The log's button and the box", () => {
  test("the button to the right of Pause empties the log", async ({ page }) => {
    const cleared: string[] = [];
    await page.route("**/api/v1/projects/*/agent/thread", (route) =>
      route.fulfill({ json: THREAD }));
    await page.route("**/api/v1/projects/*/agent/clear", (route) => {
      cleared.push(route.request().url());
      return route.fulfill({ json: THREAD.state });
    });
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await toLabelTab(page);
    const open = page.locator(".tool-icon[aria-pressed]");
    if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

    const controls = page.locator(".assist-controls");
    const pause = controls.getByRole("button", { name: /^(pause|一時停止|resume|再開)$/i });
    const clear = controls.getByRole("button", { name: /^(clear|削除)$/i });
    const [p, c] = [await pause.boundingBox(), await clear.boundingBox()];
    expect(p && c && c.x > p.x).toBeTruthy();

    await clear.click();
    await expect(page.locator(".assist-log .assist-entry")).toHaveCount(0);
    expect(cleared.length).toBe(1);
  });

  test("what was typed goes when a run starts, not when it ends", async ({ page }) => {
    let release: () => void = () => {};
    const held = new Promise<void>((r) => { release = r; });
    await page.route("**/api/v1/projects/*/agent/thread", (route) =>
      route.fulfill({ json: { state: THREAD.state, entries: [] } }));
    await page.route("**/api/v1/projects/*/agent/run", async (route) => {
      await held;                       // the run is still going
      return route.fulfill({ body: "", headers: { "content-type": "application/x-ndjson" } });
    });
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await toLabelTab(page);
    const open = page.locator(".tool-icon[aria-pressed]");
    if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

    const box = page.locator("#assist-instruction");
    await box.fill("机の上の歯車をラベル付けして");
    await page.getByRole("button", { name: /run|実行/i }).first().click();
    await expect(box).toHaveValue("");
    release();
  });

  test("the model picker and the state sit in the head, beside AI output", async ({ page }) => {
    await page.route("**/api/v1/projects/*/agent/thread", (route) =>
      route.fulfill({ json: { state: THREAD.state, entries: [] } }));
    await page.route("**/api/v1/projects/*/agent/models", (route) =>
      route.fulfill({ json: { models: ["qwen3.8:27b", "gemma4:31b"], selected: "qwen3.8:27b" } }));
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await toLabelTab(page);
    const open = page.locator(".tool-icon[aria-pressed]");
    if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

    await expect(page.locator(".assist-head select.assist-model")).toHaveValue("qwen3.8:27b");
    await expect(page.locator(".assist-controls select")).toHaveCount(0);
    await expect(page.locator(".assist-head .badge")).toHaveCount(1);
    await expect(page.locator(".assist-label")).toHaveCount(0);
  });

  test("the kind of model server is chosen in the head, and the models are asked for again",
    async ({ page }) => {
      // The panel offers names only: the kind stands for an address the
      // server keeps, and what goes back is the name.
      let stored = { ...CONNECTION };
      const sent: Array<Record<string, unknown>> = [];
      let listings = 0;
      await page.route("**/api/v1/projects/*/agent/thread", (route) =>
        route.fulfill({ json: { state: { ...THREAD.state, connection: stored }, entries: [] } }));
      await page.route("**/api/v1/agent/connection", (route) => {
        const body = route.request().postDataJSON() as Record<string, unknown>;
        sent.push(body);
        if (typeof body.backend === "string") {
          stored = { ...stored, backend: body.backend, base_url: "http://127.0.0.1:1234/v1", source: "file" };
        }
        return route.fulfill({ json: { connection: stored } });
      });
      await page.route("**/api/v1/projects/*/agent/models", (route) => {
        listings += 1;
        const none = stored.backend === "lmstudio";
        return route.fulfill({ json: {
          models: none ? [] : ["a-vision-model"], selected: "", why: none ? "connection refused" : "",
          backend: stored.backend, base_url: stored.base_url } });
      });
      await page.goto("/");
      await waitForApi(page);
      await selectFirstProject(page);
      await toLabelTab(page);
      const open = page.locator(".tool-icon[aria-pressed]");
      if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

      const kind = page.locator(".assist-head select.assist-kind");
      await expect(kind).toHaveValue("ollama");
      await expect(page.locator(".assist-head select.assist-model")).toHaveValue("a-vision-model");
      const before = listings;

      await kind.selectOption("lmstudio");
      await expect(kind).toHaveValue("lmstudio");
      expect(sent).toContainEqual({ backend: "lmstudio" });
      expect(sent.every((body) => !("base_url" in body) && !("api_key_env" in body))).toBe(true);
      await expect.poll(() => listings).toBeGreaterThan(before);
      // nothing answers at that address: the panel says which, and why
      await expect(page.locator(".assist .notice")).toContainText("127.0.0.1:1234");
      await expect(page.locator(".assist .notice")).toContainText("connection refused");
      await expect(page.locator(".assist-head select.assist-model")).toHaveCount(0);
    });
});

test.describe("Who walks the parts, and a question's answers", () => {
  test("the model can be chosen to walk the parts; the choice goes with the run and is kept",
    async ({ page }) => {
      let sent: Record<string, unknown> | null = null;
      await page.route("**/api/v1/projects/*/agent/thread", (route) =>
        route.fulfill({ json: { state: THREAD.state, entries: [] } }));
      await page.route("**/api/v1/projects/*/agent/run", (route) => {
        sent = JSON.parse(route.request().postData() || "{}");
        return route.fulfill({ body: "", headers: { "content-type": "application/x-ndjson" } });
      });
      const openPanel = async () => {
        await waitForApi(page);
        await selectFirstProject(page);
        await toLabelTab(page);
        const open = page.locator(".tool-icon[aria-pressed]");
        if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();
      };
      await page.goto("/");
      await openPanel();

      // Two answers, so two radio buttons, under the box with the others.
      const walkers = page.locator(".assist-controls").getByRole("radiogroup");
      const program = walkers.getByRole("radio", { name: /^(the program|プログラム)$/ });
      const model = walkers.getByRole("radio", { name: /^(the model|モデル)$/ });
      await expect(program).toBeChecked();
      await expect(model).not.toBeChecked();

      const box = page.locator("#assist-instruction");
      const run = page.getByRole("button", { name: /run|実行/i }).first();
      await box.fill("机の上の歯車をラベル付けして");
      await run.click();
      await expect.poll(() => sent && (sent as Record<string, unknown>).driver).toBe("harness");

      await model.check();
      sent = null;
      await box.fill("机の上の歯車をラベル付けして");
      await run.click();
      await expect.poll(() => sent && (sent as Record<string, unknown>).driver).toBe("model");

      // A way of working with the model, not a fact about one run: kept.
      await page.reload();
      await openPanel();
      await expect(page.locator(".assist-controls").getByRole("radio", { name: /^(the model|モデル)$/ }))
        .toBeChecked();
    });

  test("the answers of the question waiting for a reply are buttons, and a press answers it",
    async ({ page }) => {
      const replies: string[] = [];
      let answered = false;
      const older = { type: "question", text: "1 枚目を書きました。このままにしますか。",
        choices: ["このままにする", "取り消す", "止める"] };
      const standing = { type: "question", text: "倍率 100% で進めますか。",
        choices: ["100% で進める", "倍率を選び直す", "やめる"] };
      const waiting = { ...THREAD.state, running: true, waiting_for_reply: true, question: standing.text };
      const goingOn = { ...THREAD.state, running: true, waiting_for_reply: false, question: null };
      await page.route("**/api/v1/projects/*/agent/thread", (route) => route.fulfill({
        json: answered
          ? { state: goingOn, entries: [older, { type: "you", text: "このままにする" }, standing,
            { type: "you", text: replies[0] }] }
          : { state: waiting, entries: [older, { type: "you", text: "このままにする" }, standing] },
      }));
      await page.route("**/api/v1/projects/*/agent/reply", (route) => {
        replies.push(JSON.parse(route.request().postData() || "{}").text);
        answered = true;
        return route.fulfill({ json: goingOn });
      });
      await page.goto("/");
      await waitForApi(page);
      await selectFirstProject(page);
      await toLabelTab(page);
      const open = page.locator(".tool-icon[aria-pressed]");
      if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

      // Only the question a reply waits on: nothing waits on the older one.
      const choices = page.locator(".assist-log .assist-choices");
      await expect(choices).toHaveCount(1);
      await expect(choices.getByRole("button")).toHaveText(standing.choices);

      await choices.getByRole("button", { name: "倍率を選び直す" }).click();
      await expect.poll(() => replies).toEqual(["倍率を選び直す"]);
      await expect(page.locator(".assist-log .assist-choices")).toHaveCount(0);
      expect(replies.length).toBe(1);
    });
});
