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
      name: "教師を読み、対象に名前を付ける",
      text: "the person labels each whole gear, one mask per gear; the table is not it",
    },
    { type: "context", used: 1600, budget: 8000, pct: 20 },
    {
      type: "rehearsal", step: 4, item_id: "img003", ok: false,
      text: "リハーサル img003: 重なり 0.912、教師の 1 個中 1 個に到達、はみ出し 2.4%、"
        + "自分の 1 マスクが 3 片 \u2014 教師の物体 1 個が自分の 3 片に分かれています"
        + "（セグメンターが部品で答えています）",
    },
    {
      type: "failed", step: 5, name: "accept_mask", n: 3,
      text: "400 from /projects/p/datasets/annotate/img003/sam-segment: "
        + "points/labels or box required",
    },
  ],
};

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
      await expect(rehearsal).toContainText("部品で答えています");
      await expect(rehearsal).toHaveClass(/assist-rehearsal-bad/);
      await expect(log.locator(".assist-entry.assist-theirs.assist-rehearsal")).toHaveCount(1);

      // What it understood, numbered, on the model's own side of the log.
      const stepped = log.locator(".assist-step");
      await expect(stepped).toHaveCount(1);
      await expect(stepped).toContainText("STEP 2/4");
      await expect(stepped).toContainText("教師を読み、対象に名前を付ける");
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
      .getByRole("button", { name: /^(step view|STEP表示)$/i });
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

    await expect(page.locator(".assist-head select")).toHaveValue("qwen3.8:27b");
    await expect(page.locator(".assist-controls select")).toHaveCount(0);
    await expect(page.locator(".assist-head .badge")).toHaveCount(1);
    await expect(page.locator(".assist-label")).toHaveCount(0);
  });
});
