// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * What the MCP bridge writes has to appear on screen as it lands.
 *
 * A labelling run works on an image (the feed records its SAM prompts there,
 * without coordinates), writes its mask a couple of seconds later, and moves
 * on to the next image. The screen follows it from image to image. The masks
 * appeared sometimes and sometimes not -- and less often once the run wrote
 * each image once instead of three times, because a later write had been what
 * covered for a missed reload.
 *
 * The bridge is played here by the test: /agent/note standing in for the SAM
 * prompt the server records, and the mask PUT, with its header, in its order
 * and at its pace. No segmenter runs. Each image's mask as shown live is
 * compared with the same image opened after a reload, which is what the
 * server holds.
 *
 * Runs in its own throwaway projects, never in the shared seeds.
 */
import { request as pwRequest, type APIRequestContext, type Page } from "@playwright/test";
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectProjectByName, switchTab, SEL } from "../helpers";
import { SEED_IMAGES } from "../helpers/seed-assets";

const API = "http://localhost:8002/api/v1";
const ITEMS = ["live-01", "live-02", "live-03", "live-04"] as const;
const BRIDGE = { "X-Seg-Agent": "mcp/write" };

function overlayPixelCount(page: Page): Promise<number> {
  return page.evaluate(() => {
    const canvas = document.querySelectorAll(".canvas-inner canvas")[1] as HTMLCanvasElement | undefined;
    if (!canvas || canvas.width === 0) return -1;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return -1;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let n = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i]! > 0) n += 1;
    return n;
  });
}

/** 64x64 raw mask: 255 unpainted, a side x side square of class 1. */
function square(side: number): Buffer {
  const m = new Uint8Array(64 * 64).fill(255);
  for (let y = 8; y < 8 + side; y++) for (let x = 8; x < 8 + side; x++) m[y * 64 + x] = 1;
  return Buffer.from(m);
}

async function scratch(request: APIRequestContext, name: string): Promise<string> {
  const projects = (await (await request.get(`${API}/projects`)).json()) as Array<{ id: string; name: string }>;
  const old = projects.find((p) => p.name === name)?.id;
  if (old) await request.delete(`${API}/projects/${old}`);
  const created = await request.post(`${API}/projects`, { data: { name } });
  expect(created.ok()).toBeTruthy();
  const pid = ((await created.json()) as { id: string }).id;
  const imgs = ["seed-01.png", "seed-02.png", "seed-03.png", "seed-01.png"];
  for (const [i, item] of ITEMS.entries()) {
    const up = await request.post(`${API}/projects/${pid}/datasets/annotate/upload`, {
      multipart: { files: { name: `${item}.png`, mimeType: "image/png",
                            buffer: Buffer.from(SEED_IMAGES[imgs[i]!]!, "base64") } },
    });
    expect(up.ok()).toBeTruthy();
  }
  await request.put(`${API}/projects/${pid}/classes`, { data: {
    version: 1, ignore_index: 255, next_class_id: 2,
    classes: [{ id: 0, name: "background", color: [0, 0, 0], active: true },
              { id: 1, name: "class1", color: [255, 0, 0], active: true }],
  } });
  // Blank mask files, as a project whose masks were cleared by a person has.
  for (const item of ITEMS) {
    const put = await request.put(`${API}/projects/${pid}/datasets/annotate/masks/${item}.png?raw=1&w=64&h=64`,
      { headers: { "content-type": "application/octet-stream" }, data: square(0) });
    expect(put.ok()).toBeTruthy();
  }
  return pid;
}

async function open(page: Page, name: string, item: string): Promise<void> {
  await page.goto("/");
  await waitForApi(page);
  await selectProjectByName(page, name);
  await switchTab(page, "annotate");
  await page.locator(SEL.imageListItem).filter({ hasText: item }).first().click();
  await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout: 15_000 });
  await expect(page.locator(SEL.headerProjectName)).toContainText(name);
}

async function shownOn(page: Page, item: string): Promise<number> {
  await page.locator(SEL.imageListItem).filter({ hasText: item }).first().click();
  await expect(page.locator(`${SEL.imageListItem}.active`)).toContainText(item);
  await page.waitForTimeout(400);
  return overlayPixelCount(page);
}

async function playTheBridge(request: APIRequestContext, pid: string, noteToWrite: number,
                             writeToNext: number): Promise<void> {
  for (const [i, item] of ITEMS.slice(1).entries()) {
    const side = 8 * (i + 1);
    await request.post(`${API}/agent/note`, {
      headers: { ...BRIDGE, "X-Seg-Agent-Tool": "sam_segment" },
      data: { action: "sam_segment", project_id: pid, item_id: item },
    });
    await new Promise((r) => setTimeout(r, noteToWrite));
    const put = await request.put(`${API}/projects/${pid}/datasets/annotate/masks/${item}.png?raw=1&w=64&h=64`, {
      headers: { ...BRIDGE, "X-Seg-Agent-Tool": "write_kept", "content-type": "application/octet-stream" },
      data: square(side),
    });
    expect(put.ok()).toBeTruthy();
    await new Promise((r) => setTimeout(r, writeToNext));
  }
}

for (const [label, noteToWrite, writeToNext] of [
  ["at a run's pace", 2000, 1500],
  ["faster than a poll", 300, 700],
] as const) {
  test.describe(`A labelling run is shown as it goes, ${label}`, () => {
    const NAME = `zz-e2e-agent-live-${noteToWrite}`;
    test.afterAll(async () => {
      const ctx = await pwRequest.newContext();
      try {
        const projects = (await (await ctx.get(`${API}/projects`)).json()) as Array<{ id: string; name: string }>;
        const pid = projects.find((p) => p.name === NAME)?.id;
        if (pid) await ctx.delete(`${API}/projects/${pid}`).catch(() => {});
      } finally {
        await ctx.dispose();
      }
    });

    test("each image the bridge wrote shows what it wrote", async ({ page, request }) => {
      await page.addInitScript(() => localStorage.setItem("seg.agentFollow", "1"));
      const pid = await scratch(request, NAME);
      await open(page, NAME, "live-01");
      await playTheBridge(request, pid, noteToWrite, writeToNext);
      // the screen follows the run to the last image it wrote, and shows it there
      await expect(page.locator(`${SEL.imageListItem}.active`)).toContainText("live-04", { timeout: 10_000 });
      await expect.poll(() => overlayPixelCount(page), { timeout: 5_000 }).toBeGreaterThan(0);
      const live: Record<string, number> = {};
      for (const item of ITEMS.slice(1)) live[item] = await shownOn(page, item);
      // what the server holds, opened fresh
      const truth: Record<string, number> = {};
      for (const item of ITEMS.slice(1)) {
        await open(page, NAME, item);
        await page.waitForTimeout(400);
        truth[item] = await overlayPixelCount(page);
      }
      console.log(`[agent-live ${label}] live`, JSON.stringify(live), "truth", JSON.stringify(truth));
      for (const item of ITEMS.slice(1)) expect(truth[item]).toBeGreaterThan(0);
      expect(live).toEqual(truth);
    });
  });
}

async function note(request: APIRequestContext, pid: string, item: string): Promise<void> {
  await request.post(`${API}/agent/note`, {
    headers: { ...BRIDGE, "X-Seg-Agent-Tool": "sam_segment" },
    data: { action: "sam_segment", project_id: pid, item_id: item },
  });
}

async function put(request: APIRequestContext, pid: string, item: string, side: number): Promise<void> {
  const res = await request.put(`${API}/projects/${pid}/datasets/annotate/masks/${item}.png?raw=1&w=64&h=64`, {
    headers: { ...BRIDGE, "X-Seg-Agent-Tool": "write_kept", "content-type": "application/octet-stream" },
    data: square(side),
  });
  expect(res.ok()).toBeTruthy();
}

test.describe("A labelling run is shown as it goes, when the timing is against it", () => {
  const NAME = "zz-e2e-agent-live-race";
  test.afterAll(async () => {
    const ctx = await pwRequest.newContext();
    try {
      const projects = (await (await ctx.get(`${API}/projects`)).json()) as Array<{ id: string; name: string }>;
      const pid = projects.find((p) => p.name === NAME)?.id;
      if (pid) await ctx.delete(`${API}/projects/${pid}`).catch(() => {});
    } finally {
      await ctx.dispose();
    }
  });

  test("a reload that lands late is not painted onto the image the screen moved on to", async ({ page, request }) => {
    await page.addInitScript(() => localStorage.setItem("seg.agentFollow", "1"));
    const pid = await scratch(request, NAME);
    let slow = false;
    await page.route("**/datasets/annotate/masks/live-02.png*", async (route) => {
      if (slow && route.request().method() === "GET") await new Promise((r) => setTimeout(r, 3500));
      await route.continue();
    });
    await open(page, NAME, "live-01");
    await note(request, pid, "live-02");
    await page.waitForTimeout(1500);
    slow = true;                            // live-02 read back slowly from here on
    await put(request, pid, "live-02", 8);
    await page.waitForTimeout(600);
    await note(request, pid, "live-03");   // the screen moves on while live-02 is still loading
    await expect(page.locator(`${SEL.imageListItem}.active`)).toContainText("live-03", { timeout: 10_000 });
    await page.waitForTimeout(4000);        // live-02's late answer has landed by now
    expect(await overlayPixelCount(page), "live-03 is not written yet: nothing on it").toBe(0);
    slow = false;
    await put(request, pid, "live-03", 16);
    await expect.poll(() => overlayPixelCount(page), { timeout: 6_000 }).toBe(256);
    expect(await shownOn(page, "live-02")).toBe(64);
  });

  test("a backlog read in one poll shows every image it wrote", async ({ page, request }) => {
    await page.addInitScript(() => localStorage.setItem("seg.agentFollow", "1"));
    const pid = await scratch(request, NAME);
    let holding = false;
    await page.route("**/api/v1/agent/activity*", async (route) => {
      if (!holding) return route.continue();
      const since = Number(new URL(route.request().url()).searchParams.get("since") || "0");
      // as a hidden tab sees it: nothing new, then all of it at once
      return route.fulfill({ json: { active: true, active_window_s: 30, events: [], last: null, seq: since } });
    });
    await open(page, NAME, "live-01");
    await page.waitForTimeout(1500);
    holding = true;
    for (const [i, item] of ["live-02", "live-03", "live-04"].entries()) {
      await note(request, pid, item);
      await put(request, pid, item, 8 * (i + 1));
    }
    holding = false;
    await expect(page.locator(`${SEL.imageListItem}.active`)).toContainText("live-04", { timeout: 10_000 });
    await expect.poll(() => overlayPixelCount(page), { timeout: 6_000 }).toBe(576);
    expect(await shownOn(page, "live-02")).toBe(64);
    expect(await shownOn(page, "live-03")).toBe(256);
  });
});
