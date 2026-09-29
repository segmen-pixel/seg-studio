// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * The bulk per-class 消去 button acts on the server; the canvas and the
 * server must both reflect it, and no stale autosave may write the cleared
 * markings back. Markings used to resurrect through exactly that race: the
 * 600ms autosave debounce (or the fire-and-forget save on image switch)
 * PUT the old mask AFTER the server-side clear.
 *
 * This spec is DESTRUCTIVE, so it runs inside its own throwaway project
 * created through the API — never in the shared seeds, and never anywhere a
 * project-picker fallback could take it (that mistake has been made once).
 */
import { request as pwRequest, type APIRequestContext, type Page } from "@playwright/test";
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectProjectByName, switchTab, SEL } from "../helpers";
import { SEED_IMAGES, seedMask } from "../helpers/seed-assets";

const API = "http://localhost:8002/api/v1";
const SCRATCH = "zz-e2e-clear-scratch";
const ITEMS = ["seed-01", "seed-02"] as const;

/** Opaque pixels on the mask overlay canvas; -1 when the canvas is not up. */
function overlayPixelCount(page: Page): Promise<number> {
  return page.evaluate(() => {
    const canvas = document.querySelectorAll(
      ".canvas-inner canvas",
    )[1] as HTMLCanvasElement | undefined;
    if (!canvas || canvas.width === 0) return -1;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return -1;
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let n = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i]! > 0) n += 1;
    return n;
  });
}

/** Create (or reuse) the scratch project and paint fresh class-1 masks. */
async function ensureScratchProject(request: APIRequestContext): Promise<string> {
  const projects = (await (await request.get(`${API}/projects`)).json()) as Array<{ id: string; name: string }>;
  let pid = projects.find((p) => p.name === SCRATCH)?.id;
  if (!pid) {
    const created = await request.post(`${API}/projects`, { data: { name: SCRATCH } });
    expect(created.ok()).toBeTruthy();
    pid = ((await created.json()) as { id: string }).id;
  }
  const index = await (await request.get(`${API}/projects/${pid}/datasets/annotate`)).json() as { items?: Array<{ id: string }> };
  const have = new Set((index.items ?? []).map((it) => it.id));
  for (const item of ITEMS) {
    if (have.has(item)) continue;
    const up = await request.post(`${API}/projects/${pid}/datasets/annotate/upload`, {
      multipart: { files: { name: `${item}.png`, mimeType: "image/png", buffer: Buffer.from(SEED_IMAGES[`${item}.png`]!, "base64") } },
    });
    expect(up.ok()).toBeTruthy();
  }
  const classes = {
    version: 1, ignore_index: 255,
    classes: [
      { id: 0, name: "background", color: [0, 0, 0], active: true },
      { id: 1, name: "class1", color: [255, 0, 0], active: true },
    ],
    next_class_id: 2,
  };
  await request.put(`${API}/projects/${pid}/classes`, { data: classes });
  const mask = Buffer.from(seedMask());
  for (const item of ITEMS) {
    const put = await request.put(
      `${API}/projects/${pid}/datasets/annotate/masks/${item}.png?raw=1&w=64&h=64`,
      { headers: { "content-type": "application/octet-stream" }, data: mask },
    );
    expect(put.ok()).toBeTruthy();
  }
  // The UI reads the annotate index ONCE at project open, and an item it
  // sees as hasMask=false never gets its mask fetched (loadMaskFor
  // short-circuits) — so wait until the index reflects the PUTs before any
  // page.goto, or a busy server turns this into a permanent-blank flake.
  for (const item of ITEMS) {
    await expect
      .poll(() => classIdsOf(request, pid, item), { timeout: 10_000 })
      .toContain(1);
  }
  return pid;
}

async function classIdsOf(
  request: APIRequestContext, pid: string, itemId: string,
): Promise<number[]> {
  const res = await request.get(`${API}/projects/${pid}/datasets/annotate`);
  if (!res.ok()) return [];
  const items = ((await res.json()) as { items?: Array<{ id: string; annotation?: { classIds?: number[] } }> }).items ?? [];
  return items.find((it) => it.id === itemId)?.annotation?.classIds ?? [];
}

/** Open the scratch project's annotate tab and click the given item row. */
async function openScratchAnnotate(page: Page, itemName: string): Promise<void> {
  await page.goto("/");
  await waitForApi(page);
  await selectProjectByName(page, SCRATCH);
  await switchTab(page, "annotate");
  await page.locator(SEL.imageListItem).filter({ hasText: itemName }).first().click();
  await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout: 15_000 });
  // Belt and braces before anything destructive: we are where we think.
  await expect(page.locator(SEL.headerProjectName)).toContainText(SCRATCH);
}

async function clearClassOnSelection(page: Page): Promise<void> {
  await page.locator(SEL.imageListItem).filter({ hasText: "seed-02" }).first()
    .click({ modifiers: ["Control"] });
  page.once("dialog", (d) => void d.accept());
  await page.locator(".class-list-window .card .class-clear-btn").first().click();
}

test.describe("Class clear refresh", () => {
  test.afterAll(async () => {
    // Best-effort cleanup; the zz- name keeps a leftover harmless.
    const ctx = await pwRequest.newContext();
    try {
      const projects = (await (await ctx.get(`${API}/projects`)).json()) as Array<{ id: string; name: string }>;
      const pid = projects.find((p) => p.name === SCRATCH)?.id;
      if (pid) await ctx.delete(`${API}/projects/${pid}`).catch(() => {});
    } finally {
      await ctx.dispose();
    }
  });

  test("bulk clear empties the canvas and the server, and survives reload", async ({ page, request }) => {
    const pid = await ensureScratchProject(request);
    await openScratchAnnotate(page, "seed-01");
    // The mask must actually be on the overlay before clearing proves anything.
    await expect.poll(() => overlayPixelCount(page), { timeout: 10_000 }).toBeGreaterThan(0);

    await clearClassOnSelection(page);

    // The canvas refreshes without an image switch — this used to stay stale.
    await expect.poll(() => overlayPixelCount(page), { timeout: 10_000 }).toBe(0);

    // The server agrees, and KEEPS agreeing after the autosave debounce
    // window — a stale PUT landing here is the resurrection bug.
    await expect.poll(() => classIdsOf(request, pid, "seed-01"), { timeout: 10_000 }).not.toContain(1);
    await page.waitForTimeout(1500);
    expect(await classIdsOf(request, pid, "seed-01")).not.toContain(1);
    expect(await classIdsOf(request, pid, "seed-02")).not.toContain(1);

    // And a full reload shows the cleared state, not a cached resurrection.
    await openScratchAnnotate(page, "seed-01");
    await expect.poll(() => overlayPixelCount(page), { timeout: 10_000 }).toBe(0);
  });

  test("clearing right after painting does not resurrect through the autosave", async ({ page, request }) => {
    const pid = await ensureScratchProject(request);
    await openScratchAnnotate(page, "seed-01");

    // Paint a brush stroke so a debounced autosave is pending.
    const stack = page.locator(SEL.canvasStack);
    const box = await stack.boundingBox();
    if (!box) { test.skip(true, "canvas not laid out"); return; }
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    await page.mouse.move(cx - 30, cy);
    await page.mouse.down();
    await page.mouse.move(cx + 30, cy, { steps: 8 });
    await page.mouse.up();
    await expect.poll(() => overlayPixelCount(page), { timeout: 10_000 }).toBeGreaterThan(0);

    // Within the 600ms debounce: extend the selection and clear the class.
    await clearClassOnSelection(page);

    await expect.poll(() => overlayPixelCount(page), { timeout: 10_000 }).toBe(0);
    // Give any stale save every chance to land, then insist it did not.
    await page.waitForTimeout(1500);
    expect(await classIdsOf(request, pid, "seed-01")).not.toContain(1);
  });
});
