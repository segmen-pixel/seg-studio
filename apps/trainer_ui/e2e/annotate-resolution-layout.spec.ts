// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";

const EXPECTED_ERRORS = /favicon|Failed to load resource|net::ERR_|AbortError|the server responded with a status of 4/i;

function useConsoleErrorCollector(page: Page) {
  const errors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  return {
    assertNoErrors() {
      const filtered = errors.filter((entry) => !EXPECTED_ERRORS.test(entry));
      expect(filtered, `Unexpected console errors:\n${filtered.join("\n")}`).toHaveLength(0);
    },
  };
}

async function waitForApi(page: Page) {
  await expect(page.locator(".api-connecting-banner")).toHaveCount(0, { timeout: 90_000 });
}

async function switchTab(page: Page, tab: "projects" | "annotate") {
  const TAB_INDEX: Record<string, number> = { projects: 0, annotate: 1, training: 2 };
  const idx = TAB_INDEX[tab] ?? 0;
  await page.evaluate((i) => {
    const btns = document.querySelectorAll('.tabs-fixed button');
    if (btns[i]) (btns[i] as HTMLElement).click();
  }, idx);
  await page.waitForSelector(`.app-shell.tab-${tab}`, { timeout: 10_000 });
}

async function selectFirstProject(page: Page) {
  await switchTab(page, "projects");
  const tile = page.locator(".project-tile").first();
  await expect(tile).toBeVisible({ timeout: 10_000 });
  await tile.click();
  await expect(tile).toHaveClass(/active/);
}

const VIEWPORTS = [
  { width: 1920, height: 1080 },
  { width: 1600, height: 900 },
  { width: 1366, height: 768 },
  { width: 1280, height: 720 },
] as const;

test.describe("Annotate Resolution Layout", () => {
  for (const viewport of VIEWPORTS) {
    test(`right tool rail stays inside canvas at ${viewport.width}x${viewport.height}`, async ({ page }) => {
      const col = useConsoleErrorCollector(page);
      await page.setViewportSize(viewport);
      await page.goto("/");
      await waitForApi(page);
      await selectFirstProject(page);
      await switchTab(page, "annotate");
      await page.waitForSelector(".canvas-stack", { timeout: 15_000 });

      const canvas = page.locator(".canvas-stack").first();
      const rail = page.locator(".overlay-tools.right").first();
      await expect(canvas).toBeVisible();
      await expect(rail).toBeVisible();

      const metrics = await page.evaluate(() => {
        const canvasEl = document.querySelector(".canvas-stack") as HTMLElement | null;
        const railEl = document.querySelector(".overlay-tools.right") as HTMLElement | null;
        if (!canvasEl || !railEl) return null;
        const canvasRect = canvasEl.getBoundingClientRect();
        const railRect = railEl.getBoundingClientRect();
        return {
          canvasTop: canvasRect.top,
          canvasRight: canvasRect.right,
          canvasBottom: canvasRect.bottom,
          railTop: railRect.top,
          railRight: railRect.right,
          railBottom: railRect.bottom,
          railClientHeight: railEl.clientHeight,
          railScrollHeight: railEl.scrollHeight,
        };
      });

      expect(metrics).not.toBeNull();
      expect(metrics!.railTop).toBeGreaterThanOrEqual(metrics!.canvasTop + 4);
      expect(metrics!.railRight).toBeLessThanOrEqual(metrics!.canvasRight - 4);
      expect(metrics!.railBottom).toBeLessThanOrEqual(metrics!.canvasBottom - 4);
      expect(metrics!.railClientHeight).toBeGreaterThan(0);
      expect(metrics!.railScrollHeight).toBeGreaterThanOrEqual(metrics!.railClientHeight);

      col.assertNoErrors();
    });
  }
});
