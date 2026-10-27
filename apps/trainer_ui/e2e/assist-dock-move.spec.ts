// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

import { selectFirstProject, waitForApi } from "./helpers";

/**
 * The conversation panel sits in the bottom-right corner, which is also where
 * the picture is on a wide frame and where the toasts arrive. It has to be
 * possible to put it somewhere else, and for it to stay there.
 */
/** The labelling tab, waited for by the thing this test needs from it: the
 *  button at the foot of the tool column that opens the conversation. */
async function toLabelTab(page: import("@playwright/test").Page): Promise<void> {
  await page.locator(".tabs-fixed button").nth(1).click();
  await page.locator(".tool-icon[aria-pressed]").waitFor({ state: "visible", timeout: 30_000 });
}

test.describe("The assist panel can be moved", () => {
  test("drags by its bar, stays where it was put, and goes home on a double-click",
    async ({ page }) => {
      await page.goto("/");
      await waitForApi(page);
      await selectFirstProject(page);
      await toLabelTab(page);

      const open = page.locator(".tool-icon[aria-pressed]");
      await expect(open).toBeVisible();
      if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();

      const dock = page.locator(".assist-dock");
      await expect(dock).toBeVisible();
      const before = await dock.boundingBox();
      expect(before).not.toBeNull();

      const bar = page.locator(".assist-dock-bar");
      const grab = await bar.boundingBox();
      expect(grab).not.toBeNull();
      await page.mouse.move(grab!.x + grab!.width / 2, grab!.y + grab!.height / 2);
      await page.mouse.down();
      await page.mouse.move(grab!.x + grab!.width / 2 - 150, grab!.y + grab!.height / 2 - 90,
                            { steps: 8 });
      await page.mouse.up();

      const moved = await dock.boundingBox();
      expect(moved!.x).toBeLessThan(before!.x - 100);
      expect(moved!.y).toBeLessThan(before!.y - 50);

      // Put somewhere means put there: a reload finds it in the same place.
      await page.reload();
      await waitForApi(page);
      await toLabelTab(page);
      await expect(dock).toBeVisible();
      const afterReload = await dock.boundingBox();
      expect(Math.abs(afterReload!.x - moved!.x)).toBeLessThan(4);
      expect(Math.abs(afterReload!.y - moved!.y)).toBeLessThan(4);

      await bar.dblclick();
      const home = await dock.boundingBox();
      expect(Math.abs(home!.x - before!.x)).toBeLessThan(4);
      expect(Math.abs(home!.y - before!.y)).toBeLessThan(4);
    });

  test("stays where it was put through a window that has no size for a moment",
    async ({ page }) => {
      await page.goto("/");
      await waitForApi(page);
      await selectFirstProject(page);
      await toLabelTab(page);

      const open = page.locator(".tool-icon[aria-pressed]");
      if ((await open.getAttribute("aria-pressed")) !== "true") await open.click();
      const dock = page.locator(".assist-dock");
      await expect(dock).toBeVisible();
      await page.locator(".assist-dock-bar").dblclick();
      const home = await dock.boundingBox();
      expect(home).not.toBeNull();

      // A tab opened in the background, or a pane not shown yet, reports a
      // window of no size. Measured against that, the panel was pushed past
      // the right and bottom edges and stayed there.
      await page.evaluate(() => {
        const keys = ["innerWidth", "innerHeight"] as const;
        const kept = keys.map((k) => Object.getOwnPropertyDescriptor(window, k));
        for (const k of keys) Object.defineProperty(window, k, { configurable: true, get: () => 0 });
        (window as unknown as { sizedAgain: () => void }).sizedAgain = () => {
          keys.forEach((k, i) => {
            if (kept[i]) Object.defineProperty(window, k, kept[i]!);
            else delete (window as unknown as Record<string, unknown>)[k];
          });
        };
        window.dispatchEvent(new Event("resize"));
      });
      const unsized = await dock.boundingBox();
      expect(Math.abs(unsized!.x - home!.x)).toBeLessThan(4);
      expect(Math.abs(unsized!.y - home!.y)).toBeLessThan(4);
      await page.evaluate(() => {
        (window as unknown as { sizedAgain: () => void }).sizedAgain();
        window.dispatchEvent(new Event("resize"));
      });
      expect(await page.evaluate(() => window.innerWidth)).toBeGreaterThan(0);

      // Put well to the left, then the window shrinks: the panel is held far
      // enough on screen to grab. Grown again, it is back where the person
      // left it, not where the small window held it.
      const bar = page.locator(".assist-dock-bar");
      const grab = await bar.boundingBox();
      const at = { x: grab!.x + grab!.width / 2, y: grab!.y + grab!.height / 2 };
      await page.mouse.move(at.x, at.y);
      await page.mouse.down();
      await page.mouse.move(at.x - 400, at.y, { steps: 8 });
      await page.mouse.up();
      const put = await dock.boundingBox();
      expect(put!.x).toBeLessThan(home!.x - 300);

      const full = page.viewportSize()!;
      await page.setViewportSize({ width: 500, height: full.height });
      await expect.poll(async () => {
        const box = await dock.boundingBox();
        return box!.x + box!.width;
      }).toBeGreaterThanOrEqual(119);
      await page.setViewportSize(full);
      await expect.poll(async () => Math.abs((await dock.boundingBox())!.x - put!.x)).toBeLessThan(4);

      await bar.dblclick();                               // home again, for whoever opens it next
    });
});
