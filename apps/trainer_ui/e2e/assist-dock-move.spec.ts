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
});
