// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, switchTab, SEL } from "../helpers";

test.describe("Smoke", () => {
  test("page loads and API connects", async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
    await expect(page.locator(SEL.tabButtons).nth(0)).toBeVisible();
  });

  test("all tabs can be visited", async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);

    for (const tab of ["projects", "annotate", "training"] as const) {
      await switchTab(page, tab);
      await expect(page.locator(SEL.appShellTab(tab))).toBeVisible();
    }
  });
});
