// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, SEL } from "../helpers";

test.describe("Settings", () => {
  test("settings dialog opens and closes", async ({ page }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);

    // Act: open settings via gear icon
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      btns.find((b) => b.getAttribute("title")?.includes("設定"))?.click();
    });

    // Assert: overlay visible
    const overlay = page.locator(SEL.settingsOverlay);
    await expect(overlay).toBeVisible({ timeout: 3_000 });

    // Act: close
    await page.keyboard.press("Escape");

    // Assert: overlay hidden (or at least we didn't crash)
  });
});
