// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi } from "../helpers";

test.describe("i18n", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
  });

  test("switching language from Japanese to English updates UI text", async ({ page }) => {
    // Arrange: ensure we start in Japanese
    await page.evaluate(() => localStorage.setItem("seg-lang", "ja"));
    await page.reload();
    await waitForApi(page);

    const langBtn = page.locator(".lang-toggle");
    await expect(langBtn).toBeVisible({ timeout: 5_000 });

    // Capture text before switch
    const labelBefore = await langBtn.locator(".lang-toggle-label").textContent();
    expect(labelBefore).toBe("EN"); // shows "EN" when current lang is "ja"

    // Act: click to switch to English
    await langBtn.click();

    // Assert: label changes to "JP" (current lang is now "en")
    await expect(langBtn.locator(".lang-toggle-label")).toHaveText("JP");
  });

  test("language setting persists after page reload", async ({ page }) => {
    // Arrange: set language to English
    await page.evaluate(() => localStorage.setItem("seg-lang", "en"));
    await page.reload();
    await waitForApi(page);

    const langBtn = page.locator(".lang-toggle");
    await expect(langBtn.locator(".lang-toggle-label")).toHaveText("JP");

    // Act: reload
    await page.reload();
    await waitForApi(page);

    // Assert: still English
    await expect(langBtn.locator(".lang-toggle-label")).toHaveText("JP");
  });
});
