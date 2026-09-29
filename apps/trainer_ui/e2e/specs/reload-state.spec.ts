// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectFirstProject, SEL } from "../helpers";

test.describe("Reload State", () => {
  test("project selection is restored after reload", async ({ page, projects }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);

    // Act: select the first project
    await projects.goto();
    await projects.expectHasProjects();
    await projects.selectFirst();

    // Capture which project is active
    const activeTile = page.locator(SEL.projectTileActive);
    const activeName = await activeTile.textContent();

    // Act: reload
    await page.reload();
    await waitForApi(page);

    // Assert: the same project should be selected after reload
    const restoredTile = page.locator(SEL.projectTileActive);
    await expect(restoredTile).toBeVisible({ timeout: 10_000 });
    const restoredName = await restoredTile.textContent();
    expect(restoredName).toBe(activeName);
  });

  test("language setting is restored after reload", async ({ page }) => {
    // Arrange: set to English via localStorage
    await page.goto("/");
    await waitForApi(page);
    await page.evaluate(() => localStorage.setItem("seg-lang", "en"));
    await page.reload();
    await waitForApi(page);

    // Capture reference
    const langLabel = page.locator(".lang-toggle .lang-toggle-label");
    await expect(langLabel).toHaveText("JP"); // "JP" when current lang is "en"

    // Act: reload again
    await page.reload();
    await waitForApi(page);

    // Assert: still English
    await expect(langLabel).toHaveText("JP");
  });
});
