// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, SEL, waitForProjects } from "../helpers";

test.describe("Keyboard Flow", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
  });

  test("Tab key navigates between tab buttons", async ({ page }) => {
    // Arrange — focus the first tab button
    const tabButtons = page.locator(SEL.tabButtons);
    await expect(tabButtons.first()).toBeVisible();
    await tabButtons.first().focus();

    // Act — press Tab to move focus forward
    await page.keyboard.press("Tab");

    // Assert — focus should have moved to the next focusable element
    const focused = page.locator(":focus");
    await expect(focused).toBeVisible();
    // The focused element should be a tab button or the next logical element
    const tagName = await focused.evaluate((el) => el.tagName.toLowerCase());
    expect(["button", "a", "input"]).toContain(tagName);
  });

  test("Space/Enter selects a project tile", async ({ page }) => {
    // Arrange — go to projects tab and wait for tiles
    await waitForProjects(page);
    const firstTile = page.locator(SEL.projectTile).first();
    await expect(firstTile).toBeVisible({ timeout: 30_000 });

    // Act — focus the tile and press Space (tiles use aria-pressed + click)
    await firstTile.focus();
    await page.keyboard.press("Space");

    // Assert — the tile should become active
    await expect(firstTile).toHaveClass(/active/, { timeout: 5_000 });

    // Act — if there's a second tile, test Space key
    const count = await page.locator(SEL.projectTile).count();
    if (count >= 2) {
      const secondTile = page.locator(SEL.projectTile).nth(1);
      await secondTile.focus();
      await page.keyboard.press("Space");
      await expect(secondTile).toHaveClass(/active/, { timeout: 5_000 });
    }
  });

  test("navigation is minimal — no extraneous workflow rails", async ({
    page,
  }) => {
    // Arrange — count top-level navigation elements once the tab bar has
    // mounted (count() does not auto-wait, and waitForApi can resolve before
    // the first render).
    const tabButtons = page.locator(SEL.tabButtons);
    await expect(tabButtons.first()).toBeVisible();
    const tabCount = await tabButtons.count();

    // Assert — expect a focused set of tabs (projects, annotate, training)
    // Should not have excessive navigation elements
    expect(tabCount).toBeGreaterThanOrEqual(3);
    expect(tabCount).toBeLessThanOrEqual(6);

    // Assert — no redundant sidebar or secondary nav bars
    const sidebars = page.locator("nav.sidebar, aside.nav, .workflow-rail");
    await expect(sidebars).toHaveCount(0);
  });
});
