// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * @heavy — Large image and canvas-intensive tests.
 * These are slower and should run with limited parallelism.
 */
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectFirstProject, switchTab, SEL } from "../helpers";

test.describe("Canvas Heavy (@heavy)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
  });

  test("large image loads without timeout or crash", async ({ page }) => {
    // Arrange
    await selectFirstProject(page);
    await switchTab(page, "annotate");

    const images = page.locator(SEL.imageListItem);
    if ((await images.count()) === 0) {
      test.skip(true, "No images in project");
      return;
    }

    // Act: click first image and wait for canvas
    await images.first().click();

    // Assert: canvas loads within extended timeout
    await expect(page.locator(SEL.canvasStack)).toBeVisible({ timeout: 30_000 });
    await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout: 30_000 });
  });

  test("rapid image switching does not crash canvas", async ({ page }) => {
    // Arrange
    await selectFirstProject(page);
    await switchTab(page, "annotate");

    const images = page.locator(SEL.imageListItem);
    const count = await images.count();
    if (count < 3) {
      test.skip(true, "Need at least 3 images for rapid switching");
      return;
    }

    // Act: rapidly click through images
    for (let i = 0; i < Math.min(count, 5); i++) {
      await images.nth(i).click();
    }

    // Assert: canvas is still functional (no crash)
    await expect(page.locator(SEL.canvasStack)).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout: 15_000 });
  });
});
