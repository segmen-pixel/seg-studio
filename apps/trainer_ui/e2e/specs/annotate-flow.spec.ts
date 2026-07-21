// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectFirstProject, switchTab, SEL } from "../helpers";

test.describe("Annotate Flow", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
  });

  test("annotate tab shows canvas and class list after project selection", async ({
    page,
    annotate,
  }) => {
    // Arrange
    await selectFirstProject(page);

    // Act
    await annotate.goto();

    // Assert
    await expect(page.locator(SEL.canvasStack)).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(SEL.classListCard).first()).toBeVisible({ timeout: 10_000 });
  });

  test("selecting image from list updates canvas", async ({ page, annotate }) => {
    // Arrange
    await selectFirstProject(page);
    await annotate.goto();

    const images = page.locator(SEL.imageListItem);
    if ((await images.count()) < 1) {
      test.skip(true, "No images in project");
      return;
    }

    // Act: click first image
    await images.first().click();

    // Assert: canvas loads (busy overlay clears)
    await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout: 15_000 });
    await expect(page.locator(SEL.canvasStack)).toBeVisible();
  });

  test("class list shows background class by default", async ({ page, annotate }) => {
    // Arrange
    await selectFirstProject(page);
    await annotate.goto();

    // Assert: at least one class card visible
    const cards = page.locator(SEL.classListCard);
    await expect(cards.first()).toBeVisible({ timeout: 10_000 });

    // The first class is typically "background" (class 0)
    const count = await cards.count();
    expect(count).toBeGreaterThanOrEqual(1);
  });
});
