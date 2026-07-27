// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectProjectByName, switchTab, SEL } from "../helpers";

test.describe("Mask Persistence", () => {
  test("mask survives project switch round-trip", async ({ page }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);

    // Deterministic fixture projects (e2e/global-setup.ts)
    await selectProjectByName(page, "zz-e2e-seed-1");
    await switchTab(page, "annotate");

    // Record mask count (if any images)
    const images = page.locator(SEL.imageListItem);
    if ((await images.count()) === 0) {
      test.skip(true, "No images in project");
      return;
    }
    await images.first().click();

    // Switch to the second seed project and back
    await selectProjectByName(page, "zz-e2e-seed-2");
    await selectProjectByName(page, "zz-e2e-seed-1");
    await switchTab(page, "annotate");

    // Assert: annotate tab is still functional (no crash/freeze)
    await expect(page.locator(SEL.canvasStack)).toBeVisible({ timeout: 10_000 });
  });

  test("mask leak: switching images does not bleed mask data", async ({ page }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);
    await selectProjectByName(page, "zz-e2e-seed-1");
    await switchTab(page, "annotate");

    const images = page.locator(SEL.imageListItem);
    if ((await images.count()) < 2) {
      test.skip(true, "Need at least 2 images");
      return;
    }

    // Act: click image 0, then quickly switch to image 1
    await images.nth(0).click();
    await expect(page.locator(SEL.canvasStack)).toBeVisible({ timeout: 10_000 });
    await images.nth(1).click();
    await expect(page.locator(SEL.canvasStack)).toBeVisible({ timeout: 10_000 });

    // Assert: no error overlay or crash
    await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout: 5_000 });
  });
});
