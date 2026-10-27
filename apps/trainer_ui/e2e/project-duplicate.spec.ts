// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";
import { waitForApi } from "./helpers";

/**
 * Duplicating asks two things, which is why it is a dialog and not a confirm:
 * what the copy is called, and whether the labels come with it. Neither has a
 * default that is right often enough to guess, and the second is the reason
 * the feature exists -- the same pictures, labelled again a different way.
 *
 * This covers the wiring and the defaults. Making a copy for real is covered
 * by the API tests, which can clean up after themselves; doing it here would
 * leave a project behind on a failed run.
 */
test.describe("Duplicate a project", () => {
  test("offers a name and a choice about the labels", async ({ page }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);
    const tile = page.locator(".project-tile").first();
    await expect(tile).toBeVisible();
    const before = await page.locator(".project-tile").count();
    await tile.hover();

    // Act
    await tile.locator(".project-tile-actions .models-action-btn").nth(3).click();

    // Assert
    const panel = page.locator(".settings-overlay .settings-panel");
    await expect(panel).toBeVisible();
    const name = panel.locator("input[type='text'], input:not([type])").first();
    await expect(name).not.toHaveValue("");
    const labels = panel.locator("input[type='checkbox']");
    await expect(labels).toBeChecked();

    // Closing it changes nothing.
    await panel.locator(".settings-close-btn").click();
    await expect(panel).toHaveCount(0);
    await expect(page.locator(".project-tile")).toHaveCount(before);
  });
});

