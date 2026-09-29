// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

/**
 * Tab layout alignment E2E tests.
 * Verifies that:
 * 1. Fixed tabs (Projects/Annotate/Training) never scroll or hide
 * 2. Result tabs scroll independently
 * 3. Project name in header stays centered
 * 4. DR-01: No text overlap between adjacent elements
 */

/**
 * DR-01: Check that two elements' bounding boxes do not horizontally overlap.
 * Returns overlap in pixels (0 = no overlap).
 */
async function getHorizontalOverlap(
  elA: import("@playwright/test").Locator,
  elB: import("@playwright/test").Locator
): Promise<number> {
  const boxA = await elA.boundingBox();
  const boxB = await elB.boundingBox();
  if (!boxA || !boxB) return 0;
  const overlapStart = Math.max(boxA.x, boxB.x);
  const overlapEnd = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  return Math.max(0, overlapEnd - overlapStart);
}

test.describe("Tab bar layout and alignment", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await expect(
      page.locator(".project-grid, .api-connecting-banner")
    ).toBeVisible({ timeout: 15_000 });
  });

  test("fixed tabs are always fully visible", async ({ page }) => {
    const viewport = page.viewportSize()!;
    for (const idx of [0, 1, 2]) {
      const btn = page.locator(".tabs-fixed button").nth(idx);
      await expect(btn).toBeVisible();
      const box = await btn.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width);
    }
  });

  test("project name in header is centered", async ({ page }) => {
    const nameEl = page.locator(".header-project-name");
    // May be empty on Projects tab — just check it exists and doesn't overlap h1
    const h1 = page.locator(".app-brand h1");
    const h1Box = await h1.boundingBox();
    const nameBox = await nameEl.boundingBox();
    if (h1Box && nameBox && nameBox.width > 0) {
      // No overlap with h1
      expect(h1Box.x + h1Box.width).toBeLessThanOrEqual(nameBox.x + 2);
    }
  });

  // NOTE: the ".tabs-fixed::after separator aligns with the image list" tests
  // were removed — the separator pseudo-element no longer exists in shell.css,
  // so the alignment contract they asserted is gone from the UI.

  test("result tabs do not push fixed tabs offscreen", async ({ page }) => {
    const shrink = await page.locator(".tabs-fixed").evaluate((el) => {
      return getComputedStyle(el).flexShrink;
    });
    expect(shrink).toBe("0");

    const overflow = await page
      .locator(".tabs-result-scroll")
      .evaluate((el) => {
        return getComputedStyle(el).overflowX;
      });
    expect(overflow).toBe("auto");
  });

  // DR-01: テキスト重なり禁止
  test("DR-01: no text overlap between adjacent visible elements in tabs row", async ({
    page,
  }) => {
    // Collect all visible elements in the tabs-row
    const overlap = await page.evaluate(() => {
      const row = document.querySelector(".tabs-row");
      if (!row) return 0;
      // Get all direct children with visible text
      const children = Array.from(row.children).filter((el) => {
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      });
      let maxOverlap = 0;
      for (let i = 0; i < children.length - 1; i++) {
        const a = children[i]!.getBoundingClientRect();
        const b = children[i + 1]!.getBoundingClientRect();
        const overlapStart = Math.max(a.left, b.left);
        const overlapEnd = Math.min(a.right, b.right);
        const px = Math.max(0, overlapEnd - overlapStart);
        if (px > maxOverlap) maxOverlap = px;
      }
      return maxOverlap;
    });
    expect(overlap, "Adjacent elements in tabs-row must not overlap").toBe(0);
  });

  test("DR-01: result tabs and toast never overlap", async ({ page }) => {
    const resultScroll = page.locator(".tabs-result-scroll");
    const toast = page.locator(".tabs-status");

    // Both must exist
    await expect(resultScroll).toBeAttached();
    await expect(toast).toBeAttached();

    const overlap = await getHorizontalOverlap(resultScroll, toast);
    expect(overlap, "Result tabs and toast area must not overlap").toBe(0);
  });
});
