// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

/**
 * Description-mode tooltip E2E tests.
 * Verifies that hovering data-desc elements in desc-mode shows visible,
 * non-clipped tooltips in the correct direction.
 */

test.describe("Description mode tooltips", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    // Wait for API connection
    await expect(page.locator(".project-grid, .api-connecting-banner")).toBeVisible({ timeout: 15_000 });
    // Enable desc mode by clicking the ? button
    const descToggle = page.locator('button[title="説明モード"]');
    await descToggle.click();
    // Verify desc-mode class is applied
    // desc-mode is applied to several containers now; assert the app shell
    await expect(page.locator(".app-shell.desc-mode")).toBeVisible();
  });

  test("header buttons show tooltips below (not clipped by top)", async ({ page }) => {
    const headerButtons = page.locator('.header-actions [data-desc]');
    const count = await headerButtons.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < Math.min(count, 3); i++) {
      const btn = headerButtons.nth(i);
      const descPos = await btn.getAttribute("data-desc-pos");
      expect(descPos).toBe("bottom");

      // Park the mouse away first: the tooltip listens for mouseover, which
      // only fires on an element boundary cross (the beforeEach click leaves
      // the pointer sitting on the first header button).
      await page.mouse.move(0, 0);
      await btn.hover();
      await page.waitForTimeout(100);
      // Tooltips are a body-level .desc-tooltip element (position: fixed),
      // placed below the hovered element when data-desc-pos="bottom".
      const tip = page.locator(".desc-tooltip.visible");
      await expect(tip).toBeVisible();
      const btnBox = (await btn.boundingBox())!;
      const tipBox = (await tip.boundingBox())!;
      expect(tipBox.y, "tooltip should render below the header button").toBeGreaterThanOrEqual(btnBox.y + btnBox.height);
      expect(tipBox.y, "tooltip should not be clipped by the top edge").toBeGreaterThanOrEqual(0);
    }
  });

  test("sidebar toolbar buttons show tooltips below", async ({ page }) => {
    await page.locator(".tabs-fixed button").nth(1).click();
    await page.waitForTimeout(500);

    const sidebarBtns = page.locator('.sidebar-toolbar [data-desc]');
    const count = await sidebarBtns.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const btn = sidebarBtns.nth(i);
      const descPos = await btn.getAttribute("data-desc-pos");
      expect(descPos).toBe("bottom");
    }
  });

  test("project tile action buttons show tooltips below", async ({ page }) => {
    const tiles = page.locator(".project-tile-actions [data-desc]");
    await tiles.first().waitFor({ state: "visible", timeout: 15_000 }).catch(() => {});
    const count = await tiles.count();
    if (count === 0) {
      test.skip(true, "No projects available for testing");
      return;
    }

    for (let i = 0; i < Math.min(count, 4); i++) {
      const btn = tiles.nth(i);
      const descPos = await btn.getAttribute("data-desc-pos");
      expect(descPos).toBe("bottom");
    }
  });

  test("all data-desc elements near top edge use bottom position", async ({ page }) => {
    const allDesc = page.locator("[data-desc]:visible");
    const count = await allDesc.count();

    for (let i = 0; i < count; i++) {
      const el = allDesc.nth(i);
      const box = await el.boundingBox();
      if (!box) continue;

      if (box.y < 80) {
        const pos = await el.getAttribute("data-desc-pos");
        expect.soft(pos, `Element at y=${box.y} should have bottom tooltip`).toBe("bottom");
      }
    }
  });

  test("tooltip not clipped: hover each visible data-desc and check the tooltip is within viewport", async ({ page }) => {
    // Collect all visible data-desc elements on Projects tab
    const allDesc = page.locator("[data-desc]:visible");
    const count = await allDesc.count();
    expect(count).toBeGreaterThan(0);

    const viewport = page.viewportSize()!;
    const tip = page.locator(".desc-tooltip.visible");
    await page.mouse.move(0, 0); // ensure the first hover crosses a boundary

    // The JS tooltip clamps every placement the same way, so a representative
    // sample keeps the runtime bounded (hovering 100+ tile buttons timed out).
    const SAMPLE = Math.min(count, 30);
    for (let i = 0; i < SAMPLE; i++) {
      const el = allDesc.nth(i);
      const box = await el.boundingBox();
      if (!box) continue;
      const desc = await el.getAttribute("data-desc");
      if (!desc) continue; // empty desc renders no tooltip

      await el.hover();
      await page.waitForTimeout(50); // instant tooltip; small settle for layout

      // The body-level .desc-tooltip clamps itself to the viewport in JS —
      // measure the real element instead of estimating a pseudo-element.
      const tipBox = await tip.boundingBox();
      expect.soft(tipBox, `"${desc}" should show a tooltip`).not.toBeNull();
      if (!tipBox) continue;

      const margin = 2; // allow tiny rounding

      expect.soft(
        tipBox.y >= -margin,
        `"${desc}" tooltip top (${tipBox.y.toFixed(0)}) should not be above viewport`
      ).toBeTruthy();

      expect.soft(
        tipBox.x >= -margin,
        `"${desc}" tooltip left (${tipBox.x.toFixed(0)}) should not be left of viewport`
      ).toBeTruthy();

      expect.soft(
        tipBox.y + tipBox.height <= viewport.height + margin,
        `"${desc}" tooltip bottom (${(tipBox.y + tipBox.height).toFixed(0)}) should not exceed viewport height (${viewport.height})`
      ).toBeTruthy();

      expect.soft(
        tipBox.x + tipBox.width <= viewport.width + margin,
        `"${desc}" tooltip right (${(tipBox.x + tipBox.width).toFixed(0)}) should not exceed viewport width (${viewport.width})`
      ).toBeTruthy();
    }
  });

  test("tooltip not overlapping other text: z-index check", async ({ page }) => {
    // Switch to Annotate tab where sidebar has image list below toolbar
    await page.locator(".tabs-fixed button").nth(1).click();
    await page.waitForTimeout(500);

    const sidebarBtns = page.locator('.sidebar-toolbar [data-desc]');
    const count = await sidebarBtns.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const btn = sidebarBtns.nth(i);
      await btn.hover();
      await page.waitForTimeout(100);

      // The tooltip is a body-level element: position fixed (so ancestor
      // overflow can never clip it) with a z-index above every panel.
      const tipStyle = await page.evaluate(() => {
        const tip = document.querySelector(".desc-tooltip.visible");
        if (!tip) return null;
        const s = getComputedStyle(tip);
        return { position: s.position, zIndex: parseInt(s.zIndex || "0", 10), parent: tip.parentElement?.tagName };
      });
      expect.soft(tipStyle, `Sidebar button #${i} should show a tooltip`).not.toBeNull();
      if (!tipStyle) continue;
      expect.soft(tipStyle.position, "tooltip must be fixed-positioned (unclippable)").toBe("fixed");
      expect.soft(tipStyle.zIndex, "tooltip z-index should be >= 9999").toBeGreaterThanOrEqual(9999);
      expect.soft(tipStyle.parent, "tooltip must be attached to body").toBe("BODY");
    }
  });

  test("Annotate tab tool buttons have tooltips", async ({ page }) => {
    await page.locator(".tabs-fixed button").nth(1).click();
    await page.waitForTimeout(500);

    const toolBtns = page.locator('.overlay-tools [data-desc]');
    const count = await toolBtns.count();
    expect(count).toBeGreaterThan(3);
  });

  test("Training tab buttons have tooltips", async ({ page }) => {
    await page.locator(".tabs-fixed button").nth(2).click();
    await page.waitForTimeout(500);

    const trainDesc = page.locator('.tab-panel.active [data-desc]');
    const count = await trainDesc.count();
    expect(count).toBeGreaterThan(0);
  });
});
