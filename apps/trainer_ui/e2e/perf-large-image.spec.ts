// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

test.setTimeout(90_000);

test("large image: brush mark → switch → verify no false OK", async ({ page }) => {
  const logs: string[] = [];
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("[DEBUG]") || text.includes("[PERF]")) {
      logs.push(text);
      console.log(text);
    }
  });

  await page.goto("/ui/");
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(1000);

  const tile = page.locator(".project-tile").first();
  if (await tile.count() === 0) { test.skip(); return; }
  await tile.first().click();
  await page.waitForTimeout(500);

  // Open Annotate tab
  await page.locator(".tabs-fixed button").nth(1).click();
  await expect(page.locator(".app-shell.tab-annotate")).toBeVisible({ timeout: 8000 });
  await page.waitForTimeout(2000);

  // Select first image
  const items = page.locator(".image-list-dropzone .card.list-item-flat");
  await expect.poll(() => items.count(), { timeout: 20000 }).toBeGreaterThan(0);
  await items.first().click();
  await page.waitForTimeout(3000);

  // Get first image ID
  const firstId = await page.evaluate(() => {
    const active = document.querySelector(".image-list-dropzone .card.list-item-flat.active");
    return active?.getAttribute("data-id") || active?.textContent || "unknown";
  });
  console.log("First image:", firstId);

  // Check: no OK badge on first image initially
  const firstBadgeBefore = await items.first().locator(".clean-badge").count();
  console.log("OK badge before marking:", firstBadgeBefore);

  // Brush stroke on first image
  await page.keyboard.press("B");
  await page.waitForTimeout(300);
  const box = await page.locator(".canvas-stack").boundingBox();
  if (!box) { console.log("No canvas"); return; }

  // Draw a thick stroke
  for (let row = 0; row < 5; row++) {
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.4 + row * 3);
    await page.mouse.down();
    for (let i = 0; i < 20; i++) {
      await page.mouse.move(
        box.x + box.width * 0.3 + i * 5,
        box.y + box.height * 0.4 + row * 3
      );
    }
    await page.mouse.up();
    await page.waitForTimeout(50);
  }
  await page.waitForTimeout(500);
  console.log("Brush strokes done");

  // Press ArrowDown to switch to next image
  await page.keyboard.press("ArrowDown");
  // Wait for debounce + load + autoSave
  await page.waitForTimeout(6000);

  // Check: does first image have OK badge? (IT SHOULD NOT)
  const firstBadgeAfter = await items.first().locator(".clean-badge").count();
  console.log("OK badge after switch:", firstBadgeAfter);

  // Also check via API
  const apiResult = await page.evaluate(async () => {
    const res = await fetch("/projects/" + (window as any).__currentProjectId + "/datasets/annotate");
    if (!res.ok) return null;
    const data = await res.json();
    const first = data.items?.[0];
    return first ? {
      id: first.id,
      name: first.name,
      hasMask: first.annotation?.hasMask,
      hasForeground: first.annotation?.hasForeground,
    } : null;
  });
  console.log("API annotation state:", JSON.stringify(apiResult));

  // Take screenshot
  await page.screenshot({ path: "e2e/test-results/ok-badge-test.png" });

  // THE CRITICAL ASSERTION
  expect(firstBadgeAfter, "First image should NOT have OK badge after brush marking").toBe(0);
});
