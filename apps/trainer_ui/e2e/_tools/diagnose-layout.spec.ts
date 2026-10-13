// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.join(__dirname, "screenshots");

test("diagnose layout cutoff issues", async ({ page }) => {
  // Use a typical desktop viewport
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto("/");
  await page.waitForTimeout(2000); // Wait for API data to load

  // Projects tab
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "diag-projects.png") });

  // Check if content is clipped (scrollHeight > clientHeight)
  const projectsOverflow = await page.evaluate(() => {
    const panel = document.querySelector(".tab-panel.active > div");
    if (!panel) return { scrollH: 0, clientH: 0, clipped: false };
    return {
      scrollH: panel.scrollHeight,
      clientH: panel.clientHeight,
      clipped: panel.scrollHeight > panel.clientHeight + 2,
    };
  });
  console.log("Projects overflow:", JSON.stringify(projectsOverflow));

  // Annotate tab
  await page.locator(".tabs-fixed button").nth(1).click();
  await page.waitForTimeout(1000);
  const activeClass1 = await page.locator(".app-shell").getAttribute("class");
  console.log("After annotate click, app-shell class:", activeClass1);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "diag-annotate.png") });

  // Training tab
  await page.locator(".tabs-fixed button").nth(2).click();
  await page.waitForTimeout(1000);
  const activeClass2 = await page.locator(".app-shell").getAttribute("class");
  console.log("After training click, app-shell class:", activeClass2);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "diag-training.png") });

  // Check training panel overflow
  const trainingOverflow = await page.evaluate(() => {
    const panel = document.querySelector(".tab-panel.active > div");
    if (!panel) return { scrollH: 0, clientH: 0, clipped: false };
    return {
      scrollH: panel.scrollHeight,
      clientH: panel.clientHeight,
      clipped: panel.scrollHeight > panel.clientHeight + 2,
    };
  });
  console.log("Training overflow:", JSON.stringify(trainingOverflow));

  // Results tab — dynamic tab, just capture current state
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "diag-results.png") });

  // Go back to projects and scroll down to check if licenses section is clipped
  await page.locator(".tabs-fixed button").nth(0).click();
  await page.waitForTimeout(500);

  // Try to scroll the inner panel to the bottom
  await page.evaluate(() => {
    const panel = document.querySelector(".tab-panel.active > div");
    if (panel) panel.scrollTop = panel.scrollHeight;
  });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "diag-projects-scrolled.png") });

  // Check if Licenses section is visible after scroll
  const licensesVisible = await page.evaluate(() => {
    const el = document.querySelector(".licenses");
    if (!el) return { exists: false, visible: false, rect: null };
    const rect = el.getBoundingClientRect();
    return {
      exists: true,
      visible: rect.bottom > 0 && rect.top < window.innerHeight,
      rect: { top: rect.top, bottom: rect.bottom, height: rect.height },
    };
  });
  console.log("Licenses section:", JSON.stringify(licensesVisible));
});
