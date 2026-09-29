// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.join(__dirname, "screenshots");

// Common 4:3 resolutions
const VIEWPORTS = [
  { label: "1024x768", width: 1024, height: 768 },
  { label: "1280x960", width: 1280, height: 960 },
];

const TABS = ["projects", "annotate", "training", "results"] as const;

for (const vp of VIEWPORTS) {
  test.describe(`4:3 layout @ ${vp.label}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } });

    for (const tab of TABS) {
      test(`${tab} tab`, async ({ page }) => {
        await page.goto("/");
        await page.waitForTimeout(1500);

        if (tab !== "projects") {
          const TAB_INDEX: Record<string, number> = { annotate: 1, training: 2, results: 0 };
          const idx = TAB_INDEX[tab] ?? 0;
          await page.evaluate((i) => {
            const btns = document.querySelectorAll('.tabs-fixed button');
            if (btns[i]) (btns[i] as HTMLElement).click();
          }, idx);
          await page.waitForTimeout(800);
        }

        // Measure overflow
        const overflow = await page.evaluate(() => {
          const panel = document.querySelector(".tab-panel.active > div");
          if (!panel) return null;
          return {
            scrollW: panel.scrollWidth,
            clientW: panel.clientWidth,
            scrollH: panel.scrollHeight,
            clientH: panel.clientHeight,
            hClip: panel.scrollWidth > panel.clientWidth + 2,
            vClip: panel.scrollHeight > panel.clientHeight + 2,
          };
        });
        console.log(`[${vp.label}] ${tab}:`, JSON.stringify(overflow));

        // Check for horizontal overflow on body (content wider than viewport)
        const bodyOverflow = await page.evaluate(() => {
          return {
            bodyScrollW: document.body.scrollWidth,
            windowW: window.innerWidth,
            hOverflow: document.body.scrollWidth > window.innerWidth + 2,
          };
        });
        console.log(`[${vp.label}] ${tab} body:`, JSON.stringify(bodyOverflow));

        await page.screenshot({
          path: path.join(SCREENSHOT_DIR, `4x3-${vp.label}-${tab}.png`),
        });
      });
    }
  });
}
