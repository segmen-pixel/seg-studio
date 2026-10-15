// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.join(__dirname, "screenshots");

const TABS = ["projects", "annotate", "training", "results"] as const;
const TAB_INDEX: Record<string, number> = { projects: 0, annotate: 1, training: 2, results: 0 };

test.describe("Debug screenshots", () => {
  for (const tab of TABS) {
    test(`capture ${tab} tab`, async ({ page }) => {
      await page.goto("/");
      // Click the tab button by index
      const idx = TAB_INDEX[tab] ?? 0;
      await page.evaluate((i) => {
        const btns = document.querySelectorAll('.tabs-fixed button');
        if (btns[i]) (btns[i] as HTMLElement).click();
      }, idx);
      // Wait for any async rendering
      await page.waitForTimeout(500);
      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, `${tab}.png`),
        fullPage: true,
      });
    });
  }
});
