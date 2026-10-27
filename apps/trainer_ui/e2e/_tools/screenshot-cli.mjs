#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Debug screenshot tool — capture the running app UI for debugging.
 *
 * Usage:
 *   node e2e/screenshot-cli.mjs [--project NAME] [--tab TAB] [--lang en|ja] [--viewport WxH] [--scale N]
 *                              [--click SELECTOR]... [--out FILE] [--url URL]
 *
 * Examples:
 *   node e2e/screenshot-cli.mjs                          # Projects tab
 *   node e2e/screenshot-cli.mjs --project cookie --tab results
 *   node e2e/screenshot-cli.mjs --tab training
 *
 * Default URL: http://localhost:8002/ui/
 *
 * The doc images were made with (project: the bundled tutorial sample, Score sub-tab open):
 *   --project "<project name>" --tab training --lang ja --viewport 1440x900 --scale 2 \
 *     --click "button:has-text('スコア')" --out ../../../docs/ja/assets/handbook/10b_training_form.png
 *   ... --click .training-mode-btn --click "button:has-text('スコア')" --out .../catalog_training.png
 *   ... --lang en --viewport 1400x875 --scale 1 --click "button:has-text('Score')" --out ../../../docs/images/screenshot_training.png
 * Output: e2e/screenshots/debug-latest.png
 */
import { chromium } from "playwright-core";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_DIR = path.join(__dirname, "screenshots");
const OUTPUT_FILE = path.join(SCREENSHOT_DIR, "debug-latest.png");

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { url: "http://localhost:8002/ui/", project: null, tab: null, lang: null, viewport: null, scale: 1, click: [], out: null };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--url" && args[i + 1]) opts.url = args[++i];
    else if (args[i] === "--project" && args[i + 1]) opts.project = args[++i];
    else if (args[i] === "--tab" && args[i + 1]) opts.tab = args[++i];
    else if (args[i] === "--lang" && args[i + 1]) opts.lang = args[++i];
    else if (args[i] === "--viewport" && args[i + 1]) opts.viewport = args[++i];
    else if (args[i] === "--scale" && args[i + 1]) opts.scale = Number(args[++i]);
    else if (args[i] === "--click" && args[i + 1]) opts.click.push(args[++i]);
    else if (args[i] === "--out" && args[i + 1]) opts.out = args[++i];
    else if (!args[i].startsWith("--")) opts.url = args[i];
  }
  return opts;
}

async function main() {
  const opts = parseArgs();
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const [vw, vh] = (opts.viewport || "1920x1080").split("x").map(Number);
  const page = await browser.newPage({ viewport: { width: vw, height: vh }, deviceScaleFactor: opts.scale || 1 });
  const outFile = opts.out ? path.resolve(opts.out) : OUTPUT_FILE;
  // Mark the first-launch tutorial as done so it never sits on top of a screenshot.
  await page.addInitScript((lang) => {
    localStorage.setItem("seg-tutorial-state", JSON.stringify({ completed: true, skipped: false, lastStep: 0, mode: "expert" }));
    if (lang) localStorage.setItem("seg-lang", lang);
  }, opts.lang);

  try {
    await page.goto(opts.url, { waitUntil: "networkidle", timeout: 30_000 });

    // Wait for loading screen to disappear
    try {
      await page.waitForFunction(
        () => !document.querySelector(".loading-container"),
        { timeout: 60_000 },
      );
    } catch {
      // Loading screen might not exist
    }
    await page.waitForTimeout(300);

    // Click project tile if specified
    if (opts.project) {
      // A plain string matches case-insensitively and survives names with ( ) or .
      const tile = page.locator(".project-tile").filter({ hasText: opts.project }).first();
      await tile.click({ timeout: 5_000 });
      await page.waitForTimeout(500);
    }

    // Click tab if specified (by its tutorial anchor, so the label language does not matter)
    if (opts.tab) {
      await page.locator(`[data-tutorial-step="${opts.tab}-tab"]`).click({ timeout: 5_000 });
      await page.waitForSelector(`.app-shell.tab-${opts.tab}`, { timeout: 10_000 });
      await page.waitForTimeout(500);
    }

    // Optional extra clicks (a mode card, a sub-tab, ...) before the capture, in order
    for (const sel of opts.click) {
      await page.locator(sel).first().click({ timeout: 5_000 });
      await page.waitForTimeout(600);
    }

    await page.screenshot({ path: outFile, fullPage: !opts.viewport });
    console.log(outFile);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error("Screenshot failed:", err.message);
  process.exit(1);
});
