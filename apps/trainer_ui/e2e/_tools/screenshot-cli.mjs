#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Debug screenshot tool — capture the running app UI for debugging.
 *
 * Usage:
 *   node e2e/screenshot-cli.mjs [--project NAME] [--tab TAB] [--url URL]
 *
 * Examples:
 *   node e2e/screenshot-cli.mjs                          # Projects tab
 *   node e2e/screenshot-cli.mjs --project cookie --tab results
 *   node e2e/screenshot-cli.mjs --tab training
 *
 * Default URL: http://localhost:8002/ui/
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
  const opts = { url: "http://localhost:8002/ui/", project: null, tab: null };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--url" && args[i + 1]) opts.url = args[++i];
    else if (args[i] === "--project" && args[i + 1]) opts.project = args[++i];
    else if (args[i] === "--tab" && args[i + 1]) opts.tab = args[++i];
    else if (!args[i].startsWith("--")) opts.url = args[i];
  }
  return opts;
}

async function main() {
  const opts = parseArgs();
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

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
      const tile = page.locator(".project-tile").filter({ hasText: new RegExp(opts.project, "i") }).first();
      await tile.click({ timeout: 5_000 });
      await page.waitForTimeout(500);
    }

    // Click tab if specified
    if (opts.tab) {
      const tabBtn = page.getByRole("button", { name: new RegExp(opts.tab, "i") });
      await tabBtn.click({ timeout: 5_000 });
      await page.waitForTimeout(500);
    }

    await page.screenshot({ path: OUTPUT_FILE, fullPage: true });
    console.log(OUTPUT_FILE);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error("Screenshot failed:", err.message);
  process.exit(1);
});
