// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SCREENSHOT_DIR = path.join(__dirname, "screenshots");

test("capture light mode projects tab", async ({ page }) => {
  await page.goto("/");
  // Click theme toggle: system -> light
  const toggle = page.getByRole("button", { name: /^Theme:/i });
  await toggle.click();
  await page.waitForTimeout(500);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "projects-light.png"),
    fullPage: true,
  });
});

test("capture dark mode projects tab", async ({ page }) => {
  await page.goto("/");
  const toggle = page.getByRole("button", { name: /^Theme:/i });
  await toggle.click(); // system -> light
  await toggle.click(); // light -> dark
  await page.waitForTimeout(500);
  await page.screenshot({
    path: path.join(SCREENSHOT_DIR, "projects-dark.png"),
    fullPage: true,
  });
});
