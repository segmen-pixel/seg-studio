#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { chromium } from "playwright-core";

const url = process.argv[2] || "http://localhost:8002/ui/";

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

  await page.goto(url, { waitUntil: "networkidle", timeout: 30_000 });
  try {
    await page.waitForFunction(() => !document.querySelector(".loading-container"), { timeout: 60_000 });
  } catch {}
  await page.waitForTimeout(300);

  // Click cookie project
  const tile = page.locator(".project-tile").filter({ hasText: /cookie/i }).first();
  await tile.click({ timeout: 5_000 });
  await page.waitForTimeout(500);

  // Click Results tab
  await page.getByRole("button", { name: /results/i }).click({ timeout: 5_000 });
  await page.waitForTimeout(500);

  // Inspect list items
  const info = await page.evaluate(() => {
    const items = document.querySelectorAll(".card.list-item-flat");
    const results = [];
    for (const item of items) {
      const cs = getComputedStyle(item);
      results.push({
        text: item.textContent?.trim().slice(0, 30),
        height: cs.height,
        display: cs.display,
        gridTemplateColumns: cs.gridTemplateColumns,
        offsetHeight: item.offsetHeight,
        children: item.children.length,
        childTags: [...item.children].map(c => `${c.tagName}.${c.className}`),
      });
    }
    // Also check parent
    const parent = items[0]?.parentElement;
    const parentCs = parent ? getComputedStyle(parent) : null;
    return {
      itemCount: items.length,
      items: results.slice(0, 3),
      parent: parent ? {
        className: parent.className,
        display: parentCs?.display,
        height: parentCs?.height,
        overflow: parentCs?.overflow,
      } : null,
    };
  });

  console.log(JSON.stringify(info, null, 2));
  await browser.close();
}

main().catch((err) => {
  console.error("Inspect failed:", err.message);
  process.exit(1);
});
