// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * EN-mode leak sweep — walks the main tabs with the language forced to
 * English and asserts that no Japanese characters appear in UI chrome
 * (buttons, labels, headings, placeholders, titles, tooltips, aria).
 *
 * Guards the whole class of "hardcoded Japanese bypasses the i18n
 * dictionaries" bugs (25 instances fixed in the 2026-07 pre-OSS audit).
 * User content (project tiles, image lists, logs, side panels) is excluded
 * because real data may legitimately contain Japanese.
 */
import type { Page } from "@playwright/test";
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, switchTab, selectProjectByName, SEL } from "../helpers";

const SWEEP_SELECTOR = [
  "button",
  "label",
  "h1",
  "h2",
  "h3",
  "h4",
  "th",
  "[placeholder]",
  "[title]",
  "[data-desc]",
  "[aria-label]",
].join(",");

// User-content containers where Japanese is legitimate (project names,
// image file names, training logs, results lists).
const EXCLUDE_CLOSEST = [
  ".project-tile",
  ".image-list-dropzone",
  ".side-panel",
  "[class*='log']",
].join(",");

async function sweepJapanese(page: Page): Promise<string[]> {
  return page.evaluate(
    ({ sel, exclude }) => {
      const jp = /[\u3040-\u30ff\u3400-\u9fff]/;
      const hits = new Set<string>();
      for (const el of Array.from(document.querySelectorAll<HTMLElement>(sel))) {
        if (el.closest(exclude)) continue;
        const candidates = [
          el.textContent ?? "",
          el.getAttribute("placeholder") ?? "",
          el.getAttribute("title") ?? "",
          el.getAttribute("data-desc") ?? "",
          el.getAttribute("aria-label") ?? "",
        ];
        for (const raw of candidates) {
          const text = raw.trim();
          if (text && jp.test(text)) {
            const cls = (el.className || "").toString().split(" ")[0];
            hits.add(`<${el.tagName.toLowerCase()}${cls ? "." + cls : ""}> ${text.slice(0, 60)}`);
            break;
          }
        }
      }
      return Array.from(hits);
    },
    { sel: SWEEP_SELECTOR, exclude: EXCLUDE_CLOSEST },
  );
}

test.describe("EN mode leak sweep", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("seg-lang", "en"));
    await page.goto("/");
    await waitForApi(page);
  });

  test("no Japanese characters in UI chrome across tabs", async ({ page }) => {
    await selectProjectByName(page, "zz-e2e-seed-1");

    const leaks: string[] = [];
    for (const tab of ["projects", "annotate", "training"] as const) {
      await switchTab(page, tab);
      await page.waitForTimeout(500);
      for (const hit of await sweepJapanese(page)) leaks.push(`[${tab}] ${hit}`);
    }

    // Dynamic results tab only exists when the current project has a
    // completed run; sweep it when present (switchTab has no index for it).
    const resultTab = page.locator(SEL.tabResult).first();
    if (await resultTab.isVisible({ timeout: 1_000 }).catch(() => false)) {
      await resultTab.click();
      await page.waitForTimeout(500);
      for (const hit of await sweepJapanese(page)) leaks.push(`[results] ${hit}`);
    }

    expect(leaks, `Japanese leaked into EN mode:\n${leaks.join("\n")}`).toEqual([]);
  });
});
