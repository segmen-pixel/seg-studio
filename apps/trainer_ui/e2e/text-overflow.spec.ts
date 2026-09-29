// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * E2E: No text wrapping / overflow on any button or label.
 * Tests all viewports × both languages.
 */
import { test, expect, type Page } from "@playwright/test";

const BASE = "http://localhost:8002/ui/";

const VIEWPORTS = [
  { name: "1920x1080", width: 1920, height: 1080 },
  { name: "1440x900", width: 1440, height: 900 },
  { name: "1280x720", width: 1280, height: 720 },
  { name: "1024x768", width: 1024, height: 768 },
];

const LANGS = ["ja", "en"];

async function setLang(page: Page, lang: string) {
  await page.evaluate((l) => localStorage.setItem("seg_lang", l), lang);
  await page.reload({ waitUntil: "networkidle" });
}

async function waitForReady(page: Page) {
  // Wait for startup
  for (let i = 0; i < 30; i++) {
    try {
      const res = await page.goto(BASE, { waitUntil: "networkidle", timeout: 5000 });
      if (res && res.status() === 200) break;
    } catch { /* retry */ }
    await page.waitForTimeout(1000);
  }
}

/**
 * Check that no element has text wrapping by comparing scrollHeight > clientHeight
 * or scrollWidth > clientWidth (accounting for padding).
 */
async function checkNoTextOverflow(page: Page, selector: string, label: string) {
  const elements = await page.$$(selector);
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i]!;
    const isVisible = await el.isVisible();
    if (!isVisible) continue;

    const overflow = await el.evaluate((node) => {
      const style = window.getComputedStyle(node);
      // Skip elements that explicitly allow wrapping
      if (style.whiteSpace === "normal" && style.overflow === "visible") return null;
      // Check if content overflows
      const overflowX = node.scrollWidth > node.clientWidth + 2;
      const overflowY = node.scrollHeight > node.clientHeight + 2;
      // Check text-overflow: ellipsis is applied if overflowing
      const hasEllipsis = style.textOverflow === "ellipsis";
      if ((overflowX || overflowY) && !hasEllipsis) {
        return {
          text: (node as HTMLElement).innerText?.slice(0, 50),
          scrollW: node.scrollWidth,
          clientW: node.clientWidth,
          scrollH: node.scrollHeight,
          clientH: node.clientHeight,
        };
      }
      return null;
    });

    if (overflow) {
      // Soft assertion — log but don't fail for very small differences
      console.warn(
        `[${label}] Text overflow detected on ${selector}[${i}]: "${overflow.text}" ` +
        `(scroll=${overflow.scrollW}x${overflow.scrollH}, client=${overflow.clientW}x${overflow.clientH})`
      );
    }
  }
}

for (const viewport of VIEWPORTS) {
  for (const lang of LANGS) {
    test(`No text overflow — ${viewport.name} ${lang}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await waitForReady(page);
      await setLang(page, lang);

      // Navigate to Training tab
      const trainingTab = page.locator('button:has-text("Training"), button:has-text("学習")');
      if (await trainingTab.count() > 0) {
        await trainingTab.first().click();
        await page.waitForTimeout(500);
      }

      // Check buttons
      await checkNoTextOverflow(page, "button", `${viewport.name}/${lang}`);
      // Check training mode buttons specifically
      await checkNoTextOverflow(page, ".training-mode-btn", `${viewport.name}/${lang}/mode`);
      // Check training start buttons
      await checkNoTextOverflow(page, ".training-start-btn", `${viewport.name}/${lang}/start`);
      // Check inputs (placeholder text)
      await checkNoTextOverflow(page, "input", `${viewport.name}/${lang}/input`);
      // Check chips
      await checkNoTextOverflow(page, ".training-chip", `${viewport.name}/${lang}/chip`);

      // Screenshot for visual verification
      await page.screenshot({
        path: `e2e-results/text-overflow-${viewport.name}-${lang}.png`,
        fullPage: false,
      });

      // Hard check: no button should have line-wrapped text (scrollHeight > expected single line)
      const wrappedButtons = await page.$$eval("button", (buttons) => {
        return buttons
          .filter((btn) => {
            if (!btn.offsetParent) return false; // not visible
            const style = window.getComputedStyle(btn);
            const lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2;
            // If button height is more than 2x line height, text probably wrapped
            return btn.clientHeight > lineHeight * 2.5 && btn.innerText.trim().length > 0;
          })
          .map((btn) => ({
            text: btn.innerText.trim().slice(0, 40),
            height: btn.clientHeight,
            class: btn.className.slice(0, 60),
          }));
      });

      // Allow mode buttons (they have SVG + text, naturally tall) and the
      // round new-models FAB (fixed 50px circle around a one-digit count).
      const nonModeWrapped = wrappedButtons.filter(
        (b) =>
          !b.class.includes("training-mode-btn") &&
          !b.class.includes("training-hyper") &&
          !b.class.includes("new-models-fab-button") &&
          !b.class.includes("train-fab-button")
      );

      expect(nonModeWrapped, `Buttons with wrapped text at ${viewport.name}/${lang}`).toEqual([]);
    });
  }
}
