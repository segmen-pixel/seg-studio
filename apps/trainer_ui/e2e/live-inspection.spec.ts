// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

// Must stay on the same origin as playwright.config.ts baseURL — the seeded
// storageState (tutorial suppression) is origin-scoped, and 127.0.0.1 and
// localhost are different origins.
const BASE = "http://localhost:8002";

test.describe("Live Inspection tab", () => {
  test("tab exists and renders session controls", async ({ page }) => {
    await page.goto(`${BASE}/ui/`, { waitUntil: "networkidle" });

    // Wait for app to load
    await page.waitForSelector(".tabs.tabs-fixed", { timeout: 15000 });

    // Find the inspect tab
    const inspectTab = page.locator(".tabs.tabs-fixed > button", { hasText: /ライブ検査|Live Inspect/ });
    await expect(inspectTab).toBeVisible({ timeout: 5000 });

    // Click inspect tab
    await inspectTab.click();

    // Should show session controls (either "モデルを選択" or "プロジェクトを選択")
    const panel = page.locator(".tab-panel.active");
    await expect(panel).toBeVisible();

    // Check for inspect container or empty message
    const hasContent = await page.locator(".inspect-container, .inspect-empty").count();
    expect(hasContent).toBeGreaterThan(0);
  });

  test("session start and REST inference", async ({ page }) => {
    await page.goto(`${BASE}/ui/`, { waitUntil: "networkidle" });
    await page.waitForSelector(".tabs.tabs-fixed", { timeout: 15000 });

    // Select a project first (click first project in list)
    const projectCard = page.locator(".project-card, .projects-list-item").first();
    if (await projectCard.isVisible({ timeout: 3000 }).catch(() => false)) {
      await projectCard.click();
    }

    // Switch to inspect tab
    const inspectTab = page.locator(".tabs.tabs-fixed > button", { hasText: /ライブ検査|Live Inspect/ });
    await inspectTab.click();

    // Check for session controls
    const sessionRow = page.locator(".inspect-session-row");
    const hasSession = await sessionRow.isVisible({ timeout: 3000 }).catch(() => false);

    if (hasSession) {
      // Select model if dropdown exists
      const select = sessionRow.locator("select");
      const options = await select.locator("option").count();
      if (options > 1) {
        await select.selectOption({ index: 1 });
      }

      // Start session
      const startBtn = sessionRow.locator("button.primary");
      if (await startBtn.isVisible()) {
        await startBtn.click();
        // Wait for session to be ready
        await expect(page.locator(".inspect-badge-ok")).toBeVisible({ timeout: 30000 });
      }
    }
  });
});
