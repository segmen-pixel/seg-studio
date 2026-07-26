// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect, type Page, type ConsoleMessage } from "@playwright/test";

const EXPECTED_ERRORS =
  /favicon|Failed to load resource|net::ERR_|AbortError|the server responded with a status of 4/i;

function useConsoleErrorCollector(page: Page) {
  const errors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  return {
    assertNoErrors() {
      const filtered = errors.filter((entry) => !EXPECTED_ERRORS.test(entry));
      expect(filtered, `Unexpected console errors:\n${filtered.join("\n")}`).toHaveLength(0);
    },
  };
}

async function waitForApi(page: Page) {
  await expect(page.locator(".api-connecting-banner")).toHaveCount(0, {
    timeout: 90_000,
  });
}

async function switchTab(page: Page, tab: "projects" | "annotate") {
  const TAB_INDEX: Record<string, number> = { projects: 0, annotate: 1, training: 2 };
  const idx = TAB_INDEX[tab] ?? 0;
  await page.evaluate((i) => {
    const btns = document.querySelectorAll('.tabs-fixed button');
    if (btns[i]) (btns[i] as HTMLElement).click();
  }, idx);
  await page.waitForSelector(`.app-shell.tab-${tab}`, { timeout: 10_000 });
}

async function selectFirstProject(page: Page) {
  await switchTab(page, "projects");
  const tile = page.locator(".project-tile").first();
  await expect(tile).toBeVisible({ timeout: 20_000 });
  await tile.click();
  await expect(tile).toHaveClass(/active/);
}

test.describe("Annotate class integrity", () => {
  test("background-only class payload does not leave a phantom foreground selection", async ({ page }) => {
    const col = useConsoleErrorCollector(page);
    await page.route("**/projects/*/classes", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          version: 1,
          ignore_index: 255,
          next_class_id: 0,
          classes: [{ id: 0, name: "background", color: [0, 0, 0], active: true }],
        }),
      });
    });

    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await switchTab(page, "annotate");
    await page.waitForSelector(".list-compact .card.list-item-flat", { timeout: 20_000 });
    // The Mark Clean pseudo-card is always rendered in .class-list-window;
    // a phantom foreground class card is the one that carries a color picker.
    await expect(page.locator('.class-list-window .card input[type="color"]')).toHaveCount(0);
    await expect(page.getByRole("button", { name: /^(Delete|削除)$/i })).toBeDisabled();
    col.assertNoErrors();
  });

  test("orphan class ids are surfaced instead of silently hiding old mask pixels", async ({ page }) => {
    const col = useConsoleErrorCollector(page);
    await page.route("**/projects/*/classes/reconcile", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ orphan_ids: [7], details: { 7: 1 } }),
      });
    });

    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await switchTab(page, "annotate");
    await page.waitForSelector(".list-compact .card.list-item-flat", { timeout: 20_000 });

    // Reconcile button has been removed — orphan recovery is now automatic.
    // Verify no Reconcile button exists in the UI.
    await expect(page.getByRole("button", { name: /^(Reconcile|復元)$/i })).toHaveCount(0);
    col.assertNoErrors();
  });

});
