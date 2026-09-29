// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, selectProjectByName, switchTab, SEL } from "../helpers";

test.describe("Report Flow", () => {
  test.beforeEach(async ({ page }) => {
    // Skip the first-launch tutorial overlay (it intercepts pointer events).
    await page.addInitScript(() => {
      localStorage.setItem(
        "seg-tutorial-state",
        JSON.stringify({ completed: true, skipped: true, lastStep: 0, mode: null }),
      );
    });
    await page.goto("/");
    await waitForApi(page);
  });

  test("generate report opens an embedded, same-origin-framable report tab", async ({ page }) => {
    test.setTimeout(180_000); // server-side report rendering can take a while
    // Trained fixture project (completed runs) — deterministic regardless of tile order
    await selectProjectByName(page, "zz-e2e-seed-2");

    // Need a results view (requires a completed training run). Reuse an open
    // result tab if present, otherwise open one from the training run list.
    let resultTab = page.locator(SEL.tabResult).first();
    if (!(await resultTab.isVisible({ timeout: 3_000 }).catch(() => false))) {
      await switchTab(page, "training");
      const openResults = page.getByRole("button", { name: /結果を見る|Results/ }).first();
      if (!(await openResults.isVisible({ timeout: 3_000 }).catch(() => false))) {
        test.skip(true, "No completed training run — cannot generate a report");
        return;
      }
      await openResults.click();
      resultTab = page.locator(SEL.tabResult).first();
    } else {
      await resultTab.click();
    }
    await expect(page.locator(SEL.resultsLayout)).toBeVisible({ timeout: 10_000 });

    // Open the report modal from the results header.
    // Results header uses the short label (results.reportShort); the long
    // "レポート生成" label lives on the training run list.
    const reportBtn = page.getByRole("button", { name: /レポート|Report/i }).first();
    await expect(reportBtn).toBeVisible({ timeout: 10_000 });
    await reportBtn.click();

    // Keep generation fast/reliable: drop the heaviest section (per-image hard cases).
    const hardCase = page.locator('label:has-text("Hard Case") input[type="checkbox"]').first();
    if (await hardCase.isChecked().catch(() => false)) {
      await hardCase.uncheck();
    }

    // Generate & preview (HTML only).
    const genBtn = page.getByRole("button", { name: /生成してプレビュー|Generate & Preview/ });
    await expect(genBtn).toBeVisible();
    await genBtn.click();

    // Report generation renders charts server-side, then the in-app report
    // tab + viewer iframe appear. Wait on the iframe directly with a generous budget.
    const iframe = page.getByTestId("report-iframe");
    await expect(iframe).toBeVisible({ timeout: 120_000 });

    // A dedicated report tab button appears next to the result tab. Use
    // `button.tab-report` so it doesn't also match the app-shell's tab-report class.
    await expect(page.locator("button.tab-report").first()).toBeVisible();
    const src = await iframe.getAttribute("src");
    expect(src).toBeTruthy();
    expect(src!).toContain("/reports/");
    expect(src!).toContain("report.html");
    expect(src!).toMatch(/[?&]t=\d+/); // cache-buster present

    // The report HTML must be framable same-origin (regression guard).
    const resp = await page.request.get(src!);
    expect(resp.status()).toBe(200);
    expect((resp.headers()["x-frame-options"] || "").toUpperCase()).toBe("SAMEORIGIN");

    // And the embedded document actually renders content inside the iframe.
    const frame = page.frameLocator('[data-testid="report-iframe"]');
    await expect(frame.locator("h1, h2").first()).toBeVisible({ timeout: 15_000 });
  });
});
