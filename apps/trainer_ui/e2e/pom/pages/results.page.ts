// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { expect, type Page, type Locator } from "@playwright/test";
import { SEL, switchTab } from "../../helpers";

export class ResultsPage {
  readonly page: Page;
  readonly layout: Locator;

  constructor(page: Page) {
    this.page = page;
    this.layout = page.locator(SEL.resultsLayout);
  }

  /** Navigate to the Results tab (first result tab if multiple). */
  async goto(): Promise<void> {
    // Results tabs are dynamic (index 3+). Click the first one after training tabs.
    const resultTab = this.page.locator(SEL.tabResult).first();
    const visible = await resultTab.isVisible().catch(() => false);
    if (visible) {
      await resultTab.click();
    } else {
      // Fallback: use keyboard or index-based approach
      await switchTab(this.page, "results");
    }
  }

  /** Assert results layout is visible. */
  async expectVisible(): Promise<void> {
    await expect(this.layout).toBeVisible({ timeout: 10_000 });
  }
}
