// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { expect, type Page, type Locator } from "@playwright/test";
import { SEL } from "../../helpers";

export class SettingsDialog {
  readonly page: Page;
  readonly overlay: Locator;

  constructor(page: Page) {
    this.page = page;
    this.overlay = page.locator(SEL.settingsOverlay);
  }

  /** Open the settings dialog via the header gear icon. */
  async open(): Promise<void> {
    const btn = this.page.locator(`${SEL.appHeader} button`).filter({ hasText: /settings|設定/i }).first();
    // fallback: try the gear icon by title attribute
    const gear = this.page.locator('button[title*="settings"], button[title*="設定"]').first();
    const target = (await btn.isVisible().catch(() => false)) ? btn : gear;
    await target.click();
    await expect(this.overlay).toBeVisible({ timeout: 5_000 });
  }

  /** Close the settings dialog. */
  async close(): Promise<void> {
    // Click the overlay backdrop or close button
    const closeBtn = this.overlay.locator('button[aria-label="Close"]').first();
    const hasClose = await closeBtn.isVisible().catch(() => false);
    if (hasClose) {
      await closeBtn.click();
    } else {
      await this.overlay.click({ position: { x: 10, y: 10 } });
    }
    await expect(this.overlay).toBeHidden({ timeout: 3_000 });
  }

  /** Assert the settings dialog is visible. */
  async expectVisible(): Promise<void> {
    await expect(this.overlay).toBeVisible({ timeout: 5_000 });
  }
}
