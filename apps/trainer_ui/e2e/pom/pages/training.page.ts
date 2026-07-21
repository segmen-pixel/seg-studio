// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { expect, type Page, type Locator } from "@playwright/test";
import { SEL, switchTab } from "../../helpers";

export class TrainingPage {
  readonly page: Page;
  readonly configSection: Locator;
  readonly hyperToggle: Locator;
  readonly groupBasic: Locator;
  readonly runList: Locator;
  readonly emptyState: Locator;

  constructor(page: Page) {
    this.page = page;
    this.configSection = page.locator(SEL.trainingConfigSection);
    this.hyperToggle = page.locator(SEL.trainingHyperToggle);
    this.groupBasic = page.locator(SEL.trainingGroupBasic);
    this.runList = page.locator(SEL.trainingRunList);
    this.emptyState = page.locator(SEL.trainingEmptyState);
  }

  /** Navigate to the Training tab. */
  async goto(): Promise<void> {
    await switchTab(this.page, "training");
  }

  /** Open the hyperparameter settings panel. */
  async openHyperparams(): Promise<void> {
    await expect(this.configSection).toBeVisible({ timeout: 15_000 });
    // Only click toggle if panel is not already open
    const expanded = await this.hyperToggle.getAttribute("aria-expanded");
    if (expanded !== "true") {
      await this.hyperToggle.click();
    }
    await expect(this.groupBasic).toBeVisible({ timeout: 5_000 });
  }

  /** Set a hyperparameter value by input name/label. */
  async setParam(label: string, value: string): Promise<void> {
    const input = this.page
      .locator(SEL.trainingParamGroupInput)
      .filter({ hasText: label })
      .locator("input")
      .first();
    await input.fill(value);
  }

  /** Click the Start Training button (first visible). */
  async clickStart(): Promise<void> {
    const btn = this.page.getByRole("button", { name: /start|開始/i }).first();
    await expect(btn).toBeEnabled({ timeout: 5_000 });
    await btn.click();
  }

  /** Assert training layout is visible. */
  async expectVisible(): Promise<void> {
    await expect(this.page.locator(SEL.trainingLayout)).toBeVisible({ timeout: 10_000 });
  }
}
