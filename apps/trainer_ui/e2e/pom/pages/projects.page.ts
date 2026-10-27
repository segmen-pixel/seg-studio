// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { expect, type Page, type Locator } from "@playwright/test";
import { SEL, switchTab, waitForProjects } from "../../helpers";

export class ProjectsPage {
  readonly page: Page;
  readonly tiles: Locator;
  readonly createRow: Locator;
  readonly currentTitle: Locator;

  constructor(page: Page) {
    this.page = page;
    this.tiles = page.locator(SEL.projectTile);
    this.createRow = page.locator(SEL.projectsCreateRow);
    this.currentTitle = page.locator(SEL.projectsCurrentTitle);
  }

  /** Navigate to the Projects tab. */
  async goto(): Promise<void> {
    await switchTab(this.page, "projects");
    await waitForProjects(this.page);
  }

  /** Select the first project tile. */
  async selectFirst(): Promise<void> {
    const tile = this.tiles.first();
    await expect(tile).toBeVisible({ timeout: 30_000 });
    await tile.click();
    await expect(tile).toHaveClass(/active/);
  }

  /** Select a project by display name. Falls back to first if not found. */
  async selectByName(name: string): Promise<void> {
    const target = this.tiles.filter({ hasText: name }).first();
    const found = await target.isVisible().catch(() => false);
    if (found) {
      await target.click();
      await expect(target).toHaveClass(/active/);
    } else {
      await this.selectFirst();
    }
  }

  /** Select a project by tile index. */
  async selectByIndex(index: number): Promise<void> {
    const tile = this.tiles.nth(index);
    await expect(tile).toBeVisible({ timeout: 10_000 });
    await tile.click();
    await expect(tile).toHaveClass(/active/);
  }

  /** Get the number of project tiles. */
  async tileCount(): Promise<number> {
    return this.tiles.count();
  }

  /** Assert that at least one project tile is visible. */
  async expectHasProjects(): Promise<void> {
    await expect(this.tiles.first()).toBeVisible({ timeout: 10_000 });
  }
}
