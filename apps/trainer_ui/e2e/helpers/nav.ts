// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Navigation helpers — replaces 10+ duplicate definitions of switchTab, selectProject, etc.
 */
import { expect, type Page } from "@playwright/test";
import { SEL } from "./selectors";
import { waitForTab, waitForProjects } from "./waiters";

/** Tab name → button index mapping. */
const TAB_INDEX: Record<string, number> = {
  projects: 0,
  annotate: 1,
  training: 2,
};

export type TabName = "projects" | "annotate" | "training" | "results";

/**
 * Switch to a named tab and wait for it to become active.
 * For dynamic result tabs (index 3+), pass "results".
 */
export async function switchTab(page: Page, tab: TabName): Promise<void> {
  const idx = TAB_INDEX[tab] ?? 0;
  await page.evaluate((i) => {
    const btns = document.querySelectorAll(".tabs-fixed button");
    if (btns[i]) (btns[i] as HTMLElement).click();
  }, idx);
  await waitForTab(page, tab);
}

/**
 * Select the first project tile and wait for it to become active.
 */
export async function selectFirstProject(page: Page): Promise<void> {
  await switchTab(page, "projects");
  await waitForProjects(page);
  const tile = page.locator(SEL.projectTile).first();
  await expect(tile).toBeVisible({ timeout: 30_000 });
  await tile.click();
  await expect(tile).toHaveClass(/active/);
}

/**
 * Select a project by name. Falls back to the first tile if not found.
 */
export async function selectProjectByName(page: Page, name: string): Promise<void> {
  await switchTab(page, "projects");
  await waitForProjects(page);

  const target = page.locator(SEL.projectTile, { hasText: name }).first();
  const found = await target.isVisible().catch(() => false);

  if (found) {
    await target.click();
    await expect(target).toHaveClass(/active/);
  } else {
    const fallback = page.locator(SEL.projectTile).first();
    await fallback.click();
    await expect(fallback).toHaveClass(/active/);
  }
}

/**
 * Select a project by tile index.
 */
export async function selectProjectByIndex(page: Page, index: number): Promise<void> {
  await switchTab(page, "projects");
  await waitForProjects(page);
  const tile = page.locator(SEL.projectTile).nth(index);
  await expect(tile).toBeVisible({ timeout: 10_000 });
  await tile.click();
  await expect(tile).toHaveClass(/active/);
}
