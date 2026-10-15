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
  // Wait for the strip to exist before clicking into it. The click goes
  // through page.evaluate, so a tab bar that has not mounted yet makes it a
  // silent no-op and the wait below then times out on a selector nothing
  // was ever asked to show -- which reads as "the tab is broken" rather
  // than "the page was still loading". Straight after a reload, that is the
  // difference between a green run and a flake.
  await page.locator(SEL.tabButtons).first().waitFor({ state: "attached", timeout: 30_000 });
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
 * Select a project by name. Scrolls the tile into view — a long project
 * list puts the old seed projects far below the fold — and FAILS when the
 * tile does not exist: the silent first-tile fallback this used to have
 * could send a destructive spec into a real project, and it also made
 * read-only sweeps (the EN leak check) walk whatever project happened to
 * sort first.
 */
export async function selectProjectByName(page: Page, name: string): Promise<void> {
  await switchTab(page, "projects");
  await waitForProjects(page);

  const target = page.locator(SEL.projectTile, { hasText: name }).first();
  // Tiles render asynchronously (fetch + per-project summaries), so a
  // snapshot check reads zero tiles long before the one we want exists —
  // which is exactly why the old isVisible() probe always chose the
  // fallback. Wait for the tile; a timeout here is a loud, honest failure.
  await target.waitFor({ state: "attached", timeout: 15_000 });
  await target.scrollIntoViewIfNeeded();
  await target.click();
  await expect(target).toHaveClass(/active/);
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
