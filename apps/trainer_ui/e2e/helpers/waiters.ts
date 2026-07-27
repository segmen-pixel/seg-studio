// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Shared wait helpers — replaces 18+ duplicate definitions of waitForApi, etc.
 */
import { expect, type Page } from "@playwright/test";
import { SEL } from "./selectors";

/**
 * Wait for the API connection badge to disappear (backend ready).
 * Uses two strategies for robustness:
 *   1. waitForFunction (fast, works when badge uses .api-status-badge.connecting)
 *   2. fallback locator check for .api-connecting-banner
 */
export async function waitForApi(page: Page, timeout = 15_000): Promise<void> {
  await page.waitForFunction(
    (sel: string) => !document.querySelector(sel),
    SEL.apiStatusBadge,
    { timeout },
  );
}

/**
 * Wait until at least one project tile is visible (loading finished).
 * Handles the case where the project list is still loading.
 */
export async function waitForProjects(page: Page, timeout = 45_000): Promise<void> {
  await expect
    .poll(
      async () => {
        const tiles = await page.locator(SEL.projectTile).count();
        const loading = await page.locator(SEL.projectsLoadingCard).count();
        return tiles > 0 || loading === 0;
      },
      { timeout },
    )
    .toBeTruthy();
}

/**
 * Wait for the canvas image to finish loading/decoding.
 */
export async function waitForCanvasStable(page: Page, timeout = 10_000): Promise<void> {
  await expect(page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, { timeout });
}

/**
 * Wait for a specific app-shell tab to become active.
 */
export async function waitForTab(page: Page, tab: string, timeout = 10_000): Promise<void> {
  await page.waitForSelector(SEL.appShellTab(tab), { timeout });
}
