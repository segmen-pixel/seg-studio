// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

import { selectFirstProject, waitForApi } from "./helpers";

/**
 * Deleting a whole class left the 要確認 badge on screen.
 *
 * Not on disk: the purge clears draft/draftRun/draftReason server-side. The
 * badge renders from an items array the browser fetched before the delete, and
 * the class-delete handler was the one mutation that neither invalidated the
 * ten-second response cache nor asked for the list again. Deleting the class
 * from chosen IMAGES did both, which is why that path looked fine.
 */
/**
 * Two rows on purpose. The first is the one the annotator selects, and the
 * active row's annotation gets rebuilt from scratch by an unrelated effect --
 * which silently drops draft/draftRun/draftReason and would clear the badge by
 * accident. The row that proves the point is the one nothing else touches.
 */
function items(draft: boolean) {
  const row = (id: string, flagged: boolean) => ({
    id, name: `${id}.png`, filename: `${id}.png`, set: "none", width: 64, height: 64,
    annotation: {
      hasMask: true, hasForeground: true, classIds: flagged && draft ? [1] : [],
      revision: 3, markedClean: false,
      ...(flagged && draft
        ? { draft: true, draftRun: "review", draftReason: "fewer than the teachers show" }
        : {}),
    },
  });
  return { version: 1, items: [row("zz-e2e-active", false), row("zz-e2e-watched", true)] };
}

test.describe("Deleting a class clears the review badge", () => {
  test("the badge goes without a page reload", async ({ page }) => {
    let purged = false;
    await page.route("**/api/v1/projects/*/datasets/annotate*", (route) =>
      route.fulfill({ json: items(!purged) }));
    await page.route("**/api/v1/projects/*/classes/*/purge", (route) => {
      purged = true;
      return route.fulfill({ json: { status: "ok", purged: { annotate_masks_updated: 1 } } });
    });

    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await page.locator(".tabs-fixed button").nth(1).click();
    await page.locator(".tool-icon[aria-pressed]").waitFor({ state: "visible", timeout: 30_000 });

    const badge = page.locator(".review-badge");
    await expect(badge).toHaveCount(1);

    // Delete is disabled while the background class is the active one, so pick
    // a real class first. The first row in the window is the mark-clean row.
    const del = page.locator(".class-actions-row button.danger");
    const rows = page.locator(".class-list-window .class-item-row");
    for (let i = 1; i < (await rows.count()); i++) {
      await rows.nth(i).click();
      if (await del.isEnabled()) break;
    }
    await expect(del).toBeEnabled();
    page.once("dialog", (d) => d.accept());      // if it asks
    await del.click();

    // The server has already stopped reporting it; the screen has to agree.
    await expect(badge).toHaveCount(0, { timeout: 15_000 });
    expect(purged).toBe(true);
  });
});
