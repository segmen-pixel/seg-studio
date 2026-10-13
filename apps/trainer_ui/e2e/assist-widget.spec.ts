// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";
import { waitForApi } from "./helpers";

/**
 * A labelling run keeps going after you leave the project it belongs to -- the
 * server drives it whether or not a screen is listening -- but the Assist panel
 * lives inside the annotate tab and returns null off it. So there was no way,
 * from anywhere else, to see that anything was being labelled at all, let alone
 * which project.
 *
 * The widget answers that from any tab. It sits INSIDE .fab-stack, which is the
 * one element that coordinates the bottom-right corner; anything positioned
 * beside the stack instead of in it collides with whatever else is there.
 */
const TWO_RUNS = {
  items: [
    {
      project_id: "aaaaaaaaaaaa",
      project_name: "zz-e2e-plate-count",
      paused: false,
      waiting_for_reply: false,
      turns: 42,
      item_id: "img042",
    },
    {
      project_id: "bbbbbbbbbbbb",
      project_name: "zz-e2e-screws",
      paused: false,
      waiting_for_reply: true,
      turns: 7,
      item_id: "img011",
    },
  ],
};

test.describe("Auto-labelling widget", () => {
  test("names the project being labelled, and sits in the corner stack", async ({ page }) => {
    await page.route("**/api/v1/agent/runs", (route) => route.fulfill({ json: TWO_RUNS }));
    await page.goto("/");
    await waitForApi(page);

    const button = page.locator(".assist-fab-button");
    await expect(button).toBeVisible();
    await expect(page.locator(".assist-fab-count")).toHaveText("2");

    // In the stack, not beside it: the stack is what keeps the corner ordered.
    await expect(page.locator(".fab-stack > .assist-fab")).toHaveCount(1);

    await button.click();
    const panel = page.locator(".assist-fab-panel");
    await expect(panel).toBeVisible();
    await expect(panel).toContainText("zz-e2e-plate-count");
    await expect(panel).toContainText("zz-e2e-screws");
    await expect(panel).toContainText("img042");

    // A run waiting for an answer is the state worth seeing, and it is said in
    // a word and a glyph -- never by colour alone.
    const waiting = panel.locator(".assist-fab-state.waiting");
    await expect(waiting).toHaveCount(1);
    await expect(waiting).not.toBeEmpty();
  });

  test("is absent when nothing is being labelled", async ({ page }) => {
    await page.route("**/api/v1/agent/runs", (route) => route.fulfill({ json: { items: [] } }));
    await page.goto("/");
    await waitForApi(page);
    await expect(page.locator(".assist-fab")).toHaveCount(0);
  });
});

