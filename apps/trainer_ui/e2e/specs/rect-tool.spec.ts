// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * The rectangle tool has to paint, and one undo has to take the whole box back.
 *
 * Both are easy to get silently wrong here. Tool dispatch is a flat chain of
 * `if (tool === ...)` with no default, so an id nobody handled is a no-op that
 * still looks like a working button; and an undo entry whose prev equals its
 * next reverts nothing while looking exactly like history.
 *
 * The tool is selected by clicking its button and the selection is asserted
 * before anything is drawn. Reaching for the keyboard shortcut instead is how
 * the first version of this test fooled itself: the click that picks a class
 * leaves focus in that class's name input, so "R" was typed into the name and
 * the brush -- still selected -- painted a fat stroke that passed every pixel
 * assertion here.
 */
import type { Page } from "@playwright/test";
import { test, expect } from "../fixtures/base.fixture";
import { selectFirstProject, waitForApi } from "../helpers";

/** Painted mask pixels: non-transparent pixels on the overlay canvas.
 *  DOM order inside .canvas-inner is image, overlay, ui. */
async function paintedPixels(page: Page): Promise<number> {
  return page.evaluate(() => {
    const overlay = document.querySelectorAll(".canvas-inner canvas")[1] as
      | HTMLCanvasElement
      | undefined;
    if (!overlay) return -1;
    const ctx = overlay.getContext("2d");
    if (!ctx) return -1;
    const { data } = ctx.getImageData(0, 0, overlay.width, overlay.height);
    let painted = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) painted += 1;
    return painted;
  });
}

/** Tool rail buttons, in TOOL_LABELS order minus the two the rail filters out. */
const RAIL = ".overlay-tools.right button";
const BRUSH = 0;
const RECT = 1;

test.describe("Rectangle tool", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
  });

  test("sits between brush and eraser and is reachable by shortcut", async ({ page, annotate }) => {
    await selectFirstProject(page);
    await annotate.goto();

    const buttons = page.locator(RAIL);
    await expect(buttons.nth(BRUSH)).toHaveAttribute("title", /Brush|ブラシ/);
    await expect(buttons.nth(RECT)).toHaveAttribute("title", /Rectangle|矩形/);
    await expect(buttons.nth(RECT + 1)).toHaveAttribute("title", /Eraser|消しゴム/);

    // The rail button is not a text field, so the next keydown reaches window.
    await buttons.nth(BRUSH).click();
    await expect(buttons.nth(BRUSH)).toHaveClass(/active/);
    await page.keyboard.press("R");
    await expect(buttons.nth(RECT)).toHaveClass(/active/);
    await expect(buttons.nth(BRUSH)).not.toHaveClass(/active/);
  });

  test("drag fills the box, and one undo takes it back", async ({ page, annotate }) => {
    await selectFirstProject(page);
    await annotate.goto();
    // Asserted, not skipped: the e2e seed always provides images and a
    // foreground class, so an empty list is a broken fixture worth failing on.
    // A skip here would guard nothing while looking like it guarded something.
    await expect.poll(() => annotate.imageList.count(), { timeout: 20_000 }).toBeGreaterThan(0);
    await annotate.selectImage(0);
    await annotate.waitForReady();

    // No class is clicked on purpose. A foreground class is already active on
    // load, and SEL.classListCard also matches the mark-clean row above the
    // class list -- clicking that marks the image as verified-clean and leaves
    // history in a state where the undo below lands somewhere else entirely.
    // If no class were active the paint guard would drop the press and the
    // pixel assertion would fail, which is the report we want anyway.

    const rect = page.locator(RAIL).nth(RECT);
    await rect.click();
    await expect(rect).toHaveClass(/active/);
    await expect(page.locator(".canvas-stack")).toHaveCSS("cursor", "crosshair");

    // Poll rather than read once: the canvas stack re-renders when the tool
    // changes, and a read that lands mid-render finds no canvas and returns -1.
    await expect.poll(() => paintedPixels(page), { timeout: 10_000 }).toBeGreaterThanOrEqual(0);
    const before = await paintedPixels(page);

    // .canvas-inner is the image itself; .canvas-stack includes the letterbox
    // around it, and a drag that starts out there never reaches the image at
    // all (screenToImage returns null and the press is dropped).
    const img = await page.locator(".canvas-inner").boundingBox();
    expect(img).not.toBeNull();
    const x = img!.x + img!.width * 0.1;
    const y = img!.y + img!.height * 0.1;
    const toX = img!.x + img!.width * 0.9;
    const toY = img!.y + img!.height * 0.9;
    await page.mouse.move(x, y);
    await page.mouse.down();
    // Two moves, not one: the draft is updated on move and committed on up, so
    // a single jump would still pass while a broken move handler went unseen.
    await page.mouse.move((x + toX) / 2, (y + toY) / 2);
    await page.mouse.move(toX, toY);
    await page.mouse.up();

    await expect.poll(() => paintedPixels(page), { timeout: 10_000 }).toBeGreaterThan(before);

    await page.keyboard.press("Control+z");
    await expect.poll(() => paintedPixels(page), { timeout: 10_000 }).toBe(before);
  });
});
