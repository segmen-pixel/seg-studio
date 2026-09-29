// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { expect, type Page, type Locator } from "@playwright/test";
import { SEL, switchTab, waitForCanvasStable } from "../../helpers";
import { CanvasViewport } from "../components/canvas-viewport";

export class AnnotatePage {
  readonly page: Page;
  readonly canvas: CanvasViewport;
  readonly imageList: Locator;
  readonly classList: Locator;

  constructor(page: Page) {
    this.page = page;
    this.canvas = new CanvasViewport(page);
    this.imageList = page.locator(SEL.imageListItem);
    this.classList = page.locator(SEL.classListCard);
  }

  /** Navigate to the Annotate tab. */
  async goto(): Promise<void> {
    await switchTab(this.page, "annotate");
  }

  /** Wait for the canvas to be ready (image loaded, no busy overlay). */
  async waitForReady(): Promise<void> {
    await waitForCanvasStable(this.page);
  }

  /** Select an image from the image list by index. */
  async selectImage(index: number): Promise<void> {
    const item = this.imageList.nth(index);
    await expect(item).toBeVisible({ timeout: 10_000 });
    await item.click();
  }

  /** Select a class from the class list by name. */
  async selectClass(name: string): Promise<void> {
    const card = this.classList.filter({ hasText: name }).first();
    await expect(card).toBeVisible();
    await card.click();
  }

  /** Get the number of classes in the class list. */
  async classCount(): Promise<number> {
    return this.classList.count();
  }

  /** Check that the annotate layout is visible. */
  async expectVisible(): Promise<void> {
    await expect(this.page.locator(SEL.annotateLayout)).toBeVisible({ timeout: 10_000 });
  }
}
