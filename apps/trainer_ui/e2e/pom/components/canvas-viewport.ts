// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * CanvasViewport — abstracts canvas interactions in image-coordinate space.
 *
 * Design decisions:
 *   - All public methods accept image coordinates, not screen coordinates.
 *   - Coordinate conversion is handled internally.
 *   - deviceScaleFactor=1 + fixed viewport assumed for stability.
 *   - No waitForTimeout — uses semantic waits only.
 */
import { expect, type Page, type Locator } from "@playwright/test";
import { SEL } from "../../helpers";

export interface Point {
  x: number;
  y: number;
}

export class CanvasViewport {
  readonly page: Page;
  readonly stack: Locator;

  constructor(page: Page) {
    this.page = page;
    this.stack = page.locator(SEL.canvasStack);
  }

  /** Wait for the canvas stack to be visible and the busy overlay to disappear. */
  async waitForReady(): Promise<void> {
    await expect(this.stack).toBeVisible({ timeout: 15_000 });
    await expect(this.page.locator(SEL.annotatorBusyOverlay)).toHaveCount(0, {
      timeout: 15_000,
    });
  }

  /**
   * Get the bounding box of the canvas stack in screen coordinates.
   * Used internally for coordinate conversion.
   */
  private async getBounds() {
    const box = await this.stack.boundingBox();
    if (!box) throw new Error("Canvas stack not visible — cannot get bounds");
    return box;
  }

  /**
   * Convert image coordinates to screen coordinates.
   * For now, assumes a simple mapping where the canvas fills the stack.
   * TODO: account for zoom/pan transforms when needed.
   */
  private async toScreen(imgX: number, imgY: number): Promise<{ x: number; y: number }> {
    const box = await this.getBounds();
    return { x: box.x + imgX, y: box.y + imgY };
  }

  /** Click at an image coordinate. */
  async clickAt(imgX: number, imgY: number): Promise<void> {
    const { x, y } = await this.toScreen(imgX, imgY);
    await this.page.mouse.click(x, y);
  }

  /** Perform a brush stroke from one point to another. */
  async brushStroke(from: Point, to: Point, steps = 5): Promise<void> {
    const start = await this.toScreen(from.x, from.y);
    const end = await this.toScreen(to.x, to.y);
    await this.page.mouse.move(start.x, start.y);
    await this.page.mouse.down();
    await this.page.mouse.move(end.x, end.y, { steps });
    await this.page.mouse.up();
  }

  /** Draw a polygon by clicking a series of points, then closing. */
  async drawPolygon(points: Point[]): Promise<void> {
    if (points.length < 3) throw new Error("Polygon requires at least 3 points");
    for (const pt of points) {
      await this.clickAt(pt.x, pt.y);
    }
    // Close polygon by clicking the first point again
    await this.clickAt(points[0].x, points[0].y);
  }

  /** Zoom by scrolling the mouse wheel over the canvas center. */
  async zoom(delta: number): Promise<void> {
    const box = await this.getBounds();
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    await this.page.mouse.move(cx, cy);
    await this.page.mouse.wheel(0, delta);
  }

  /** Pan by dragging from center with offset. */
  async pan(dx: number, dy: number): Promise<void> {
    const box = await this.getBounds();
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    // Middle-button or shift+drag depending on UI implementation
    await this.page.mouse.move(cx, cy);
    await this.page.mouse.down({ button: "middle" });
    await this.page.mouse.move(cx + dx, cy + dy, { steps: 3 });
    await this.page.mouse.up({ button: "middle" });
  }
}
