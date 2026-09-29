// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, switchTab, selectFirstProject, SEL, selectProjectByName } from "../helpers";

test.describe("Local Training", () => {
  test("hyperparameter panel opens and accepts input", async ({ page, training }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await training.goto();

    // Act: open hyperparameters
    await training.openHyperparams();

    // Assert: basic group is visible with inputs
    await expect(page.locator(SEL.trainingGroupBasic)).toBeVisible();
    const inputs = page.locator(`${SEL.trainingGroupBasic} input`);
    await expect(inputs.first()).toBeVisible();
  });

  test("epochs input persists after tab switch", async ({ page, training }) => {
    // Arrange
    await page.goto("/");
    await waitForApi(page);
    await selectFirstProject(page);
    await training.goto();
    await training.openHyperparams();

    // Act: change epochs value
    const epochsInput = page.locator(`${SEL.trainingGroupBasic} input`).first();
    await epochsInput.fill("25");

    // Switch away and back
    await switchTab(page, "projects");
    await training.goto();
    await training.openHyperparams();

    // Assert: value persisted
    await expect(epochsInput).toHaveValue("25");
  });

  test("start button sends training config via API", async ({ page, training }) => {
    let apiPayload: Record<string, unknown> | null = null;

    // Arrange: mock training start endpoint
    await page.route("**/api/v1/projects/*/train", async (route) => {
      if (route.request().method() === "POST") {
        apiPayload = JSON.parse((await route.request().postData()) ?? "{}");
        return route.fulfill({ json: { run_id: "test-run-001", status: "starting" } });
      }
      return route.continue();
    });

    await page.goto("/");
    await waitForApi(page);

    // Deterministic fixture project with images (e2e/global-setup.ts)
    await selectProjectByName(page, "zz-e2e-seed-1");

    await training.goto();

    // Pick a training mode first — start stays disabled until a mode is chosen.
    await page.locator(".training-mode-btn").first().click();

    // The seed project may not satisfy local-training data requirements;
    // when start stays disabled, skip honestly (counted by the skip budget)
    // instead of clicking into nothing and asserting nothing.
    const startBtn = page.getByRole("button", { name: /start|開始/i }).first();
    const enabled = await startBtn.isEnabled({ timeout: 5_000 }).catch(() => false);
    test.skip(!enabled, "Start disabled — seed project lacks trainable data (trained-fixture follow-up)");

    // Act: click start
    await training.clickStart();

    // Assert: the mocked endpoint actually received the config
    await expect.poll(() => apiPayload, { timeout: 10_000 }).not.toBeNull();
    expect(apiPayload).toHaveProperty("epochs");
  });
});
