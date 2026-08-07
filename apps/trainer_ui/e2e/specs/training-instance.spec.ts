// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
// Instance training mode (v0.9.8 M3): mode button, synthesis section render,
// and the instance_* payload shape. Fast render/validation only — the real
// instance train smoke is dev-box @heavy territory.
import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, SEL, selectProjectByName } from "../helpers";

test.describe("Instance Training Mode", () => {
  test("instance mode reveals the synthesis section and hides detailed settings", async ({ page, training }) => {
    await page.goto("/");
    await waitForApi(page);
    await selectProjectByName(page, "zz-e2e-seed-1");
    await training.goto();

    // 4th mode button exists and selects.
    const instanceBtn = page.locator(SEL.trainingModeBtn).nth(3);
    await instanceBtn.click();
    await expect(instanceBtn).toHaveClass(/active/);

    // Instance mode keeps the same detailed-settings toggle as every other mode
    // (it used to be hidden here, leaving the busiest form permanently open).
    await expect(page.locator(SEL.trainingHyperToggle)).toBeVisible();
    if (await page.locator(SEL.instanceSection).count() === 0) {
      await page.locator(SEL.trainingHyperToggle).click();
    }
    await expect(page.locator(SEL.instanceSection)).toBeVisible();
    await expect(page.locator(SEL.instancePreviewBtn)).toBeVisible();

    // Patch size decides the scale objects reach the model at, for training
    // and inference alike, and was API-only until the docs started telling
    // people to change it.
    await expect(page.locator("[data-testid='select-instance-patch-size']")).toBeVisible();
    await expect(page.locator("[data-testid='input-instance-patch-size']")).toHaveValue("768");

    // Switching back to standard restores the semantic form.
    await page.locator(SEL.trainingModeBtn).first().click();
    await expect(page.locator(SEL.instanceSection)).toHaveCount(0);
    await expect(page.locator(SEL.trainingHyperToggle)).toBeVisible();
  });

  test("start sends training_mode=instance with synthesis fields", async ({ page, training }) => {
    let apiPayload: Record<string, unknown> | null = null;
    await page.route("**/api/v1/projects/*/train", async (route) => {
      if (route.request().method() === "POST") {
        apiPayload = JSON.parse((await route.request().postData()) ?? "{}");
        return route.fulfill({ json: { run_id: "test-run-inst", status: "starting" } });
      }
      return route.continue();
    });

    await page.goto("/");
    await waitForApi(page);
    await selectProjectByName(page, "zz-e2e-seed-1");
    await training.goto();

    await page.locator(SEL.trainingModeBtn).nth(3).click();
    // The synthesis fields live behind the detailed-settings toggle, same as
    // every other mode.
    if (await page.locator(SEL.instanceSection).count() === 0) {
      await page.locator(SEL.trainingHyperToggle).click();
    }
    await page.locator("[data-testid='input-instance-n-train']").fill("120");
    await page.locator("[data-testid='input-instance-n-train']").blur();
    await page.locator("[data-testid='input-instance-patch-size']").fill("512");
    await page.locator("[data-testid='input-instance-patch-size']").blur();

    const startBtn = page.getByRole("button", { name: /start|開始/i }).first();
    const enabled = await startBtn.isEnabled({ timeout: 5_000 }).catch(() => false);
    test.skip(!enabled, "Start disabled — seed project lacks trainable data (trained-fixture follow-up)");

    await training.clickStart();

    await expect.poll(() => apiPayload, { timeout: 10_000 }).not.toBeNull();
    expect(apiPayload).toMatchObject({
      training_mode: "instance",
      instance_model_size: "small",
      instance_n_train: 120,
      instance_patch_size: 512,
      k_folds: 1,
      iterative_mode: false,
    });
    expect(apiPayload).toHaveProperty("instance_objects_min");
    expect(apiPayload).toHaveProperty("instance_stack_pair_prob");
  });
});
