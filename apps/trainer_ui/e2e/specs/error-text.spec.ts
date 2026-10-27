// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type { Page } from "@playwright/test";

import { test, expect } from "../fixtures/base.fixture";
import { waitForApi, SEL } from "../helpers";

/**
 * The trainer answers in English. A coded error is shown in the screen's
 * language; a refusal that carries only the code its status falls under is
 * shown as the server said it. Every refusal used to be shown as
 * "[NSS-1006] a required parameter is missing", whatever had been refused.
 *
 * The answers are put in the trainer's place, so the test does not depend on
 * finding a request the trainer refuses the same way in every version.
 */
async function answerCreateWith(page: Page, status: number, error: Record<string, string>): Promise<void> {
  await page.route("**/api/v1/projects", (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    return route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify({ error: { correlation_id: "req_e2e", ...error }, detail: error.message }),
    });
  });
}

async function openIn(page: Page, lang: "ja" | "en"): Promise<void> {
  await page.goto("/");
  await waitForApi(page);
  await page.evaluate((l) => localStorage.setItem("seg-lang", l), lang);
  await page.reload();
  await waitForApi(page);
}

async function createProject(page: Page): Promise<void> {
  await page.locator(".projects-create-row input").first().fill("zz-e2e-never-created");
  await page.locator('[data-tutorial-step="create-project-btn"]').click();
}

test.describe("Server errors on the screen", () => {
  test.afterEach(async ({ page }) => {
    await page.evaluate(() => localStorage.setItem("seg-lang", "ja"));
  });

  test("a refusal with no code of its own is shown with the server's reason", async ({ page, projects }) => {
    await openIn(page, "ja");
    await projects.goto();
    await answerCreateWith(page, 409, {
      code: "NSS-1006", kind: "uncoded", message: "A project with this name already exists",
    });

    await createProject(page);

    const toast = page.locator(SEL.toast);
    await expect(toast).toContainText("A project with this name already exists");
    await expect(toast).not.toContainText("NSS-1006");
    await expect(toast).not.toContainText("必須パラメータ");
    // A failure said in Japanese is drawn as one, and stays to be read.
    await expect(toast).toHaveClass(/toast-error/);
    await page.waitForTimeout(4500);
    await expect(toast).toContainText("A project with this name already exists");
  });

  test("a coded error is shown in the screen's language", async ({ page, projects }) => {
    await openIn(page, "ja");
    await projects.goto();
    await answerCreateWith(page, 404, { code: "NSS-2001", kind: "usual", message: "Project not found." });

    await createProject(page);
    await expect(page.locator(SEL.toast)).toContainText("[NSS-2001] プロジェクトが見つかりません");

    await openIn(page, "en");
    await projects.goto();
    await answerCreateWith(page, 404, { code: "NSS-2001", kind: "usual", message: "Project not found." });
    await createProject(page);
    await expect(page.locator(SEL.toast)).toContainText("[NSS-2001] Project not found");
  });

  test("a coded error that says more keeps what the server said beside the words", async ({ page, projects }) => {
    await openIn(page, "ja");
    await projects.goto();
    const said = "The SAM model file (mobile_sam.pt) could not be downloaded.";
    await answerCreateWith(page, 503, { code: "NSS-5011", kind: "specific", message: said });

    await createProject(page);

    const toast = page.locator(SEL.toast);
    await expect(toast).toContainText("[NSS-5011] SAM のモデルファイルをダウンロードできませんでした");
    await expect(toast).toContainText(`（${said}）`);
  });
});
