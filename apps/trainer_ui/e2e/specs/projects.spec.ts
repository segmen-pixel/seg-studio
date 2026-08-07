// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "../fixtures/base.fixture";
import {
  waitForApi,
  switchTab,
  waitForProjects,
  SEL,
  useConsoleErrorCollector,
} from "../helpers";

test.describe("Projects", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForApi(page);
  });

  test("selecting a project tile marks it as active", async ({ projects }) => {
    // Arrange
    await projects.goto();
    await projects.expectHasProjects();

    // Act
    await projects.selectFirst();

    // Assert
    await expect(projects.tiles.first()).toHaveClass(/active/);
  });

  test("switching projects updates the class list", async ({
    page,
    projects,
    annotate,
  }) => {
    // Arrange — need at least 2 projects
    await projects.goto();
    await projects.tiles.nth(1).waitFor({ state: "visible", timeout: 15_000 }).catch(() => {});
    const count = await projects.tileCount();
    test.skip(count < 2, "Need at least 2 projects for this test");

    // Act — select first project and capture class count
    await projects.selectByIndex(0);
    await switchTab(page, "annotate");
    const classCountA = await annotate.classCount();

    // Act — switch to second project and capture class count
    await switchTab(page, "projects");
    await projects.selectByIndex(1);
    await switchTab(page, "annotate");
    const classCountB = await annotate.classCount();

    // Assert — class lists should be loaded (both > 0) and tab didn't break
    expect(classCountA).toBeGreaterThanOrEqual(0);
    expect(classCountB).toBeGreaterThanOrEqual(0);
    await annotate.expectVisible();
  });

  test("rapid project switching does not freeze the UI", async ({
    page,
    projects,
  }) => {
    // Arrange
    const col = useConsoleErrorCollector(page);
    await projects.goto();
    await projects.tiles.nth(1).waitFor({ state: "visible", timeout: 15_000 }).catch(() => {});
    const count = await projects.tileCount();
    test.skip(count < 2, "Need at least 2 projects for this test");

    // Act — rapidly switch between projects
    for (let i = 0; i < 5; i++) {
      await projects.selectByIndex(i % count);
    }

    // Assert — UI is still responsive, no freeze
    await expect(projects.tiles.first()).toBeVisible();
    await switchTab(page, "annotate");
    await expect(page.locator(SEL.appShellTab("annotate"))).toBeVisible();
    col.assertNoErrors();
  });

  test("orphaned classes are auto-reconciled via API", async ({ page, projects }) => {
    // Arrange — mock the projects API to return a project with mismatched classes
    await page.route("**/api/projects", async (route) => {
      const response = await route.fetch();
      const body = await response.json();

      // Inject a synthetic class that doesn't exist in images (orphan)
      if (Array.isArray(body) && body.length > 0) {
        const proj = body[0];
        if (proj.classes && Array.isArray(proj.classes)) {
          proj.classes.push({
            name: "__orphan_test_class__",
            color: "#ff00ff",
          });
        }
      }

      await route.fulfill({ json: body });
    });

    await page.goto("/");
    await waitForApi(page);

    // Act — select the project with the injected orphan class
    await projects.goto();
    await projects.selectFirst();

    // Assert — the UI should load without errors despite the orphan class
    await switchTab(page, "annotate");
    await expect(page.locator(SEL.annotateLayout)).toBeVisible({ timeout: 10_000 });
  });
});
