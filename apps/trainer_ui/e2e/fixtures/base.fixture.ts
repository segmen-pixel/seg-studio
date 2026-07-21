// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Base test fixture — provides shared page setup for all E2E tests.
 *
 * Usage:
 *   import { test, expect } from "../fixtures/base.fixture";
 *
 * This fixture:
 *   - Navigates to the app
 *   - Waits for API connection
 *   - Provides POM instances (projects, annotate, training, results, settings)
 */
import { test as base, expect } from "@playwright/test";
import { waitForApi } from "../helpers";
import { ProjectsPage, AnnotatePage, TrainingPage, ResultsPage, SettingsDialog } from "../pom";

type Fixtures = {
  projects: ProjectsPage;
  annotate: AnnotatePage;
  training: TrainingPage;
  results: ResultsPage;
  settings: SettingsDialog;
  _noRealTraining: void;
};

export const test = base.extend<Fixtures>({
  // Safety net: e2e must never launch a real training job on the dev GPU.
  // Page-level routes registered by individual specs take precedence, so
  // submit-contract tests still observe their own mocks.
  _noRealTraining: [
    async ({ context }, use) => {
      const fake = { run_id: "e2e-guard-fake", status: "starting" };
      for (const url of ["**/api/v1/projects/*/train"]) {
        await context.route(url, (route) =>
          route.request().method() === "POST"
            ? route.fulfill({ json: fake })
            : route.continue(),
        );
      }
      await use();
    },
    { auto: true },
  ],
  projects: async ({ page }, use) => {
    await use(new ProjectsPage(page));
  },
  annotate: async ({ page }, use) => {
    await use(new AnnotatePage(page));
  },
  training: async ({ page }, use) => {
    await use(new TrainingPage(page));
  },
  results: async ({ page }, use) => {
    await use(new ResultsPage(page));
  },
  settings: async ({ page }, use) => {
    await use(new SettingsDialog(page));
  },
});

export { expect };
