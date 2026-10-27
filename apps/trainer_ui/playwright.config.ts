// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  testIgnore: ["_tools/**", "_archive/**"],
  outputDir: "./e2e/test-results",
  timeout: 60_000,
  retries: 0,
  workers: 1,
  use: {
    baseURL: "http://localhost:8002/ui",
    // Pre-seed localStorage so the first-launch tutorial overlay (which
    // intercepts pointer events) never auto-opens during e2e runs.
    storageState: "./e2e/fixtures/storage-state.json",
    viewport: { width: 1920, height: 1080 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  // NOTE: run WITHOUT --reporter CLI flag — it would override this list and
  // silently drop the skip-budget gate.
  reporter: [
    ["dot"],
    ["html", { outputFolder: "./e2e/html-report", open: "never" }],
    ["./e2e/skip-budget-reporter.ts"],
  ],
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
