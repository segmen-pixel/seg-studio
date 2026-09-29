// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Shared assertion helpers — replaces 4+ duplicate useConsoleErrorCollector definitions.
 */
import { expect, type Page, type ConsoleMessage } from "@playwright/test";

/** Browser network errors that are expected (missing models, hardware, etc.) */
const EXPECTED_ERRORS =
  /favicon|Failed to load resource|net::ERR_|AbortError|the server responded with a status of 4/i;

/**
 * Collect console errors throughout a test. Call `assertNoErrors()` at end.
 *
 * @example
 * ```ts
 * test("no console errors", async ({ page }) => {
 *   const col = useConsoleErrorCollector(page);
 *   await page.goto("/");
 *   // ... test actions ...
 *   col.assertNoErrors();
 * });
 * ```
 */
export function useConsoleErrorCollector(page: Page) {
  const errors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  return {
    errors,
    assertNoErrors(ignore: RegExp = EXPECTED_ERRORS) {
      const filtered = errors.filter((e) => !ignore.test(e));
      expect(
        filtered,
        `Unexpected console errors:\n${filtered.join("\n")}`,
      ).toHaveLength(0);
    },
  };
}
