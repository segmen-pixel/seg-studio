// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Skip-budget reporter — turns silent coverage erosion into a failure.
 *
 * Many specs use data-dependent guards (test.skip when server data is
 * missing). Individually reasonable, collectively they let the suite decay
 * to smoke-only without anyone noticing (observed drift: 6 -> 8 skips
 * between consecutive days). This reporter fails the run when the number
 * of skipped tests exceeds the budget.
 *
 * Override with E2E_SKIP_BUDGET=<n> (default 4: report-flow, results-flow x2
 * and training-local start legitimately skip until a trained-run fixture
 * is seeded).
 */
import type { FullResult, Reporter, TestCase, TestResult } from "@playwright/test/reporter";

class SkipBudgetReporter implements Reporter {
  private skipped: string[] = [];

  onTestEnd(test: TestCase, result: TestResult): void {
    if (result.status === "skipped") {
      this.skipped.push(test.titlePath().slice(2).join(" › "));
    }
  }

  onEnd(result: FullResult): { status?: FullResult["status"] } | void {
    const budget = Number(process.env.E2E_SKIP_BUDGET ?? "4");
    if (this.skipped.length > 0) {
      console.log(`\n[skip-budget] ${this.skipped.length} skipped (budget: ${budget})`);
      for (const name of this.skipped) console.log(`  - ${name}`);
    }
    if (this.skipped.length > budget && result.status === "passed") {
      console.error(
        `[skip-budget] FAIL: ${this.skipped.length} skipped tests exceed the budget of ${budget}. ` +
          "Data-dependent guards are eroding coverage — reseed the fixtures " +
          "(e2e/global-setup.ts) or consciously raise E2E_SKIP_BUDGET.",
      );
      return { status: "failed" };
    }
  }
}

export default SkipBudgetReporter;
