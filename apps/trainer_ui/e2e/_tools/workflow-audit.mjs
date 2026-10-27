// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const BASE_URL = process.env.NSS_UI_URL ?? "http://localhost:8002/ui";
const API_BASE = BASE_URL.replace(/\/ui\/?$/, "");
const TARGET_PROJECT_NAME = process.env.NSS_AUDIT_PROJECT ?? "Gear";
const EXPECTED_NOISE = /favicon|AbortError|net::ERR_|Failed to load resource|\.webm/i;

function nowStamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function classifyLatency(ms) {
  if (ms <= 800) return "comfortable";
  if (ms <= 2_000) return "noticeable";
  if (ms <= 5_000) return "slow";
  return "frustrating";
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function writeJson(filePath, value) {
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  return res.json();
}

async function waitForApiReady(page) {
  const banner = page.locator(".api-connecting-banner");
  const started = Date.now();
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
  try {
    await banner.waitFor({ state: "hidden", timeout: 90_000 });
  } catch {
    const count = await banner.count();
    if (count > 0) throw new Error("API connection banner never cleared.");
  }
  return Date.now() - started;
}

async function switchTab(page, tabName) {
  await page.getByRole("button", { name: new RegExp(`^${tabName}$`, "i") }).click();
  await page.locator(`.app-shell.tab-${tabName.toLowerCase()}`).waitFor({ state: "visible", timeout: 10_000 });
}

async function timed(report, key, fn) {
  const started = Date.now();
  const value = await fn();
  const ms = Date.now() - started;
  report.timings[key] = { ms, band: classifyLatency(ms) };
  if (ms > 2_000) {
    report.discomforts.push({
      step: key,
      ms,
      band: classifyLatency(ms),
      note: ms > 5_000 ? "Likely frustrating in normal use." : "User will notice this pause.",
    });
  }
  return value;
}

async function snapshot(page, outputDir, name) {
  await page.screenshot({ path: path.join(outputDir, `${name}.png`), fullPage: true });
  await fs.writeFile(path.join(outputDir, `${name}.html`), await page.content(), "utf8");
}

function sanitizeConsoleMessage(message) {
  return {
    type: message.type(),
    text: message.text(),
    location: message.location(),
  };
}

async function main() {
  const stamp = nowStamp();
  const outputDir = path.resolve("e2e", "audit-artifacts", stamp);
  await ensureDir(outputDir);

  const report = {
    started_at: new Date().toISOString(),
    base_url: BASE_URL,
    api_base: API_BASE,
    target_project_name: TARGET_PROJECT_NAME,
    selected_project: null,
    selected_run: null,
    predicted_image_id: null,
    timings: {},
    observations: {},
    console: {
      warnings: [],
      errors: [],
      page_errors: [],
    },
    network: {
      failed_requests: [],
      bad_responses: [],
    },
    blockers: [],
    discomforts: [],
    scenario: [
      "Open app and wait for API readiness",
      "Select an existing project from Projects",
      "Open Annotate and navigate images with the keyboard",
      "Trigger quick learning from the status bar and observe training feedback",
      "Open Training and inspect run visibility / next-step clarity",
      "Open Results with an existing model and inspect a predicted image",
    ],
  };

  const projects = await fetchJson(`${API_BASE}/projects`);
  const selectedProject =
    projects.find((project) => project.name === TARGET_PROJECT_NAME) ??
    projects[0] ??
    null;
  if (!selectedProject) {
    throw new Error("No projects available for workflow audit.");
  }
  report.selected_project = { id: selectedProject.id, name: selectedProject.name };

  const runs = await fetchJson(`${API_BASE}/projects/${selectedProject.id}/train/runs`);
  const targetRun =
    [...runs]
      .filter((run) => run.has_model)
      .sort((a, b) => Date.parse(b.updated_at || b.created_at || "") - Date.parse(a.updated_at || a.created_at || ""))[0] ??
    null;
  if (targetRun) {
    report.selected_run = {
      run_id: targetRun.run_id,
      model_name: targetRun.model_name,
      status: targetRun.status,
    };
    try {
      const predictStatus = await fetchJson(
        `${API_BASE}/projects/${selectedProject.id}/train/runs/${targetRun.run_id}/predict/status?backend=onnx`,
      );
      report.predicted_image_id = predictStatus.predicted?.[0] ?? null;
    } catch (error) {
      report.blockers.push({
        area: "results",
        message: `Could not read prediction status for target run: ${String(error)}`,
      });
    }
  } else {
    report.blockers.push({
      area: "training/results",
      message: "No existing run with a model was found. Results audit will be partial.",
    });
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const page = await context.newPage();

  page.on("console", (message) => {
    if (message.type() === "warning") {
      report.console.warnings.push(sanitizeConsoleMessage(message));
    } else if (message.type() === "error" && !EXPECTED_NOISE.test(message.text())) {
      report.console.errors.push(sanitizeConsoleMessage(message));
    }
  });
  page.on("pageerror", (error) => {
    report.console.page_errors.push(String(error));
  });
  page.on("requestfailed", (request) => {
    if (!EXPECTED_NOISE.test(request.url())) {
      report.network.failed_requests.push({
        url: request.url(),
        method: request.method(),
        failure: request.failure(),
      });
    }
  });
  page.on("response", (response) => {
    if (response.status() >= 500 && !EXPECTED_NOISE.test(response.url())) {
      report.network.bad_responses.push({
        url: response.url(),
        status: response.status(),
      });
    }
  });

  try {
    report.timings.api_ready = {
      ms: await waitForApiReady(page),
      band: "pending",
    };
    report.timings.api_ready.band = classifyLatency(report.timings.api_ready.ms);
    if (report.timings.api_ready.ms > 2_000) {
      report.discomforts.push({
        step: "api_ready",
        ms: report.timings.api_ready.ms,
        band: report.timings.api_ready.band,
        note: "The initial connection banner stayed visible long enough to be noticeable.",
      });
    }

    await timed(report, "projects_tile_ready", async () => {
      const tile = page.locator(".project-tile").filter({ hasText: selectedProject.name }).first();
      await tile.waitFor({ state: "visible", timeout: 20_000 });
    });

    await timed(report, "projects_select_project", async () => {
      const tile = page.locator(".project-tile").filter({ hasText: selectedProject.name }).first();
      await tile.click();
      await page.waitForFunction(
        (projectName) => {
          return Array.from(document.querySelectorAll(".project-tile")).some((node) =>
            node.classList.contains("active") && (node.textContent || "").includes(projectName),
          );
        },
        selectedProject.name,
        { timeout: 10_000 },
      );
    });

    report.observations.projects = {
      helper_copy: await page.locator(".projects-helper, .project-summary-copy").allInnerTexts().catch(() => []),
    };
    await snapshot(page, outputDir, "projects");

    // Scope all Annotate queries to the visible tab panel to avoid matching
    // stale/hidden duplicates in other panels.
    const annotatePanel = page.locator(".tab-panel.active");

    await timed(report, "annotate_tab_ready", async () => {
      await switchTab(page, "Annotate");
      await annotatePanel.getByRole("listbox", { name: "Annotate images" }).waitFor({ state: "visible", timeout: 20_000 });
      await annotatePanel.locator(".image-list-dropzone .card.list-item-flat").first().waitFor({ state: "visible", timeout: 20_000 });
    });

    await timed(report, "annotate_first_canvas", async () => {
      const firstItem = annotatePanel.locator(".image-list-dropzone .card.list-item-flat").first();
      await firstItem.click();
      await page.waitForFunction(() => {
        const panel = document.querySelector(".tab-panel.active");
        const canvas = panel && panel.querySelector(".canvas-area canvas");
        return !!canvas && canvas.width > 0 && canvas.height > 0;
      }, { timeout: 15_000 });
    });

    await timed(report, "annotate_keyboard_next_image", async () => {
      const list = annotatePanel.getByRole("listbox", { name: "Annotate images" });
      const active = annotatePanel.locator(".image-list-dropzone .card.list-item-flat.active");
      const before = await active.getAttribute("data-image-id");
      await list.focus();
      await page.keyboard.press("ArrowDown");
      await page.waitForFunction(
        (prevId) => {
          const panel = document.querySelector(".tab-panel.active");
          const next = panel && panel.querySelector(".image-list-dropzone .card.list-item-flat.active");
          return next && next.getAttribute("data-image-id") !== prevId;
        },
        before,
        { timeout: 10_000 },
      );
    });

    report.observations.annotate = {
      quick_model_title: await annotatePanel.locator(".model-assist-status-title").innerText(),
      quick_model_copy: await annotatePanel.locator(".model-assist-status-copy").innerText(),
    };
    await snapshot(page, outputDir, "annotate");

    // Use the always-visible status bar button instead of the unreliable context menu.
    const quickTrainBtn = annotatePanel.locator(".model-assist-status-actions button.primary").filter({ hasText: /Start Quick Learning|Retrain/i }).first();
    if (await quickTrainBtn.count()) {
      const previousTitle = await annotatePanel.locator(".model-assist-status-title").innerText();
      await timed(report, "annotate_start_quick_learning", async () => {
        await quickTrainBtn.click();
        await page.waitForFunction(
          (titleText) => {
            const panel = document.querySelector(".tab-panel.active");
            const node = panel && panel.querySelector(".model-assist-status-title");
            return node && (node.textContent || "").trim() !== titleText.trim();
          },
          previousTitle,
          { timeout: 30_000 },
        );
      }).catch((error) => {
        report.discomforts.push({
          area: "annotate/quick-learning",
          message: `Quick learning did not update the status bar within 30 s (may be slow hardware): ${String(error)}`,
        });
      });

      const statusTitle = await annotatePanel.locator(".model-assist-status-title").innerText();
      report.observations.annotate.quick_model_title_after_start = statusTitle;
      if (/training/i.test(statusTitle)) {
        await timed(report, "annotate_stop_quick_learning", async () => {
          const stopBtn = annotatePanel.locator(".model-assist-status-actions button.ghost").filter({ hasText: /Stop/i }).first();
          await stopBtn.click();
          await page.waitForFunction(() => {
            const panel = document.querySelector(".tab-panel.active");
            const node = panel && panel.querySelector(".model-assist-status-title");
            return node && !/Quick model is training/i.test(node.textContent || "");
          }, { timeout: 120_000 });
        }).catch((error) => {
          report.discomforts.push({
            area: "annotate/quick-learning",
            message: `Quick learning stop/completion feedback did not clear within 120 s (training may still be running): ${String(error)}`,
          });
        });
      }
    } else {
      report.discomforts.push({
        area: "annotate/quick-learning",
        message: "Start Quick Learning button was not visible in the status bar (model may already exist).",
      });
    }

    // Scope Training queries to the visible tab panel.
    const trainingPanel = page.locator(".tab-panel.active");

    await timed(report, "training_tab_ready", async () => {
      await switchTab(page, "Training");
      await trainingPanel.locator(".training-run-list").waitFor({ state: "visible", timeout: 20_000 });
    });
    report.observations.training = {
      summary_title: await trainingPanel.locator(".training-summary-title").innerText().catch(() => null),
      summary_copy: await trainingPanel.locator(".training-summary-copy").innerText().catch(() => null),
      run_count: await trainingPanel.locator(".training-run-list .card").count(),
      running_badges: await trainingPanel.locator(".training-run-list .run-pulse-dot, .training-run-list .run-reserved-badge").count(),
    };
    await snapshot(page, outputDir, "training");

    await timed(report, "results_tab_ready", async () => {
      let openedFromTraining = false;
      if (targetRun) {
        const trainingRunCard = trainingPanel.locator(".training-run-list .card").filter({
          hasText: targetRun.model_name ?? targetRun.run_id.slice(0, 8),
        }).first();
        if (await trainingRunCard.count()) {
          const openResultsButton = trainingRunCard.locator('[title="Open Results"]').first();
          if (await openResultsButton.count()) {
            await trainingRunCard.click();
            await openResultsButton.click();
            openedFromTraining = true;
          }
        }
      }
      if (!openedFromTraining) {
        const resultsTab = page.getByRole("button", { name: /^Results$/i });
        if (await resultsTab.count()) {
          await resultsTab.click();
        } else {
          throw new Error("No Results entry point was available from Training or the top tab bar.");
        }
      }
      await page.locator(".tab-panel.active .results-layout").waitFor({ state: "visible", timeout: 20_000 });
      await page.locator(".tab-panel.active").getByRole("listbox", { name: "Prediction images" }).waitFor({ state: "visible", timeout: 20_000 });
    });

    // Re-bind to the now-active Results panel.
    const resultsPanel = page.locator(".tab-panel.active");

    if (targetRun) {
      const runLocator = resultsPanel.locator(".results-run-list .card").filter({
        hasText: targetRun.model_name ?? targetRun.run_id.slice(0, 8),
      }).first();
      if (await runLocator.count()) {
        await timed(report, "results_select_run", async () => {
          await runLocator.click();
          await page.waitForTimeout(500);
        });
      }
    }

    if (!report.observations.results) report.observations.results = {};
    if (!report.predicted_image_id) {
      // No inference has been run — verify the UI shows a "not predicted" state
      // rather than a stuck loading spinner.
      const panelText = await resultsPanel.innerText().catch(() => "");
      if (panelText.includes("Loading prediction") || panelText.includes("Loading selected prediction")) {
        report.observations.results.prediction_state = "loading";
      } else {
        report.observations.results.prediction_state = "not_inferred";
      }
    }

    if (report.predicted_image_id) {
      const targetImageId = report.predicted_image_id;
      const activeResultsImageId = await resultsPanel.locator(".results-image-list .card.active").getAttribute("data-image-id").catch(() => null);
      if (activeResultsImageId === targetImageId) {
        report.observations.results.active_image_matches_predicted = true;
      } else {
        const imageLocator = resultsPanel.locator(`.results-image-list [data-image-id="${targetImageId}"]`).first();
        if (await imageLocator.count()) {
        try {
          await timed(report, "results_load_predicted_image", async () => {
            await imageLocator.scrollIntoViewIfNeeded();
            await imageLocator.click({ force: true });
            await page.waitForFunction(() => {
              const panel = document.querySelector(".tab-panel.active");
              if (!panel) return false;
              return Array.from(panel.querySelectorAll(".section-title")).some((node) =>
                (node.textContent || "").trim() === "Image Prediction",
              ) && Array.from(panel.querySelectorAll(".card")).some((node) =>
                (node.textContent || "").includes("Mean confidence"),
              );
            }, { timeout: 20_000 });
          });
        } catch (error) {
          report.blockers.push({
            area: "results",
            message: `Predicted image ${targetImageId} did not load usable prediction details: ${String(error)}`,
          });
        }
        } else {
          report.observations.results.predicted_image_not_directly_visible = targetImageId;
        }
      }
    }

    await timed(report, "results_keyboard_next_image", async () => {
      const list = resultsPanel.getByRole("listbox", { name: "Prediction images" });
      const active = resultsPanel.locator(".results-image-list .card.active");
      const before = await active.getAttribute("data-image-id");
      await list.focus();
      await page.keyboard.press("ArrowDown");
      await page.waitForFunction(
        (prevId) => {
          const panel = document.querySelector(".tab-panel.active");
          const next = panel && panel.querySelector(".results-image-list .card.active");
          return next && next.getAttribute("data-image-id") !== prevId;
        },
        before,
        { timeout: 10_000 },
      );
    });

    // Wait for prediction section to stabilize (loading clears in <100ms typically)
    await page.waitForFunction(() => {
      const panel = document.querySelector(".tab-panel.active");
      if (!panel) return false;
      const text = panel.textContent || "";
      return !text.includes("Loading prediction") && !text.includes("Loading selected prediction");
    }, { timeout: 5000 }).catch(() => {});

    report.observations.results = {
      ...report.observations.results,
      preview_title: await resultsPanel.locator(".results-preview-title").innerText().catch(() => null),
      preview_subtitle: await resultsPanel.locator(".results-preview-subtitle").innerText().catch(() => null),
      image_prediction_text: await resultsPanel.locator(".section").filter({ hasText: "Image Prediction" }).first().innerText().catch(() => null),
    };
    await snapshot(page, outputDir, "results");
  } finally {
    await browser.close();
  }

  if (report.console.errors.length > 0) {
    report.blockers.push({
      area: "browser-console",
      message: `${report.console.errors.length} unexpected console error(s) were captured.`,
    });
  }
  if (report.console.page_errors.length > 0) {
    report.blockers.push({
      area: "browser-runtime",
      message: `${report.console.page_errors.length} page error(s) were captured.`,
    });
  }
  // Tutorial video 404s are expected (some videos don't exist yet) — exclude from blocker count.
  const nonTutorialFailedRequests = report.network.failed_requests.filter(
    (r) => !r.url.includes("/tutorials/"),
  );
  const nonTutorialBadResponses = report.network.bad_responses.filter(
    (r) => !r.url.includes("/tutorials/"),
  );
  if (nonTutorialFailedRequests.length > 0 || nonTutorialBadResponses.length > 0) {
    report.blockers.push({
      area: "network",
      message: `Network issues captured: ${nonTutorialFailedRequests.length} failed request(s), ${nonTutorialBadResponses.length} server error response(s).`,
    });
  }

  report.finished_at = new Date().toISOString();
  report.summary = {
    blocker_count: report.blockers.length,
    discomfort_count: report.discomforts.length,
    console_error_count: report.console.errors.length,
    warning_count: report.console.warnings.length,
    artifact_dir: outputDir,
  };

  const reportPath = path.join(outputDir, "workflow-audit-report.json");
  await writeJson(reportPath, report);

  console.log(
    JSON.stringify(
      {
        report_path: reportPath,
        blocker_count: report.summary.blocker_count,
        discomfort_count: report.summary.discomfort_count,
        selected_project: report.selected_project,
        selected_run: report.selected_run,
      },
      null,
      2,
    ),
  );
}

main().catch(async (error) => {
  console.error(error);
  process.exitCode = 1;
});
