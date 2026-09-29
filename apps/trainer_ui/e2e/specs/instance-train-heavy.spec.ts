// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * @heavy — Full instance-mode training smoke (design_instance_segmentation_v098 §6).
 *
 * Seeds a dedicated project with synthetic images + raw masks, runs a real
 * 1-epoch RF-DETR-Seg Nano training through the API (compose → subprocess
 * train → calibration → contract), then exercises the instance predict
 * route. Needs the dev-box GPU + rfdetr — skips honestly when the GPU is
 * busy or the dependency is absent (counted by the skip budget).
 */
import { test, expect, request as pwRequest, type APIRequestContext } from "@playwright/test";
import { encodePngRgb } from "../helpers/png";

const API = "http://localhost:8002/api/v1";
// Must sort AFTER the zz-e2e-seed-* projects: several specs pick the first
// project alphabetically, and a smoke project sorting first hijacks them
// (observed: report-flow clicking the smoke run's disabled report button).
const PROJECT_NAME = "zz-e2e-zz-inst-smoke";
const SIZE = 128;
const TRAIN_TIMEOUT_MS = 8 * 60_000;

/** 3 separated 20x28 rectangles per image; deterministic per-image offsets. */
function seedImageAndMask(k: number): { png: Buffer; mask: Uint8Array } {
  const rgb = new Uint8Array(SIZE * SIZE * 3).fill(225 - k * 2);
  const mask = new Uint8Array(SIZE * SIZE).fill(255);
  const spots: Array<[number, number]> = [
    [12 + (k % 3) * 2, 10],
    [78, 16 + (k % 4) * 2],
    [40, 88 + (k % 3) * 2],
  ];
  for (const [ox, oy] of spots) {
    for (let y = oy; y < oy + 28; y++) {
      for (let x = ox; x < ox + 20; x++) {
        const i = y * SIZE + x;
        mask[i] = 1;
        rgb[i * 3] = 70;
        rgb[i * 3 + 1] = 90 + k * 3;
        rgb[i * 3 + 2] = 180;
      }
    }
  }
  return { png: encodePngRgb(SIZE, SIZE, rgb), mask };
}

async function seedProject(ctx: APIRequestContext): Promise<string> {
  const projects = (await (await ctx.get(`${API}/projects`)).json()) as Array<{ id: string; name: string }>;
  let proj = projects.find((p) => p.name === PROJECT_NAME);
  if (!proj) {
    proj = (await (await ctx.post(`${API}/projects`, { data: { name: PROJECT_NAME } })).json()) as {
      id: string; name: string;
    };
  }
  const index = await ctx.get(`${API}/projects/${proj.id}/datasets/annotate`);
  const items = index.ok() ? (((await index.json()) as { items?: unknown[] }).items ?? []) : [];
  if (items.length < 8) {
    for (let k = 0; k < 8; k++) {
      const { png, mask } = seedImageAndMask(k);
      const name = `inst-${String(k).padStart(2, "0")}`;
      const up = await ctx.post(`${API}/projects/${proj.id}/datasets/annotate/upload`, {
        multipart: { files: { name: `${name}.png`, mimeType: "image/png", buffer: png } },
      });
      expect(up.ok()).toBeTruthy();
      const put = await ctx.put(
        `${API}/projects/${proj.id}/datasets/annotate/masks/${name}.png?raw=1&w=${SIZE}&h=${SIZE}`,
        { headers: { "content-type": "application/octet-stream" }, data: Buffer.from(mask) },
      );
      expect(put.ok()).toBeTruthy();
    }
  }
  return proj.id;
}

test.describe("Instance Training Smoke (@heavy)", () => {
  test("instance mode trains end-to-end and serves instances.json", async () => {
    test.setTimeout(TRAIN_TIMEOUT_MS + 120_000);
    const ctx = await pwRequest.newContext();
    try {
      const gs = await ctx.get(`${API}/train/global-status`);
      const busy = gs.ok() ? ((await gs.json()) as { gpu_busy?: boolean }).gpu_busy === true : true;
      test.skip(busy, "GPU busy — skipping instance train smoke");

      const pid = await seedProject(ctx);

      // Preview doubles as the rfdetr-free part of the pipeline check.
      const preview = await ctx.post(`${API}/projects/${pid}/train/instance-preview`, {
        data: { instance_objects_min: 2, instance_objects_max: 3, n_samples: 2 },
      });
      expect(preview.ok()).toBeTruthy();
      expect(((await preview.json()) as { samples: unknown[] }).samples.length).toBe(2);

      const start = await ctx.post(`${API}/projects/${pid}/train`, {
        data: {
          model_name: "e2e-inst-smoke",
          training_mode: "instance",
          epochs: 1,
          auto_epochs: false,
          batch_size: 2,
          instance_n_train: 8,
          instance_n_val: 2,
          instance_objects_min: 2,
          instance_objects_max: 3,
          instance_seed: 7,
        },
      });
      expect(start.ok()).toBeTruthy();
      const runId = ((await start.json()) as { run_id: string }).run_id;

      const deadline = Date.now() + TRAIN_TIMEOUT_MS;
      let status = "";
      while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 10_000));
        const rr = await ctx.get(`${API}/projects/${pid}/train/runs`);
        const list = rr.ok() ? ((await rr.json()) as Array<{ run_id: string; status: string }>) : [];
        status = list.find((r) => r.run_id === runId)?.status ?? "";
        if (["completed", "failed", "stopped"].includes(status)) break;
      }
      if (status === "failed") {
        const logs = await ctx.get(`${API}/projects/${pid}/train/runs/${runId}/logs`);
        const text = logs.ok() ? ((await logs.json()) as { log?: string }).log ?? "" : "";
        test.skip(text.includes("rfdetr is not installed"),
          "rfdetr not installed — skipping instance train smoke");
        expect(status, `run failed:\n${text.slice(-1500)}`).toBe("completed");
      }
      expect(status).toBe("completed");

      // Contract-backed predict route (triggers one real inference).
      const inst = await ctx.get(
        `${API}/projects/${pid}/train/runs/${runId}/predict/inst-00/instances.json`,
        { timeout: 120_000 },
      );
      expect(inst.ok()).toBeTruthy();
      const body = (await inst.json()) as { count: number; threshold: number; dedup_iou: number };
      expect(body.count).toBeGreaterThanOrEqual(0);
      expect(body.dedup_iou).toBeCloseTo(0.7, 5);
      expect(typeof body.threshold).toBe("number");

      // Success: remove the smoke project so it never leaks into other
      // specs' project lists. On failure it stays behind for debugging
      // (the zz-e2e-zz- prefix keeps it sorted last either way).
      await ctx.delete(`${API}/projects/${pid}`);
    } finally {
      await ctx.dispose();
    }
  });
});
