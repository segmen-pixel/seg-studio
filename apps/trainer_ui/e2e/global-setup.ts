// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Global setup — idempotent seeding of deterministic test projects.
 *
 * Many specs guard on server data ("need >= 2 projects", "no images") and
 * silently skip when it is missing, so effective coverage used to depend on
 * whatever happened to be on the dev server. This creates two clearly-named
 * seed projects (zz-e2e-seed-1/2) with 3 tiny images and one painted mask
 * each, so those guards stay satisfied on any instance.
 *
 * Never deletes or modifies anything else. Safe to re-run: existing seed
 * projects are detected by name and left as-is (images are re-uploaded
 * only when the annotate index is empty).
 */
import { request, type FullConfig } from "@playwright/test";
import { SEED_IMAGES, seedMask } from "./helpers/seed-assets";

const API = "http://localhost:8002/api/v1";
export const SEED_PROJECT_NAMES = ["zz-e2e-seed-1", "zz-e2e-seed-2"];

export default async function globalSetup(_config: FullConfig): Promise<void> {
  const ctx = await request.newContext();
  try {
    const res = await ctx.get(`${API}/projects`);
    if (!res.ok()) {
      console.warn(`[e2e-seed] GET /projects -> ${res.status()}; skipping seeding`);
      return;
    }
    const projects = (await res.json()) as Array<{ id: string; name: string }>;
    const seeded: Record<string, { id: string; name: string }> = {};

    for (const name of SEED_PROJECT_NAMES) {
      let proj = projects.find((p) => p.name === name);
      if (!proj) {
        const created = await ctx.post(`${API}/projects`, { data: { name } });
        if (!created.ok()) {
          console.warn(`[e2e-seed] create ${name} -> ${created.status()}; skipping`);
          continue;
        }
        proj = (await created.json()) as { id: string; name: string };
        console.log(`[e2e-seed] created project ${name} (${proj.id})`);
      }
      seeded[name] = proj;

      const index = await ctx.get(`${API}/projects/${proj.id}/datasets/annotate`);
      const items = index.ok()
        ? (((await index.json()) as { items?: unknown[] }).items ?? [])
        : [];
      if (items.length < Object.keys(SEED_IMAGES).length) {
        // The endpoint accepts repeated "files" fields; upload one file per
        // request for clarity.
        for (const [fname, b64] of Object.entries(SEED_IMAGES)) {
          const up = await ctx.post(`${API}/projects/${proj.id}/datasets/annotate/upload`, {
            multipart: { files: { name: fname, mimeType: "image/png", buffer: Buffer.from(b64, "base64") } },
          });
          if (!up.ok()) console.warn(`[e2e-seed] upload ${fname} -> ${up.status()}`);
        }
      }

      // Paint masks (class 1 square) on the first two images so region
      // labels / mask flows always have data. Idempotent, cheap.
      const mask = Buffer.from(seedMask());
      for (const item of ["seed-01", "seed-02"]) {
        const put = await ctx.put(
          `${API}/projects/${proj.id}/datasets/annotate/masks/${item}.png?raw=1&w=64&h=64`,
          { headers: { "content-type": "application/octet-stream" }, data: mask },
        );
        if (!put.ok()) console.warn(`[e2e-seed] mask put ${item} -> ${put.status()}`);
      }
      console.log(`[e2e-seed] seeded ${name}: ${Object.keys(SEED_IMAGES).length} images + 2 masks`);
    }

    // Trained fixture: report/results specs need a completed run on seed-2.
    // Train one tiny run the first time only (~2 min on the dev GPU), and
    // never when the GPU is busy with real work.
    const seed2 = seeded["zz-e2e-seed-2"];
    if (seed2) {
      const runsRes = await ctx.get(`${API}/projects/${seed2.id}/train/runs`);
      const runs = runsRes.ok() ? ((await runsRes.json()) as Array<{ status: string }>) : [];
      const hasCompleted = runs.some((r) => r.status === "completed");
      if (!hasCompleted) {
        const gs = await ctx.get(`${API}/train/global-status`);
        const busy = gs.ok() ? ((await gs.json()) as { gpu_busy?: boolean }).gpu_busy === true : true;
        if (busy) {
          console.warn("[e2e-seed] GPU busy — skipping trained-fixture run (run-dependent specs will skip)");
        } else {
          console.log("[e2e-seed] training the fixture run on zz-e2e-seed-2 (first time only, ~2 min)…");
          const start = await ctx.post(`${API}/projects/${seed2.id}/train`, {
            data: { model_name: "e2e-fixture", training_mode: "standard", epochs: 5, auto_epochs: false },
          });
          if (!start.ok()) {
            console.warn(`[e2e-seed] fixture training start -> ${start.status()}; run-dependent specs will skip`);
          } else {
            const deadline = Date.now() + 5 * 60_000;
            let finalStatus = "";
            while (Date.now() < deadline) {
              await new Promise((r) => setTimeout(r, 5_000));
              const rr = await ctx.get(`${API}/projects/${seed2.id}/train/runs`);
              const list = rr.ok() ? ((await rr.json()) as Array<{ status: string }>) : [];
              finalStatus = list[0]?.status ?? "";
              if (["completed", "failed", "stopped"].includes(finalStatus)) break;
            }
            console.log(`[e2e-seed] fixture run status: ${finalStatus || "timeout"}`);
          }
        }
      }
    }
  } finally {
    await ctx.dispose();
  }
}
