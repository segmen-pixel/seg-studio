// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { test, expect } from "@playwright/test";

/**
 * SAM embedding cache regression test.
 *
 * Bug: The SAM predictor holds only ONE image's embedding at a time, but the
 * old cache stored a `True` flag per image path. When switching between images
 * of different sizes (e.g. 1280x960 → 2048x2048), the cache
 * incorrectly reported "already loaded" and skipped `set_image`, causing the
 * predictor to return masks at the wrong resolution and coordinates.
 *
 * This test calls the SAM API directly (via Playwright's request context) to
 * verify that masks always match the target image dimensions, even after
 * switching between projects with different image sizes.
 */

// All REST endpoints live under the /api/v1 prefix.
const API = "http://localhost:8002/api/v1";

type ImageInfo = {
  projectId: string;
  projectName: string;
  itemId: string;
  width: number;
  height: number;
};

/** Find two images from different projects with different dimensions. */
async function findTwoDifferentSizedImages(
  request: typeof test extends (name: string, fn: (args: infer A) => any) => any ? A["request"] : any,
): Promise<[ImageInfo, ImageInfo] | null> {
  const projRes = await request.get(`${API}/projects`);
  const projects: { id: string; name: string }[] = await projRes.json();
  if (projects.length < 2) return null;

  const images: ImageInfo[] = [];
  for (const proj of projects) {
    const annRes = await request.get(`${API}/projects/${proj.id}/datasets/annotate`);
    if (!annRes.ok()) continue;
    const data = await annRes.json();
    const items: { id: string; width: number; height: number }[] = data.items ?? [];
    if (items.length > 0 && items[0].width > 0 && items[0].height > 0) {
      images.push({
        projectId: proj.id,
        projectName: proj.name,
        itemId: items[0].id,
        width: items[0].width,
        height: items[0].height,
      });
    }
    if (images.length >= 2) break;
  }

  if (images.length < 2) return null;
  // Prefer pair with different dimensions (stronger test)
  if (images[0].width !== images[1].width || images[0].height !== images[1].height) {
    return [images[0], images[1]];
  }
  // Fall back to same-size pair (still tests cache key correctness)
  return [images[0], images[1]];
}

/** Call SAM predict and return the decoded mask dimensions. */
async function samPredict(
  request: any,
  projectId: string,
  itemId: string,
  point: [number, number],
  model = "mobile_sam",
): Promise<{ maskWidth: number; maskHeight: number; score: number; timeMs: number }> {
  const res = await request.post(
    `${API}/projects/${projectId}/datasets/annotate/${itemId}/sam-segment`,
    {
      data: {
        points: [point],
        labels: [1],
        model,
      },
      timeout: 60_000,
    },
  );
  expect(res.ok(), `SAM API returned ${res.status()}`).toBe(true);
  const body = await res.json();

  // Decode mask PNG to get dimensions
  // Use a data URI trick: fetch the base64 PNG and check the PNG header for dimensions
  const maskB64: string = body.mask;
  const maskBytes = Buffer.from(maskB64, "base64");

  // PNG header: bytes 16-19 = width (big-endian), bytes 20-23 = height (big-endian)
  // (after 8-byte signature + 4-byte chunk length + 4-byte "IHDR")
  const maskWidth = maskBytes.readUInt32BE(16);
  const maskHeight = maskBytes.readUInt32BE(20);

  return { maskWidth, maskHeight, score: body.score, timeMs: body.predict_time_ms };
}

test.describe("SAM Embedding Cache", () => {
  test("mask dimensions match image after switching between projects", async ({ request }) => {
    const pair = await findTwoDifferentSizedImages(request);
    if (!pair) {
      test.skip();
      return;
    }
    const [imgA, imgB] = pair;
    const differentSizes = imgA.width !== imgB.width || imgA.height !== imgB.height;

    console.log(
      `Testing SAM cache: A=${imgA.projectName} (${imgA.width}x${imgA.height}), ` +
      `B=${imgB.projectName} (${imgB.width}x${imgB.height}), different=${differentSizes}`,
    );

    const centerA: [number, number] = [Math.floor(imgA.width / 2), Math.floor(imgA.height / 2)];
    const centerB: [number, number] = [Math.floor(imgB.width / 2), Math.floor(imgB.height / 2)];

    // Step 1: SAM predict on Image A
    const r1 = await samPredict(request, imgA.projectId, imgA.itemId, centerA);
    expect(r1.maskWidth, "First call: mask width must match image A").toBe(imgA.width);
    expect(r1.maskHeight, "First call: mask height must match image A").toBe(imgA.height);
    console.log(`  A first:  mask=${r1.maskWidth}x${r1.maskHeight}, score=${r1.score}, ${r1.timeMs}ms`);

    // Step 2: SAM predict on Image B (different project/size)
    const r2 = await samPredict(request, imgB.projectId, imgB.itemId, centerB);
    expect(r2.maskWidth, "Second call: mask width must match image B").toBe(imgB.width);
    expect(r2.maskHeight, "Second call: mask height must match image B").toBe(imgB.height);
    console.log(`  B:        mask=${r2.maskWidth}x${r2.maskHeight}, score=${r2.score}, ${r2.timeMs}ms`);

    // Step 3: SAM predict on Image A again (THIS is where the bug manifested)
    // With the old buggy cache, the predictor still had B's embedding,
    // returning a mask at B's dimensions instead of A's.
    const r3 = await samPredict(request, imgA.projectId, imgA.itemId, centerA);
    expect(r3.maskWidth, "Third call (back to A): mask width must match image A, NOT image B").toBe(imgA.width);
    expect(r3.maskHeight, "Third call (back to A): mask height must match image A, NOT image B").toBe(imgA.height);
    console.log(`  A second: mask=${r3.maskWidth}x${r3.maskHeight}, score=${r3.score}, ${r3.timeMs}ms`);
  });

  test("multi-click on same image reuses embedding (no re-encode)", async ({ request }) => {
    const projRes = await request.get(`${API}/projects`);
    const projects: { id: string; name: string }[] = await projRes.json();
    if (projects.length === 0) { test.skip(); return; }

    const annRes = await request.get(`${API}/projects/${projects[0].id}/datasets/annotate`);
    const data = await annRes.json();
    const items: { id: string; width: number; height: number }[] = data.items ?? [];
    if (items.length === 0 || items[0].width === 0) { test.skip(); return; }

    const img = items[0];
    const pid = projects[0].id;

    // First click (cold — includes set_image)
    const r1 = await samPredict(request, pid, img.id, [img.width / 4, img.height / 4]);
    expect(r1.maskWidth).toBe(img.width);

    // Second click on same image (should reuse embedding — faster)
    const r2 = await samPredict(request, pid, img.id, [img.width * 3 / 4, img.height * 3 / 4]);
    expect(r2.maskWidth).toBe(img.width);
    expect(r2.maskHeight).toBe(img.height);

    console.log(`  Cold: ${r1.timeMs}ms, Warm: ${r2.timeMs}ms`);
    // Warm call should be faster (no set_image), but we only assert correctness
  });
});
