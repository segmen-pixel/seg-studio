// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { copyFileSync, readFileSync } from "fs";
import { fileURLToPath } from "url";

const pkg = JSON.parse(readFileSync("./package.json", "utf-8"));
const here = fileURLToPath(new URL(".", import.meta.url));

// Ship the repo-level third-party notices inside every built bundle dir.
// BSD/MIT terms of bundled npm packages (openseadragon, react, zustand,
// ...) require the copyright notice to travel with the distribution;
// minification strips source comments, so the aggregated notices file
// must ride along with dist/ on every channel (installer, zip, docker).
const copyThirdPartyNotices = () => ({
  name: "copy-third-party-notices",
  closeBundle() {
    copyFileSync(
      `${here}/../../THIRD_PARTY_NOTICES.md`,
      `${here}/dist/THIRD_PARTY_NOTICES.md`,
    );
  },
});

export default defineConfig(({ mode }) => ({
  base: "./",
  plugins: [react(), copyThirdPartyNotices()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __BUILD_DATE__: JSON.stringify(new Date().toISOString().slice(0, 10)),
  },
  // Strip console.log/debug/trace and debugger statements from prod bundles
  // so [DBG] logs from development don't leak into the shipped UI.
  // console.warn / console.error are preserved so real problems still surface.
  esbuild: mode === "production" ? {
    pure: ["console.log", "console.debug", "console.trace", "console.info"],
    drop: ["debugger"],
    // Keep /*! ... */ and @license banners through minification —
    // stripping them breaks the BSD/MIT notice-retention terms.
    legalComments: "inline",
  } : undefined,
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://localhost:8002", ws: true },
      "/v2": "http://localhost:8002",
      "/ws": { target: "ws://localhost:8002", ws: true },
      "/health": "http://localhost:8002",
      "/version": "http://localhost:8002",
      "/startup-status": "http://localhost:8002",
    },
  },
}));
