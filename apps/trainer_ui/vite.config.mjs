// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { copyFileSync, readFileSync } from "fs";
import { fileURLToPath } from "url";

const pkg = JSON.parse(readFileSync("./package.json", "utf-8"));

// Shown after the version in the header and the About dialog. Empty for a
// final release; a release-candidate branch carries "RC" so a machine with
// more than one build installed says which one is running without anyone
// opening a terminal. Clear this in the commit that cuts the final release.
const APP_CHANNEL = "";
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
    __APP_CHANNEL__: JSON.stringify(APP_CHANNEL),
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
