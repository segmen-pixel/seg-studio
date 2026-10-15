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

// The dev server relays the API to the trainer on this machine, so every
// relayed request reaches the trainer from 127.0.0.1. The trainer reads that
// peer address as "the operator's own browser" and lets it in without the LAN
// token, which is only true while the browser really is on this machine. The
// start scripts bind the dev server to 127.0.0.1, but `npm run dev -- --host`
// is one flag away, and through it anyone on the network could read the token
// from /api/v1/auth/token. So a request from any other address is stamped
// X-Forwarded-For, which the trainer reads as relayed and holds to the token.
// Not http-proxy's `xfwd`: that stamps every request, the operator's own too,
// and a trainer with no token (the loopback default) then refuses the dev UI.
const LOOPBACK_PEERS = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);
const stampRemotePeers = (proxy) => {
  const stamp = (proxyReq, req) => {
    const peer = req.socket?.remoteAddress ?? "";
    if (!LOOPBACK_PEERS.has(peer)) proxyReq.setHeader("X-Forwarded-For", peer || "unknown");
  };
  proxy.on("proxyReq", stamp);
  proxy.on("proxyReqWs", stamp);
};

// Every entry is an object. Vite expands a bare string target to
// { target, changeOrigin: true }, which rewrites Host to localhost:8002 while
// the browser's Origin stays localhost:5173; the trainer's CSRF check compares
// the two, so every POST to /v2 (the Live tab's camera and session calls)
// would be refused with a 403.
const toTrainer = (target, extra = {}) => ({ target, configure: stampRemotePeers, ...extra });

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
      "/api": toTrainer("http://localhost:8002", { ws: true }),
      "/v2": toTrainer("http://localhost:8002"),
      "/ws": toTrainer("ws://localhost:8002", { ws: true }),
      "/health": toTrainer("http://localhost:8002"),
      "/version": toTrainer("http://localhost:8002"),
      "/startup-status": toTrainer("http://localhost:8002"),
    },
  },
}));
