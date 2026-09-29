// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * Barrel export — import everything from "@e2e/helpers" or "./helpers".
 */
export { SEL } from "./selectors";
export { waitForApi, waitForProjects, waitForCanvasStable, waitForTab } from "./waiters";
export { switchTab, selectFirstProject, selectProjectByName, selectProjectByIndex } from "./nav";
export type { TabName } from "./nav";
export { useConsoleErrorCollector } from "./assertions";
