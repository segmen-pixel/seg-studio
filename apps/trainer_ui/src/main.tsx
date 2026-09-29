// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./ErrorBoundary";
import TokenGate from "./app/components/TokenGate";
import { I18nProvider } from "./i18n";
import "./styles/index.css";

/* Log unhandled promise rejections to the console */
window.onunhandledrejection = (event: PromiseRejectionEvent) => {
  console.error("[UnhandledRejection]", event.reason);
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <I18nProvider>
        <TokenGate>
          <App />
        </TokenGate>
      </I18nProvider>
    </ErrorBoundary>
  </React.StrictMode>
);
