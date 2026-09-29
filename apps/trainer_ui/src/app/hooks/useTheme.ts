// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useEffect, useState } from "react";
import type { ThemeMode } from "../types";

function applyTheme(mode: ThemeMode) {
  if (mode === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", mode);
  }
}

export function useTheme() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem("seg-theme");
    return (saved === "light" || saved === "dark" || saved === "system") ? saved : "system";
  });

  useEffect(() => {
    applyTheme(themeMode);
    localStorage.setItem("seg-theme", themeMode);
  }, [themeMode]);

  function cycleTheme() {
    setThemeMode((prev) => {
      if (prev === "system") return "light";
      if (prev === "light") return "dark";
      return "system";
    });
  }

  return { themeMode, cycleTheme } as const;
}
