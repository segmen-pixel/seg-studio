// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import ja from "./ja";
import en from "./en";

export type Lang = "ja" | "en";
export type TranslationKey = keyof typeof ja;

const dictionaries = { ja, en } as const;

/* ── Context ── */

interface I18nContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: TranslationKey) => string;
}

const I18nContext = createContext<I18nContextValue>({
  lang: "ja",
  setLang: () => {},
  t: (key) => key,
});

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    const stored = localStorage.getItem("seg-lang");
    if (stored === "en" || stored === "ja") return stored;
    return "ja";
  });

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    localStorage.setItem("seg-lang", l);
  }, []);

  const t = useCallback((key: TranslationKey): string => {
    return dictionaries[lang][key] ?? key;
  }, [lang]);

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}

/**
 * Format an error for user display. Uses i18n-mapped NSS code if available,
 * otherwise falls back to the error message.
 * Works with both ApiError (from api.ts) and plain Error.
 */
export function formatError(err: unknown, lang: Lang = "ja"): string {
  if (!err) return dictionaries[lang]["error.generic"];
  const dict = dictionaries[lang];
  // ApiError with NSS code
  if (err && typeof err === "object" && "code" in err) {
    const code = (err as { code: string | null }).code;
    if (code) {
      const key = `error.${code}` as TranslationKey;
      if (key in dict) return `[${code}] ${dict[key]}`;
      // Code exists but no i18n mapping — use the error message
      const msg = err instanceof Error ? err.message : String(err);
      return `[${code}] ${msg}`;
    }
  }
  if (err instanceof Error) return err.message;
  return String(err);
}


