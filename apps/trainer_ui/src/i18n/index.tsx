// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import ja from "./ja";
import en from "./en";
import { errorText, storedLang, type ErrorKind, type Lang } from "./errorText";

export type { Lang };
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
  const [lang, setLangState] = useState<Lang>(storedLang);

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
 * Format an error for user display: a coded server error with its code and
 * in the screen's language (errorText), anything else as it says itself.
 * Works with both ApiError (from api.ts) and plain Error.
 */
export function formatError(err: unknown, lang: Lang = "ja"): string {
  if (!err) return dictionaries[lang]["error.generic"];
  if (err && typeof err === "object" && "code" in err) {
    const api = err as { code: string | null; kind?: ErrorKind; serverMessage?: string };
    const said = api.serverMessage ?? (err instanceof Error ? err.message : String(err));
    const kind = api.kind ?? "uncoded";
    // An uncoded error carries the bucket code of its status, which names
    // nothing: shown without it.
    if (api.code && kind !== "uncoded") return `[${api.code}] ${errorText(api.code, kind, said, lang)}`;
    return said;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}


