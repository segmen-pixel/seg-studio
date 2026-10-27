// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import ja from "./ja";
import en from "./en";

export type Lang = "ja" | "en";

/** What a server error's message is to its code (trainer_api error_handlers.KINDS). */
export type ErrorKind = "usual" | "specific" | "uncoded";

const dictionaries = { ja, en } as const;

/** The language the screen was last set to; Japanese where nothing is stored. */
export function storedLang(): Lang {
  try {
    const stored = localStorage.getItem("seg-lang");
    if (stored === "en" || stored === "ja") return stored;
  } catch {
    // no storage: the default
  }
  return "ja";
}

/**
 * A server error in the screen's language.
 *
 * The trainer answers in English. A coded error whose message is the code's
 * usual one is said in the dictionary's words. One that says more -- a file's
 * name, a limit -- keeps what the server said beside them, in Japanese; in
 * English the server's sentence already is that. An uncoded error (an
 * HTTPException, which carries only the bucket code of its status) is said
 * as the server said it: the bucket's words in its place told a person "a
 * required parameter is missing" whatever had been refused.
 */
export function errorText(code: string | null, kind: ErrorKind, message: string, lang: Lang): string {
  if (!code || kind === "uncoded") return message;
  const usual = (dictionaries[lang] as Record<string, string>)[`error.${code}`];
  if (!usual) return message;
  if (kind === "usual") return usual;
  return lang === "ja" ? `${usual}（${message}）` : message;
}
