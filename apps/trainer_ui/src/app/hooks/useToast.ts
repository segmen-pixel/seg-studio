// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useRef, useState } from "react";

/**
 * Whether a status message reports a failure: it is then drawn as one and
 * stays until it is clicked. Asked of the message's own words, in both of
 * the screen's languages -- with the English words alone, a Japanese
 * "作成失敗: ..." was drawn as an ordinary note and gone after four seconds.
 */
export function isFailureText(msg: string): boolean {
  return /fail|error|失敗|エラー/i.test(msg);
}

export function useToast() {
  const [toastMsg, setToastMsg] = useState("");
  const [toastCopied, setToastCopied] = useState(false);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastHoveredRef = useRef(false);

  const showToast = useCallback((msg: string) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToastMsg(msg);
    setToastCopied(false);
    const isError = isFailureText(msg);
    if (msg && !isError) {
      toastTimer.current = setTimeout(() => {
        if (!toastHoveredRef.current) setToastMsg("");
      }, 4000);
    }
  }, []);

  const handleToastClick = useCallback(() => {
    if (toastMsg) {
      navigator.clipboard.writeText(toastMsg).then(() => {
        setToastCopied(true);
        setTimeout(() => { setToastCopied(false); setToastMsg(""); }, 800);
      }).catch(() => {});
    }
  }, [toastMsg]);

  const handleToastHoverEnter = useCallback(() => {
    toastHoveredRef.current = true;
    if (toastTimer.current) { clearTimeout(toastTimer.current); toastTimer.current = null; }
  }, []);

  const handleToastHoverLeave = useCallback(() => {
    toastHoveredRef.current = false;
    if (toastMsg && !isFailureText(toastMsg)) {
      toastTimer.current = setTimeout(() => setToastMsg(""), 2000);
    }
  }, [toastMsg]);

  return {
    toastMsg, toastCopied, showToast,
    handleToastClick, handleToastHoverEnter, handleToastHoverLeave,
  } as const;
}
