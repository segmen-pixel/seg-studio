// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import { useCallback, useRef, useState } from "react";

export function useToast() {
  const [toastMsg, setToastMsg] = useState("");
  const [toastCopied, setToastCopied] = useState(false);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toastHoveredRef = useRef(false);

  const showToast = useCallback((msg: string) => {
    if (toastTimer.current) clearTimeout(toastTimer.current);
    setToastMsg(msg);
    setToastCopied(false);
    const isError = /fail|error/i.test(msg);
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
    if (toastMsg && !/fail|error/i.test(toastMsg)) {
      toastTimer.current = setTimeout(() => setToastMsg(""), 2000);
    }
  }, [toastMsg]);

  return {
    toastMsg, toastCopied, showToast,
    handleToastClick, handleToastHoverEnter, handleToastHoverLeave,
  } as const;
}
