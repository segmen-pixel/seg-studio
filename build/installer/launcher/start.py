# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8002
URL = f"http://localhost:{PORT}/ui/"


def _open_browser() -> None:
    time.sleep(2.0)
    try:
        webbrowser.open(URL)
    except Exception:
        pass


def main() -> None:
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        "apps.trainer_api.app.main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
