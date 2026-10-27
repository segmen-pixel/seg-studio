# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio as an app: start the local services, open the screen, and stay
in the taskbar and the tray (Windows) or the Dock and the menu bar (macOS)
with Open, Logs and Stop.

The installers create the entry point. On Windows it is a Start menu and a
desktop shortcut (scripts/windows/_create_shortcuts.ps1) that runs this file
with the venv's pythonw.exe, so no console window opens; on macOS it is
Seg-Studio.app (scripts/macos/create_app_macos.sh), whose executable runs it
with the venv's Python from inside the app, so that the Dock shows Seg-Studio.
Nothing is downloaded and nothing is signed: the
shortcut and the app are written on the machine itself, after the install.

The services are the ones start-windows.bat and start-macos.sh start, through
the same scripts, so there is one way of starting them. The app runs the
start script without a console, without the development screen server
(SEG_NO_DEV_UI=1) and without letting the script open the browser
(SEG_NO_BROWSER=1); it waits for the API itself and then opens
http://localhost:8002/ui/. Stop runs the stop script, after asking when a
training, an inference or an AI support run is going, and only when the
services are this folder's or the app started them.

The API's /startup-status names the checkout it runs from (``checkout``, from
checkout_id). When the Seg-Studio on the port comes from another folder, or
from a version that does not say, the app asks before it stops that one and
starts its own, instead of opening someone else's screen as if it were its own.

``--self-test`` shows the tray icon or the menu bar item for a moment and
exits without starting anything (an installer or a person can check that the
front end works on this machine).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8002"
UI_URL = "http://localhost:8002/ui/"
ICON = ROOT / "scripts" / "installer_assets" / "seg-studio.ico"
#: The bundle identifier create_app_macos.sh gives Seg-Studio.app.
MACOS_BUNDLE_ID = "com.segmen-pixel.seg-studio.checkout"
#: The Windows AppUserModelID of the app's taskbar button, which
#: _create_shortcuts.ps1 also gives the shortcuts: the taskbar then takes the
#: running app for the Start menu's Seg-Studio, and pins that shortcut.
APP_USER_MODEL_ID = "SegmenPixel.SegStudio"
#: The macOS menu bar image: black with the S cut out, shown as a template.
MENUBAR_ICON = ROOT / "scripts" / "installer_assets" / "seg-studio-menubar.png"
#: How long a start may take before the app says it failed. The first start
#: after an install builds caches and loads the models' code: a minute or two.
START_TIMEOUT_S = 240

LABELS = {
    "ja": {
        "open": "Seg-Studio を開く",
        "logs": "ログのフォルダを開く",
        "quit": "停止して終了",
        "starting": "Seg-Studio を起動しています…",
        "running": "Seg-Studio（動作中）",
        "stopping": "Seg-Studio を停止しています…",
        "busy": "学習または AI サポートの実行中です。停止すると中断されます。停止しますか？",
        "start_failed": "Seg-Studio を起動できませんでした。ログのフォルダの app_start.log を確認してください。",
        "port_taken": "ポート 8002 を別のプログラムが使っているため、Seg-Studio を起動できません。",
        "no_menu": "メニューバーの部品（pyobjc）が入っていないため、メニューバーには表示しません。",
        "running_ask": "Seg-Studio は動作中です。",
        "yes": "停止する",
        "no": "キャンセル",
        "elsewhere": "別のフォルダの Seg-Studio か、古い版の Seg-Studio が動いています。"
                     "それを停止して、このフォルダの Seg-Studio を起動しますか？",
        "elsewhere_no": "「いいえ」を選ぶと、動いている方をそのまま開きます。",
        "elsewhere_busy": "別のフォルダの Seg-Studio か、古い版の Seg-Studio で、学習または AI サポートが"
                          "実行中です。停止すると中断されます。それでも停止して、このフォルダの Seg-Studio を"
                          "起動しますか？",
        "elsewhere_busy_no": "「いいえ」を選ぶと、動いている方をそのまま開きます。",
        "stop_failed": "動いている Seg-Studio を停止できませんでした。ログのフォルダの app_start.log を確認してください。",
        "restart": "停止して起動",
        "keep": "そのまま開く",
    },
    "en": {
        "open": "Open Seg-Studio",
        "logs": "Open the log folder",
        "quit": "Stop and quit",
        "starting": "Seg-Studio is starting…",
        "running": "Seg-Studio (running)",
        "stopping": "Seg-Studio is stopping…",
        "busy": "A training or an AI support run is going. Stopping ends it. Stop anyway?",
        "start_failed": "Seg-Studio could not start. See app_start.log in the log folder.",
        "port_taken": "Another program is using port 8002, so Seg-Studio cannot start.",
        "no_menu": "The menu bar library (pyobjc) is not installed, so the app shows no menu bar item.",
        "running_ask": "Seg-Studio is running.",
        "yes": "Stop",
        "no": "Cancel",
        "elsewhere": "A Seg-Studio from another folder, or an older version, is running. "
                     "Stop it and start the Seg-Studio in this folder?",
        "elsewhere_no": "No opens the one that is running.",
        "elsewhere_busy": "A Seg-Studio from another folder, or an older version, is running a training or "
                          "an AI support run. Stopping ends it. Stop it anyway and start the Seg-Studio in "
                          "this folder?",
        "elsewhere_busy_no": "No opens the one that is running.",
        "stop_failed": "The running Seg-Studio could not be stopped. See app_start.log in the log folder.",
        "restart": "Stop it and start",
        "keep": "Open the running one",
    },
}

log = logging.getLogger("seg_studio_app")


def lang_from(code: str | None) -> str:
    """The app's language from a locale or language code: Japanese or English."""
    return "ja" if str(code or "").lower().replace("-", "_").startswith("ja") else "en"


def log_dir(platform: str = sys.platform) -> Path:
    """Where the start scripts write their logs, and where the app writes its own."""
    return ROOT / "logs" / ("windows" if platform == "win32" else "macos")


def service_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment the start script runs in: the person's, plus the app's switches.

    SEG_HIDE_TOKEN: the start script's output goes to app_start.log, so a LAN
    start says where the token is kept rather than printing it there."""
    env = dict(os.environ if base is None else base)
    env.update({"SEG_NO_BROWSER": "1", "SEG_NO_DEV_UI": "1", "SEG_NO_PAUSE": "1", "SEG_HIDE_TOKEN": "1"})
    return env


def _cmd_exe() -> str:
    """cmd.exe by its full path: a bare name is looked for in the working directory first."""
    return os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", "cmd.exe")


# On Windows the scripts are named relative to the checkout, which is the
# working directory they run in: given the full path, cmd.exe takes the quotes
# off one that holds a space together with ( ) or &, and splits it there.
def start_command(platform: str = sys.platform) -> list[str]:
    """The start script, as the person would run it."""
    if platform == "win32":
        return [_cmd_exe(), "/d", "/c", r"scripts\windows\start_local_windows.bat"]
    return ["/bin/bash", str(ROOT / "scripts" / "macos" / "start_local_macos.sh")]


def stop_command(platform: str = sys.platform) -> list[str]:
    """The stop script, as the person would run it."""
    if platform == "win32":
        return [_cmd_exe(), "/d", "/c", r"scripts\windows\stop_local_windows.bat"]
    return ["/bin/bash", str(ROOT / "scripts" / "macos" / "stop_local_macos.sh")]


def _hidden(platform: str = sys.platform) -> dict:
    """How a script is started so that no window and no terminal comes with it."""
    if platform == "win32":
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {"start_new_session": True}


def checkout_id(root: Path = ROOT) -> str:
    """A short name for a checkout's folder. The trainer API's /startup-status
    gives the same for the folder it runs from (app/core/startup_state.py).

    It is made from the folder's identity on disk (device and file number),
    not from its path: the app and the API may spell the path differently (a
    path typed in another case stays that way on macOS), and the name must not
    tell anyone the path or the user name in it. Where the file system has no
    file numbers, the path is used."""
    try:
        st = os.stat(root)
    except OSError:
        st = None
    key = f"{st.st_dev}:{st.st_ino}" if st and st.st_ino else os.path.normcase(os.path.realpath(root))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


#: What a reply that cannot be read raises: no connection, a timeout, a
#: program that does not speak HTTP, or a body that is not the JSON expected.
_UNREADABLE = (OSError, http.client.HTTPException, ValueError, TypeError, AttributeError)


def _get(path: str, base: str = API, timeout: float = 3.0) -> tuple[int, bytes]:
    # Opened without a proxy: on a machine that names one and makes no
    # exception for local addresses, urlopen would ask the proxy for 127.0.0.1.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + path, timeout=timeout) as r:
        return r.status, r.read()


def api_state(base: str = API, root: Path = ROOT) -> str:
    """'ready', 'starting' (Seg-Studio answers but is still starting), 'absent'
    (nothing listens), 'other' (something that is not Seg-Studio holds the port),
    or 'elsewhere' (a Seg-Studio from another folder, or one too old to say)."""
    try:
        _, body = _get("/startup-status", base)
    except urllib.error.HTTPError:
        return "other"
    except OSError:              # nothing listens (URLError, a timeout)
        return "absent"
    except http.client.HTTPException:
        return "other"           # something listens, and it does not speak HTTP
    try:
        data = json.loads(body.decode("utf-8"))
    except ValueError:
        return "other"
    if not isinstance(data, dict) or "ready" not in data:
        return "other"
    if data.get("checkout") != checkout_id(root):
        return "elsewhere"
    return "ready" if data.get("ready") else "starting"


def busy(base: str = API) -> bool:
    """Whether stopping now would end a training, an inference or an AI support run.

    gpu_busy alone is true only while every CUDA device is taken, so a run on
    the Mac's GPU, on the CPU or on one of two cards does not set it.
    owner_kind names a job holding a device, training_runs counts the runs the
    API is running or holds queued (on any device, the CPU's included), and
    inference is a batch prediction."""
    try:
        _, body = _get("/api/v1/train/global-status", base)
        status = json.loads(body.decode("utf-8"))
        if (status.get("gpu_busy") or status.get("owner_kind") or status.get("inference")
                or int(status.get("queued_count") or 0) > 0 or int(status.get("training_runs") or 0) > 0):
            return True
        _, body = _get("/api/v1/agent/runs", base)
        return bool((json.loads(body.decode("utf-8")) or {}).get("items"))
    except _UNREADABLE:
        return False


class Controller:
    """What the tray and the menu bar do, without the tray and the menu bar.

    ``say(key)`` shows a state in the front end's tooltip or title,
    ``alert(key)`` tells the person something went wrong, and ``ask(key)``
    asks a yes-or-no question and returns True for yes; all three are called
    from worker threads, and each front end hands them to its own UI thread.
    """

    def __init__(self, platform: str = sys.platform, say=None, alert=None, ask=None, base: str = API,
                 root: Path = ROOT) -> None:
        self.platform = platform
        self.say = say or (lambda key: None)
        self.alert = alert or (lambda key: None)
        self.ask = ask or (lambda key: False)
        self.base = base
        self.root = root
        # Held only to read and set the flags, never across a start or a stop:
        # a quit must not wait for a start that can take minutes.
        self._lock = threading.Lock()
        self._stopping = False
        self._opening = False
        #: Whether this app ran the start script, and the script while it runs.
        self.started = False
        self._start_proc: subprocess.Popen | None = None

    def ensure_running(self, open_ui: bool = True) -> bool:
        """Start the services unless they run, wait for them, then open the screen."""
        with self._lock:
            if self._stopping:
                return False
            if self._opening:
                return True      # a start or an open is under way, and it opens the screen
            self._opening = True
        try:
            return self._ensure_running(open_ui)
        finally:
            with self._lock:
                self._opening = False

    def _ensure_running(self, open_ui: bool) -> bool:
        state = api_state(self.base, self.root)
        if state == "elsewhere":
            if not self.ask("elsewhere_busy" if busy(self.base) else "elsewhere"):
                if open_ui:
                    webbrowser.open(UI_URL)
                return True
            if self._stopping:
                return False
            self.stop()
            state = self._wait_gone()
            if state == "elsewhere":
                self.alert("stop_failed")
                return False
        if self._stopping:
            return False
        if state == "other":
            self.alert("port_taken")
            return False
        if state == "absent":
            self.say("starting")
            if not self._start():
                if not self._stopping:
                    self.alert("start_failed")
                return False
        elif state == "starting":
            self.say("starting")
            if not self._wait_ready():
                if not self._stopping:
                    self.alert("start_failed")
                return False
        self.say("running")
        if open_ui:
            webbrowser.open(UI_URL)
        return True

    def _start(self) -> bool:
        out_dir = log_dir(self.platform)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "app_start.log", "ab") as out:
            out.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start ===\n".encode())
            out.flush()
            self._start_proc = subprocess.Popen(
                start_command(self.platform), cwd=str(ROOT), env=service_env(),
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **_hidden(self.platform))
        self.started = True
        return self._wait_ready(self._start_proc)

    def _wait_ready(self, proc: subprocess.Popen | None = None, timeout: float = START_TIMEOUT_S) -> bool:
        """Whether this folder's API answers ready in time. A start script that
        ended with an error (it says why in app_start.log) or a stop that
        began ends the wait at once."""
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self._stopping:
            if api_state(self.base, self.root) == "ready":
                return True
            if proc is not None and proc.poll() not in (None, 0):
                return False
            time.sleep(1.0)
        return False

    def _wait_gone(self, timeout: float = 20.0) -> str:
        """After a stop: the port's state once nothing answers, or as it is at the timeout."""
        end = time.monotonic() + timeout
        state = api_state(self.base, self.root)
        while state != "absent" and time.monotonic() < end:
            time.sleep(0.5)
            state = api_state(self.base, self.root)
        return state

    def owns_services(self) -> bool:
        """Whether Stop and quit should run the stop script: the services on the
        port are this folder's, or this app started them. A program the app
        found on the port, or a Seg-Studio from another folder the person chose
        to keep, is left running."""
        return self.started or api_state(self.base, self.root) in ("ready", "starting")

    def stop(self) -> bool:
        """Run the stop script and wait for it; False when it could not run.

        A start script still at work is let finish first (it ends once the API
        answers), so that what it starts is there to be stopped."""
        self.say("stopping")
        proc = self._start_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=START_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                log.warning("the start script is still running; stopping anyway")
        try:
            out_dir = log_dir(self.platform)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "app_start.log", "ab") as out:
                out.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} stop ===\n".encode())
                out.flush()
                subprocess.run(stop_command(self.platform), cwd=str(ROOT), env=service_env(),
                               stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                               timeout=180, **_hidden(self.platform))
        except (OSError, subprocess.SubprocessError):
            log.exception("the stop script did not run to its end")
            return False
        return True

    def begin_stop(self) -> bool:
        """Whether this call is the one that stops: a second Stop while one runs does nothing."""
        with self._lock:
            if self._stopping:
                return False
            self._stopping = True
            return True

    def cancel_stop(self) -> None:
        """The person said no to stopping: Stop may be asked again."""
        with self._lock:
            self._stopping = False

    def open_logs(self) -> None:
        folder = log_dir(self.platform)
        folder.mkdir(parents=True, exist_ok=True)
        if self.platform == "win32":
            os.startfile(str(folder))  # type: ignore[attr-defined]  # noqa: S606 - a folder of ours
        else:
            subprocess.Popen(["open", str(folder)])


def _setup_logging(platform: str = sys.platform) -> None:
    folder = log_dir(platform)
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(folder / "app.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    self_test = "--self-test" in args
    # Set by Seg-Studio.app's executable to run the venv; the services must not inherit it.
    os.environ.pop("__PYVENV_LAUNCHER__", None)
    _setup_logging()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        if sys.platform == "win32":
            import _tray_windows

            return _tray_windows.run(self_test=self_test)
        if sys.platform == "darwin":
            import _menubar_macos

            return _menubar_macos.run(self_test=self_test)
        # Linux and the rest: no tray here; start, open, and leave the services running.
        return 0 if (self_test or Controller().ensure_running()) else 1
    except Exception:  # noqa: BLE001 - written to app.log, which is all a windowless app has
        log.exception("the app failed")
        raise


if __name__ == "__main__":
    sys.exit(main())
