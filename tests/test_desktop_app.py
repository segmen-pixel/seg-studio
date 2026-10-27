# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The Seg-Studio app: the tray (Windows) and menu bar (macOS) entry point the
installers create, and the start-script switches it relies on."""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "desktop_app"))

import seg_studio_app as app  # noqa: E402


def _serve(routes: dict[str, tuple[int, bytes]]):
    """A local HTTP server answering each path with (status, body)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the stdlib's name
            status, body = routes.get(self.path, (404, b"not found"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


HERE_ID = app.checkout_id()


def _status(ready: bool, checkout: str | None = HERE_ID) -> bytes:
    data = {"ready": ready, "steps": []}
    if checkout is not None:
        data["checkout"] = checkout
    return json.dumps(data).encode()


@pytest.mark.parametrize(("body", "want"), [
    (_status(True), "ready"),
    (_status(False), "starting"),
    (_status(True, checkout="0123456789ab"), "elsewhere"),     # another folder's Seg-Studio
    (_status(True, checkout=None), "elsewhere"),               # one too old to say
    (b"<html>a different server</html>", "other"),
    (b'{"status": "ok"}', "other"),
])
def test_the_app_tells_seg_studio_from_anything_else_on_the_port(body, want):
    server, base = _serve({"/startup-status": (200, body)})
    try:
        assert app.api_state(base) == want
    finally:
        server.shutdown()


def test_nothing_on_the_port_is_absent():
    assert app.api_state(f"http://127.0.0.1:{_free_port()}") == "absent"


@pytest.mark.parametrize(("status", "runs", "want"), [
    ({"gpu_busy": False, "queued_count": 0}, {"items": []}, False),
    ({"gpu_busy": True, "queued_count": 0}, {"items": []}, True),
    ({"gpu_busy": False, "queued_count": 2}, {"items": []}, True),
    ({"gpu_busy": False, "queued_count": 0}, {"items": [{"run_id": "r"}]}, True),
    # A run that does not take every CUDA device: on the Mac's GPU, on one of
    # two cards, on the CPU, and a batch prediction.
    ({"gpu_busy": False, "queued_count": 0, "owner_kind": "training", "device": "mps"}, {"items": []}, True),
    ({"gpu_busy": False, "queued_count": 0, "training_runs": 1}, {"items": []}, True),
    ({"gpu_busy": False, "queued_count": 0, "inference": {"active": True}}, {"items": []}, True),
    ({"gpu_busy": False, "queued_count": 0, "training_runs": 0, "inference": None}, {"items": []}, False),
])
def test_stop_asks_first_when_a_training_or_an_ai_run_is_going(status, runs, want):
    server, base = _serve({"/api/v1/train/global-status": (200, json.dumps(status).encode()),
                           "/api/v1/agent/runs": (200, json.dumps(runs).encode())})
    try:
        assert app.busy(base) is want
    finally:
        server.shutdown()


def test_an_api_that_does_not_answer_is_not_busy():
    assert app.busy(f"http://127.0.0.1:{_free_port()}") is False


def test_a_program_that_does_not_speak_http_is_another_program():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        def answer():
            conn, _ = listener.accept()
            with conn:
                conn.recv(1024)
                conn.sendall(b"SSH-2.0-not-http\r\n")
        threading.Thread(target=answer, daemon=True).start()
        assert app.api_state(f"http://127.0.0.1:{port}") == "other"


def test_a_port_held_by_another_program_is_said_and_nothing_is_started(monkeypatch):
    server, base = _serve({"/startup-status": (200, b"<html></html>")})
    said, alerts = [], []
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: pytest.fail("started over another program"))
    try:
        ok = app.Controller(say=said.append, alert=alerts.append, base=base).ensure_running(open_ui=False)
    finally:
        server.shutdown()
    assert ok is False and alerts == ["port_taken"] and said == []


def test_the_api_names_the_same_checkout_as_the_app():
    path = ROOT / "apps/trainer_api/app/core/startup_state.py"
    spec = importlib.util.spec_from_file_location("startup_state_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.checkout_id() == app.checkout_id() == module.startup_state["checkout"]
    assert re.fullmatch(r"[0-9a-f]{12}", app.checkout_id())
    assert app.checkout_id(ROOT / "elsewhere") != app.checkout_id()


def test_the_checkout_is_named_by_the_folder_not_by_how_its_path_is_spelt(tmp_path):
    # A path typed in another case stays that way on macOS, and the API and the
    # app may come by the folder through different spellings.
    spelt = Path(str(ROOT).swapcase())
    if spelt.exists():                         # a file system that ignores case
        assert app.checkout_id(spelt) == app.checkout_id(ROOT)
    link = tmp_path / "link"
    try:
        link.symlink_to(ROOT, target_is_directory=True)
    except OSError:
        pytest.skip("no symlinks here")
    assert app.checkout_id(link) == app.checkout_id(ROOT)
    # The name carries nothing of the path: not the hash of the path itself.
    import hashlib
    path_hash = hashlib.sha256(os.path.normcase(str(ROOT)).encode("utf-8")).hexdigest()[:12]
    assert app.checkout_id(ROOT) != path_hash


def test_a_seg_studio_from_another_folder_is_stopped_only_when_the_person_says_so(monkeypatch):
    server, base = _serve({"/startup-status": (200, _status(True, checkout="0123456789ab"))})
    opened, stopped, started, asked = [], [], [], []
    monkeypatch.setattr(app.webbrowser, "open", opened.append)
    monkeypatch.setattr(app.Controller, "stop", lambda self: (stopped.append(1), server.shutdown(),
                                                              server.server_close()))
    monkeypatch.setattr(app.Controller, "_start", lambda self: started.append(1) or True)
    try:
        # No: the running one is opened as it is.
        keep = app.Controller(base=base, ask=lambda key: asked.append(key) or False)
        assert keep.ensure_running(open_ui=True) is True
        assert (asked, stopped, started, opened) == (["elsewhere"], [], [], [app.UI_URL])
        # Yes: it is stopped, and this folder's Seg-Studio is started.
        restart = app.Controller(base=base, ask=lambda key: True)
        assert restart.ensure_running(open_ui=False) is True
        assert (stopped, started) == ([1], [1])
    finally:
        server.shutdown()


def test_another_folders_run_is_named_before_it_is_stopped(monkeypatch):
    server, base = _serve({"/startup-status": (200, _status(True, checkout="0123456789ab"))})
    asked = []
    monkeypatch.setattr(app, "busy", lambda base=None: True)
    monkeypatch.setattr(app.webbrowser, "open", lambda url: None)
    try:
        app.Controller(base=base, ask=lambda key: asked.append(key) or False).ensure_running(open_ui=True)
    finally:
        server.shutdown()
    assert asked == ["elsewhere_busy"]


def test_a_running_seg_studio_is_opened_without_starting_it(monkeypatch):
    server, base = _serve({"/startup-status": (200, _status(True))})
    opened = []
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: pytest.fail("started a second time"))
    monkeypatch.setattr(app.webbrowser, "open", opened.append)
    try:
        assert app.Controller(base=base).ensure_running(open_ui=True) is True
    finally:
        server.shutdown()
    assert opened == [app.UI_URL]


def test_nothing_starts_while_a_stop_is_going(monkeypatch):
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: pytest.fail("started while stopping"))
    controller = app.Controller(base=f"http://127.0.0.1:{_free_port()}")
    assert controller.begin_stop() is True
    assert controller.begin_stop() is False          # a second Stop does nothing
    assert controller.ensure_running(open_ui=False) is False
    controller.cancel_stop()
    assert controller.begin_stop() is True


def test_a_stop_does_not_wait_for_a_start():
    # A start can take minutes. Stop and quit must not wait for it: on macOS the
    # quit is asked on the main thread, which would hang, and could deadlock
    # with a question the start puts to that thread.
    entered, release, alerts = threading.Event(), threading.Event(), []
    controller = app.Controller(alert=alerts.append, base=f"http://127.0.0.1:{_free_port()}")

    def slow_start():
        entered.set()
        release.wait(10)
        return controller._wait_ready()
    controller._start = slow_start
    worker = threading.Thread(target=controller.ensure_running, kwargs={"open_ui": False})
    worker.start()
    assert entered.wait(5)
    began = time.monotonic()
    assert controller.begin_stop() is True
    assert time.monotonic() - began < 0.5
    release.set()
    worker.join(5)
    assert not worker.is_alive() and alerts == []      # the wait ends; a stop is no failure


def test_a_start_script_that_fails_is_said_at_once(monkeypatch, tmp_path):
    class Ended:                                     # the script exited with an error
        def poll(self):
            return 1
    alerts = []
    monkeypatch.setattr(app, "log_dir", lambda platform=sys.platform: tmp_path)
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: Ended())
    began = time.monotonic()
    controller = app.Controller(alert=alerts.append, base=f"http://127.0.0.1:{_free_port()}")
    assert controller.ensure_running(open_ui=False) is False
    assert alerts == ["start_failed"] and time.monotonic() - began < 10
    assert controller.started is True


def test_stop_lets_a_start_script_finish_first(monkeypatch, tmp_path):
    order = []

    class Running:
        def poll(self):
            return None

        def wait(self, timeout=None):
            order.append("start script ended")
            return 0
    monkeypatch.setattr(app, "log_dir", lambda platform=sys.platform: tmp_path)
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: order.append("stop script"))
    controller = app.Controller()
    controller._start_proc = Running()
    assert controller.stop() is True
    assert order == ["start script ended", "stop script"]

    def too_slow(*_a, **_k):
        raise subprocess.TimeoutExpired("stop", 180)
    monkeypatch.setattr(app.subprocess, "run", too_slow)
    assert app.Controller().stop() is False          # said, not raised into the quit


def test_stop_and_quit_leaves_what_is_not_this_folders():
    for body, owned in ((b"<html></html>", False),                          # another program
                        (_status(True, checkout="0123456789ab"), False),    # another folder's, kept
                        (_status(False), True),                             # this folder's, starting
                        (_status(True), True)):                             # this folder's
        server, base = _serve({"/startup-status": (200, body)})
        try:
            assert app.Controller(base=base).owns_services() is owned, body
        finally:
            server.shutdown()
    nothing = app.Controller(base=f"http://127.0.0.1:{_free_port()}")
    assert nothing.owns_services() is False
    nothing.started = True                            # this app ran the start script
    assert nothing.owns_services() is True


def test_the_app_starts_the_scripts_without_the_dev_server_or_the_browser():
    env = app.service_env({"PATH": "x", "SEG_PROJECTS_DIR": "D:/p"})
    assert env["SEG_NO_DEV_UI"] == env["SEG_NO_BROWSER"] == env["SEG_NO_PAUSE"] == "1"
    assert env["SEG_HIDE_TOKEN"] == "1"               # its log is a file: no LAN token in it
    assert env["SEG_PROJECTS_DIR"] == "D:/p" and env["PATH"] == "x"
    for command in (app.start_command("darwin"), app.stop_command("darwin")):
        assert Path(command[-1]).is_file(), command
    for command in (app.start_command("win32"), app.stop_command("win32")):
        # cmd.exe by its full path; the script relative to the checkout, so that
        # no quoting of a path with a space and ( ) or & reaches cmd.exe.
        assert PureWindowsPath(command[0]).is_absolute() and PureWindowsPath(command[0]).name == "cmd.exe"
        assert not PureWindowsPath(command[-1]).is_absolute(), command
        assert (ROOT / PureWindowsPath(command[-1]).as_posix()).is_file(), command
    assert app._hidden("win32") == {"creationflags": 0x08000000}


def test_the_menus_say_the_same_things_in_both_languages():
    assert set(app.LABELS["ja"]) == set(app.LABELS["en"])
    assert app.lang_from("ja_JP") == app.lang_from("ja-JP") == "ja"
    assert app.lang_from("en-GB") == app.lang_from("") == app.lang_from(None) == "en"


@pytest.mark.parametrize("rel", ["scripts/windows/start_local_windows.bat", "scripts/macos/start_local_macos.sh"])
def test_the_start_scripts_honour_the_apps_switches(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert "SEG_NO_DEV_UI" in text and "SEG_NO_BROWSER" in text
    if rel.endswith(".bat"):
        assert re.search(r'^if not "%SEG_NO_BROWSER%"=="1" start "" "http', text, re.M)
        assert re.search(r'^if "%SEG_NO_DEV_UI%"=="1" \(', text, re.M)
        assert 'if "%SEG_HIDE_TOKEN%"=="1" (\n    echo   LAN access token: the "api_token" entry' in text
    else:
        assert 'if [ "${SEG_NO_BROWSER:-0}" != "1" ]; then\n    open "http' in text
        assert 'if [ "${SEG_NO_DEV_UI:-0}" = "1" ]; then' in text
        assert 'if [ "${SEG_HIDE_TOKEN:-0}" = "1" ]; then\n    echo "  LAN access token: the' in text
    # The token itself is printed only where SEG_HIDE_TOKEN is not set.
    assert len(re.findall(r"LAN access token: (?:!SEG_API_TOKEN!|\$SEG_API_TOKEN)", text)) == 1


def test_the_windows_installer_creates_the_shortcuts_unless_told_not_to():
    text = (ROOT / "scripts/windows/install_windows.bat").read_text(encoding="utf-8")
    assert re.search(r'^if "%NO_SHORTCUTS%"=="1" goto :after_shortcuts', text, re.M)
    # The app serves the built screen: with none, no shortcut to an empty one.
    assert 'if not exist "%REPO_ROOT%\\apps\\trainer_ui\\dist\\index.html" (' in text
    assert re.search(r"_create_shortcuts\.ps1\" -RepoRoot", text)
    assert '"--no-shortcuts"' in text and "--no-shortcuts        " in text
    ps1 = (ROOT / "scripts/windows/_create_shortcuts.ps1").read_text(encoding="utf-8")
    assert r".venv-windows\Scripts\pythonw.exe" in ps1 and r"scripts\desktop_app\seg_studio_app.py" in ps1
    # The shortcuts and the running app share one taskbar ID, so the button pins the shortcut.
    assert f"$appId = '{app.APP_USER_MODEL_ID}'" in ps1 and "[SegStudioAppId]::Set($path, $appId)" in ps1
    # The native PROPVARIANT is 24 bytes on 64-bit Windows, and the store's
    # methods return their HRESULT as it is.
    assert "[StructLayout(LayoutKind.Explicit, Size = 24)]" in ps1 and ps1.count("[PreserveSig] int ") == 5
    tray = (ROOT / "scripts/desktop_app/_tray_windows.py").read_text(encoding="utf-8")
    assert "SetCurrentProcessExplicitAppUserModelID(app.APP_USER_MODEL_ID)" in tray


def test_the_macos_installer_creates_the_app_unless_told_not_to():
    text = (ROOT / "scripts/macos/install_macos.sh").read_text(encoding="utf-8")
    assert "--no-app)             NO_APP=1 ;;" in text
    assert 'bash "$REPO_ROOT/scripts/macos/create_app_macos.sh"' in text
    assert '[ ! -f "$REPO_ROOT/apps/trainer_ui/dist/index.html" ]' in text
    assert re.search(r'^\s*\$PIP install --no-deps -r "\$REPO_ROOT/apps/trainer_api/requirements-macos-app\.txt"',
                     text, re.M)


def test_the_menu_bar_library_is_pinned():
    pins = [ln for ln in (ROOT / "apps/trainer_api/requirements-macos-app.txt").read_text().splitlines()
            if ln and not ln.startswith("#")]
    assert pins and all(re.fullmatch(r"[a-z0-9-]+==\d+(\.\d+)+", ln) for ln in pins), pins
    assert {ln.split("==")[0] for ln in pins} == {"pyobjc-core", "pyobjc-framework-cocoa"}
    workflow = (ROOT / ".github/workflows/license-check.yml").read_text(encoding="utf-8")
    assert "apps/trainer_api/requirements-macos-app.txt" in workflow


@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="needs bash")
def test_create_app_writes_a_menu_bar_app_for_this_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    for rel in ("apps/trainer_api/app/main.py", "pyproject.toml", "scripts/macos/create_app_macos.sh",
                "scripts/installer_assets/make_icon.py", "scripts/installer_assets/seg-studio.ico"):
        dest = checkout / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    venv_python = checkout / ".venv-macos/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    venv_python.chmod(0o755)
    apps = tmp_path / "Applications"
    env = dict(os.environ, SEG_APP_DIR=str(apps))
    out = subprocess.run(["bash", str(checkout / "scripts/macos/create_app_macos.sh")], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    bundle = apps / "Seg-Studio.app" / "Contents"
    plist = (bundle / "Info.plist").read_text()
    assert "LSUIElement" not in plist                     # a Dock icon while it runs
    assert f"<string>{app.MACOS_BUNDLE_ID}</string>" in plist
    launcher = bundle / "MacOS" / "seg-studio"
    assert os.access(launcher, os.X_OK)
    script = launcher.read_text()
    assert 'APP_PY="$REPO/scripts/desktop_app/seg_studio_app.py"' in script
    assert 'export __PYVENV_LAUNCHER__="$REPO/.venv-macos/bin/python"' in script
    assert 'exec "$HERE/python" "$APP_PY" "$@"' in script
    # A new copy of Python's executable comes in by a rename: macOS kills a
    # signed executable whose file is written over.
    assert 'cp -f "$GUI" "$HERE/python.new" && mv -f "$HERE/python.new" "$HERE/python"' in script
    assert str(checkout) not in script.splitlines()[1]      # the path only in the quoted REPO=
    assert script.rstrip().endswith('exec "$REPO/.venv-macos/bin/python" "$APP_PY" "$@"')
    if shutil.which("iconutil"):
        assert (bundle / "Resources" / "AppIcon.icns").stat().st_size > 0
    # A second run replaces its own app; an app it did not make is left alone.
    assert subprocess.run(["bash", str(checkout / "scripts/macos/create_app_macos.sh")], env=env).returncode == 0
    (bundle / "Info.plist").write_text(plist.replace(app.MACOS_BUNDLE_ID, "org.example.other"))
    again = subprocess.run(["bash", str(checkout / "scripts/macos/create_app_macos.sh")], env=env,
                           capture_output=True, text=True)
    assert again.returncode == 1 and "org.example.other" in (bundle / "Info.plist").read_text()


def _pixels(image):
    import numpy as np

    return np.asarray(image.convert("RGBA"), dtype=np.int16)


def test_the_committed_icons_are_what_make_icon_draws(tmp_path):
    from PIL import Image

    sys.path.insert(0, str(ROOT / "scripts" / "installer_assets"))
    import make_icon

    with Image.open(app.ICON) as ico:
        assert {(n, n) for n in make_icon.ICO_SIZES} <= set(ico.info["sizes"])
        for n in (16, 32, 256):
            assert abs(_pixels(ico.ico.getimage((n, n))) - _pixels(make_icon.render(n))).max() <= 2, n
    with Image.open(ROOT / "apps/trainer_ui/public/favicon.ico") as fav:
        assert set(fav.info["sizes"]) == {(n, n) for n in make_icon.FAVICON_SIZES}
    with Image.open(app.MENUBAR_ICON) as bar:
        drawn = make_icon.render(make_icon.MENUBAR_PX, template=True, inset=make_icon.MENUBAR_INSET)
        assert bar.size == drawn.size and abs(_pixels(bar) - _pixels(drawn)).max() <= 2
        px = _pixels(bar)
        assert px[..., :3][px[..., 3] > 0].max() == 0     # a template: black, the shape is in the alpha
    make_icon.write_iconset(tmp_path / "AppIcon.iconset")
    names = sorted(p.name for p in (tmp_path / "AppIcon.iconset").iterdir())
    assert len(names) == 2 * len(make_icon.ICONSET)
    with Image.open(tmp_path / "AppIcon.iconset" / "icon_512x512@2x.png") as big:
        assert big.size == (1024, 1024)
    menubar = (ROOT / "scripts/desktop_app/_menubar_macos.py").read_text(encoding="utf-8")
    assert "app.MENUBAR_ICON" in menubar and "setTemplate_(True)" in menubar


def test_the_macos_app_is_in_the_dock_and_every_quit_stops_the_services():
    menubar = (ROOT / "scripts/desktop_app/_menubar_macos.py").read_text(encoding="utf-8")
    assert "setActivationPolicy_(NSApplicationActivationPolicyRegular)" in menubar
    assert "Accessory" not in menubar
    for hook in ("def applicationShouldTerminate_", "def applicationDockMenu_",
                 "def applicationShouldHandleReopen_hasVisibleWindows_", "setMainMenu_"):
        assert hook in menubar, hook
    # The menus' Stop and quit goes through applicationShouldTerminate_, like Cmd-Q and the Dock's Quit.
    assert re.search(r"def quit_\(self, _sender\):\n\s+NSApplication\.sharedApplication\(\)\.terminate_\(None\)",
                     menubar)


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS front end")
@pytest.mark.skipif(sys.platform == "win32", reason="the macOS front end")
def test_a_dialog_without_pyobjc_quotes_what_it_shows():
    import _menubar_macos

    assert _menubar_macos._as_text('a "b" \\ c') == '"a \\"b\\" \\\\ c"'


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS front end")
def test_the_macos_delegate_class_builds_with_its_selectors():
    pytest.importorskip("AppKit")
    import _menubar_macos

    cls = _menubar_macos.delegate_class()     # pyobjc checks every method's selector here
    for selector in (b"applicationShouldTerminate:", b"applicationDockMenu:", b"open:", b"logs:",
                     b"quit:", b"ask:", b"stopped:", b"applicationShouldHandleReopen:hasVisibleWindows:"):
        assert cls.instancesRespondToSelector_(selector), selector


@pytest.mark.skipif(sys.platform != "win32", reason="the Win32 tray")
def test_the_tray_declares_the_shell_structure_at_its_documented_size():
    import ctypes

    import _tray_windows

    # NOTIFYICONDATAW with guidItem and hBalloonIcon: 976 bytes on 64-bit Windows.
    assert ctypes.sizeof(_tray_windows.NOTIFYICONDATAW) == (976 if ctypes.sizeof(ctypes.c_void_p) == 8 else 956)
    _tray_windows._declare()   # the signature table itself must load
    assert ctypes.sizeof(_tray_windows.PROPVARIANT) == (24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16)
    command = _tray_windows.relaunch_command()
    assert command.endswith('seg_studio_app.py"') and "python" in command.lower()


class _FakeController:
    def __init__(self):
        self.calls = []

    def ensure_running(self, open_ui=True):
        self.calls.append("open")
        return True

    def begin_stop(self):
        self.calls.append("begin_stop")
        return True

    def owns_services(self):
        self.calls.append("owns_services")
        return True

    def cancel_stop(self):
        self.calls.append("cancel_stop")

    def stop(self):
        self.calls.append("stop")


@pytest.mark.skipif(sys.platform != "win32", reason="the Win32 taskbar button")
def test_the_taskbar_button_opens_the_screen_and_closing_it_stops(monkeypatch):
    import time

    import _tray_windows as tw

    tw._declare()
    monkeypatch.setattr(tw.app, "busy", lambda base=None: False)
    ran = []
    monkeypatch.setattr(tw.Tray, "_in_background", lambda self, fn: (ran.append(fn), fn()))
    tray = tw.Tray("en")
    tray.controller = _FakeController()
    tray.create()
    try:
        assert tray.show_in_taskbar() is True                  # visible, and minimized
        # The window names itself for the taskbar (and clears it again).
        assert tw.window_identity(tray.hwnd, [(tw.PID_RELAUNCH_NAME, "Seg-Studio")]) is True
        # A click on the button asks to restore the window: the screen opens instead.
        tw.user32.SendMessageW(tray.hwnd, tw.WM_SYSCOMMAND, tw.SC_RESTORE, 0)
        assert tray.controller.calls == ["open"] and tw.user32.IsIconic(tray.hwnd)
        # A double-click on the tray icon (up, double-click, up) opens one screen.
        tray._last_open = float("-inf")
        for event in (0x0202, 0x0203, 0x0202):
            tw.user32.SendMessageW(tray.hwnd, tw.WM_TRAY, 0, event)
        assert tray.controller.calls == ["open", "open"]
        # Close window on the button is Stop and quit.
        tw.user32.SendMessageW(tray.hwnd, tw.WM_SYSCOMMAND, tw.SC_CLOSE, 0)
        deadline = time.monotonic() + 5
        while "stop" not in tray.controller.calls and time.monotonic() < deadline:
            time.sleep(0.05)
        assert tray.controller.calls == ["open", "open", "begin_stop", "owns_services", "stop"]
        assert tw.user32.IsWindow(tray.hwnd)                   # the window ends with WM_END, from its loop
    finally:
        tw.user32.DestroyWindow(tray.hwnd)
