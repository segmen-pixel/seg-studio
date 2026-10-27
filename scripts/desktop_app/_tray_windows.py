# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The Windows taskbar button and tray icon of the Seg-Studio app, on the Win32
API through ctypes.

No third-party tray library: the common one is LGPL, and the API it wraps is a
window, one Shell_NotifyIconW call and a popup menu. Left click on the tray
icon opens the screen; right click shows Open / Log folder / Stop and quit.

Windows 11 puts a new tray icon out of sight, behind the ^ arrow, so the same
window also stands in the taskbar for as long as Seg-Studio runs, the way the
Dock shows it on macOS: a minimized window, 1 x 1 pixel and off the screen.
Clicking the button opens the screen instead of restoring the window, and
closing it (Close window in the button's menu) is Stop and quit. The process
has the AppUserModelID the shortcuts have, so the button carries the
Start menu's Seg-Studio and pinning it pins that shortcut; the window also
names itself, its icon and how to start it again (the shell's relaunch
properties), for when there is no such shortcut.

Work that waits (starting, stopping, asking the API) runs on worker threads,
which hand UI changes to the window's thread with PostMessageW: the tooltip,
a message box, the end.
"""
from __future__ import annotations

import ctypes
import hashlib
import sys
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path

import seg_studio_app as app

user32 = ctypes.WinDLL("user32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WM_DESTROY, WM_CLOSE, WM_NULL, WM_SIZE, WM_SETICON = 0x0002, 0x0010, 0x0000, 0x0005, 0x0080
WM_SYSCOMMAND, SC_RESTORE, SC_MAXIMIZE, SC_CLOSE = 0x0112, 0xF120, 0xF030, 0xF060
SIZE_RESTORED, SIZE_MAXIMIZED = 0, 2
WS_CAPTION, WS_SYSMENU, WS_MINIMIZEBOX, WS_EX_APPWINDOW = 0x00C00000, 0x00080000, 0x00020000, 0x00040000
SW_SHOWMINNOACTIVE, ICON_SMALL, ICON_BIG = 7, 0, 1
GWL_STYLE, WS_VISIBLE = -16, 0x10000000
WM_LBUTTONUP, WM_RBUTTONUP = 0x0202, 0x0205
WM_APP = 0x8000
WM_TRAY = WM_APP + 1      # the icon's mouse messages
WM_SAY = WM_APP + 2       # a new tooltip
WM_ALERT = WM_APP + 3     # a message box
WM_END = WM_APP + 4       # remove the icon and leave the loop
#: Stop and quit, as the menu item does, for a script that cannot click the
#: menu (a test, an uninstaller): PostMessageW to the window of class
#: SegStudioTray in the person's session.
WM_STOP_REQUEST = WM_APP + 5
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP = 0x1, 0x2, 0x4
MF_STRING, MF_SEPARATOR = 0x0, 0x800
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x2, 0x100, 0x80
IMAGE_ICON, LR_LOADFROMFILE = 1, 0x10
SM_CXICON, SM_CYICON, SM_CXSMICON, SM_CYSMICON = 11, 12, 49, 50
MB_OK, MB_YESNO, MB_ICONERROR, MB_ICONQUESTION, MB_ICONWARNING = 0x0, 0x4, 0x10, 0x20, 0x30
MB_SETFOREGROUND, MB_TOPMOST, IDYES = 0x10000, 0x40000, 6
ERROR_ALREADY_EXISTS = 183
IDI_APPLICATION = 32512
CMD_OPEN, CMD_LOGS, CMD_QUIT = 1, 2, 3


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]


class PROPVARIANT(ctypes.Structure):
    """VT_EMPTY or VT_LPWSTR only: the type, three reserved words, the pointer, padding to 24 bytes."""
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3), ("pointer", ctypes.c_void_p),
                ("padding", ctypes.c_void_p)]


def _guid(text: str) -> GUID:
    return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


IID_IPROPERTYSTORE = "886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"
FMTID_APPUSERMODEL = "9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"
PID_RELAUNCH_COMMAND, PID_RELAUNCH_ICON, PID_RELAUNCH_NAME, PID_APP_ID = 2, 3, 4, 5
VT_EMPTY, VT_LPWSTR = 0, 31


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT),
                ("szInfoTitle", wintypes.WCHAR * 64), ("dwInfoFlags", wintypes.DWORD), ("guidItem", GUID),
                ("hBalloonIcon", wintypes.HICON)]


def _declare() -> None:
    """Argument and result types, so that 64-bit handles are not cut to 32 bits.

    A list of pairs, not a dict: a ctypes function pointer cannot be a dict key.
    """
    sig = [
        (user32.RegisterClassW, (wintypes.ATOM, [ctypes.POINTER(WNDCLASSW)])),
        (user32.CreateWindowExW, (wintypes.HWND, [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                 wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID])),
        (user32.DefWindowProcW, (LRESULT, [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM])),
        (user32.GetMessageW, (wintypes.BOOL, [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT])),
        (user32.TranslateMessage, (wintypes.BOOL, [ctypes.POINTER(wintypes.MSG)])),
        (user32.DispatchMessageW, (LRESULT, [ctypes.POINTER(wintypes.MSG)])),
        (user32.PostMessageW, (wintypes.BOOL, [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM])),
        (user32.PostQuitMessage, (None, [ctypes.c_int])),
        (user32.DestroyWindow, (wintypes.BOOL, [wintypes.HWND])),
        (user32.RegisterWindowMessageW, (wintypes.UINT, [wintypes.LPCWSTR])),
        (user32.LoadImageW, (wintypes.HANDLE, [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                              ctypes.c_int, ctypes.c_int, wintypes.UINT])),
        (user32.LoadIconW, (wintypes.HICON, [wintypes.HINSTANCE, wintypes.LPVOID])),
        (user32.GetSystemMetrics, (ctypes.c_int, [ctypes.c_int])),
        (user32.CreatePopupMenu, (wintypes.HMENU, [])),
        (user32.AppendMenuW, (wintypes.BOOL, [wintypes.HMENU, wintypes.UINT, ctypes.c_size_t, wintypes.LPCWSTR])),
        (user32.TrackPopupMenu, (ctypes.c_int, [wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, wintypes.HWND, wintypes.LPVOID])),
        (user32.DestroyMenu, (wintypes.BOOL, [wintypes.HMENU])),
        (user32.GetCursorPos, (wintypes.BOOL, [ctypes.POINTER(wintypes.POINT)])),
        (user32.SetForegroundWindow, (wintypes.BOOL, [wintypes.HWND])),
        (user32.MessageBoxW, (ctypes.c_int, [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT])),
        (user32.ShowWindow, (wintypes.BOOL, [wintypes.HWND, ctypes.c_int])),
        (user32.SendMessageW, (LRESULT, [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM])),
        (user32.SetWindowTextW, (wintypes.BOOL, [wintypes.HWND, wintypes.LPCWSTR])),
        (user32.IsIconic, (wintypes.BOOL, [wintypes.HWND])),
        (user32.GetWindowLongW, (ctypes.c_long, [wintypes.HWND, ctypes.c_int])),
        (user32.IsWindow, (wintypes.BOOL, [wintypes.HWND])),
        (shell32.Shell_NotifyIconW, (wintypes.BOOL, [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)])),
        (shell32.SetCurrentProcessExplicitAppUserModelID, (ctypes.c_long, [wintypes.LPCWSTR])),
        (shell32.SHGetPropertyStoreForWindow, (ctypes.c_long, [wintypes.HWND, ctypes.POINTER(GUID),
                                                               ctypes.POINTER(ctypes.c_void_p)])),
        (kernel32.GetModuleHandleW, (wintypes.HMODULE, [wintypes.LPCWSTR])),
        (kernel32.CreateMutexW, (wintypes.HANDLE, [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR])),
        (kernel32.GetUserDefaultUILanguage, (wintypes.WORD, [])),
    ]
    for fn, (res, args) in sig:
        fn.restype, fn.argtypes = res, args


def window_identity(hwnd, values: list[tuple[int, str | None]]) -> bool:
    """Write AppUserModel properties on a window through its shell property
    store (IPropertyStore, by its vtable); None clears one."""
    store = ctypes.c_void_p()
    if shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(_guid(IID_IPROPERTYSTORE)), ctypes.byref(store)) < 0:
        return False
    vtable = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    # IUnknown (QueryInterface, AddRef, Release), then GetCount, GetAt, GetValue, SetValue, Commit.
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])
    set_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY),
                                   ctypes.POINTER(PROPVARIANT))(vtable[6])
    commit = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vtable[7])
    ok = True
    try:
        for pid, text in values:
            key = PROPERTYKEY(_guid(FMTID_APPUSERMODEL), pid)
            value = PROPVARIANT()
            buffer = None
            if text is not None:
                buffer = ctypes.create_unicode_buffer(text)   # SetValue copies it
                value.vt, value.pointer = VT_LPWSTR, ctypes.cast(buffer, ctypes.c_void_p)
            ok = set_value(store, ctypes.byref(key), ctypes.byref(value)) >= 0 and ok
        ok = commit(store) >= 0 and ok
    finally:
        release(store)
    return ok


def relaunch_command() -> str:
    """How the taskbar starts the app again from a pin: the venv's pythonw, without a console."""
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    return f'"{pythonw if pythonw.exists() else python}" "{Path(app.__file__).resolve()}"'


def ui_lang() -> str:
    """Japanese when Windows shows its own menus in Japanese (LANG_JAPANESE is 0x11)."""
    return "ja" if (kernel32.GetUserDefaultUILanguage() & 0x3FF) == 0x11 else "en"


class Tray:
    def __init__(self, lang: str) -> None:
        self.text = app.LABELS[lang]
        self.tip = "Seg-Studio"
        self.alert_key = ""
        self.hwnd = None
        self.in_taskbar = False
        self._last_open = float("-inf")
        self._proc = WNDPROC(self._wndproc)   # kept: the window calls it for as long as it lives
        self._taskbar_created = user32.RegisterWindowMessageW("TaskbarCreated")
        self.controller = app.Controller(say=self._say, alert=self._alert, ask=self._ask)

    # --- from worker threads -------------------------------------------------
    def _say(self, key: str) -> None:
        self.tip = self.text.get(key, "Seg-Studio")
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_SAY, 0, 0)

    def _alert(self, key: str) -> None:
        self.alert_key = key
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_ALERT, 0, 0)

    def _ask(self, key: str) -> bool:
        text = self.text[key] + ("\n\n" + self.text[key + "_no"] if key + "_no" in self.text else "")
        flags = MB_YESNO | MB_ICONQUESTION | MB_SETFOREGROUND | MB_TOPMOST
        return user32.MessageBoxW(None, text, "Seg-Studio", flags) == IDYES

    def _in_background(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def _open(self) -> None:
        now = time.monotonic()
        if now - self._last_open < 1.0:     # the second click of a double-click: one screen
            return
        self._last_open = now
        self._in_background(lambda: self.controller.ensure_running(open_ui=True))

    def _quit(self) -> None:
        def stop():
            if not self.controller.begin_stop():
                return
            try:
                # Only this folder's services, or the ones this app started:
                # a program found on the port, or a Seg-Studio from another
                # folder the person chose to keep, is left running.
                if self.controller.owns_services():
                    if app.busy():
                        flags = MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST
                        if user32.MessageBoxW(None, self.text["busy"], "Seg-Studio", flags) != IDYES:
                            self.controller.cancel_stop()
                            return
                    self.controller.stop()
            except Exception:  # noqa: BLE001 - the app still quits; app.log says why
                app.log.exception("stopping the services failed")
            user32.PostMessageW(self.hwnd, WM_END, 0, 0)
        self._in_background(stop)

    # --- the window's thread -------------------------------------------------
    def _icon_data(self, flags: int) -> NOTIFYICONDATAW:
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = self.hwnd
        nid.uID = 1
        nid.uFlags = flags
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = self.hicon
        nid.szTip = self.tip[:127]
        return nid

    def add_icon(self) -> bool:
        return bool(shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(self._icon_data(NIF_MESSAGE | NIF_ICON | NIF_TIP))))

    def remove_icon(self) -> None:
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(self._icon_data(0)))

    def _menu(self) -> None:
        menu = user32.CreatePopupMenu()
        user32.AppendMenuW(menu, MF_STRING, CMD_OPEN, self.text["open"])
        user32.AppendMenuW(menu, MF_STRING, CMD_LOGS, self.text["logs"])
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, CMD_QUIT, self.text["quit"])
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        # Without the foreground, the menu does not close when the person clicks elsewhere.
        user32.SetForegroundWindow(self.hwnd)
        cmd = user32.TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY,
                                    pt.x, pt.y, 0, self.hwnd, None)
        user32.PostMessageW(self.hwnd, WM_NULL, 0, 0)
        user32.DestroyMenu(menu)
        if cmd == CMD_OPEN:
            self._open()
        elif cmd == CMD_LOGS:
            self.controller.open_logs()
        elif cmd == CMD_QUIT:
            self._quit()

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_TRAY:
            if lparam == WM_LBUTTONUP:              # a double-click is two of these
                self._open()
            elif lparam == WM_RBUTTONUP:
                self._menu()
            return 0
        if msg == WM_SAY:
            shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(self._icon_data(NIF_TIP)))
            user32.SetWindowTextW(hwnd, self.tip)     # the taskbar button's name
            return 0
        if msg == WM_SYSCOMMAND and (wparam & 0xFFF0) in (SC_RESTORE, SC_MAXIMIZE):
            self._open()                              # a click on the taskbar button
            return 0
        if msg == WM_SIZE and wparam in (SIZE_RESTORED, SIZE_MAXIMIZED) and self.in_taskbar:
            user32.ShowWindow(hwnd, SW_SHOWMINNOACTIVE)   # restored some other way: back to the taskbar
            self._open()
            return 0
        if msg == WM_CLOSE or (msg == WM_SYSCOMMAND and (wparam & 0xFFF0) == SC_CLOSE):
            self._quit()                              # Close window on the taskbar button, Alt-F4
            return 0
        if msg == WM_ALERT:
            user32.MessageBoxW(None, self.text.get(self.alert_key, self.alert_key), "Seg-Studio",
                               MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST)
            return 0
        if msg == WM_STOP_REQUEST:
            self._quit()
            return 0
        if msg == WM_END:
            self.remove_icon()
            # The shell asks for these to be cleared before the window goes.
            window_identity(hwnd, [(pid, None) for pid in (PID_APP_ID, PID_RELAUNCH_COMMAND,
                                                           PID_RELAUNCH_ICON, PID_RELAUNCH_NAME)])
            user32.DestroyWindow(hwnd)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        if msg == self._taskbar_created:   # Explorer restarted: the icon is gone, put it back
            self.add_icon()
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _load_icon(self, cx: int, cy: int):
        return user32.LoadImageW(None, str(app.ICON), IMAGE_ICON, user32.GetSystemMetrics(cx),
                                 user32.GetSystemMetrics(cy), LR_LOADFROMFILE) \
            or user32.LoadIconW(None, ctypes.c_void_p(IDI_APPLICATION))

    def create(self) -> None:
        # Before the first window: the taskbar groups windows by this ID.
        shell32.SetCurrentProcessExplicitAppUserModelID(app.APP_USER_MODEL_ID)
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW()
        wc.lpfnWndProc = self._proc
        wc.hInstance = hinst
        wc.lpszClassName = "SegStudioTray"
        user32.RegisterClassW(ctypes.byref(wc))
        # A top-level window rather than a message-only one: only a top-level
        # window hears "TaskbarCreated" after Explorer restarts, and only one
        # can stand in the taskbar. It is never shown other than minimized.
        self.hwnd = user32.CreateWindowExW(WS_EX_APPWINDOW, "SegStudioTray", "Seg-Studio",
                                           WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                           -10000, -10000, 1, 1, None, None, hinst, None)
        self.hicon = self._load_icon(SM_CXSMICON, SM_CYSMICON)
        self.hicon_big = self._load_icon(SM_CXICON, SM_CYICON)
        user32.SendMessageW(self.hwnd, WM_SETICON, ICON_SMALL, self.hicon)
        user32.SendMessageW(self.hwnd, WM_SETICON, ICON_BIG, self.hicon_big)
        # The relaunch properties first and the ID last, as the shell reads them.
        window_identity(self.hwnd, [(PID_RELAUNCH_COMMAND, relaunch_command()),
                                    (PID_RELAUNCH_ICON, f"{app.ICON},0"),
                                    (PID_RELAUNCH_NAME, "Seg-Studio"),
                                    (PID_APP_ID, app.APP_USER_MODEL_ID)])

    def show_in_taskbar(self) -> bool:
        """The taskbar button: the window, minimized, without taking the focus."""
        user32.ShowWindow(self.hwnd, SW_SHOWMINNOACTIVE)
        if not user32.IsIconic(self.hwnd):
            # A process's first ShowWindow may follow the shortcut's "Run" setting instead.
            user32.ShowWindow(self.hwnd, SW_SHOWMINNOACTIVE)
        self.in_taskbar = True
        # The window's own flag: IsWindowVisible also asks the desktop, which a
        # session without a screen (SSH, a service) never shows.
        shown = user32.GetWindowLongW(self.hwnd, GWL_STYLE) & WS_VISIBLE
        return bool(shown and user32.IsIconic(self.hwnd))

    def loop(self) -> None:
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


def _single_instance() -> bool:
    """Whether this is the only app for this checkout (a mutex named after its folder)."""
    name = "Local\\SegStudioApp-" + hashlib.sha1(str(app.ROOT).lower().encode()).hexdigest()[:12]
    kernel32.CreateMutexW(None, False, name)
    return ctypes.get_last_error() != ERROR_ALREADY_EXISTS


def run(self_test: bool = False) -> int:
    _declare()
    if not self_test and not _single_instance():
        # The app is already in the tray: a second click on the shortcut opens the screen.
        return 0 if app.Controller().ensure_running(open_ui=True) else 1
    tray = Tray(ui_lang())
    tray.create()
    shown = tray.add_icon()
    in_taskbar = tray.show_in_taskbar()
    app.log.info("tray icon %s, taskbar button %s", "shown" if shown else "could not be shown",
                 "shown" if in_taskbar else "could not be shown")
    if self_test:
        tray.remove_icon()
        user32.DestroyWindow(tray.hwnd)
        print("tray icon shown" if shown else "tray icon could not be shown")
        print("taskbar button shown" if in_taskbar else "taskbar button could not be shown")
        return 0 if shown and in_taskbar else 1
    tray._open()
    tray.loop()
    return 0


if __name__ == "__main__":
    sys.exit(run(self_test="--self-test" in sys.argv))
