# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The macOS front end of the Seg-Studio app, on AppKit through pyobjc (MIT).

A Dock icon for as long as Seg-Studio runs, so that it is plain that it does,
and a menu bar item for when the menu bar has room for it (macOS hides items
that do not fit). Clicking the Dock icon, or opening Seg-Studio.app again,
opens the screen; the Dock icon's menu, the app menu and the menu bar item have
Open / Log folder / Stop and quit, and quitting the app in any way (Cmd-Q, Quit
in the Dock, logging out) stops the services, after asking when a training or
an AI support run is going. No window.

install_macos.sh installs pyobjc from apps/trainer_api/requirements-macos-app.txt;
without it the app still starts the services and opens the screen, and asks
with a dialog (osascript) whether to open or stop when it is opened again
while they run.
"""
from __future__ import annotations

import fcntl
import subprocess
import sys
import threading

import seg_studio_app as app


def ui_lang() -> str:
    try:
        from Foundation import NSLocale

        langs = NSLocale.preferredLanguages()
        return app.lang_from(str(langs[0]) if langs else "")
    except Exception:  # noqa: BLE001 - no pyobjc: ask the system
        try:
            out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True,
                                 timeout=5).stdout
            first = next((ln.strip(' ",()') for ln in out.splitlines() if ln.strip(' ",()')), "")
            return app.lang_from(first)
        except (OSError, subprocess.SubprocessError):
            return "en"


def _lock():
    """Hold a lock for as long as the app runs; None when another app holds it."""
    folder = app.log_dir("darwin")
    folder.mkdir(parents=True, exist_ok=True)
    handle = open(folder / "app.lock", "w")  # noqa: SIM115 - held until the process ends
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _as_text(value: str) -> str:
    """value as an AppleScript string literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dialog(message: str, buttons: list[str], default: str) -> str:
    """A choice through osascript, for the app without pyobjc; '' when dismissed."""
    quoted = ", ".join(_as_text(b) for b in buttons)
    script = (f'display dialog {_as_text(message)} with title "Seg-Studio" buttons {{{quoted}}} '
              f'default button {_as_text(default)}')
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True).stdout
    return out.split("button returned:", 1)[-1].strip() if "button returned:" in out else ""


def _run_without_menu(text: dict) -> int:
    controller = app.Controller(platform="darwin",
                                alert=lambda key: _dialog(text.get(key, key), ["OK"], "OK"),
                                ask=lambda key: _dialog(text[key], [text["keep"], text["restart"]],
                                                        text["restart"]) == text["restart"])
    if app.api_state() == "ready":
        choice = _dialog(text["running_ask"], [text["quit"], text["open"]], text["open"])
        if choice == text["quit"]:
            # api_state() said ready: the services are this folder's.
            if not app.busy() or _dialog(text["busy"], [text["no"], text["yes"]], text["no"]) == text["yes"]:
                controller.stop()
            return 0
        if choice != text["open"]:
            return 0
    return 0 if controller.ensure_running(open_ui=True) else 1


_DELEGATE = None


def delegate_class():
    """The app delegate, defined once per process (an Objective-C class name can be registered once)."""
    global _DELEGATE
    if _DELEGATE is not None:
        return _DELEGATE
    import objc
    from AppKit import (
        NSAlert,
        NSAlertFirstButtonReturn,
        NSApplication,
        NSImage,
        NSMenu,
        NSMenuItem,
        NSStatusBar,
        NSTerminateCancel,
        NSTerminateLater,
        NSTerminateNow,
        NSVariableStatusItemLength,
    )
    from Foundation import NSObject
    from PyObjCTools import AppHelper

    def choose(message: str, yes: str, no: str) -> bool:
        box = NSAlert.alloc().init()
        box.setMessageText_("Seg-Studio")
        box.setInformativeText_(message)
        box.addButtonWithTitle_(yes)
        box.addButtonWithTitle_(no)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        return box.runModal() == NSAlertFirstButtonReturn

    class SegStudioMenuBarDelegate(NSObject):
        def initWithController_text_selfTest_(self, controller, text, self_test):
            self = objc.super(SegStudioMenuBarDelegate, self).init()
            if self is None:
                return None
            self.controller, self.text, self.self_test = controller, text, bool(self_test)
            self.item = None
            self.dock_menu = None
            self.stopped = False
            return self

        @objc.python_method
        def entries(self, keys):
            """A menu of Open / Log folder / Stop and quit entries, in the order given (None: a separator)."""
            menu = NSMenu.alloc().init()
            actions = {"open": "open:", "logs": "logs:", "quit": "quit:"}
            for key in keys:
                if key is None:
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                entry = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    self.text[key], actions[key], "q" if key == "quit" else "")
                entry.setTarget_(self)
                menu.addItem_(entry)
            return menu

        def build(self):
            """The menu bar item, the app menu and the Dock icon's menu."""
            # The app menu (the bold one beside the Apple menu), with Cmd-Q.
            main, app_item = NSMenu.alloc().init(), NSMenuItem.alloc().init()
            app_item.setSubmenu_(self.entries(("open", "logs", None, "quit")))
            main.addItem_(app_item)
            NSApplication.sharedApplication().setMainMenu_(main)
            self.dock_menu = self.entries(("open", "logs"))   # the Dock adds Quit itself
            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            image = NSImage.alloc().initWithContentsOfFile_(str(app.MENUBAR_ICON))
            if image is not None:
                image.setSize_((18, 18))
                image.setTemplate_(True)   # drawn in the menu bar's colour, light or dark
                self.item.button().setImage_(image)
            else:
                self.item.button().setTitle_("Seg")
            self.item.button().setToolTip_("Seg-Studio")
            menu = self.entries(("open", "logs", None, "quit"))
            self.item.setMenu_(menu)
            return menu

        def applicationDockMenu_(self, _app):
            return self.dock_menu

        def applicationDidFinishLaunching_(self, _note):
            self.build()
            if self.self_test:
                AppHelper.callLater(1.0, NSApplication.sharedApplication().terminate_, None)
            else:
                self.open_(None)

        def applicationShouldHandleReopen_hasVisibleWindows_(self, _app, _flag):
            self.open_(None)
            return False

        def say_(self, key):
            if self.item is not None:
                self.item.button().setToolTip_(self.text.get(key, "Seg-Studio"))

        def alert_(self, key):
            box = NSAlert.alloc().init()
            box.setMessageText_("Seg-Studio")
            box.setInformativeText_(self.text.get(key, key))
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            box.runModal()

        def open_(self, _sender):
            threading.Thread(target=lambda: self.controller.ensure_running(open_ui=True), daemon=True).start()

        def logs_(self, _sender):
            self.controller.open_logs()

        def confirm_(self, key):
            """Yes or no, on the main thread: the person decides whether to stop a run."""
            return choose(self.text[key], self.text["yes"], self.text["no"])

        def ask_(self, box):
            """The controller's question, on the main thread; the answer goes back in box."""
            box["answer"] = choose(self.text[box["key"]], self.text["restart"], self.text["keep"])

        def quit_(self, _sender):
            NSApplication.sharedApplication().terminate_(None)

        def applicationShouldTerminate_(self, _app):
            """Every way of quitting comes here: stop the services first, asking when a run is going."""
            if self.self_test or self.stopped:
                return NSTerminateNow
            if not self.controller.begin_stop():
                return NSTerminateCancel        # a stop is going; the app quits when it ends
            try:
                # Only this folder's services, or the ones this app started:
                # a program found on the port, or a Seg-Studio from another
                # folder the person chose to keep, is left running.
                if not self.controller.owns_services():
                    return NSTerminateNow
                if app.busy() and not self.confirm_("busy"):
                    self.controller.cancel_stop()
                    return NSTerminateCancel
            except Exception:  # noqa: BLE001 - the app still quits; app.log says why
                app.log.exception("checking the services before quitting failed")

            def stop():
                try:
                    self.controller.stop()
                finally:
                    AppHelper.callAfter(self.stopped_, None)

            threading.Thread(target=stop, daemon=True).start()
            return NSTerminateLater

        def stopped_(self, _none):
            self.stopped = True
            NSApplication.sharedApplication().replyToApplicationShouldTerminate_(True)

    _DELEGATE = SegStudioMenuBarDelegate
    return _DELEGATE


def run(self_test: bool = False) -> int:
    text = app.LABELS[ui_lang()]
    try:
        cls = delegate_class()
        from AppKit import NSApplication, NSApplicationActivationPolicyRegular, NSImage
        from Foundation import NSBundle, NSMutableDictionary
        from PyObjCTools import AppHelper
    except ImportError:
        app.log.warning("pyobjc is not installed: no menu bar item")
        return 0 if self_test else _run_without_menu(text)

    held = None if self_test else _lock()
    if not self_test and held is None:
        # The app is already running: opening it again opens the screen.
        return 0 if app.Controller(platform="darwin").ensure_running(open_ui=True) else 1

    nsapp = NSApplication.sharedApplication()
    nsapp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    if NSBundle.mainBundle().bundleIdentifier() != app.MACOS_BUNDLE_ID:
        # Run by a python outside Seg-Studio.app (no framework Python to copy
        # into it): the Dock says Python, but shows Seg-Studio's icon.
        icon = NSImage.alloc().initWithContentsOfFile_(str(app.ICON))
        if icon is not None:
            nsapp.setApplicationIconImage_(icon)
    holder = {}

    def ask(key):
        box = NSMutableDictionary.dictionaryWithDictionary_({"key": key})
        holder["d"].performSelectorOnMainThread_withObject_waitUntilDone_("ask:", box, True)
        return bool(box.get("answer"))

    controller = app.Controller(platform="darwin",
                                say=lambda key: AppHelper.callAfter(holder["d"].say_, key),
                                alert=lambda key: AppHelper.callAfter(holder["d"].alert_, key),
                                ask=ask)
    holder["d"] = cls.alloc().initWithController_text_selfTest_(controller, text, self_test)
    nsapp.setDelegate_(holder["d"])
    AppHelper.runEventLoop()
    return 0


if __name__ == "__main__":
    sys.exit(run(self_test="--self-test" in sys.argv))
