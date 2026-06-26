#!/usr/bin/env python3
"""Native always-on-top meeting-detection HUD (Notion/Granola-style).

A borderless floating panel pinned to the bottom-centre of the screen, shown
when a meeting is detected (mic in use / known meeting app) and you're NOT
already recording. Click 開始轉譯 to open the live page; ✕ dismisses until the
call ends. PyObjC only — no Xcode. meeting_watch.py execs into this when AppKit
is importable, else falls back to its notification poll-loop.

Network polling runs on a daemon thread (localhost /detect) and hands the result
to the main thread, so the AppKit run loop / UI never blocks on a request.
"""
import os
import sys
import threading
import time
import urllib.request

PORT = os.environ.get("MEETING_PORT", "8765")
DETECT = f"http://127.0.0.1:{PORT}/detect"
LIVE = f"http://127.0.0.1:{PORT}/live"
POLL_S = float(os.environ.get("MEETING_WATCH_POLL_S", "5"))

from AppKit import (  # noqa: E402
    NSApplication, NSApp, NSPanel, NSView, NSColor, NSTextField, NSButton,
    NSFont, NSScreen, NSObject, NSTimer, NSWorkspace,
    NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSStatusWindowLevel, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSBezelStyleRounded, NSTextAlignmentLeft)
from Foundation import NSMakeRect, NSMakePoint, NSURL  # noqa: E402
import objc  # noqa: E402

W, H = 470, 84  # panel size


def _detect():
    try:
        with urllib.request.urlopen(DETECT, timeout=3) as r:
            import json
            return json.load(r)
    except Exception:
        return None


def _label(frame, size, color, bold=False):
    t = NSTextField.alloc().initWithFrame_(frame)
    t.setBezeled_(False)
    t.setDrawsBackground_(False)
    t.setEditable_(False)
    t.setSelectable_(False)
    t.setTextColor_(color)
    t.setFont_(NSFont.boldSystemFontOfSize_(size) if bold
               else NSFont.systemFontOfSize_(size))
    t.setAlignment_(NSTextAlignmentLeft)
    return t


class HUD(NSObject):
    def init(self):
        self = objc.super(HUD, self).init()
        if self is None:
            return None
        self._visible = False
        self._dismissed = False  # ✕/開始 pressed -> suppress until the call ends
        self._pending = None
        self._build()
        return self

    def _build(self):
        rect = NSMakeRect(0, 0, W, H)
        mask = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, mask, NSBackingStoreBuffered, False)
        panel.setLevel_(NSStatusWindowLevel)  # float above normal app windows
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setHidesOnDeactivate_(False)

        bg = NSView.alloc().initWithFrame_(rect)
        bg.setWantsLayer_(True)
        bg.layer().setCornerRadius_(18.0)
        bg.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(0.13, 0.97).CGColor())
        panel.setContentView_(bg)

        icon = _label(NSMakeRect(20, H / 2 - 16, 34, 30), 22, NSColor.whiteColor())
        icon.setStringValue_("📝")
        bg.addSubview_(icon)

        self._title = _label(NSMakeRect(60, 44, 300, 24), 15,
                             NSColor.whiteColor(), bold=True)
        self._title.setStringValue_("偵測到會議")
        bg.addSubview_(self._title)

        self._sub = _label(NSMakeRect(60, 20, 300, 20), 12,
                          NSColor.colorWithCalibratedWhite_alpha_(0.72, 1.0))
        self._sub.setStringValue_("開始轉譯?")
        bg.addSubview_(self._sub)

        btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 142, 24, 118, 36))
        btn.setTitle_("開始轉譯")
        btn.setBezelStyle_(NSBezelStyleRounded)
        btn.setFont_(NSFont.boldSystemFontOfSize_(13))
        btn.setTarget_(self)
        btn.setAction_("start:")
        btn.setKeyEquivalent_("\r")
        bg.addSubview_(btn)

        x = NSButton.alloc().initWithFrame_(NSMakeRect(W - 26, H - 24, 18, 18))
        x.setTitle_("✕")
        x.setBordered_(False)
        x.setFont_(NSFont.systemFontOfSize_(12))
        x.setTarget_(self)
        x.setAction_("dismiss:")
        bg.addSubview_(x)

        self._panel = panel

    def _reposition(self):
        scr = NSScreen.mainScreen()
        if scr is None:
            return
        f = scr.frame()
        x = f.origin.x + (f.size.width - W) / 2
        y = f.origin.y + 60  # 60pt above the bottom edge
        self._panel.setFrameOrigin_(NSMakePoint(x, y))

    def show_(self, app):
        self._sub.setStringValue_((f"{app} 進行中 — 開始轉譯?" if app else "開始轉譯?"))
        self._reposition()
        self._panel.orderFrontRegardless()
        self._visible = True

    def hide(self):
        self._panel.orderOut_(None)
        self._visible = False

    def start_(self, sender):
        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(LIVE))
        self._dismissed = True
        self.hide()

    def dismiss_(self, sender):
        self._dismissed = True
        self.hide()

    def refresh(self):
        """Main-thread: apply the latest /detect result the poller stashed."""
        d = self._pending
        if d is None:
            return
        meeting = bool(d.get("meeting"))
        recording = bool(d.get("recording"))
        if not meeting:
            self._dismissed = False  # call ended -> re-arm for the next one
        show = meeting and not recording and not self._dismissed
        if show and not self._visible:
            self.show_(d.get("app"))
        elif not show and self._visible:
            self.hide()

    def _poll_loop(self):
        while True:
            self._pending = _detect()
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "refresh", None, False)
            time.sleep(POLL_S)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    hud = HUD.alloc().init()
    threading.Thread(target=hud._poll_loop, daemon=True).start()
    NSApp.run()


if __name__ == "__main__":
    if "--check" in sys.argv:  # build + exercise UI methods (no run loop)
        NSApplication.sharedApplication()
        h = HUD.alloc().init()
        h.show_("Zoom")
        assert h._visible
        h._pending = {"meeting": False, "recording": False}
        h.refresh()  # meeting ended -> should hide
        assert not h._visible
        h._pending = {"meeting": True, "recording": False, "app": "Teams"}
        h.refresh()  # detected -> show
        assert h._visible
        h._pending = {"meeting": True, "recording": True, "app": "Teams"}
        h.refresh()  # now recording -> hide
        assert not h._visible
        print("hud ok")
    else:
        main()
