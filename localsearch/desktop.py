"""Optional desktop shell: a native window, a global hotkey and a tray icon.

    python -m localsearch.app --window                 # or "desktop": true in settings.json

The web UI is unchanged; it is hosted in a native window (pywebview -> WebView2,
which ships with Windows 10/11) instead of a browser tab. A global hotkey
(default Ctrl+Alt+Space) brings the window to the front with the search box
focused -- the Everything/Spotlight reflex. A tray icon (pystray, optional)
offers Show / Quit. Every piece degrades: no pywebview -> the browser, as before;
no pystray -> no tray; hotkey taken by another app -> logged, window still works.

Windows-only behaviour here (RegisterHotKey, WebView2) could not be exercised
in CI; the hotkey parser is unit-tested.
"""
import logging
import sys
import threading

log = logging.getLogger("localsearch.desktop")

MOD = {"alt": 0x0001, "ctrl": 0x0002, "control": 0x0002, "shift": 0x0004, "win": 0x0008}
MOD_NOREPEAT = 0x4000
VK = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "backquote": 0xC0, "`": 0xC0}
VK.update({f"f{i}": 0x6F + i for i in range(1, 13)})


def parse_hotkey(spec):
    """'ctrl+alt+space' -> (modifiers, virtual key) for RegisterHotKey."""
    mods, key = 0, None
    for part in (p.strip().lower() for p in spec.split("+") if p.strip()):
        if part in MOD:
            mods |= MOD[part]
        elif part in VK:
            key = VK[part]
        elif len(part) == 1 and part.isalnum():
            key = ord(part.upper())
        else:
            raise ValueError("unknown key in hotkey: %r" % part)
    if key is None or not mods:
        raise ValueError("a hotkey needs at least one modifier and one key: %r" % spec)
    return mods | MOD_NOREPEAT, key


def _hotkey_thread(spec, on_press):
    """RegisterHotKey + a message loop on its own thread (Windows only)."""
    import ctypes
    from ctypes import wintypes
    mods, vk = parse_hotkey(spec)
    user32 = ctypes.windll.user32
    if not user32.RegisterHotKey(None, 1, mods, vk):
        log.warning("hotkey %s is taken by another program; not registered", spec)
        return
    log.info("global hotkey %s registered", spec)
    msg = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312:            # WM_HOTKEY
                try:
                    on_press()
                except Exception as e:           # never let the loop die
                    log.warning("hotkey handler failed: %s", e)
    finally:
        user32.UnregisterHotKey(None, 1)


def _tray(window, quit_cb, title):
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception:
        return None
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=(37, 99, 235, 255))
    d.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
    icon = pystray.Icon("scry", img, title, menu=pystray.Menu(
        pystray.MenuItem("Show / 显示", lambda: _show(window), default=True),
        pystray.MenuItem("Quit / 退出", lambda: quit_cb())))
    threading.Thread(target=icon.run, daemon=True, name="scry-tray").start()
    return icon


def _show(window):
    window.show()
    window.restore()
    try:
        window.evaluate_js("document.getElementById('q').focus();"
                           "document.getElementById('q').select();")
    except Exception:
        pass


def available():
    if sys.platform != "win32":
        return False
    try:
        import webview  # noqa: F401
        return True
    except Exception:
        return False


def run(url, serve, shutdown, hotkey="ctrl+alt+space", title="Scry", on_ready=None):
    """Serve the app on a background thread and host the UI in a native window
    (blocks until the window closes). Returns False if the shell is unavailable
    (the caller then falls back to the browser)."""
    if not available():
        return False
    import webview
    threading.Thread(target=serve, daemon=True, name="scry-http").start()
    window = webview.create_window(title, url, width=1040, height=780, min_size=(640, 480))
    icon = None

    def quit_app():
        try:
            if icon is not None:
                icon.stop()
        finally:
            window.destroy()

    def on_start():
        nonlocal icon
        icon = _tray(window, quit_app, title)
        if on_ready is not None:
            on_ready(lambda: _show(window))     # lets a second launch raise this window
        if hotkey:
            threading.Thread(target=_hotkey_thread, args=(hotkey, lambda: _show(window)),
                             daemon=True, name="scry-hotkey").start()

    try:
        webview.start(on_start)
    finally:
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
        shutdown()
    return True
