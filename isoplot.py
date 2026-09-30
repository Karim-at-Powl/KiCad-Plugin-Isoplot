"""Net Isoplot - KiCad IPC API plugin entry point.

KiCad starts this script in its own Python process when the toolbar button is
pressed (see plugin.json). It opens a window showing the geodesic
(along-copper) distance from the selected pad/via to every reachable point on
the same net, and keeps it up to date while you keep working in KiCad.

Only one window runs at a time: pressing the button again brings the existing
window to the front and switches it to the current selection.

The process has no console (pythonw), so diagnostics go to a log file in the
temp directory (``net_isoplot.log``).
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LOG_PATH = os.path.join(tempfile.gettempdir(), "net_isoplot.log")
_PORT_FILE = os.path.join(tempfile.gettempdir(), "net_isoplot.port")
_HELLO = b"net-isoplot:raise\n"
_ACK = b"ok\n"

log = logging.getLogger("isoplot")


def _notify_running_instance():
    """Ask an already running window to come forward. True if one answered."""
    try:
        with open(_PORT_FILE) as f:
            port = int(f.read().strip())
    except (OSError, ValueError):
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0) as s:
            s.sendall(_HELLO)
            return s.recv(16) == _ACK
    except OSError:
        return False


def _listen_for_instances(on_raise):
    """Accept "come forward" requests from later launches on a local port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    with open(_PORT_FILE, "w") as f:
        f.write(str(server.getsockname()[1]))

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(1.0)
                try:
                    if conn.recv(64) == _HELLO:
                        conn.sendall(_ACK)
                        on_raise()
                except OSError:
                    pass

    threading.Thread(target=serve, name="isoplot-instance", daemon=True).start()
    return server


def _enable_hidpi():
    """Draw at the display's real resolution.

    KiCad's pythonw.exe declares no DPI awareness in its manifest, so Windows
    renders the window at 96 DPI and stretches the bitmap on scaled displays,
    which blurs everything. This must run before the first window exists.
    """
    if sys.platform != "win32":
        return
    import ctypes
    try:  # Windows 10 1703+: per-monitor v2, rescales when moved between displays
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:  # Windows 8.1+: per-monitor
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def _install_excepthooks():
    def hook(exc_type, exc, tb):
        log.critical("uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def main():
    logging.basicConfig(filename=LOG_PATH, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _install_excepthooks()
    if _notify_running_instance():
        log.info("another window is already open; asked it to come forward")
        return

    _enable_hidpi()
    import wx
    from live import LiveSession
    from viewer import IsoplotFrame

    app = wx.App(False)
    frame = IsoplotFrame(icon_path=os.path.join(HERE, "icons", "icon-48.png"))
    session = LiveSession(frame)
    frame.attach(session)

    def come_forward():
        frame.raise_window()
        session.refresh(adopt_selection=True)

    server = _listen_for_instances(lambda: wx.CallAfter(come_forward))

    def on_close(evt):
        session.stop()
        server.close()
        try:
            os.remove(_PORT_FILE)
        except OSError:
            pass
        evt.Skip()

    frame.Bind(wx.EVT_CLOSE, on_close)
    frame.Show()
    session.start()
    app.MainLoop()
    log.info("closed")


if __name__ == "__main__":
    main()
