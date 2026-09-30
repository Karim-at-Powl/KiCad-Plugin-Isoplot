"""Background workers that keep the viewer in sync with KiCad.

Two daemon threads:

* the *poller* owns the KiCad API connection. Several times a second it reads
  the selection; about once a second (slower on big nets) it re-reads the seed
  net and hands changed geometry to the solver. The IPC API has no change
  notifications, so polling is the only option; a fingerprint of the net's
  items keeps unchanged boards from being re-solved.
* the *solver* computes distance fields: a quick coarse pass first so the view
  reacts immediately, then a fine pass. A newer job cancels the one in progress.

Everything the UI sees goes through ``wx.CallAfter``.
"""

from __future__ import annotations

import logging
import threading
import time

import wx
from kipy import KiCad
from kipy.errors import ApiError, ConnectionError as KiCadConnectionError
from kipy.proto.common.types import DocumentType

import distance_field as df
import kicad_source as ks

log = logging.getLogger(__name__)

SELECTION_POLL_S = 0.25
MIN_GEOMETRY_POLL_S = 1.0
BUSY_NOTICE_S = 0.5            # report "KiCad is busy" after this long
GIVE_UP_AFTER_S = 30.0          # close once KiCad has been unreachable this long
API_TIMEOUT_MS = 3000

COARSE_CELLS = 250_000          # grid cells summed over layers
COARSE_MIN_PITCH_NM = 100_000
FINE_CELLS = 2_500_000
FINE_MIN_PITCH_NM = 50_000

MM = 1e6


def _call_if_alive(window, method, args):
    if window:  # False once the wx window has been destroyed
        getattr(window, method)(*args)


class LiveSession:
    """Drives an ``IsoplotFrame`` from a running KiCad."""

    def __init__(self, frame):
        self._frame = frame
        self._follow = True
        self._adopt = False      # take the current selection even when pinned
        self._force = False      # re-read and re-solve even if unchanged
        self._outline_request = False
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._solver = _Solver(self._post)
        self._thread = threading.Thread(target=self._run, name="isoplot-poller",
                                        daemon=True)

    # -- control (any thread) -------------------------------------------------
    def start(self):
        self._solver.start()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._solver.stop()

    def set_follow(self, follow):
        self._follow = bool(follow)
        self._wake.set()

    def refresh(self, adopt_selection=False):
        self._force = True
        self._adopt = self._adopt or adopt_selection
        self._wake.set()

    def refresh_outline(self):
        self._outline_request = True
        self._wake.set()

    def _post(self, method, *args):
        wx.CallAfter(_call_if_alive, self._frame, method, args)

    # -- poller thread --------------------------------------------------------
    def _run(self):
        kicad = None
        reader = None
        seed_ids = ()
        last_fp = None
        next_geometry = 0.0
        outline_fp = None
        unreachable_since = None
        busy_since = None
        busy_shown = False
        last_status = None

        def status(text):
            nonlocal last_status
            if text != last_status:
                last_status = text
                self._post("set_status", text)

        while not self._stop.is_set():
            now = time.monotonic()
            try:
                if kicad is None:
                    kicad = KiCad(client_name="net-isoplot", timeout_ms=API_TIMEOUT_MS)
                if reader is None:
                    reader = ks.BoardReader(kicad.get_board())
                    seed_ids, last_fp, outline_fp = (), None, None
                    log.info("attached to board %s", reader.name)
                    status("Connected to %s" % reader.name)
                unreachable_since = None

                # The outline is read when attaching to a board and on request
                # ("Update Board Outline"), not polled.
                if outline_fp is None or self._outline_request:
                    fp, rings, chains = reader.board_outline()
                    self._outline_request = False
                    if fp != outline_fp:
                        outline_fp = fp
                        self._post("set_outline", rings, chains)

                force, self._force = self._force, False
                adopt, self._adopt = self._adopt, False
                selected = reader.selected_seed_ids()
                if selected and selected != seed_ids and (self._follow or adopt or not seed_ids):
                    seed_ids, last_fp = selected, None

                if not seed_ids:
                    status("Select a pad or via in the PCB editor")
                elif last_fp is None or force or now >= next_geometry:
                    if not self._same_board(kicad, reader):
                        reader = None
                        continue
                    t0 = time.perf_counter()
                    geometry = reader.fetch(seed_ids, None if force else last_fp)
                    elapsed = time.perf_counter() - t0
                    next_geometry = time.monotonic() + max(MIN_GEOMETRY_POLL_S, 5 * elapsed)
                    if geometry is not None:
                        log.info("net %s changed (read in %.0f ms)", geometry.net_name,
                                 elapsed * 1e3)
                        last_fp = geometry.fingerprint
                        last_status = None
                        self._solver.submit(geometry)
                busy_since = None
                if busy_shown:
                    busy_shown = False
                    self._post("set_busy", False)
            except ks.SeedError as e:
                seed_ids, last_fp = (), None
                status("%s Select a pad or via." % e)
            except KiCadConnectionError as e:
                kicad = reader = None
                unreachable_since = unreachable_since or now
                status("Waiting for KiCad...")
                log.info("connection problem: %s", e)
                if now - unreachable_since > GIVE_UP_AFTER_S:
                    log.info("KiCad unreachable for %.0f s, closing", GIVE_UP_AFTER_S)
                    self._post("Close")
                    return
            except ApiError as e:
                if ks.is_busy(e):
                    # An interactive tool (routing, via placement, drag...) is
                    # active; KiCad refuses all API calls until it ends.
                    busy_since = busy_since or now
                    if not busy_shown and now - busy_since >= BUSY_NOTICE_S:
                        busy_shown = True
                        self._post("set_busy", True)
                else:
                    # No board open, the board was closed/replaced, or a stale
                    # document: start over with a fresh board handle.
                    log.info("API error: %s", e)
                    reader = None
                    status("Open a board in the PCB editor")
            except Exception:
                log.exception("poller error")
                reader = None
                status("Unexpected error (see log); retrying")
            self._wake.wait(SELECTION_POLL_S)
            self._wake.clear()

    @staticmethod
    def _same_board(kicad, reader):
        docs = kicad.get_open_documents(DocumentType.DOCTYPE_PCB)
        return bool(docs) and docs[0] == reader.board.document


class _Solver:
    """Solves the latest submitted NetGeometry on a worker thread."""

    def __init__(self, post):
        self._post = post
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._job = None
        self._gen = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="isoplot-solver",
                                        daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopped = True
        self._event.set()

    def submit(self, geometry):
        with self._lock:
            self._job = geometry
            self._gen += 1
        self._event.set()

    def _run(self):
        while True:
            self._event.wait()
            self._event.clear()
            if self._stopped:
                return
            with self._lock:
                geometry, gen = self._job, self._gen
            try:
                self._solve(geometry, gen)
            except df.Cancelled:
                pass
            except Exception:
                log.exception("solve failed")
                self._post("set_status", "Solve failed (see log)")

    def _solve(self, geometry, gen):
        def cancel():
            return self._stopped or self._gen != gen

        prims = geometry.prims
        if prims.num_layers == 0:
            self._post("show_message", "No copper found on net %s." % geometry.net_name)
            return
        fine = df.pitch_for_budget(prims.bbox, prims.num_layers, FINE_CELLS,
                                   FINE_MIN_PITCH_NM)
        coarse = df.pitch_for_budget(prims.bbox, prims.num_layers, COARSE_CELLS,
                                     COARSE_MIN_PITCH_NM)
        passes = [coarse, fine] if coarse > 1.25 * fine else [fine]
        for i, pitch in enumerate(passes):
            final = i == len(passes) - 1
            self._post("set_status", "Computing %s (grid %.2f mm)..."
                       % (geometry.net_name, pitch / MM))
            t0 = time.perf_counter()
            field = df.solve(prims, pitch, cancel=cancel)
            elapsed = time.perf_counter() - t0
            if cancel():
                return
            log.info("solved %s @ %.3f mm in %.2f s", geometry.net_name, pitch / MM, elapsed)
            self._post("show_result", geometry, field, final, elapsed)
