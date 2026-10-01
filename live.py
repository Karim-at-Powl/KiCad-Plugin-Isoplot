"""Background workers that keep the viewer in sync with KiCad.

Two daemon threads:

* the *poller* owns the KiCad API connection. Several times a second it reads
  the selection; about once a second (slower on big nets) it re-reads the seed
  net and hands changed geometry to the solver. The IPC API has no change
  notifications, so polling is the only option; a fingerprint of the net's
  items keeps unchanged boards from being re-solved.
* the *solver* computes distance fields in two passes: a quick one so the view
  reacts immediately (a coarse grid on big nets, the fine grid on small ones),
  then the fine grid with longer moves, which is about three times slower but
  cuts the distance error from ~2.7 % to ~0.75 %. Each pass replaces the view
  as it finishes. The solver also prepares the result for display, keeping
  that work off the UI thread. A newer job cancels the one in progress.

Everything the UI sees goes through ``wx.CallAfter``.
"""

from __future__ import annotations

import copy
import logging
import threading
import time

import wx
from kipy import KiCad
from kipy.errors import ApiError, ConnectionError as KiCadConnectionError
from kipy.proto.common.types import DocumentType

import distance_field as df
import kicad_source as ks
import viewer

log = logging.getLogger(__name__)

SELECTION_POLL_S = 0.1          # a selection query takes well under 1 ms
MIN_GEOMETRY_POLL_S = 1.0
BUSY_NOTICE_S = 0.5            # report "KiCad is busy" after this long
GIVE_UP_AFTER_S = 30.0          # close once KiCad has been unreachable this long
API_TIMEOUT_MS = 3000

COARSE_CELLS = 250_000          # grid cells summed over layers
COARSE_MIN_PITCH_NM = 100_000
FINE_CELLS = 2_500_000
FINE_MIN_PITCH_NM = 50_000
# Below this many (estimated) copper cells on the fine grid, the fine pass is
# quick enough on its own; a coarse pass first would only add a redraw.
COARSE_ABOVE_CELLS = 400_000
# Move reach of the quick passes and of the last, more accurate one (see
# distance_field.move_set: 16 and 48 moves).
QUICK_REACH = 2
FINAL_REACH = 4

MM = 1e6


def _call_if_alive(window, method, args):
    if window:  # False once the wx window has been destroyed
        getattr(window, method)(*args)


class Channel:
    """One object of the delta pair: a pad or via by id, its net and label."""

    def __init__(self, item_id, net, label):
        self.id, self.net, self.label = item_id, net, label

    def __eq__(self, other):
        return isinstance(other, Channel) and vars(self) == vars(other)

    def __repr__(self):
        return "Channel(%r, %r, %r)" % (self.id, self.net, self.label)


class SeedPair:
    """The two objects delta mode compares, as picked by selecting in KiCad.

    The pair is the last two different pads/vias selected on one net. An
    object keeps its slot (1 or 2, and so its colour) while it stays in the
    pair: a new one takes the free slot, else replaces the one selected
    longer ago. Selecting two at once sets both. Selecting one on another
    net starts a new pair there.
    """

    def __init__(self):
        self.slots = [None, None]       # Channel or None
        self._recent = []               # slot indices, most recently selected last

    @property
    def complete(self):
        return None not in self.slots

    @property
    def ids(self):
        return tuple(c.id for c in self.slots if c is not None)

    @property
    def net(self):
        return next((c.net for c in self.slots if c is not None), None)

    def _touch(self, i):
        self._recent = [j for j in self._recent if j != i] + [i]

    def select(self, chosen):
        """Take a selection (Channels, in KiCad's order). Returns None, or why
        it was not taken."""
        if not chosen:
            return None
        if len(chosen) > 2:
            return "Delta mode compares two pads/vias; %d are selected." % len(chosen)
        if not all(c.net for c in chosen):
            return "The selected pad/via is not connected to a net."
        if len(chosen) == 2:
            a, b = chosen
            if a.net != b.net:
                return "The two selected are on different nets (%s, %s)." % (a.net, b.net)
            old = [c.id if c else None for c in self.slots]
            if a.id == old[1] or b.id == old[0]:
                a, b = b, a             # whichever was already in the pair stays put
            self.slots, self._recent = [a, b], [0, 1]
            return None
        (c,) = chosen
        if self.net is not None and c.net != self.net:
            self.slots, self._recent = [c, None], [0]
            return None
        for i, s in enumerate(self.slots):
            if s is not None and s.id == c.id:
                self.slots[i] = c       # refreshed label
                self._touch(i)
                return None
        i = self.slots.index(None) if None in self.slots else self._recent[0]
        self.slots[i] = c
        self._touch(i)
        return None

    def swap(self):
        self.slots.reverse()
        self._recent = [1 - i for i in self._recent]

    def drop(self, ids):
        """Empty the slots of these ids (e.g. deleted objects)."""
        for i, s in enumerate(self.slots):
            if s is not None and s.id in ids:
                self.slots[i] = None
                self._recent = [j for j in self._recent if j != i]

    def clear(self):
        self.slots, self._recent = [None, None], []

    def prompt(self):
        """What to do next while the pair is incomplete (None once complete)."""
        if self.complete:
            return None
        if not self.ids:
            return ("Delta mode: select two pads or vias on one net\n"
                    "(one after the other, or both at once).")
        i = 0 if self.slots[0] is not None else 1
        return ("Delta mode: now select a second pad or via on %s\n(%d is %s)."
                % (self.net, i + 1, self.slots[i].label))


class LiveSession:
    """Drives an ``IsoplotFrame`` from a running KiCad."""

    def __init__(self, frame):
        self._frame = frame
        self._follow = True
        self._adopt = False      # take the current selection even when pinned
        self._force = False      # re-read and re-solve even if unchanged
        self._outline_request = False
        self._delta = False      # delta mode: compare the two objects of a SeedPair
        self._mode_changed = False
        self._swap = False
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

    def set_mode(self, delta):
        """Isoplot (False) or delta mode (True)."""
        self._delta = bool(delta)
        self._mode_changed = True
        self._solver.cancel()           # a result of the other mode is of no use now
        self._wake.set()

    def swap_pair(self):
        self._swap = True
        self._wake.set()

    def _post(self, method, *args):
        wx.CallAfter(_call_if_alive, self._frame, method, args)

    # -- poller thread --------------------------------------------------------
    def _run(self):
        kicad = None
        features = None             # ks.KiCadFeatures of the connected KiCad
        reader = None
        seed_ids = ()               # isoplot mode: the seeds
        pair = SeedPair()           # delta mode: the two objects
        last_selected = None        # selection as last read (delta mode acts on changes)
        shown_channels = None       # pair labels as last sent to the window
        refs_tried = None           # pair ids the footprint references were read for
        prompt_shown = None         # delta prompt the view shows, if any
        last_fp = None
        next_geometry = 0.0
        outline_fp = None
        holes = None                # board_holes() as last sent to the window
        unreachable_since = None
        busy_since = None
        busy_shown = False
        last_status = None
        changed_at = None           # when a new selection was noticed

        def status(text):
            nonlocal last_status
            if text != last_status:
                last_status = text
                self._post("set_status", text)

        def busy(is_busy):
            """Report "KiCad is busy" once it has lasted a moment."""
            nonlocal busy_since, busy_shown
            if not is_busy:
                busy_since = None
                if busy_shown:
                    busy_shown = False
                    self._post("set_busy", False)
                return
            busy_since = busy_since or now
            if not busy_shown and now - busy_since >= BUSY_NOTICE_S:
                busy_shown = True
                self._post("set_busy", True,
                           reader is not None and reader.single_pad_selected())

        while not self._stop.is_set():
            now = time.monotonic()
            try:
                if kicad is None:
                    kicad = KiCad(client_name="net-isoplot", timeout_ms=API_TIMEOUT_MS)
                    features = None
                if features is None:
                    # Once per connection: what this KiCad's API offers.
                    features = ks.KiCadFeatures.of(kicad)
                    log.info("connected to %s", features)
                if reader is None:
                    reader = ks.BoardReader(kicad.get_board(), features)
                    seed_ids, last_fp, outline_fp, holes = (), None, None, None
                    log.info("attached to board %s", reader.name)
                    status("Connected to %s" % reader.name)
                unreachable_since = None

                # The outline is read when attaching to a board and on request
                # ("Update Board Outline"), not polled. KiCad refuses it while
                # busy (e.g. one pad selected); it is simply tried again.
                refresh_holes = False
                if outline_fp is None or self._outline_request:
                    outline = self._try_while_busy(reader.board_outline)
                    if outline is not None:
                        fp, rings, chains = outline
                        refresh_holes, self._outline_request = self._outline_request, False
                        if fp != outline_fp:
                            outline_fp = fp
                            self._post("set_outline", rings, chains)

                force, self._force = self._force, False
                adopt, self._adopt = self._adopt, False
                delta = self._delta
                if self._mode_changed:
                    # Take the current selection in the new mode; without
                    # one, carry the seeds over.
                    self._mode_changed = False
                    force = adopt = True
                    last_fp = prompt_shown = shown_channels = None
                    self._post("set_note", None)
                    if delta and not pair.ids and 1 <= len(seed_ids) <= 2:
                        pair.select(self._channels(reader, map(reader.known_item, seed_ids)))
                    elif not delta and not seed_ids:
                        seed_ids = pair.ids
                selected = reader.selected_seed_ids()
                if delta:
                    if self._swap:
                        self._swap = False
                        pair.swap()
                        last_fp = None
                    if selected and (selected != last_selected or adopt) and (
                            self._follow or adopt or not pair.complete):
                        before = list(pair.slots)
                        self._post("set_note",
                                   pair.select(self._channels(reader, reader.selected_items())))
                        if pair.slots != before:
                            last_fp = None
                            changed_at = time.perf_counter()
                    refs_tried = self._label_pair(reader, pair, refs_tried)
                    labels = (tuple(c.label if c else None for c in pair.slots), pair.net)
                    if labels != shown_channels:
                        shown_channels = labels
                        self._post("set_channels", *labels)
                elif selected and selected != seed_ids and (self._follow or adopt or not seed_ids):
                    seed_ids, last_fp = selected, None
                    changed_at = time.perf_counter()
                last_selected = selected
                # Kept current so a seed can be resolved while KiCad is busy.
                # The board's pad drills come from it too, so moved holes show
                # up within a few seconds.
                reader.keep_snapshot_fresh(now=refresh_holes)
                board_holes = reader.board_holes()
                if board_holes is not None and board_holes is not holes and board_holes != holes:
                    holes = board_holes
                    self._post("set_holes", holes)

                seeds = (pair.ids if pair.complete else ()) if delta else seed_ids
                if delta and not pair.complete:
                    prompt = pair.prompt()
                    if prompt != prompt_shown:
                        prompt_shown = prompt
                        self._post("show_message", prompt)
                    status("Select pads or vias in the PCB editor")
                    busy(False)
                elif not seeds:
                    status("Select a pad or via in the PCB editor")
                    busy(False)
                elif last_fp is None or force or now >= next_geometry:
                    if not self._same_board(kicad, reader):
                        reader = None
                        continue
                    t0 = time.perf_counter()
                    geometry = reader.fetch(seeds, None if force else last_fp)
                    elapsed = time.perf_counter() - t0
                    next_geometry = time.monotonic() + max(MIN_GEOMETRY_POLL_S, 5 * elapsed)
                    if geometry is not None and delta and len(geometry.seed_ids) < 2:
                        # One of the pair moved to another net (or is gone).
                        gone = set(pair.ids) - set(geometry.seed_ids)
                        names = [c.label for c in pair.slots if c and c.id in gone]
                        pair.drop(gone)
                        self._post("set_note", "%s is no longer on %s."
                                   % (" and ".join(names), geometry.net_name))
                        last_fp = None
                        continue
                    if geometry is not None:
                        geometry.delta = delta
                        prompt_shown = None
                        log.info("net %s changed (read in %.0f ms%s)", geometry.net_name,
                                 elapsed * 1e3, ", from the snapshot" if reader.stale else "")
                        last_fp = geometry.fingerprint
                        last_status = None
                        geometry.changed_at = changed_at or t0
                        changed_at = None
                        self._solver.submit(geometry)
                    # From the snapshot: fine for a new selection, but edits
                    # only show once KiCad answers again.
                    busy(reader.stale)
            except ks.SeedError as e:
                if self._delta:
                    pair.clear()
                    self._post("set_note", str(e))
                else:
                    seed_ids = ()
                    status("%s Select a pad or via." % e)
                last_fp = None
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
                    # active; KiCad refuses item reads until it ends.
                    busy(True)
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
    def _channels(reader, items):
        """Pads/vias as Channels (None entries skipped)."""
        return [Channel(i.id.value, i.net.name, reader.label(i)) for i in items if i is not None]

    @staticmethod
    def _label_pair(reader, pair, refs_tried):
        """Refresh the pair's labels. Pads are named after their footprint,
        which is read once more when a pad of the pair isn't known yet (and
        again on the next poll while KiCad is busy). Returns the pair ids the
        footprints have now been read for."""
        items = [reader.known_item(i) for i in pair.ids]
        if pair.ids != refs_tried and reader.needs_refs(i for i in items if i):
            if reader.refresh_refs():
                refs_tried = pair.ids
        for c in pair.slots:
            item = reader.known_item(c.id) if c else None
            if item is not None:
                c.label = reader.label(item)
        return refs_tried

    @staticmethod
    def _try_while_busy(read):
        """``read()``, or None if KiCad is busy."""
        try:
            return read()
        except ApiError as e:
            if ks.is_busy(e):
                return None
            raise

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

    def cancel(self):
        """Drop the job in progress (its result would not be shown)."""
        with self._lock:
            self._job = None
            self._gen += 1

    def _run(self):
        while True:
            self._event.wait()
            self._event.clear()
            if self._stopped:
                return
            with self._lock:
                geometry, gen = self._job, self._gen
            if geometry is None:
                continue
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
        big = df.copper_area(prims) / float(fine) ** 2 > COARSE_ABOVE_CELLS
        # One quick pass for a first picture (coarse grid on big nets), then
        # the accurate one on the fine grid.
        if big and coarse > 1.25 * fine:
            passes = [(coarse, QUICK_REACH), (fine, FINAL_REACH)]
        else:
            passes = [(fine, QUICK_REACH), (fine, FINAL_REACH)]
        for i, (pitch, reach) in enumerate(passes):
            final = i == len(passes) - 1
            self._post("set_status", "computing %s (grid %.2f mm, %d directions)..."
                       % (geometry.net_name, pitch / MM, len(df.move_set(reach))))
            t0 = time.perf_counter()
            if getattr(geometry, "delta", False):
                # From each object of the pair on its own; the view shows
                # which one is nearer.
                field, other = (df.solve(_from_seed(prims, shapes), pitch, cancel=cancel,
                                         reach=reach) for shapes in geometry.seed_shapes)
                display = viewer.prepare_delta(field, other)
            else:
                field = df.solve(prims, pitch, cancel=cancel, reach=reach)
                display = viewer.prepare_layers(field)   # here, not on the UI thread
            elapsed = time.perf_counter() - t0
            if cancel():
                return
            log.info("solved %s @ %.3f mm, %d moves in %.2f s (%.0f ms after the change was seen)",
                     geometry.net_name, pitch / MM, field.num_moves, elapsed,
                     (time.perf_counter() - geometry.changed_at) * 1e3)
            self._post("show_result", geometry, field, display, final, elapsed)


def _from_seed(prims, shapes):
    """``prims`` measured from one seed's copper (``shapes``: its seed discs
    and polys, see NetGeometry.seed_shapes) instead of all the seeds."""
    one = copy.copy(prims)
    one.seed_discs, one.seed_polys = shapes
    one.sources = []
    return one
