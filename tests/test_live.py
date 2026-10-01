"""Tests for the delta-mode pair logic in live.py (no KiCad needed).

Needs wxPython and kicad-python importable (the plugin's Python environment):
    python tests/test_live.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # the fake board

from live import Channel, SeedPair


def ch(name, net="GND"):
    return Channel(name, net, "label " + name)


def ids(pair):
    return [c.id if c else None for c in pair.slots]


def test_fills_one_after_the_other():
    pair = SeedPair()
    assert "select two" in pair.prompt()
    assert pair.select([ch("a")]) is None
    assert ids(pair) == ["a", None] and not pair.complete
    assert "second pad or via on GND" in pair.prompt() and "1 is label a" in pair.prompt()
    pair.select([ch("a")])                      # the same again: nothing changes
    assert ids(pair) == ["a", None]
    pair.select([ch("b")])
    assert ids(pair) == ["a", "b"] and pair.complete and pair.prompt() is None


def test_a_third_object_replaces_the_one_selected_longer_ago():
    pair = SeedPair()
    for name in "ab":
        pair.select([ch(name)])
    pair.select([ch("c")])                      # a is older: c takes slot 1
    assert ids(pair) == ["c", "b"]
    pair.select([ch("d")])                      # now b is older
    assert ids(pair) == ["c", "d"]
    pair.select([ch("c")])                      # re-selecting c makes d the older one
    pair.select([ch("e")])
    assert ids(pair) == ["c", "e"]


def test_two_at_once():
    pair = SeedPair()
    pair.select([ch("a"), ch("b")])
    assert ids(pair) == ["a", "b"]
    pair.select([ch("c"), ch("a")])             # a keeps its slot
    assert ids(pair) == ["a", "c"]
    pair.select([ch("b"), ch("d")])
    assert ids(pair) == ["b", "d"]
    note = pair.select([ch("x", "GND"), ch("y", "VCC")])
    assert "different nets" in note and ids(pair) == ["b", "d"]
    note = pair.select([ch("p"), ch("q"), ch("r")])
    assert "3 are selected" in note and ids(pair) == ["b", "d"]
    assert "not connected" in pair.select([ch("n", "")])


def test_another_net_starts_a_new_pair():
    pair = SeedPair()
    for name in "ab":
        pair.select([ch(name)])
    pair.select([ch("v", "VCC")])
    assert ids(pair) == ["v", None] and pair.net == "VCC"
    pair.select([ch("w", "VCC")])
    assert ids(pair) == ["v", "w"]


def test_swap_drop_and_clear():
    pair = SeedPair()
    for name in "ab":
        pair.select([ch(name)])
    pair.swap()
    assert ids(pair) == ["b", "a"]
    pair.select([ch("c")])                      # a (now slot 2) is still the older one
    assert ids(pair) == ["b", "c"]
    pair.drop({"c"})
    assert ids(pair) == ["b", None] and "1 is label b" in pair.prompt()
    pair.drop({"b"})
    pair.select([ch("d")])
    assert ids(pair) == ["d", None]
    pair.select([ch("e")])
    pair.drop({"d"})
    assert "2 is label e" in pair.prompt()
    pair.select([ch("f")])                      # fills the free slot 1
    assert ids(pair) == ["f", "e"]
    pair.clear()
    assert ids(pair) == [None, None] and pair.net is None


def test_session_walks_through_delta_mode():
    """The real poller and solver against a fake KiCad: the prompts while the
    pair fills, the labels above the scale, then a delta result."""
    from types import SimpleNamespace as NS
    import wx
    import live
    import viewer
    from test_kicad_source import board_fixture

    board = board_fixture()
    board.document = "doc"
    board.selection = ()
    board.get_shapes = lambda: []
    pad = board.items[0]
    pad.proto.number = "1"
    board.get_footprints = lambda: [
        NS(reference_field=NS(text=NS(value="U1")), definition=NS(pads=[pad]))]

    class FakeKiCad:
        def __init__(self, **_):
            pass

        def get_version(self):
            return NS(major=9, minor=0, patch=5, full_version="9.0.5")

        def get_board(self):
            return board

        def get_open_documents(self, _):
            return ["doc"]

    real_kicad = live.KiCad
    live.KiCad = FakeKiCad
    app = wx.App(False)
    frame = viewer.IsoplotFrame()
    session = live.LiveSession(frame)
    frame.attach(session)
    seen = {}

    def snap(name):
        seen[name] = (frame._view._message, [t.GetLabel() for t in frame._channel_text],
                      frame._view._heat, frame.GetStatusBar().GetStatusText())

    steps = [
        (lambda: (frame._mode_delta.SetValue(True), frame._on_mode(None)), None),
        (lambda: None, "empty"),
        (lambda: setattr(board, "selection", ("pad1",)), None),
        (lambda: None, "one"),
        (lambda: setattr(board, "selection", ("pad1", "v1", "t1x")), None),
        (lambda: setattr(board, "selection", ("v1",)), None),
        (lambda: None, "two"),
    ]

    def run(i=0):
        if i == len(steps):
            session.stop()
            frame.Destroy()
            return
        action, name = steps[i]
        action()
        if name:
            snap(name)
        wx.CallLater(700, run, i + 1)

    try:
        session.start()
        wx.CallLater(700, run)
        app.MainLoop()
    finally:
        live.KiCad = real_kicad
    message, labels, heat, _ = seen["empty"]
    assert "select two pads or vias" in message, message
    assert labels == ["1: select a pad or via", "2: select a pad or via"], labels
    message, labels, heat, _ = seen["one"]
    assert "second pad or via on SIG" in message and "U1 pad 1" in message, message
    assert labels == ["1: U1 pad 1", "2: select a pad or via"], labels
    message, labels, heat, footer = seen["two"]
    assert labels == ["1: U1 pad 1", "2: Via at 20.00, 0.00 mm"], labels
    assert heat is not None and heat.delta and not message, (message, heat)
    # pad edge to via ~19 mm along the track, plus the via's 1.5 mm
    assert 19 * 1e6 < heat.delta_scale < 22 * 1e6, heat.delta_scale


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("OK: %d tests passed" % len(tests))
