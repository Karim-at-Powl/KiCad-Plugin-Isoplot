"""Tests for the heatmap rendering in viewer.py (needs wxPython, no KiCad).

Run with KiCad's Python:  python tests/test_viewer.py
"""
import os
import sys

import numpy as np
import wx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distance_field as df
import viewer

MM = 1_000_000


def square(x0, y0, s):
    return [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]


def heatmap(dist, pitch, max_distance, layers=None):
    """A viewer.Heatmap straight from a (layers, ny, nx) distance array."""
    L, ny, nx = dist.shape
    field = df.DistanceField(nx, ny, L, pitch, (0.0, 0.0), dist)
    prepared = viewer.prepare_layers(field)
    layers = range(L) if layers is None else layers
    return viewer.Heatmap([prepared[li] for li in layers], (0.0, 0.0), pitch, max_distance)


def test_interpolation_is_exact_on_a_linear_field():
    """Bilinear sampling reproduces a field that is linear in x."""
    ny, nx, pitch = 20, 30, 100_000
    dist = np.tile(np.arange(nx, dtype=float) * pitch, (ny, 1))[None]
    heat = heatmap(dist, pitch, dist.max())
    for x in (0.0, 123_456.0, 1_234_567.0, 2_850_000.0):
        assert abs(heat.sample(x, 1_000_000.0) - x) < 1.0, x


def test_no_data_does_not_bleed_in():
    """Next to cells without copper the value comes from the copper side only,
    and far from copper there is no value at all."""
    pitch = 100_000
    dist = np.full((1, 10, 10), np.inf)
    dist[0, :, :5] = 7.0 * MM
    heat = heatmap(dist, pitch, 7.0 * MM)
    assert abs(heat.sample(4.6 * pitch, 5 * pitch) - 7.0 * MM) < 1.0
    assert np.isnan(heat.sample(8 * pitch, 5 * pitch))


def _render(prims, layers, w=200, h=200):
    field = df.solve(prims, 100_000)
    prepared = viewer.prepare_layers(field)
    heat = viewer.Heatmap([prepared[li] for li in layers], field.origin, field.pitch_nm,
                          field.max_distance_nm)
    shapes = [viewer.CopperShapes(prims, li) for li in layers]
    view = (w / (20.0 * MM), (10.0 * MM, 10.0 * MM))   # 20 x 20 mm fills the image
    return viewer.render_heatmap(heat, shapes, view, w, h, block=2), field


def test_render_follows_the_exact_copper_edge():
    """A 10 mm square from (5, 5) mm: opaque inside, transparent outside, and
    the edge falls where the polygon is, not on a grid cell boundary."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(5 * MM, 5 * MM, 10 * MM)])]
    p.sources = [(0, 6 * MM, 6 * MM)]
    p.bbox = (5 * MM, 5 * MM, 15 * MM, 15 * MM)
    r, _ = _render(p, [0])
    bw, bh = r.bitmap.GetWidth(), r.bitmap.GetHeight()
    assert bw < 150 and bh < 150, "rendered more than the net's area"
    part = np.frombuffer(bytes(r.bitmap.ConvertToImage().GetAlphaBuffer()),
                         dtype=np.uint8).reshape(bh, bw)
    alpha = np.zeros((200, 200), dtype=np.uint8)   # back into viewport pixels
    alpha[r.y:r.y + bh, r.x:r.x + bw] = part[:200 - r.y, :200 - r.x]
    row = alpha[100]
    assert row[20] == 0 and row[180] == 0            # 2 mm / 18 mm: outside
    assert row[60] > 140 and row[140] > 140          # inside
    inside = np.flatnonzero(row > 70)                # the 5 mm / 15 mm edges
    assert abs(inside[0] - 50) <= 1 and abs(inside[-1] - 149) <= 1, inside[[0, -1]]
    assert r.value_at(20, 100) is None
    assert 0 <= r.value_at(60, 60) < 2 * MM


def test_nearer_layer_shows_where_layers_overlap():
    """Layer 1 overlaps layer 0 but is only reached through a far via, so the
    overlap shows layer 0's (smaller) distance."""
    p = df.NetPrimitives()
    p.num_layers = 2
    p.polys = [(0, [square(2 * MM, 2 * MM, 8 * MM)]), (1, [square(2 * MM, 2 * MM, 16 * MM)])]
    p.segments = [(0, 10 * MM, 6 * MM, 17 * MM, 6 * MM, int(0.3 * MM))]
    p.vias = [(17 * MM, 6 * MM, [0, 1])]
    p.sources = [(0, 3 * MM, 3 * MM)]
    p.bbox = (2 * MM, 2 * MM, 18 * MM, 18 * MM)
    r, field = _render(p, [0, 1])
    near = r.value_at(40, 40)       # (4, 4) mm: on both layers
    assert near is not None and near < 2 * MM, near
    far = r.value_at(160, 160)      # (16, 16) mm: layer 1 only
    assert far is not None and far > 10 * MM, far


def test_hidden_vias_and_pads_are_not_drawn():
    """A pad and a via on a track: hiding one kind drops only its shapes, and
    the hidden copper reads as "no copper" while the track still shows."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(2 * MM, 9 * MM, 2 * MM)]), (0, [square(12 * MM, 2 * MM, 6 * MM)])]
    p.segments = [(0, 3 * MM, 10 * MM, 17 * MM, 10 * MM, int(0.3 * MM)),
                  (0, 15 * MM, 8 * MM, 15 * MM, 10 * MM, int(0.3 * MM))]  # to the zone
    p.discs = [(0, 17 * MM, 10 * MM, 1 * MM)]
    p.sources = [(0, 3 * MM, 10 * MM)]
    p.bbox = (2 * MM, 2 * MM, 18 * MM, 11 * MM)
    discs, polys = ["via"], ["pad", "zone"]
    everything = viewer.CopperShapes(p, 0, discs, polys)
    assert len(everything.discs) == 1 and len(everything.polys) == 2
    no_vias = viewer.CopperShapes(p, 0, discs, polys, hidden={"via"})
    assert len(no_vias.discs) == 0 and len(no_vias.polys) == 2
    no_pads = viewer.CopperShapes(p, 0, discs, polys, hidden={"pad"})
    assert len(no_pads.discs) == 1 and len(no_pads.polys) == 1
    assert no_pads.poly_boxes[0][0] == 12 * MM, "the zone must stay"

    field = df.solve(p, 100_000)
    heat = viewer.Heatmap(viewer.prepare_layers(field), field.origin, field.pitch_nm,
                          field.max_distance_nm)
    view = (200 / (20.0 * MM), (10.0 * MM, 10.0 * MM))      # 10 px per mm
    bare = viewer.CopperShapes(p, 0, discs, polys, hidden={"pad", "via"})
    shown = viewer.render_heatmap(heat, [everything], view, 200, 200, block=2)
    hidden = viewer.render_heatmap(heat, [bare], view, 200, 200, block=2)
    for px, py in ((25, 95), (175, 93)):                   # on the pad, on the via
        assert shown.value_at(px, py) is not None, (px, py)
        assert hidden.value_at(px, py) is None, (px, py)
    assert hidden.value_at(100, 100) is not None           # the track still shows
    assert hidden.value_at(140, 50) is not None            # and so does the zone


def test_marks_follow_layers_and_switches():
    """Pad outlines once per pad (not per layer), via drills only on shown
    layers, and each only while its switch is on."""
    p = df.NetPrimitives()
    p.num_layers = 2
    pad = square(0, 0, 2 * MM)
    p.polys = [(0, [pad]), (1, [pad]), (1, [square(5 * MM, 0, 3 * MM)])]
    kinds = ["pad", "pad", "zone"]
    vias = [(6 * MM, MM, 300_000, 300_000, 0.0, [1])]
    m = viewer.Marks(p, kinds, vias, [0, 1])
    assert len(m.pad_rings) == 1 and m.via_holes.tolist() == [[6 * MM, MM, 3e5, 3e5, 0.0]]
    assert len(viewer.Marks(p, kinds, vias, [0]).via_holes) == 0   # the via is on layer 1 only
    m = viewer.Marks(p, kinds, vias, [0, 1], hidden={"pad"})
    assert not m.pad_rings and len(m.via_holes) == 1
    assert len(viewer.Marks(p, kinds, vias, [0, 1], hidden={"via"}).via_holes) == 0
    # A through-hole pad's copper gets no outline (its drill marks it).
    assert not viewer.Marks(p, ["tht", "tht", "zone"], vias, [0, 1]).pad_rings
    assert set(viewer._SWITCHED["pad"]) == {"pad", "tht"}, "the Pads switch hides both"


def test_board_holes_are_drawn_without_a_net():
    """The board's drills show with the outline alone, cut out in the
    background grey, before any pad or via is selected."""
    frame = wx.Frame(None)
    try:
        view = viewer._BoardView(frame, wx.Size(200, 200))
        view.SetSize(200, 200)
        view.set_outline([square(0, 0, 20 * MM)], [])
        view.set_holes([(10 * MM, 10 * MM, 4 * MM, 4 * MM, 0.0)])
        bmp = wx.Bitmap(200, 200, 24)
        dc = wx.MemoryDC(bmp)
        view.draw(dc)
        dc.SelectObject(wx.NullBitmap)
        img = bmp.ConvertToImage()
        assert (img.GetRed(100, 100), img.GetGreen(100, 100)) == viewer._BG_COLOUR[:2]
        assert (img.GetRed(100, 40), img.GetGreen(100, 40)) == viewer._BOARD_COLOUR[:2]
    finally:
        frame.Destroy()


def test_oblong_hole_outline():
    """A 2 x 1 slot turned 90 degrees stands upright, with round ends."""
    pts = viewer._oblong(0.0, 0.0, 1.0, 0.5, 90.0)
    assert np.allclose(pts[0], pts[-1]), "the outline must be closed"
    assert np.allclose(np.abs(pts).max(axis=0), (0.5, 1.0))
    assert np.allclose(np.abs(viewer._oblong(0.0, 0.0, 0.5, 1.0, 0.0)).max(axis=0), (0.5, 1.0))


def test_board_view_draws_outline_and_messages():
    """Painting with a message up must not fail (it used to leave the view
    black at startup), and the startup note goes once the outline is in."""
    frame = wx.Frame(None)
    try:
        view = viewer._BoardView(frame, wx.Size(300, 200))
        view.SetSize(300, 200)
        bmp = wx.Bitmap(300, 200, 24)
        dc = wx.MemoryDC(bmp)
        view.draw(dc)                                # "Starting..." only
        view.set_outline([square(0, 0, 50 * MM)], [])
        assert view._message is None
        view.draw(dc)                                # the bare board
        view.set_message("No layers selected\nsecond line")
        view.draw(dc)
        dc.SelectObject(wx.NullBitmap)
        pixel = bmp.ConvertToImage()
        board = viewer._BOARD_COLOUR
        assert (pixel.GetRed(150, 20), pixel.GetGreen(150, 20)) == board[:2], "no board drawn"
    finally:
        frame.Destroy()


def _trace_pair():
    """An 18 mm trace on layer 0 along y = 10 mm, object 1 a 1 mm disc at its
    left end (x = 1 mm), object 2 at its right end (x = 19 mm)."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.segments = [(0, 1 * MM, 10 * MM, 19 * MM, 10 * MM, 1 * MM)]
    p.bbox = (0.5 * MM, 9.5 * MM, 19.5 * MM, 10.5 * MM)
    fields = []
    for x in (1, 19):
        p.seed_discs = [(0, x * MM, 10 * MM, 0.5 * MM)]
        fields.append(df.solve(p, 100_000))
    return p, fields


def _pixels(r, w=200, h=200):
    """A Render's RGBA as viewport pixels (h, w, 4)."""
    img = r.bitmap.ConvertToImage()
    bw, bh = img.GetWidth(), img.GetHeight()
    rgb = np.frombuffer(bytes(img.GetDataBuffer()), dtype=np.uint8).reshape(bh, bw, 3)
    a = np.frombuffer(bytes(img.GetAlphaBuffer()), dtype=np.uint8).reshape(bh, bw, 1)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[r.y:r.y + bh, r.x:r.x + bw] = np.concatenate([rgb, a], axis=2)[:h - r.y, :w - r.x]
    return out


def test_delta_colours_and_values():
    """Nearer to 1: object 1's colour, the middle white, nearer to 2: object
    2's colour; the readout values are the distance to the nearer object and
    the difference."""
    p, (f1, f2) = _trace_pair()
    d = viewer.prepare_delta(f1, f2)
    assert 16.5 * MM < d.scale < 18.5 * MM, d.scale      # seed edge to seed edge
    heat = viewer.Heatmap(d.layers, f1.origin, f1.pitch_nm, f1.max_distance_nm, d.scale)
    view = (200 / (20.0 * MM), (10.0 * MM, 10.0 * MM))   # 10 px per mm
    r = viewer.render_heatmap(heat, [viewer.CopperShapes(p, 0)], view, 200, 200, block=2)
    px = _pixels(r)[100]
    red, blue = viewer.CHANNEL_RGB
    assert px[30, 0] > 180 and px[30, 2] < 120, px[30]     # x = 3 mm: like object 1
    assert px[170, 2] > 150 and px[170, 0] < 100, px[170]  # x = 17 mm: like object 2
    assert px[100, :3].min() > 225, px[100]                # x = 10 mm: white
    key, delta = r.values_at(50, 100)                      # x = 5 mm
    assert abs(key - 3.5 * MM) < 0.3 * MM, key
    assert abs(delta + 10 * MM) < 0.5 * MM, delta   # changes twice as fast
    key, delta = r.values_at(150, 100)
    assert abs(delta - 10 * MM) < 0.5 * MM, delta
    assert r.values_at(100, 50) == (None, None)            # off copper
    assert heat.sample_delta(5 * MM, 10 * MM)[1] < -9.5 * MM
    lines = viewer._delta_readout(key, delta, d.scale)
    assert lines == ["2 is nearer: %.2f mm" % (key / MM), "1 is %.2f mm farther" % (delta / MM)]
    assert viewer._delta_readout(5 * MM, 0.001 * MM, d.scale)[1] == "Equal distance"


def test_delta_over_overlapping_layers_is_the_difference_of_the_isoplots():
    """Two poured layers joined by a via in the middle, object 1 on layer 0,
    object 2 on layer 1. Each object's distance at a point is the nearest
    over the layers there (as its isoplot shows), and the delta is their
    difference: it has no seam where one layer becomes the nearer one."""
    p = df.NetPrimitives()
    p.num_layers = 2
    p.polys = [(0, [square(0, 0, 20 * MM)]), (1, [square(0, 0, 20 * MM)])]
    p.vias = [(10 * MM, 10 * MM, [0, 1])]
    p.bbox = (0, 0, 20 * MM, 20 * MM)
    fields = []
    for li, x in ((0, 2), (1, 18)):
        p.seed_discs = [(li, x * MM, 10 * MM, 0.5 * MM)]
        fields.append(df.solve(p, 100_000))
    f1, f2 = fields
    d = viewer.prepare_delta(f1, f2)
    heat = viewer.Heatmap(d.layers, f1.origin, f1.pitch_nm, f1.max_distance_nm, d.scale)
    iso = [viewer.Heatmap(viewer.prepare_layers(f), f.origin, f.pitch_nm, f.max_distance_nm)
           for f in fields]
    view = (200 / (20.0 * MM), (10.0 * MM, 10.0 * MM))   # 10 px per mm
    r = viewer.render_heatmap(heat, [viewer.CopperShapes(p, li) for li in (0, 1)],
                              view, 200, 200, block=2)
    for py in (31, 51, 151):
        row = np.array([r.values_at(px, py)[1] for px in range(11, 190, 2)])
        assert np.abs(np.diff(row)).max() < 0.5 * MM, (py, np.abs(np.diff(row)).max())
        for px in range(11, 190, 20):
            x, y = (px + 1 - 100) / 10.0 * MM + 10 * MM, (py + 1 - 100) / 10.0 * MM + 10 * MM
            want = iso[0].sample(x, y) - iso[1].sample(x, y)
            got = r.values_at(px, py)[1]
            assert abs(got - want) < 0.2 * MM, (px, py, got / MM, want / MM)


def test_delta_on_copper_reached_from_one_object_only():
    """Copper only object 1 reaches counts as nearest to 1 (full colour), and
    the readout says the other is not connected."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(0, 0, 4 * MM)]), (0, [square(10 * MM, 0, 4 * MM)])]
    p.bbox = (0, 0, 14 * MM, 4 * MM)
    p.sources = [(0, 1 * MM, 1 * MM)]
    f1 = df.solve(p, 100_000)
    p.sources = [(0, 11 * MM, 1 * MM)]
    f2 = df.solve(p, 100_000)
    d = viewer.prepare_delta(f1, f2)
    assert d.scale == 1.0                                  # never both: no scale
    heat = viewer.Heatmap(d.layers, f1.origin, f1.pitch_nm, f1.max_distance_nm, d.scale)
    key, delta = heat.sample_delta(3 * MM, 2 * MM)        # object 1's square only
    assert delta == -viewer._ONE_SIDED and abs(key - 2.24 * MM) < 0.2 * MM, (key, delta)
    assert heat.sample_delta(12 * MM, 2 * MM)[1] == viewer._ONE_SIDED
    assert np.isnan(heat.sample_delta(7 * MM, 2 * MM)[0])  # between them: no copper
    assert viewer._delta_readout(2 * MM, -2.0, 1.0) == ["2.00 mm to 1", "Not connected to 2"]


class _FakeGeometry:
    def __init__(self, prims, delta):
        self.prims, self.delta = prims, delta
        self.disc_kinds = ["via"] * len(prims.discs)
        self.poly_kinds = []
        self.via_holes = []
        self.layer_names, self.layer_colors = ["F.Cu"], [(200, 52, 52)]
        self.net_name = "SIG"
        self.seed_count = 2
        self.seed_shapes = [([(0, 1 * MM, 10 * MM, 0.5 * MM)], []),
                            ([(0, 19 * MM, 10 * MM, 0.5 * MM)], [])]
        self.unfilled_zones = 0


def test_frame_switches_mode_and_shows_a_delta_result():
    frame = viewer.IsoplotFrame()
    try:
        modes = []
        frame.attach(type("S", (), {"set_mode": lambda self, d: modes.append(d)})())
        p, (f1, f2) = _trace_pair()
        frame._mode_delta.SetValue(True)
        frame._on_mode(None)
        assert modes == [True] and frame._delta
        assert frame._view._board_colour == viewer._DELTA_BOARD_COLOUR
        frame.set_channels(("R1 pad 1", None), "SIG")
        assert frame._channel_text[0].GetLabel() == "1: R1 pad 1"
        assert frame._channel_text[1].GetLabel() == "2: select a pad or via"
        # An isoplot result arriving after the switch is dropped.
        frame.show_result(_FakeGeometry(p, False), f1, viewer.prepare_layers(f1), True, 0.1)
        assert frame._field is None
        frame.show_result(_FakeGeometry(p, True), f1, viewer.prepare_delta(f1, f2), True, 0.1)
        heat = frame._view._heat
        assert heat is not None and heat.delta
        assert frame._delta_top.GetLabel().endswith("mm  (1 nearer)")
        seeds = frame._view._seeds
        assert [rgb for rgb, _ in seeds] == list(viewer._DELTA_SEED_RGB)
        frame.set_note("Delta mode compares two pads/vias; 3 are selected.")
        assert "3 are selected" in frame.GetStatusBar().GetStatusText()
        frame._mode_iso.SetValue(True)
        frame._on_mode(None)
        assert modes == [True, False] and frame._view._heat is None
        assert "3 are selected" not in frame.GetStatusBar().GetStatusText()
    finally:
        frame.Destroy()


if __name__ == "__main__":
    app = wx.App(False)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("OK: %d tests passed" % len(tests))
