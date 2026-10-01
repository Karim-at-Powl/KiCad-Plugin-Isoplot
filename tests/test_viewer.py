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


if __name__ == "__main__":
    app = wx.App(False)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("OK: %d tests passed" % len(tests))
