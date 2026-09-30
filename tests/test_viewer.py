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


if __name__ == "__main__":
    app = wx.App(False)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("OK: %d tests passed" % len(tests))
