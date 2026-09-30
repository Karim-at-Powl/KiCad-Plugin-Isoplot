"""Standalone tests for distance_field (no KiCad needed).

Run from anywhere:  python tests/test_distance_field.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distance_field as df

MM = 1_000_000  # nm per mm


def square(x0, y0, s):
    return [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]


def test_L_net_across_via():
    """Layer 0: trace (0,0)->(20,0)->(20,10); via at (20,10) to layer 1, trace
    (20,10)->(25,10). Source at (0,0): furthest point ~35 mm along copper."""
    p = df.NetPrimitives()
    p.num_layers = 2
    w = 0.25 * MM
    p.segments = [
        (0, 0, 0, 20 * MM, 0, w),
        (0, 20 * MM, 0, 20 * MM, 10 * MM, w),
        (1, 20 * MM, 10 * MM, 25 * MM, 10 * MM, w),
    ]
    p.vias = [(20 * MM, 10 * MM, [0, 1])]
    p.sources = [(0, 0, 0)]
    p.bbox = (-1 * MM, -1 * MM, 26 * MM, 11 * MM)
    f = df.solve(p, pitch_nm=int(0.15 * MM))
    assert f.dist.shape == (2, f.ny, f.nx)
    assert 33.0 < f.max_distance_nm / MM < 37.0, f.max_distance_nm / MM
    assert np.isfinite(f.dist[1]).sum() > 50, "via did not bridge to layer 1"


def test_square_polygon():
    """10 mm filled square, source near one corner: furthest ~12.7 mm."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(0, 0, 10 * MM)])]
    p.sources = [(0, 1 * MM, 1 * MM)]
    p.bbox = (0, 0, 10 * MM, 10 * MM)
    f = df.solve(p, pitch_nm=int(0.2 * MM))
    cells = int(np.isfinite(f.dist[0]).sum())
    assert 2500 <= cells <= 2800, cells  # 50x50 interior + boundary ring
    assert 11.0 < f.max_distance_nm / MM < 15.0, f.max_distance_nm / MM


def test_polygon_hole_forces_detour():
    """Square with a large square hole: going around the hole is longer than
    the straight line, and hole cells are not copper."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(0, 0, 10 * MM), square(2 * MM, 2 * MM, 6 * MM)])]
    p.sources = [(0, 5 * MM, 1 * MM)]
    p.bbox = (0, 0, 10 * MM, 10 * MM)
    f = df.solve(p, pitch_nm=int(0.2 * MM))
    ix = int(round((5 * MM - f.origin[0]) / f.pitch_nm))
    iy_hole = int(round((5 * MM - f.origin[1]) / f.pitch_nm))
    iy_far = int(round((9 * MM - f.origin[1]) / f.pitch_nm))
    assert not np.isfinite(f.dist[0, iy_hole, ix]), "hole was filled"
    # Straight line would be 8 mm; around the hole it's > 11 mm.
    assert f.dist[0, iy_far, ix] / MM > 11.0, f.dist[0, iy_far, ix] / MM


def test_seed_polygon_measures_from_pad_edge():
    """A 4 mm seed pad at the start of a trace: distance counts from the pad
    edge, not its centre."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(-2 * MM, -2 * MM, 4 * MM)])]
    p.segments = [(0, 0, 0, 20 * MM, 0, int(0.3 * MM))]
    p.seed_polys = [(0, [square(-2 * MM, -2 * MM, 4 * MM)])]
    p.bbox = (-2 * MM, -2 * MM, 20 * MM, 2 * MM)
    f = df.solve(p, pitch_nm=int(0.1 * MM))
    assert 17.5 < f.max_distance_nm / MM < 18.5, f.max_distance_nm / MM


def test_cancel():
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(0, 0, 30 * MM)])]
    p.sources = [(0, 0, 0)]
    p.bbox = (0, 0, 30 * MM, 30 * MM)
    try:
        df.solve(p, pitch_nm=int(0.1 * MM), cancel=lambda: True)
    except df.Cancelled:
        return
    raise AssertionError("solve ignored cancel")


def test_pitch_for_budget():
    bbox = (0, 0, 100 * MM, 50 * MM)
    pitch = df.pitch_for_budget(bbox, 4, 1_000_000, int(0.05 * MM))
    nx, ny, _ = df._grid_dims(bbox, pitch, 2 * pitch)
    assert nx * ny * 4 <= 1_000_000
    assert pitch % 10_000 == 0
    assert df.pitch_for_budget((0, 0, MM, MM), 1, 1_000_000, 50_000) == 50_000


def benchmark():
    """Rough solve speed on a fully poured 60 x 40 mm, 2-layer net."""
    p = df.NetPrimitives()
    p.num_layers = 2
    p.polys = [(0, [square(0, 0, 60 * MM)]), (1, [square(0, 0, 60 * MM)])]
    p.vias = [(30 * MM, 20 * MM, [0, 1])]
    p.sources = [(0, 0, 0)]
    p.bbox = (0, 0, 60 * MM, 40 * MM)
    pitch = df.pitch_for_budget(p.bbox, 2, 1_000_000, 50_000)
    t = time.perf_counter()
    f = df.solve(p, pitch)
    dt = time.perf_counter() - t
    cells = int(np.isfinite(f.dist).sum())
    print("benchmark: %d copper cells @ %.3f mm in %.2f s (%.2f us/cell)"
          % (cells, pitch / MM, dt, 1e6 * dt / max(1, cells)))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    benchmark()
    print("OK: %d tests passed" % len(tests))
