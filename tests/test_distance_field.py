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


def _via_net(num_layers, via_layers, layer_z):
    """A 10 mm trace on layer 0 to a via, then a 5 mm trace on each other
    layer of the via."""
    p = df.NetPrimitives()
    p.num_layers = num_layers
    w = int(0.25 * MM)
    p.segments = [(0, 0, 0, 10 * MM, 0, w)]
    p.segments += [(li, 10 * MM, 0, 15 * MM, 0, w) for li in via_layers if li]
    p.vias = [(10 * MM, 0, list(via_layers))]
    p.sources = [(0, 0, 0)]
    p.bbox = (-1 * MM, -1 * MM, 16 * MM, 1 * MM)
    p.layer_z = layer_z
    return p


def test_via_length_from_stackup():
    """Through the via, each layer is farther by its height difference from
    the layer the path came in on."""
    pitch = int(0.1 * MM)
    flat = df.solve(_via_net(3, [0, 1, 2], None), pitch)
    z = [0.0, 0.2 * MM, 1.6 * MM]
    tall = df.solve(_via_net(3, [0, 1, 2], z), pitch)
    assert np.array_equal(flat.dist[0], tall.dist[0])   # entry layer untouched
    for li in (1, 2):
        fin = np.isfinite(flat.dist[li])
        assert np.allclose(tall.dist[li][fin] - flat.dist[li][fin], z[li]), li
    # The heights decide the order, not the layer numbers (here 2 sits
    # between 0 and 1), and a skipped layer is only passed through.
    z = [0.0, 1.6 * MM, 0.2 * MM]
    f = df.solve(_via_net(3, [0, 1, 2], z), pitch)
    fin = np.isfinite(flat.dist[1])
    assert np.allclose(f.dist[1][fin] - flat.dist[1][fin], 1.6 * MM)
    f = df.solve(_via_net(4, [0, 3], [0.0, 0.5 * MM, 1.0 * MM, 1.5 * MM]), pitch)
    fin = np.isfinite(flat.dist[1])
    assert np.allclose(f.dist[3][fin] - flat.dist[1][fin], 1.5 * MM)
    assert not np.isfinite(f.dist[1]).any() and not np.isfinite(f.dist[2]).any()


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


def test_search_strategies_agree():
    """Heap-only, wavefront-only and the default mix give identical fields on
    a net with a pour, a hole, thin traces and a via."""
    p = df.NetPrimitives()
    p.num_layers = 2
    p.polys = [(0, [square(0, 0, 12 * MM), square(3 * MM, 3 * MM, 5 * MM)])]
    p.segments = [(0, 12 * MM, 6 * MM, 30 * MM, 6 * MM, int(0.2 * MM)),
                  (1, 30 * MM, 6 * MM, 30 * MM, 20 * MM, int(0.3 * MM)),
                  (1, 30 * MM, 20 * MM, 5 * MM, 20 * MM, int(0.3 * MM))]
    p.vias = [(30 * MM, 6 * MM, [0, 1])]
    p.seed_discs = [(0, 1 * MM, 1 * MM, int(0.4 * MM))]
    p.bbox = (0, 0, 31 * MM, 21 * MM)
    pitch = int(0.1 * MM)
    for reach, p.layer_z in ((2, None), (df.MAX_REACH, None), (2, [0.0, 1.6 * MM])):
        ref = df.solve(p, pitch, reach=reach, to_heap=10 ** 9, to_vector=10 ** 9).dist
        for th, tv in ((0, 0), (df.TO_HEAP, df.TO_VECTOR), (4, 8)):
            got = df.solve(p, pitch, reach=reach, to_heap=th, to_vector=tv).dist
            assert np.array_equal(np.isfinite(ref), np.isfinite(got)), (reach, th, tv)
            fin = np.isfinite(ref)
            assert np.allclose(ref[fin], got[fin], rtol=0, atol=1e-6), (reach, th, tv)
        assert np.isfinite(ref[1]).sum() > 100, "via did not bridge to layer 1"


def test_move_sets():
    """8/16/32/48 moves, neighbours first; a diagonal crosses no other cell,
    a knight's move the two beside its line."""
    assert [len(df.move_set(r)) for r in range(1, df.MAX_REACH + 1)] == [8, 16, 32, 48]
    for r in range(1, df.MAX_REACH + 1):
        moves = df.move_set(r)
        assert len({(dx, dy) for dx, dy, _ in moves}) == len(moves)
        assert {(dx, dy) for dx, dy, _ in moves[:8]} == \
            {(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - {(0, 0)}
    assert df._crossed(1, 1) == ()
    assert set(df._crossed(2, 1)) == {(1, 0), (1, 1)}
    assert set(df._crossed(-1, 2)) == {(0, 1), (-1, 1)}
    assert len(df._crossed(4, 3)) == 6


# Worst-case overestimate of a grid path per reach (1 / cos of half the
# largest angle between move directions), plus slack for the grid.
_MAX_ERROR = {2: 0.03, 4: 0.009}


def test_accuracy_against_exact_distances():
    """The grid path overestimates by at most ~2.7 % with 16 moves and ~0.75 %
    with 48: straight traces at awkward angles, and an open pour around a
    round pad."""
    pitch = int(0.05 * MM)
    for reach, max_error in _MAX_ERROR.items():
        for deg in (10, 22.5, 30):
            a = math.radians(deg)
            x1, y1 = 30 * MM * math.cos(a), 30 * MM * math.sin(a)
            p = df.NetPrimitives()
            p.num_layers = 1
            p.segments = [(0, 0, 0, x1, y1, int(0.2 * MM))]
            p.seed_discs = [(0, 0, 0, int(0.3 * MM))]
            p.bbox = (-MM, -MM, x1 + MM, y1 + MM)
            f = df.solve(p, pitch, reach=reach)
            ix = int(round((x1 - f.origin[0]) / pitch))
            iy = int(round((y1 - f.origin[1]) / pitch))
            exact = 30 * MM - 0.3 * MM
            assert abs(f.dist[0, iy, ix] / exact - 1) < max_error, \
                (reach, deg, f.dist[0, iy, ix] / exact)

        p = df.NetPrimitives()
        p.num_layers = 1
        p.polys = [(0, [square(0, 0, 20 * MM)])]
        p.seed_discs = [(0, 10 * MM, 10 * MM, int(0.3 * MM))]
        p.bbox = (0, 0, 20 * MM, 20 * MM)
        f = df.solve(p, pitch, reach=reach)
        xs = f.origin[0] + np.arange(f.nx) * pitch
        ys = f.origin[1] + np.arange(f.ny) * pitch
        X, Y = np.meshgrid(xs, ys)
        exact = np.hypot(X - 10 * MM, Y - 10 * MM) - 0.3 * MM
        far = np.isfinite(f.dist[0]) & (exact > 3 * MM) & (np.abs(X - 10 * MM) < 9.5 * MM) \
            & (np.abs(Y - 10 * MM) < 9.5 * MM)
        rel = f.dist[0][far] / exact[far] - 1
        assert rel.max() < max_error and rel.min() > -0.01, (reach, rel.min(), rel.max())


def test_longer_moves_only_shorten():
    """Each move set contains the smaller ones, so distances can only drop
    as the reach grows, and the same cells are reached."""
    p = df.NetPrimitives()
    p.num_layers = 1
    p.polys = [(0, [square(0, 0, 10 * MM), square(3 * MM, 3 * MM, 4 * MM)])]
    p.segments = [(0, 10 * MM, 5 * MM, 25 * MM, 12 * MM, int(0.15 * MM))]
    p.seed_discs = [(0, 1 * MM, 1 * MM, int(0.3 * MM))]
    p.bbox = (0, 0, 25 * MM, 12 * MM)
    fields = [df.solve(p, int(0.1 * MM), reach=r).dist for r in range(1, df.MAX_REACH + 1)]
    for a, b in zip(fields, fields[1:]):
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        fin = np.isfinite(a)
        assert (b[fin] <= a[fin] + 1e-6).all()


def test_long_moves_do_not_jump_gaps():
    """Two parallel traces one cell apart, joined only at one end: the far end
    of the second trace must be reached the long way round, whatever the
    reach of the moves."""
    pitch = int(0.1 * MM)
    p = df.NetPrimitives()
    p.num_layers = 1
    w = int(0.05 * MM)   # rasterises to single-cell-wide traces
    p.segments = [(0, 0, 0, 10 * MM, 0, w),
                  (0, 10 * MM, 0, 10 * MM, 2 * pitch, w),
                  (0, 10 * MM, 2 * pitch, 0, 2 * pitch, w)]
    p.sources = [(0, 0, 0)]
    p.bbox = (-MM, -MM, 11 * MM, MM)
    for reach in range(1, df.MAX_REACH + 1):
        f = df.solve(p, pitch, reach=reach)
        ix = int(round((0 - f.origin[0]) / pitch))
        iy = int(round((2 * pitch - f.origin[1]) / pitch))
        assert f.dist[0, iy, ix] / MM > 19.0, (reach, f.dist[0, iy, ix] / MM)


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
    for reach in (2, df.MAX_REACH):
        t = time.perf_counter()
        f = df.solve(p, pitch, reach=reach)
        dt = time.perf_counter() - t
        cells = int(np.isfinite(f.dist).sum())
        print("benchmark: %d copper cells @ %.3f mm, %d moves in %.2f s (%.2f us/cell)"
              % (cells, pitch / MM, f.num_moves, dt, 1e6 * dt / max(1, cells)))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    benchmark()
    print("OK: %d tests passed" % len(tests))
