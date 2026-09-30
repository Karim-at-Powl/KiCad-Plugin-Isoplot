"""Standalone smoke test for distance_field (no pcbnew needed).

Run from anywhere:  python tests/test_distance_field.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distance_field as df

MM = 1_000_000  # nm per mm


def build_L_net():
    """Layer 0: horizontal trace (0,0)->(20mm,0), then vertical up to (20,10).
    Via at (20,10) down to layer 1: short trace (20,10)->(25,10).
    Source at (0,0). Expected furthest point ~ (25,10): 20+10+5 = 35mm geodesic.
    """
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
    return p


def run(label, pitch_mm):
    p = build_L_net()
    f = df.solve(p, pitch_nm=int(pitch_mm * MM))
    maxmm = f.max_distance_nm / MM
    # furthest reachable cell on layer 1
    reached1 = sum(1 for v in f.dist[1] if v is not None)
    print("%-22s pitch=%.2fmm grid=%dx%d max=%.2fmm (expect ~35) L1cells=%d"
          % (label, pitch_mm, f.nx, f.ny, maxmm, reached1))
    return maxmm


def build_poly_net():
    """A single 10mm x 10mm filled square (polygon) on layer 0, source near one
    corner. Mimics a pad sitting on a continuous pour: copper is described *only*
    by a polygon, so this exercises the poly() rasterisation path that zones and
    rectangular pads rely on. Furthest point ~ opposite corner: ~14.1mm."""
    p = df.NetPrimitives()
    p.num_layers = 1
    s = 10 * MM
    p.polys = [(0, [(0, 0), (s, 0), (s, s), (0, s)])]
    p.sources = [(0, 1 * MM, 1 * MM)]
    p.bbox = (-1 * MM, -1 * MM, 11 * MM, 11 * MM)
    return p


def run_poly(label):
    p = build_poly_net()
    f = df.solve(p, pitch_nm=int(0.2 * MM))
    cells = sum(1 for v in f.dist[0] if v is not None)
    maxmm = f.max_distance_nm / MM
    print("%-22s copper_cells=%d max=%.2fmm (expect ~12.7)" % (label, cells, maxmm))
    return cells, maxmm


print("HAS_NUMPY =", df.HAS_NUMPY)
m1 = run("numpy path" if df.HAS_NUMPY else "pure path", 0.15)

# Polygon coverage on whichever path the environment provides (regression for
# zone/pour copper being dropped on the NumPy path).
pc1, pm1 = run_poly("numpy" if df.HAS_NUMPY else "pure")
assert pc1 > 1000, "polygon fill produced almost no copper: %d cells" % pc1
assert 11.0 < pm1 < 15.0, "polygon geodesic out of range: %s" % pm1

# Force pure-python path and re-run to confirm identical-ish results.
df.HAS_NUMPY = False
m2 = run("forced pure-python", 0.15)
pc2, pm2 = run_poly("forced pure-python")

assert 33.0 < m1 < 37.0, "geodesic distance out of expected range: %s" % m1
assert 33.0 < m2 < 37.0, "pure-python geodesic out of range: %s" % m2
assert pc2 > 1000, "pure-python polygon fill produced almost no copper: %d" % pc2
assert 11.0 < pm2 < 15.0, "pure-python polygon geodesic out of range: %s" % pm2
print("OK: both paths within tolerance")
