"""Core geometry-free computation for the Net Isoplot plugin.

This module intentionally has **no dependency on pcbnew**. It takes plain
numeric primitives (all coordinates in KiCad internal units = nanometres),
rasterises the copper of a single net onto a multi-layer grid, and computes the
geodesic (along-copper) distance from a set of source cells to every reachable
copper cell using a multi-source Dijkstra.

Keeping this module pcbnew-free means the heavy/important logic can be unit
tested with synthetic input using any plain Python interpreter.

NumPy is used *only* to accelerate rasterisation when it happens to be
importable; the algorithm and its results are identical without it. The
distance solve itself is pure Python and operates on a flat ``bytearray`` mask,
so it behaves the same regardless of whether NumPy is present.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

try:
    import numpy as _np
    HAS_NUMPY = True
except Exception:  # pragma: no cover - depends on environment
    _np = None
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Input / output containers
# ---------------------------------------------------------------------------

class NetPrimitives:
    """Plain description of one net's copper, ready for rasterisation.

    All coordinates and sizes are in nanometres (KiCad internal units).

    Attributes:
        num_layers: number of copper layers involved (indices 0..num_layers-1).
        bbox: (xmin, ymin, xmax, ymax) of all copper, in nm.
        segments: list of (layer, x0, y0, x1, y1, width).
        discs:    list of (layer, cx, cy, radius).
        polys:    list of (layer, [(x, y), ...]) filled polygon outlines.
        vias:     list of (cx, cy, [layer, ...]) connecting layers at a point.
        sources:  list of (layer, x, y) seed points (distance 0).
    """

    def __init__(self):
        self.num_layers = 0
        self.bbox = (0, 0, 0, 0)
        self.segments = []
        self.discs = []
        self.polys = []
        self.vias = []
        self.sources = []


class DistanceField:
    """Result of a solve.

    Attributes:
        nx, ny: grid dimensions.
        num_layers: number of layers.
        pitch_nm: grid pitch in nm.
        origin: (xmin, ymin) world coordinate of cell (0, 0) centre, in nm.
        dist: list (len num_layers) of flat float lists (len nx*ny); value is the
              geodesic distance in nm, or ``None`` for unreachable / non-copper.
        max_distance_nm: largest reachable distance (0.0 if only the source).
    """

    def __init__(self, nx, ny, num_layers, pitch_nm, origin):
        self.nx = nx
        self.ny = ny
        self.num_layers = num_layers
        self.pitch_nm = pitch_nm
        self.origin = origin
        self.dist = [[None] * (nx * ny) for _ in range(num_layers)]
        self.max_distance_nm = 0.0

    def cell_center_world(self, ix, iy):
        ox, oy = self.origin
        return (ox + ix * self.pitch_nm, oy + iy * self.pitch_nm)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _grid_dims(bbox, pitch_nm, margin_nm):
    xmin, ymin, xmax, ymax = bbox
    xmin -= margin_nm
    ymin -= margin_nm
    xmax += margin_nm
    ymax += margin_nm
    nx = max(1, int(math.ceil((xmax - xmin) / pitch_nm)) + 1)
    ny = max(1, int(math.ceil((ymax - ymin) / pitch_nm)) + 1)
    return nx, ny, (xmin, ymin)


# ---------------------------------------------------------------------------
# Rasterisation (pure-Python, with optional NumPy fast paths)
# ---------------------------------------------------------------------------

class _Rasteriser:
    def __init__(self, nx, ny, origin, pitch_nm):
        self.nx = nx
        self.ny = ny
        self.ox, self.oy = origin
        self.pitch = float(pitch_nm)
        self.half = self.pitch * 0.5
        if HAS_NUMPY:
            self._np_mask = _np.zeros((ny, nx), dtype=bool)
        self._mask = bytearray(nx * ny)  # used when numpy absent

    # -- world <-> cell -----------------------------------------------------
    def cell_of(self, x, y):
        ix = int(round((x - self.ox) / self.pitch))
        iy = int(round((y - self.oy) / self.pitch))
        return ix, iy

    def _set(self, ix, iy):
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self._mask[iy * self.nx + ix] = 1

    def cx(self, ix):
        return self.ox + ix * self.pitch

    def cy(self, iy):
        return self.oy + iy * self.pitch

    # -- primitives ---------------------------------------------------------
    def disc(self, cx, cy, r):
        rr = r + self.half
        ix0, iy0 = self.cell_of(cx - rr, cy - rr)
        ix1, iy1 = self.cell_of(cx + rr, cy + rr)
        rr2 = rr * rr
        if HAS_NUMPY:
            self._disc_np(cx, cy, rr2, ix0, iy0, ix1, iy1)
            return
        for iy in range(max(0, iy0), min(self.ny, iy1 + 1)):
            wy = self.cy(iy) - cy
            for ix in range(max(0, ix0), min(self.nx, ix1 + 1)):
                wx = self.cx(ix) - cx
                if wx * wx + wy * wy <= rr2:
                    self._mask[iy * self.nx + ix] = 1

    def segment(self, x0, y0, x1, y1, width):
        rr = width * 0.5 + self.half
        minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
        miny, maxy = (y0, y1) if y0 <= y1 else (y1, y0)
        ix0, iy0 = self.cell_of(minx - rr, miny - rr)
        ix1, iy1 = self.cell_of(maxx + rr, maxy + rr)
        rr2 = rr * rr
        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        if HAS_NUMPY:
            self._segment_np(x0, y0, dx, dy, seg_len2, rr2, ix0, iy0, ix1, iy1)
            return
        for iy in range(max(0, iy0), min(self.ny, iy1 + 1)):
            py = self.cy(iy)
            for ix in range(max(0, ix0), min(self.nx, ix1 + 1)):
                px = self.cx(ix)
                if _pt_seg_dist2(px, py, x0, y0, dx, dy, seg_len2) <= rr2:
                    self._mask[iy * self.nx + ix] = 1

    def poly(self, pts):
        """Scanline-fill a closed polygon (even-odd)."""
        if len(pts) < 3:
            return
        ys = [p[1] for p in pts]
        iy0 = max(0, self.cell_of(0, min(ys))[1])
        iy1 = min(self.ny - 1, self.cell_of(0, max(ys))[1])
        n = len(pts)
        for iy in range(iy0, iy1 + 1):
            scan_y = self.cy(iy)
            xs = []
            for i in range(n):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % n]
                if (ay <= scan_y < by) or (by <= scan_y < ay):
                    t = (scan_y - ay) / (by - ay)
                    xs.append(ax + t * (bx - ax))
            if not xs:
                continue
            xs.sort()
            row = iy * self.nx
            for k in range(0, len(xs) - 1, 2):
                ixa = max(0, self.cell_of(xs[k], 0)[0])
                ixb = min(self.nx - 1, self.cell_of(xs[k + 1], 0)[0])
                if ixb < ixa:
                    continue
                if HAS_NUMPY:
                    # Must target the same buffer finalize() returns, otherwise
                    # every polygon fill (zones/pours, rectangular pads) is lost.
                    self._np_mask[iy, ixa:ixb + 1] = True
                else:
                    for ix in range(ixa, ixb + 1):
                        self._mask[row + ix] = 1

    # -- numpy fast paths ---------------------------------------------------
    def _disc_np(self, cx, cy, rr2, ix0, iy0, ix1, iy1):
        ix0 = max(0, ix0); iy0 = max(0, iy0)
        ix1 = min(self.nx - 1, ix1); iy1 = min(self.ny - 1, iy1)
        if ix1 < ix0 or iy1 < iy0:
            return
        xs = self.ox + _np.arange(ix0, ix1 + 1) * self.pitch - cx
        ys = self.oy + _np.arange(iy0, iy1 + 1) * self.pitch - cy
        d2 = ys[:, None] ** 2 + xs[None, :] ** 2
        self._np_mask[iy0:iy1 + 1, ix0:ix1 + 1] |= (d2 <= rr2)

    def _segment_np(self, x0, y0, dx, dy, seg_len2, rr2, ix0, iy0, ix1, iy1):
        ix0 = max(0, ix0); iy0 = max(0, iy0)
        ix1 = min(self.nx - 1, ix1); iy1 = min(self.ny - 1, iy1)
        if ix1 < ix0 or iy1 < iy0:
            return
        xs = self.ox + _np.arange(ix0, ix1 + 1) * self.pitch
        ys = self.oy + _np.arange(iy0, iy1 + 1) * self.pitch
        px = xs[None, :] - x0
        py = ys[:, None] - y0
        if seg_len2 <= 0:
            d2 = px ** 2 + py ** 2
        else:
            t = (px * dx + py * dy) / seg_len2
            t = _np.clip(t, 0.0, 1.0)
            cxp = px - t * dx
            cyp = py - t * dy
            d2 = cxp ** 2 + cyp ** 2
        self._np_mask[iy0:iy1 + 1, ix0:ix1 + 1] |= (d2 <= rr2)

    def finalize(self):
        """Return the layer mask as a flat bytearray of 0/1."""
        if HAS_NUMPY:
            return bytearray(self._np_mask.reshape(-1).astype('uint8').tobytes())
        return self._mask


def _pt_seg_dist2(px, py, x0, y0, dx, dy, seg_len2):
    if seg_len2 <= 0:
        ex = px - x0
        ey = py - y0
        return ex * ex + ey * ey
    t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = px - (x0 + t * dx)
    ey = py - (y0 + t * dy)
    return ex * ex + ey * ey


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def build_masks(prims, pitch_nm, margin_nm, progress=None):
    """Rasterise every layer's copper. Returns (nx, ny, origin, masks)."""
    nx, ny, origin = _grid_dims(prims.bbox, pitch_nm, margin_nm)
    masks = []
    for layer in range(prims.num_layers):
        r = _Rasteriser(nx, ny, origin, pitch_nm)
        for (li, x, y, rad) in prims.discs:
            if li == layer:
                r.disc(x, y, rad)
        for (li, x0, y0, x1, y1, w) in prims.segments:
            if li == layer:
                r.segment(x0, y0, x1, y1, w)
        for (li, pts) in prims.polys:
            if li == layer:
                r.poly(pts)
        masks.append(r.finalize())
        if progress:
            progress(layer + 1, prims.num_layers)
    return nx, ny, origin, masks


def solve(prims, pitch_nm, margin_nm=0, progress=None):
    """Rasterise + multi-source Dijkstra. Returns a DistanceField."""
    nx, ny, origin = _grid_dims(prims.bbox, pitch_nm, margin_nm)
    _, _, _, masks = build_masks(prims, pitch_nm, margin_nm, progress)

    field = DistanceField(nx, ny, prims.num_layers, pitch_nm, origin)
    ox, oy = origin
    layer_stride = nx * ny

    def cell_idx(x, y):
        ix = int(round((x - ox) / pitch_nm))
        iy = int(round((y - oy) / pitch_nm))
        if 0 <= ix < nx and 0 <= iy < ny:
            return iy * nx + ix
        return None

    # Vertical (via) connections: cell offset -> list of layers connected there.
    via_links = {}  # cell_offset -> set(layers)
    for (x, y, layers) in prims.vias:
        off = cell_idx(x, y)
        if off is None:
            continue
        s = via_links.setdefault(off, set())
        s.update(layers)
        for li in layers:
            if li < prims.num_layers:
                masks[li][off] = 1  # ensure the via lands on copper

    ortho = float(pitch_nm)
    diag = float(pitch_nm) * math.sqrt(2.0)
    # neighbour offsets (dx, dy, cost)
    nb = [(1, 0, ortho), (-1, 0, ortho), (0, 1, ortho), (0, -1, ortho),
          (1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag)]

    INF = float('inf')
    dist = [[INF] * layer_stride for _ in range(prims.num_layers)]

    heap = []
    for (li, x, y) in prims.sources:
        off = cell_idx(x, y)
        if off is None or li >= prims.num_layers:
            continue
        masks[li][off] = 1  # force the seed to be copper
        if dist[li][off] > 0.0:
            dist[li][off] = 0.0
            heapq.heappush(heap, (0.0, li, off))

    max_d = 0.0
    while heap:
        d, li, off = heapq.heappop(heap)
        if d > dist[li][off]:
            continue
        if d > max_d:
            max_d = d
        ix = off % nx
        iy = off // nx
        mask = masks[li]
        # in-plane neighbours
        for dx, dy, cost in nb:
            jx = ix + dx
            jy = iy + dy
            if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                continue
            noff = jy * nx + jx
            if not mask[noff]:
                continue
            nd = d + cost
            if nd < dist[li][noff]:
                dist[li][noff] = nd
                heapq.heappush(heap, (nd, li, noff))
        # vertical (via) neighbours: same cell, other connected layers
        linked = via_links.get(off)
        if linked:
            for lj in linked:
                if lj == li or lj >= prims.num_layers:
                    continue
                if not masks[lj][off]:
                    continue
                if d < dist[lj][off]:
                    dist[lj][off] = d
                    heapq.heappush(heap, (d, lj, off))

    for li in range(prims.num_layers):
        dl = dist[li]
        out = field.dist[li]
        for i in range(layer_stride):
            v = dl[i]
            if v != INF:
                out[i] = v
    field.max_distance_nm = max_d
    return field
