"""Core geometry-free computation for the Net Isoplot plugin.

This module has **no dependency on KiCad**. It takes plain numeric primitives
(all coordinates in KiCad internal units = nanometres), rasterises the copper
of a single net onto a multi-layer grid, and computes the geodesic
(along-copper) distance from the seed copper to every reachable copper cell
using a multi-source Dijkstra.

Keeping this module KiCad-free means the heavy logic can be unit tested with
synthetic input, and it can run on a worker thread without touching the API.

Rasterisation is vectorised with NumPy. The Dijkstra itself is a tight
pure-Python loop over flat lists (NumPy has no fast priority queue); it checks a
``cancel`` callback periodically so a stale solve can be abandoned when the
board or the selection changes.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


class Cancelled(Exception):
    """Raised by :func:`solve` when its ``cancel`` callback returns True."""


# ---------------------------------------------------------------------------
# Input / output containers
# ---------------------------------------------------------------------------

class NetPrimitives:
    """Plain description of one net's copper, ready for rasterisation.

    All coordinates and sizes are in nanometres (KiCad internal units). A
    *ring* is a list of (x, y) vertices of a closed polygon; a list of rings is
    filled with the even-odd rule, so an outline followed by its holes works.

    Attributes:
        num_layers: number of copper layers involved (indices 0..num_layers-1).
        bbox: (xmin, ymin, xmax, ymax) of all copper, in nm.
        segments: list of (layer, x0, y0, x1, y1, width).
        discs:    list of (layer, cx, cy, radius).
        polys:    list of (layer, [ring, ...]) filled polygons.
        vias:     list of (cx, cy, [layer, ...]) connecting layers at a point.
        sources:  list of (layer, x, y) seed points (distance 0).
        seed_discs: list of (layer, cx, cy, radius) seed copper (distance 0).
        seed_polys: list of (layer, [ring, ...]) seed copper (distance 0).
    """

    def __init__(self):
        self.num_layers = 0
        self.bbox = (0, 0, 0, 0)
        self.segments = []
        self.discs = []
        self.polys = []
        self.vias = []
        self.sources = []
        self.seed_discs = []
        self.seed_polys = []


class DistanceField:
    """Result of a solve.

    Attributes:
        nx, ny: grid dimensions.
        num_layers: number of layers.
        pitch_nm: grid pitch in nm.
        origin: (x, y) world coordinate of the centre of cell (0, 0), in nm.
        dist: float64 array of shape (num_layers, ny, nx); geodesic distance in
              nm, ``inf`` for unreachable / non-copper cells.
        max_distance_nm: largest reachable distance (0.0 if only the seed).
    """

    def __init__(self, nx, ny, num_layers, pitch_nm, origin, dist):
        self.nx = nx
        self.ny = ny
        self.num_layers = num_layers
        self.pitch_nm = pitch_nm
        self.origin = origin
        self.dist = dist
        finite = dist[np.isfinite(dist)]
        self.max_distance_nm = float(finite.max()) if finite.size else 0.0

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


def pitch_for_budget(bbox, num_layers, max_cells, min_pitch_nm):
    """Smallest pitch (>= ``min_pitch_nm``, rounded up to 10 um) for which the
    grid over ``bbox`` has at most ``max_cells`` cells summed over all layers."""
    xmin, ymin, xmax, ymax = bbox
    area = max(1.0, float(xmax - xmin)) * max(1.0, float(ymax - ymin))
    pitch = max(float(min_pitch_nm),
                math.sqrt(area * max(1, num_layers) / float(max_cells)))
    step = 10_000
    pitch = int(math.ceil(pitch / step)) * step
    # Margins and rounding add a few rows/columns; nudge until it fits.
    while True:
        nx, ny, _ = _grid_dims(bbox, pitch, 2 * pitch)
        if nx * ny * max(1, num_layers) <= max_cells:
            return pitch
        pitch += step


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

class _Rasteriser:
    """Marks the cells of one layer whose centre lies on copper.

    Discs and segments are grown by half a cell and polygon boundaries mark
    every cell they pass through, so the raster is slightly conservative
    (oversized) and thin copper stays connected on a coarse grid.
    """

    def __init__(self, nx, ny, origin, pitch_nm):
        self.nx = nx
        self.ny = ny
        self.ox, self.oy = origin
        self.pitch = float(pitch_nm)
        self.half = self.pitch * 0.5
        self.mask = np.zeros((ny, nx), dtype=bool)

    def _window(self, xmin, ymin, xmax, ymax):
        """Clipped inclusive cell window covering a world-space box."""
        ix0 = max(0, int(math.floor((xmin - self.ox) / self.pitch)))
        iy0 = max(0, int(math.floor((ymin - self.oy) / self.pitch)))
        ix1 = min(self.nx - 1, int(math.ceil((xmax - self.ox) / self.pitch)))
        iy1 = min(self.ny - 1, int(math.ceil((ymax - self.oy) / self.pitch)))
        return ix0, iy0, ix1, iy1

    def disc(self, cx, cy, r):
        rr = r + self.half
        ix0, iy0, ix1, iy1 = self._window(cx - rr, cy - rr, cx + rr, cy + rr)
        if ix1 < ix0 or iy1 < iy0:
            return
        xs = self.ox + np.arange(ix0, ix1 + 1) * self.pitch - cx
        ys = self.oy + np.arange(iy0, iy1 + 1) * self.pitch - cy
        d2 = ys[:, None] ** 2 + xs[None, :] ** 2
        self.mask[iy0:iy1 + 1, ix0:ix1 + 1] |= d2 <= rr * rr

    def segment(self, x0, y0, x1, y1, width):
        rr = width * 0.5 + self.half
        ix0, iy0, ix1, iy1 = self._window(min(x0, x1) - rr, min(y0, y1) - rr,
                                          max(x0, x1) + rr, max(y0, y1) + rr)
        if ix1 < ix0 or iy1 < iy0:
            return
        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        px = (self.ox + np.arange(ix0, ix1 + 1) * self.pitch)[None, :] - x0
        py = (self.oy + np.arange(iy0, iy1 + 1) * self.pitch)[:, None] - y0
        if seg_len2 <= 0:
            d2 = px ** 2 + py ** 2
        else:
            t = np.clip((px * dx + py * dy) / seg_len2, 0.0, 1.0)
            d2 = (px - t * dx) ** 2 + (py - t * dy) ** 2
        self.mask[iy0:iy1 + 1, ix0:ix1 + 1] |= d2 <= rr * rr

    def rings(self, rings):
        """Fill a polygon given as rings (even-odd), plus its boundary cells."""
        rings = [np.asarray(r, dtype=np.float64) for r in rings if len(r) >= 3]
        if not rings:
            return
        a = np.concatenate(rings)
        b = np.concatenate([np.roll(r, -1, axis=0) for r in rings])
        ix0, iy0, ix1, iy1 = self._window(a[:, 0].min(), a[:, 1].min(),
                                          a[:, 0].max(), a[:, 1].max())
        if ix1 < ix0 or iy1 < iy0:
            return
        w = ix1 - ix0 + 1
        h = iy1 - iy0 + 1
        wox = self.ox + ix0 * self.pitch
        woy = self.oy + iy0 * self.pitch

        # Interior: for every edge, find the rows whose scanline (through the
        # cell centres) it crosses, and count crossings left of each cell.
        ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        keep = ay != by
        ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
        ylo = np.minimum(ay, by)
        yhi = np.maximum(ay, by)
        r0 = np.clip(np.ceil((ylo - woy) / self.pitch), 0, h).astype(np.int64)
        r1 = np.clip(np.ceil((yhi - woy) / self.pitch), 0, h).astype(np.int64)
        counts = np.maximum(r1 - r0, 0)
        total = int(counts.sum())
        if total:
            e = np.repeat(np.arange(counts.size), counts)
            starts = np.cumsum(counts) - counts
            row = r0[e] + (np.arange(total) - starts[e])
            scan_y = woy + row * self.pitch
            t = (scan_y - ay[e]) / (by[e] - ay[e])
            x = ax[e] + t * (bx[e] - ax[e])
            col = np.clip(np.ceil((x - wox) / self.pitch), 0, w).astype(np.int64)
            hits = np.bincount(row * (w + 1) + col, minlength=h * (w + 1))
            parity = np.cumsum(hits.reshape(h, w + 1), axis=1)[:, :w] & 1
            self.mask[iy0:iy1 + 1, ix0:ix1 + 1] |= parity.astype(bool)

        # Boundary: sample every edge at half-cell spacing and mark the cells
        # containing the samples.
        seg_len = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        n = np.maximum(1, np.ceil(seg_len / self.half).astype(np.int64)) + 1
        e = np.repeat(np.arange(n.size), n)
        starts = np.cumsum(n) - n
        t = (np.arange(int(n.sum())) - starts[e]) / (n[e] - 1)
        sx = a[e, 0] + t * (b[e, 0] - a[e, 0])
        sy = a[e, 1] + t * (b[e, 1] - a[e, 1])
        cx = np.rint((sx - self.ox) / self.pitch).astype(np.int64)
        cy = np.rint((sy - self.oy) / self.pitch).astype(np.int64)
        ok = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        self.mask[cy[ok], cx[ok]] = True

    def cell_of(self, x, y):
        ix = int(round((x - self.ox) / self.pitch))
        iy = int(round((y - self.oy) / self.pitch))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None


def _rasterise_layer(nx, ny, origin, pitch_nm, discs, segments, polys):
    r = _Rasteriser(nx, ny, origin, pitch_nm)
    for (x, y, rad) in discs:
        r.disc(x, y, rad)
    for (x0, y0, x1, y1, w) in segments:
        r.segment(x0, y0, x1, y1, w)
    for rings in polys:
        r.rings(rings)
    return r.mask


def _by_layer(items, num_layers):
    out = [[] for _ in range(num_layers)]
    for item in items:
        if 0 <= item[0] < num_layers:
            out[item[0]].append(item[1:] if len(item) > 2 else item[1])
    return out


def build_masks(prims, pitch_nm, margin_nm):
    """Rasterise every layer's copper and seed copper.

    Returns (nx, ny, origin, masks, seed_masks); both masks are bool arrays of
    shape (num_layers, ny, nx).
    """
    nx, ny, origin = _grid_dims(prims.bbox, pitch_nm, margin_nm)
    L = prims.num_layers
    discs = _by_layer(prims.discs, L)
    segments = _by_layer(prims.segments, L)
    polys = _by_layer(prims.polys, L)
    seed_discs = _by_layer(prims.seed_discs, L)
    seed_polys = _by_layer(prims.seed_polys, L)
    masks = np.zeros((L, ny, nx), dtype=bool)
    seeds = np.zeros((L, ny, nx), dtype=bool)
    for li in range(L):
        masks[li] = _rasterise_layer(nx, ny, origin, pitch_nm,
                                     discs[li], segments[li], polys[li])
        if seed_discs[li] or seed_polys[li]:
            seeds[li] = _rasterise_layer(nx, ny, origin, pitch_nm,
                                         seed_discs[li], (), seed_polys[li])
    return nx, ny, origin, masks, seeds


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve(prims, pitch_nm, margin_nm=None, cancel=None):
    """Rasterise + multi-source Dijkstra. Returns a DistanceField.

    ``cancel`` is an optional zero-argument callable polled during the solve;
    when it returns True the solve stops by raising :class:`Cancelled`.
    """
    if margin_nm is None:
        margin_nm = 2 * pitch_nm
    nx, ny, origin, masks, seed_masks = build_masks(prims, pitch_nm, margin_nm)
    L = prims.num_layers
    ox, oy = origin

    def cell_of(x, y):
        ix = int(round((x - ox) / pitch_nm))
        iy = int(round((y - oy) / pitch_nm))
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None

    # Seeds are always copper; point sources and vias force their own cell.
    masks |= seed_masks
    for (li, x, y) in prims.sources:
        c = cell_of(x, y)
        if c is not None and 0 <= li < L:
            masks[li, c[1], c[0]] = True
            seed_masks[li, c[1], c[0]] = True
    via_cells = []  # (ix, iy, layers)
    for (x, y, layers) in prims.vias:
        c = cell_of(x, y)
        layers = sorted({li for li in layers if 0 <= li < L})
        if c is None or len(layers) < 2:
            continue
        for li in layers:
            masks[li, c[1], c[0]] = True
        via_cells.append((c[0], c[1], layers))

    # Pad the grid by one empty cell on every side so the in-plane neighbour
    # step never needs a bounds check (and never wraps into another layer).
    nxp, nyp = nx + 2, ny + 2
    stride = nxp * nyp
    padded = np.zeros((L, nyp, nxp), dtype=bool)
    padded[:, 1:-1, 1:-1] = masks
    mask = padded.reshape(-1).tobytes()

    def gidx(li, ix, iy):
        return li * stride + (iy + 1) * nxp + (ix + 1)

    links = {}  # global cell -> other-layer cells joined by a via/PTH
    for (ix, iy, layers) in via_cells:
        cells = [gidx(li, ix, iy) for li in layers]
        for g in cells:
            links.setdefault(g, set()).update(h for h in cells if h != g)

    ortho = float(pitch_nm)
    diag = float(pitch_nm) * math.sqrt(2.0)
    steps = ((1, ortho), (-1, ortho), (nxp, ortho), (-nxp, ortho),
             (nxp + 1, diag), (nxp - 1, diag), (-nxp + 1, diag), (-nxp - 1, diag))

    INF = math.inf
    dist = [INF] * (L * stride)
    li_s, iy_s, ix_s = np.nonzero(seed_masks)
    heap = []
    for li, iy, ix in zip(li_s.tolist(), iy_s.tolist(), ix_s.tolist()):
        g = gidx(li, ix, iy)
        dist[g] = 0.0
        heap.append((0.0, g))
    heapq.heapify(heap)

    pop = heapq.heappop
    push = heapq.heappush
    get_links = links.get
    popped = 0
    while heap:
        d, g = pop(heap)
        if d > dist[g]:
            continue
        popped += 1
        if cancel is not None and not popped & 0x3FFF and cancel():
            raise Cancelled()
        for step, cost in steps:
            h = g + step
            if mask[h]:
                nd = d + cost
                if nd < dist[h]:
                    dist[h] = nd
                    push(heap, (nd, h))
        joined = get_links(g)
        if joined:
            for h in joined:
                if d < dist[h]:
                    dist[h] = d
                    push(heap, (d, h))

    arr = np.array(dist, dtype=np.float64).reshape(L, nyp, nxp)[:, 1:-1, 1:-1]
    return DistanceField(nx, ny, L, pitch_nm, origin, np.ascontiguousarray(arr))
