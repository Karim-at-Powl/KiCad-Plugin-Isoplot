"""The Net Isoplot window: a heatmap of the distance field that stays open next
to KiCad and redraws whenever the live session delivers a new result.

Neither KiCad API can put pixels on the board canvas, so the heatmap lives in
its own window, with a per-layer toggle list, switches to hide the net's vias
and pads, a legend and a cursor readout of the along-copper distance and board
position. In delta mode it shows instead which of two pads/vias each point is
nearer to, and by how much.

The heatmap is rendered at screen resolution rather than as an enlarged grid:
the copper's outline comes from the exact shapes read from KiCad (drawn
anti-aliased), and only the colour inside comes from the distance grid,
interpolated between cells. Edges stay sharp at any zoom, whatever the pitch.
"""

from __future__ import annotations

import numpy as np
import wx

# Background the transparent heatmap is composited over (light grey so the
# black seed pad and the copper colours both read well).
_BG_COLOUR = (215, 215, 215)       # outside the board
_BOARD_COLOUR = (245, 245, 245)    # inside the board outline
_EDGE_COLOUR = (70, 70, 70)        # Edge.Cuts
_PAD_COLOUR = (85, 85, 85)         # pad outlines
_HOLE_FILL = (184, 115, 51)        # via holes: copper ...
_HOLE_RIM = (240, 200, 60)         # ... with a golden rim
_OPACITY = 0.6
_VIEW_W = 760                      # initial board view size (DIP)
_VIEW_H = 620
_SETTLE_MS = 120                   # re-render this long after the last zoom/pan
_STARTING = "Starting..."          # shown until the board outline arrives
_NO_HOLES = np.zeros((0, 5))
# The kinds of copper (kicad_source.NetGeometry) each "Show" switch hides.
_SWITCHED = {"via": ("via",), "pad": ("pad", "tht")}
MM = 1e6


# ---------------------------------------------------------------------------
# Colour map  (near = red / hot ... far = blue / cool)
# ---------------------------------------------------------------------------

_STOPS = [
    (0.00, (255, 0, 0)),      # seed / closest = red (hot)
    (0.25, (255, 255, 0)),
    (0.50, (0, 255, 0)),
    (0.75, (0, 255, 255)),
    (1.00, (0, 0, 255)),      # furthest = blue (cool)
]


# Delta mode: nearer to object 1 ... equal (white) ... nearer to object 2.
# Drawn more opaque, over a darker board, so white copper still shows.
CHANNEL_RGB = ((214, 39, 40), (31, 119, 180))
_DELTA_STOPS = [
    (0.00, CHANNEL_RGB[1]),
    (0.50, (255, 255, 255)),
    (1.00, CHANNEL_RGB[0]),
]
_DELTA_OPACITY = 0.85
_DELTA_BOARD_COLOUR = (222, 222, 222)
# The objects' own copper, marked in a dark shade of their colour.
_DELTA_SEED_RGB = tuple(tuple(int(0.55 * c) for c in rgb) for rgb in CHANNEL_RGB)
# Beyond the scale: copper reached from only one of the two objects.
_ONE_SIDED = 2.0


def colormap(frac, stops=_STOPS):
    if frac <= 0:
        return stops[0][1]
    if frac >= 1:
        return stops[-1][1]
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f0 <= frac <= f1:
            t = (frac - f0) / (f1 - f0)
            return tuple(int(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


def _lut32(stops):
    """The colour map as 256 packed RGBA words (bytes R, G, B, A in memory),
    opaque, plus a transparent entry 256 for "no data"."""
    lut = np.array([colormap(i / 255.0, stops) for i in range(256)], dtype=np.uint32)
    out = np.zeros(257, dtype=np.uint32)
    out[:256] = lut[:, 0] | lut[:, 1] << 8 | lut[:, 2] << 16 | np.uint32(0xFF) << 24
    return out


def _alpha32(opacity):
    """An AND-mask per copper coverage that scales alpha down to ``opacity``."""
    return np.array([0x00FFFFFF | round(c * opacity) << 24 for c in range(256)],
                    dtype=np.uint32)


_LUT32, _ALPHA32 = _lut32(_STOPS), _alpha32(_OPACITY)
_LUT32_DELTA, _ALPHA32_DELTA = _lut32(_DELTA_STOPS), _alpha32(_DELTA_OPACITY)


# ---------------------------------------------------------------------------
# What the view draws (plain NumPy, board coordinates in nm)
# ---------------------------------------------------------------------------

def prepare_layers(field):
    """A DistanceField's layers as (value, weight) float32 array pairs with a
    one-cell border, for interpolation that skips cells without data. Pure
    NumPy, so it runs on the solver thread rather than the UI thread.

    Empty cells next to copper take their smallest neighbour's value, so a
    sample on the exact copper edge (which can fall just outside the
    rasterised copper) still finds data.
    """
    out = []
    for dist in field.dist:
        v = np.pad(dist.astype(np.float32), 1, constant_values=np.inf)
        v[np.isinf(v)] = np.nan
        # 3 x 3 minimum ignoring NaN, done separably (rows, then columns).
        near = v.copy()
        np.fmin(near[:, 1:], v[:, :-1], out=near[:, 1:])
        np.fmin(near[:, :-1], v[:, 1:], out=near[:, :-1])
        rows = near.copy()
        np.fmin(near[1:], rows[:-1], out=near[1:])
        np.fmin(near[:-1], rows[1:], out=near[:-1])
        empty = np.isnan(v)
        v[empty] = near[empty]          # only empty cells take the neighbours' value
        v[0] = v[-1] = v[:, 0] = v[:, -1] = np.nan   # keep the border empty
        ok = ~np.isnan(v)
        v[~ok] = 0.0
        out.append((v, ok.astype(np.float32)))
    return out


class DeltaLayers:
    """Delta mode's layers (see :func:`prepare_delta`).

    Attributes:
        layers: per layer (value 1, weight 1, value 2, weight 2): the layer's
                distances from object 1 and from object 2, each as
                :func:`prepare_layers` makes them.
        scale:  the largest |delta| where both objects reach (at most their
                distance from each other), in nm.

    The two are kept apart, not combined per layer: at a point, each
    object's distance is the nearest over the layers there (what its own
    isoplot shows), and the delta is the difference of those. Taking the
    delta of whichever layer is nearer to either object would jump wherever
    another layer becomes the nearer one.
    """

    def __init__(self, layers, scale):
        self.layers = layers
        self.scale = scale


def prepare_delta(field, other):
    """Two DistanceFields on one grid (from object 1 and from object 2) as
    :class:`DeltaLayers`. Pure NumPy, for the solver thread."""
    both = np.isfinite(field.dist) & np.isfinite(other.dist)
    scale = float(np.abs(field.dist[both] - other.dist[both]).max()) if both.any() else 0.0
    layers = [a + b for a, b in zip(prepare_layers(field), prepare_layers(other))]
    return DeltaLayers(layers, scale or 1.0)


def _combine(d1, d2, scale):
    """(distance to the nearer object, delta) from the distances to object 1
    and 2 (arrays, NaN = none). Copper reached from only one of them gets
    delta -/+ _ONE_SIDED * scale."""
    dist = np.fmin(d1, d2)
    with np.errstate(invalid="ignore"):
        delta = d1 - d2
    beyond = np.float32(_ONE_SIDED * scale)
    only1, only2 = np.isnan(d2) & ~np.isnan(d1), np.isnan(d1) & ~np.isnan(d2)
    delta[only1] = -beyond
    delta[only2] = beyond
    return dist, delta


class Heatmap:
    """Per-layer distances of the shown layers, ready to be sampled.

    In delta mode (``delta_scale`` given) each layer carries the distances
    from both objects (see :class:`DeltaLayers`)."""

    def __init__(self, layers, origin, pitch, max_distance, delta_scale=None):
        """``layers``: the shown layers from :func:`prepare_layers`, or from
        ``DeltaLayers.layers`` with its ``scale`` as ``delta_scale``."""
        self._layers = layers
        self.delta_scale = delta_scale
        ny, nx = layers[0][0].shape
        ny, nx = ny - 2, nx - 2
        self._pitch = float(pitch)
        self._x0 = origin[0] - self._pitch   # centre of bordered cell (0, 0)
        self._y0 = origin[1] - self._pitch
        self._n = (nx + 2, ny + 2)
        self.max_distance = float(max_distance)
        half = self._pitch / 2.0
        self.box = (origin[0] - half, origin[1] - half,
                    origin[0] + (nx - 0.5) * self._pitch, origin[1] + (ny - 0.5) * self._pitch)

    @property
    def num_layers(self):
        return len(self._layers)

    def axis(self, coords, along_x):
        """Cell index and fraction for board coordinates along one axis."""
        o, n = (self._x0, self._n[0]) if along_x else (self._y0, self._n[1])
        f = (np.asarray(coords, dtype=np.float64) - o) / self._pitch
        i = np.clip(np.floor(f), 0, n - 2).astype(np.intp)
        return i, np.clip(f - i, 0.0, 1.0).astype(np.float32)

    @property
    def delta(self):
        return self.delta_scale is not None

    def layer(self, li, iy, ty, ix, tx):
        """Bilinear distances of one layer on the grid of rows (iy, ty) x
        columns (ix, tx); NaN where there is no data. Cells without data are
        left out and the other weights renormalised, so values don't bleed in
        from outside the copper. Separable, so it costs a few array passes."""
        return self.channels(li, iy, ty, ix, tx)[0]

    def channels(self, li, iy, ty, ix, tx):
        """The distances of one layer as :meth:`layer`, one array per
        object: [from the seed], or in delta mode [from 1, from 2]."""
        c0 = int(ix.min())
        jx = ix - c0
        c1 = int(ix.max()) + 2
        arrays = self._layers[li]
        out = []
        for value, weight in zip(arrays[0::2], arrays[1::2]):
            den = self._across(weight[:, c0:c1], iy, ty, jx, tx)
            num = self._across(value[:, c0:c1], iy, ty, jx, tx)
            with np.errstate(invalid="ignore", divide="ignore"):
                num /= den
            num[den < 1e-4] = np.nan
            out.append(num)
        return out

    @staticmethod
    def _across(a, iy, ty, jx, tx):
        """Separable bilinear lookup of ``a`` (rows first, then columns)."""
        top = a[iy]
        r = a[iy + 1]
        r -= top
        r *= ty[:, None]
        r += top
        left = r[:, jx]
        right = r[:, jx + 1]
        right -= left
        right *= tx
        right += left
        return right

    def sample(self, x, y):
        """Nearest distance over all layers at one board point (nm), NaN if none."""
        return self.sample_delta(x, y)[0]

    def sample_delta(self, x, y):
        """(nearest distance over all layers, its delta) at one board point
        (nm); NaN if none, delta None unless in delta mode."""
        ix, tx = self.axis([x], True)
        iy, ty = self.axis([y], False)
        nearest = None              # per object, the nearest over the layers
        for li in range(self.num_layers):
            vals = self.channels(li, iy, ty, ix, tx)
            nearest = vals if nearest is None else [np.fmin(a, b) for a, b in zip(nearest, vals)]
        if nearest is None:
            return np.nan, None
        if not self.delta:
            return float(nearest[0][0, 0]), None
        dist, delta = _combine(*nearest, self.delta_scale)
        return float(dist[0, 0]), (float(delta[0, 0]) if np.isfinite(dist[0, 0]) else None)


def _delta_readout(value, delta, scale):
    """Cursor readout lines in delta mode: the distance to the nearer object
    and how much farther the other one is."""
    if not np.isfinite(value) or delta is None:
        return ["Not connected to 1 or 2"]
    near = 1 if delta < 0 else 2
    if abs(delta) > 0.5 * (1.0 + _ONE_SIDED) * scale:
        return ["%.2f mm to %d" % (value / MM, near), "Not connected to %d" % (3 - near)]
    if abs(delta) < 0.005 * MM:
        return ["%.2f mm to both" % (value / MM), "Equal distance"]
    return ["%d is nearer: %.2f mm" % (near, value / MM),
            "%d is %.2f mm farther" % (3 - near, abs(delta) / MM)]


def _bridged(rings):
    """One point list for a polygon with holes. Each ring is reached from the
    first vertex and left the same way, so even-odd filling cancels the
    bridges."""
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None
    anchor = rings[0][0]
    pts = []
    for ring in rings:
        pts.append(anchor)
        pts.extend(ring)
        pts.append(ring[0])
    return np.asarray(pts, dtype=np.float64)


class CopperShapes:
    """The copper of one layer as arrays (nm), for drawing."""

    def __init__(self, prims, layer, disc_kinds=None, poly_kinds=None, hidden=()):
        """``disc_kinds`` / ``poly_kinds``: what each of ``prims.discs`` /
        ``prims.polys`` came from ("via", "pad", "tht", "zone"); kinds in ``hidden``
        are not drawn. Hidden copper still counts in the distances."""
        def shown(kinds, i):
            return kinds is None or kinds[i] not in hidden

        self.segments = np.array([s[1:] for s in prims.segments if s[0] == layer],
                                 dtype=np.float64).reshape(-1, 5)   # x0 y0 x1 y1 width
        self.discs = np.array([d[1:] for i, d in enumerate(prims.discs)
                               if d[0] == layer and shown(disc_kinds, i)],
                              dtype=np.float64).reshape(-1, 3)      # x y r
        polys = (_bridged(rings) for i, (li, rings) in enumerate(prims.polys)
                 if li == layer and shown(poly_kinds, i))
        self.polys = [p for p in polys if p is not None]
        self.poly_boxes = np.array([(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
                                    for p in self.polys], dtype=np.float64).reshape(-1, 4)


class Marks:
    """The net's SMD pad outlines and via drills on the shown layers, drawn
    on top of the heatmap so they stand out. Board nm. Through-hole pads get
    no outline: their drill (drawn with the board, see _BoardView.set_holes)
    marks them, and in a pour the pad outline stands for nothing physical."""

    def __init__(self, prims, poly_kinds, via_holes, layers, hidden=()):
        """``poly_kinds`` / ``via_holes``: as in kicad_source.NetGeometry;
        ``layers``: the shown layer indices; kinds in ``hidden`` are left out."""
        shown = set(layers)
        self.pad_rings = []
        if "pad" not in hidden:
            seen = set()    # a pad on several layers has the same outline on each
            for (li, rings), kind in zip(prims.polys, poly_kinds):
                if kind != "pad" or li not in shown:
                    continue
                for ring in rings:
                    if len(ring) < 3:
                        continue
                    key = (len(ring), tuple(ring[0]), tuple(ring[len(ring) // 2]))
                    if key in seen:
                        continue
                    seen.add(key)
                    self.pad_rings.append(np.asarray(list(ring) + [ring[0]], dtype=np.float64))
        self.ring_boxes = np.array([(r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())
                                    for r in self.pad_rings], dtype=np.float64).reshape(-1, 4)
        self.via_holes = np.array([h[:5] for h in via_holes
                                   if "via" not in hidden and shown & set(h[5])],
                                  dtype=np.float64).reshape(-1, 5)   # x y width height angle


def _oblong(cx, cy, a, b, angle_deg, n=11):   # odd n: a point on each tip
    """Outline (screen points) of a slot with half-sizes ``a`` along its own x
    and ``b`` along its own y, turned by ``angle_deg`` the KiCad way
    (anticlockwise on screen, y pointing down)."""
    r = min(a, b)
    t = np.linspace(-np.pi / 2, np.pi / 2, n)
    cap = np.column_stack((abs(a - b) + r * np.cos(t), r * np.sin(t)))
    local = np.vstack((cap, -cap, cap[:1]))     # closed, so the rim has no gap
    if b > a:
        local = local[:, ::-1]          # the long side is the slot's own y
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    x, y = local[:, 0], local[:, 1]
    return np.column_stack((cx + x * c + y * s, cy - x * s + y * c))


def _screen_poly(pts, s, ox, oy):
    """Board points -> screen points, dropping steps of under a quarter pixel."""
    q = np.round((pts * s + (ox, oy)) * 4.0) / 4.0
    if len(q) > 3:
        keep = np.empty(len(q), dtype=bool)
        keep[0] = True
        np.any(q[1:] != q[:-1], axis=1, out=keep[1:])
        q = q[keep]
    return q


def _copper_coverage(shapes, view, bmp, buf, min_px):
    """Anti-aliased coverage (uint8, h x w) of the copper in the viewport,
    drawn on the scratch bitmap ``bmp`` and read through ``buf`` (h, w, 3)."""
    s, (cx, cy) = view
    h, w = buf.shape[:2]
    ox, oy = w / 2.0 - cx * s, h / 2.0 - cy * s
    vx0, vy0, vx1, vy1 = -ox / s, -oy / s, (w - ox) / s, (h - oy) / s   # viewport (nm)

    dc = wx.MemoryDC(bmp)
    dc.SetBackground(wx.BLACK_BRUSH)
    dc.Clear()
    gc = wx.GraphicsContext.Create(dc)
    gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
    gc.SetPen(wx.TRANSPARENT_PEN)
    gc.SetBrush(wx.WHITE_BRUSH)

    b = shapes.poly_boxes
    if len(b):
        seen = (b[:, 0] <= vx1) & (b[:, 2] >= vx0) & (b[:, 1] <= vy1) & (b[:, 3] >= vy0)
        for i in np.flatnonzero(seen):
            q = _screen_poly(shapes.polys[i], s, ox, oy)
            if len(q) >= 3:
                gc.DrawLines(q, wx.ODDEVEN_RULE)

    d = shapes.discs
    if len(d):
        d = d[(d[:, 0] + d[:, 2] >= vx0) & (d[:, 0] - d[:, 2] <= vx1)
              & (d[:, 1] + d[:, 2] >= vy0) & (d[:, 1] - d[:, 2] <= vy1)]
        for x, y, r in zip(d[:, 0] * s + ox, d[:, 1] * s + oy, np.maximum(d[:, 2] * s, min_px / 2.0)):
            gc.DrawEllipse(x - r, y - r, 2 * r, 2 * r)

    g = shapes.segments
    if len(g):
        hw = g[:, 4] / 2.0
        g = g[(np.minimum(g[:, 0], g[:, 2]) - hw <= vx1) & (np.maximum(g[:, 0], g[:, 2]) + hw >= vx0)
              & (np.minimum(g[:, 1], g[:, 3]) - hw <= vy1) & (np.maximum(g[:, 1], g[:, 3]) + hw >= vy0)]
        for width in np.unique(g[:, 4]):
            rows = g[g[:, 4] == width]
            pen = wx.GraphicsPenInfo(wx.WHITE, max(width * s, min_px)).Cap(wx.CAP_ROUND)
            gc.SetPen(gc.CreatePen(pen))
            path = gc.CreatePath()
            for x0, y0, x1, y1 in zip(rows[:, 0] * s + ox, rows[:, 1] * s + oy,
                                      rows[:, 2] * s + ox, rows[:, 3] * s + oy):
                path.MoveToPoint(x0, y0)
                path.AddLineToPoint(x1, y1)
            gc.StrokePath(path)

    del gc
    dc.SelectObject(wx.NullBitmap)
    bmp.CopyToBuffer(buf, wx.BitmapBufferFormat_RGB)
    return buf[:, :, 0].copy()


class Render:
    """One rendered heatmap for a viewport."""

    def __init__(self, bitmap, coverage, dist, block, delta=None):
        self.bitmap = bitmap        # RGBA wx.Bitmap, device pixels, or None if empty
        self.coverage = coverage    # (h, w) uint8, copper of all shown layers
        self.dist = dist            # (ceil(h/block), ceil(w/block)) float32, NaN = none
        self.delta = delta          # the same for the delta (delta mode), else None
        self.block = block
        self.x = self.y = 0         # position in the viewport

    def value_at(self, px, py):
        """Distance shown at a viewport pixel: None off copper, inf on
        unreached copper."""
        return self.values_at(px, py)[0]

    def values_at(self, px, py):
        """(distance, delta) shown at a viewport pixel: distance as in
        :meth:`value_at`; delta None unless in delta mode with a distance."""
        px, py = px - self.x, py - self.y
        cov = self.coverage
        if not (0 <= py < cov.shape[0] and 0 <= px < cov.shape[1]) or not cov[py, px]:
            return None, None
        at = py // self.block, px // self.block
        v = self.dist[at]
        if not np.isfinite(v):
            return np.inf, None
        return float(v), (None if self.delta is None else float(self.delta[at]))


def render_heatmap(heat, layer_shapes, view, w, h, min_px=1.0, block=1):
    """Render the heatmap for a w x h viewport (``layer_shapes`` matches the
    heatmap's layers).

    Only the part of the viewport the net's grid covers is rendered, so a
    small net costs little. The copper outline is drawn at device resolution.
    Colour is computed per ``block`` x ``block`` pixels (below a DIP on
    high-DPI screens, so it can't be seen, and it saves most of the work).
    Where layers overlap, the one nearer to the seed shows, decided per pixel
    from the exact copper, so layer boundaries are as sharp as the outline.
    """
    s, (cx, cy) = view
    bx0, by0, bx1, by1 = heat.box
    x0 = max(0, int(np.floor((bx0 - cx) * s + w / 2.0)) - 2)
    y0 = max(0, int(np.floor((by0 - cy) * s + h / 2.0)) - 2)
    x1 = min(w, int(np.ceil((bx1 - cx) * s + w / 2.0)) + 2)
    y1 = min(h, int(np.ceil((by1 - cy) * s + h / 2.0)) + 2)
    if x1 <= x0 or y1 <= y0:
        return Render(None, np.zeros((0, 0), np.uint8), np.zeros((0, 0), np.float32), block)
    rw, rh = x1 - x0, y1 - y0
    sub = (s, ((x0 + rw / 2.0 - w / 2.0) / s + cx, (y0 + rh / 2.0 - h / 2.0) / s + cy))
    r = _render_rect(heat, layer_shapes, sub, rw, rh, min_px, block)
    r.x, r.y = x0, y0
    return r


def _render_rect(heat, layer_shapes, view, w, h, min_px, block):
    s, (cx, cy) = view
    bh, bw = -(-h // block), -(-w // block)
    py = np.arange(bh) * block + block // 2   # block centres (the last may be
    px = np.arange(bw) * block + block // 2   # past the edge; it is cut off)
    edge = (bh * block - h, bw * block - w)
    pad = ((0, bh - int((py < h).sum())), (0, bw - int((px < w).sum())))
    ix, tx = heat.axis((px + 0.5 - w / 2.0) / s + cx, True)
    iy, ty = heat.axis((py + 0.5 - h / 2.0) / s + cy, False)

    canvas = wx.Bitmap(w, h, 24)
    buf = np.empty((h, w, 3), dtype=np.uint8)
    coverage = None
    # Per object (the seed, or the two in delta mode): the nearest distance
    # over the layers covering each block centre, and over all layers.
    n = 2 if heat.delta else 1
    shown = [np.full((bh, bw), np.nan, dtype=np.float32) for _ in range(n)]
    anywhere = [np.full((bh, bw), np.nan, dtype=np.float32) for _ in range(n)]
    for li, shapes in enumerate(layer_shapes):
        cov = _copper_coverage(shapes, view, canvas, buf, min_px)
        if coverage is None:
            coverage = cov
        else:
            np.maximum(coverage, cov, out=coverage)
        at_centres = np.pad(cov[block // 2::block, block // 2::block], pad)
        for k, d in enumerate(heat.channels(li, iy, ty, ix, tx)):
            np.fmin(anywhere[k], d, out=anywhere[k])
            d[at_centres == 0] = np.nan
            np.fmin(shown[k], d, out=shown[k])
    # Anti-aliased edge pixels whose block centre is off copper take the
    # nearest layer's value regardless.
    off = np.logical_and.reduce([np.isnan(a) for a in shown])
    nearest = [np.where(off, a, b) for a, b in zip(anywhere, shown)]
    if heat.delta:
        dist, delta = _combine(*nearest, heat.delta_scale)
    else:
        dist, delta = nearest[0], None

    if heat.delta:
        # Nearer to 1 (delta < 0) is the top of the scale: object 1's colour.
        q = delta * np.float32(-127.5 / heat.delta_scale) + np.float32(127.5)
        lut, alpha = _LUT32_DELTA, _ALPHA32_DELTA
    else:
        q = dist * np.float32(255.0 / (heat.max_distance or 1.0))
        lut, alpha = _LUT32, _ALPHA32
    np.clip(q, 0, 255, out=q)
    q[np.isnan(dist)] = 256   # copper the seed can't reach stays uncoloured
    small = lut[q.astype(np.intp)]
    rgba = np.empty((bh * block, bw * block), dtype=np.uint32)
    for i in range(block):          # strided copies beat a broadcast here
        for j in range(block):
            rgba[i::block, j::block] = small
    if edge[0] or edge[1]:
        rgba[h:, :] = 0
        rgba[:, w:] = 0
    visible = rgba[:h, :w]
    np.bitwise_and(visible, alpha[coverage], out=visible)
    bitmap = wx.Bitmap.FromBufferRGBA(bw * block, bh * block, rgba.view(np.uint8))
    return Render(bitmap, coverage, dist, block, delta)



# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class _GradientBar(wx.Panel):
    """Colour scale (top = the end of ``stops``: far = blue in isoplot mode,
    nearer to object 1 in delta mode) with a frame.

    Built as a vertical stack of solid-colour child panels rather than by
    owner-drawing. On wxMSW the BG_STYLE_PAINT + AutoBufferedPaintDC paint path
    rendered the bar blank/white, while a plain ``wx.Panel`` with
    ``SetBackgroundColour`` draws reliably. The parent's black background shows
    through a 1 px border as the frame.
    """

    def __init__(self, parent, size, stops=_STOPS, bands=210):
        super().__init__(parent, size=size)
        self.SetBackgroundColour(wx.Colour(0, 0, 0))
        stack = wx.BoxSizer(wx.VERTICAL)
        for i in range(bands):
            band = wx.Panel(self)
            band.SetBackgroundColour(wx.Colour(*colormap(1.0 - i / float(bands - 1), stops)))
            stack.Add(band, 1, wx.EXPAND)
        frame = wx.BoxSizer(wx.VERTICAL)
        frame.Add(stack, 1, wx.EXPAND | wx.ALL, 1)
        self.SetSizer(frame)


class _BoardView(wx.Panel):
    """Owner-drawn board view in world coordinates (nm).

    The frame of reference is the board outline, so the view doesn't jump when
    switching between nets of different size. The mouse wheel zooms around the
    cursor, dragging pans, and a double-click fits the whole board again.

    The heatmap is rendered for exactly the visible area at device resolution.
    While zooming or panning the last render is stretched as a preview, and a
    fresh one is made once the view has been still for a moment. The composed
    picture is kept, so the cursor readout in the top-left corner can be
    redrawn on its own as the mouse moves.
    """

    _MARGIN = 12      # DIP around the fitted board
    _ZOOM_STEP = 1.25

    def __init__(self, parent, size):
        super().__init__(parent, size=size)
        self._heat = None               # Heatmap
        self._shapes = None             # [CopperShapes] per heatmap layer
        self._content = 0               # bumped whenever heat/shapes change
        self._seeds = ()                # (rgb, ("disc", x, y, r) / ("poly", points)), nm
        self._marks = None              # Marks: the net's pad outlines and via drills
        self._holes = _NO_HOLES         # the board's pad drills: x y width height angle
        self._rings = []                # closed board outline rings (nm)
        self._chains = []               # open outline pieces (nm)
        self._message = _STARTING
        self._zoom = 1.0
        self._center = None             # world point at the view centre; None = fitted
        self._drag = None
        self._cache = None              # (content, view, size, Render)
        self._settled = True
        self._settle = wx.Timer(self)
        self._composed = None           # the view as last drawn, without the readout
        self._dirty = True              # _composed needs redrawing
        self._mouse = None              # last mouse position over the view
        self._readout = None            # (lines, rect) of the cursor readout
        self._board_colour = _BOARD_COLOUR
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_TIMER, self._on_settle, self._settle)
        self.Bind(wx.EVT_SIZE, lambda e: (self._view_changed(), e.Skip()))
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, lambda e: setattr(self, "_drag", None))
        self.Bind(wx.EVT_LEFT_DCLICK, lambda e: self.fit())
        self.Bind(wx.EVT_MOTION, self._on_mouse_motion)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)

    # -- content ------------------------------------------------------------
    def set_outline(self, rings, chains):
        old = self._bounds()
        self._rings, self._chains = rings, chains
        if self._message == _STARTING and (rings or chains):
            self._message = None    # the board is up; the footer says what to do
        if self._bounds() != old:
            self.fit()
        self._invalidate()

    def set_holes(self, holes):
        """The board's pad drills, (x, y, width, height, angle) each; drawn
        like the outline's cut-outs, whatever net is shown."""
        self._holes = np.array(holes, dtype=np.float64).reshape(-1, 5)
        self._invalidate()

    def set_heatmap(self, heat, shapes, seeds, marks=None):
        had_frame = self._bounds() is not None
        self._heat, self._shapes, self._seeds, self._marks = heat, shapes, seeds, marks
        self._content += 1
        self._message = None
        if not had_frame:
            self.fit()
        self._invalidate()

    def set_board_colour(self, rgb):
        self._board_colour = rgb
        self._invalidate()

    def set_message(self, text):
        self._heat = self._shapes = None
        self._content += 1
        self._seeds = ()
        self._marks = None
        self._message = text
        self._invalidate()

    def fit(self):
        self._zoom = 1.0
        self._center = None
        self._view_changed()

    def _invalidate(self):
        self._dirty = True
        self.Refresh()

    def _view_changed(self):
        self._settled = False
        self._settle.StartOnce(_SETTLE_MS)
        self._invalidate()

    def _on_settle(self, _evt):
        self._settled = True
        self._invalidate()

    # -- view transform -----------------------------------------------------
    def _bounds(self):
        """World box of the fitted view: the board outline, else the heatmap."""
        pts = [p for ring in self._rings for p in ring] + [p for c in self._chains for p in c]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        if self._heat is not None:
            return self._heat.box
        return None

    def _view(self):
        """(scale px/nm, world centre), or None when there is nothing to show."""
        box = self._bounds()
        w, h = self.GetClientSize()
        m = self.FromDIP(self._MARGIN)
        if box is None or w < 2 * m + 2 or h < 2 * m + 2:
            return None
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        fit = min((w - 2 * m) / bw, (h - 2 * m) / bh)
        center = self._center or ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        return fit * self._zoom, center

    def to_screen(self, x, y, view):
        s, (cx, cy) = view
        w, h = self.GetClientSize()
        return ((x - cx) * s + w / 2.0, (y - cy) * s + h / 2.0)

    def to_world(self, px, py, view):
        s, (cx, cy) = view
        w, h = self.GetClientSize()
        return ((px - w / 2.0) / s + cx, (py - h / 2.0) / s + cy)

    def _screen_pts(self, pts, view):
        s, (cx, cy) = view
        w, h = self.GetClientSize()
        return (np.asarray(pts, dtype=np.float64) - (cx, cy)) * s + (w / 2.0, h / 2.0)

    # -- painting -----------------------------------------------------------
    def _on_paint(self, _evt):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetClientSize()
        comp = self._composed
        if self._dirty or comp is None or tuple(comp.GetSize()) != (w, h):
            comp = wx.Bitmap(max(1, w), max(1, h), 24)
            mdc = wx.MemoryDC(comp)
            self.draw(mdc)
            mdc.SelectObject(wx.NullBitmap)
            self._composed, self._dirty = comp, False
            if self._mouse is not None:   # the value under a still mouse may have changed
                self._readout = self._make_readout(*self._mouse)
        dc.DrawBitmap(comp, 0, 0)
        self._draw_readout(dc)

    def draw(self, dc):
        dc.SetBackground(wx.Brush(wx.Colour(*_BG_COLOUR)))
        dc.Clear()
        view = self._view()
        if view is not None:
            # Vector parts anti-aliased through a GraphicsContext; the heatmap
            # bitmap through the plain DC (GDI AlphaBlend is far faster than
            # GDI+ for large bitmaps). A context flushes when it is deleted.
            gc = self._gc(dc)
            self._draw_board_area(gc, view)
            del gc
            if self._heat is not None:
                self._draw_heatmap(dc, view)
            gc = self._gc(dc)
            self._draw_overlay(gc, view)
            del gc
        if self._message:
            dc.SetFont(self.GetFont())
            dc.SetTextForeground(wx.Colour(60, 60, 60))
            tw, th = dc.GetMultiLineTextExtent(self._message)   # a wx.Size
            w, h = self.GetClientSize()
            dc.DrawText(self._message, (w - tw) // 2, (h - th) // 2)

    # -- cursor readout (top-left corner) -----------------------------------
    def _make_readout(self, px, py):
        """(lines, rect) describing the board point under pixel (px, py)."""
        view = self._view()
        if view is None:
            return None
        x, y = self.to_world(px, py, view)
        lines = []
        if self._heat is not None:
            value, delta = self.value_at(px, py, view)
            if value is None:
                lines.append("No copper here")
            elif self._heat.delta:
                lines += _delta_readout(value, delta, self._heat.delta_scale)
            elif not np.isfinite(value):
                lines.append("Not connected to the selection")
            else:
                lines.append("%.2f mm from selection" % (value / MM))
        lines.append("x %.2f   y %.2f mm" % (x / MM, y / MM))
        dc = wx.ClientDC(self)
        pad = self.FromDIP(6)
        width = height = 0
        for i, text in enumerate(lines):
            dc.SetFont(self._readout_font(i, len(lines)))
            tw, th = dc.GetTextExtent(text)
            width, height = max(width, tw), height + th
        at = self.FromDIP(8)
        return lines, wx.Rect(at, at, width + 2 * pad, height + 2 * pad)

    def _readout_font(self, i, n):
        font = self.GetFont()
        return font.Bold() if i == 0 and n > 1 else font

    def _draw_readout(self, dc):
        if self._readout is None:
            return
        lines, rect = self._readout
        gc = self._gc(dc)
        gc.SetPen(wx.Pen(wx.Colour(150, 150, 150)))
        gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255, 225)))
        gc.DrawRoundedRectangle(rect.x + 0.5, rect.y + 0.5, rect.width - 1, rect.height - 1,
                                self.FromDIP(4))
        del gc
        pad = self.FromDIP(6)
        y = rect.y + pad
        dc.SetTextForeground(wx.Colour(30, 30, 30))
        for i, text in enumerate(lines):
            dc.SetFont(self._readout_font(i, len(lines)))
            dc.DrawText(text, rect.x + pad, y)
            y += dc.GetTextExtent(text)[1]

    def _set_readout(self, readout):
        old = self._readout
        self._readout = readout
        for r in (old, readout):
            if r is not None:
                self.RefreshRect(wx.Rect(r[1]).Inflate(2, 2), eraseBackground=False)

    def _on_leave(self, _evt):
        self._mouse = None
        self._set_readout(None)

    @staticmethod
    def _gc(dc):
        gc = wx.GraphicsContext.Create(dc)
        gc.SetAntialiasMode(wx.ANTIALIAS_DEFAULT)
        return gc

    def _draw_board_area(self, gc, view):
        if self._rings:  # even-odd, so cut-outs stay grey
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.SetBrush(wx.Brush(wx.Colour(*self._board_colour)))
            gc.DrawLines(self._screen_pts(_bridged(self._rings), view), wx.ODDEVEN_RULE)

    def _draw_overlay(self, gc, view):
        """Pad outlines, board outline, seed copper and drills, on top of the
        heatmap."""
        if self._marks is not None:
            self._draw_pad_outlines(gc, view)
        gc.SetPen(gc.CreatePen(wx.GraphicsPenInfo(wx.Colour(*_EDGE_COLOUR),
                                                  1.5 * self.GetDPIScaleFactor())))
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        for poly in self._rings + self._chains:
            gc.StrokeLines(self._screen_pts(poly, view))

        gc.SetPen(wx.TRANSPARENT_PEN)
        for rgb, shape in self._seeds:
            gc.SetBrush(wx.Brush(wx.Colour(*rgb)))
            if shape[0] == "disc":
                _, x, y, r = shape
                sx, sy = self.to_screen(x, y, view)
                r = max(self.FromDIP(2), r * view[0])
                gc.DrawEllipse(sx - r, sy - r, 2 * r, 2 * r)
            else:
                gc.DrawLines(self._screen_pts(shape[1], view), wx.ODDEVEN_RULE)
        self._draw_holes(gc, view)

    def _world_viewport(self, view):
        w, h = self.GetClientSize()
        return self.to_world(0, 0, view) + self.to_world(w, h, view)

    def _draw_pad_outlines(self, gc, view):
        b = self._marks.ring_boxes
        if not len(b):
            return
        x0, y0, x1, y1 = self._world_viewport(view)
        gc.SetPen(gc.CreatePen(wx.GraphicsPenInfo(wx.Colour(*_PAD_COLOUR),
                                                  1.5 * self.GetDPIScaleFactor())))
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        seen = (b[:, 0] <= x1) & (b[:, 2] >= x0) & (b[:, 1] <= y1) & (b[:, 3] >= y0)
        for i in np.flatnonzero(seen):
            gc.StrokeLines(self._screen_pts(self._marks.pad_rings[i], view))

    def _draw_holes(self, gc, view):
        """Each drill at its real size, never smaller than a few pixels so
        vias stay visible zoomed out. The net's vias are copper with a golden
        rim; the board's pad drills (any net, plated or not) are cut out like
        the board outline's cut-outs (the background grey, edged like
        Edge.Cuts)."""
        dpi = self.GetDPIScaleFactor()
        if self._marks is not None:
            self._draw_drills(gc, view, self._marks.via_holes, _HOLE_FILL, _HOLE_RIM,
                              lambda r: min(max(0.15 * r, dpi), 0.4 * r, 4 * dpi))
        self._draw_drills(gc, view, self._holes, _BG_COLOUR, _EDGE_COLOUR,
                          lambda r: min(1.5 * dpi, 0.4 * r))

    def _draw_drills(self, gc, view, holes, fill, rim_colour, rim_width):
        """``rim_width(radius)``: rim in px for a hole of ``radius`` px."""
        if not len(holes):
            return
        x0, y0, x1, y1 = self._world_viewport(view)
        r = holes[:, 2:4].max(axis=1) / 2.0
        holes = holes[(holes[:, 0] + r >= x0) & (holes[:, 0] - r <= x1)
                      & (holes[:, 1] + r >= y0) & (holes[:, 1] - r <= y1)]
        s = view[0]
        min_r = 1.5 * self.GetDPIScaleFactor()
        gc.SetBrush(wx.Brush(wx.Colour(*fill)))
        rim_colour = wx.Colour(*rim_colour)

        def rim_pen(radius):
            rim = rim_width(radius)
            gc.SetPen(gc.CreatePen(wx.GraphicsPenInfo(rim_colour, rim)))
            return rim

        pts = self._screen_pts(holes[:, :2], view)
        round_ = holes[:, 2] == holes[:, 3]
        for d in np.unique(holes[round_, 2]):     # one path per drill size
            radius = max(d / 2.0 * s, min_r)
            inner = radius - rim_pen(radius) / 2.0   # the rim ends at the drill edge
            path = gc.CreatePath()
            for x, y in pts[round_ & (holes[:, 2] == d)]:
                path.AddCircle(x, y, inner)
            gc.DrawPath(path)
        for (x, y), (_, _, w, h, angle) in zip(pts[~round_], holes[~round_]):   # slots
            a, b = max(w / 2.0 * s, min_r), max(h / 2.0 * s, min_r)
            inset = rim_pen(min(a, b)) / 2.0
            n = 2 * int(np.clip(min(a, b) / 2.0, 5, 40)) + 1    # smooth ends at any zoom
            gc.DrawLines(_oblong(x, y, a - inset, b - inset, angle, n))

    def _current_render(self, view):
        cache = self._cache
        if cache is not None and cache[:3] == (self._content, view, tuple(self.GetClientSize())):
            return cache[3]
        return None

    def _draw_heatmap(self, dc, view):
        size = tuple(self.GetClientSize())
        cache = self._cache
        render = self._current_render(view)
        if render is None and (self._settled or cache is None or cache[0] != self._content):
            scale = self.GetDPIScaleFactor()
            render = render_heatmap(self._heat, self._shapes, view, size[0], size[1],
                                    min_px=scale, block=max(1, int(scale)))
            self._cache = (self._content, view, size, render)
        if render is not None:
            if render.bitmap is not None:
                dc.DrawBitmap(render.bitmap, render.x, render.y, True)
            return
        # Preview: the last render, moved and scaled to the current view.
        _, (s0, c0), (w0, h0), old = cache
        if old.bitmap is None:
            return
        bw, bh = old.bitmap.GetWidth(), old.bitmap.GetHeight()
        x, y = self.to_screen(c0[0] + (old.x - w0 / 2.0) / s0,
                              c0[1] + (old.y - h0 / 2.0) / s0, view)
        k = view[0] / s0
        src = wx.MemoryDC(old.bitmap)
        dc.StretchBlit(int(round(x)), int(round(y)), int(round(bw * k)), int(round(bh * k)),
                       src, 0, 0, bw, bh)
        src.SelectObject(wx.NullBitmap)

    def value_at(self, px, py, view):
        """(distance, delta) shown at a pixel: distance None off copper, inf
        on unreached copper; delta None unless in delta mode."""
        if self._heat is None:
            return None, None
        render = self._current_render(view)
        if render is not None:
            return render.values_at(px, py)
        v, d = self._heat.sample_delta(*self.to_world(px + 0.5, py + 0.5, view))  # mid-zoom/pan
        return (v, d) if np.isfinite(v) else (None, None)

    # -- mouse --------------------------------------------------------------
    def _on_wheel(self, evt):
        view = self._view()
        if view is None:
            return
        steps = evt.GetWheelRotation() / float(evt.GetWheelDelta() or 120)
        new_zoom = min(500.0, max(0.5, self._zoom * self._ZOOM_STEP ** steps))
        # Keep the world point under the cursor where it is.
        px, py = evt.GetPosition()
        ux, uy = self.to_world(px, py, view)
        s_new = view[0] * new_zoom / self._zoom
        w, h = self.GetClientSize()
        self._zoom = new_zoom
        self._center = (ux - (px - w / 2.0) / s_new, uy - (py - h / 2.0) / s_new)
        self._view_changed()

    def _on_left_down(self, evt):
        view = self._view()
        if view is not None:
            self._drag = (evt.GetPosition(), view[1])
            self.CaptureMouse()

    def _on_left_up(self, _evt):
        self._drag = None
        if self.HasCapture():
            self.ReleaseMouse()

    def _on_mouse_motion(self, evt):
        view = self._view()
        if self._drag is not None and evt.LeftIsDown() and view is not None:
            (x0, y0), (cx, cy) = self._drag
            x, y = evt.GetPosition()
            self._center = (cx - (x - x0) / view[0], cy - (y - y0) / view[0])
            self._view_changed()
        self._mouse = tuple(evt.GetPosition())
        self._set_readout(self._make_readout(*self._mouse))


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

def _short(text, n=30):
    return text if len(text) <= n else text[:n - 1] + "\u2026"


def _scale_borders(sizer, ratio):
    """Scale every border in a sizer tree (sizer borders are device pixels)."""
    for item in sizer.GetChildren():
        item.SetBorder(int(round(item.GetBorder() * ratio)))
        if item.IsSizer():
            _scale_borders(item.GetSizer(), ratio)


class IsoplotFrame(wx.Frame):
    def __init__(self, icon_path=None):
        super().__init__(None, title="Net Isoplot")
        self.SetSize(self.FromDIP(wx.Size(1040, 780)))
        if icon_path:
            self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_PNG))
        self._session = None
        self._geometry = None
        self._field = None
        self._display = None        # the field's layers from prepare_layers()
        self._summary = ""          # footer: what the last result shows
        self._layer_names = None
        self._unticked = set()      # layer names the user switched off
        self._checks = []           # (layer_index, wx.CheckBox)
        self._kind_checks = {}      # "via"/"pad" -> wx.CheckBox ("Show" switches)
        self._fixed = []            # (window, size in DIP), re-applied on DPI change
        self._status = ""           # last status text (restored after "busy")
        self._note = None           # delta mode: why a selection was not taken
        self._busy = False
        self._delta = False         # delta mode (else isoplot)
        self._build_ui()
        self.CreateStatusBar()
        self.SetStatusText("Connecting to KiCad...")
        self.Bind(wx.EVT_DPI_CHANGED, self._on_dpi_changed)

    def attach(self, session):
        self._session = session

    # -- UI -----------------------------------------------------------------
    def _fixed_size(self, window, dip_size):
        self._fixed.append((window, dip_size))
        window.SetMinSize(self.FromDIP(wx.Size(*dip_size)))
        return window

    def _build_ui(self):
        self._panel = p = wx.Panel(self)
        d = self.FromDIP
        outer = wx.BoxSizer(wx.VERTICAL)

        body = wx.BoxSizer(wx.HORIZONTAL)
        self._view = _BoardView(p, d(wx.Size(_VIEW_W, _VIEW_H)))
        body.Add(self._view, 1, wx.EXPAND | wx.ALL, d(8))

        right = wx.BoxSizer(wx.VERTICAL)
        mode = wx.BoxSizer(wx.HORIZONTAL)
        self._mode_iso = wx.RadioButton(p, label="Isoplot", style=wx.RB_GROUP)
        self._mode_iso.SetToolTip("Distance along the copper from the selection")
        self._mode_delta = wx.RadioButton(p, label="Delta")
        self._mode_delta.SetToolTip("Which of two pads/vias each point is nearer to,\n"
                                    "and by how much. Select the two in KiCad, one\n"
                                    "after the other or both at once.")
        for rb in (self._mode_iso, self._mode_delta):
            rb.Bind(wx.EVT_RADIOBUTTON, self._on_mode)
            mode.Add(rb, 0, wx.RIGHT, d(12))
        right.Add(mode, 0, wx.BOTTOM, d(12))
        self._right = right
        self._legend_iso = self._make_legend()
        self._legend_delta = self._make_delta_legend()
        right.Add(self._legend_iso, 0, wx.BOTTOM, d(16))
        right.Add(self._legend_delta, 0, wx.BOTTOM, d(16))
        right.Hide(self._legend_delta)
        right.Add(wx.StaticText(p, label="Layers:"), 0, wx.BOTTOM, d(4))
        self._layers = wx.BoxSizer(wx.VERTICAL)
        right.Add(self._layers, 0)
        right.Add(wx.StaticText(p, label="Show:"), 0, wx.TOP | wx.BOTTOM, d(4))
        for kind, label, tip in (
                ("via", "Vias", "Draw the net's vias, with the drill at its real size.\n"
                                "Hidden vias still join the layers in the distances."),
                ("pad", "Pads", "Draw the net's pads (SMD pads outlined in grey).\n"
                                "Hidden pads still count in the distances; the "
                                "selected copper stays marked in black.")):
            cb = wx.CheckBox(p, label=label)
            cb.SetValue(True)
            cb.SetToolTip(tip)
            cb.Bind(wx.EVT_CHECKBOX, self._redraw)
            right.Add(cb, 0, wx.BOTTOM, d(6))
            self._kind_checks[kind] = cb
        right.AddStretchSpacer(1)

        self._follow = wx.CheckBox(p, label="Follow selection")
        self._follow.SetValue(True)
        self._follow.SetToolTip("Recompute when you select another pad or via.\n"
                                "Untick to keep the current one (in delta mode: the\n"
                                "pair) while you click around.")
        self._follow.Bind(wx.EVT_CHECKBOX, self._on_follow)
        right.Add(self._follow, 0, wx.TOP, d(8))
        self._on_top = wx.CheckBox(p, label="Always on top")
        self._on_top.Bind(wx.EVT_CHECKBOX, self._on_always_on_top)
        right.Add(self._on_top, 0, wx.TOP, d(4))
        refresh = wx.Button(p, label="Refresh")
        refresh.SetToolTip("Re-read the board and recompute now")
        refresh.Bind(wx.EVT_BUTTON, lambda e: self._session and self._session.refresh())
        right.Add(refresh, 0, wx.TOP | wx.EXPAND, d(8))
        outline = wx.Button(p, label="Update Board Outline")
        outline.SetToolTip("Re-read Edge.Cuts and the board's drill holes now\n"
                           "(holes also follow edits by themselves within a few seconds)")
        outline.Bind(wx.EVT_BUTTON,
                     lambda e: self._session and self._session.refresh_outline())
        right.Add(outline, 0, wx.TOP | wx.EXPAND, d(4))
        right.Add(wx.StaticText(p, label="Wheel: zoom    Drag: pan\nDouble-click: whole board"),
                  0, wx.TOP, d(8))

        body.Add(right, 0, wx.EXPAND | wx.ALL, d(8))
        outer.Add(body, 1, wx.EXPAND)
        p.SetSizer(outer)

    def _make_legend(self):
        p = self._panel
        d = self.FromDIP
        box = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(p, label="Net distance")
        tf = title.GetFont()
        tf.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(tf)
        box.Add(title, 0, wx.BOTTOM, d(2))
        self._legend_net = wx.StaticText(p, label="")
        box.Add(self._legend_net, 0, wx.BOTTOM, d(8))
        row = wx.BoxSizer(wx.HORIZONTAL)
        bar = self._fixed_size(_GradientBar(p, d(wx.Size(34, 212))), (34, 212))
        row.Add(bar, 0, wx.RIGHT, d(8))
        labels = wx.BoxSizer(wx.VERTICAL)
        self._legend_far = wx.StaticText(p, label="- mm  (furthest)")
        self._legend_mid = wx.StaticText(p, label="- mm")
        labels.Add(self._legend_far, 0)
        labels.AddStretchSpacer(1)
        labels.Add(self._legend_mid, 0)
        labels.AddStretchSpacer(1)
        labels.Add(wx.StaticText(p, label="0.00 mm  (selected)"), 0)
        row.Add(labels, 0, wx.EXPAND)
        box.Add(row, 0)
        return box

    def _make_delta_legend(self):
        p = self._panel
        d = self.FromDIP
        box = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(p, label="Distance delta")
        tf = title.GetFont()
        tf.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(tf)
        box.Add(title, 0, wx.BOTTOM, d(2))
        self._delta_net = wx.StaticText(p, label="")
        box.Add(self._delta_net, 0, wx.BOTTOM, d(6))
        self._channel_text = []
        for rgb in CHANNEL_RGB:     # which objects were read in
            row = wx.BoxSizer(wx.HORIZONTAL)
            chip = self._fixed_size(wx.Panel(p, size=d(wx.Size(14, 14))), (14, 14))
            chip.SetBackgroundColour(wx.Colour(*rgb))
            row.Add(chip, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, d(6))
            text = wx.StaticText(p, label="")
            row.Add(text, 0, wx.ALIGN_CENTER_VERTICAL)
            box.Add(row, 0, wx.BOTTOM, d(4))
            self._channel_text.append(text)
        self.set_channels((None, None), None)
        row = wx.BoxSizer(wx.HORIZONTAL)
        bar = self._fixed_size(_GradientBar(p, d(wx.Size(34, 180)), _DELTA_STOPS), (34, 180))
        row.Add(bar, 0, wx.RIGHT, d(8))
        labels = wx.BoxSizer(wx.VERTICAL)
        self._delta_top = wx.StaticText(p, label="- mm  (1 nearer)")
        self._delta_bottom = wx.StaticText(p, label="- mm  (2 nearer)")
        labels.Add(self._delta_top, 0)
        labels.AddStretchSpacer(1)
        labels.Add(wx.StaticText(p, label="equal distance"), 0)
        labels.AddStretchSpacer(1)
        labels.Add(self._delta_bottom, 0)
        row.Add(labels, 0, wx.EXPAND)
        box.Add(row, 0, wx.TOP, d(4))
        swap = wx.Button(p, label="Swap 1 and 2")
        swap.SetToolTip("Exchange the two objects (and so the colours)")
        swap.Bind(wx.EVT_BUTTON, lambda e: self._session and self._session.swap_pair())
        box.Add(swap, 0, wx.TOP, d(8))
        return box

    def _rebuild_layer_list(self, names, colors):
        self._layers.Clear(delete_windows=True)
        self._fixed = [(w, s) for (w, s) in self._fixed if w]  # drop destroyed swatches
        self._checks = []
        d = self.FromDIP
        for li, name in enumerate(names):
            row = wx.BoxSizer(wx.HORIZONTAL)
            swatch = self._fixed_size(wx.Panel(self._panel, size=d(wx.Size(24, 24))), (24, 24))
            swatch.SetBackgroundColour(wx.Colour(*colors[li]))
            row.Add(swatch, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, d(8))
            cb = wx.CheckBox(self._panel, label=name)
            cb.SetValue(name not in self._unticked)
            cb.Bind(wx.EVT_CHECKBOX, self._on_toggle)
            row.Add(cb, 0, wx.ALIGN_CENTER_VERTICAL)
            self._layers.Add(row, 0, wx.BOTTOM, d(6))
            self._checks.append((li, cb))
        self._layer_names = list(names)
        self._panel.Layout()

    def _on_dpi_changed(self, evt):
        """Moved to a display with other scaling: wx rescales fonts and native
        controls; sizer borders and our fixed sizes are ours to update."""
        old, new = evt.GetOldDPI().y, evt.GetNewDPI().y
        if old and new and old != new:
            _scale_borders(self._panel.GetSizer(), new / float(old))
            for window, (w, h) in self._fixed:
                window.SetMinSize(wx.Size(int(round(w * new / 96.0)), int(round(h * new / 96.0))))
            self._panel.Layout()
        evt.Skip()

    # -- called by the live session (on the UI thread) ------------------------
    def set_status(self, text):
        self._status = text
        self._update_footer()

    def set_busy(self, busy, pad_selected=False):
        """KiCad refuses item reads while a tool is active - which includes
        having a single pad selected (see kicad_source.BoardReader)."""
        self._busy = "pad" if busy and pad_selected else busy
        self._update_footer()

    def set_note(self, text):
        """Delta mode: why the last selection was not taken (None: it was)."""
        self._note = text
        self._update_footer()

    def set_channels(self, labels, net):
        """Delta mode: the two objects read in (label or None per slot) and
        their net, shown above the scale."""
        for i, (text, label) in enumerate(zip(self._channel_text, labels)):
            text.SetLabel("%d: %s" % (i + 1, _short(label) if label else "select a pad or via"))
        self._delta_net.SetLabel(_short(net or ""))
        self._panel.Layout()

    def _update_footer(self):
        """One line: the grid shown, then what is going on."""
        if self._busy == "pad":
            status = ("KiCad holds back board edits while one pad is selected - "
                      "they show after your next click in KiCad")
        elif self._busy:
            status = ("KiCad is busy (a tool is active) - updates resume when you "
                      "leave the tool (Esc)")
        else:
            status = self._status
        note = self._note if self._delta else None
        self.SetStatusText("      ".join(t for t in (self._summary, note, status) if t))

    def set_outline(self, rings, chains):
        self._view.set_outline(rings, chains)

    def set_holes(self, holes):
        self._view.set_holes(holes)

    def show_message(self, text):
        self._geometry = self._field = self._display = None
        self._summary = ""
        self._view.set_message(text)
        self._update_footer()

    def show_result(self, geometry, field, display, final, seconds):
        """``display``: from prepare_layers(), or prepare_delta() for a
        delta-mode geometry (then ``field`` is the one from object 1)."""
        delta = getattr(geometry, "delta", False)
        if delta != self._delta:
            return                  # solved just before the mode changed
        self._geometry = geometry
        self._field = field
        self._display = display
        if geometry.layer_names != self._layer_names:
            self._rebuild_layer_list(geometry.layer_names, geometry.layer_colors)

        if delta:
            scale_mm = display.scale / MM
            self._delta_net.SetLabel(_short(geometry.net_name or ""))
            self._delta_top.SetLabel("%.2f mm  (1 nearer)" % scale_mm)
            self._delta_bottom.SetLabel("%.2f mm  (2 nearer)" % scale_mm)
        else:
            max_mm = field.max_distance_nm / MM
            self._legend_net.SetLabel((geometry.net_name or "")[:26])
            self._legend_far.SetLabel("%.2f mm  (furthest)" % max_mm)
            self._legend_mid.SetLabel("%.2f mm" % (max_mm / 2))
        self._panel.Layout()
        self._redraw()

        # Net and furthest distance are in the legend; the footer adds the rest.
        seeds = "" if delta or geometry.seed_count == 1 else "%d seeds      " % geometry.seed_count
        self._summary = ("%sgrid %d x %d @ %.3f mm, %d directions"
                         % (seeds, field.nx, field.ny, field.pitch_nm / MM, field.num_moves))
        status = "solved in %.2f s" % seconds
        if not final:
            status += ", refining..."
        if geometry.unfilled_zones:
            status += ("      %d zone(s) unfilled - press B in KiCad to fill"
                       % geometry.unfilled_zones)
        self.set_status(status)

    def raise_window(self):
        if self.IsIconized():
            self.Iconize(False)
        self.Show()
        self.Raise()

    # -- drawing ------------------------------------------------------------
    def _redraw(self, *_):
        field = self._field
        if field is None:
            return
        layers = [li for (li, cb) in self._checks if cb.GetValue()]
        if not layers:
            self._view.set_message("No layers selected")
            return
        if field.max_distance_nm <= 0.0:
            self._view.set_message(
                "The net has only the selected copper (nothing to measure).\n"
                "If it clearly has more copper, fill its zones (press B).")
            return
        if not np.isfinite(field.dist[layers]).any():
            self._view.set_message("No copper on these layers")
            return
        if self._delta:
            prepared, scale = self._display.layers, self._display.scale
        else:
            prepared, scale = self._display, None
        heat = Heatmap([prepared[li] for li in layers], field.origin, field.pitch_nm,
                       field.max_distance_nm, scale)
        g = self._geometry
        hidden = {kind for switch, cb in self._kind_checks.items() if not cb.GetValue()
                  for kind in _SWITCHED[switch]}
        shapes = [CopperShapes(g.prims, li, g.disc_kinds, g.poly_kinds, hidden)
                  for li in layers]
        marks = Marks(g.prims, g.poly_kinds, g.via_holes, layers, hidden)
        self._view.set_heatmap(heat, shapes, self._seed_shapes(), marks)

    def _seed_shapes(self):
        """Seed copper outlines in board coordinates (nm), as (rgb, shape):
        black, or in delta mode a dark shade of each object's colour."""
        g = self._geometry
        if self._delta:
            groups = zip(_DELTA_SEED_RGB, g.seed_shapes)
        else:
            groups = [((0, 0, 0), (g.prims.seed_discs, g.prims.seed_polys))]
        out = []
        for rgb, (discs, polys) in groups:
            out += [(rgb, ("disc", x, y, r)) for (_, x, y, r) in discs]
            polys = (_bridged(rings) for (_, rings) in polys)
            out += [(rgb, ("poly", p)) for p in polys if p is not None]
        return out

    # -- events -------------------------------------------------------------
    def _on_toggle(self, evt):
        cb = evt.GetEventObject()
        (self._unticked.discard if cb.GetValue() else self._unticked.add)(cb.GetLabel())
        self._redraw()

    def _on_mode(self, _evt):
        delta = self._mode_delta.GetValue()
        if delta == self._delta:
            return
        self._delta = delta
        self._right.Show(self._legend_iso, not delta)
        self._right.Show(self._legend_delta, delta)
        self._view.set_board_colour(_DELTA_BOARD_COLOUR if delta else _BOARD_COLOUR)
        self._note = None
        self.show_message("")       # until the new mode's result (or prompt) arrives
        self._panel.Layout()
        if self._session:
            self._session.set_mode(delta)

    def _on_follow(self, _evt):
        if self._session:
            self._session.set_follow(self._follow.GetValue())

    def _on_always_on_top(self, _evt):
        style = self.GetWindowStyleFlag()
        if self._on_top.GetValue():
            style |= wx.STAY_ON_TOP
        else:
            style &= ~wx.STAY_ON_TOP
        self.SetWindowStyleFlag(style)
