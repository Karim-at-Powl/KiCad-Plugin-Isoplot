"""The Net Isoplot window: a heatmap of the distance field that stays open next
to KiCad and redraws whenever the live session delivers a new result.

Neither KiCad API can put pixels on the board canvas, so the heatmap lives in
its own window, with a per-layer toggle list, a legend and a cursor readout of
the along-copper distance and board position.

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
_OPACITY = 0.6
_VIEW_W = 760                      # initial board view size (DIP)
_VIEW_H = 620
_SETTLE_MS = 120                   # re-render this long after the last zoom/pan
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


def colormap(frac):
    if frac <= 0:
        return _STOPS[0][1]
    if frac >= 1:
        return _STOPS[-1][1]
    for (f0, c0), (f1, c1) in zip(_STOPS, _STOPS[1:]):
        if f0 <= frac <= f1:
            t = (frac - f0) / (f1 - f0)
            return tuple(int(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
    return _STOPS[-1][1]


_LUT = np.array([colormap(i / 255.0) for i in range(256)], dtype=np.uint8)

# The same as packed RGBA words (bytes R, G, B, A in memory), opaque, plus a
# transparent entry 256 for "no data"; and an AND-mask per copper coverage
# that scales alpha down to the heatmap opacity.
_LUT32 = np.zeros(257, dtype=np.uint32)
_LUT32[:256] = (_LUT[:, 0].astype(np.uint32) | _LUT[:, 1].astype(np.uint32) << 8
                | _LUT[:, 2].astype(np.uint32) << 16 | np.uint32(0xFF) << 24)
_ALPHA32 = np.array([0x00FFFFFF | round(c * _OPACITY) << 24 for c in range(256)],
                    dtype=np.uint32)


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


class Heatmap:
    """Per-layer distances of the shown layers, ready to be sampled."""

    def __init__(self, layers, origin, pitch, max_distance):
        """``layers``: the shown layers from :func:`prepare_layers`."""
        self._layers = layers
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

    def layer(self, li, iy, ty, ix, tx):
        """Bilinear distances of one layer on the grid of rows (iy, ty) x
        columns (ix, tx); NaN where there is no data. Cells without data are
        left out and the other weights renormalised, so values don't bleed in
        from outside the copper. Separable, so it costs a few array passes."""
        c0 = int(ix.min())
        jx = ix - c0
        c1 = int(ix.max()) + 2
        num, den = (self._across(a[:, c0:c1], iy, ty, jx, tx) for a in self._layers[li])
        with np.errstate(invalid="ignore", divide="ignore"):
            num /= den
        num[den < 1e-4] = np.nan
        return num

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
        ix, tx = self.axis([x], True)
        iy, ty = self.axis([y], False)
        vals = [self.layer(li, iy, ty, ix, tx)[0, 0] for li in range(self.num_layers)]
        return float(np.fmin.reduce(vals)) if vals else np.nan


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

    def __init__(self, prims, layer):
        self.segments = np.array([s[1:] for s in prims.segments if s[0] == layer],
                                 dtype=np.float64).reshape(-1, 5)   # x0 y0 x1 y1 width
        self.discs = np.array([d[1:] for d in prims.discs if d[0] == layer],
                              dtype=np.float64).reshape(-1, 3)      # x y r
        polys = (_bridged(rings) for (li, rings) in prims.polys if li == layer)
        self.polys = [p for p in polys if p is not None]
        self.poly_boxes = np.array([(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
                                    for p in self.polys], dtype=np.float64).reshape(-1, 4)


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

    def __init__(self, bitmap, coverage, dist, block):
        self.bitmap = bitmap        # RGBA wx.Bitmap, device pixels, or None if empty
        self.coverage = coverage    # (h, w) uint8, copper of all shown layers
        self.dist = dist            # (ceil(h/block), ceil(w/block)) float32, NaN = none
        self.block = block
        self.x = self.y = 0         # position in the viewport

    def value_at(self, px, py):
        """Distance shown at a viewport pixel: None off copper, inf on
        unreached copper."""
        px, py = px - self.x, py - self.y
        cov = self.coverage
        if not (0 <= py < cov.shape[0] and 0 <= px < cov.shape[1]) or not cov[py, px]:
            return None
        v = self.dist[py // self.block, px // self.block]
        return float(v) if np.isfinite(v) else np.inf


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
    shown = np.full((bh, bw), np.nan, dtype=np.float32)     # nearest covering layer
    anywhere = np.full((bh, bw), np.nan, dtype=np.float32)  # nearest layer at all
    for li, shapes in enumerate(layer_shapes):
        cov = _copper_coverage(shapes, view, canvas, buf, min_px)
        if coverage is None:
            coverage = cov
        else:
            np.maximum(coverage, cov, out=coverage)
        d = heat.layer(li, iy, ty, ix, tx)
        np.fmin(anywhere, d, out=anywhere)
        at_centres = np.pad(cov[block // 2::block, block // 2::block], pad)
        d[at_centres == 0] = np.nan
        np.fmin(shown, d, out=shown)
    # Anti-aliased edge pixels whose block centre is off copper take the
    # nearest layer's value regardless.
    dist = np.where(np.isnan(shown), anywhere, shown)

    q = dist * np.float32(255.0 / (heat.max_distance or 1.0))
    np.clip(q, 0, 255, out=q)
    q[np.isnan(q)] = 256   # copper the seed can't reach stays uncoloured
    small = _LUT32[q.astype(np.intp)]
    rgba = np.empty((bh * block, bw * block), dtype=np.uint32)
    for i in range(block):          # strided copies beat a broadcast here
        for j in range(block):
            rgba[i::block, j::block] = small
    if edge[0] or edge[1]:
        rgba[h:, :] = 0
        rgba[:, w:] = 0
    visible = rgba[:h, :w]
    np.bitwise_and(visible, _ALPHA32[coverage], out=visible)
    bitmap = wx.Bitmap.FromBufferRGBA(bw * block, bh * block, rgba.view(np.uint8))
    return Render(bitmap, coverage, dist, block)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class _GradientBar(wx.Panel):
    """Colour scale (top = far = blue ... bottom = near = red) with a frame.

    Built as a vertical stack of solid-colour child panels rather than by
    owner-drawing. On wxMSW the BG_STYLE_PAINT + AutoBufferedPaintDC paint path
    rendered the bar blank/white, while a plain ``wx.Panel`` with
    ``SetBackgroundColour`` draws reliably. The parent's black background shows
    through a 1 px border as the frame.
    """

    def __init__(self, parent, size, bands=210):
        super().__init__(parent, size=size)
        self.SetBackgroundColour(wx.Colour(0, 0, 0))
        stack = wx.BoxSizer(wx.VERTICAL)
        for i in range(bands):
            band = wx.Panel(self)
            band.SetBackgroundColour(wx.Colour(*colormap(1.0 - i / float(bands - 1))))
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
        self._seeds = ()                # ("disc", x, y, r) / ("poly", points) in nm
        self._rings = []                # closed board outline rings (nm)
        self._chains = []               # open outline pieces (nm)
        self._message = "Starting..."
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
        if self._bounds() != old:
            self.fit()
        self._invalidate()

    def set_heatmap(self, heat, shapes, seeds):
        had_frame = self._bounds() is not None
        self._heat, self._shapes, self._seeds = heat, shapes, seeds
        self._content += 1
        self._message = None
        if not had_frame:
            self.fit()
        self._invalidate()

    def set_message(self, text):
        self._heat = self._shapes = None
        self._content += 1
        self._seeds = ()
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
            tw, th, _ = dc.GetMultiLineTextExtent(self._message)
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
            value = self.value_at(px, py, view)
            if value is None:
                lines.append("No copper here")
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
            gc.SetBrush(wx.Brush(wx.Colour(*_BOARD_COLOUR)))
            gc.DrawLines(self._screen_pts(_bridged(self._rings), view), wx.ODDEVEN_RULE)

    def _draw_overlay(self, gc, view):
        """Board outline and seed copper, on top of the heatmap."""
        gc.SetPen(gc.CreatePen(wx.GraphicsPenInfo(wx.Colour(*_EDGE_COLOUR),
                                                  1.5 * self.GetDPIScaleFactor())))
        for poly in self._rings + self._chains:
            gc.StrokeLines(self._screen_pts(poly, view))

        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.SetBrush(wx.BLACK_BRUSH)
        for shape in self._seeds:
            if shape[0] == "disc":
                _, x, y, r = shape
                sx, sy = self.to_screen(x, y, view)
                r = max(self.FromDIP(2), r * view[0])
                gc.DrawEllipse(sx - r, sy - r, 2 * r, 2 * r)
            else:
                gc.DrawLines(self._screen_pts(shape[1], view), wx.ODDEVEN_RULE)

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
        """Distance shown at a pixel: None off copper, inf on unreached copper."""
        if self._heat is None:
            return None
        render = self._current_render(view)
        if render is not None:
            return render.value_at(px, py)
        v = self._heat.sample(*self.to_world(px + 0.5, py + 0.5, view))  # mid-zoom/pan
        return v if np.isfinite(v) else None

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
        self._fixed = []            # (window, size in DIP), re-applied on DPI change
        self._status = ""           # last status text (restored after "busy")
        self._busy = False
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
        right.Add(self._make_legend(), 0, wx.BOTTOM, d(16))
        right.Add(wx.StaticText(p, label="Layers:"), 0, wx.BOTTOM, d(4))
        self._layers = wx.BoxSizer(wx.VERTICAL)
        right.Add(self._layers, 0)
        right.AddStretchSpacer(1)

        self._follow = wx.CheckBox(p, label="Follow selection")
        self._follow.SetValue(True)
        self._follow.SetToolTip("Recompute when you select another pad or via.\n"
                                "Untick to keep the current one while you click around.")
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
        outline.SetToolTip("Re-read Edge.Cuts after editing the board outline")
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

    def set_busy(self, busy):
        """KiCad refuses API calls while an interactive tool is running."""
        self._busy = busy
        self._update_footer()

    def _update_footer(self):
        """One line: the grid shown, then what is going on."""
        if self._busy:
            status = ("KiCad is busy (a tool is active) - updates resume when you "
                      "leave the tool (Esc)")
        else:
            status = self._status
        self.SetStatusText("      ".join(t for t in (self._summary, status) if t))

    def set_outline(self, rings, chains):
        self._view.set_outline(rings, chains)

    def show_message(self, text):
        self._geometry = self._field = self._display = None
        self._summary = ""
        self._view.set_message(text)
        self._update_footer()

    def show_result(self, geometry, field, display, final, seconds):
        self._geometry = geometry
        self._field = field
        self._display = display
        if geometry.layer_names != self._layer_names:
            self._rebuild_layer_list(geometry.layer_names, geometry.layer_colors)

        max_mm = field.max_distance_nm / MM
        self._legend_net.SetLabel((geometry.net_name or "")[:26])
        self._legend_far.SetLabel("%.2f mm  (furthest)" % max_mm)
        self._legend_mid.SetLabel("%.2f mm" % (max_mm / 2))
        self._panel.Layout()
        self._redraw()

        # Net and furthest distance are in the legend; the footer adds the rest.
        seeds = "" if geometry.seed_count == 1 else "%d seeds      " % geometry.seed_count
        self._summary = ("%sgrid %d x %d @ %.3f mm"
                         % (seeds, field.nx, field.ny, field.pitch_nm / MM))
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
        heat = Heatmap([self._display[li] for li in layers], field.origin, field.pitch_nm,
                       field.max_distance_nm)
        shapes = [CopperShapes(self._geometry.prims, li) for li in layers]
        self._view.set_heatmap(heat, shapes, self._seed_shapes())

    def _seed_shapes(self):
        """Seed copper outlines in board coordinates (nm)."""
        prims = self._geometry.prims
        out = [("disc", x, y, r) for (_, x, y, r) in prims.seed_discs]
        polys = (_bridged(rings) for (_, rings) in prims.seed_polys)
        out += [("poly", p) for p in polys if p is not None]
        return out

    # -- events -------------------------------------------------------------
    def _on_toggle(self, evt):
        cb = evt.GetEventObject()
        (self._unticked.discard if cb.GetValue() else self._unticked.add)(cb.GetLabel())
        self._redraw()

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
