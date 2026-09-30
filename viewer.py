"""The Net Isoplot window: a heatmap of the distance field that stays open next
to KiCad and redraws whenever the live session delivers a new result.

Neither KiCad API can put pixels on the board canvas, so the heatmap lives in
its own window, with a per-layer toggle list, a legend and a cursor readout of
the along-copper distance and board position.
"""

from __future__ import annotations

import numpy as np
import wx

# Background the transparent heatmap is composited over (light grey so the
# black seed pad and the copper colours both read well).
_BG_COLOUR = (235, 235, 235)
_OPACITY = 0.6
_VIEW_W = 760
_VIEW_H = 620
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


def heatmap_image(dist, maxd):
    """RGBA ``wx.Image`` from a (ny, nx) distance array (inf = no copper), or
    None if there is no copper at all."""
    mask = np.isfinite(dist)
    if not mask.any():
        return None
    idx = np.zeros(dist.shape, dtype=np.int32)
    idx[mask] = (np.clip(dist[mask] / (maxd or 1.0), 0.0, 1.0) * 255).astype(np.int32)
    rgb = _LUT[idx]
    rgb[~mask] = 0
    ny, nx = dist.shape
    img = wx.Image(nx, ny)
    img.SetData(rgb.tobytes())
    img.InitAlpha()
    img.SetAlpha((mask.astype(np.uint8) * int(round(_OPACITY * 255))).tobytes())
    return img


class _GradientBar(wx.Panel):
    """Colour scale (top = far = blue ... bottom = near = red) with a frame.

    Built as a vertical stack of solid-colour child panels rather than by
    owner-drawing. On wxMSW the BG_STYLE_PAINT + AutoBufferedPaintDC paint path
    rendered the bar blank/white, while a plain ``wx.Panel`` with
    ``SetBackgroundColour`` draws reliably. The parent's black background shows
    through a 1 px border as the frame.
    """

    def __init__(self, parent, size=(34, 212), bands=210):
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


class _HeatmapPanel(wx.Panel):
    """Owner-drawn heatmap view.

    Holds the grid-resolution heatmap image, scales it to fit the client area
    (cached per size), and draws the seed copper filled black on top.
    """

    def __init__(self, parent, size, on_motion):
        super().__init__(parent, size=size)
        self._img = None
        self._bmp = None
        self._bmp_key = None
        self._seed_shapes = ()
        self._message = "Starting..."
        self._on_motion = on_motion
        self._scale = 0.0
        self._off_x = 0
        self._off_y = 0
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))
        self.Bind(wx.EVT_MOTION, lambda e: self._on_motion(e.GetPosition()))
        self.Bind(wx.EVT_LEAVE_WINDOW, lambda e: self._on_motion(None))

    def set_image(self, img, seed_shapes=(), message=None):
        self._img = img
        self._bmp = None
        self._seed_shapes = seed_shapes
        self._message = message
        self.Refresh()

    def cell_at(self, pos):
        """Map a client-coordinate position to image-grid (ix, iy), or None."""
        if self._img is None or self._scale <= 0:
            return None
        ix = int((pos[0] - self._off_x) / self._scale)
        iy = int((pos[1] - self._off_y) / self._scale)
        if 0 <= ix < self._img.GetWidth() and 0 <= iy < self._img.GetHeight():
            return ix, iy
        return None

    def _on_paint(self, _evt):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(wx.Colour(*_BG_COLOUR)))
        dc.Clear()
        w, h = self.GetClientSize()
        if self._message is not None:
            dc.SetTextForeground(wx.Colour(60, 60, 60))
            dc.DrawText(self._message, 20, 20)
            return
        if self._img is None or w < 2 or h < 2:
            return

        iw, ih = self._img.GetWidth(), self._img.GetHeight()
        self._scale = min(w / float(iw), h / float(ih))
        sw = max(1, int(iw * self._scale))
        sh = max(1, int(ih * self._scale))
        self._off_x = (w - sw) // 2
        self._off_y = (h - sh) // 2
        if self._bmp_key != (sw, sh) or self._bmp is None:
            self._bmp = wx.Bitmap(self._img.Scale(sw, sh, wx.IMAGE_QUALITY_NORMAL))
            self._bmp_key = (sw, sh)
        dc.DrawBitmap(self._bmp, self._off_x, self._off_y, True)

        dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
        dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0)))
        s, ox, oy = self._scale, self._off_x, self._off_y
        for shape in self._seed_shapes:
            if shape[0] == "disc":
                _, gx, gy, gr = shape
                dc.DrawCircle(int(ox + gx * s), int(oy + gy * s), max(2, int(gr * s)))
            else:
                pts = [wx.Point(int(ox + gx * s), int(oy + gy * s)) for (gx, gy) in shape[1]]
                if len(pts) >= 3:
                    dc.DrawPolygon(pts)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

class IsoplotFrame(wx.Frame):
    def __init__(self, icon_path=None):
        super().__init__(None, title="Net Isoplot", size=(1040, 780))
        if icon_path:
            self.SetIcon(wx.Icon(icon_path, wx.BITMAP_TYPE_PNG))
        self._session = None
        self._geometry = None
        self._field = None
        self._composite = None      # (ny, nx) distances currently displayed
        self._layer_names = None
        self._unticked = set()      # layer names the user switched off
        self._checks = []           # (layer_index, wx.CheckBox)
        self._build_ui()
        self.CreateStatusBar()
        self.SetStatusText("Connecting to KiCad...")

    def attach(self, session):
        self._session = session

    # -- UI -----------------------------------------------------------------
    def _build_ui(self):
        self._panel = p = wx.Panel(self)
        outer = wx.BoxSizer(wx.VERTICAL)

        self._info = wx.StaticText(p, label="Select a pad or via in the PCB editor.")
        outer.Add(self._info, 0, wx.ALL, 10)
        self._cursor = wx.StaticText(p, label="Cursor: -")
        f = self._cursor.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        self._cursor.SetFont(f)
        outer.Add(self._cursor, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        body = wx.BoxSizer(wx.HORIZONTAL)
        self._view = _HeatmapPanel(p, (_VIEW_W, _VIEW_H), self._on_motion)
        body.Add(self._view, 1, wx.EXPAND | wx.ALL, 8)

        right = wx.BoxSizer(wx.VERTICAL)
        right.Add(self._make_legend(), 0, wx.BOTTOM, 16)
        right.Add(wx.StaticText(p, label="Layers:"), 0, wx.BOTTOM, 4)
        self._layers = wx.BoxSizer(wx.VERTICAL)
        right.Add(self._layers, 0)
        right.AddStretchSpacer(1)

        self._follow = wx.CheckBox(p, label="Follow selection")
        self._follow.SetValue(True)
        self._follow.SetToolTip("Recompute when you select another pad or via.\n"
                                "Untick to keep the current one while you click around.")
        self._follow.Bind(wx.EVT_CHECKBOX, self._on_follow)
        right.Add(self._follow, 0, wx.TOP, 8)
        self._on_top = wx.CheckBox(p, label="Always on top")
        self._on_top.Bind(wx.EVT_CHECKBOX, self._on_always_on_top)
        right.Add(self._on_top, 0, wx.TOP, 4)
        refresh = wx.Button(p, label="Refresh")
        refresh.SetToolTip("Re-read the board and recompute now")
        refresh.Bind(wx.EVT_BUTTON, lambda e: self._session and self._session.refresh())
        right.Add(refresh, 0, wx.TOP, 8)

        body.Add(right, 0, wx.EXPAND | wx.ALL, 8)
        outer.Add(body, 1, wx.EXPAND)
        p.SetSizer(outer)

    def _make_legend(self):
        p = self._panel
        box = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(p, label="Net distance")
        tf = title.GetFont()
        tf.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(tf)
        box.Add(title, 0, wx.BOTTOM, 2)
        self._legend_net = wx.StaticText(p, label="")
        box.Add(self._legend_net, 0, wx.BOTTOM, 8)
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(_GradientBar(p, size=(34, 212)), 0, wx.RIGHT, 8)
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
        self._checks = []
        for li, name in enumerate(names):
            row = wx.BoxSizer(wx.HORIZONTAL)
            swatch = wx.Panel(self._panel, size=(24, 24))
            swatch.SetBackgroundColour(wx.Colour(*colors[li]))
            row.Add(swatch, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
            cb = wx.CheckBox(self._panel, label=name)
            cb.SetValue(name not in self._unticked)
            cb.Bind(wx.EVT_CHECKBOX, self._on_toggle)
            row.Add(cb, 0, wx.ALIGN_CENTER_VERTICAL)
            self._layers.Add(row, 0, wx.BOTTOM, 6)
            self._checks.append((li, cb))
        self._layer_names = list(names)
        self._panel.Layout()

    # -- called by the live session (on the UI thread) ------------------------
    def set_status(self, text):
        self.SetStatusText(text)

    def show_message(self, text):
        self._geometry = self._field = self._composite = None
        self._view.set_image(None, message=text)
        self._info.SetLabel(text)

    def show_result(self, geometry, field, final, seconds):
        self._geometry = geometry
        self._field = field
        if geometry.layer_names != self._layer_names:
            self._rebuild_layer_list(geometry.layer_names, geometry.layer_colors)

        max_mm = field.max_distance_nm / MM
        self._legend_net.SetLabel((geometry.net_name or "")[:26])
        self._legend_far.SetLabel("%.2f mm  (furthest)" % max_mm)
        self._legend_mid.SetLabel("%.2f mm" % (max_mm / 2))
        seeds = "" if geometry.seed_count == 1 else "  (%d seeds)" % geometry.seed_count
        self._info.SetLabel(
            "Net: %s%s    Furthest: %.2f mm along copper\n"
            "Grid: %d x %d @ %.3f mm    Red = near, blue = far"
            % (geometry.net_name, seeds, max_mm, field.nx, field.ny, field.pitch_nm / MM))
        self._panel.Layout()
        self._redraw()

        status = "%s: solved in %.1f s" % (geometry.net_name, seconds)
        if not final:
            status += ", refining..."
        if geometry.unfilled_zones:
            status += ("    %d zone(s) unfilled - press B in KiCad to fill"
                       % geometry.unfilled_zones)
        self.SetStatusText(status)

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
            self._composite = None
            self._view.set_image(None, message="No layers selected")
            return
        if field.max_distance_nm <= 0.0:
            self._composite = None
            self._view.set_image(None, message=(
                "The net has only the selected copper (nothing to measure).\n"
                "If it clearly has more copper, fill its zones (press B)."))
            return
        self._composite = field.dist[layers].min(axis=0)
        img = heatmap_image(self._composite, field.max_distance_nm)
        if img is None:
            self._view.set_image(None, message="No copper on these layers")
            return
        self._view.set_image(img, self._seed_shapes())

    def _seed_shapes(self):
        """Seed copper in image coordinates (cell (0,0) spans 0..1)."""
        f = self._field
        ox, oy = f.origin
        pitch = float(f.pitch_nm)

        def g(x, y):
            return ((x - ox) / pitch + 0.5, (y - oy) / pitch + 0.5)

        out = []
        prims = self._geometry.prims
        for (_, x, y, r) in prims.seed_discs:
            gx, gy = g(x, y)
            out.append(("disc", gx, gy, r / pitch))
        for (_, rings) in prims.seed_polys:
            if rings:
                out.append(("poly", [g(x, y) for (x, y) in rings[0]]))
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

    def _on_motion(self, pos):
        cell = self._view.cell_at(pos) if pos is not None else None
        if cell is None or self._composite is None:
            self._cursor.SetLabel("Cursor: -")
            return
        ix, iy = cell
        x, y = self._field.cell_center_world(ix, iy)
        where = "(x %.2f, y %.2f mm)" % (x / MM, y / MM)
        v = self._composite[iy, ix]
        if np.isfinite(v):
            self._cursor.SetLabel("Cursor: %.2f mm from selection   %s" % (v / MM, where))
        else:
            self._cursor.SetLabel("Cursor: (no copper here)   %s" % where)
