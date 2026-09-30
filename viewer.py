"""The Net Isoplot window: a heatmap of the distance field that stays open next
to KiCad and redraws whenever the live session delivers a new result.

Neither KiCad API can put pixels on the board canvas, so the heatmap lives in
its own window, with a per-layer toggle list, a legend and a cursor readout of
the along-copper distance and board position.
"""

from __future__ import annotations

import math

import numpy as np
import wx

# Background the transparent heatmap is composited over (light grey so the
# black seed pad and the copper colours both read well).
_BG_COLOUR = (215, 215, 215)       # outside the board
_BOARD_COLOUR = (245, 245, 245)    # inside the board outline
_EDGE_COLOUR = (70, 70, 70)        # Edge.Cuts
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


class _BoardView(wx.Panel):
    """Owner-drawn board view in world coordinates (nm).

    The frame of reference is the board outline, so the view doesn't jump when
    switching between nets of different size. The heatmap image is placed at
    its true position; the mouse wheel zooms around the cursor, dragging pans,
    and a double-click fits the whole board again.
    """

    _MARGIN = 12      # px around the fitted board
    _ZOOM_STEP = 1.25

    def __init__(self, parent, size, on_motion):
        super().__init__(parent, size=size)
        self._img = None
        self._img_origin = (0.0, 0.0)   # world position of the image's top-left
        self._img_pitch = 1.0           # nm per image pixel
        self._seeds = ()                # ("disc", x, y, r) / ("poly", [(x, y)...]) in nm
        self._rings = []                # closed board outline rings (nm)
        self._chains = []               # open outline pieces (nm)
        self._message = "Starting..."
        self._zoom = 1.0
        self._center = None             # world point at the view centre; None = fitted
        self._drag = None
        self._cache_key = None
        self._cache_bmp = None
        self._on_motion = on_motion
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, lambda e: setattr(self, "_drag", None))
        self.Bind(wx.EVT_LEFT_DCLICK, lambda e: self.fit())
        self.Bind(wx.EVT_MOTION, self._on_mouse_motion)
        self.Bind(wx.EVT_LEAVE_WINDOW, lambda e: self._on_motion(None))

    # -- content ------------------------------------------------------------
    def set_outline(self, rings, chains):
        old = self._bounds()
        self._rings, self._chains = rings, chains
        if self._bounds() != old:
            self.fit()
        self.Refresh()

    def set_heatmap(self, img, origin, pitch, seeds):
        """``origin`` is the world centre of pixel (0, 0); ``pitch`` nm/pixel."""
        had_frame = self._bounds() is not None
        self._img = img
        self._img_pitch = float(pitch)
        self._img_origin = (origin[0] - pitch / 2.0, origin[1] - pitch / 2.0)
        self._seeds = seeds
        self._message = None
        if not had_frame:
            self.fit()
        self.Refresh()

    def set_message(self, text):
        self._img = None
        self._seeds = ()
        self._message = text
        self.Refresh()

    def fit(self):
        self._zoom = 1.0
        self._center = None
        self.Refresh()

    # -- view transform -----------------------------------------------------
    def _bounds(self):
        """World box of the fitted view: the board outline, else the heatmap."""
        pts = [p for ring in self._rings for p in ring] + [p for c in self._chains for p in c]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (min(xs), min(ys), max(xs), max(ys))
        if self._img is not None:
            x0, y0 = self._img_origin
            return (x0, y0, x0 + self._img.GetWidth() * self._img_pitch,
                    y0 + self._img.GetHeight() * self._img_pitch)
        return None

    def _view(self):
        """(scale px/nm, world centre), or None when there is nothing to show."""
        box = self._bounds()
        w, h = self.GetClientSize()
        if box is None or w < 2 * self._MARGIN + 2 or h < 2 * self._MARGIN + 2:
            return None
        bw = max(1.0, box[2] - box[0])
        bh = max(1.0, box[3] - box[1])
        fit = min((w - 2 * self._MARGIN) / bw, (h - 2 * self._MARGIN) / bh)
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

    # -- painting -----------------------------------------------------------
    def _on_paint(self, _evt):
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(wx.Colour(*_BG_COLOUR)))
        dc.Clear()
        view = self._view()
        if view is not None:
            self._draw_board(dc, view)
        if self._message:
            dc.SetTextForeground(wx.Colour(60, 60, 60))
            dc.DrawText(self._message, 20, 20)

    def _draw_board(self, dc, view):
        def pts(poly):
            return [wx.Point(int(sx), int(sy))
                    for sx, sy in (self.to_screen(x, y, view) for (x, y) in poly)]

        if self._rings:  # board area (even-odd, so cut-outs stay grey)
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.SetBrush(wx.Brush(wx.Colour(*_BOARD_COLOUR)))
            # One polygon through all rings; each bridge from the anchor is
            # traversed out and back, so even-odd filling cancels it.
            anchor = self._rings[0][0]
            joined = []
            for ring in self._rings:
                joined += [anchor] + ring + [ring[0]]
            dc.DrawPolygon(pts(joined), fill_style=wx.ODDEVEN_RULE)

        if self._img is not None:
            self._draw_heatmap(dc, view)

        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen(wx.Colour(*_EDGE_COLOUR), 2))
        for poly in self._rings + self._chains:
            dc.DrawLines(pts(poly))

        dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
        dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0)))
        for shape in self._seeds:
            if shape[0] == "disc":
                _, x, y, r = shape
                sx, sy = self.to_screen(x, y, view)
                dc.DrawCircle(int(sx), int(sy), max(2, int(r * view[0])))
            elif len(shape[1]) >= 3:
                dc.DrawPolygon(pts(shape[1]))

    def _draw_heatmap(self, dc, view):
        """Crop the heatmap to the visible area and scale only that part."""
        s = view[0]
        w, h = self.GetClientSize()
        x0, y0 = self._img_origin
        p = self._img_pitch
        iw, ih = self._img.GetWidth(), self._img.GetHeight()
        vx0, vy0 = self.to_world(0, 0, view)
        vx1, vy1 = self.to_world(w, h, view)
        ix0 = max(0, math.floor((vx0 - x0) / p))
        iy0 = max(0, math.floor((vy0 - y0) / p))
        ix1 = min(iw, math.ceil((vx1 - x0) / p))
        iy1 = min(ih, math.ceil((vy1 - y0) / p))
        if ix1 <= ix0 or iy1 <= iy0:
            return
        sx, sy = self.to_screen(x0 + ix0 * p, y0 + iy0 * p, view)
        sw = max(1, int(round((ix1 - ix0) * p * s)))
        sh = max(1, int(round((iy1 - iy0) * p * s)))
        key = (id(self._img), ix0, iy0, ix1, iy1, sw, sh)
        if key != self._cache_key:
            sub = self._img.GetSubImage(wx.Rect(ix0, iy0, ix1 - ix0, iy1 - iy0))
            self._cache_bmp = wx.Bitmap(sub.Scale(sw, sh, wx.IMAGE_QUALITY_NORMAL))
            self._cache_key = key
        dc.DrawBitmap(self._cache_bmp, int(round(sx)), int(round(sy)), True)

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
        self.Refresh()

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
            self.Refresh()
            view = self._view()
        self._on_motion(self.to_world(*evt.GetPosition(), view) if view else None)


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
        self._status = ""           # last status text (restored after "busy")
        self._busy = False
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
        self._view = _BoardView(p, (_VIEW_W, _VIEW_H), self._on_motion)
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
        right.Add(refresh, 0, wx.TOP | wx.EXPAND, 8)
        outline = wx.Button(p, label="Update Board Outline")
        outline.SetToolTip("Re-read Edge.Cuts after editing the board outline")
        outline.Bind(wx.EVT_BUTTON,
                     lambda e: self._session and self._session.refresh_outline())
        right.Add(outline, 0, wx.TOP | wx.EXPAND, 4)
        right.Add(wx.StaticText(p, label="Wheel: zoom    Drag: pan\nDouble-click: whole board"),
                  0, wx.TOP, 8)

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
        self._status = text
        if not self._busy:
            self.SetStatusText(text)

    def set_busy(self, busy):
        """KiCad refuses API calls while an interactive tool is running."""
        self._busy = busy
        self.SetStatusText("KiCad is busy (a tool is active) - updates resume when you "
                           "leave the tool (Esc)" if busy else self._status)

    def set_outline(self, rings, chains):
        self._view.set_outline(rings, chains)

    def show_message(self, text):
        self._geometry = self._field = self._composite = None
        self._view.set_message(text)
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
            self._composite = None
            self._view.set_message("No layers selected")
            return
        if field.max_distance_nm <= 0.0:
            self._composite = None
            self._view.set_message(
                "The net has only the selected copper (nothing to measure).\n"
                "If it clearly has more copper, fill its zones (press B).")
            return
        self._composite = field.dist[layers].min(axis=0)
        img = heatmap_image(self._composite, field.max_distance_nm)
        if img is None:
            self._view.set_message("No copper on these layers")
            return
        self._view.set_heatmap(img, field.origin, field.pitch_nm, self._seed_shapes())

    def _seed_shapes(self):
        """Seed copper outlines in board coordinates (nm)."""
        prims = self._geometry.prims
        out = [("disc", x, y, r) for (_, x, y, r) in prims.seed_discs]
        out += [("poly", rings[0]) for (_, rings) in prims.seed_polys if rings]
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

    def _on_motion(self, world):
        if world is None:
            self._cursor.SetLabel("Cursor: -")
            return
        x, y = world
        where = "(x %.2f, y %.2f mm)" % (x / MM, y / MM)
        f = self._field
        v = np.inf
        if f is not None and self._composite is not None:
            ix = int(round((x - f.origin[0]) / f.pitch_nm))
            iy = int(round((y - f.origin[1]) / f.pitch_nm))
            if 0 <= ix < f.nx and 0 <= iy < f.ny:
                v = self._composite[iy, ix]
        if np.isfinite(v):
            self._cursor.SetLabel("Cursor: %.2f mm from selection   %s" % (v / MM, where))
        else:
            self._cursor.SetLabel("Cursor: (no copper here)   %s" % where)
