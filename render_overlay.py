"""Render a DistanceField as a heatmap and show it in a wxPython window.

KiCad 9's Python API exposes no way to load pixel data onto the canvas: the
``REFERENCE_IMAGE`` / ``BITMAP_BASE`` classes returned by
``PCB_REFERENCE_IMAGE.GetReferenceImage()`` are not SWIG-wrapped, so there is no
``ReadImageFile`` / ``SetImage`` to call. We therefore composite the heatmap into
wx bitmaps in memory and present them in a dialog with a per-layer toggle list, a
legend and a live cursor-distance readout, rather than placing reference images
on the board.

Image generation uses the bundled wxPython only (no PIL/matplotlib). NumPy is
used to build the pixel buffers quickly when available, with a pure-Python
fallback.
"""

from __future__ import annotations

import wx

try:
    import numpy as _np
    HAS_NUMPY = True
except Exception:
    _np = None
    HAS_NUMPY = False

GROUP_NAME = "Powl_NetIsoplot"

# Background the transparent heatmap is composited over (light grey so the
# black origin pad and the copper colours both read well).
_BG_COLOUR = (235, 235, 235)
# Initial on-screen size of the heatmap area (it scales to fit, aspect kept,
# and grows with the window).
_VIEW_W = 760
_VIEW_H = 620


# ---------------------------------------------------------------------------
# Colour map  (near = red / hot ... far = blue / cool)
# ---------------------------------------------------------------------------

_STOPS = [
    (0.00, (255, 0, 0)),      # selected / closest = red (hot)
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
    for i in range(len(_STOPS) - 1):
        f0, c0 = _STOPS[i]
        f1, c1 = _STOPS[i + 1]
        if f0 <= frac <= f1:
            t = (frac - f0) / (f1 - f0)
            return tuple(int(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
    return _STOPS[-1][1]


def _colormap_lut(n=256):
    return [colormap(i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _image_from_flat(dist, nx, ny, maxd, opacity):
    """Build an RGBA ``wx.Image`` from a flat distance buffer.

    ``dist`` is a flat list (len nx*ny) of geodesic distance in nm or ``None``
    where there is no reachable copper. Returns the image, or ``None`` if the
    buffer holds no copper at all.
    """
    maxd = maxd or 1.0
    alpha_val = int(max(0, min(255, round(opacity * 255))))
    lut = _colormap_lut()

    if HAS_NUMPY:
        arr = _np.array([(-1.0 if v is None else v) for v in dist],
                        dtype=_np.float64)
        return _image_from_np(arr, nx, ny, maxd, alpha_val, lut)

    if not any(v is not None for v in dist):
        return None
    rgb = bytearray(nx * ny * 3)
    alpha = bytearray(nx * ny)
    for i, v in enumerate(dist):
        if v is None:
            continue
        frac = v / maxd
        frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
        c = lut[int(frac * 255)]
        j = i * 3
        rgb[j], rgb[j + 1], rgb[j + 2] = c
        alpha[i] = alpha_val
    img = wx.Image(nx, ny)
    img.SetData(bytes(rgb))
    img.InitAlpha()
    img.SetAlpha(bytes(alpha))
    return img


def _image_from_np(arr, nx, ny, maxd, alpha_val, lut):
    """Build an RGBA ``wx.Image`` from a flat NumPy array (<0 == no copper)."""
    mask = arr >= 0.0
    if not bool(mask.any()):
        return None
    frac = _np.zeros_like(arr)
    frac[mask] = _np.clip(arr[mask] / maxd, 0.0, 1.0)
    lut_np = _np.array(lut, dtype=_np.uint8)
    idx = (frac * 255).astype(_np.int32)
    rgb = lut_np[idx]                       # (n, 3)
    rgb[~mask] = 0
    rgb = rgb.reshape(ny, nx, 3)
    alpha = (mask.astype(_np.uint8) * alpha_val).reshape(ny, nx)
    img = wx.Image(nx, ny)
    img.SetData(rgb.tobytes())
    img.InitAlpha()
    img.SetAlpha(alpha.tobytes())
    return img


class _GradientBar(wx.Panel):
    """Colour scale (top = far = blue ... bottom = near = red) with a frame.

    Built as a vertical stack of solid-colour child panels rather than by
    owner-drawing. On this wxMSW build the BG_STYLE_PAINT + AutoBufferedPaintDC
    paint path rendered the bar blank/white whether it drew a bitmap or solid
    rectangles, while a plain ``wx.Panel`` with ``SetBackgroundColour`` (the
    same widget the layer swatches use) draws reliably. So each gradient band
    is just such a panel; the parent's black background shows through a 1 px
    border as the frame.
    """

    def __init__(self, parent, size=(34, 212), bands=210):
        super().__init__(parent, size=size)
        self.SetBackgroundColour(wx.Colour(0, 0, 0))  # shows as the frame
        stack = wx.BoxSizer(wx.VERTICAL)
        for i in range(bands):
            # top band (i=0) = far = blue (frac 1.0); bottom = near = red.
            r, g, b = colormap(1.0 - i / float(bands - 1))
            band = wx.Panel(self)
            band.SetBackgroundColour(wx.Colour(r, g, b))
            stack.Add(band, 1, wx.EXPAND)
        frame = wx.BoxSizer(wx.VERTICAL)
        frame.Add(stack, 1, wx.EXPAND | wx.ALL, 1)
        self.SetSizer(frame)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class _HeatmapPanel(wx.Panel):
    """Owner-drawn heatmap view.

    Holds the raw (grid-resolution) heatmap image and scales it to fit the
    current client size on every paint, so the view follows the window. Also
    draws the selection's copper filled black, and (unlike wx.StaticBitmap)
    reliably delivers mouse-motion events on wxMSW for the cursor readout.
    """

    def __init__(self, parent, size, on_motion, bg=_BG_COLOUR):
        super().__init__(parent, size=size)
        self._img = None
        self._seed_shapes = ()
        self._message = None
        self._on_motion = on_motion
        self._bg = bg
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
        dc.SetBackground(wx.Brush(wx.Colour(*self._bg)))
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
        dc.DrawBitmap(wx.Bitmap(self._img.Scale(sw, sh, wx.IMAGE_QUALITY_NORMAL)),
                      self._off_x, self._off_y, True)

        dc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
        dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0)))
        for shape in self._seed_shapes:
            if shape[0] == "disc":
                _, gx, gy, gr = shape
                dc.DrawCircle(int(self._off_x + gx * self._scale),
                              int(self._off_y + gy * self._scale),
                              max(2, int(gr * self._scale)))
            elif shape[0] == "poly":
                pts = [wx.Point(int(self._off_x + gx * self._scale),
                                int(self._off_y + gy * self._scale))
                       for (gx, gy) in shape[1]]
                if len(pts) >= 3:
                    dc.DrawPolygon(pts)


class _OverlayDialog(wx.Dialog):
    def __init__(self, parent, field, extraction, opacity, seed_shapes):
        super().__init__(parent, title="Net Isoplot",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._field = field
        self._nx, self._ny = field.nx, field.ny
        self._maxd = field.max_distance_nm or 1.0
        self._opacity = opacity
        self._seed_shapes = seed_shapes
        self._lut = _colormap_lut()
        self._alpha_val = int(max(0, min(255, round(opacity * 255))))

        # Per-layer distance buffers (NumPy float arrays, <0 == no copper), and
        # which layers actually carry copper.
        self._layer_arr = []
        self._has_copper = []
        for li in range(field.num_layers):
            present = any(v is not None for v in field.dist[li])
            self._has_copper.append(present)
            if HAS_NUMPY:
                self._layer_arr.append(
                    _np.array([(-1.0 if v is None else v)
                               for v in field.dist[li]], dtype=_np.float64))
            else:
                self._layer_arr.append(field.dist[li])

        self._cur_flat = None        # flat distance buffer currently displayed

        self._build_ui(field, extraction)
        self._rebuild()

    # -- UI -----------------------------------------------------------------
    def _build_ui(self, field, extraction):
        outer = wx.BoxSizer(wx.VERTICAL)

        max_mm = field.max_distance_nm / 1e6
        self._info = wx.StaticText(self, label=(
            "Net: %s    Furthest: %.2f mm (along copper)\n"
            "Grid: %d x %d @ %.3f mm    Red = near, blue = far"
            % (extraction.net_name, max_mm, field.nx, field.ny,
               field.pitch_nm / 1e6)))
        outer.Add(self._info, 0, wx.ALL, 10)

        self._cursor = wx.StaticText(self, label="Cursor: —")
        f = self._cursor.GetFont()
        f.SetWeight(wx.FONTWEIGHT_BOLD)
        self._cursor.SetFont(f)
        outer.Add(self._cursor, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        body = wx.BoxSizer(wx.HORIZONTAL)
        self._image_ctrl = _HeatmapPanel(self, (_VIEW_W, _VIEW_H),
                                         self._on_motion)
        body.Add(self._image_ctrl, 1, wx.EXPAND | wx.ALL, 8)

        right = wx.BoxSizer(wx.VERTICAL)
        right.Add(self._make_legend(max_mm, extraction.net_name), 0, wx.BOTTOM, 16)
        right.Add(wx.StaticText(self, label="Layers:"), 0, wx.BOTTOM, 4)

        self._checks = []   # (layer_index, wx.CheckBox)
        for li in range(field.num_layers):
            if not self._has_copper[li]:
                continue
            rowsz = wx.BoxSizer(wx.HORIZONTAL)
            colour = (extraction.layer_colors[li]
                      if li < len(extraction.layer_colors) else (180, 180, 180))
            swatch = wx.Panel(self, size=(32, 32))
            swatch.SetBackgroundColour(wx.Colour(*colour))
            rowsz.Add(swatch, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
            name = (extraction.layer_names[li]
                    if li < len(extraction.layer_names) else "Layer %d" % li)
            cb = wx.CheckBox(self, label=name)
            cb.SetValue(True)
            cb.Bind(wx.EVT_CHECKBOX, self._on_toggle)
            rowsz.Add(cb, 0, wx.ALIGN_CENTER_VERTICAL)
            right.Add(rowsz, 0, wx.BOTTOM, 6)
            self._checks.append((li, cb))

        body.Add(right, 0, wx.ALL, 8)
        outer.Add(body, 1, wx.EXPAND)

        btns = self.CreateButtonSizer(wx.OK)
        if btns:
            outer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        self.SetSizerAndFit(outer)

    def _make_legend(self, max_mm, net_name):
        """Legend as real widgets: title + net name + owner-drawn gradient bar
        flanked by far/mid/near distance labels."""
        box = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(self, label="Net distance")
        tf = title.GetFont()
        tf.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(tf)
        box.Add(title, 0, wx.BOTTOM, 2)
        box.Add(wx.StaticText(self, label=(net_name or "")[:26]), 0, wx.BOTTOM, 8)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(_GradientBar(self, size=(34, 212)), 0, wx.RIGHT, 8)
        labels = wx.BoxSizer(wx.VERTICAL)
        labels.Add(wx.StaticText(self, label="%.2f mm  (furthest)" % max_mm), 0)
        labels.AddStretchSpacer(1)
        labels.Add(wx.StaticText(self, label="%.2f mm" % (max_mm * 0.5)), 0)
        labels.AddStretchSpacer(1)
        labels.Add(wx.StaticText(self, label="0.00 mm  (selected)"), 0)
        row.Add(labels, 0, wx.EXPAND)
        box.Add(row, 0)
        return box

    # -- selection / compositing -------------------------------------------
    def _selected_layers(self):
        return [li for (li, cb) in self._checks if cb.GetValue()]

    def _composite_flat(self, layers):
        """Nearest distance per cell across ``layers`` (flat list, None=empty)."""
        n = self._nx * self._ny
        if HAS_NUMPY:
            best = _np.full(n, _np.inf, dtype=_np.float64)
            for li in layers:
                arr = self._layer_arr[li]
                best = _np.minimum(best, _np.where(arr >= 0.0, arr, _np.inf))
            return [None if v == _np.inf else float(v) for v in best]
        best = [None] * n
        for li in layers:
            for i, v in enumerate(self._layer_arr[li]):
                if v is not None and (best[i] is None or v < best[i]):
                    best[i] = v
        return best

    def _rebuild(self, *_):
        layers = self._selected_layers()
        if not layers:
            self._image_ctrl.set_image(None, message="No layers selected")
            self._cur_flat = None
            return

        if HAS_NUMPY and len(layers) == 1:
            arr = self._layer_arr[layers[0]]
            img = _image_from_np(arr, self._nx, self._ny, self._maxd,
                                 self._alpha_val, self._lut)
            self._cur_flat = arr
        else:
            flat = self._composite_flat(layers)
            img = _image_from_flat(flat, self._nx, self._ny, self._maxd,
                                   self._opacity)
            self._cur_flat = flat

        if img is None:
            self._image_ctrl.set_image(None, message="No copper on these layers")
            return
        self._image_ctrl.set_image(img, self._seed_shapes)

    _on_toggle = _rebuild

    # -- hover --------------------------------------------------------------
    def _value_at(self, idx):
        v = self._cur_flat[idx]
        if HAS_NUMPY and not isinstance(self._cur_flat, list):
            return None if v < 0.0 else float(v)
        return v

    def _on_motion(self, pos):
        if pos is None or self._cur_flat is None:
            self._cursor.SetLabel("Cursor: —")
            return
        cell = self._image_ctrl.cell_at(pos)
        if cell is None:
            self._cursor.SetLabel("Cursor: —")
            return
        ix, iy = cell
        v = self._value_at(iy * self._nx + ix)
        if v is None:
            self._cursor.SetLabel("Cursor: (no copper here)")
        else:
            self._cursor.SetLabel("Cursor: %.2f mm from selection" % (v / 1e6))


def remove_previous(board):
    """Delete any leftover overlay group from earlier on-canvas attempts.

    Older versions placed reference images grouped under ``GROUP_NAME``; this
    cleans them up so the board doesn't accumulate dead groups. Returns count.
    """
    removed = 0
    for grp in list(board.Groups()):
        if grp.GetName() == GROUP_NAME:
            for item in list(grp.GetItems()):
                board.Remove(item)
                removed += 1
            board.Remove(grp)
    return removed


def _seed_shapes_grid(field, extraction):
    """Convert the selection's copper shapes (nm) to image-grid coordinates."""
    ox, oy = field.origin
    pitch = float(field.pitch_nm)

    def g(x, y):
        return ((x - ox) / pitch, (y - oy) / pitch)

    out = []
    for shape in getattr(extraction, "seed_shapes", []):
        if shape[0] == "disc":
            _, cx, cy, r = shape
            gx, gy = g(cx, cy)
            out.append(("disc", gx, gy, r / pitch))
        elif shape[0] == "poly":
            out.append(("poly", [g(x, y) for (x, y) in shape[1]]))
    return out


def render(field, extraction, opacity=0.6, parent=None):
    """Show the heatmap in a modal dialog. Returns a dict with summary info."""
    max_mm = field.max_distance_nm / 1e6
    any_copper = any(any(v is not None for v in field.dist[li])
                     for li in range(field.num_layers))
    if not any_copper:
        wx.MessageBox("No copper to display for this net.", "Net Isoplot",
                      wx.OK | wx.ICON_WARNING)
        return {"layers_rendered": 0, "max_distance_mm": max_mm,
                "grid": (field.nx, field.ny)}

    dlg = _OverlayDialog(parent, field, extraction, opacity,
                         _seed_shapes_grid(field, extraction))
    try:
        dlg.ShowModal()
    finally:
        dlg.Destroy()

    rendered = sum(1 for li in range(field.num_layers)
                   if any(v is not None for v in field.dist[li]))
    return {
        "layers_rendered": rendered,
        "max_distance_mm": max_mm,
        "grid": (field.nx, field.ny),
    }
