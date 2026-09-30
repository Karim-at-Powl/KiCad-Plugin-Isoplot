"""Net Isoplot - KiCad action plugin.

Select one pad or via, click the toolbar button: the plugin computes the
geodesic (along-copper) distance from the selection to every reachable point on
the same net and draws a colour heatmap overlay (blue = near, red = far) with a
legend. Useful for spotting the furthest points on a net to find/shorten loops.
"""

from __future__ import annotations

import os
import traceback

import wx
import pcbnew

from . import net_geometry
from . import distance_field
from . import render_overlay

MM = 1_000_000  # nm per mm

# Default grid pitch. Finer when NumPy is available (rasterisation is cheap).
PITCH_NUMPY_NM = int(0.12 * MM)
PITCH_PURE_NM = int(0.20 * MM)
# Safety cap on grid cells per layer to keep the pure-Python solve responsive.
MAX_CELLS = 3_000_000


def _info(msg, title="Net Isoplot", style=wx.OK | wx.ICON_INFORMATION):
    dlg = wx.MessageDialog(None, msg, title, style)
    dlg.ShowModal()
    dlg.Destroy()


def _find_selected_seed(board):
    """Return a single selected PAD or PCB_VIA, or (None, reason)."""
    try:
        sel = list(pcbnew.GetCurrentSelection())
    except Exception:
        sel = []
    seeds = []
    for item in sel:
        cls = item.GetClass()
        if cls == "PAD" or cls in ("PCB_VIA", "VIA"):
            seeds.append(item)
    if len(seeds) == 0:
        return None, ("Select exactly one pad or via, then run the plugin.\n"
                      "(Nothing suitable is currently selected.)")
    if len(seeds) > 1:
        return None, "Select only ONE pad or via (found %d)." % len(seeds)
    return seeds[0], None


def _choose_pitch(prims):
    base = PITCH_NUMPY_NM if distance_field.HAS_NUMPY else PITCH_PURE_NM
    xmin, ymin, xmax, ymax = prims.bbox
    w = max(1, xmax - xmin)
    h = max(1, ymax - ymin)
    pitch = base
    # Enlarge pitch if the grid would exceed the cell cap.
    while (w / pitch + 2) * (h / pitch + 2) > MAX_CELLS:
        pitch = int(pitch * 1.5)
    return pitch


class NetIsoplotPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Net Isoplot (distance heatmap)"
        self.category = "Analysis"
        self.description = ("Heatmap of along-copper distance from a selected "
                            "pad/via to the rest of its net")
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")
        self.dark_icon_file_name = self.icon_file_name

    def Run(self):
        try:
            self._run()
        except Exception:
            _info("Net Isoplot failed:\n\n%s" % traceback.format_exc(),
                  style=wx.OK | wx.ICON_ERROR)

    def _run(self):
        board = pcbnew.GetBoard()
        seed, reason = _find_selected_seed(board)
        if seed is None:
            _info(reason, style=wx.OK | wx.ICON_WARNING)
            return

        net_code = seed.GetNetCode()
        if net_code <= 0:
            _info("The selected item is not assigned to a net.",
                  style=wx.OK | wx.ICON_WARNING)
            return

        busy = wx.BusyInfo("Net Isoplot: extracting net geometry...")
        try:
            extraction = net_geometry.extract(board, net_code, seed)
        finally:
            del busy
        if extraction is None or extraction.prims.num_layers == 0:
            _info("No copper found on the selected net.",
                  style=wx.OK | wx.ICON_WARNING)
            return

        prims = extraction.prims
        pitch = _choose_pitch(prims)

        busy = wx.BusyInfo("Net Isoplot: computing distance field "
                           "(grid pitch %.3f mm)..." % (pitch / MM))
        try:
            field = distance_field.solve(prims, pitch_nm=pitch,
                                         margin_nm=pitch * 2)
        finally:
            del busy

        if field.max_distance_nm <= 0.0:
            _info("The net has only the selected point of copper "
                  "(nothing to measure).\n\n"
                  "If the net clearly has more copper, make sure its zones are "
                  "filled (press 'B') and try again.",
                  style=wx.OK | wx.ICON_INFORMATION)
            return

        # Clean up any stale on-canvas group left by earlier plugin versions.
        try:
            render_overlay.remove_previous(board)
        except Exception:
            pass

        render_overlay.render(field, extraction)
