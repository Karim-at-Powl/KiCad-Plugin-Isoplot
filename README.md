# Net Isoplot

A KiCad 9 action plugin. Select a pad or via, click the toolbar button, and the
plugin draws a **distance heatmap** over the selected net: red (hot) at the
selection, cooling through green to blue at the points furthest away *measured
along the copper* (traces, vias and filled pours), across all layers connected
through vias. Intended as a first-aid visual for spotting long return paths /
loops.

## How it works

1. **`net_geometry.py`** reads the selected net's copper from the board via the
   `pcbnew` SWIG API (tracks, arcs, vias, pads, filled zone polygons) and emits
   plain numeric primitives (nm), plus a dense copper-layer index map.
2. **`distance_field.py`** (no `pcbnew` dependency) rasterises that copper onto a
   multi-layer grid and runs a multi-source Dijkstra from the selected
   pad/via, giving the geodesic distance to every reachable copper cell. Vias
   bridge layers. NumPy accelerates rasterisation when present; the algorithm is
   identical without it.
3. **`render_overlay.py`** turns the distance field into RGBA heatmap images
   (built with bundled wxPython - no PIL/matplotlib) and shows them in a dialog
   with: a per-layer toggle list (each layer named, with its KiCad layer colour
   as a swatch, all on by default), a colour legend, the selected pad/via drawn
   filled black at its true size and shape, and a live readout of the
   along-copper distance under the cursor. With several layers enabled the view
   shows the per-cell *nearest* distance across them; untick layers to inspect
   one at a time.

   > **Why a dialog and not an on-canvas overlay?** KiCad 9's Python API does not
   > expose any way to load pixel data onto the board: the `REFERENCE_IMAGE` /
   > `BITMAP_BASE` object returned by `PCB_REFERENCE_IMAGE.GetReferenceImage()`
   > is not SWIG-wrapped, so there is no `ReadImageFile`/`SetImage` to call.
   > Placing the heatmap as a `PCB_REFERENCE_IMAGE` is therefore not possible in
   > v9 scripting; the dialog is the supported alternative.

## Dependencies

**None required.** Runs on KiCad's bundled Python + wxPython out of the box, so
it is plug-and-play via the Plugin & Content Manager. If **NumPy** happens to be
installed in KiCad's Python, it is used automatically for a finer default grid
and faster rasterisation.

## Install (development)

Copy this folder into your KiCad 3rd-party plugins directory (it already is, if
you are reading this there):

```
<KiCad documents>/9.0/3rdparty/plugins/Powl_NetIsoplot/
```

Then in the PCB editor: **Tools -> External Plugins -> Refresh Plugins**. A
toolbar button "Net Isoplot" appears.

## Use

1. In the PCB editor, click a single pad or via that belongs to a net.
2. Run **Net Isoplot** from the toolbar / External Plugins menu.
3. Read the heatmap dialog: red = close, blue = furthest along the copper; the
   solid black shape is the selected pad/via. Tick/untick layers in the list to
   isolate them, and move the cursor over the heatmap to read the distance at
   that point. The legend shows the 0 -> furthest scale in mm.

## Notes / limitations (v0.1)

- Distance is an **8-connected grid geodesic** (good Euclidean approximation,
  slightly conservative). Grid pitch defaults to 0.12 mm (NumPy) / 0.20 mm
  (pure Python) and is automatically coarsened for very large nets.
- Pad shapes are approximated (circle -> disc, others -> oriented rectangle).
  Adequate for connectivity/distance; not a DRC-grade copper model.
- The heatmap is shown in a **modal dialog**, not painted on the board canvas,
  because KiCad 9's Python API has no way to push an image onto a layer (see
  "How it works" above). The window is resizable and the heatmap scales to fit
  it; cell pitch (not the on-screen size) bounds the spatial accuracy.
