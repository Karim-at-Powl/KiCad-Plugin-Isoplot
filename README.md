# Net Isoplot

A KiCad plugin that shows a **live distance heatmap** of a net. Select a pad or
via in the PCB editor, press the toolbar button, and a window opens that
colours every point of that net by its distance *measured along the copper*
(traces, arcs, vias, pads and filled pours, across all layers joined by vias
and plated holes). It runs red (hot) at the selection, cools through green, and
ends in blue at the furthest points. It's a quick visual for spotting long
return paths and loops.

The window stays open beside KiCad and keeps up with your work:

- **Select another pad or via** and it switches to that one. Untick *Follow
  selection* to pin the current seed while you click around.
- **Edit the board** (move, route, delete, undo, refill zones) and it
  recomputes. A coarse result appears within a moment and is then refined.
- Select **several pads/vias on one net** to measure from all of them at once.
- The view is framed by the **board outline** (Edge.Cuts), so it doesn't jump
  between nets. The mouse wheel zooms, dragging pans, and a double-click shows
  the whole board. The outline is read once. After editing it, press **Update
  Board Outline**.
- The cursor readout gives the distance and board coordinates under the mouse.

While an interactive tool is active in KiCad (routing, placing a via,
dragging...), KiCad answers every API request with "busy". The window says so
and catches up as soon as you leave the tool (Esc).

## Requirements

- KiCad **9.0.5 or newer** (IPC API plugin; built with KiCad 11 in mind, where
  the old SWIG Python API is removed).
- The KiCad API must be enabled: **Preferences → Plugins → Enable KiCad API**,
  then restart KiCad.
- On first use KiCad creates a Python environment for the plugin and installs
  `requirements.txt` (`kicad-python`) into it, which needs an internet
  connection once. wxPython and NumPy come from KiCad's bundled Python.

## Install (development)

The plugin lives in KiCad's 3rd-party plugins directory, which KiCad scans for
`plugin.json`:

```
<KiCad documents>/9.0/3rdparty/plugins/Powl_NetIsoplot/
```

Restart KiCad. A "Net Isoplot" button appears in the PCB editor's toolbar.
Pressing it while the window is already open brings the window forward and
switches it to the current selection.

## How it works

Each module has a single job, and only one of them talks to KiCad:

| File | Role |
| --- | --- |
| `isoplot.py` | Entry point KiCad launches (`plugin.json`). Single-instance handling, logging, starts the window. |
| `live.py` | Background threads. The **poller** reads the selection 4×/s and re-reads the seed net about once a second. The IPC API has no change events, so a fingerprint of the net's items skips unchanged boards. The **solver** runs a coarse pass, then a fine one; a newer job cancels the current one. |
| `kicad_source.py` | The only module that uses the KiCad API (`kipy`). It fetches the net's tracks, arcs, vias, pads (exact per-layer polygons from KiCad) and filled zones, and honours unconnected-layer removal. Output is plain primitives in nm. |
| `distance_field.py` | KiCad-free core. Rasterises copper onto a multi-layer grid (NumPy) and runs a multi-source Dijkstra from the seed copper. |
| `viewer.py` | The wxPython window: heatmap, legend, layer toggles, cursor readout. |

> **Why a separate window and not an on-canvas overlay?** Neither KiCad API
> (SWIG in v9, IPC in v9–v11) can put pixel data onto the board canvas, so the
> heatmap is drawn in its own window.

## Tests

```
python tests/test_distance_field.py     # no dependencies beyond NumPy
python tests/test_kicad_source.py       # needs kicad-python (no KiCad required)
```

## Troubleshooting

The plugin runs without a console. Its log is written to `net_isoplot.log` in
the system temp directory (`%TEMP%` on Windows). If the window says *Waiting
for KiCad...*, check that the API is enabled. If a pour looks missing, fill the
zones (press **B**). Unfilled zones are reported in the status bar.

## Notes / limitations

- Distance is an **8-connected grid geodesic**: slightly long, by up to about 8%
  on diagonal-ish paths. Grid pitch adapts to the net's size: 0.05 mm on
  small nets, coarser on large pours.
- Distances are measured from the edge of the selected copper, and via/hole
  transitions count as zero length.
- Zone copper is only as current as the last zone fill.
