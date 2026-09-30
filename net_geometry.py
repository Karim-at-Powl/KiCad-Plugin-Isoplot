"""Extract one net's copper from a pcbnew board into plain primitives.

This is the only module that depends on ``pcbnew``. It converts tracks, arcs,
vias, pads and filled zone polygons of a single net into a
``distance_field.NetPrimitives`` instance (all coordinates in nm), and records a
dense layer-index mapping plus human-readable layer names for the legend/render.

All shape approximations are deliberately conservative (slightly oversized) so
the rasterised copper stays connected on a coarse grid.
"""

from __future__ import annotations

import math

import pcbnew

from . import distance_field as df


class NetExtraction:
    def __init__(self, prims, layer_ids, layer_names, layer_colors, net_name):
        self.prims = prims              # distance_field.NetPrimitives
        self.layer_ids = layer_ids      # dense index -> pcbnew layer id
        self.layer_names = layer_names  # dense index -> str
        self.layer_colors = layer_colors  # dense index -> (r, g, b)
        self.net_name = net_name
        self.seed_shapes = []           # copper of the selection, in nm


# KiCad "KiCad Default" theme copper-layer colours (the built-in default theme).
# The COLOR_SETTINGS object is not exposed to Python, so we mirror the shipped
# palette here, keyed by canonical layer key, to draw layer swatches.
_COPPER_RGB = {
    "f": (200, 52, 52), "b": (77, 127, 196),
    "in1": (127, 200, 127), "in2": (206, 125, 44), "in3": (79, 203, 203),
    "in4": (219, 98, 139), "in5": (167, 165, 198), "in6": (40, 204, 217),
    "in7": (232, 178, 167), "in8": (242, 237, 161), "in9": (141, 203, 129),
    "in10": (237, 124, 51), "in11": (91, 195, 235), "in12": (247, 111, 142),
    "in13": (167, 165, 198), "in14": (40, 204, 217), "in15": (232, 178, 167),
    "in16": (242, 237, 161), "in17": (237, 124, 51), "in18": (91, 195, 235),
    "in19": (247, 111, 142), "in20": (167, 165, 198), "in21": (40, 204, 217),
    "in22": (232, 178, 167), "in23": (242, 237, 161), "in24": (237, 124, 51),
    "in25": (91, 195, 235), "in26": (247, 111, 142), "in27": (167, 165, 198),
    "in28": (40, 204, 217), "in29": (232, 178, 167), "in30": (242, 237, 161),
}


def _build_layer_keymap():
    km = {}
    if hasattr(pcbnew, "F_Cu"):
        km[pcbnew.F_Cu] = "f"
    if hasattr(pcbnew, "B_Cu"):
        km[pcbnew.B_Cu] = "b"
    for i in range(1, 31):
        cid = getattr(pcbnew, "In%d_Cu" % i, None)
        if cid is not None:
            km[cid] = "in%d" % i
    return km


_LAYER_KEYMAP = _build_layer_keymap()


def _layer_color(lid):
    return _COPPER_RGB.get(_LAYER_KEYMAP.get(lid), (180, 180, 180))


def _pt(v):
    return (int(v.x), int(v.y))


def _angle_deg(pad):
    a = pad.GetOrientation()
    if hasattr(a, "AsDegrees"):
        return a.AsDegrees()
    try:
        return float(a) / 10.0  # legacy tenths-of-degree
    except TypeError:
        return 0.0


def _sample_arc(start, mid, end, max_seg_nm):
    """Return a polyline (list of (x,y)) approximating an arc through 3 points."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return [start, end]
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) +
          (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) +
          (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)

    def norm(a):
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return a

    # Decide sweep direction so that the mid angle lies between start and end.
    span_ccw = norm(a3 - a1)
    mid_ccw = norm(a2 - a1)
    if mid_ccw <= span_ccw:
        sweep = span_ccw          # counter-clockwise
    else:
        sweep = -(2 * math.pi - span_ccw)  # clockwise
    n = max(2, int(abs(sweep) * r / max(1.0, max_seg_nm)) + 1)
    pts = []
    for i in range(n + 1):
        a = a1 + sweep * (i / n)
        pts.append((ux + r * math.cos(a), uy + r * math.sin(a)))
    return pts


def extract(board, net_code, selected_item):
    """Build NetPrimitives for ``net_code``. ``selected_item`` is the seed pad/via."""
    cu_stack = list(board.GetEnabledLayers().CuStack())
    stack_pos = {lid: i for i, lid in enumerate(cu_stack)}

    # Raw accumulation with native layer ids; remapped to dense indices later.
    raw_segments = []   # (lid, x0, y0, x1, y1, width)
    raw_discs = []      # (lid, cx, cy, r)
    raw_polys = []      # (lid, [(x,y)...])
    raw_vias = []       # (cx, cy, [lid...])
    raw_sources = []    # (lid, x, y)
    used = set()

    def use(lid):
        if lid in stack_pos:
            used.add(lid)

    def span_layers(top_lid, bot_lid):
        if top_lid not in stack_pos or bot_lid not in stack_pos:
            return [l for l in (top_lid, bot_lid) if l in stack_pos]
        a, b = stack_pos[top_lid], stack_pos[bot_lid]
        if a > b:
            a, b = b, a
        return cu_stack[a:b + 1]

    # --- tracks / arcs / vias ------------------------------------------------
    for t in board.GetTracks():
        if t.GetNetCode() != net_code:
            continue
        cls = t.GetClass()
        if cls in ("PCB_VIA", "VIA"):
            pos = _pt(t.GetPosition())
            r = t.GetWidth() / 2.0
            layers = [l for l in span_layers(t.TopLayer(), t.BottomLayer())]
            for lid in layers:
                use(lid)
                raw_discs.append((lid, pos[0], pos[1], r))
            if len(layers) >= 2:
                raw_vias.append((pos[0], pos[1], layers))
        elif cls in ("PCB_ARC", "ARC"):
            lid = t.GetLayer()
            use(lid)
            w = t.GetWidth()
            start = _pt(t.GetStart())
            end = _pt(t.GetEnd())
            mid = _pt(t.GetMid()) if hasattr(t, "GetMid") else None
            pts = _sample_arc(start, mid, end, 300_000) if mid else [start, end]
            for i in range(len(pts) - 1):
                ax, ay = pts[i]
                bx, by = pts[i + 1]
                raw_segments.append((lid, ax, ay, bx, by, w))
        else:  # PCB_TRACK
            lid = t.GetLayer()
            use(lid)
            s = _pt(t.GetStart())
            e = _pt(t.GetEnd())
            raw_segments.append((lid, s[0], s[1], e[0], e[1], t.GetWidth()))

    # --- pads ---------------------------------------------------------------
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetCode() != net_code:
                continue
            pos = _pt(pad.GetPosition())
            size = pad.GetSize()
            sx, sy = int(size.x), int(size.y)
            pad_layers = [lid for lid in cu_stack if pad.IsOnLayer(lid)]
            shape = pad.GetShape()
            is_circle = (shape == getattr(pcbnew, "PAD_SHAPE_CIRCLE", -99))
            for lid in pad_layers:
                use(lid)
                if is_circle:
                    raw_discs.append((lid, pos[0], pos[1], max(sx, sy) / 2.0))
                else:
                    raw_polys.append((lid, _oriented_rect(pos, sx, sy,
                                                          _angle_deg(pad))))
            # Through-hole / multi-layer pad bridges its copper layers.
            if len(pad_layers) >= 2:
                raw_vias.append((pos[0], pos[1], pad_layers))

    # --- filled zones -------------------------------------------------------
    for zone in board.Zones():
        if zone.GetNetCode() != net_code:
            continue
        if not zone.IsFilled():
            continue
        if hasattr(zone, "GetIsRuleArea") and zone.GetIsRuleArea():
            continue
        for lid in zone.GetLayerSet().CuStack():
            try:
                ps = zone.GetFilledPolysList(lid)
            except TypeError:
                ps = zone.GetFilledPolysList()
            if ps is None or ps.OutlineCount() == 0:
                continue
            use(lid)
            for i in range(ps.OutlineCount()):
                chain = ps.Outline(i)
                pts = [( int(chain.CPoint(j).x), int(chain.CPoint(j).y) )
                       for j in range(chain.PointCount())]
                if len(pts) >= 3:
                    raw_polys.append((lid, pts))

    # --- seed (selected item) ----------------------------------------------
    # seed_shapes describes the actual copper of the selection (in nm) so the
    # renderer can draw it at true size/shape; seed positions feed the solver.
    seed_shapes = []
    sel_cls = selected_item.GetClass()
    if sel_cls in ("PCB_VIA", "VIA"):
        pos = _pt(selected_item.GetPosition())
        layers = span_layers(selected_item.TopLayer(), selected_item.BottomLayer())
        for lid in layers:
            use(lid)
            raw_sources.append((lid, pos[0], pos[1]))
        seed_shapes.append(("disc", pos[0], pos[1],
                            selected_item.GetWidth() / 2.0))
    else:  # PAD
        pos = _pt(selected_item.GetPosition())
        pad_layers = [lid for lid in cu_stack if selected_item.IsOnLayer(lid)]
        for lid in pad_layers:
            use(lid)
            raw_sources.append((lid, pos[0], pos[1]))
        size = selected_item.GetSize()
        sx, sy = int(size.x), int(size.y)
        if selected_item.GetShape() == getattr(pcbnew, "PAD_SHAPE_CIRCLE", -99):
            seed_shapes.append(("disc", pos[0], pos[1], max(sx, sy) / 2.0))
        else:
            seed_shapes.append(("poly", _oriented_rect(pos, sx, sy,
                                                        _angle_deg(selected_item))))

    if not used:
        return None

    # --- dense remap --------------------------------------------------------
    layer_ids = sorted(used, key=lambda l: stack_pos[l])
    idx = {lid: i for i, lid in enumerate(layer_ids)}
    layer_names = [board.GetLayerName(lid) for lid in layer_ids]
    layer_colors = [_layer_color(lid) for lid in layer_ids]

    prims = df.NetPrimitives()
    prims.num_layers = len(layer_ids)
    prims.segments = [(idx[l], a, b, c, d, w) for (l, a, b, c, d, w) in raw_segments
                      if l in idx]
    prims.discs = [(idx[l], x, y, r) for (l, x, y, r) in raw_discs if l in idx]
    prims.polys = [(idx[l], pts) for (l, pts) in raw_polys if l in idx]
    prims.sources = [(idx[l], x, y) for (l, x, y) in raw_sources if l in idx]
    prims.vias = []
    for (x, y, layers) in raw_vias:
        mapped = [idx[l] for l in layers if l in idx]
        if len(mapped) >= 2:
            prims.vias.append((x, y, mapped))

    prims.bbox = _bbox(prims)
    net_name = board.GetNetInfo().GetNetItem(net_code).GetNetname()
    ext = NetExtraction(prims, layer_ids, layer_names, layer_colors, net_name)
    ext.seed_shapes = seed_shapes
    return ext


def _oriented_rect(center, sx, sy, angle_deg):
    cx, cy = center
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hx, hy = sx / 2.0, sy / 2.0
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca) for dx, dy in corners]


def _bbox(prims):
    xs = []
    ys = []
    for (_, x0, y0, x1, y1, w) in prims.segments:
        hw = w / 2.0
        xs += [x0 - hw, x0 + hw, x1 - hw, x1 + hw]
        ys += [y0 - hw, y0 + hw, y1 - hw, y1 + hw]
    for (_, x, y, r) in prims.discs:
        xs += [x - r, x + r]
        ys += [y - r, y + r]
    for (_, pts) in prims.polys:
        for (x, y) in pts:
            xs.append(x)
            ys.append(y)
    for (_, x, y) in prims.sources:
        xs.append(x)
        ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))
