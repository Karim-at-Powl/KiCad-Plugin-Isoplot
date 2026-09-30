"""Read one net's copper from a running KiCad through the IPC API.

This is the only module that talks to KiCad (through ``kipy``, the
kicad-python package). It resolves the seed pads/vias, fetches the net's
tracks, arcs, vias, pads and filled zones, and converts them into a
``distance_field.NetPrimitives`` (all coordinates in nm) plus the layer names
and colours the viewer needs.

Pads use KiCad's own polygon for each layer, so every pad shape (round rect,
oval, trapezoid, chamfered, custom) and per-layer padstacks come out exact.
Vias and pads only count on the layers where KiCad actually flashes copper
(unconnected-layer removal is honoured).
"""

from __future__ import annotations

import hashlib
import math

from kipy.board_types import (ArcTrack, BoardArc, BoardBezier, BoardCircle, BoardPolygon,
                              BoardRectangle, BoardSegment, Pad, Track, Via, Zone)
from kipy.errors import ApiError
from kipy.proto.board.board_types_pb2 import BoardLayer, PadStackType, ZoneType
from kipy.proto.common import ApiStatusCode
from kipy.proto.common.types import KiCadObjectType
from kipy.util.board_layer import canonical_name, iter_copper_layers

import distance_field as df

_COPPER_ORDER = list(iter_copper_layers())          # F.Cu, In1.Cu .. In30.Cu, B.Cu
_STACK_POS = {lid: i for i, lid in enumerate(_COPPER_ORDER)}
_F_CU, _IN1_CU, _B_CU = _COPPER_ORDER[0], _COPPER_ORDER[1], _COPPER_ORDER[-1]

_NET_ITEM_TYPES = [KiCadObjectType.KOT_PCB_TRACE, KiCadObjectType.KOT_PCB_ARC,
                   KiCadObjectType.KOT_PCB_VIA, KiCadObjectType.KOT_PCB_PAD,
                   KiCadObjectType.KOT_PCB_ZONE]
_SEED_TYPES = [KiCadObjectType.KOT_PCB_PAD, KiCadObjectType.KOT_PCB_VIA]
_COPPER_ZONES = (ZoneType.ZT_COPPER, ZoneType.ZT_TEARDROP)

_ARC_SEG_NM = 100_000   # max chord length when sampling arcs

# KiCad "KiCad Default" theme copper-layer colours, for the layer swatches (the
# API does not expose the colour theme).
_INNER_RGB = [(127, 200, 127), (206, 125, 44), (79, 203, 203), (219, 98, 139),
              (167, 165, 198), (40, 204, 217), (232, 178, 167), (242, 237, 161),
              (141, 203, 129), (237, 124, 51), (91, 195, 235), (247, 111, 142)]
_COPPER_RGB = {"F.Cu": (200, 52, 52), "B.Cu": (77, 127, 196)}
_COPPER_RGB.update({"In%d.Cu" % i: _INNER_RGB[(i - 1) % len(_INNER_RGB)]
                    for i in range(1, 31)})


class SeedError(Exception):
    """The seed pads/vias no longer exist or are not on a net."""


def is_busy(exc):
    """True if ``exc`` means "KiCad is busy (e.g. mid-drag), try again later"."""
    return isinstance(exc, ApiError) and exc.code in (ApiStatusCode.AS_BUSY,
                                                      ApiStatusCode.AS_NOT_READY)


class NetGeometry:
    """Everything the solver and viewer need about one net (plain data)."""

    def __init__(self, prims, layer_names, layer_colors, net_name, fingerprint,
                 seed_count, unfilled_zones):
        self.prims = prims                  # distance_field.NetPrimitives
        self.layer_names = layer_names      # dense index -> str
        self.layer_colors = layer_colors    # dense index -> (r, g, b)
        self.net_name = net_name
        self.fingerprint = fingerprint
        self.seed_count = seed_count
        self.unfilled_zones = unfilled_zones


class BoardReader:
    """Reads seeds and net copper from one open board."""

    def __init__(self, board):
        self.board = board
        self._have_layer_names = True

    @property
    def name(self):
        return self.board.name

    def selected_seed_ids(self):
        """IDs of the pads/vias currently selected in the PCB editor."""
        items = self.board.get_selection(_SEED_TYPES)
        return tuple(sorted(i.id.value for i in items if isinstance(i, (Pad, Via))))

    def board_outline(self):
        """The board's Edge.Cuts as (fingerprint, closed rings, open chains).

        Board-level shapes only; Edge.Cuts drawn inside footprints (e.g. slots)
        are not included.
        """
        shapes = [s for s in self.board.get_shapes() if s.layer == BoardLayer.BL_Edge_Cuts]
        h = hashlib.blake2b(digest_size=16)
        for s in shapes:
            h.update(s.proto.SerializeToString(deterministic=True))
        rings, chains = _chain([pl for s in shapes for pl in _shape_polylines(s)])
        return h.hexdigest(), rings, chains

    def fetch(self, seed_ids, previous_fingerprint=None):
        """Return the NetGeometry for the net of ``seed_ids``, or None if the
        net's copper is unchanged since ``previous_fingerprint``."""
        copper = sorted((lid for lid in self.board.get_enabled_layers()
                         if lid in _STACK_POS), key=_STACK_POS.get)
        items = self.board.get_items(_NET_ITEM_TYPES)

        by_id = {i.id.value: i for i in items if isinstance(i, (Pad, Via))}
        seeds = [by_id[s] for s in seed_ids if s in by_id]
        if not seeds:
            raise SeedError("The selected pad/via no longer exists.")
        net = seeds[0].net.name
        if not net:
            raise SeedError("The selected pad/via is not connected to a net.")
        seeds = [s for s in seeds if s.net.name == net]
        net_items = [i for i in items if _net_name(i) == net]

        fingerprint = _fingerprint(net_items, seeds, copper)
        if fingerprint == previous_fingerprint:
            return None
        return self._convert(net, net_items, seeds, copper, fingerprint)

    # -- conversion ---------------------------------------------------------
    def _convert(self, net, items, seeds, copper, fingerprint):
        on_board = set(copper)
        padstacks = [i for i in items if isinstance(i, (Pad, Via))]
        present = self._presence(padstacks, copper)
        pads = [i for i in items if isinstance(i, Pad)]

        segments, discs, polys, vias = [], [], [], []
        seed_discs, seed_polys = [], []
        unfilled = 0

        for item in items:
            if isinstance(item, Track):
                if item.layer in on_board:
                    segments.append((item.layer, item.start.x, item.start.y,
                                     item.end.x, item.end.y, item.width))
            elif isinstance(item, ArcTrack):
                if item.layer in on_board:
                    pts = _sample_arc((item.start.x, item.start.y),
                                      (item.mid.x, item.mid.y),
                                      (item.end.x, item.end.y), _ARC_SEG_NM)
                    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                        segments.append((item.layer, ax, ay, bx, by, item.width))
            elif isinstance(item, Via):
                layers = present.get(item.id.value, [])
                pos = item.position
                for lid in layers:
                    discs.append((lid, pos.x, pos.y, _via_diameter(item, lid) / 2.0))
                if len(layers) >= 2:
                    vias.append((pos.x, pos.y, layers))
            elif isinstance(item, Pad):
                layers = present.get(item.id.value, [])
                if len(layers) >= 2:  # plated hole bridges its copper layers
                    vias.append((item.position.x, item.position.y, layers))
            elif isinstance(item, Zone):
                if item.type not in _COPPER_ZONES:
                    continue
                if not item.filled:
                    unfilled += 1
                    continue
                for lid, shapes in item.filled_polygons.items():
                    if lid in on_board:
                        for shape in shapes:
                            polys.append((lid, _rings(shape)))

        for lid in copper:
            on_layer = [p for p in pads if lid in present.get(p.id.value, ())]
            if on_layer:
                for shape in self.board.get_pad_shapes_as_polygons(on_layer, lid):
                    polys.append((lid, _rings(shape)))

        for seed in seeds:
            for lid in present.get(seed.id.value, []):
                if isinstance(seed, Via):
                    seed_discs.append((lid, seed.position.x, seed.position.y,
                                       _via_diameter(seed, lid) / 2.0))
                else:
                    shape = self.board.get_pad_shapes_as_polygons(seed, lid)
                    if shape is not None:
                        seed_polys.append((lid, _rings(shape)))

        used = ({s[0] for s in segments} | {d[0] for d in discs}
                | {p[0] for p in polys} | {d[0] for d in seed_discs}
                | {p[0] for p in seed_polys})
        layer_ids = [lid for lid in copper if lid in used]
        idx = {lid: i for i, lid in enumerate(layer_ids)}

        prims = df.NetPrimitives()
        prims.num_layers = len(layer_ids)
        prims.segments = [(idx[s[0]],) + s[1:] for s in segments]
        prims.discs = [(idx[d[0]],) + d[1:] for d in discs]
        prims.polys = [(idx[p[0]], p[1]) for p in polys]
        prims.seed_discs = [(idx[d[0]],) + d[1:] for d in seed_discs]
        prims.seed_polys = [(idx[p[0]], p[1]) for p in seed_polys]
        prims.vias = []
        for (x, y, layers) in vias:
            mapped = [idx[lid] for lid in layers if lid in idx]
            if len(mapped) >= 2:
                prims.vias.append((x, y, mapped))
        prims.bbox = _bbox(prims)

        return NetGeometry(
            prims,
            [self._layer_name(lid) for lid in layer_ids],
            [_COPPER_RGB.get(canonical_name(lid), (180, 180, 180)) for lid in layer_ids],
            net, fingerprint, len(seeds), unfilled)

    def _presence(self, items, copper):
        """Map item id -> copper layers (stack order) where it has copper."""
        if not items:
            return {}
        try:
            found = self.board.check_padstack_presence_on_layers(items, copper)
            return {item.id.value: [lid for lid in copper if layers.get(lid)]
                    for item, layers in found.items()}
        except ApiError as e:
            if is_busy(e):
                raise
            # Older KiCad: fall back to the padstack's nominal layer set.
            return {i.id.value: [lid for lid in copper if lid in set(i.padstack.layers)]
                    for i in items}

    def _layer_name(self, lid):
        if self._have_layer_names:  # needs KiCad 9.0.8+ for user layer names
            try:
                return self.board.get_layer_name(lid)
            except ApiError as e:
                if is_busy(e):
                    raise
                self._have_layer_names = False
        return canonical_name(lid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _net_name(item):
    if isinstance(item, Zone):
        net = item.net
        return net.name if net is not None else None
    return item.net.name


def _fingerprint(items, seeds, copper):
    h = hashlib.blake2b(digest_size=16)
    h.update(bytes(str(copper), "ascii"))
    for s in seeds:
        h.update(s.id.value.encode())
    for item in items:
        h.update(item.proto.SerializeToString(deterministic=True))
    return h.hexdigest()


def _via_diameter(via, lid):
    ps = via.padstack
    sizes = {cl.layer: cl.size.x for cl in ps.copper_layers}
    if ps.type == PadStackType.PST_CUSTOM and lid in sizes:
        return sizes[lid]
    if ps.type == PadStackType.PST_FRONT_INNER_BACK:
        key = lid if lid in (_F_CU, _B_CU) else _IN1_CU
        if key in sizes:
            return sizes[key]
    return sizes.get(_F_CU) or next(iter(sizes.values()), 0)


def _rings(shape):
    """PolygonWithHoles -> [outline ring, hole ring, ...] as (x, y) lists."""
    proto = shape.proto
    return [_ring(proto.outline)] + [_ring(h) for h in proto.holes]


def _ring(polyline):
    pts = []
    for node in polyline.nodes:
        if node.HasField("point"):
            pts.append((node.point.x_nm, node.point.y_nm))
        elif node.HasField("arc"):
            a = node.arc
            pts.extend(_sample_arc((a.start.x_nm, a.start.y_nm),
                                   (a.mid.x_nm, a.mid.y_nm),
                                   (a.end.x_nm, a.end.y_nm), _ARC_SEG_NM))
    return pts


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
    tau = 2 * math.pi
    # Sweep direction: the one for which the mid angle lies between start/end.
    span_ccw = (a3 - a1) % tau
    if (a2 - a1) % tau <= span_ccw:
        sweep = span_ccw
    else:
        sweep = span_ccw - tau
    n = max(2, int(abs(sweep) * r / max(1.0, max_seg_nm)) + 1)
    return [(ux + r * math.cos(a1 + sweep * i / n), uy + r * math.sin(a1 + sweep * i / n))
            for i in range(n + 1)]


def _shape_polylines(shape):
    """A board graphic shape as a list of polylines ((x, y) lists, nm)."""
    if isinstance(shape, BoardSegment):
        return [[(shape.start.x, shape.start.y), (shape.end.x, shape.end.y)]]
    if isinstance(shape, BoardArc):
        return [_sample_arc((shape.start.x, shape.start.y), (shape.mid.x, shape.mid.y),
                            (shape.end.x, shape.end.y), _ARC_SEG_NM)]
    if isinstance(shape, BoardCircle):
        c, r = shape.center, shape.radius()
        n = 72
        return [[(c.x + r * math.cos(2 * math.pi * i / n), c.y + r * math.sin(2 * math.pi * i / n))
                 for i in range(n + 1)]]
    if isinstance(shape, BoardRectangle):
        (x0, y0), (x1, y1) = (shape.top_left.x, shape.top_left.y), \
            (shape.bottom_right.x, shape.bottom_right.y)
        return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
    if isinstance(shape, BoardPolygon):
        return [ring + ring[:1] for poly in shape.polygons for ring in _rings(poly) if ring]
    if isinstance(shape, BoardBezier):
        p = [(v.x, v.y) for v in (shape.start, shape.control1, shape.control2, shape.end)]
        n = 24
        pts = []
        for i in range(n + 1):
            t = i / n
            a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
            pts.append((a * p[0][0] + b * p[1][0] + c * p[2][0] + d * p[3][0],
                        a * p[0][1] + b * p[1][1] + c * p[2][1] + d * p[3][1]))
        return [pts]
    return []


def _chain(polylines, tol=1_000):
    """Join polylines that share endpoints (within ``tol`` nm).

    Returns (closed rings, open chains); outlines drawn as separate segments
    and arcs become closed rings the viewer can fill.
    """
    def close(a, b):
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

    pending = [list(pl) for pl in polylines if len(pl) >= 2]
    rings, chains = [], []
    while pending:
        chain = pending.pop()
        grown = True
        while grown and not (len(chain) > 3 and close(chain[0], chain[-1])):
            grown = False
            for i, pl in enumerate(pending):
                if close(chain[-1], pl[0]):
                    chain = chain + pl[1:]
                elif close(chain[-1], pl[-1]):
                    chain = chain + pl[-2::-1]
                elif close(chain[0], pl[-1]):
                    chain = pl[:-1] + chain
                elif close(chain[0], pl[0]):
                    chain = pl[:0:-1] + chain
                else:
                    continue
                pending.pop(i)
                grown = True
                break
        if len(chain) > 3 and close(chain[0], chain[-1]):
            rings.append(chain)
        else:
            chains.append(chain)
    return rings, chains


def _bbox(prims):
    xmin = ymin = math.inf
    xmax = ymax = -math.inf

    def grow(x0, y0, x1, y1):
        nonlocal xmin, ymin, xmax, ymax
        xmin = min(xmin, x0)
        ymin = min(ymin, y0)
        xmax = max(xmax, x1)
        ymax = max(ymax, y1)

    for (_, x0, y0, x1, y1, w) in prims.segments:
        hw = w / 2.0
        grow(min(x0, x1) - hw, min(y0, y1) - hw, max(x0, x1) + hw, max(y0, y1) + hw)
    for (_, x, y, r) in prims.discs + prims.seed_discs:
        grow(x - r, y - r, x + r, y + r)
    for (_, rings) in prims.polys + prims.seed_polys:
        for ring in rings:
            if ring:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                grow(min(xs), min(ys), max(xs), max(ys))
    if xmin == math.inf:
        return (0, 0, 0, 0)
    return (xmin, ymin, xmax, ymax)
