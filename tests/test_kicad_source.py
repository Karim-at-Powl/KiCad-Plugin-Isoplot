"""Offline tests for kicad_source: real kipy item types, fake board API.

Needs the kicad-python package (``kipy``) importable, but no running KiCad.
Run:  python tests/test_kicad_source.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kipy.board_types import BoardArc, BoardCircle, BoardSegment, Pad, Track, Via, Zone
from kipy.errors import ApiError
from kipy.geometry import PolygonWithHoles, Vector2
from kipy.proto.board import board_types_pb2 as bt
from kipy.proto.board.board_types_pb2 import BoardLayer
from kipy.proto.common import ApiStatusCode
from kipy.proto.common.types import base_types_pb2 as base

import distance_field as df
import kicad_source as ks

MM = 1_000_000
F, IN1, B = ks._F_CU, ks._IN1_CU, ks._B_CU


def _rect_poly(cx, cy, w, h):
    p = base.PolygonWithHoles()
    for (x, y) in ((cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                   (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)):
        node = p.outline.nodes.add()
        node.point.x_nm, node.point.y_nm = int(x), int(y)
    p.outline.closed = True
    return p


def track(uid, net, layer, x0, y0, x1, y1, w=250_000):
    t = bt.Track()
    t.id.value = uid
    t.net.name = net
    t.layer = layer
    t.start.x_nm, t.start.y_nm, t.end.x_nm, t.end.y_nm = x0, y0, x1, y1
    t.width.value_nm = w
    return Track(proto=t)


def via(uid, net, x, y, dia=600_000):
    v = bt.Via()
    v.id.value = uid
    v.net.name = net
    v.position.x_nm, v.position.y_nm = x, y
    v.type = bt.ViaType.VT_THROUGH
    v.pad_stack.type = bt.PadStackType.PST_NORMAL
    v.pad_stack.layers.extend([F, B])
    cl = v.pad_stack.copper_layers.add()
    cl.layer = F
    cl.size.x_nm = cl.size.y_nm = dia
    return Via(proto=v)


def smd_pad(uid, net, x, y):
    p = bt.Pad()
    p.id.value = uid
    p.net.name = net
    p.position.x_nm, p.position.y_nm = x, y
    p.type = bt.PadType.PT_SMD
    p.pad_stack.layers.extend([F])
    return Pad(proto=p)


def zone(uid, net, layer, rect, filled=True):
    z = bt.Zone()
    z.id.value = uid
    z.type = bt.ZoneType.ZT_COPPER
    z.layers.extend([layer])
    z.copper_settings.net.name = net
    z.filled = filled
    if filled:
        fp = z.filled_polygons.add()
        fp.layer = layer
        fp.shapes.polygons.add().CopyFrom(_rect_poly(*rect))
    return Zone(proto=z)


class FakeBoard:
    name = "fake.kicad_pcb"

    def __init__(self, items, selection=()):
        self.items = items
        self.selection = selection
        self.calls = []

    def get_enabled_layers(self):
        return [F, B, 40, 41]  # two copper layers + some non-copper ids

    def get_items(self, types):
        return list(self.items)

    def get_selection(self, types):
        return [i for i in self.items if i.id.value in self.selection]

    def check_padstack_presence_on_layers(self, items, layers):
        out = {}
        for i in items:
            on = {F, B} if isinstance(i, Via) else {F}
            out[i] = {lid: lid in on for lid in layers}
        return out

    def get_pad_shapes_as_polygons(self, pads, layer):
        self.calls.append(("pad_shapes", layer))
        single = isinstance(pads, Pad)
        pads = [pads] if single else pads
        shapes = [PolygonWithHoles(proto=_rect_poly(p.position.x, p.position.y, 2 * MM, 1 * MM))
                  for p in pads if layer == F]
        if single:
            return shapes[0] if shapes else None
        return shapes

    def get_layer_name(self, lid):
        raise ApiError("unhandled", code=ApiStatusCode.AS_UNHANDLED)


def board_fixture():
    """Pad on F -> 20 mm track -> via -> 10 mm track on B + a pour on B.
    Plus a track on another net and an unfilled zone."""
    items = [
        smd_pad("pad1", "SIG", 0, 0),
        track("t1", "SIG", F, 0, 0, 20 * MM, 0),
        via("v1", "SIG", 20 * MM, 0),
        track("t2", "SIG", B, 20 * MM, 0, 20 * MM, 10 * MM),
        zone("z1", "SIG", B, (20 * MM, 12 * MM, 4 * MM, 4 * MM)),
        zone("z2", "SIG", B, (0, 0, 1 * MM, 1 * MM), filled=False),
        track("other", "GND", F, 0, 5 * MM, 30 * MM, 5 * MM),
    ]
    return FakeBoard(items, selection=("pad1",))


def test_selection_and_fetch():
    board = board_fixture()
    reader = ks.BoardReader(board)
    seeds = reader.selected_seed_ids()
    assert seeds == ("pad1",), seeds
    g = reader.fetch(seeds)
    p = g.prims
    assert g.net_name == "SIG"
    assert g.layer_names == ["F.Cu", "B.Cu"], g.layer_names
    assert p.num_layers == 2
    assert len(p.segments) == 2, "other-net track leaked in: %r" % (p.segments,)
    assert len(p.vias) == 1 and p.vias[0][2] == [0, 1]
    assert len(p.discs) == 2 and all(d[3] == 300_000 for d in p.discs)
    assert len(p.seed_polys) == 1 and p.seed_polys[0][0] == 0
    assert g.unfilled_zones == 1
    # zone fill + pad polygon on F
    assert sum(1 for q in p.polys if q[0] == 1) == 1
    assert sum(1 for q in p.polys if q[0] == 0) == 1
    # unchanged board -> no new geometry
    assert reader.fetch(seeds, g.fingerprint) is None
    # moving a track changes the fingerprint
    board.items[1].proto.end.x_nm = 21 * MM
    assert reader.fetch(seeds, g.fingerprint) is not None

    f = df.solve(p, int(0.1 * MM))
    far = f.max_distance_nm / MM
    # pad edge (1 mm) -> 20 mm -> via -> ~14 mm down to the far pour edge
    assert 30.0 < far < 36.0, far


def test_seed_errors():
    board = board_fixture()
    reader = ks.BoardReader(board)
    try:
        reader.fetch(("gone",))
    except ks.SeedError:
        pass
    else:
        raise AssertionError("missing seed not reported")
    board.items.append(smd_pad("nc", "", 0, 0))
    try:
        reader.fetch(("nc",))
    except ks.SeedError:
        pass
    else:
        raise AssertionError("unconnected seed not reported")


class FakeBoardV10(FakeBoard):
    """KiCad 10.0.1+: per-item and per-net queries; the whole-board read
    must not be used."""

    def __init__(self, items, selection=(), refuse=None):
        super().__init__(items, selection)
        self.refuse = refuse     # ApiStatusCode to answer the new queries with

    def get_items(self, types):
        self.calls.append("get_items")
        return super().get_items(types)

    def get_items_by_id(self, ids):
        self.calls.append("get_items_by_id")
        if self.refuse is not None:
            raise ApiError("refused", code=self.refuse)
        wanted = {k.value for k in ids}
        return [i for i in self.items if i.id.value in wanted]

    def get_items_by_net(self, net, types):
        self.calls.append("get_items_by_net")
        return [i for i in self.items if ks._net_name(i) == net.name]


def _same_geometry(a, b):
    for attr in ("num_layers", "bbox", "segments", "discs", "vias", "sources",
                 "seed_discs", "seed_polys", "polys"):
        assert getattr(a.prims, attr) == getattr(b.prims, attr), attr
    assert (a.net_name, a.layer_names, a.seed_count, a.unfilled_zones) == \
        (b.net_name, b.layer_names, b.seed_count, b.unfilled_zones)


def test_per_net_query_matches_whole_board_read():
    """KiCad 10: only the seed net is fetched, and the result is the same."""
    old = ks.BoardReader(board_fixture()).fetch(("pad1",))
    board = FakeBoardV10(board_fixture().items, selection=("pad1",))
    reader = ks.BoardReader(board)
    new = reader.fetch(("pad1",))
    _same_geometry(old, new)
    assert "get_items" not in board.calls and "get_items_by_net" in board.calls
    assert reader.fetch(("pad1",), new.fingerprint) is None      # unchanged board


def test_falls_back_on_kicad_9():
    """KiCad 9 answers the new queries with AS_UNHANDLED: whole-board read,
    same result, and the new queries are not tried again."""
    old = ks.BoardReader(board_fixture()).fetch(("pad1",))
    board = FakeBoardV10(board_fixture().items, selection=("pad1",),
                         refuse=ApiStatusCode.AS_UNHANDLED)
    reader = ks.BoardReader(board)
    _same_geometry(old, reader.fetch(("pad1",)))
    reader.fetch(("pad1",))
    assert board.calls.count("get_items_by_id") == 1, board.calls
    assert board.calls.count("get_items") == 2, board.calls


def test_features_from_version():
    f9 = ks.KiCadFeatures((9, 0, 5))
    assert f9.padstack_presence and not f9.layer_names and not f9.net_queries
    assert ks.KiCadFeatures((9, 0, 8)).layer_names
    assert not ks.KiCadFeatures((10, 0, 0)).net_queries
    f10 = ks.KiCadFeatures((10, 0, 1))
    assert f10.net_queries and f10.layer_names
    unknown = ks.KiCadFeatures()
    assert unknown.net_queries and unknown.layer_names, "unknown version: try everything"
    assert "9.0.5" in str(f9)


def test_version_decides_before_any_request():
    """On KiCad 9 the per-net query and layer-name calls are never sent, and
    the result equals the whole-board read."""
    board = FakeBoardV10(board_fixture().items, selection=("pad1",))
    board.get_layer_name = lambda lid: (_ for _ in ()).throw(AssertionError("layer name asked"))
    reader = ks.BoardReader(board, ks.KiCadFeatures((9, 0, 5)))
    _same_geometry(ks.BoardReader(board_fixture()).fetch(("pad1",)), reader.fetch(("pad1",)))
    assert "get_items_by_id" not in board.calls and "get_items_by_net" not in board.calls
    # ... and on 10.0.1 the per-net query is used straight away.
    board = FakeBoardV10(board_fixture().items, selection=("pad1",))
    ks.BoardReader(board, ks.KiCadFeatures((10, 0, 1))).fetch(("pad1",))
    assert "get_items" not in board.calls and "get_items_by_net" in board.calls


def test_per_net_query_busy_and_seed_errors():
    board = FakeBoardV10(board_fixture().items, refuse=ApiStatusCode.AS_BUSY)
    reader = ks.BoardReader(board)
    try:
        reader.fetch(("pad1",))
    except ApiError as e:
        assert ks.is_busy(e)
    else:
        raise AssertionError("busy KiCad not reported")
    assert reader._have_net_queries, "a busy KiCad must not disable the per-net query"

    reader = ks.BoardReader(FakeBoardV10(board_fixture().items))
    for seeds in (("gone",), ("nc",)):
        reader.board.items.append(smd_pad("nc", "", 0, 0))
        try:
            reader.fetch(seeds)
        except ks.SeedError:
            pass
        else:
            raise AssertionError("seed error not reported for %r" % (seeds,))


def test_via_diameter_front_inner_back():
    v = via("v", "N", 0, 0)
    ps = v.proto.pad_stack
    ps.type = bt.PadStackType.PST_FRONT_INNER_BACK
    for lid, d in ((IN1, 400_000), (B, 500_000)):
        cl = ps.copper_layers.add()
        cl.layer = lid
        cl.size.x_nm = d
    assert ks._via_diameter(v, F) == 600_000
    assert ks._via_diameter(v, ks._COPPER_ORDER[5]) == 400_000
    assert ks._via_diameter(v, B) == 500_000


def _edge_segment(x0, y0, x1, y1):
    s = BoardSegment()
    s.layer = BoardLayer.BL_Edge_Cuts
    s.start, s.end = Vector2.from_xy(x0, y0), Vector2.from_xy(x1, y1)
    return s


def _edge_arc(start, mid, end):
    a = BoardArc()
    a.layer = BoardLayer.BL_Edge_Cuts
    a.start, a.mid, a.end = (Vector2.from_xy(*p) for p in (start, mid, end))
    return a


def test_board_outline_chains_segments_and_arcs():
    """A 20 x 10 mm board with one rounded corner, drawn as separate segments
    in mixed directions, plus a round cut-out and a silkscreen line."""
    r = 2 * MM
    c = int(r * (1 - 1 / math.sqrt(2)))
    shapes = [
        _edge_segment(0, 0, 20 * MM - r, 0),
        _edge_arc((20 * MM - r, 0), (20 * MM - c, c), (20 * MM, r)),
        _edge_segment(20 * MM, 10 * MM, 20 * MM, r),        # reversed
        _edge_segment(20 * MM, 10 * MM, 0, 10 * MM),
        _edge_segment(0, 0, 0, 10 * MM),                     # reversed
    ]
    hole = BoardCircle()
    hole.layer = BoardLayer.BL_Edge_Cuts
    hole.center, hole.radius_point = Vector2.from_xy(5 * MM, 5 * MM), Vector2.from_xy(6 * MM, 5 * MM)
    silk = _edge_segment(0, 0, MM, MM)
    silk.layer = BoardLayer.BL_F_SilkS
    board = board_fixture()
    board.get_shapes = lambda: shapes + [hole, silk]
    fp, rings, chains = ks.BoardReader(board).board_outline()
    assert len(rings) == 2 and not chains, (len(rings), len(chains))
    outer = max(rings, key=lambda ring: max(p[0] for p in ring) - min(p[0] for p in ring))
    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    assert [round(v) for v in (min(xs), min(ys), max(xs), max(ys))] == [0, 0, 20 * MM, 10 * MM]
    # moving an edge changes the fingerprint
    shapes[3].end = Vector2.from_xy(0, 11 * MM)
    assert ks.BoardReader(board).board_outline()[0] != fp


def test_chain_leaves_open_pieces_open():
    rings, chains = ks._chain([[(0, 0), (10, 0)], [(10, 0), (10, 10)]], tol=0)
    assert not rings and len(chains) == 1 and len(chains[0]) == 3


def test_sample_arc_both_directions():
    r = 10 * MM
    for mid in ((r / math.sqrt(2), r / math.sqrt(2)), (-r / math.sqrt(2), -r / math.sqrt(2))):
        pts = ks._sample_arc((r, 0), mid, (0, r), 1 * MM)
        assert all(abs(math.hypot(x, y) - r) < 1 for x, y in pts)
        assert pts[0] == (r, 0.0) or math.dist(pts[0], (r, 0)) < 1
        assert math.dist(pts[-1], (0, r)) < 1
        # the sampled path passes near the requested mid point
        assert min(math.dist(p, mid) for p in pts) < 1 * MM
    # quarter arc (ccw) is short, three-quarter arc (cw) is long
    short = ks._sample_arc((r, 0), (r / math.sqrt(2), r / math.sqrt(2)), (0, r), 1 * MM)
    long_ = ks._sample_arc((r, 0), (-r / math.sqrt(2), -r / math.sqrt(2)), (0, r), 1 * MM)
    assert len(long_) > 2 * len(short)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("OK: %d tests passed" % len(tests))
