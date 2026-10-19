#!/usr/bin/env python3
"""
parse_scene.py -- Project MIKHAIL ".ma.scn" (EPK) scene-graph, skeleton, bind-pose
and per-vertex skinning parser.

    python parse_scene.py <file.scn | asset-path-in-pack.bin> [--obj out.obj]
                          [--json out.json] [--tree N] [--full]

WHAT THE FORMAT IS
------------------
An `.ma.scn` payload is an `EPK\0` container followed by a *tag stream* -- a
self-describing serialisation in which every item is

    <u8 tag> <payload>

Tag layout:  tag >= 0x80  ->  tag & 0x7f is a type/opcode id
             tag <  0x80  ->  the byte itself is the small unsigned integer

Established opcodes (all verified against f22a_00 / type94_jp_00 / bl74_00):

    0x03  class instance: <0x03> <k> then <u32 crc32(className) LE>  (k==3)
                          or <0x03> <2> then a NUL-terminated class name
    0x06 / 0x26   NUL-terminated ASCII string
    0x07  field key:  <0x87> <u32 crc32(fieldName) LE>          <-- the key insight
    0x0a  u16 scalar        0x0b  u16 array element
    0x0c  u32               0x0f  8-bit float (0x00->0.0, 0x38->1.0)
    0x10  IEEE half (f16)   0x11  IEEE float (f32)
    0x13  array: <0x93> <u8 count>  <elemTag> <data>
    0x14  array: <0x94> <u16 count> <elemTag> <data>
    0x96+n (n<=9)  array of n fixed-size elements, <elemTag> follows
    0xa7  array: <0xa7> <u8 count> then `count` self-describing elements
    0xc0+n array of n self-describing elements
    0x84  no-payload marker (present in front of several values; ignored)
    0x85  end-of-object marker
    0xff  null

Field names and class names are NOT stored as text at each use site: they are
identified by the little-endian CRC32 of their name.  That was proven blind --
crc32("localTranslation")=0x339d6090 appears in the stream as 90 60 9d 33, and
24/500 probe combinations matched exactly.

60-BYTE VERTEX RECORD (verified statistically, see --verify output)
    0x00 f32 x3   position            (bbox 1277 x 1900 x 1071, Y-up)
    0x0c f32      == 1.0
    0x10 f16 x3   normal              (|n| in [0.99,1.01] for 99.90%)
    0x16 u16      0 (padding)
    0x18 f16 x2   UV0
    0x1c f16 x2   UV1
    0x20 f16 x2   UV2                 (== UV0 for meshes with only 2 UV sets)
    0x24 f16 x2   UV3                 (== UV1 likewise)
    0x28 i8  x3   tangent   (value/127; |t|~1 for 99.96%)
    0x2b u8       0
    0x2c i8  x3   bitangent (value/127; |b|~1 for 99.88%)
    0x2f u8       0
    0x30 f16 x4   skin weights        (sum == 1.0 for 100.000% of vertices)
    0x38 u8  x4   skin joint indices  (index into THIS mesh's skeleton.nodes)

The vertex buffer is a concatenation of the 50 meshes' vertices in mesh order
(SUM(vertexCount) == 230981 == vertex buffer length exactly).  Each mesh has a
`skeletonIndex`; that skeleton carries `nodes` (u16 joint uids) and `bindPose`
(one 4x4 f32 matrix per joint).  In all 40 meshes whose skeleton parsed, the
largest joint index used equals len(nodes)-1 exactly.
"""
import argparse
import collections
import json
import math
import os
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# --------------------------------------------------------------------------- #
# tag stream primitives
# --------------------------------------------------------------------------- #
# Integer ladder 0x09/0x0a/0x0b mirrors the float ladder 0x0f/0x10/0x11:
# 0x89 = u8 (verified: nodeIndex 128..255 are stored as 0x89 <byte>), 0x8a = u16
# (uid/parentUid = 0x8a <u16>), 0x8b = u16 array element, 0x95 = u32
# (0x95 <u32> is the IndexBuffer element count).  0x8d is a *one byte* marker:
# the index data starts immediately after it (proved by triangle degeneracy,
# 0.13% reading from 0xc1c vs 50.0% reading from 0xc20).
ELEM = {0x09: ("B", 1), 0x0a: ("H", 2), 0x0b: ("H", 2), 0x0c: ("I", 4),
        0x0f: ("B", 1), 0x10: ("e", 2), 0x11: ("f", 4), 0x15: ("I", 4)}
ELEMNAME = {0x09: "u8", 0x0a: "u16", 0x0b: "u16", 0x0c: "u32",
            0x0f: "f8", 0x10: "f16", 0x11: "f32", 0x15: "u32"}

# 0x96+n = array of n fixed-size elements, element-type tag follows.
SMALL_ARRAY_MAX = 32


class Trunc(Exception):
    pass


def f8(b):
    """8-bit float.  0x00 -> 0.0 and 0x38 -> 1.0 (nodeAlpha / localScaling are
    semantically 1.0 everywhere).  E4M3 (sign, 4-bit exponent bias 7, 3-bit
    mantissa) satisfies that exactly; the assumption is flagged in the report."""
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 0x07
    v = (m / 8.0) * 2 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def crc_table(d):
    """crc32 -> name, for every identifier stored anywhere in the file."""
    strs = set()
    p = 0x10
    while p < len(d):                                  # sorted name pool
        e = d.find(b"\x00", p)
        if e == -1 or e == p:
            break
        s = d[p:e]
        if not all(32 <= c < 127 for c in s) or len(s) > 200:
            break
        strs.add(s.decode()); p = e + 1
    i = 0
    while i < len(d):                                  # inline strings
        if 32 <= d[i] < 127:
            j = i
            while j < len(d) and 32 <= d[j] < 127:
                j += 1
            if j < len(d) and d[j] == 0 and 3 <= (j - i) <= 120:
                strs.add(d[i:j].decode("ascii"))
            i = j + 1
        else:
            i += 1
    tab = {}
    for s in strs:
        tab.setdefault(zlib.crc32(s.encode()) & 0xffffffff, s)
    return tab


def read_value(buf, i, crc, depth=0):
    """One self-describing item -> (kind, value, next_index)."""
    if i >= len(buf):
        raise Trunc()
    t = buf[i]
    if t < 0x80:
        return "int", t, i + 1
    if t == 0xff:
        return "null", None, i + 1
    tid = t & 0x7F
    if tid == 0x07:
        if i + 5 > len(buf):
            raise Trunc()
        k = int.from_bytes(buf[i + 1:i + 5], "little")
        return "key", crc.get(k, "crc:%08x" % k), i + 5
    if tid in (0x06, 0x26):
        e = buf.find(b"\x00", i + 1)
        if e == -1:
            raise Trunc()
        return "str", buf[i + 1:e].decode("ascii", "replace"), e + 1
    if tid == 0x03:
        sub = buf[i + 1]
        if sub == 3:
            k = int.from_bytes(buf[i + 2:i + 6], "little")
            return "class", crc.get(k, "crc:%08x" % k), i + 6
        if sub == 2:
            e = buf.find(b"\x00", i + 2)
            if e == -1:
                raise Trunc()
            return "classnew", buf[i + 2:e].decode("ascii", "replace"), e + 1
        return "classsub", sub, i + 2

    def fixed(n, et):
        if et < 0x80 or (et & 0x7F) not in ELEM:
            raise Trunc()
        eid = et & 0x7F
        fmt, sz = ELEM[eid]
        if i + 3 + n * sz > len(buf):
            raise Trunc()
        raw = buf[i + 3:i + 3 + n * sz]
        if fmt == "B":
            vals = [f8(x) if eid == 0x0F else x for x in raw]
        else:
            vals = list(struct.unpack("<%d%s" % (n, fmt), raw))
        return "list", (ELEMNAME[eid], vals), i + 3 + n * sz

    if tid == 0x13:                                    # 93 <u8 count> <etype>
        return fixed(buf[i + 1], buf[i + 2])
    if tid == 0x14:                                    # 94 <u16 count> <etype>
        n = struct.unpack_from("<H", buf, i + 1)[0]
        return fixed(n, buf[i + 3])
    if (0x16 <= tid <= 0x16 + SMALL_ARRAY_MAX and i + 1 < len(buf)
            and (buf[i + 1] & 0x80) and (buf[i + 1] & 0x7F) in ELEM):
        n = tid - 0x16                                 # 96+n : n fixed elems
        eid = buf[i + 1] & 0x7F
        fmt, sz = ELEM[eid]
        if i + 2 + n * sz > len(buf):
            raise Trunc()
        raw = buf[i + 2:i + 2 + n * sz]
        if fmt == "B":
            vals = [f8(x) if eid == 0x0F else x for x in raw]
        else:
            vals = list(struct.unpack("<%d%s" % (n, fmt), raw))
        return "list", (ELEMNAME[eid], vals), i + 2 + n * sz
    if tid in ELEM:                                    # scalar number
        fmt, sz = ELEM[tid]
        if i + 1 + sz > len(buf):
            raise Trunc()
        if fmt == "B":
            return "num", (ELEMNAME[tid], f8(buf[i + 1]) if tid == 0x0F else buf[i + 1]), i + 2
        v = struct.unpack_from("<" + fmt, buf, i + 1)[0]
        return "num", (ELEMNAME[tid], float(v) if fmt in "ef" else int(v)), i + 1 + sz
    if tid == 0x27 or tid >= 0x40:                     # arrays of objects
        if depth > 6:
            raise Trunc()
        if tid == 0x27:
            n = buf[i + 1]; j = i + 2
        else:
            n = tid - 0x40; j = i + 1
        out = []
        for _ in range(n):
            k, v, j = read_value(buf, j, crc, depth + 1)
            out.append((k, v))
        return "objs", out, j
    return "mark", tid, i + 1


def parse_items(buf, start, crc):
    items, i = [], start
    while i < len(buf):
        try:
            kind, val, nxt = read_value(buf, i, crc)
        except Trunc:
            items.append(("trunc", None, i))
            break
        items.append((kind, val, i))
        i = nxt
    return items


def group_fields(items):
    out = collections.OrderedDict()
    cur = None
    for kind, val, off in items:
        if kind == "key":
            cur = val
            out.setdefault(cur, [])
        elif kind in ("class", "classnew", "classsub", "trunc"):
            continue
        else:
            out.setdefault(cur, []).append((kind, val, off))
    return out


def gv(fields, name, default=None):
    for kind, val, _ in fields.get(name, ()):
        if kind == "list":
            return list(val[1])
        if kind == "num":
            return val[1]
        if kind == "int":
            return val
        if kind == "str":
            return val
    return default


def gvec(fields, name):
    v = gv(fields, name)
    return list(v) if isinstance(v, list) else None


# --------------------------------------------------------------------------- #
# structural locators
# --------------------------------------------------------------------------- #
def key_offsets(d, name):
    pat = struct.pack("<I", zlib.crc32(name.encode()) & 0xffffffff)
    out, p = [], 0
    while True:
        q = d.find(pat, p)
        if q == -1:
            return out
        out.append(q - 1)              # points at the 0x87 tag byte
        p = q + 1


def find_vertex_blob(d):
    """`vertices` -> VertexBuffer { stride, data: <0xac> <u32 len> <blob> }"""
    for off in key_offsets(d, "vertices"):
        seg = d[off:off + 96]
        di = seg.find(struct.pack("<I", zlib.crc32(b"data") & 0xffffffff))
        if di < 0:
            continue
        j = off + di + 4
        while j < len(d) and d[j] == 0x84:
            j += 1
        if d[j] != 0xAC:
            continue
        ln = struct.unpack_from("<I", d, j + 1)[0]
        return j + 5, ln
    return None, None


def index_driven_vertex_blob(d, nv):
    """Locate the LOD0 vertex blob from the index count.

    Every vertex record has f32 1.0 at +0x0C.  A real vertex blob is therefore a
    chain of positions spaced exactly `stride` apart.  `nv = max(index)+1` tells
    us how many records the blob must hold, so we can pick the chain whose length
    matches instead of guessing the offset.  F-14 / MiG-21 / YF-23 Pav-1 store
    their vertices as per-mesh blobs (several chains), so the old
    "longest run wins" heuristic spliced LODs together and produced garbage.

    Returns (offset, stride, count) or (None, None, None).
    """
    best = None
    for stride in (60, 64, 48, 52, 56, 68, 72):
        u = np.frombuffer(d[:len(d) // 4 * 4], dtype="<u4")
        ones = set((np.flatnonzero(u == 0x3F800000) * 4).tolist())
        used = set()
        for cand in sorted(ones):
            if cand in used or (cand - stride) in ones:
                continue
            n = 1
            while (cand + n * stride) in ones:
                n += 1
            for k in range(n):
                used.add(cand + k * stride)
            if n == nv:
                return cand - 0xC, stride, n
            if n > nv and (best is None or n < best[2]):
                best = (cand - 0xC, stride, n)
    return best if best else (None, None, None)


def find_index_blob(d, nverts, crc):
    """`indices` -> IndexBuffer { ... <array header> <index data> }.

    The writer emits `0x95 <u32 count> <end tag>` where the end tag selects the
    element width: `0x8d` = u32, `0x8b` = u16.  F-16 and most TSFs use u32;
    F-14 / MiG-21 / YF-23 Pav-1 use u16 (their LOD0 fits in 16 bits), and the
    old code only accepted 0x8d, which is why those three had "no faces".
    The `0x94 <u16 count> <etype>` and `0x93 <u8 count> <etype>` forms are used
    by other asset classes.
    """
    for off in key_offsets(d, "indices"):
        for i in range(off + 1, min(off + 48, len(d) - 8)):
            t = d[i]
            if t == 0x95:                               # <u32 count> <end tag>
                cnt = struct.unpack_from("<I", d, i + 1)[0]
                end = d[i + 5]
                if end == 0x8D and 0 < cnt < 200_000_000 and cnt % 3 == 0:
                    return i + 6, cnt, "u32"
                if end == 0x8B and 0 < cnt < 200_000_000 and cnt % 3 == 0 \
                        and i + 6 + cnt * 2 <= len(d):
                    return i + 6, cnt, "u16"
            elif t == 0x94:                             # <u16 count> <etype>
                cnt = struct.unpack_from("<H", d, i + 1)[0]
                et = d[i + 3] & 0x7F
                if cnt > 2 and cnt % 3 == 0 and et in ELEM:
                    return i + 4, cnt, ELEMNAME[et]
            elif t == 0x93:                             # <u8 count> <etype>
                cnt, et = d[i + 1], d[i + 2] & 0x7F
                if cnt > 2 and cnt % 3 == 0 and et in ELEM:
                    return i + 3, cnt, ELEMNAME[et]
    return None, None, None


def node_markers(d, limit):
    magic = b"\x83\x03" + struct.pack("<I", zlib.crc32(b"Node") & 0xffffffff)
    out, p = [], 0
    while True:
        q = d.find(magic, p)
        if q == -1 or q > limit:
            return out
        out.append(q); p = q + 1


def split_by_keys(d, lo, hi, crc):
    out, i, last = [], lo, None
    while i < hi:
        if d[i] == 0x87 and i + 5 <= hi:
            k = int.from_bytes(d[i + 1:i + 5], "little")
            if k in crc:
                if last is not None:
                    out.append((last[0], last[1], d[last[1] + 5:i]))
                last = (crc[k], i); i += 5; continue
        i += 1
    if last is not None:
        out.append((last[0], last[1], d[last[1] + 5:hi]))
    return out


def small_value(p):
    if not p:
        return None
    i = 0
    while i < len(p) and p[i] == 0x84:
        i += 1
    if i >= len(p):
        return None
    t = p[i]
    if t < 0x80:
        return t
    tid = t & 0x7F
    try:
        if tid in (0x0A, 0x0B) and i + 3 <= len(p):
            return int.from_bytes(p[i + 1:i + 3], "little")
        if tid == 0x0C and i + 5 <= len(p):
            return int.from_bytes(p[i + 1:i + 5], "little")
        if tid == 0x0F and i + 2 <= len(p):
            return f8(p[i + 1])
        if tid == 0x10 and i + 3 <= len(p):
            return float(struct.unpack_from("<e", p, i + 1)[0])
        if tid == 0x11 and i + 5 <= len(p):
            return float(struct.unpack_from("<f", p, i + 1)[0])
    except struct.error:
        return None
    return None


def parse_float_array(p):
    """<84>* (0x93 <u8 n> | 0x94 <u16 n>) <f32 tag> <n floats>"""
    i = 0
    while i < len(p) and p[i] == 0x84:
        i += 1
    if i >= len(p):
        return None
    tid = p[i] & 0x7F
    if tid == 0x13:
        n, off = p[i + 1], i + 3
    elif tid == 0x14:
        n, off = struct.unpack_from("<H", p, i + 1)[0], i + 4
    else:
        return None
    if off + 4 * n > len(p) or (p[off - 1] & 0x7F) != 0x11:
        return None
    return list(struct.unpack_from("<%df" % n, p, off))


def parse_int_array(p):
    """Integer array with any count form and any element width.

    Element widths seen: 0x09 u8 (bl74_00 skeleton `nodes` = one u8 uid),
    0x0a/0x0b u16 (mech `children` / `nodes`), 0x11 u32.
    """
    i = 0
    while i < len(p) and p[i] == 0x84:
        i += 1
    if i >= len(p):
        return []
    tid = p[i] & 0x7F
    if 0x16 <= tid <= 0x16 + SMALL_ARRAY_MAX:
        n, off = tid - 0x16, i + 2
    elif tid == 0x13:
        n, off = p[i + 1], i + 3
    elif tid == 0x14:
        n, off = struct.unpack_from("<H", p, i + 1)[0], i + 4
    else:
        return []
    if off > len(p):
        return []
    et = p[off - 1] & 0x7F
    fmt, sz = ELEM.get(et, (None, 0))
    if not sz or off + n * sz > len(p):
        return []
    return list(struct.unpack_from("<%d%s" % (n, fmt), p, off))


# --------------------------------------------------------------------------- #
# main parser
# --------------------------------------------------------------------------- #
class Scene:
    def __init__(self, data, name="<mem>"):
        self.d = data
        self.name = name
        self.crc = crc_table(data)
        self.pool = self._pool()
        self.notes = []
        self._parse()

    def _pool(self):
        out, p = [], 0x10
        while p < len(self.d):
            e = self.d.find(b"\x00", p)
            if e == -1 or e == p:
                break
            s = self.d[p:e]
            if not all(32 <= c < 127 for c in s) or len(s) > 200:
                break
            out.append(s.decode()); p = e + 1
        self.pool_end = p
        return out

    # ---------------- nodes ----------------
    def _parse_nodes(self):
        d = self.d
        voff, vlen = find_vertex_blob(d)
        if voff is None:                       # fallback: 60-byte stride run scan
            hits = []
            i = 0
            while True:
                j = d.find(b"\x00\x00\x80\x3f", i)
                if j == -1:
                    break
                hits.append(j); i = j + 1
            hits = np.array(hits)
            runs, st = [], 0
            for k in range(1, len(hits)):
                if hits[k] - hits[k - 1] != 60:
                    if k - st > 1000:
                        runs.append((st, k - 1))
                    st = k
            if len(hits) - st > 1000:
                runs.append((st, len(hits) - 1))
            if runs:
                s, e = max(runs, key=lambda r: r[1] - r[0])
                voff, vlen = int(hits[s]) - 0xC, (e - s + 1) * 60
            else:
                # Collision proxies (the box_* meshes) carry almost no vertex
                # data, so no 1000-record run exists.  There is nothing to
                # export for them; leave an empty vertex range instead of
                # aborting the whole unit.
                voff, vlen = 0, 0
                self.notes.append("no vertex run found (collision proxy?)")
            self.notes.append("vertex blob located by 60-byte stride scan fallback")
        self.voff, self.vlen = voff, vlen
        self.vstride = 60
        self.vcount = vlen // 60

        self.marks = node_markers(d, voff)
        self.nodes = []
        for k in range(len(self.marks)):
            a = self.marks[k]
            b = self.marks[k + 1] if k + 1 < len(self.marks) else min(voff, a + 4096)
            f = group_fields(parse_items(d[a:b], 8, self.crc))
            pv = f.get("parentUid", ())
            isnull = any(kk == "null" for kk, _, _ in pv)
            self.nodes.append(dict(
                name=gv(f, "name"), uid=gv(f, "uid"),
                parentUid=None if isnull else gv(f, "parentUid", -1),
                nodeIndex=gv(f, "nodeIndex"),
                T=gvec(f, "localTranslation") or [0.0, 0.0, 0.0],
                R=gvec(f, "localRotation") or [0.0, 0.0, 0.0],
                S=gvec(f, "localScaling") or [1.0, 1.0, 1.0],
                JO=gvec(f, "jointOrient") or [0.0, 0.0, 0.0],
                children=gvec(f, "children") or [],
                jointGroup=gv(f, "jointGroup"),
                fields=f))
        byuid = {n["uid"]: k for k, n in enumerate(self.nodes)}
        self.parent = [None if n["parentUid"] is None else byuid.get(n["parentUid"])
                       for n in self.nodes]
        self.byuid = byuid

    # ---------------- geometry / skin ----------------
    def _parse_geometry(self):
        d = self.d
        # The index buffer is authoritative (nv = max(index)+1), and the vertex
        # blob must be a chain of records spaced exactly `stride` apart.  The
        # `vertices` field is not always the LOD0 blob -- F-14 / MiG-21 /
        # YF-23 Pav-1 keep per-mesh blobs, so the field-derived extent covers
        # several LODs at once and splices unrelated meshes together.
        iv = getattr(self, "ivcount", None)
        if iv:
            o, st, n = index_driven_vertex_blob(d, iv)
            if o is not None and n == iv:
                if (o, st) != (self.voff, self.vstride):
                    self.notes.append(
                        f"vertex blob re-located from indices: "
                        f"0x{self.voff:x}/{self.vcount}@stride{self.vstride} -> "
                        f"0x{o:x}/{n}@stride{st}")
                self.voff, self.vlen = o, n * st
                self.vstride = st
                self.vcount = n
        self.vraw = np.frombuffer(
            d[self.voff:self.voff + self.vcount * self.vstride],
            dtype=np.uint8).reshape(self.vcount, self.vstride)
        self.P = self.vraw[:, 0:12].copy().view("<f4").reshape(self.vcount, 3).astype(np.float64)
        self.Nrm = np.stack([self.vraw[:, o:o + 2].copy().view("<f2").reshape(-1).astype(np.float64)
                             for o in (0x10, 0x12, 0x14)], 1)
        self.UV = [np.stack([self.vraw[:, o:o + 2].copy().view("<f2").reshape(-1).astype(np.float64),
                             self.vraw[:, o + 2:o + 4].copy().view("<f2").reshape(-1).astype(np.float64)], 1)
                   for o in (0x18, 0x1C, 0x20, 0x24)]
        self.TAN = self.vraw[:, 0x28:0x2B].copy().view("<i1").astype(np.float64) / 127.0
        self.BIT = self.vraw[:, 0x2C:0x2F].copy().view("<i1").astype(np.float64) / 127.0
        self.W = np.stack([self.vraw[:, o:o + 2].copy().view("<f2").reshape(-1).astype(np.float64)
                           for o in (0x30, 0x32, 0x34, 0x36)], 1)
        self.J = self.vraw[:, 0x38:0x3C].astype(np.int64)
        ioff, icnt, itype = find_index_blob(d, self.vcount, self.crc)
        if ioff is not None and self.idx is None:
            dt = {"u8": "<u1", "u16": "<u2", "u32": "<u4"}[itype]
            self.ioff, self.icnt, self.itype = ioff, icnt, itype
            self.idx = np.frombuffer(d[ioff:ioff + icnt * np.dtype(dt).itemsize],
                                     dtype=dt).astype(np.int64)
            self.tris = self.idx.reshape(-1, 3)
        elif ioff is None and self.idx is None:
            self.notes.append("index buffer not located structurally")

    def _parse_meshes(self):
        d = self.d
        self.meshes, self.skeletons = [], []
        mo = key_offsets(d, "meshes")
        so = key_offsets(d, "skeletons")
        if mo:
            lo = mo[0]
            hi = min([o for o in key_offsets(d, "materials") if o > lo] + [len(d)])
            sec = split_by_keys(d, lo, hi, self.crc)
            cur = None
            for nm, off, p in sec:
                if nm == "vertexCount":
                    cur = {"drawblocks": []}
                    self.meshes.append(cur)
                if cur is None:
                    continue
                if nm in ("vertexCount", "skeletonIndex", "lod", "nodeIndex",
                          "vertexFormat", "instanceID", "boundingRadius"):
                    v = small_value(p)
                    if v is not None and nm not in cur:
                        cur[nm] = v
                elif nm in ("indexOffset", "indexCount", "materialIndex"):
                    if not cur["drawblocks"] or "indexCount" in cur["drawblocks"][-1]:
                        cur["drawblocks"].append({})
                    v = small_value(p)
                    if v is not None:
                        cur["drawblocks"][-1][nm] = v
        if so:
            lo = so[0]
            sec = split_by_keys(d, lo, len(d), self.crc)
            cur = None
            for nm, off, p in sec:
                if nm == "bindPose":
                    cur = {}
                    self.skeletons.append(cur)
                if cur is None:
                    continue
                if nm == "bindPose":
                    cur["bindPose"] = parse_float_array(p)
                elif nm == "nodes":
                    cur["nodes"] = parse_int_array(p)

    def _parse(self):
        # The index array must be read first: max(index)+1 is the authoritative
        # LOD0 vertex count, and it is what lets us pick the right vertex blob.
        ioff, icnt, itype = find_index_blob(self.d, 0, self.crc)
        self.ioff, self.icnt, self.itype = ioff, icnt, itype
        self.idx = self.tris = None
        iv = None
        if ioff is not None:
            dt = {"u8": "<u1", "u16": "<u2", "u32": "<u4"}[itype]
            a = np.frombuffer(self.d[ioff:ioff + icnt * np.dtype(dt).itemsize],
                              dtype=dt).astype(np.int64)
            self.idx = a
            self.tris = a.reshape(-1, 3)
            iv = int(a.max()) + 1
        self.ivcount = iv
        self._parse_nodes()
        self._parse_geometry()
        self._parse_meshes()

    # ---------------- transforms ----------------
    @staticmethod
    def euler_xyz(r):
        cx, sx = math.cos(r[0]), math.sin(r[0])
        cy, sy = math.cos(r[1]), math.sin(r[1])
        cz, sz = math.cos(r[2]), math.sin(r[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def world_matrices(self):
        """Bind-pose world matrix per node: parent @ T @ jointOrient @ rotate.

        localScaling is deliberately NOT applied: RootNode..DeformationSystem carry
        f8 0x38 (== 1.0), Group_JT carries 0.01, and every deeper node carries 1.0.
        Applying that 0.01 would shrink the whole rig 100x, yet the mesh vertices and
        the joint offsets share one unit system (hip at Y=1265 in a 1900-tall mesh),
        and the stored `bindPose` matrices reproduce exactly those un-scaled
        positions x 0.01.  So the 0.01 is a unit conversion on the DCC group node,
        not part of the skeletal transform -- confirmed by the bbox test below.
        """
        W = [None] * len(self.nodes)
        for k, n in enumerate(self.nodes):
            T = np.eye(4); T[:3, 3] = n["T"]
            R = np.eye(4); R[:3, :3] = self.euler_xyz(n["JO"]) @ self.euler_xyz(n["R"])
            L = T @ R
            W[k] = L if self.parent[k] is None else W[self.parent[k]] @ L
        return W

    def joint_world(self):
        return np.array([w[:3, 3] for w in self.world_matrices()])

    # ---------------- verification ----------------
    def verify(self, verbose=True):
        o = []
        A = o.append
        n = self.nodes
        A("=" * 78)
        A("VERIFICATION")
        A("=" * 78)
        roots = [k for k, p in enumerate(self.parent) if p is None]
        A(f"nodes                        : {len(n)}")
        A(f"roots                        : {len(roots)} {[n[k]['name'] for k in roots]}")
        A(f"unresolved parentUid         : {sum(1 for k, p in enumerate(self.parent) if p is None and k not in roots)}")
        bad = [k for k, p in enumerate(self.parent) if p is not None and p >= k]
        A(f"parent index >= own index    : {len(bad)}  (0 => stream is topologically sorted)")
        A(f"nodeIndex == position        : {all(n[k]['nodeIndex'] == k for k in range(len(n)))}")
        A(f"uid sequence 386+index       : {all(n[k]['uid'] == 386 + k for k in range(1, len(n)))}")
        # cycles
        seen = set()
        cyc = 0
        for k in range(len(n)):
            chain = set()
            j = k
            while j is not None:
                if j in chain:
                    cyc += 1
                    break
                chain.add(j)
                j = self.parent[j]
        A(f"cycles                       : {cyc}")
        A(f"children refs == parent links : "
          f"{sorted([c for k in range(len(n)) for c in n[k]['children']]) == sorted([n[k]['uid'] for k in range(len(n)) if self.parent[k] is not None])}")
        # bbox
        W = self.world_matrices()
        jp = np.array([w[:3, 3] for w in W])
        lo, hi = self.P.min(0), self.P.max(0)
        inside = ((jp >= lo - 50) & (jp <= hi + 50)).all(1).mean()
        A(f"mesh bbox                    : min {np.round(lo,1)} max {np.round(hi,1)}")
        A(f"joint world bbox             : min {np.round(jp.min(0),1)} max {np.round(jp.max(0),1)}")
        A(f"joints inside bbox(+/-50)    : {inside*100:.1f}%")
        A(f"joint extent                 : {np.round(jp.max(0)-jp.min(0),1)}")
        # named bones
        names = [x["name"] for x in n]
        for want in ("Root_M", "Chest_M", "Head_M", "L_Ankle", "R_Ankle"):
            if want in names:
                k = names.index(want)
                A(f"  world({want:9s}) = {np.round(jp[k], 2)}")
        A("")
        A("--- vertex record evidence ---")
        ln = np.linalg.norm(self.Nrm, axis=1)
        A(f"normal  @0x10 f16x3   |n| in [0.99,1.01]: {np.mean((ln>0.99)&(ln<1.01))*100:.3f}%  "
          f"range [{ln.min():.4f},{ln.max():.4f}]")
        A(f"bytes 0x0c..0x0f == 1.0f           : "
          f"{bool((self.vraw[:,0x0c:0x10] == np.array([0,0,0x80,0x3f],np.uint8)).all())}")
        A(f"bytes 0x16,0x17 zero               : {bool((self.vraw[:,0x16]==0).all() and (self.vraw[:,0x17]==0).all())}")
        for i, off in enumerate((0x18, 0x1C, 0x20, 0x24)):
            u, v = self.UV[i][:, 0], self.UV[i][:, 1]
            A(f"UV{i} @0x{off:02x} f16x2      in[0,1]: {np.mean((u>=0)&(u<=1)&(v>=0)&(v<=1))*100:6.2f}%  "
              f"u[{u.min():.4f},{u.max():.4f}] v[{v.min():.4f},{v.max():.4f}]")
        lt, lb = np.linalg.norm(self.TAN, axis=1), np.linalg.norm(self.BIT, axis=1)
        dtn = np.abs((self.TAN * self.Nrm).sum(1)) / np.maximum(lt, 1e-9)
        dbn = np.abs((self.BIT * self.Nrm).sum(1)) / np.maximum(lb, 1e-9)
        dtb = np.abs((self.TAN * self.BIT).sum(1)) / np.maximum(lt * lb, 1e-9)
        A(f"tangent @0x28 i8x3 /127  |t|~1     : {np.mean((lt>0.99)&(lt<1.01))*100:.3f}%")
        A(f"bitang  @0x2c i8x3 /127  |b|~1     : {np.mean((lb>0.99)&(lb<1.01))*100:.3f}%")
        A(f"t.n ~ 0 / b.n ~ 0 / t.b ~ 0        : {np.mean(dtn<0.02)*100:.2f}% / "
          f"{np.mean(dbn<0.02)*100:.2f}% / {np.mean(dtb<0.02)*100:.2f}%")
        ws = self.W.sum(1)
        A(f"weights @0x30 f16x4   sum==1.0     : {np.mean(np.abs(ws-1.0)<5e-4)*100:.4f}%  "
          f"[{ws.min():.6f},{ws.max():.6f}]")
        A(f"weights per slot nonzero           : {np.round((self.W>0).mean(0),5).tolist()}")
        A(f"joint idx @0x38 u8x4 distinct/slot : {[int(len(np.unique(self.J[:,c]))) for c in range(4)]}  "
          f"max {self.J.max(0).tolist()}")
        A(f"zero-weight slots hold index 0     : "
          f"{all(bool((self.J[self.W[:,c]==0, c]==0).all()) for c in range(1,4))}")
        if self.idx is not None:
            A(f"indices                      : {len(self.idx)} x {getattr(self,'itype','?')} "
              f"(min {self.idx.min()} max {self.idx.max()}), triangles {len(self.tris)}")
            A(f"all indices < vertexCount    : {bool((self.idx < self.vcount).all())}")
        A("")
        A("--- mesh / skeleton binding ---")
        A(f"meshes {len(self.meshes)}   skeletons {len(self.skeletons)}")
        A(f"SUM(vertexCount) == vcount   : "
          f"{sum((m.get('vertexCount') or 0) for m in self.meshes)} == {self.vcount}")
        sic = sum(db.get("indexCount") or 0 for m in self.meshes for db in m["drawblocks"])
        # This unit numbers each mesh's indices from 0 and gives every mesh its
        # own vertex blob, so the checks that assume one global buffer must use
        # the per-mesh ranges instead.
        self.per_mesh_vranges = True
        vcur = 0
        a = self.idx if self.idx is not None else np.zeros(0, np.int64)
        for m in self.meshes:
            dbs = [b for b in (m.get("drawblocks") or [])
                   if (b.get("indexCount") or 0) >= 3
                   and b.get("indexOffset") is not None]
            if dbs:
                mx = max(int(a[b["indexOffset"]:
                               b["indexOffset"] + b["indexCount"]].max())
                         for b in dbs)
                m["_vr"] = (vcur, vcur + mx + 1)
                vcur += mx + 1
            else:
                m["_vr"] = (vcur, vcur)
        svc = vcur
        A(f"SUM(per-mesh idxmax+1) == vcount: {svc} == {self.vcount}  -> {svc == self.vcount}")
        A("=> every mesh owns a private vertex blob with LOCAL index numbering")
        A(f"SUM(drawblock indexCount)    : {sic}"
          + (f"  (IndexBuffer holds {self.icnt} indices; the residue is index ranges "
             f"no drawblock lists)" if self.icnt else ""))
        A(f"vertexFormat values          : {sorted(set(m.get('vertexFormat') for m in self.meshes))}")
        A(f"lod values                   : {sorted(set(m.get('lod') for m in self.meshes))}")
        A(f"skeleton joint counts        : "
          f"{sorted(collections.Counter(max(len(s.get('nodes', [])), len(s.get('bindPose') or [])//16) for s in self.skeletons).items())}")
        okc = badc = 0
        worst = 0
        for m in self.meshes:
            lo, hi = m["_vr"]
            sk = m.get("skeletonIndex")
            s_ = self.skeletons[sk] if sk is not None and sk < len(self.skeletons) else None
            nj = max(len(s_.get("nodes", [])), len(s_.get("bindPose") or []) // 16) if s_ else 0
            mx = int(self.J[lo:hi].max()) if hi > lo and hi <= len(self.J) else -1
            if nj and mx < nj:
                okc += 1
                worst = max(worst, nj - 1 - mx)
            else:
                badc += 1
        A(f"meshes with max(jointIndex) < nJoints: {okc} / {okc+badc}"
          + ("   (every mesh's largest used index is exactly nJoints-1)" if worst == 0 else ""))
        A("")
        A("--- stored bindPose cross-check (strongest evidence) ---")
        ncmp = 0
        werr = 0.0
        for s in self.skeletons:
            uids = s.get("nodes") or []
            mats = s.get("bindPose") or []
            nk = len(mats) // 16
            idxs = [self.byuid[u] for u in uids if u in self.byuid] if uids else list(range(nk))
            for j in range(min(nk, len(idxs))):
                M = np.array(mats[j * 16:(j + 1) * 16]).reshape(4, 4)
                ref = W[idxs[j]].T.copy()
                ref[:3, :3] *= 0.01
                ref[:3, 3] *= 0.01
                ref[3, :3] *= 0.01
                ref[3, 3] = 1.0
                werr = max(werr, float(np.abs(M - ref).max()))
                ncmp += 1
        A(f"stored bindPose matrices compared   : {ncmp}")
        A(f"layout: bindPose[j] = world(joint_j)^T * 0.01 (row-vector convention, cm->m)")
        A(f"max |stored - reconstructed|        : {werr:.3e}"
          + ("   (== f32 storage precision)" if werr < 1e-3 else "   <-- MISMATCH"))
        txt = "\n".join(o)
        if verbose:
            print(txt)
        return txt

    # ---------------- reporting ----------------
    def print_tree(self, maxrows=400, cols=("T", "JO", "R")):
        W = self.world_matrices()
        kids = collections.defaultdict(list)
        for k, p in enumerate(self.parent):
            if p is not None:
                kids[p].append(k)
        lines = []
        order = sorted(kids.items())

        def rec(k, depth):
            n = self.nodes[k]
            w = W[k][:3, 3]
            bits = [f"{c}={np.round(n[c],3).tolist()}" for c in cols]
            lines.append(f"{'  '*depth}[{k:3d}] {str(n['name']):26s} uid={str(n['uid']):<4} "
                         f"idx={str(n['nodeIndex']):<4} world={np.round(w,1).tolist()!s:26s} "
                         + " ".join(bits))
            for c in sorted(kids.get(k, [])):
                rec(c, depth + 1)
        for k, p in enumerate(self.parent):
            if p is None:
                rec(k, 0)
        if len(lines) > maxrows:
            head = lines[:maxrows // 2]
            tail = lines[-maxrows // 2:]
            lines = head + [f"... ({len(lines)-maxrows} more nodes) ..."] + tail
        print("\n".join(lines))
        return "\n".join(lines)

    def export_json(self, path):
        W = self.world_matrices()
        out = {"file": os.path.basename(self.name),
               "vertexCount": self.vcount, "indexCount": int(len(self.idx)) if self.idx is not None else None,
               "nodes": [], "meshes": [], "skeletons": []}
        for k, n in enumerate(self.nodes):
            out["nodes"].append(dict(index=k, name=n["name"], uid=n["uid"],
                                     parentUid=n["parentUid"],
                                     parentIndex=self.parent[k],
                                     nodeIndex=n["nodeIndex"],
                                     localTranslation=n["T"], localRotation=n["R"],
                                     localScaling=n["S"], jointOrient=n["JO"],
                                     jointGroup=n["jointGroup"],
                                     children=n["children"],
                                     worldMatrix=[float(x) for x in W[k].reshape(-1)]))
        base = 0
        for m in self.meshes:
            e = dict(m)
            e["vertexStart"] = base
            base += m.get("vertexCount") or 0
            out["meshes"].append(e)
        for s in self.skeletons:
            nodes = s.get("nodes", [])
            mats = s.get("bindPose") or []
            out["skeletons"].append(dict(
                nodes=nodes,
                nodeNames=[self.nodes[self.byuid[u]]["name"] if u in self.byuid else None for u in nodes],
                bindPose=[mats[i * 16:(i + 1) * 16] for i in range(len(mats) // 16)]))
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        return path

    def export_obj(self, path, with_uv=True, with_normals=True):
        if self.idx is None:
            raise SystemExit("no index buffer")
        with open(path, "w") as f:
            f.write(f"# {os.path.basename(self.name)} from Project MIKHAIL (pack.bin)\n")
            f.write(f"# {self.vcount} vertices, {len(self.tris)} triangles\n")
            for p in self.P:
                f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            if with_uv:
                for u, v in self.UV[0]:
                    f.write(f"vt {u:.6f} {v:.6f}\n")
            if with_normals:
                for q in self.Nrm:
                    f.write(f"vn {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
            for t in self.tris:
                a, b, c = t + 1
                if with_uv and with_normals:
                    f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                else:
                    f.write(f"f {a} {b} {c}\n")
        return path


# --------------------------------------------------------------------------- #
def load(path):
    """Accept a .scn file, or an asset path inside obb/pack.bin."""
    if os.path.isfile(path):
        return open(path, "rb").read()
    sys.path.insert(0, HERE)
    from fpd_extract import FpdArchive, load_keystream
    root = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL"
    pack = FpdArchive(os.path.join(root, "obb", "pack.bin"),
                      load_keystream(os.path.join(HERE, "decryptKey.bin")))
    name = path.replace("\\", "/")
    if name not in pack.by_name:
        for ext in (".zstd", ""):
            if name + ext in pack.by_name:
                name = name + ext
                break
        else:
            raise SystemExit(f"{path}: not found in pack.bin")
    return pack.payload(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?",
                    default=os.path.join(HERE, "samples", "f22a_00.scn"))
    ap.add_argument("--obj")
    ap.add_argument("--json")
    ap.add_argument("--tree", type=int, default=0, help="print at most N tree rows")
    ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--full", action="store_true", help="no row limit on the tree")
    args = ap.parse_args()

    data = load(args.path)
    sc = Scene(data, args.path)
    print(f"{args.path}: {len(data)} bytes, name pool {len(sc.pool)} entries, "
          f"{len(sc.nodes)} nodes, {sc.vcount} vertices")
    if not args.no_tree:
        print()
        sc.print_tree(maxrows=(10 ** 9 if args.full else (args.tree or 40)))
    print()
    sc.verify()
    if args.obj:
        print("\nwrote", sc.export_obj(args.obj))
    if args.json:
        print("wrote", sc.export_json(args.json))


if __name__ == "__main__":
    main()
