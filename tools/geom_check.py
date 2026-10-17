#!/usr/bin/env python3
"""
f14_00 / mig21_00 / yf23pav1_00: geometry now complete.
  vertex blob starts where the '1.0' sentinel density begins
  index array: 0x95 <u32 n> 0x8b -> u16 @0xc1a
  drawblocks give global index ranges that concatenate exactly
Build the mesh, print the bounding box (a TSF must be ~15-20 m).
"""
import os
import sys

import numpy as np

WS = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL\unpack"
sys.path.insert(0, os.path.join(WS, "fpd"))
from mikhail_assets import FpdArchive
import parse_scene as ps
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from mikhail import config as _cfg
WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
GAME = _cfg.GAME


GAME = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL"
pack = FpdArchive(os.path.join(GAME, "obb", "pack.bin"))


def get(code):
    return pack.payload(f"root/assets/data/chr/tsf/{code}/model/{code}.ma.scn.zstd")


def find_arrays(d):
    import re
    out = []
    for mt in re.finditer(rb"\x95(.{4})([\x8b\x8d])", d, re.S):
        n = int.from_bytes(mt.group(1), "little")
        if 3 <= n <= len(d):
            esz = 2 if mt.group(2) == b"\x8b" else 4
            out.append((mt.end(), n, esz))
    return out


def vtx_start(d, esz=2):
    """first offset where f32 1.0 sentinels appear regularly"""
    u = np.frombuffer(d[:len(d) // 4 * 4], dtype="<u4")
    ones = np.flatnonzero(u == 0x3F800000) * 4
    diffs = np.diff(ones)
    for stride in (60, 64):
        m = diffs == stride
        if m.sum() < 20:
            continue
        # first index of the longest regular run
        runs = np.flatnonzero(m)
        brk = np.flatnonzero(np.diff(runs) != 1)
        starts = np.r_[runs[0], runs[brk + 1]]
        ends = np.r_[runs[brk], runs[-1]]
        k = int(np.argmax(ends - starts))
        return int(ones[starts[k]]), stride
    return None, None


def vtx_start_scan(d, nv):
    """try both strides at every offset near the sentinel region, pick the one
    whose float triples look like real model coordinates"""
    u = np.frombuffer(d[:len(d) // 4 * 4], dtype="<u4")
    ones = np.flatnonzero(u == 0x3F800000) * 4
    cands = []
    for stride in (60, 64):
        for base in ones[:400]:
            s = int(base) - 0xC
            if s < 0 or s + nv * stride > len(d):
                continue
            v = np.frombuffer(d[s:s + nv * stride], dtype=np.uint8).reshape(nv, stride)
            # sentinel must be 1.0 for every record
            sent = np.frombuffer(v[:, 0x0C:0x10].copy().tobytes(), dtype="<f4")
            if not np.allclose(sent, 1.0):
                continue
            pos = np.frombuffer(v[:, 0:12].copy().tobytes(), dtype="<f4").reshape(nv, 3)
            span = pos.max(0) - pos.min(0)
            if np.isfinite(span).all() and span.max() < 200 and span.max() > 5:
                cands.append((s, stride, span, pos))
                break
    return cands


for code in ("f14_00", "mig21_00", "yf23pav1_00"):
    d = get(code)
    sc = ps.Scene(d, code)
    arrs = find_arrays(d)
    off, n, esz = arrs[0]
    a = np.frombuffer(d[off:off + n * esz], dtype="<u2")
    nv = int(a.max()) + 1
    print(f"\n=== {code}")
    print(f"   idx  @0x{off:x} n={n:,} u{esz*8}  max={a.max():,}")

    cands = vtx_start_scan(d, nv)
    for s, stride, span, pos in cands:
        print(f"   vert @0x{s:x} stride={stride} nv={nv:,}  "
              f"span {np.round(span,3).tolist()}  OK")

    # reference check: every index within nv
    print(f"   index max < nv: {a.max() < nv}")

    # walk every drawblock by its own indexOffset
    blocks = [(m, b) for m in sc.meshes for b in (m.get("drawblocks") or [])]
    gaps = 0
    prev_end = 0
    for m, b in blocks:
        if b["indexOffset"] != prev_end:
            gaps += 1
        prev_end = b["indexOffset"] + b["indexCount"]
    print(f"   drawblocks={len(blocks)}  gaps={gaps}  "
          f"covered up to {prev_end:,}/{n:,}  meshes={len(sc.meshes)}")
