#!/usr/bin/env python3
"""
Disambiguate the skin offset.  A real weight group is 4 float16 values in [0,1]
(non-negative), summing to 1.0, followed by 4 single-byte joint indices all
< nJoints.  Reject windows whose bytes include joint bytes pretending to be
weights by requiring every f16 weight to be in [0, 1] and the joint bytes to be
strictly small.
"""
import os
import sys

import numpy as np

WS = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL\unpack"
sys.path.insert(0, os.path.join(WS, "fpd"))
from mikhail_assets import FpdArchive
import parse_scene as ps
import mesh_blobs
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from mikhail import config as _cfg
WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
GAME = _cfg.GAME


GAME = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL"
pack = FpdArchive(os.path.join(GAME, "obb", "pack.bin"))


def eval_offset(sl, o, nj):
    n = len(sl)
    raw = sl[:, o:o + 8]
    W = np.frombuffer(raw.tobytes(), dtype="<f2").reshape(n, 4).astype(np.float64)
    nan = ~np.isfinite(W)
    W = np.where(nan, -1.0, W)
    inrange = ((W >= 0.0) & (W <= 1.0)).all(1)
    s1 = np.abs(W.sum(1) - 1.0) < 5e-3
    J = sl[:, o + 8:o + 12].astype(np.int64)
    jok = (J < nj).all(1) if nj else np.zeros(n, bool)
    return float(np.mean(inrange & s1 & jok))


for code in ("f14_00", "mig21_00", "yf23pav1_00", "f16_00"):
    d = pack.payload(f"root/assets/data/chr/tsf/{code}/model/{code}.ma.scn.zstd")
    sc = ps.Scene(d, code)
    a = sc.idx.astype(np.int64)
    pool = mesh_blobs.candidate_pool(d)
    groups = {}
    print(f"\n=== {code}")
    for mi, m in enumerate(sc.meshes):
        dbs = [b for b in (m.get("drawblocks") or [])
               if (b.get("indexCount") or 0) >= 3]
        if not dbs:
            continue
        tri = np.concatenate([a[b["indexOffset"]:b["indexOffset"] + b["indexCount"]]
                              for b in dbs]).reshape(-1, 3)
        used = np.unique(tri.ravel())
        need = int(used.max()) + 1
        nj = len(sc.skeletons[m["skeletonIndex"]].get("nodes") or [])
        bestw, bb = -1.0, None
        for voff, stride, vn, pos, nrm in pool:
            if vn < need:
                continue
            w = mesh_blobs.winding_score(pos, nrm, tri)
            if w > bestw:
                bestw, bb = w, (voff, stride, vn)
        if bb is None or bestw < 0.9:
            continue
        voff, stride, vn = bb
        v = np.frombuffer(d[voff:voff + vn * stride],
                          dtype=np.uint8).reshape(vn, stride)
        sl = np.ascontiguousarray(v[used])
        scores = [(eval_offset(sl, o, nj), o) for o in range(0x20, 0x39)]
        scores.sort(reverse=True)
        top = scores[0]
        groups.setdefault(top[1], []).append(mi)
        print(f"   mesh{mi:3d} lod={m.get('lod')} nJ={nj:>3} wind={bestw:7.3%} "
              f"-> best offset 0x{top[1]:02x} strict={top[0]:7.2%}   "
              f"runner 0x{scores[1][1]:02x}={scores[1][0]:6.2%}")
    print(f"   groups: { {f'0x{k:02x}': len(v) for k, v in groups.items()} }")
