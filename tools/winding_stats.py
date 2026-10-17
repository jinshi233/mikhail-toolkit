#!/usr/bin/env python3
"""
The blob search is already optimal (only one candidate each), so the mismatch is
not a wrong offset.  Check whether it is merely a winding convention (i.e. swap
two corners) or genuinely wrong data.  Also compare face/vertex topology stats
against a unit known to be correct.
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


def report(code):
    d = get(code)
    sc = ps.Scene(d, code)
    pos, nrm, tri = sc.P, None, sc.tris
    nrm = np.stack([sc.vraw[:, o:o + 2].copy().view("<f2").reshape(-1)
                    for o in (0x10, 0x12, 0x14)], 1).astype(np.float64)
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-9
    f = fn[ok] / ln[ok, None]
    vn = nrm[tri[ok]].mean(1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    dot = (f * vn).sum(1)
    # topology: how many distinct vertices, edges per vertex, duplicate tris
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]],
                                tri[:, [2, 0]]]), axis=1)
    ue = np.unique(e, axis=0)
    ut = np.unique(np.sort(tri, axis=1), axis=0)
    print(f"{code:14s} nv={sc.vcount:>8,} tris={len(tri):>8,} "
          f"fwd={float((dot > 0).mean()):.3%} rev={float((dot < 0).mean()):.3%} "
          f"meddot={np.median(dot):+.3f}")
    print(f"                edges={len(ue):,} (tris*1.5={int(len(tri)*1.5):,}) "
          f"uniq_tris={len(ut):,} dup={len(tri)-len(ut):,} "
          f"zero-area={int((~ok).sum()):,} "
          f"tris/nv={len(tri)/sc.vcount:.3f}")
    # average valence
    val = np.bincount(e.reshape(-1), minlength=sc.vcount)
    print(f"                avg valence={val[val > 0].mean():.3f}  "
          f"unused verts={int((val == 0).sum()):,}")


for c in ("f16_00", "f14_00", "mig21_00", "yf23pav1_00", "type00a_00"):
    report(c)
