#!/usr/bin/env python3
"""
Audit what each LOD0 mesh actually is, instead of trusting a hardcoded part order.

For every unit, every LOD0 mesh: vertex count, world-space bounding box, its
skeleton's joints, and the vertices' dominant joint.  A mesh whose bbox covers
most of the model but whose skeleton has only torso joints is the "full body
mesh, no limb weights" the user is describing; a mesh order that differs from
LOD0_PARTS explains materials landing on the wrong parts.
"""
import os
import sys

import numpy as np

WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "mikhail"))
from mikhail_assets import FpdArchive
import parse_scene as ps
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

GAME = _cfg.GAME
pack = FpdArchive(_cfg.PACK)

UNITS = sys.argv[1:] or ["f22a_00", "f16_00", "f14_00", "type00a_00"]

for unit in UNITS:
    name = f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd"
    try:
        d = pack.payload(name)
    except Exception as exc:
        print(f"{unit}: {exc}")
        continue
    sc = ps.Scene(d, unit)
    nodes = [n["name"] for n in sc.nodes]
    # world-space extent of every LOD0 mesh, using its own vertex slice
    lod0 = [m for m in sc.meshes if m.get("lod") in (None, 0)]
    print(f"\n=== {unit}   LOD0 meshes={len(lod0)}  vcount={sc.vcount:,}")

    base = 0
    for mi, m in enumerate(sc.meshes):
        vc = m.get("vertexCount") or 0
        if m.get("lod") in (None, 0) and vc and base + vc <= len(sc.P):
            P = sc.P[base:base + vc]
            W = sc.W[base:base + vc]
            J = sc.J[base:base + vc]
            bb = np.round(P.max(0) - P.min(0), 0).astype(int)
            dom = int(np.bincount(J[np.arange(len(J)), W.argmax(1)],
                                  minlength=len(nodes)).argmax())
            sk = sc.skeletons[m.get("skeletonIndex") or 0]
            snames = [nodes[sc.byuid[u]] for u in (sk.get("nodes") or [])
                      if u in sc.byuid]
            print(f"  mesh{mi:>3} vc={vc:>7,} "
                  f"bbox={bb.tolist()} dom={nodes[dom] if dom < len(nodes) else '?'}")
            print(f"        skeleton({len(snames)}): {snames[:8]}")
        base += vc
