#!/usr/bin/env python3
"""Inspect a BETA unit's structure so the exporter can be adapted."""
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

for code in ("tank_00", "destroyer_00", "grappler_00", "fort_00"):
    name = f"root/assets/data/chr/beta/{code}/model/{code}.ma.scn.zstd"
    try:
        d = pack.payload(name)
    except Exception as e:
        print(f"{code}: MISSING ({e})")
        continue
    sc = ps.Scene(d, code)
    nodes = [n["name"] for n in sc.nodes]
    meshes = sc.meshes
    lod0 = [m for m in meshes if m.get("lod") in (None, 0)]
    span = (sc.P.max(0) - sc.P.min(0)) if sc.vcount else np.zeros(3)
    print(f"\n=== {code}  file={len(d):,}")
    print(f"    nodes={len(nodes)} skeletons={len(sc.skeletons)} "
          f"meshes={len(meshes)} (LOD0 {len(lod0)})")
    print(f"    vcount={sc.vcount:,} vstride={sc.vstride} idx={len(sc.idx) if sc.idx is not None else 0:,}")
    print(f"    span_cm={np.round(span,0).tolist()} -> m={np.round(span/100,2).tolist()}")
    print(f"    skeletons sizes={sorted({len(s.get('nodes') or []) for s in sc.skeletons})}")
    print(f"    joint names (skel0)={(sc.skeletons[0].get('nodes') and [nodes[sc.byuid[u]] for u in sc.skeletons[0]['nodes'] if u in sc.byuid])}")
    print(f"    drawblock sum={sum(b.get('indexCount',0) for m in meshes for b in (m.get('drawblocks') or [])):,}")
    for mi, m in enumerate(meshes[:6]):
        dbs = m.get("drawblocks") or []
        print(f"      mesh{mi}: lod={m.get('lod')} vc={m.get('vertexCount')} "
              f"fmt={m.get('vertexFormat')} dbs={len(dbs)} skIdx={m.get('skeletonIndex')}")
