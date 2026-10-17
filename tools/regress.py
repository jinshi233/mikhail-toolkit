#!/usr/bin/env python3
"""
Regression check for the parser change.

Invariant that must now hold for EVERY TSF:  vcount == max(index)+1, because the
vertex blob is selected by that number.  Also check drawblock indices stay in
range and the geometry span is TSF-sized (cm -> 1000..2500).
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

# every TSF model entry in the archive
codes = sorted({e.name.split("/")[5] for e in pack.entries
                if "/chr/tsf/" in e.name
                and e.name.endswith(".ma.scn.zstd")})
print(f"{len(codes)} TSF model entries\n")

hdr = (f"{'code':16s} {'idx':>9s} {'nv':>8s} {'max+1':>8s} {'tris':>8s} "
       f"{'st':>3s} {'meshvc':>8s} {'span_cm (y)':>12s}  flags")
print(hdr)
print("-" * len(hdr))

bad = []
for code in codes:
    name = f"root/assets/data/chr/tsf/{code}/model/{code}.ma.scn.zstd"
    try:
        d = pack.payload(name)
        sc = ps.Scene(d, code)
    except Exception as e:
        print(f"{code:16s} ERROR {type(e).__name__}: {e}")
        bad.append(code)
        continue

    nv = sc.vcount
    m1 = int(sc.idx.max()) + 1 if sc.idx is not None else -1
    span = (sc.P.max(0) - sc.P.min(0)) if nv else np.zeros(3)
    meshvc = sum(m.get("vertexCount") or 0 for m in sc.meshes)
    flags = []
    if sc.idx is None:
        flags.append("NO-IDX")
    elif nv != m1:
        flags.append(f"NV!=MAX+1({m1})")
    if not (1000 < span[1] < 2600):
        flags.append(f"SPAN:{span[1]:.0f}")
    if meshvc and meshvc != nv:
        flags.append(f"MESHVC={meshvc}")
    if flags:
        bad.append(code)
    print(f"{code:16s} {len(sc.idx) if sc.idx is not None else 0:>9,} "
          f"{nv:>8,} {m1:>8,} {len(sc.tris) if sc.tris is not None else 0:>8,} "
          f"{sc.vstride:>3d} {meshvc:>8,} {span[1]:>12.1f}  {' '.join(flags)}")

print(f"\n{len(codes)-len(bad)}/{len(codes)} clean")
if bad:
    print("needs work:", ", ".join(bad))
