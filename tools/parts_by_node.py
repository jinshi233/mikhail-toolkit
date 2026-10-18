#!/usr/bin/env python3
"""
Authoritative part list per unit: Mesh.nodeIndex -> node.name.

Every Mesh record carries `nodeIndex`, and those nodes are named
`chest, head, j_unit, l_arm, r_arm, leg, l_b_mount, r_b_mount, l_g_mount,
r_g_mount` (10 per LOD group).  This replaces the positional LOD0_PARTS table,
which is wrong for any unit that orders its meshes differently.
"""
import os
import sys
from collections import Counter

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

PARTS = {"chest", "head", "j_unit", "l_arm", "r_arm", "leg",
         "l_b_mount", "r_b_mount", "l_g_mount", "r_g_mount"}

units = sorted({e.name.split("/")[5] for e in pack.entries
                if "/chr/tsf/" in e.name and e.name.endswith(".ma.scn.zstd")})
print(f"{'unit':<18} {'n':>2}  parts (LOD0, by node name)")
print("-" * 104)
bad = []
for unit in units:
    d = pack.payload(f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd")
    sc = ps.Scene(d, unit)
    nbi = {n.get("nodeIndex"): n.get("name") for n in sc.nodes}
    names, unknown = [], 0
    for m in sc.meshes:
        if m.get("lod") not in (None, 0):
            continue
        nm = nbi.get(m.get("nodeIndex"))
        names.append(nm if nm in PARTS else f"<{nm}>")
        if nm not in PARTS:
            unknown += 1
    have = Counter(n for n in names if n in PARTS)
    miss = sorted(PARTS - set(have))
    dup = [p for p, c in have.items() if c > 1]
    flag = ""
    if miss:
        flag += " MISS:" + ",".join(miss)
    if dup:
        flag += " DUP:" + ",".join(dup)
    if unknown:
        flag += f" UNKNOWN:{unknown}"
    if flag:
        bad.append(unit)
    print(f"{unit:<18} {len(names):>2}  {','.join(names)}{flag}")
print("-" * 104)
print(f"{len(bad)}/{len(units)} units with a missing/duplicate/unknown part")
print("bad:", bad)
