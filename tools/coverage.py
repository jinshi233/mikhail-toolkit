#!/usr/bin/env python3
"""
What does each LOD0 mesh actually cover vertically?  A part that spans nearly the
whole model height is the "full body mesh" the user is seeing; a missing chest
shows up as an uncovered band between the head and the legs.
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

UNIT = sys.argv[1] if len(sys.argv) > 1 else "f22a_00"
d = pack.payload(f"root/assets/data/chr/tsf/{UNIT}/model/{UNIT}.ma.scn.zstd")
sc = ps.Scene(d, UNIT)
nodes = [n["name"] for n in sc.nodes]

P = sc.P
ymin, ymax = P[:, 1].min(), P[:, 1].max()
H = ymax - ymin
print(f"=== {UNIT}  model Y range {ymin:.0f}..{ymax:.0f} (height {H:.0f} cm)")

base = 0
rows = []
for mi, m in enumerate(sc.meshes):
    vc = m.get("vertexCount") or 0
    if vc == 0:
        continue
    if base + vc > len(P):
        break
    sl = P[base:base + vc]
    lo, hi = sl[:, 1].min(), sl[:, 1].max()
    rows.append((mi, vc, lo, hi, (hi - lo) / H))
    base += vc

print(f"{'mesh':>5} {'verts':>8} {'Ymin':>7} {'Ymax':>7} {'cover':>7}  bar (0=feet 1=head)")
for mi, vc, lo, hi, cov in rows:
    a = int((lo - ymin) / H * 40)
    b = int((hi - ymin) / H * 40)
    bar = " " * a + "#" * max(1, b - a)
    print(f"{mi:>5} {vc:>8,} {lo:>7.0f} {hi:>7.0f} {cov:>7.1%}  |{bar:<40}|")

# vertical coverage of the union
print("\nvertical occupancy (any mesh), 20 bands feet->head:")
bands = np.zeros(20, int)
for mi, vc, lo, hi, cov in rows:
    a = int((lo - ymin) / H * 20)
    b = min(19, int((hi - ymin) / H * 20))
    bands[a:b + 1] += 1
for i, n in enumerate(bands):
    print(f"   {i:>2} {'#' * n:<12} {n} mesh(es)")
