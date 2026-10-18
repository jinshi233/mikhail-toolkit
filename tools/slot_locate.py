#!/usr/bin/env python3
"""
Which mask slot does the visor belong to?

For a part's vertices, sample the `_cm` mask at each vertex's UV and report where
each slot lands in 3D.  The visor is the small group on the FRONT of the head, so
this identifies the slot that should be UserColor_Yellow instead of guessing.
"""
import os
import sys

import numpy as np

WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "mikhail"))
from mikhail_assets import FpdArchive
import parse_scene as ps
import txb_fast
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

GAME = _cfg.GAME
pack = FpdArchive(_cfg.PACK)

UNIT = sys.argv[1] if len(sys.argv) > 1 else "f22a_00"
PART = sys.argv[2] if len(sys.argv) > 2 else "head"

d = pack.payload(f"root/assets/data/chr/tsf/{UNIT}/model/{UNIT}.ma.scn.zstd")
sc = ps.Scene(d, UNIT)
nbi = {n.get("nodeIndex"): n.get("name") for n in sc.nodes}

# the mask
tn = [e.name for e in pack.entries
      if f"/chr/tsf/{UNIT}/" in e.name and f"{PART}_cm.png" in e.name
      and e.name.endswith(".txb.desktop.zstd")]
if not tn:
    raise SystemExit(f"no {PART}_cm mask")
mw, mh, _fmt, mask = txb_fast.decode(pack.payload(tn[0]))
mc = mask[:, :, :3].astype(np.int16)
print(f"mask {mw}x{mh}")


def slot_of(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    if r > 180 and g < 60 and b < 60:
        return "red"
    if g > 180 and r < 60 and b < 60:
        return "green"
    if b > 180 and r < 60 and g < 60:
        return "blue"
    return "black"


base = 0
for mi, m in enumerate(sc.meshes):
    vc = m.get("vertexCount") or 0
    if nbi.get(m.get("nodeIndex")) != PART:
        base += vc
        continue
    P = sc.P[base:base + vc]
    UV = sc.UV[base:base + vc, 0]
    groups = {}
    for i in range(0, vc, 7):                    # sample every 7th vertex
        u, v = float(UV[i, 0]), float(UV[i, 1])
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            continue
        x = min(mw - 1, int(u * (mw - 1)))
        y = min(mh - 1, int(v * (mh - 1)))
        s = slot_of(mc[y, x])
        groups.setdefault(s, []).append(P[i])
    print(f"\nmesh{mi} part={PART} vc={vc}")
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    for s, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        a = np.array(pts)
        frac = len(a) / vc
        print(f"   {s:<6} {len(a):>7} samples ({frac:>5.1%})  "
              f"X {a[:,0].min():>7.0f}..{a[:,0].max():<7.0f} "
              f"Y {a[:,1].min():>7.0f}..{a[:,1].max():<7.0f} "
              f"Z {a[:,2].min():>7.0f}..{a[:,2].max():<7.0f}")
    print(f"   (part Y range {ymin:.0f}..{ymax:.0f})")
    base += vc
