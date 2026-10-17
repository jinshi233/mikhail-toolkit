#!/usr/bin/env python3
"""
Diagnose the current per-unit export quality.

Checks, per unit:
  * how many LOD0 meshes Blender will build, and how many vertex groups each
    mesh has vs how many its skeleton declares
  * vertex/index totals from the JSON as written
  * how many vertices in each mesh got NO joint influence at all
"""
import json
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else r"export"
units = sorted(d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d)))
print(f"{'unit':<18}{'LOD0msh':>8}{'verts':>10}{'faces':>9}"
      f"{'unweighted':>11}  meshes with no weights")
print("-" * 96)
bad = []
for u in units:
    js = os.path.join(root, u, u + ".json")
    if not os.path.exists(js):
        continue
    j = json.load(open(js))
    lod0 = [m for m in j["meshes"] if m.get("lod") in (None, 0)]
    tv = sum(m["vertexCount"] for m in lod0)
    tf = sum(sum((b.get("indexCount") or 0) // 3 for b in m["drawblocks"])
             for m in lod0)
    unw = 0
    empty = []
    for m in lod0:
        J = np.asarray(m["joints"], dtype=np.int64)
        W = np.asarray(m["weights"], dtype=np.float64)
        if not len(J):
            empty.append(m["index"])
            continue
        s = W.sum(1)
        n = int((s <= 1e-9).sum())
        unw += n
        if n:
            empty.append(m["index"])
    flag = ""
    if unw or empty:
        flag = f"  <== {len(empty)} meshes, e.g. {empty[:6]}"
        bad.append(u)
    print(f"{u:<18}{len(lod0):>8}{tv:>10,}{tf:>9,}{unw:>11,}{flag}")
print("-" * 96)
print(f"{len(units)} units, {len(bad)} with unweighted vertices: {bad}")
