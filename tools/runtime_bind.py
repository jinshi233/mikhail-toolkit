#!/usr/bin/env python3
"""
Runtime truth: read the exported JSON exactly as blender_rig does -- positions,
weights and joints are already per-mesh in meshes_out -- and report, per LOD0
mesh, the bones that actually carry weight.
"""
import json
import sys

import numpy as np

for unit in ("f14_00", "mig21_00", "yf23pav1_00"):
    j = json.load(open(rf"export\{unit}\{unit}.json"))
    print(f"\n=== {unit}")
    for m in j["meshes"]:
        if m.get("lod") not in (None, 0):
            continue
        names = m["jointNames"]
        W = np.asarray(m["weights"], dtype=np.float64)
        J = np.asarray(m["joints"], dtype=np.int64)
        nz = W > 1e-3
        per = np.zeros(len(names))
        for c in range(W.shape[1]):
            sel = nz[:, c] & (J[:, c] >= 0) & (J[:, c] < len(names))
            np.add.at(per, J[sel, c], W[sel, c])
        used = [names[i] for i in range(len(names)) if per[i] > 0]
        unused = [names[i] for i in range(len(names)) if per[i] <= 0]
        bad = int((~((W.sum(1) > 0.97) & (W.sum(1) < 1.03))).sum())
        jmax = int(J.max()) if J.size else -1
        print(f"  mesh{m['index']:>3} {str(m.get('part')):<8} verts={len(W):>6} "
              f"joints={len(names):>3} weighted={len(used):>3} jmax={jmax:>3} "
              f"weightsum_bad={bad}")
        if unused:
            print(f"      NO WEIGHT: {unused}")
