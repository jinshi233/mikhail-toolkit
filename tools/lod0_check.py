#!/usr/bin/env python3
"""
LOD0 vertex block = the FIRST stride-60 sentinel chain, length == idxmax+1.
  f14_00      @0x14f500, 14,713 records
  mig21_00    @0x0fcee0, 28,233 records
  yf23pav1_00 @0x0972b0, 21,663 records
Build the full mesh with the u16 indices and verify it is a real TSF.
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


UNITS = [("f14_00", 0x14F500, 0x41F2B),
         ("mig21_00", 0x0FCEE0, 0x55083),
         ("yf23pav1_00", 0x0972B0, 0x44D8)]

for code, voff, n in UNITS:
    d = get(code)
    a = np.frombuffer(d[0xC1A:0xC1A + n * 2], dtype="<u2").astype(np.int32)
    nv = int(a.max()) + 1
    v = np.frombuffer(d[voff:voff + nv * 60], dtype=np.uint8).reshape(nv, 60)
    pos = np.frombuffer(v[:, 0:12].copy().tobytes(), dtype="<f4").reshape(nv, 3)
    nrm = np.frombuffer(v[:, 0x10:0x16].copy().tobytes(),
                        dtype="<f2").reshape(nv, 3)
    uvs = np.frombuffer(v[:, 0x18:0x1c].copy().tobytes(),
                        dtype="<f2").reshape(nv, 2)
    wts = np.frombuffer(v[:, 0x30:0x38].copy().tobytes(),
                        dtype="<f2").reshape(nv, 4)
    jnt = v[:, 0x38:0x3C]
    tri = a.reshape(-1, 3)
    deg = ((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2])
           | (tri[:, 0] == tri[:, 2]))
    span = pos.max(0) - pos.min(0)
    # triangle areas for sanity
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    print(f"\n=== {code}  nv={nv:,}  tris={len(tri):,}")
    print(f"   bbox X[{pos[:,0].min():8.3f},{pos[:,0].max():8.3f}] "
          f"Y[{pos[:,1].min():8.3f},{pos[:,1].max():8.3f}] "
          f"Z[{pos[:,2].min():8.3f},{pos[:,2].max():8.3f}]")
    print(f"   span {np.round(span,3).tolist()}")
    print(f"   normal len  mean={np.linalg.norm(nrm,axis=1).mean():.4f} "
          f"(want 1.0)")
    print(f"   uv range    u[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] "
          f"v[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}]")
    print(f"   weights sum mean={wts.sum(1).mean():.5f} (want 1.0)")
    print(f"   joints      max={jnt.max()} uniq={len(np.unique(jnt)):,}")
    print(f"   degenerate  {deg.sum():,} ({deg.mean():.4%})")
    print(f"   tri area    mean={area.mean():.6f} max={area.max():.4f} "
          f"zero={(area<1e-12).sum():,}")
