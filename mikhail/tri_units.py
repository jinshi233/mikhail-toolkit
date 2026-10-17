#!/usr/bin/env python3
"""
Correct extraction for f14_00 / mig21_00 / yf23pav1_00.

Structure (verified):
  * index array is u16 @0xc1a; its values are GLOBAL indices into a vertex pool
  * the drawblock chain partitions it into per-mesh contiguous blocks
  * each mesh references a contiguous, dense range of the pool, and that range
    is a slice of one of the pool's stride-60/64 blobs
  * every pool blob is a sentinel chain, so its extent is exact

So: per mesh, pick the blob with the best winding, take the mesh's contiguous
index range as a local slice of it, and renumber densely from 0.
"""
import os
import sys

import numpy as np

STRIDES = (60, 64)
# Blob selection threshold.  LOD0 meshes reach 97.7-100%; the coarser LODs sit a
# little lower, and they are not exported anyway.
MIN_WINDING = 0.90

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from .mikhail_assets import FpdArchive  # noqa: E402
except ImportError:
    from mikhail_assets import FpdArchive  # noqa: E402
import parse_scene as ps
try:
    from . import mesh_blobs  # noqa: E402
except ImportError:
    import mesh_blobs  # noqa: E402

try:
    from . import config as _cfg
except ImportError:
    import config as _cfg
GAME = _cfg.GAME
pack = FpdArchive(os.path.join(GAME, "obb", "pack.bin"))


def solve_unit(d, scene):
    """-> list over meshes of dict(tri, blob, keep, vlo, vhi, nverts, wind) or None.

    Blob choice uses TWO oracles, and the combination is what makes it unique:
      geometry : triangle winding must agree with the stored vertex normals
      skinning : weights must sum to 1.0 and joint indices must be < nJoints
    Geometry alone picked a blob whose skinning was garbage for F-14 meshes
    0/1/2/6 (leg hand jmax=255 against an 18-joint skeleton), which is exactly
    the "missing bindings" symptom: those vertices were dropped.
    """
    a = scene.idx.astype(np.int64)
    pool = mesh_blobs.candidate_pool(d)
    out = [None] * len(scene.meshes)
    for mi, m in enumerate(scene.meshes):
        dbs = [b for b in (m.get("drawblocks") or [])
               if (b.get("indexCount") or 0) >= 3
               and b.get("indexOffset") is not None]
        if not dbs:
            continue
        tri = np.concatenate([a[b["indexOffset"]:b["indexOffset"] + b["indexCount"]]
                              for b in dbs]).reshape(-1, 3)
        if not len(tri):
            continue
        used = np.unique(tri.ravel())
        vlo, vhi = int(used.min()), int(used.max())
        need = vhi + 1
        # the blob must actually hold every vertex the indices address
        if need <= 0:
            continue
        nj = len(scene.skeletons[m["skeletonIndex"]].get("nodes") or [])
        best = None
        for voff, stride, vn, pos, nrm in pool:
            if vn < need:
                continue
            # winding is the blob selector: it is content-based and, in practice,
            # uniquely identifies the right blob per mesh
            w = mesh_blobs.winding_score(pos, nrm, tri)
            if w < MIN_WINDING:
                continue
            if best is None or w > best[0]:
                best = (w, (voff, stride, vn))
        if best is None:
            continue
        w, blob = best
        voff, stride, vn = blob
        remap = np.zeros(vn + 1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        out[mi] = dict(tri=remap[tri], blob=blob, vlo=vlo, vhi=vhi,
                       wind=w, dense=len(used) == vhi + 1 - vlo,
                       keep=used, nverts=len(used))
    return out


if __name__ == "__main__":
    for code in ("f14_00", "mig21_00", "yf23pav1_00"):
        d = pack.payload(
            f"root/assets/data/chr/tsf/{code}/model/{code}.ma.scn.zstd")
        sc = ps.Scene(d, code)
        sol = solve_unit(d, sc)
        got = sum(1 for x in sol if x)
        winds = [x["wind"] for x in sol if x]
        dense = sum(1 for x in sol if x and x["dense"])
        print(f"\n=== {code}: {got}/{len(sol)} meshes, "
              f"median winding {np.median(winds):.4%}, min {min(winds):.4%}, "
              f"dense ranges {dense}/{got}")
        for mi in range(min(6, len(sol))):
            x = sol[mi]
            if not x:
                continue
            print(f"   mesh{mi:3d} range [{x['vlo']:,},{x['vhi']:,}) "
                  f"tris {len(x['tri']):,} blob 0x{x['blob'][0]:x}/"
                  f"{x['blob'][1]} wind {x['wind']:.4%} dense={x['dense']}")
