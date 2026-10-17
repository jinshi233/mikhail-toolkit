#!/usr/bin/env python3
"""
Per-mesh vertex-blob solver.

Most TSFs keep one contiguous LOD0 vertex buffer, so `vcount == max(index)+1`
and the indices are global.  F-14 / MiG-21 / YF-23 Pav-1 instead give every mesh
a PRIVATE vertex blob and number that mesh's indices from 0, so the concatenated
index array looks full of duplicate triangles and a global read mis-aligns the
geometry.

Winding agreement between each triangle's geometric normal and the stored vertex
norms is a reliable oracle (correct data scores 99.4-100%), so use it to choose
the blob for each mesh, then rewrite the indices into a local space.
"""
import numpy as np

STRIDES = (60, 64, 48)


def detect_skin_offsets(sl, nj, stride):
    """Find where this mesh's skin weights and joint indices really live.

    The record is not fixed-layout across meshes: 60-byte records put weights at
    0x30 / joints at 0x38, while 64-byte records put them at 0x34 / 0x3C -- the
    4 extra bytes sit earlier in the record, so the skin block shifts with it.
    Only those two alignments are legal (wider scans latch onto the UV pair just
    before the skin block and score well while reporting only 1 joint).

    Returns (weight_offset, joint_offset), preferring the alignment that gets
    every weighted slot below the skeleton size.
    """
    n = len(sl)
    nj = max(1, nj)
    best = None
    for o in (0x30, 0x34):
        if o + 12 > stride:
            continue
        W = np.frombuffer(sl[:, o:o + 8].copy().tobytes(),
                          dtype="<f2").reshape(n, 4).astype(np.float64)
        W = np.where(np.isfinite(W), W, -1.0)
        good = ((W >= 0.0) & (W <= 1.0)).all(1) & \
               (np.abs(W.sum(1) - 1.0) < 5e-3)
        J = sl[:, o + 8:o + 12].astype(np.int64)
        msk = W > 1e-3
        jok = (~msk | (J < nj)).all(1)
        # prefer more distinct joints actually used, then the cleaner candidate
        njoint = len(set(J[msk & good[:, None]].tolist()))
        score = (njoint, float((good & jok).mean()))
        if best is None or score > best[0]:
            best = (score, o)
    if best is None:
        # no skin block in this record layout (kakasi's 48-byte static records),
        # so report an unskinned mesh
        return None
    o = best[1]
    return o, o + 8


def skin_of(sl, nj, stride):
    """Return (weights (n,4) float64, joints (n,4) int64, offsets) using detected
    offsets, with the weight rows normalised.  offsets is None for an unskinned
    record layout (no skin block at all)."""
    off = detect_skin_offsets(sl, nj, stride)
    if off is None:
        n = len(sl)
        return (np.zeros((n, 4)), np.zeros((n, 4), dtype=np.int64), None)
    wo, jo = off
    n = len(sl)
    W = np.frombuffer(sl[:, wo:wo + 8].copy().tobytes(),
                      dtype="<f2").reshape(n, 4).astype(np.float64)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.clip(W, 0.0, None)
    J = sl[:, jo:jo + 4].astype(np.int64)
    return W, J, (wo, jo)


def sentinel_chains(d, stride, minlen=200):
    """Maximal runs of records with f32 1.0 at +0x0C, spaced exactly `stride`."""
    u = np.frombuffer(d[:len(d) // 4 * 4], dtype="<u4")
    ones = set((np.flatnonzero(u == 0x3F800000) * 4).tolist())
    used, out = set(), []
    for cand in sorted(ones):
        if cand in used or (cand - stride) in ones:
            continue
        n = 1
        while (cand + n * stride) in ones:
            n += 1
        for k in range(n):
            used.add(cand + k * stride)
        if n >= minlen:
            out.append((cand - 0xC, n))
    return out


def read_pos_nrm(d, off, stride, n):
    v = np.frombuffer(d[off:off + n * stride], dtype=np.uint8).reshape(n, stride)
    pos = np.frombuffer(v[:, 0:12].copy().tobytes(), dtype="<f4").reshape(n, 3)
    nrm = np.frombuffer(v[:, 0x10:0x16].copy().tobytes(),
                        dtype="<f2").reshape(n, 3).astype(np.float64)
    return pos, nrm


def winding_score(pos, nrm, tri):
    """Fraction of triangles whose winding agrees with the stored normals."""
    if len(tri) < 20:
        return -1.0
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-9
    if ok.sum() < 20:
        return -1.0
    f = fn[ok] / ln[ok, None]
    vn = nrm[tri[ok]].mean(1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    return float(((f * vn).sum(1) > 0).mean())


def candidate_pool(d, minlen=200):
    pool = []
    for stride in STRIDES:
        for voff, vn in sentinel_chains(d, stride, minlen):
            if voff < 0 or voff + vn * stride > len(d):
                continue
            pos, nrm = read_pos_nrm(d, voff, stride, vn)
            if not np.isfinite(pos).all():
                continue
            span = float(pos.max() - pos.min())
            if not (1.0 < span < 1e6):
                continue
            pool.append((voff, stride, vn, pos, nrm))
    return pool


def solve(d, scene):
    """Return (pool, per-mesh plan).

    For each mesh: pick the vertex blob whose winding agreement is highest, then
    keep exactly `max(index)+1` records -- the indices are LOCAL to that mesh, so
    only the slice it actually addresses belongs to it.  plan[mi] is None when
    the mesh has no usable index block.
    """
    idxarr = scene.idx
    if idxarr is None:
        return [], []
    idxarr = idxarr.astype(np.int64)
    pool = candidate_pool(d)
    plan = [None] * len(scene.meshes)
    for mi, m in enumerate(scene.meshes):
        dbs = [b for b in (m.get("drawblocks") or [])
               if (b.get("indexCount") or 0) >= 3
               and b.get("indexOffset") is not None]
        if not dbs:
            continue
        tri = np.concatenate([idxarr[b["indexOffset"]:
                                     b["indexOffset"] + b["indexCount"]]
                              for b in dbs]).reshape(-1, 3)
        if not len(tri):
            continue
        need = int(tri.max()) + 1
        best = (-1.0, None)
        for voff, stride, vn, pos, nrm in pool:
            if vn < need:
                continue
            s = winding_score(pos, nrm, tri)
            if s > best[0]:
                best = (s, (voff, stride, vn))
        if best[1] is None:
            continue
        plan[mi] = dict(blob=best[1], tri=tri, nv=need, wind=best[0])
    return pool, plan
