#!/usr/bin/env python3
"""
export_rigged.py — Project MIKHAIL  ->  rigged FBX / glTF

Pipeline (all layers verified elsewhere, see SCENE_FORMAT_FINDINGS.md):

    obb/pack.bin (FPD\0)  ->  .zstd  ->  EPK\0 .ma.scn
        parse_scene.py             node tree, world matrices, bind poses,
                                   mesh slices, joints
        vertex record (60 B)       position/normal/uv/tangent/weights/joints
        .txb                       BC7/BC5 textures -> PNG

This script merges those into one JSON per unit and then drives Blender to
produce FBX and glTF with a real armature, joint weights and PBR materials.

    python export_rigged.py --kind tsf
    python export_rigged.py --kind tsf --name f22a_00
    python export_rigged.py --kind weapon
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from .mikhail_assets import FpdArchive, classify_texture, TEX_ROLE  # noqa: E402
except ImportError:
    from mikhail_assets import FpdArchive, classify_texture, TEX_ROLE  # noqa: E402
try:
    from . import txb_fast  # noqa: E402
except ImportError:
    import txb_fast  # noqa: E402
try:
    from . import mesh_blobs  # noqa: E402
except ImportError:
    import mesh_blobs  # noqa: E402

try:
    from . import config as _cfg
except ImportError:
    import config as _cfg
GAME = _cfg.GAME
PACK = _cfg.PACK
UNPACK = os.path.dirname(HERE)
OUT_ROOT = _cfg.OUT

BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
]

UNIT_SCALE = 0.01          # game centimetres -> metres

# vertex record offsets (verified)
OFF_POS, OFF_NRM = 0x00, 0x10
OFF_UV = (0x18, 0x1C, 0x20, 0x24)
OFF_TAN, OFF_BIT = 0x28, 0x2C
OFF_W = 0x30
OFF_J = 0x38
STRIDE = 60


def find_blender() -> Optional[str]:
    for c in BLENDER_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


# ---------------------------------------------------------------- scene parse
def parse_scene_module():
    sys.path.insert(0, HERE)
    import parse_scene  # noqa
    return parse_scene


def find_units(pack: FpdArchive, kind: str) -> List[Dict]:
    units: Dict[str, Dict] = {}
    if kind == "tsf":
        pat = re.compile(r"root/assets/data/chr/tsf/([^/]+)/model/([^/]+)\.ma\.scn\.zstd$")
    elif kind == "beta":
        pat = re.compile(r"root/assets/data/chr/beta/([^/]+)/model/([^/]+)\.ma\.scn\.zstd$")
    else:
        pat = re.compile(r"root/assets/data/equip/([^/]+)/([^/]+)/model/([^/]+)\.ma\.scn\.zstd$")
    for e in pack.entries:
        if not e.name.endswith(".ma.scn.zstd") or e.size == 0:
            continue
        m = pat.match(e.name)
        if not m:
            continue
        if kind in ("tsf", "beta"):
            unit, model = m.group(1), m.group(2)
            if model != unit:
                continue
        else:
            cat, unit, model = m.group(1), m.group(2), m.group(3)
            if model != unit:
                continue
            unit = f"{cat}/{unit}"
        units.setdefault(unit, {"unit": unit, "model": e.name, "textures": [],
                                "kind": kind})
    for info in units.values():
        d = os.path.dirname(info["model"])
        for e in pack.entries:
            if not e.name.startswith(d + "/") or e.size == 0:
                continue
            if e.name.endswith(".txb.desktop.zstd"):
                role = classify_texture(e.name)
                if role:
                    info["textures"].append((role, e.name))
    return sorted(units.values(), key=lambda u: u["unit"])


# ------------------------------------------------------------------ per-unit
def tex_stem(filename: str) -> str:
    """Texture-set name for a txb filename: the part, with the PBR role suffix
    stripped.

    Splitting on the last underscore is wrong whenever a part name itself ends in
    a role-looking token -- `arm_l_nm` and `arm_r_nm` (part `arm_l_n`?) -- which
    made `arm_l` / `arm_r` lose their normal / roughness / metallic maps and left
    them with an unnamed duplicate.  Strip a known role suffix instead.
    """
    base = os.path.basename(filename).split(".")[0]
    for suf in sorted(TEX_ROLE, key=len, reverse=True):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def drawblock_spans(mesh, idx_len):
    """(offset, count) runs for a mesh.

    Prefer the parsed `indexCount`: it is exact whenever the tag walker decoded
    it, and its value is validated against the array length and the next offset.
    Only when it is missing or implausible (some BETA units encode it with a tag
    the walker does not decode, and box_laser_00 omits it entirely) fall back to
    the gap to the next offset.  Deriving unconditionally was wrong: a few units
    list a drawblock whose offset sits far before its own block, which made a
    mesh's span swallow the rest of the array.
    """
    dbs = [b for b in (mesh.get("drawblocks") or [])
           if b.get("indexOffset") is not None]
    if not dbs:
        return []
    offs = sorted({int(b["indexOffset"]) for b in dbs})
    out = []
    for k, o in enumerate(offs):
        end = offs[k + 1] if k + 1 < len(offs) else idx_len
        room = int(end) - o
        declared = None
        for b in dbs:
            if int(b["indexOffset"]) == o and b.get("indexCount"):
                declared = int(b["indexCount"])
                break
        c = declared if (declared and 3 <= declared <= room) else room
        c = min(c, room)
        if c >= 3:
            out.append((o, c // 3 * 3))
    return out


def _scene_to_rig(scene) -> Dict:
    """
    Convert a parse_scene.Scene using the parser's OWN verified exporter, so we
    inherit its bind-pose / joint-order handling exactly (its rig.json was
    cross-validated: 906/906 stored bindPose matrices reproduced, max err 1e-6).
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "rig.json")
        scene.export_json(tmp)
        rig = json.load(open(tmp))
    # export_json already emits vertexStart per mesh, in order
    for m in rig["meshes"]:
        # remember the declared count before any synthesis, so the span logic
        # below can tell a genuine value from a missing one
        m["_declared_vc"] = m.get("vertexCount") or 0
    base = 0
    for m in rig["meshes"]:
        if "vertexStart" not in m or m["vertexStart"] is None:
            m["vertexStart"] = base
        base = m["vertexStart"] + (m.get("vertexCount") or 0)
    # BETA meshes frequently omit or mis-state `vertexCount`, and a few units
    # (box_laser_00) use an indexCount encoding the tag walker does not decode,
    # so don't trust either field.  `indexOffset` IS reliable, so recover each
    # block's span from the neighbouring offsets and the array length, then take
    # a mesh's vertex span as the range its indices address.
    def blocks_of(m):
        return drawblock_spans(m, len(scene.idx) if scene.idx is not None else 0)

    def used(m):
        return bool(blocks_of(m))
    ms = rig["meshes"]
    total = rig.get("vertexCount") or 0
    # Preferred layout: the meshes' DECLARED vertexCount values partition the
    # buffer exactly.  That is the case for the great majority of units and it is
    # what the verified exports used -- 10 LOD0 meshes that Blender builds as 10
    # separate objects (chest / head / arm_l / leg / mounts ...).  Recomputing
    # spans from the index array instead collapses whole units into one mesh,
    # because their index values are absolute positions in a shared buffer.
    declared = [m.get("_declared_vc") or 0 for m in ms]
    if total and sum(declared) == total:
        cur = 0
        for m, vc_ in zip(ms, declared):
            m["vertexCount"] = vc_
            m["vertexStart"] = cur
            cur += vc_
        for m in ms:
            m.pop("_declared_vc", None)
        print(f"    layout: declared vertexCounts partition the buffer "
              f"({len([d for d in declared if d])} meshes)")
        return rig
    if scene.idx is not None and len(scene.idx):
        a = scene.idx
        ranges = []
        for m in ms:
            spans = blocks_of(m)
            if not spans:
                ranges.append(None)
                continue
            seg = np.concatenate([a[o:o + c] for o, c in spans])
            ranges.append((int(seg.min()), int(seg.max())))
        live = [r for r in ranges if r]
        # Topology, decided on the LOD0 meshes by how index values relate to the
        # buffer size:
        #  * GLOBAL numbering (most units): one shared buffer, indices are
        #    absolute, so the largest index equals the buffer's vertex count.
        #  * LOCAL numbering (F-14 / MiG-21 / YF-23 Pav-1): each mesh addresses
        #    its own blob and numbers from 0, so LOD0's largest index is far
        #    below the total and every LOD0 range starts at 0.
        total = rig.get("vertexCount") or 0
        lod0_ranges = [r for m, r in zip(ms, ranges)
                       if r and m.get("lod") in (None, 0)]
        gmax = max((hi for _lo, hi in lod0_ranges), default=0)
        all_zero = bool(lod0_ranges) and all(lo == 0 for lo, _ in lod0_ranges)
        local = (all_zero and gmax < total * 0.9)
        print(f"    topology: {'LOCAL (per-mesh blobs)' if local else 'GLOBAL index buffer'}"
              f"  lod0_max_index={gmax:,} vcount={total:,} "
              f"lod0_ranges_start_at_0={all_zero}")
        # A mesh's DECLARED vertexCount is correct whenever it is present, so
        # keep it and only synthesise the ones that are missing.
        cur = 0
        for m, r in zip(ms, ranges):
            declared = m.get("_declared_vc")
            if r is None:
                m["vertexCount"] = declared or 0
            elif declared:
                m["vertexCount"] = declared
            else:
                lo, hi = r
                m["vertexCount"] = (hi + 1) if local else (hi - lo + 1)
            m["vertexStart"] = cur
            cur += m["vertexCount"]
        # unindexed tail padding belongs to the last used mesh so the whole
        # buffer is accounted for exactly once
        total_v = rig.get("vertexCount") or 0
        if total_v > cur:
            for m in reversed(ms):
                if m["vertexCount"]:
                    m["vertexCount"] += total_v - cur
                    break
    else:
        base = 0
        for m in ms:
            m["vertexStart"] = base
            base += m.get("vertexCount") or 0
    for m in ms:
        m.pop("_declared_vc", None)
    return rig


# For TSF mechs the 10 LOD-0 meshes are the body parts below, in file order.
# This order was confirmed three ways: the mesh's own skin joints, the texture
# set names present in the pack, and the size of each texture set.
LOD0_PARTS = ["chest", "head", "j_unit", "arm_l", "mountb",
              "mountg", "leg", "arm_r", "mountb", "mountg"]

# Fallback for assets whose mesh count differs (weapons): infer the texture set
# from the joints the mesh actually deforms.  Order matters -- the more specific
# left/right rules must be tested before the generic ones.
PART_RULES = [
    ("arm_l",  ("_l,", "_l ", "elbow_l", "wrist_l", "scapula_l",
                "shoulder_l", "finger_l", "shoulder_guard1_l")),
    ("arm_r",  ("_r,", "_r ", "elbow_r", "wrist_r", "scapula_r",
                "shoulder_r", "finger_r", "shoulder_guard1_r")),
    ("mountb", ("bm_", "blade")),
    ("mountg", ("gm_", "grip")),
    ("j_unit", ("hip_unit", "thruster")),
    ("leg",    ("knee_", "ankle_", "toes_", "hip_l", "hip_r")),
    ("head",   ("head_m", "neck_m", "eye_")),
    ("chest",  ("chest_m", "spine1_m", "spine2_m", "spine3_m")),
]
DEFAULT_PART = "chest"


def part_from_joints(joint_names) -> Optional[str]:
    names = " " + ",".join(n.lower() for n in joint_names if n) + ","
    for part, keys in PART_RULES:
        if any(k in names for k in keys):
            return part
    return None


def resolve_part(mesh_index: int, lod, joint_names, n_lod0: int,
                 tex_stems=None) -> str:
    """Texture-set name for one mesh.

    TSF units have 10 LOD0 parts in a fixed file order, and most of those part
    names match a texture set directly.  Two do not: `j_unit` (hip thrusters) is
    textured by the `unitj` set.  Getting this wrong is expensive, because a
    name with no matching set used to fall back to "the first set in the unit" --
    which created a base-colour-only material, cached it under THAT part's name,
    and then handed it to the real mesh of that part, silently stripping its
    normal / roughness / metallic maps.  So map the known aliases and otherwise
    return a name only if the unit actually ships it.
    """
    ALIAS = {"j_unit": "unitj"}
    if lod in (None, 0) and n_lod0 == len(LOD0_PARTS) \
            and 0 <= mesh_index < len(LOD0_PARTS):
        want = ALIAS.get(LOD0_PARTS[mesh_index], LOD0_PARTS[mesh_index])
        if not tex_stems or want in tex_stems:
            return want
    if tex_stems:
        if "common" in tex_stems:
            return "common"
        return sorted(tex_stems)[0]
    return part_from_joints(joint_names) or DEFAULT_PART


def uv_preferred_shape(meshes_out) -> Optional[Tuple[int, int]]:
    """
    Atlas aspect ratio implied by the mesh UVs.

    The extent cannot be recovered from the base-mip byte count alone
    (2048*2048*16 == 4096*1024*16 == ...), and both readings decode without
    error -- they are just transposes of each other -- so we take the shape
    from the unwrap: the mean UV span in u maps to the width, in v to the
    height.
    """
    xs, ys = [], []
    for m in meshes_out:
        uv = m.get("uv0")
        if not uv:
            continue
        a = np.asarray(uv, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 64:
            continue
        xs.append(a[:, 0].max() - a[:, 0].min())
        ys.append(a[:, 1].max() - a[:, 1].min())
    if not xs:
        return None
    rx, ry = float(np.mean(xs)), float(np.mean(ys))
    if rx <= 1e-6 or ry <= 1e-6:
        return None
    if ry > rx * 1.15:                 # islands are taller than wide
        return (1024, 2048)
    if rx > ry * 1.15:
        return (2048, 1024)
    return (1024, 1024)


def _read_vertex_record(V, i, stride, off):
    """
    Read one attribute from vertex record `i` of a stride-`stride` array.

    vertexFormat 1 uses 60-byte records and vertexFormat 6 uses 64-byte records,
    but both place their attributes at the same offsets within the record (the
    extra 4 bytes are trailing padding).  A stride-64 buffer is therefore read
    by pacing at 64 and using the same field offsets.
    """
    return V[i, off:off + 4]


def _slice_positions(V, start, count, stride):
    """(count,3) float32 positions from a possibly stride-64 buffer."""
    base = V[start * stride:(start + count) * stride].reshape(count, stride)
    return base[:, 0:12].copy().view("<f4").reshape(count, 3)


def build_unit(pack: FpdArchive, info: Dict, ps, want_tex=True,
               tex_cache: Optional[Dict] = None,
               detach_neck: bool = False) -> Dict:
    """
    detach_neck: the blade/gun mounts ship a small bracket rigidly weighted to
    `Neck_M` (the game's own rig -- weight 1.0 on 1272 of p4's vertices).  That
    makes the mount swing when the head turns, which is correct in-game but
    surprising when posing.  With this flag those vertices are re-bound to the
    dominant joint of their own part instead, so the mount stays rigid.
    """
    name = info["unit"].replace("/", "__")
    t0 = time.time()
    print(f"  [{info['unit']}] parsing scene")
    scn = pack.payload(info["model"])
    scene = ps.Scene(scn, info["model"])
    # verify() is a diagnostic; a few units trip an empty-array check inside it
    # (meshes whose joint list is empty), which must not block the export.
    try:
        scene.verify(verbose=False)
    except Exception as exc:
        print(f"    (verify skipped: {type(exc).__name__}: {exc})")
    rig = _scene_to_rig(scene)
    vc = rig.get("vertexCount")
    ic = rig.get("indexCount")
    print(f"    {vc if vc is not None else '?'} verts  "
          f"{ic if ic is not None else '?'} indices  "
          f"{len(rig['nodes'])} nodes  {len(rig['meshes'])} meshes")

    # parse_scene locates the LOD0 vertex blob from the index buffer
    # (nv = max(index)+1) and records the stride it used.  That holds for most
    # units, but F-14 / MiG-21 / YF-23 Pav-1 give every mesh its own private blob
    # with mesh-local index numbering; detect that by checking whether the
    # triangles' winding agrees with the stored vertex normals.
    vstart = scene.voff
    STRIDE = scene.vstride
    vcount = scene.vcount
    if not vcount:
        raise ValueError("no vertices found")
    try:
        V = np.frombuffer(scn[vstart:vstart + vcount * STRIDE],
                          dtype=np.uint8).reshape(vcount, STRIDE)
    except ValueError as exc:
        raise ValueError(
            f"vertex blob @0x{vstart:x} stride {STRIDE} count {vcount} "
            f"overruns the entry ({len(scn)} B): {exc}")

    nrm_ref = np.frombuffer(V[:, 0x10:0x16].copy().tobytes(),
                            dtype="<f2").reshape(vcount, 3).astype(np.float64)
    pos_ref = np.frombuffer(V[:, 0:12].copy().tobytes(),
                            dtype="<f4").reshape(vcount, 3)
    sub = np.arange(0, len(scene.tris), max(1, len(scene.tris) // 30000))
    glob_score = mesh_blobs.winding_score(pos_ref, nrm_ref, scene.tris[sub])
    # BETA models keep one global vertex buffer whose slices are the meshes
    # (SUM(mesh vertexCount) == vcount), so the global path is the right one and
    # a per-mesh search would only mis-assign their shared blobs.
    per_mesh_ok = (info.get("kind", "tsf") != "beta")
    print(f"    global winding {glob_score:.2%}")
    per_mesh = None
    if per_mesh_ok and glob_score < 0.97:
        print(f"    global blob winding only {glob_score:.2%}; using the "
              f"per-mesh submesh extraction")
        import tri_units
        sol = tri_units.solve_unit(scn, scene)
        got = sum(1 for x in sol if x)
        winds = [x["wind"] for x in sol if x]
        print(f"    per-mesh: {got}/{len(scene.meshes)} meshes, median winding "
              f"{np.median(winds):.4%}, min {min(winds):.4%}")
        if got < len(scene.meshes):
            raise ValueError(f"only {got}/{len(scene.meshes)} meshes solved")
        per_mesh = sol
        vcount = sum(x["nverts"] for x in sol if x)
        rig["vertexCount"] = vcount
    layout = None

    meshes_out = []
    n_lod0 = sum(1 for m in rig["meshes"] if m.get("lod") in (None, 0))
    # texture sets this unit actually ships, keyed the same way the writer keys
    # its `materials` entries (filename stem without the role suffix)
    tex_stems = {tex_stem(t) for _role, t in (info.get("textures") or [])}
    mesh_vbase = 0            # running vertex offset in the merged output
    mesh_ibase = 0            # running index offset in out_tris
    local_indices = False     # per-mesh submeshes renumber from 0
    out_tris = []
    written = []              # (vertexStart, nverts, nindices) in mesh order
    for mi, mesh in enumerate(rig["meshes"]):
        voff = mesh.get("vertexStart") or 0
        vn = mesh.get("vertexCount") or 0
        if per_mesh is not None:
            x = per_mesh[mi]
            if x is None:
                print(f"    ! mesh {mi}: no index block, skipped")
                continue
            boff, bstride, bn = x["blob"]
            keep = x["keep"]
            vn = len(keep)
            voff = mesh_vbase
            # gather exactly the records this mesh references
            recs = np.frombuffer(scn[boff:boff + bn * bstride],
                                 dtype=np.uint8).reshape(bn, bstride)
            sl = np.ascontiguousarray(recs[keep])
            mesh_vbase += vn
            # triangles are already renumbered 0..vn-1
            out_tris.append(x["tri"].reshape(-1))
            mesh_ibase += x["tri"].size
            local_indices = True
            written.append((voff, vn, x["tri"].size))
        elif layout is not None:
            loc = layout[mi]
            if loc is None or vn == 0:
                print(f"    ! mesh {mi}: not located, skipped")
                continue
            off, st = loc
            if off + vn * st > len(scn):
                print(f"    ! mesh {mi}: slice past EOF, skipped")
                continue
            sl = np.frombuffer(scn[off:off + vn * st],
                               dtype=np.uint8).reshape(vn, st)
        else:
            # Global-index unit: the index values are ABSOLUTE vertex numbers.
            # Extract exactly the records this mesh references and renumber them
            # 0..n-1.  Slicing the whole buffer by the declared vertexCount looks
            # equivalent -- those counts do sum to vcount -- but they do NOT
            # correspond to the ranges the indices actually address, so most
            # triangles fell outside their own slice and Blender dropped them
            # (loose vertices, exploded limbs).  Submesh extraction is immune to
            # that mismatch and costs only duplicated vertex records.
            spans = drawblock_spans(
                mesh, len(scene.idx) if scene.idx is not None else 0)
            if not spans or vn == 0:
                print(f"    ! mesh {mi}: no index block, skipped")
                continue
            tri = np.concatenate([scene.idx[o_:o_ + c_]
                                  for (o_, c_) in spans]).astype(np.int64)
            tri = tri[:tri.size // 3 * 3].reshape(-1, 3)
            used = np.unique(tri.ravel())
            if used.max() >= len(V):
                print(f"    ! mesh {mi}: index {int(used.max())} beyond the "
                      f"vertex buffer ({len(V)}), skipped")
                continue
            sl = np.ascontiguousarray(V[used])
            vn = len(sl)
            remap = np.zeros(len(V) + 1, dtype=np.int64)
            remap[used] = np.arange(vn, dtype=np.int64)
            voff = mesh_vbase
            mesh_vbase += vn
            out_tris.append(remap[tri].reshape(-1))
            mesh_ibase += tri.size
            local_indices = True
            written.append((voff, vn, tri.size))
            mesh["drawblocks"] = [{"indexOffset": o_, "indexCount": c_}
                                  for (o_, c_) in spans]
            mesh["vertexStart"] = voff
        # mesh_vbase is advanced inside each branch that assigns voff
        pos = sl[:, OFF_POS:OFF_POS + 12].copy().view("<f4").reshape(vn, 3)
        nrm = sl[:, OFF_NRM:OFF_NRM + 6].copy().view("<f2").reshape(vn, 3)
        uv0 = sl[:, OFF_UV[0]:OFF_UV[0] + 4].copy().view("<f2").reshape(vn, 2)
        uv1 = sl[:, OFF_UV[1]:OFF_UV[1] + 4].copy().view("<f2").reshape(vn, 2)
        tan = sl[:, OFF_TAN:OFF_TAN + 3].astype(np.int8).astype(np.float64) / 127.0
        bit = sl[:, OFF_BIT:OFF_BIT + 3].astype(np.int8).astype(np.float64) / 127.0
        # skin offsets are NOT fixed: 60-byte records keep weights at 0x30 /
        # joints at 0x38, 64-byte records at 0x34 / 0x3C.  Reading the wrong one
        # made most vertices look bound to a single joint, which is exactly the
        # "missing binding" symptom.  A unit with no skeleton at all (kakasi, a
        # static prop) has no skin block either, so it comes back unskinned.
        sk_list = rig["skeletons"]
        si = mesh.get("skeletonIndex")
        nj_here = (len(sk_list[si]["nodeNames"])
                   if isinstance(si, int) and 0 <= si < len(sk_list) else 0)
        wts, jts, skin_off = mesh_blobs.skin_of(sl, nj_here, sl.shape[1])
        if skin_off is None:
            print(f"    mesh {mi}: no skin block in this record layout; "
                  f"exporting unrigged")

        # weights: some sampled records can hold NaN/Inf, so sanitise and
        # renormalise -- an all-zero row would silently unbind that vertex
        w = np.nan_to_num(wts.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)
        nj = nj_here
        # secondary weight slots of a few records carry joint numbers outside
        # this mesh's skeleton (~1% of the weight).  Left as-is they make Blender
        # DROP the whole vertex from the skin, which is the "missing binding"
        # symptom, so clamp them onto a valid joint instead.
        if nj:
            oob = (jts >= nj) | (jts < 0)
            if oob.any():
                n_fix = int(oob.any(axis=1).sum())
                wasted = float(w[oob].sum())
                jts = np.where(oob, 0, jts)
                w = np.where(oob, 0.0, w)
                print(f"    mesh {mi}: clamped {n_fix} vertices' out-of-range "
                      f"joint slots (dropped weight {wasted:.1f})")
        s = w.sum(axis=1, keepdims=True)
        zero = (s[:, 0] <= 1e-9)
        if zero.any():
            if nj:
                # fall back to a rigid bind on the vertex's first declared joint
                # so the vertex is never left unweighted
                w[zero] = 0.0
                w[np.flatnonzero(zero), 0] = 1.0
                jts[zero, 0] = np.clip(jts[zero, 0], 0, max(0, nj - 1))
                print(f"    mesh {mi}: {int(zero.sum())} vertices had no usable "
                      f"weights; bound rigidly to their first joint")
            else:
                # unrigged static prop: no skeleton to bind to at all
                w[zero] = 0.0
            s = w.sum(axis=1, keepdims=True)
        s[s <= 1e-9] = 1.0
        w = w / s

        sk = (rig["skeletons"][mesh["skeletonIndex"]]
              if nj else {"nodeNames": []})
        joint_names = sk["nodeNames"]

        # optionally detach mount geometry that the game parents to the neck
        part_guess = resolve_part(mi, mesh.get("lod"), joint_names, n_lod0)
        if detach_neck and part_guess in ("mountb", "mountg"):
            neck = [i for i, nm in enumerate(joint_names) if nm == "Neck_M"]
            if neck and jts.size and w.size:
                ni = neck[0]
                # vertices whose DOMINANT influence is the neck
                dom = jts[np.arange(len(jts)), w.argmax(axis=1)]
                onneck = (dom == ni) & (w.max(axis=1) > 0.5)
                if onneck.any():
                    others = [k for k, nm in enumerate(joint_names)
                              if nm not in (None, "Neck_M")]
                    if others:
                        counts = {k: int((dom == k).sum()) for k in others}
                        tgt = max(counts, key=counts.get)
                        jts[onneck, :] = 0
                        jts[onneck, 0] = tgt
                        w[onneck, :] = 0.0
                        w[onneck, 0] = 1.0
                        print(f"    detached {int(onneck.sum())} Neck_M verts "
                              f"of mesh {mi} -> {joint_names[tgt]}")

        if nj and len(sk.get("nodes") or []) != len(joint_names):
            raise ValueError(
                f"mesh {mi}: skeleton {mesh.get('skeletonIndex')} has "
                f"{len(sk.get('nodes') or [])} nodes but {len(joint_names)} names")
        # sanity: joint indices must address this mesh's own skeleton
        jmax = int(jts.max()) if jts.size else -1
        if nj and jmax >= len(joint_names):
            raise ValueError(
                f"mesh {mi}: joint index {jmax} >= skeleton size "
                f"{len(joint_names)} after clamping -- bug in the weight path")
        meshes_out.append({
            "index": mi,
            "vertexStart": voff,
            "vertexCount": vn,
            "localIndices": local_indices,
            "lod": mesh.get("lod"),
            "part": resolve_part(mi, mesh.get("lod"), joint_names, n_lod0,
                                 tex_stems),
            "skeletonIndex": mesh["skeletonIndex"],
            "jointNames": joint_names,
            "drawblocks": mesh.get("drawblocks", []),
            "positions": np.round(pos.astype(np.float64) * UNIT_SCALE, 6).tolist(),
            "normals": np.round(nrm.astype(np.float64), 5).tolist(),
            "tangents": np.round(tan, 5).tolist(),
            "bitangents": np.round(bit, 5).tolist(),
            "uv0": np.round(uv0.astype(np.float64), 6).tolist(),
            "uv1": np.round(uv1.astype(np.float64), 6).tolist(),
            "weights": np.round(w, 6).tolist(),
            "joints": jts.tolist(),
        })

    nodes = [{
        "index": n["index"], "name": n["name"], "uid": n["uid"],
        "parentIndex": n.get("parentIndex"),
        "worldMatrix": n["worldMatrix"],
    } for n in rig["nodes"]]
    skeletons = [{"nodes": s["nodes"], "nodeNames": s["nodeNames"],
                  "bindPose": s["bindPose"]} for s in rig["skeletons"]]

    out = {
        "name": name, "unit": info["unit"], "source": info["model"],
        "scale": UNIT_SCALE, "nodes": nodes,
        "skeletons": skeletons, "meshes": meshes_out, "materials": [],
        "vertices": int(vcount),
    }
    # one self-contained folder per model: <unit>/<unit>.json + <unit>/textures/
    unit_dir = os.path.join(OUT_ROOT, name)
    tex_dir = os.path.join(unit_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    idx = scene.idx if scene.idx is not None else np.zeros(0, np.uint32)
    if out_tris:
        # out_tris holds one contiguous run per mesh in mesh order, and every
        # index in a run is local to that mesh (0..nverts-1).  blender_rig reads
        # mesh i's run as idx[vertexStart : vertexStart + 3*nfaces], so verify
        # the cursor and the counts line up before writing anything.
        cur_v = cur_i = 0
        for k, (vs, vn_, ni) in enumerate(written):
            # each mesh's run starts exactly where the previous one ended; there
            # is no fixed index:vertex ratio, so check the vertex cursor and that
            # the run lengths are whole triangles
            if vs != cur_v:
                print(f"    !! write cursor: mesh#{k} start={vs} nverts={vn_} "
                      f"nindices={ni} running_v={cur_v}")
                raise ValueError(f"vertex cursor desync at mesh {k}: "
                                 f"{vs} != {cur_v}")
            if ni % 3:
                raise ValueError(f"index run {ni} is not a multiple of 3")
            cur_v += vn_
            cur_i += ni
        idx = np.concatenate(out_tris).reshape(-1).astype(np.uint32)
        if len(idx) != cur_i:
            raise ValueError(f"index length {len(idx)} != expected {cur_i}")
        # each mesh now carries its own extracted records, so the real total is
        # what was written -- not the source buffer's size
        vcount = cur_v
        rig["vertexCount"] = vcount
        if len(idx) and int(idx.max()) >= vcount:
            raise ValueError(f"index {int(idx.max())} out of range "
                             f"({vcount} vertices)")
        # blender_rig needs each mesh's index cursor; there is no fixed ratio
        cur_i = 0
        for m, (vs, vn_, ni) in zip(meshes_out, written):
            m["indexStart"] = cur_i
            cur_i += ni
    np.save(os.path.join(unit_dir, name + ".indices.npy"),
            np.asarray(idx, dtype=np.uint32))

    if want_tex:
        tex_cache = {} if tex_cache is None else tex_cache
        # the mesh UVs tell us the atlas shape; passing it in removes the
        # w/h ambiguity that the byte count alone cannot resolve
        pref = uv_preferred_shape(meshes_out)
        if pref:
            print(f"    atlas shape from UVs: {pref[0]}x{pref[1]}")
        for role, tname in sorted(info["textures"],
                                  key=lambda t: t[0]):
            stem = os.path.basename(tname).split(".")[0]
            part = tex_stem(tname)
            rel = f"textures/{stem}.png"
            abs_p = os.path.join(unit_dir, rel)
            if not os.path.exists(abs_p):
                key = (tname,)
                if key not in tex_cache:
                    try:
                        w_, h_, fmt_, arr = txb_fast.decode(
                            pack.payload(tname), prefer=pref)
                        tex_cache[key] = (w_, h_, fmt_, arr)
                    except Exception as exc:
                        print(f"    ! tex {stem} ({role}): {exc}")
                        tex_cache[key] = None
                tex = tex_cache[key]
                if tex is None:
                    continue
                w_, h_, fmt_, arr = tex
                from PIL import Image
                os.makedirs(os.path.dirname(abs_p), exist_ok=True)
                Image.fromarray(arr, "RGBA").save(abs_p)
                print(f"    tex {part}/{role}: {w_}x{h_} {fmt_}")
            if os.path.exists(abs_p):
                out["materials"].append({"part": part, "role": role,
                                         "png": rel})
    print(f"    done in {time.time()-t0:.1f}s")
    return out


# --------------------------------------------------------------------- blender
def run_blender(blender: str, json_path: str, outdir: str) -> bool:
    script = os.path.join(HERE, "blender_rig.py")
    cmd = [blender, "--background", "--factory-startup", "--python", script,
           "--", json_path, outdir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print("      blender|", line)
    if r.returncode != 0:
        print("      blender FAILED")
        print((r.stderr or "")[-2000:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tsf", "beta", "weapon"], default="tsf")
    ap.add_argument("--name", nargs="*", default=None,
                    help="one or more unit codes; omit for all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--no-blender", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--detach-neck", action="store_true",
                    help="re-bind mount geometry that the game parents to "
                         "Neck_M, so head rotation leaves the mounts alone")
    args = ap.parse_args()

    # BETA enemies live in their own output tree so they never mix with the TSFs
    global OUT_ROOT
    if args.kind != "tsf":
        # keep the trees apart, but honour MIKHAIL_OUT
        OUT_ROOT = os.path.join(os.path.dirname(_cfg.OUT.rstrip(os.sep)),
                                os.path.basename(_cfg.OUT) + "_" + args.kind)

    pack = FpdArchive(PACK)
    ps = parse_scene_module()
    units = find_units(pack, args.kind)
    if args.name:
        wanted = set(args.name)
        units = [u for u in units
                 if u["unit"] in wanted
                 or any(u["unit"].endswith("/" + w) or u["unit"] == w
                        for w in wanted)]
    if args.limit:
        units = units[: args.limit]
    print(f"pack.bin: {pack.count:,} entries; units: {len(units)}")
    for u in units:
        print(f"   {u['unit']}")

    blender = None if (args.no_blender or args.json_only) else find_blender()
    if blender:
        print("blender:", blender)

    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest = []
    for info in units:
        try:
            built = build_unit(pack, info, ps, want_tex=not args.no_tex,
                               detach_neck=args.detach_neck)
        except Exception as exc:
            import traceback
            print(f"  !! {info['unit']}: {exc}")
            traceback.print_exc()
            continue
        # everything for this model lives in its own folder
        unit_dir = os.path.join(OUT_ROOT, built["name"])
        os.makedirs(unit_dir, exist_ok=True)
        dst = os.path.join(unit_dir, built["name"] + ".json")
        with open(dst, "w") as f:
            json.dump(built, f)
        entry = {"name": built["name"], "unit": info["unit"],
                 "folder": unit_dir, "json": dst,
                 "meshes": len(built["meshes"]),
                 "verts": sum(m["vertexCount"] for m in built["meshes"]),
                 "nodes": len(built["nodes"]),
                 "tex": len(built["materials"]), "fbx": None, "glb": None}
        if blender:
            if run_blender(blender, dst, unit_dir):
                entry["fbx"] = os.path.join(unit_dir, built["name"] + ".fbx")
                entry["glb"] = os.path.join(unit_dir, built["name"] + ".glb")
        manifest.append(entry)
        print(f"  => {built['name']}: {entry['verts']:,} verts, "
              f"{entry['nodes']} nodes, {entry['tex']} tex -> {unit_dir}")

    mf = os.path.join(OUT_ROOT, "manifest.json")
    with open(mf, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(manifest)} units -> {OUT_ROOT}\nmanifest: {mf}")


if __name__ == "__main__":
    main()
