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

from mikhail_assets import FpdArchive, classify_texture, TEX_ROLE  # noqa: E402
import txb_fast  # noqa: E402
import mixed_walk  # noqa: E402
import mesh_blobs  # noqa: E402

GAME = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL"
PACK = os.path.join(GAME, "obb", "pack.bin")
UNPACK = os.path.dirname(HERE)
OUT_ROOT = os.path.join(UNPACK, "export")

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


def scene_texture_refs(data: bytes, model_dir: str, pack_names):
    """Texture files a scene references by relative path.

    Units do not necessarily ship their own atlas: type94_jp_00 keeps only
    leg_bc.png locally and points at siblings for the rest, e.g.
    "../../type94_02/model/head_ao.png" and
    "../../type94_jp_rs_00/model/arm_l_cm.png".  Scanning only the unit's own
    directory therefore produced parts with no texture set at all, which is why
    those meshes came out with no material or the wrong one.

    Returns [(role, entry_name)] resolved against the archive.
    """
    rels = []
    i = 0
    while True:
        j = data.find(b".png", i)
        if j < 0:
            break
        a = j
        while a > 0 and 32 <= data[a - 1] < 127:
            a -= 1
        if a > 0 and (data[a - 1] & 0x7F) in (0x06, 0x26):
            rel = data[a:j + 4].decode("ascii", "ignore")
            if "/" in rel:
                rels.append(rel)
        i = j + 4
    out, seen = [], set()
    for rel in rels:
        full = os.path.normpath(os.path.join(model_dir, rel)).replace(os.sep, "/")
        cand = full + ".txb.desktop.zstd"
        if cand in pack_names and cand not in seen:
            role = classify_texture(cand)
            if role:
                seen.add(cand)
                out.append((role, cand))
    return out



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


def drawblock_spans(mesh, idx_len, next_offsets=None):
    """(offset, count) runs for a mesh.

    Prefer the parsed `indexCount`: it is exact whenever the tag walker decoded
    it, and its value is validated against the array length and the next offset.
    Only when it is missing or implausible (some BETA units encode it with a tag
    the walker does not decode, and box_laser_00 omits it entirely) fall back to
    the gap to the next known offset.

    `next_offsets` must be the sorted offsets of EVERY drawblock in the scene, not
    just this mesh's.  Using only this mesh's offsets made the last block run to
    the end of the array: F-15J's chest block (indexCount missing) then claimed
    315,264 indices and swallowed the whole buffer, which is why that unit's
    "chest" object came out as a 192,273-vertex full-body mesh.
    """
    dbs = [b for b in (mesh.get("drawblocks") or [])
           if b.get("indexOffset") is not None]
    if not dbs:
        return []
    offs = sorted({int(b["indexOffset"]) for b in dbs})
    glob = sorted(int(o) for o in (next_offsets or []))
    out = []
    for k, o in enumerate(offs):
        end = offs[k + 1] if k + 1 < len(offs) else idx_len
        # tighten the end with the nearest block start after this one, anywhere
        # in the scene
        for g in glob:
            if g > o:
                end = min(end, g)
                break
        room = int(end) - o
        declared = None
        for b in dbs:
            if int(b["indexOffset"]) == o and b.get("indexCount"):
                declared = int(b["indexCount"])
                break
        c = declared if (declared and 3 <= declared <= room) else room
        c = min(c, room)
        mat_id = None
        for b in dbs:
            if int(b["indexOffset"]) == o:
                mat_id = b.get("materialIndex")
                break
        if c >= 3:
            out.append((o, c // 3 * 3, mat_id))
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
        # drawblock_spans now returns (offset, count, materialIndex)
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
            seg = np.concatenate([a[o:o + c] for o, c, _m in spans])
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


# Every TSF mesh record carries `nodeIndex`, and that node is named after the
# body part.  This is the authoritative mapping and it REPLACES the old
# positional table, which silently mislabelled whole unit families: the order is
# not fixed.  Most units list chest,head,j_unit,l_arm,l_b_mount,l_g_mount,leg,
# r_arm,r_b_mount,r_g_mount, but the F-15 family puts both arms LAST and every
# Type-94 puts `chest` last -- so a position-based guess gave the real chest mesh
# a mount's material and name, which is exactly "the chest is missing".
#
# Part label -> texture set.  The left and right mounts share one set each, and
# `j_unit` is textured by the set called `unitj`; this matches the material names
# recorded in the file itself (chest_mat, head_mat, unitj_mat, arm_l_mat,
# arm_r_mat, mountb_mat, mountg_mat, leg_mat).
PART_SETS = {
    "chest": "chest", "head": "head", "j_unit": "unitj",
    "l_arm": "arm_l", "r_arm": "arm_r", "leg": "leg",
    "l_b_mount": "mountb", "r_b_mount": "mountb",
    "l_g_mount": "mountg", "r_g_mount": "mountg",
}

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


def node_map(scene) -> Dict:
    """nodeIndex -> node name, for resolving a mesh to its body part."""
    return {n.get("nodeIndex"): n.get("name") for n in scene.nodes}


def part_of_node(name: Optional[str]) -> Optional[str]:
    """Part label for a mesh's node name.

    Handles the few suffixed variants: `head_shadow` (kakasi's shadow proxies)
    and `l_b_mount1` (the YF-23 ships a second pair of mounts).
    """
    if not name:
        return None
    if name in PART_SETS:
        return name
    base = name.rstrip("0123456789")
    if base in PART_SETS:
        return name              # keep `l_b_mount1` distinct for labelling
    if name.endswith("_shadow"):
        stem = name[: -len("_shadow")]
        if stem in PART_SETS or stem.rstrip("0123456789") in PART_SETS:
            return name
    return None


def texset_of(part: Optional[str], tex_stems=None) -> Optional[str]:
    """Texture set for a part label, if the unit ships it."""
    if not part:
        return None
    base = part.rstrip("0123456789").replace("_shadow", "")
    want = PART_SETS.get(base) or PART_SETS.get(part)
    if want and (not tex_stems or want in tex_stems):
        return want
    return None


def resolve_part(mesh_index: int, lod, joint_names, n_lod0: int,
                 tex_stems=None, node_name: Optional[str] = None) -> str:
    """Part label for one mesh.

    Prefers the mesh's own node name; falls back to the joints it deforms, and
    only then to position.  The material key is derived separately (`texset_of`),
    so a part can be labelled `l_b_mount`/`r_b_mount` while both use the one
    `mountb` texture set.
    """
    got = part_of_node(node_name)
    if got:
        return got
    guess = part_from_joints(joint_names)
    if guess:
        return guess
    if lod in (None, 0) and 0 <= mesh_index < 10:
        return ["chest", "head", "j_unit", "l_arm", "l_b_mount",
                "l_g_mount", "leg", "r_arm", "r_b_mount", "r_g_mount"][mesh_index]
    return DEFAULT_PART


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

    # Merge in textures the scene references elsewhere.  Units such as
    # type94_jp_00 ship almost no atlas of their own and point at sibling model
    # directories, so a local-directory scan leaves most parts without a set.
    model_dir = os.path.dirname(info["model"])
    pack_names = {e.name for e in pack.entries}
    have_tex = {t for _r, t in info.get("textures") or []}
    extra = [t for t in scene_texture_refs(scn, model_dir, pack_names)
             if t[1] not in have_tex]
    if extra:
        info.setdefault("textures", []).extend(extra)
        print(f"    +{len(extra)} texture(s) resolved from relative references")
    # verify() is a diagnostic; a few units trip an empty-array check inside it
    # (meshes whose joint list is empty), which must not block the export.
    try:
        scene.verify(verbose=False)
    except Exception as exc:
        print(f"    (verify skipped: {type(exc).__name__}: {exc})")
    rig = _scene_to_rig(scene)
    nidx2name = node_map(scene)
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
    # every drawblock start in the scene, so a block with a missing indexCount
    # can be bounded by the next block anywhere, not by the end of the array
    all_block_offsets = sorted({
        int(b["indexOffset"])
        for m_ in rig["meshes"] for b in (m_.get("drawblocks") or [])
        if b.get("indexOffset") is not None})
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
                mesh, len(scene.idx) if scene.idx is not None else 0,
                all_block_offsets)
            if not spans or vn == 0:
                print(f"    ! mesh {mi}: no index block, skipped")
                continue
            tri = np.concatenate([scene.idx[o_:o_ + c_]
                                  for (o_, c_, _m) in spans]).astype(np.int64)
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
            mesh["drawblocks"] = [{"indexOffset": o_, "indexCount": c_,
                                   "materialIndex": m_}
                                  for (o_, c_, m_) in spans]
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
        node_name = nidx2name.get(mesh.get("nodeIndex"))
        part_guess = resolve_part(mi, mesh.get("lod"), joint_names, n_lod0,
                                  node_name=node_name)
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
        label = resolve_part(mi, mesh.get("lod"), joint_names, n_lod0,
                             tex_stems, node_name=node_name)
        meshes_out.append({
            "index": mi,
            "vertexStart": voff,
            "vertexCount": vn,
            "localIndices": local_indices,
            "lod": mesh.get("lod"),
            "part": label,
            "texset": texset_of(label, tex_stems),
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
    # Which drawblocks belong to the shared "glow" material?  Every mesh carries
    # a second drawblock whose materialIndex is the SAME value across chest,
    # head, j_unit, leg and both arms -- that is the eyes / sensors material.
    # Folding it into the part material is why the visor took the atlas colour
    # instead of the palette's glow colour.
    # Count how many DISTINCT body parts use each materialIndex, LOD0 only.  The
    # glow material is the one shared by several parts (chest + head + j_unit +
    # leg + both arms); a part's own material is used by one part only, or by its
    # left/right pair at most.  Counting every LOD instead made each index appear
    # five times and flagged all of them.
    _mat_parts = {}
    for m_ in meshes_out:
        if m_.get("lod") not in (None, 0):
            continue
        for idx_ in {b.get("materialIndex") for b in (m_.get("drawblocks") or [])
                     if b.get("materialIndex") is not None}:
            _mat_parts.setdefault(idx_, set()).add(m_.get("part"))
    glow_ids = {k for k, v in _mat_parts.items() if len(v) >= 3}
    for m_ in meshes_out:
        m_["drawblockGlow"] = [bool(b.get("materialIndex") in glow_ids)
                               for b in (m_.get("drawblocks") or [])]
    if glow_ids:
        print(f"    glow material index(es): {sorted(glow_ids)}")

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
                # Opaque maps are written as RGB on purpose.  Blender's FBX
                # exporter wires the Base Color image's alpha into the Principled
                # Alpha socket, and the DCC importer then honours it -- with a
                # grey albedo the alpha channel reads ~0.19, so every mesh came in
                # almost fully transparent and rendered as a black silhouette (or
                # vanished).  No alpha channel, no alpha link.
                if arr.shape[2] == 4 and int(arr[:, :, 3].min()) >= 250:
                    Image.fromarray(arr[:, :, :3], "RGB").save(abs_p)
                else:
                    Image.fromarray(arr, "RGBA").save(abs_p)
                print(f"    tex {part}/{role}: {w_}x{h_} {fmt_}")
            if os.path.exists(abs_p):
                # `_cm` classifies as "emissive" by suffix, but it is the
                # UserColor region mask; labelling it emissive made Blender drive
                # Emission Color from flat red/blue/green blobs.
                real_role = "userColorMask" if stem.endswith("_cm") else role
                out["materials"].append({"part": part, "role": real_role,
                                         "png": rel})
        # bake the user-colour emissive maps from the region masks
        for m in write_emissive_for_unit(info, unit_dir, pack, tex_cache, pref):
            if not any(x["part"] == m["part"] and x["role"] == "emissive"
                       for x in out["materials"]):
                out["materials"].append(m)
    print(f"    done in {time.time()-t0:.1f}s")
    return out


# ---------------------------------------------------------------- user colours
# `_cm` is NOT an emissive colour map.  It is a flat REGION MASK: every texel is
# pure red, green, blue or black, and the material's `UserColor_*` parameters
# supply the colour for each slot.  Using the mask directly as an emission colour
# is what painted whole models with flat red/blue/green blotches and turned the
# eye strip blue.  The palette below is the engine default and was verified
# identical across every TSF, so it is not per-unit data.
USER_COLORS = {
    "red":   (1.00, 0.04, 0.00),
    "green": (0.22, 1.00, 0.36),
    "blue":  (0.00, 0.68, 1.00),
    "yellow": (0.80, 0.88, 0.27),
}

# Which palette entry each mask slot takes.  The visor / sensor strip sits in the
# mask's BLUE slot, so that slot maps to YELLOW -- confirmed against the model's
# own region layout (the blue region is the Y-band of the face).  `red` covers
# ~80% of the body and is the plain armour, so it gets no emission at all.
SLOT_EMISSIVE = {"blue": "yellow", "green": "green"}


def bake_user_emissive(mask_arr, strength_px=None):
    """Turn the region mask into an emissive colour map via the palette."""
    import numpy as np
    c = mask_arr[:, :, :3].astype(np.int16)
    r, g, b = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    out = np.zeros((c.shape[0], c.shape[1], 3), dtype=np.float32)
    slots = {
        "red":    (r > 170) & (g < 70) & (b < 70),
        "green":  (g > 170) & (r < 70) & (b < 70),
        "blue":   (b > 170) & (r < 70) & (g < 70),
    }
    for slot, mask in slots.items():
        target = SLOT_EMISSIVE.get(slot)
        if not target:
            continue
        col = USER_COLORS[target]
        out[mask, 0] = col[0]
        out[mask, 1] = col[1]
        out[mask, 2] = col[2]
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def write_emissive_for_unit(info, unit_dir, pack, tex_cache, pref):
    """Bake `<part>_em.png` for every part that ships a mask."""
    import numpy as np
    from PIL import Image
    made = []
    for role, tname in list(info.get("textures") or []):
        if role != "userColorMask" and "_cm" not in os.path.basename(tname):
            continue
        part = tex_stem(tname)
        stem = os.path.basename(tname).split(".")[0]
        try:
            key = (tname,)
            if key not in tex_cache or tex_cache[key] is None:
                tex_cache[key] = txb_fast.decode(pack.payload(tname), prefer=pref)
            _w, _h, _f, arr = tex_cache[key]
        except Exception:
            continue
        em = bake_user_emissive(arr)
        rel = f"textures/{part}_em.png"
        Image.fromarray(em, "RGB").save(os.path.join(unit_dir, rel))
        made.append({"part": part, "role": "emissive", "png": rel})
    return made



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
    if args.kind == "beta":
        OUT_ROOT = os.path.join(UNPACK, "export_beta")
    elif args.kind == "weapon":
        OUT_ROOT = os.path.join(UNPACK, "export_weapon")

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
