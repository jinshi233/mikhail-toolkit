#!/usr/bin/env python3
"""
Resolve materialIndex -> material name -> part, and check it against the mesh's
own skeleton.  This is the authoritative mapping: the file names its materials
after the part (`chest`, `head`, `j_unit`, `l_arm`, `r_arm`, `l_b_mount`,
`l_g_mount`, `leg`), so no positional guessing is needed.
"""
import os
import struct
import sys
import zlib

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

CRC_NAME = 0x5E237E06          # crc32("name")
CRC_MATERIAL = zlib.crc32(b"Material") & 0xFFFFFFFF
NAME_KEY = struct.pack("<I", CRC_NAME)
MAT_MARK = b"\x83\x03" + struct.pack("<I", CRC_MATERIAL)


def parse_materials(d):
    """Ordered list of material names, in the order they appear."""
    out = []
    p = 0
    while True:
        q = d.find(MAT_MARK, p)
        if q < 0:
            break
        # the name field follows within the record
        nq = d.find(NAME_KEY, q)
        nm = None
        if 0 <= nq < q + 400:
            k = nq + 4
            while k < len(d) and d[k] == 0x84:
                k += 1
            if k < len(d) and d[k] & 0x7F in (0x06, 0x26):
                e = d.find(b"\x00", k + 1)
                if 0 < e - (k + 1) < 80:
                    s = d[k + 1:e]
                    if all(32 <= c < 127 for c in s):
                        nm = s.decode("ascii")
        out.append((q, nm))
        p = q + 1
    return out


def skel_part(names):
    s = " " + " ".join(n.lower() for n in names if n) + " "
    def has(*ks):
        return any(k in s for k in ks)
    if has("hip_unit") and not has("chest_m", "knee"):
        return "j_unit"
    if has("elbow", "wrist", "finger", "shoulder"):
        return "l_arm" if has("shoulder_l", "elbow_l", "wrist_l", "scapula_l",
                              "shoulder_guard1_l") or has("pinkyfinger3_l") else "r_arm"
    if has("knee", "ankle", "toes"):
        return "leg"
    if has("head_m", "neck_m"):
        return "head"
    if has("chest_m", "spine1_m", "spine2_m", "spine3_m"):
        return "chest"
    if has("bm_", "blade", "b_mount"):
        return "l_b_mount"
    if has("gm_", "grip", "g_mount"):
        return "l_g_mount"
    return "?"


for unit in sys.argv[1:] or ["f22a_00", "f16_00", "f15j_00", "ef2000_00", "f14_00"]:
    d = pack.payload(f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd")
    sc = ps.Scene(d, unit)
    nodes = [n["name"] for n in sc.nodes]
    mats = parse_materials(d)
    names = [n for _, n in mats]
    print(f"\n=== {unit}: {len(mats)} Material records, "
          f"{len(set(n for n in names if n))} distinct names")
    from collections import Counter
    print("   names:", dict(Counter(n for n in names if n)))
    print(f"   {'mesh':>4} {'matIdx':>8} {'material':<12} {'skeleton says':<12} verdict")
    for mi, m in enumerate(sc.meshes):
        if m.get("lod") not in (None, 0):
            continue
        mid = [b.get("materialIndex") for b in (m.get("drawblocks") or [])]
        primary = mid[0] if mid else None
        nm = names[primary] if primary is not None and primary < len(names) else None
        sk = sc.skeletons[m.get("skeletonIndex") or 0]
        sn = [nodes[sc.byuid[u]] for u in (sk.get("nodes") or []) if u in sc.byuid]
        gp = skel_part(sn)
        verdict = "ok" if (nm == gp or (nm in ("l_b_mount", "l_g_mount") and gp == nm)) else "MISMATCH"
        print(f"   {mi:>4} {str(primary):>8} {str(nm):<12} {gp:<12} {verdict}")
