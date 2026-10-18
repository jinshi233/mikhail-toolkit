#!/usr/bin/env python3
"""
Parse the Material records: name, palette, and which texture is the mask.

The Material section is a list of records; each holds a `name` (chest_mat,
head_mat, ...), a `textures` array of {name: "fileN", relative: "..."} entries,
and shader parameters.  `UserColorMask` names one of those textures -- that is
the region mask, and it is what decides the eye colour together with the
`UserColor_*` palette.
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


def crc(nm):
    return struct.pack("<I", zlib.crc32(nm.encode()) & 0xFFFFFFFF)


CRC_NAME = crc("name")


def read_str(d, off):
    """value tag + NUL-terminated ASCII"""
    p = off
    for _ in range(4):
        if p >= len(d):
            return None
        t = d[p]
        if (t & 0x7F) in (0x06, 0x26):
            e = d.find(b"\x00", p + 1)
            if 0 < e - (p + 1) < 300:
                s = d[p + 1:e]
                if all(32 <= c < 127 for c in s):
                    return s.decode("ascii")
        p += 1
    return None


def read_f32x4(d, off):
    """`84 84 9a 91` then four little-endian f32."""
    for skip in range(0, 10):
        p = off + skip
        if p + 2 < len(d) and (d[p] & 0x7F) == 0x1A and (d[p + 1] & 0x7F) == 0x11:
            try:
                return list(struct.unpack_from("<4f", d, p + 2))
            except struct.error:
                return None
    return None


FIELDS = ["UserColorMask", "UserColor_Red", "UserColor_Green",
          "UserColor_Blue", "UserColor_Yellow", "emissiveColor",
          "emissiveColorIntensity", "albedoColor"]


def parse_materials(d):
    """[(name, {field: value})] in file order."""
    MAT = b"\x83\x03" + struct.pack("<I", zlib.crc32(b"Material") & 0xFFFFFFFF)
    out = []
    p = 0
    while True:
        q = d.find(MAT, p)
        if q < 0:
            break
        # the record ends at the next class marker or the next Material
        nxt = d.find(b"\x85\x85\x83", q + 6)
        rec_end = nxt if nxt > 0 else min(len(d), q + 4000)
        rec = d[q:rec_end]
        nm = None
        nq = rec.find(CRC_NAME)
        if nq >= 0:
            nm = read_str(rec, nq + 4)
        vals = {}
        for f in FIELDS:
            k = rec.find(crc(f))
            if k < 0:
                continue
            if f.startswith("UserColor_") or f in ("emissiveColor", "albedoColor"):
                v = read_f32x4(rec, k + 4)
                if v:
                    vals[f] = [round(x, 4) for x in v]
            elif f == "UserColorMask":
                v = read_str(rec, k + 4)
                if v:
                    vals[f] = v
            else:
                for skip in range(1, 8):
                    t = rec[k + 4 + skip]
                    if (t & 0x7F) == 0x11:
                        vals[f] = round(struct.unpack_from("<f", rec, k + 5 + skip)[0], 4)
                        break
        out.append((nm, vals))
        p = q + 1
    return out


for unit in sys.argv[1:] or ["f22a_00"]:
    d = pack.payload(f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd")
    print(f"=== {unit}")
    for nm, vals in parse_materials(d):
        if not vals:
            continue
        print(f"  {nm}")
        for k, v in vals.items():
            print(f"      {k:<24} {v}")
