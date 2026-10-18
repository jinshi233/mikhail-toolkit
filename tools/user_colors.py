#!/usr/bin/env python3
"""
Read the material's user palette.

`_cm` is a REGION MASK, not a colour map: it is flat red / green / blue / yellow
/ black blobs.  The actual colours are shader parameters stored in the material:

    UserColorMask            the mask texture
    UserColor_Red
    UserColor_Green
    UserColor_Blue
    UserColor_Yellow
    emissiveColor / emissiveColorIntensity

That is why eyes that should be yellow render blue: the mask selects a region and
the palette decides its colour, so reading the mask as an emissive colour gives
nonsense.
"""
import os
import struct
import sys
import zlib

import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "mikhail"))
from mikhail_assets import FpdArchive
import parse_scene as ps

GAME = _cfg.GAME
pack = FpdArchive(_cfg.PACK)

NAMES = ["UserColor_Red", "UserColor_Green", "UserColor_Blue",
         "UserColor_Yellow", "emissiveColor", "AlbedoColor", "albedoColor",
         "emissiveColorIntensity", "emissiveScale"]


def key(name):
    return struct.pack("<I", zlib.crc32(name.encode()) & 0xFFFFFFFF)


def read_value(d, off):
    """Decode the scalar/vector value that starts at `off`."""
    if off >= len(d):
        return None
    t = d[off]
    op = t & 0x7F
    if op == 0x11:            # f32
        return struct.unpack_from("<f", d, off + 1)[0]
    if op == 0x10:            # f16
        return struct.unpack_from("<e", d, off + 1)[0]
    if op == 0x0C:            # u32
        return struct.unpack_from("<I", d, off + 1)[0]
    if op in (0x0A,):
        return struct.unpack_from("<H", d, off + 1)[0]
    if op == 0x09:
        return d[off + 1]
    if op in (0x13, 0x14, 0x96, 0x97, 0x98, 0x99, 0x9A):
        # array: count then element tag
        if op == 0x13:
            n = d[off + 1]
            p = off + 2
        elif op == 0x14:
            n = struct.unpack_from("<H", d, off + 1)[0]
            p = off + 3
        else:
            n = (op - 0x96) + 1
            p = off + 1
        et = d[p] & 0x7F
        p += 1
        size = {0x0F: 1, 0x09: 1, 0x0A: 2, 0x11: 4}.get(et)
        if size is None:
            return None
        vals = []
        for i in range(n):
            q = p + i * size
            if et == 0x11:
                vals.append(struct.unpack_from("<f", d, q)[0])
            elif et == 0x10:
                vals.append(struct.unpack_from("<e", d, q)[0])
            elif et == 0x0A:
                vals.append(struct.unpack_from("<H", d, q)[0])
            else:
                vals.append(d[q])
        return vals
    return None


for unit in sys.argv[1:] or ["f22a_00", "f15j_00"]:
    d = pack.payload(f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd")
    print(f"\n=== {unit}")
    for nm in NAMES:
        k = key(nm)
        p = 0
        hits = []
        while True:
            q = d.find(k, p)
            if q < 0 or q + 8 > len(d):
                break
            v = read_value(d, q + 4)
            if v is not None:
                hits.append(v)
            p = q + 1
        if hits:
            uniq = []
            for h in hits:
                if h not in uniq:
                    uniq.append(h)
            print(f"   {nm:<24} {len(hits):>3} refs, distinct={uniq[:6]}")
