#!/usr/bin/env python3
"""
Resolve the material's texture indices against the file's texture table.

A texture reference is encoded as `87 <crc32(field)> 8a <u8 index>`, so a
material names its maps by index into the `Texture` record list.  This tells us
which FILE is `UserColorMask` -- previously assumed to be `_cm` without proof.
"""
import os
import struct
import sys
import zlib

WS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "mikhail"))
from mikhail_assets import FpdArchive
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from mikhail import config as _cfg
except ImportError:
    import config as _cfg

GAME = _cfg.GAME
pack = FpdArchive(_cfg.PACK)

TEX_MARK = b"\x83\x03" + struct.pack("<I", zlib.crc32(b"Texture") & 0xFFFFFFFF)
MAT_MARK = b"\x83\x03" + struct.pack("<I", zlib.crc32(b"Material") & 0xFFFFFFFF)


def crc(nm):
    return struct.pack("<I", zlib.crc32(nm.encode()) & 0xFFFFFFFF)


def read_name(rec, off):
    p = off
    for _ in range(4):
        if p >= len(rec):
            return None
        if (rec[p] & 0x7F) in (0x06, 0x26):
            e = rec.find(b"\x00", p + 1)
            if 0 < e - (p + 1) < 300:
                s = rec[p + 1:e]
                if all(32 <= c < 127 for c in s):
                    return s.decode("ascii")
        p += 1
    return None


def texture_table(d):
    """[relative path] in record order."""
    out = []
    p = 0
    while True:
        q = d.find(TEX_MARK, p)
        if q < 0:
            break
        rec = d[q:q + 600]
        rel = None
        k = rec.find(crc("relative"))
        if k >= 0:
            rel = read_name(rec, k + 4)
        out.append(rel)
        p = q + 1
    return out


def materials(d):
    out = []
    p = 0
    while True:
        q = d.find(MAT_MARK, p)
        if q < 0:
            break
        rec = d[q:q + 3000]
        nm = None
        k = rec.find(crc("name"))
        if k >= 0 and k < 260:
            nm = read_name(rec, k + 4)
        refs = {}
        i = 0
        while True:
            j = rec.find(b"\x87", i)
            if j < 0 or j + 8 > len(rec):
                break
            c = struct.unpack_from("<I", rec, j + 1)[0]
            if (rec[j + 5] & 0x7F) == 0x0A:
                refs[c] = rec[j + 6]
            i = j + 1
        out.append((nm, refs))
        p = q + 1
    return out


unit = sys.argv[1] if len(sys.argv) > 1 else "f22a_00"
d = pack.payload(f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd")
tex = texture_table(d)
print(f"{unit}: {len(tex)} Texture records")
for i, t in enumerate(tex[:12]):
    print(f"   [{i:>3}] {t}")

FIELDS = ["AlbedoColorTexture", "NormalTexture", "AOTexture",
          "MetallicTexture", "RoughnessTexture", "UserColorMask",
          "CamouflageTexture", "DetailAlbedoColorTexture"]
BYCRC = {zlib.crc32(f.encode()) & 0xFFFFFFFF: f for f in FIELDS}

print("\nresolved references per material:")
seen = set()
for nm, refs in materials(d):
    if not nm or nm in seen:
        continue
    seen.add(nm)
    print(f"  {nm}")
    for c, idx in sorted(refs.items()):
        f = BYCRC.get(c)
        t = tex[idx] if idx < len(tex) else "<out of range>"
        if f:
            print(f"      {f:<24} idx={idx:<4} -> {t}")
