#!/usr/bin/env python3
"""
Recover the material -> texture chain from the tag stream.

Observed at 0xe94a5c:

    87 d9 37 5b 6e | a6 "../../common_texture/scratch/scratch_rh.png" 00
    87 06 7e 23 5e | a6 "file6" 00            (0x5e237e06 = crc32("name"))

so the `textures` array is a list of {name: "fileN", <pathField>: "<path>"}.
A Material then names one of those, which is the authoritative mesh -> texture
link that positional guessing has been getting wrong.
"""
import os
import struct
import sys
import zlib
from collections import Counter, defaultdict

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

unit = sys.argv[1] if len(sys.argv) > 1 else "f22a_00"
d = pack.payload(f"root/assets/data/chr/tsf/{unit}/model/{unit}.ma.scn.zstd")
sc = ps.Scene(d, unit)

# every candidate field name we know of: pool + our guesses
CAND = set(sc.pool)
for extra in ("name", "path", "file", "textureName", "texturePath",
              "AlbedoColorTexture", "NormalTexture", "AOTexture",
              "MetallicTexture", "RoughnessTexture", "Texture",
              "textureIndex", "shaderName", "materialIndex", "materials",
              "textures", "Material", "Shader", "id"):
    CAND.add(extra)
by_crc = {zlib.crc32(c.encode()) & 0xFFFFFFFF: c for c in CAND}


def key(name):
    return struct.pack("<I", zlib.crc32(name.encode()) & 0xFFFFFFFF)


def str_at(off):
    """value tag then NUL-terminated ASCII"""
    if off >= len(d):
        return None
    tag = d[off]
    if tag & 0x7F not in (0x06, 0x26):
        tag = None
    p = off + 1 if tag is not None else off
    q = d.find(b"\x00", p)
    if q < 0 or q - p < 2 or q - p > 300:
        return None
    s = d[p:q]
    if all(32 <= c < 127 for c in s):
        return s.decode("ascii")
    return None


print(f"=== {unit}: field hashes that carry a string value")
seen = defaultdict(Counter)
p = 0
while True:
    q = d.find(b"\x87", p)
    if q < 0 or q + 10 > len(d):
        break
    crc = struct.unpack_from("<I", d, q + 1)[0]
    s = str_at(q + 5)
    if s and (crc in by_crc or "/" in s or s.endswith(".png")):
        seen[crc][s] += 1
    p = q + 1

for crc, vals in sorted(seen.items(), key=lambda kv: -sum(kv[1].values())):
    nm = by_crc.get(crc)
    print(f"\n  0x{crc:08x}  {nm if nm else '<unknown>'}")
    for v, c in vals.most_common(8):
        print(f"      {c:>4}x  {v}")
