#!/usr/bin/env python3
"""
mikhail_assets.py — Project MIKHAIL asset pipeline (fuzz-eg / AGES Mk2 engine).

Layers, all verified:

  obb/pack.bin            FPD\0 archive, XOR-scrambled (64 KiB keystream)
    entry .zstd           Zstandard frame
      EPK\0 container     fuzz-eg tag stream
        .ma.scn           static mesh scene: index + vertex buffers
        .txb              texture (BC7 / BC5 / BC1 / BC3 / raw RGBA8)

FPD layout (reverse-engineered by DaZombieKiller / kurikomoe for the
Fate/stay night REMASTERED release of the same engine; keystream verified
identical for Project MIKHAIL):

  0x00 'FPD\\0' | u32BE version | u64BE fileCount | u64BE dataStart | 32B zero
  0x38 entry table, fileCount * 32 bytes, XOR-scrambled from keystream byte 0:
        u64BE nameOffset | u64BE dataOffset | u64BE dataLength | u64BE usize
  then  zlib-deflated NUL-separated UTF-8 name blob
  each payload is XOR-scrambled independently from keystream byte 0;
  if usize != 0 the payload is zlib-deflated after descrambling.

.ma.scn layout (verified on f22a_00):

  0x00 'EPK\\0' | u16 ver | u16 flags | u32LE payloadLen | u32 zero
  0x10        sorted NUL-terminated field-name pool (the schema)
  ~0xc00      index buffer, u32LE * N, all < vertexCount
  ~0x13e118   scene / node tag stream  (skeleton, materials, nodes)
  ~0x1539a0   vertex array: vertexCount records of 60 bytes
   tail       more descriptors

60-byte vertex record (verified):

  +0x00 f32 x, +0x04 f32 y, +0x08 f32 z, +0x0c f32 = 1.0
  +0x10 f16 x3 normal, +0x16 u16 = 0
  +0x18 f16 x2 uv0
  +0x1c f16 x2 uv1
  +0x20 f16 x2 uv0 again (duplicate stream)
  +0x24 f16 x2 uv1 again (duplicate stream)
  +0x28 8 bytes, 4th byte of each u32 always 0
  +0x30 u16 x3
  +0x36 u16 = 0
  +0x38 u16 in [0,21]   +0x3a u8 in [0,12]   +0x3b = 0
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# FPD archive
# --------------------------------------------------------------------------
HDR_SIZE = 0x38
KEY_SIZE = 65536
HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from .config import KEY as DEFAULT_KEY
except ImportError:
    import config as _cfg
    DEFAULT_KEY = _cfg.KEY


def load_keystream(path: str = DEFAULT_KEY) -> np.ndarray:
    with open(path, "rb") as f:
        k = f.read()
    if len(k) != KEY_SIZE:
        raise ValueError(f"keystream must be {KEY_SIZE} bytes, got {len(k)}")
    return np.frombuffer(k, dtype=np.uint8)


def xor_stream(buf: bytes, keys: np.ndarray, key_offset: int = 0) -> bytes:
    n = len(buf)
    if n == 0:
        return buf
    idx = (np.arange(n, dtype=np.int64) + key_offset) % KEY_SIZE
    a = np.frombuffer(buf, dtype=np.uint8).copy()
    a ^= keys[idx]
    return a.tobytes()


@dataclass
class Entry:
    name: str
    offset: int
    size: int
    usize: int
    idx: int


class FpdArchive:
    def __init__(self, path: str, keys: Optional[np.ndarray] = None):
        self.path = path
        self.keys = load_keystream() if keys is None else keys
        self._f = open(path, "rb")
        head = self._f.read(HDR_SIZE)
        if head[:4] != b"FPD\x00":
            raise ValueError(f"{path}: not an FPD archive ({head[:4]!r})")
        self.version = struct.unpack(">I", head[4:8])[0]
        self.count = struct.unpack(">Q", head[8:16])[0]
        self.data_start = struct.unpack(">Q", head[16:24])[0]
        self._read_table()

    def _read_table(self):
        tbl_len = self.data_start - HDR_SIZE
        self._f.seek(HDR_SIZE)
        dec = xor_stream(self._f.read(tbl_len), self.keys, 0)
        u = np.frombuffer(dec[: self.count * 32], dtype=">u8").reshape(-1, 4)
        names_blob = zlib.decompress(dec[self.count * 32:])
        self.entries: List[Entry] = []
        for i in range(self.count):
            no, do, dl, ul = (int(x) for x in u[i])
            end = names_blob.find(b"\x00", no)
            raw = names_blob[no:end]
            try:
                nm = raw.decode("utf-8")
            except UnicodeDecodeError:
                nm = raw.decode("cp932", "replace")
            self.entries.append(Entry(nm, do, dl, ul, i))
        self.by_name: Dict[str, Entry] = {e.name: e for e in self.entries}

    def raw(self, name: str) -> bytes:
        e = self.by_name[name]
        if e.size == 0:
            return b""
        self._f.seek(self.data_start + e.offset)
        data = xor_stream(self._f.read(e.size), self.keys, 0)
        if e.usize:
            data = zlib.decompress(data)
        return data

    def payload(self, name: str) -> bytes:
        d = self.raw(name)
        if name.endswith(".zstd"):
            import zstandard
            d = zstandard.ZstdDecompressor().decompress(d, max_output_size=1 << 31)
        return d

    def find(self, substring: str, suffix: Optional[str] = None) -> List[str]:
        out = []
        for e in self.entries:
            if substring in e.name and (suffix is None or e.name.endswith(suffix)):
                out.append(e.name)
        return out


# --------------------------------------------------------------------------
# EPK container
# --------------------------------------------------------------------------
EPK_HEADER = 16


def epk_header(d: bytes) -> Tuple[int, int, int]:
    if d[:4] != b"EPK\x00":
        raise ValueError(f"not an EPK payload ({d[:4]!r})")
    ver, flags = struct.unpack_from("<HH", d, 4)
    length = struct.unpack_from("<I", d, 8)[0]
    return ver, flags, length


def epk_name_pool(d: bytes, start: int = 0x10) -> Tuple[List[str], int]:
    """Return (names, end_offset) of the sorted NUL-terminated field-name pool."""
    ident = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    names: List[str] = []
    cur = None
    pos = start
    n = len(d)
    while pos < min(n, 0x40000):
        c = d[pos]
        if c in ident:
            if cur is None:
                cur = pos
            pos += 1
        elif c == 0 and cur is not None:
            names.append(d[cur:pos].decode("ascii"))
            cur = None
            pos += 1
        else:
            break
    return names, pos


# --------------------------------------------------------------------------
# scene mesh extraction
# --------------------------------------------------------------------------
VERTEX_STRIDE = 60
ONE_F32 = b"\x00\x00\x80\x3f"          # float 1.0, at +0x0c of every vertex


def find_vertex_array(d: bytes, min_run: int = 64
                      ) -> Optional[Tuple[int, int, int]]:
    """Locate the vertex array by the stride-60 float 1.0 sentinel."""
    hits, i = [], 0
    while True:
        j = d.find(ONE_F32, i)
        if j == -1:
            break
        hits.append(j)
        i = j + 1
    if len(hits) < min_run:
        return None
    hits_arr = np.asarray(hits)
    diffs = np.diff(hits_arr)
    runs, start = [], 0
    for k in range(1, len(hits_arr)):
        if diffs[k - 1] != VERTEX_STRIDE:
            if k - start > min_run:
                runs.append((start, k - 1))
            start = k
    if len(hits_arr) - start > min_run:
        runs.append((start, len(hits_arr) - 1))
    if not runs:
        return None
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return int(hits_arr[s]) - 0xC, e - s + 1, VERTEX_STRIDE


def find_index_buffer(d: bytes, vstart: int, vcount: int,
                      min_len: int = 3000) -> Optional[Tuple[int, int]]:
    """Longest contiguous u32 run below vcount, ending before the vertices."""
    nw = vstart // 4
    u = np.frombuffer(d[: nw * 4], dtype="<u4").astype(np.int64)
    ok = u < vcount
    best, i, N = None, 0, len(u)
    while i < N:
        if ok[i]:
            j = i
            while j < N and ok[j]:
                j += 1
            if j - i >= min_len and (best is None or (j - i) > (best[1] - best[0])):
                best = (i, j)
            i = j
        else:
            i += 1
    if best is None:
        return None
    return best[0] * 4, best[1] * 4


@dataclass
class SceneMesh:
    positions: np.ndarray                 # (N,3) float32
    normals: np.ndarray                   # (N,3) float32
    uvs: List[np.ndarray]                 # list of (N,2) float32
    indices: np.ndarray                   # (M,) uint32
    vertex_start: int
    index_start: int
    names: List[str] = field(default_factory=list)
    aux: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def triangles(self) -> np.ndarray:
        return self.indices.reshape(-1, 3)


def parse_scene(d: bytes, want_aux: bool = True) -> SceneMesh:
    names, pool_end = epk_name_pool(d)
    va = find_vertex_array(d)
    if va is None:
        raise ValueError("vertex array not found")
    vstart, vcount, stride = va
    ib = find_index_buffer(d, vstart, vcount)
    if ib is None:
        raise ValueError("index buffer not found")

    V = np.frombuffer(d[vstart:vstart + vcount * stride],
                      dtype=np.uint8).reshape(vcount, stride)
    pos = V[:, 0:12].copy().view("<f4").reshape(vcount, 3).astype(np.float32)
    nrm = V[:, 0x10:0x16].copy().view("<f2").reshape(vcount, 3).astype(np.float32)
    uv0 = V[:, 0x18:0x1c].copy().view("<f2").reshape(vcount, 2).astype(np.float32)
    uv1 = V[:, 0x1c:0x20].copy().view("<f2").reshape(vcount, 2).astype(np.float32)
    idx = np.frombuffer(d[ib[0]:ib[1]], dtype="<u4").astype(np.uint32)

    aux: Dict[str, np.ndarray] = {}
    if want_aux:
        aux["raw"] = V
        aux["u16_30"] = V[:, 0x30:0x36].copy().view("<u2").reshape(vcount, 3)
        aux["u16_38"] = V[:, 0x38:0x3a].copy().view("<u2").reshape(vcount)
        aux["u8_39"] = V[:, 0x39].copy()
        aux["u8_3a"] = V[:, 0x3a].copy()
        aux["blob28"] = V[:, 0x28:0x2c].copy()
        aux["blob2c"] = V[:, 0x2c:0x30].copy()

    return SceneMesh(pos, nrm, [uv0, uv1], idx, vstart, ib[0], names, aux)


# --------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------
BC_FORMATS = {
    "BC1_UNORM": ("bc1", 8), "BC1_UNORM_SRGB": ("bc1", 8),
    "BC3_UNORM": ("bc3", 16), "BC3_UNORM_SRGB": ("bc3", 16),
    "BC4_UNORM": ("bc4", 8),
    "BC5_UNORM": ("bc5", 16), "BC5_SNORM": ("bc5", 16),
    "BC6H_UF16": ("bc6", 16), "BC6H_SF16": ("bc6", 16),
    "BC7_UNORM": ("bc7", 16), "BC7_UNORM_SRGB": ("bc7", 16),
    "BC7_SRGB": ("bc7", 16),
}
RAW_FORMATS = {
    "R8G8B8A8_SRGB": 4, "R8G8B8A8_UNORM": 4, "R8G8B8A8": 4,
    "B8G8R8A8_SRGB": 4, "B8G8R8A8_UNORM": 4,
    "R8G8B8_SRGB": 3, "R8_UNORM": 1, "R8": 1,
}


@dataclass
class Texture:
    width: int
    height: int
    format: str
    rgba: np.ndarray          # (h,w,4) uint8
    mips: int = 1


def mip_chain_bytes(w: int, h: int, bpb: int) -> int:
    """Total bytes of a block-compressed mip chain down to 4x4."""
    tot, cw, ch = 0, w, h
    while cw >= 4 and ch >= 4:
        tot += (cw // 4) * (ch // 4) * bpb
        cw //= 2
        ch //= 2
    return tot


def read_txb_format(d: bytes) -> Optional[str]:
    i = d.find(b"format\x00")
    if i < 0:
        return None
    p = i + len(b"format\x00")
    for k in range(0, 12):
        j = d.find(b"\x00", p + k)
        if j < 0:
            continue
        s = d[p + k:j]
        if s and all(32 <= c < 127 for c in s):
            return s.decode()
    return None


def read_txb_extent(d: bytes, fmt: Optional[str] = None
                    ) -> Optional[Tuple[int, int]]:
    """
    Recover the texture extent (w, h).

    The tag stream does not keep a stable name/value ordering, so rather than
    depend on it we use the payload's own size bookkeeping:

      * every mip level is preceded by its byte length as a u32
      * the base level therefore gives w*h = size / bytes-per-pixel

    We scan the header for a u32 whose value equals a complete block-compressed
    or raw mip chain, then derive the power-of-two (w, h) from it.
    """
    header = d[:0x400]
    bpp = None
    if fmt in BC_FORMATS:
        bpp = BC_FORMATS[fmt][1]
    elif fmt in RAW_FORMATS:
        bpp = RAW_FORMATS[fmt]

    def sq_candidates(size, bpp):
        """(w, h) power-of-two pairs whose mip chain equals `size`."""
        out = []
        for w in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4):
            for h in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4):
                if mip_chain_bytes(w, h, bpp) == size:
                    out.append((w, h))
        return out

    # The reliable anchor: the base mip level's own u32 byte count sits just
    # before the pixel blob, e.g. chest_bc.png.txb
    #     0x4c u32 = 4,194,304 (= 2048*2048 BC7 base level)
    #     0x50    pixels start
    #     0x58 u32 = 3 (mip count)   0x5c = 0
    # So look for any u32 that equals w*h*bpp for a power-of-two pair.
    for o in range(0x10, min(0x400, len(d) - 4)):
        v = struct.unpack_from("<I", d, o)[0]
        if v < 4096 or v > len(d) or v % bpp:
            continue
        base_elems = v // bpp
        cands = []
        for w in (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4):
            if base_elems % w:
                continue
            h = base_elems // w
            if h in (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4):
                # The base-level byte count is ambiguous (512*512*16 ==
                # 1024*512*8 == ...).  Break the tie with the full mip chain:
                # the real chain must fit inside the payload, and we prefer the
                # largest such (w, h).
                chain = mip_chain_bytes(w, h, bpp)
                if chain > len(d):
                    continue
                cands.append((-(w * h), abs(w - h), w, h))
        if cands:
            cands.sort()
            return cands[0][2], cands[0][3]
    return None


def _decode_base(d: bytes, start: int, w: int, h: int, kind: str) -> Optional[np.ndarray]:
    """Decode the base mip at `start`, returning (h,w,4) uint8 or None."""
    import texture2ddecoder as t2d
    n = (w // 4) * (h // 4) * BC_FORMATS_BPB[kind]
    if start + n > len(d):
        return None
    try:
        px = getattr(t2d, f"decode_{kind}")(d[start:start + n], w, h)
    except Exception:
        return None
    return np.frombuffer(px, dtype=np.uint8).reshape(h, w, 4)


BC_FORMATS_BPB = {"bc1": 8, "bc3": 16, "bc4": 8, "bc5": 16, "bc6": 16, "bc7": 16}


def parse_txb(d: bytes) -> Texture:
    """
    Decode a .txb payload into RGBA8.

    Determining (w, h) is ambiguous from the header alone because the byte
    counts of different power-of-two extents coincide.  We therefore decode
    every plausible extent and pick the one that yields a *coherent* image:
    real texture atlases have strong local pixel correlation, whereas a wrong
    extent produces noise.  (This is the method that was verified by eye on
    chest_bc: the correct answer 2048x2048 decodes to the F-22A chest atlas.)
    """
    fmt = read_txb_format(d)
    if fmt is None:
        raise ValueError("txb: no format field")
    if fmt in BC_FORMATS:
        kind, bpb = BC_FORMATS[fmt]
    elif fmt in RAW_FORMATS:
        kind = None
        bpb = RAW_FORMATS[fmt]
    else:
        raise ValueError(f"txb: unsupported format {fmt}")

    ext = read_txb_extent(d, fmt)

    if kind is None:
        # uncompressed: the blob is the tail of the file
        if ext is None:
            raise ValueError("txb: no extent for raw format")
        w, h = ext
        base = w * h * bpb
        start = len(d) - base
        a = np.frombuffer(d[start:start + base], dtype=np.uint8).reshape(h, w, bpb)
        if bpb == 4:
            if fmt.startswith("B8G8R8A8"):
                a = a[:, :, [2, 1, 0, 3]]
            return Texture(w, h, fmt, np.ascontiguousarray(a))
        if bpb == 3:
            return Texture(w, h, fmt,
                           np.dstack([a, np.full((h, w, 1), 255, np.uint8)]))
        return Texture(w, h, fmt,
                       np.dstack([a, a, a, np.full((h, w, 1), 255, np.uint8)]))

    # --- block compressed -----------------------------------------------
    # Header facts (verified on chest_bc.png.txb):
    #   0x4c u32 = byte size of the BASE mip level (4,194,304 = 2048*2048*16)
    #   0x50    base mip block data starts
    #   then    the remaining mip levels follow contiguously
    #
    # Square textures store their 4x4 blocks in plain raster order.  The
    # non-square ones (1024x2048, 512x1024) store the block grid TRANSPOSED
    # (column-major); reading those as raster puts a seam on every 4th pixel
    # row, which is the horizontal banding seen on those maps.  We therefore
    # decode both orders and keep the smoother result.
    best = None
    for w, h in trials:
        need = (w // 4) * (h // 4) * bpb
        if need > len(d) or need < 4096:
            continue
        for start in range(0x20, min(len(d) - need, 0x200)):
            for order_tag, order in _block_orders(w // 4, h // 4).items():
                try:
                    a = _decode_ordered(d[start:start + need], w, h, kind, bpb, order)
                except Exception:
                    continue
                if a is None:
                    continue
                g = a[:, :, :3].astype(np.float32).mean(axis=2)
                sc = float(np.abs(np.diff(g, axis=1)).mean()
                           + np.abs(np.diff(g, axis=0)).mean())
                if best is None or sc < best[0]:
                    best = (sc, w, h, start, a, order_tag)
    if best is None:
        raise ValueError("txb: no decodable extent found")
    _, w, h, start, a, order_tag = best
    if kind == "bc4":
        # decode_bc4 puts its single channel in BLUE, not red
        r = a[:, :, 2]
        a = np.dstack([r, r, r, np.full((h, w), 255, np.uint8)])
    return Texture(w, h, fmt, a)


def _block_orders(bw: int, bh: int):
    """Candidate 4x4-block orders inside a compressed mip level."""
    n = bw * bh
    lin = np.arange(n)
    by, bx = np.divmod(lin, bw)
    return {
        "raster": lin,                 # row-major block grid
        "colmajor": bx * bh + by,      # transposed block grid
    }


def _decode_ordered(blob: bytes, w: int, h: int, kind: str, bpb: int, order):
    import texture2ddecoder as t2d
    nblk = (w // 4) * (h // 4)
    if len(blob) < nblk * bpb:
        return None
    blocks = np.frombuffer(blob[:nblk * bpb], dtype=np.uint8).reshape(nblk, bpb)
    data = blocks[order].tobytes()
    px = getattr(t2d, f"decode_{kind}")(data, w, h)
    return np.frombuffer(px, dtype=np.uint8).reshape(h, w, 4)


# texture-name suffix -> PBR role
TEX_ROLE = {
    "_bc": "baseColor", "_nm": "normal", "_ao": "occlusion",
    "_mt": "metalness", "_rh": "roughness", "_cm": "emissive",
    "_d": "detail", "_sp": "specular",
}


def classify_texture(filename: str) -> Optional[str]:
    base = os.path.basename(filename)
    for suf, role in TEX_ROLE.items():
        if f"{suf}." in base or base.endswith(suf):
            return role
    return None
