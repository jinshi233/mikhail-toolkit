#!/usr/bin/env python3
"""
Fast .txb texture decoder.

Verified layout:
    pixel data starts at 0x50 always
    extent (w,h) : the base mip is (w/4)*(h/4)*bpb bytes
    format       : string in the header ('BC7_SRGB', 'BC5_UNORM', ...)
    block order  : raster for square textures, column-major for 1024x2048 /
                   512x1024 (reading those as raster puts a seam on every 4th
                   pixel row)

The extent is genuinely ambiguous from the byte count alone
(2048*2048*16 == 4096*1024*16), so instead of brute-forcing we take a hint:
the mesh UVs tell us the atlas shape.  When no hint is available we pick the
most square candidate that fits the payload.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np

import texture2ddecoder as t2d

BPB = {"bc1": 8, "bc2": 16, "bc3": 16, "bc4": 8, "bc5": 16, "bc6": 16, "bc7": 16}
KIND = {
    "BC1_UNORM": "bc1", "BC1_UNORM_SRGB": "bc1",
    "BC2_UNORM": "bc2", "BC2_UNORM_SRGB": "bc2",
    "BC3_UNORM": "bc3", "BC3_UNORM_SRGB": "bc3",
    "BC4_UNORM": "bc4", "BC4_SNORM": "bc4",
    "BC5_UNORM": "bc5", "BC5_SNORM": "bc5",
    "BC6H_UF16": "bc6", "BC6H_SF16": "bc6",
    "BC7_UNORM": "bc7", "BC7_UNORM_SRGB": "bc7", "BC7_SRGB": "bc7",
}


def _rgba(a):
    """texture2ddecoder returns BLUE-first pixels (verified with a hand-decoded
    BC1 block whose endpoint 0 is pure red: the library reports [0,0,255] where
    the format says [255,0,0]).  Swap so the array really is R,G,B,A -- without
    this every red marking in the game (the PLAAF star on the J-8, warning
    decals, faction stripes) comes out blue and every blue comes out red.
    """
    return np.ascontiguousarray(a[:, :, [2, 1, 0, 3]])


def _decode(blob: bytes, w: int, h: int, kind: str, bpb: int, colmajor: bool):
    bw, bh = w // 4, h // 4
    nblk = bw * bh
    blocks = np.frombuffer(blob[:nblk * bpb], dtype=np.uint8).reshape(nblk, bpb)
    if colmajor:
        by, bx = np.divmod(np.arange(nblk), bw)
        blocks = blocks[bx * bh + by]
    if kind == "bc2":
        # DXT3: each 16-byte block is 8 bytes of explicit alpha followed by an
        # 8-byte BC1 colour block.  texture2ddecoder has no bc2 entry point, so
        # gather the colour halves and decode those as BC1.
        col = np.concatenate([blocks[:, 8:], blocks[:, :8]], axis=1)[:, :8]
        px = t2d.decode_bc1(col.tobytes(), w, h)
        return _rgba(np.frombuffer(px, dtype=np.uint8).reshape(h, w, 4))
    px = getattr(t2d, f"decode_{kind}")(blocks.tobytes(), w, h)
    return _rgba(np.frombuffer(px, dtype=np.uint8).reshape(h, w, 4))
RAW = {"R8G8B8A8_SRGB": 4, "R8G8B8A8_UNORM": 4, "R8G8B8A8": 4,
       "B8G8R8A8_SRGB": 4, "B8G8R8A8_UNORM": 4, "R8_UNORM": 1, "R8": 1}

PIXEL_OFFSET = 0x50          # verified on square and non-square textures alike
SHAPES = [(4096, 4096), (2048, 2048), (4096, 2048), (2048, 4096),
          (2048, 1024), (1024, 2048), (1024, 1024), (2048, 512), (512, 2048),
          (1024, 512), (512, 1024), (512, 512), (1024, 256), (256, 1024),
          (512, 256), (256, 512), (256, 256), (128, 512), (512, 128),
          (128, 256), (256, 128), (128, 128), (64, 128), (128, 64), (64, 64)]


def read_format(d: bytes) -> Optional[str]:
    """
    Pixel format string from the header.

    The descriptor tail also carries a 'format' name, so an empty or short
    result there must not shadow the real one; we take the first plausible
    all-identifier value found.
    """
    i = d.find(b"format\x00")
    if i < 0:
        return None
    p = i + 7
    ident = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    for k in range(0, 12):
        j = d.find(b"\x00", p + k)
        if j < 0 or j == p + k:
            continue
        s = d[p + k:j]
        if s and all(c in ident for c in s) and len(s) >= 3:
            return s.decode()
    return None


def base_mip_bytes(d: bytes, bpb: int) -> Optional[int]:
    """
    Byte size of the base mip level.

    The value is a u32 in the 0x20..0x60 header window.  Several offsets can
    hold a value that happens to be a valid block count, so pick the one whose
    implied mip chain leaves the *smallest* descriptor tail -- the real size
    field is the one that explains the payload.
    """
    n = len(d)
    valid = set()
    for w, h in SHAPES:
        valid.add((w // 4) * (h // 4) * bpb)
    best = None
    for o in range(0x20, min(0x60, n - 4)):
        v = int.from_bytes(d[o:o + 4], "little")
        if v not in valid:
            continue
        if PIXEL_OFFSET + v > n:
            continue
        # prefer the value whose mip chain ends closest to the payload end
        chain, cw, ch = 0, None, None
        # chain length depends on the shape; recompute for the tightest shape
        for w, h in SHAPES:
            if (w // 4) * (h // 4) * bpb != v:
                continue
            c, a, b = 0, w, h
            while a >= 4 and b >= 4:
                c += (a // 4) * (b // 4) * bpb
                a //= 2
                b //= 2
            slack = n - PIXEL_OFFSET - c
            if slack < 0:
                continue
            if chain == 0 or slack < chain:
                chain = slack
        if chain == 0 and cw is None and not any(
                (w // 4) * (h // 4) * bpb == v for w, h in SHAPES):
            continue
        key = (chain, -v)
        if best is None or key < best[0]:
            best = (key, v, o)
    return best[1] if best else None


def _smooth(a: np.ndarray) -> float:
    g = a[:, :, :3].astype(np.float32).mean(2)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


def _decode(blob: bytes, w: int, h: int, kind: str, bpb: int, colmajor: bool):
    bw, bh = w // 4, h // 4
    nblk = bw * bh
    blocks = np.frombuffer(blob[:nblk * bpb], dtype=np.uint8).reshape(nblk, bpb)
    if colmajor:
        by, bx = np.divmod(np.arange(nblk), bw)
        blocks = blocks[bx * bh + by]
    if kind == "bc2":
        # DXT3: each 16-byte block is 8 bytes of explicit alpha followed by the
        # 8-byte BC1 colour block.  texture2ddecoder has no bc2 entry point, so
        # gather the colour halves and run them through BC1.
        col = blocks[:, 8:].copy()
        px = t2d.decode_bc1(col.tobytes(), w, h)
        return _rgba(np.frombuffer(px, dtype=np.uint8).reshape(h, w, 4))
    px = getattr(t2d, f"decode_{kind}")(blocks.tobytes(), w, h)
    return _rgba(np.frombuffer(px, dtype=np.uint8).reshape(h, w, 4))


def decode(d: bytes, prefer: Optional[Tuple[int, int]] = None,
           verbose: bool = False):
    """Decode a .txb payload -> (w, h, format, (h,w,4) uint8)."""
    fmt = read_format(d)
    if fmt is None:
        raise ValueError("txb: no format field")
    if fmt in KIND:
        kind = KIND[fmt]
        bpb = BPB[kind]
    elif fmt in RAW:
        kind = None
        bpb = RAW[fmt]
    else:
        raise ValueError(f"txb: unsupported format {fmt}")

    n = len(d)
    nb = base_mip_bytes(d, bpb)

    if kind is None:
        w, h = prefer or (1024, 1024)
        start = n - w * h * bpb
        a = np.frombuffer(d[start:start + w * h * bpb],
                          dtype=np.uint8).reshape(h, w, bpb)
        if bpb == 4:
            if fmt.startswith("B8G8R8A8"):
                a = a[:, :, [2, 1, 0, 3]]
            return w, h, fmt, np.ascontiguousarray(a)
        if bpb == 1:
            return w, h, fmt, np.dstack([a[:, :, 0]] * 3 +
                                        [np.full((h, w), 255, np.uint8)])
        return w, h, fmt, np.dstack([a, np.full((h, w, 1), 255, np.uint8)])

    # which shapes fit the base mip?
    shapes: List[Tuple[int, int]] = [s for s in SHAPES
                                     if (s[0] // 4) * (s[1] // 4) * bpb == nb]
    if not shapes:
        # fall back: any shape whose base mip fits the payload
        shapes = [s for s in SHAPES
                  if PIXEL_OFFSET + (s[0] // 4) * (s[1] // 4) * bpb <= n]
    if not shapes:
        raise ValueError(f"txb: no shape for {nb} bytes / {fmt}")
    # A hint from the caller (the mesh UV aspect ratio) is only a tie-break:
    # the byte count cannot distinguish 2048x2048 from 4096x1024, and the UV
    # ratio is noisy, so smoothness still decides.
    if prefer and tuple(prefer) in shapes:
        shapes = [tuple(prefer)] + [s for s in shapes if tuple(s) != tuple(prefer)]

    blob = d[PIXEL_OFFSET:]
    best = None
    for w, h in shapes[:6]:
        need = (w // 4) * (h // 4) * bpb
        if need > len(blob):
            continue
        for cm in (False, True):
            try:
                a = _decode(blob, w, h, kind, bpb, cm)
            except Exception:
                continue
            sc = _smooth(a)
            if best is None or sc < best[0]:
                best = (sc, w, h, a, cm)
    if best is None:
        raise ValueError("txb: nothing decodable")
    _, w, h, a, cm = best
    if verbose:
        print(f"   {w}x{h} {'colmajor' if cm else 'raster'} smooth={best[0]:.4f}")
    if kind == "bc4":
        r = a[:, :, 2]                       # decode_bc4 stores in BLUE
        a = np.dstack([r, r, r, np.full((h, w), 255, np.uint8)])
    return w, h, fmt, a


if __name__ == "__main__":
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mikhail_assets import FpdArchive
    try:
        from . import config as _cfg
    except ImportError:
        import config as _cfg
    pack = FpdArchive(_cfg.PACK)
    base = "root/assets/data/chr/tsf/f22a_00/model/"
    for nm in sys.argv[1:] or ["chest_bc", "mountb_bc", "leg_ao", "head_bc"]:
        d = pack.payload(base + nm + ".png.txb.desktop.zstd")
        t0 = time.time()
        w, h, fmt, a = decode(d, verbose=True)
        print(f"{nm:12s} {w}x{h} {fmt:12s} {time.time()-t0:.2f}s")
