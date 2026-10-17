#!/usr/bin/env python3
"""
Recover the archive's 64 KiB XOR keystream from the game itself.

The keystream is not shipped here: it lives as static data inside the engine
binary (`erc.dll`), and every archive payload is XOR-scrambled with it starting
at offset 0.  That gives an enormous known-plaintext: the first payload in
`obb/pack.bin` is a Zstandard frame (`28 B5 2F FD`), and a zlib-deflated one is
not far behind, so candidate windows can be verified instead of guessed.

Usage:
    python tools/make_key.py [--game DIR] [--out decryptKey.bin] [--binary erc.dll]

This is a clean-room derivation: it searches for the window whose XOR with the
ciphertext yields a valid compressed frame.
"""
import argparse
import os
import struct
import sys
import zlib

KEY_SIZE = 0x10000
HDR_SIZE = 0x38
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def entry_table(pack_path):
    """(count, dataStart, first payload offset/length) from the FPD header."""
    with open(pack_path, "rb") as f:
        head = f.read(HDR_SIZE)
        if head[:4] != b"FPD\x00":
            raise SystemExit(f"{pack_path}: not an FPD archive")
        count = struct.unpack_from(">Q", head, 8)[0]
        data_start = struct.unpack_from(">Q", head, 16)[0]
        # entry table is scrambled, so its field values need the keystream; the
        # header itself is plaintext
        return count, data_start


def find_key(binary_path, pack_path, verbose=False):
    """Search `binary_path` for the window that decrypts the archive's index.

    Known plaintext is much stronger than "looks compressed": after descrambling,
    entry 0 of the offset table is `nameOffset=0, dataOffset=0, dataLength=<size>`
    and the table as a whole is monotonic, so a wrong window is rejected almost
    immediately.  Scanning is vectorised (numpy XOR) because the binaries are tens
    of megabytes.
    """
    import numpy as np

    with open(pack_path, "rb") as f:
        head = f.read(HDR_SIZE)
        count = struct.unpack_from(">Q", head, 8)[0]
        data_start = struct.unpack_from(">Q", head, 16)[0]
        f.seek(HDR_SIZE)
        cipher = f.read(min(KEY_SIZE, data_start - HDR_SIZE))

    with open(binary_path, "rb") as f:
        blob = f.read()
    ks = np.frombuffer(blob, dtype=np.uint8)
    ct = np.frombuffer(cipher, dtype=np.uint8)
    n = len(ct)
    if len(ks) < n:
        return None, None

    # window that must decrypt to 16 zero bytes (entry 0: nameOffset/dataOffset)
    zeros = np.zeros(16, dtype=np.uint8)
    hits = []
    as_strided = np.lib.stride_tricks.sliding_window_view
    for probe_len, need in ((16, zeros),):
        win = as_strided(ks, probe_len)[: len(ks) - probe_len + 1]
        eq = np.all(win ^ ct[:probe_len] == need, axis=1) if False else None
        # XOR needs broadcasting: do it in chunks to stay inside cache
        chunk = 1 << 22
        for base in range(0, win.shape[0], chunk):
            seg = win[base:base + chunk] ^ ct[:probe_len]
            good = np.flatnonzero(np.all(seg == need, axis=1))
            for g in good[:200]:
                hits.append(base + int(g))
    if verbose:
        print(f"  {len(hits)} prefilter hit(s)")

    for off in hits:
        cand = bytes(blob[off:off + n])
        plain = bytes(a ^ b for a, b in zip(cipher, cand))
        if validate_table(plain, count):
            return blob[off:off + KEY_SIZE if off + KEY_SIZE <= len(blob) else n], off
    return None, None


def validate_table(plain, count):
    """Entry 0 must be (0, 0, size, ...) and the offsets must be monotonic."""
    if len(plain) < 32:
        return False
    name_off, data_off, data_len = struct.unpack_from(">3Q", plain, 0)
    if name_off != 0 or data_off != 0 or not (1000 < data_len < 1 << 40):
        return False
    prev = 0
    n = min(count, len(plain) // 32, 512)
    for i in range(n):
        o = i * 32
        no = struct.unpack_from(">Q", plain, o)[0]
        if no < prev or no > 1 << 32:
            return False
        prev = no
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=os.environ.get(
        "MIKHAIL_GAME", r"D:\SteamLibrary\steamapps\common\Project MIKHAIL"))
    ap.add_argument("--pack", default=None)
    ap.add_argument("--binary", default=None, help="engine binary holding the key")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "decryptKey.bin"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    pack = a.pack or os.path.join(a.game, "obb", "pack.bin")
    if not os.path.exists(pack):
        raise SystemExit(f"pack not found: {pack}")

    cands = []
    if a.binary:
        cands.append(a.binary)
    else:
        for name in ("erc.dll", "erc.exe", "ProjectMIKHAIL.exe", "smashboot.dll"):
            p = os.path.join(a.game, name)
            if os.path.exists(p):
                cands.append(p)
        # also look one level down (binaries/, engine/)
        for sub in ("binaries", "engine", "Win64", "x64"):
            d = os.path.join(a.game, sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".dll", ".exe")):
                        cands.append(os.path.join(d, f))
    if not cands:
        raise SystemExit("no engine binary found; pass --binary explicitly")

    for c in cands:
        print(f"searching {os.path.basename(c)} ({os.path.getsize(c):,} B) ...")
        key, off = find_key(c, pack, a.verbose)
        if key:
            with open(a.out, "wb") as f:
                f.write(key)
            print(f"keystream found at 0x{off:x} in {os.path.basename(c)}")
            print(f"wrote {a.out} ({len(key)} bytes)")
            return 0
    raise SystemExit("keystream not located; the engine may compute it at runtime")


if __name__ == "__main__":
    sys.exit(main())
