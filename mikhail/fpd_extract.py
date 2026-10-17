"""
Project MIKHAIL / AGES Mk2 "FPD" archive extractor.

Format (reverse-engineered by DaZombieKiller & kurikomoe for Fate/stay night
REMASTERED; the identical keystream was verified to work on Project MIKHAIL):

  offset 0x00  magic  "FPD\\0"
  offset 0x04  u32 BE  version        (1 = Android, 2 = PC/Switch)
  offset 0x08  u64 BE  file count
  offset 0x10  u64 BE  data start offset
  offset 0x18  32 bytes reserved (zeros)
  offset 0x38  entry table, file_count * 0x20 bytes, XOR-scrambled from key byte 0
                 u64 BE nameOffset (into inflated name buffer)
                 u64 BE dataOffset (relative to data start)
                 u64 BE dataLength
                 u64 BE uncompressedLength (0 = stored, else zlib)
  then         zlib-deflated NUL-separated UTF-8 name buffer

  Each file payload is XOR-scrambled independently (key byte 0 at file start).

The 64 KiB keystream in decryptKey.bin is engine-wide (shared by Fate/stay night
REMASTERED, Fate/hollow ataraxia REMASTERED and Project MIKHAIL).
"""
import argparse
import os
import struct
import sys
import zlib

import numpy as np
try:
    from . import config as _cfg
except ImportError:
    import config as _cfg

HDR_SIZE = 0x38
KEY_SIZE = 65536


def load_keystream(path):
    with open(path, "rb") as f:
        k = f.read()
    if len(k) != KEY_SIZE:
        raise SystemExit(f"key file must be {KEY_SIZE} bytes, got {len(k)}")
    return np.frombuffer(k, dtype=np.uint8)


def xor_stream(buf, keys, key_offset=0):
    """XOR `buf` (bytes) with the periodic keystream, starting at key_offset."""
    n = len(buf)
    if n == 0:
        return buf
    idx = (np.arange(n, dtype=np.int64) + key_offset) % KEY_SIZE
    a = np.frombuffer(buf, dtype=np.uint8).copy()
    a ^= keys[idx]
    return a.tobytes()


class FpdArchive:
    def __init__(self, path, keys):
        self.path = path
        self.keys = keys
        self.f = open(path, "rb")
        head = self.f.read(HDR_SIZE)
        if head[:4] != b"FPD\x00":
            raise SystemExit(f"{path}: not an FPD archive (magic {head[:4]!r})")
        self.version = struct.unpack(">I", head[4:8])[0]
        self.count = struct.unpack(">Q", head[8:16])[0]
        self.data_start = struct.unpack(">Q", head[16:24])[0]
        self.reserved = head[24:56]
        self._read_table()

    def _read_table(self):
        tbl_len = self.data_start - HDR_SIZE
        self.f.seek(HDR_SIZE)
        dec = xor_stream(self.f.read(tbl_len), self.keys, 0)
        self.raw_table = dec
        u = np.frombuffer(dec[: self.count * 32], dtype=">u8").reshape(-1, 4)
        names = zlib.decompress(dec[self.count * 32:])
        self.names_blob = names
        self.entries = []
        for i in range(self.count):
            no, do, dl, ul = (int(x) for x in u[i])
            end = names.find(b"\x00", no)
            raw = names[no:end]
            try:
                nm = raw.decode("utf-8")
            except UnicodeDecodeError:
                nm = raw.decode("cp932", "replace")
            self.entries.append({"name": nm, "offset": do, "size": dl,
                                 "usize": ul, "idx": i})
        self.by_name = {e["name"]: e for e in self.entries}

    def raw(self, name):
        e = self.by_name[name]
        if e["size"] == 0:
            return b""
        self.f.seek(self.data_start + e["offset"])
        data = xor_stream(self.f.read(e["size"]), self.keys, 0)
        if e["usize"]:
            data = zlib.decompress(data)
        return data

    def payload(self, name, zstd_layer=True):
        """Return the real payload: FPD entry -> optional zstd frame."""
        d = self.raw(name)
        if zstd_layer and name.endswith(".zstd"):
            try:
                import zstandard
            except ImportError:
                raise SystemExit("pip install zstandard")
            d = zstandard.ZstdDecompressor().decompress(d, max_output_size=1 << 30)
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--key", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "decryptKey.bin"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--out", default=None, help="extract matching entries to this dir")
    args = ap.parse_args()

    keys = load_keystream(args.key)
    a = FpdArchive(args.archive, keys)
    print(f"{args.archive}: version={a.version} files={a.count} data_start={a.data_start}",
          file=sys.stderr)

    items = a.entries
    if args.filter:
        items = [e for e in items if args.filter in e["name"]]

    if args.list or not args.out:
        for e in items:
            print(f"{e['size']:>12}  {e['usize']:>12}  {e['name']}")
        return

    for e in items:
        d = a.payload(e["name"])
        dst = os.path.join(args.out, e["name"].replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(d)
        print(f"{len(d):>12}  {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
