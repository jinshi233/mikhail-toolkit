import struct, zlib, os, sys, json
try:
    from . import config as _cfg
except ImportError:
    import config as _cfg
KEY = _cfg.KEY
base = _cfg.GAME

keys = open(KEY, "rb").read()
HDR = 0x38

def xor(data, keyoffset=0):
    out = bytearray(len(data))
    for i in range(len(data)):
        out[i] = data[i] ^ keys[(keyoffset + i) % len(keys)]
    return bytes(out)

class Fpd:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        head = self.f.read(HDR)
        assert head[:4] == b"FPD\x00", "bad magic " + repr(head[:4])
        self.version = struct.unpack(">I", head[4:8])[0]
        self.count = struct.unpack(">Q", head[8:16])[0]
        self.dstart = struct.unpack(">Q", head[16:24])[0]
        self.reserved = head[24:56]
        self.table_len = self.dstart - HDR
        self.f.seek(HDR)
        raw = self.f.read(self.table_len)
        dec = xor(raw, 0)
        self.entries = []
        for i in range(self.count):
            o = i * 32
            no, do, dl, ul = struct.unpack(">QQQQ", dec[o:o+32])
            self.entries.append((no, do, dl, ul))
        names_blob = dec[self.count * 32:]
        names = zlib.decompress(names_blob)
        for i, (no, do, dl, ul) in enumerate(self.entries):
            s = names[no:]
            e = s.find(b"\x00")
            self.entries[i] = (s[:e].decode("utf-8", "replace"), do, dl, ul)

    def read(self, idx):
        name, off, dl, ul = self.entries[idx]
        self.f.seek(self.dstart + off)
        data = xor(self.f.read(dl))
        if ul != 0:
            data = zlib.decompress(data)
        return name, data

base = r"D:\SteamLibrary\steamapps\common\Project MIKHAIL"
for rel in [r"boot.dat", r"smashboot.dat", r"data\root\assets\adv.bin",
            r"data\root\assets\gui01.bin", r"data\root\assets\gui02.bin",
            r"data\root\assets\spine.bin"]:
    p = os.path.join(base, rel)
    a = Fpd(p)
    exts = {}
    for n, o, l, u in a.entries:
        e = os.path.splitext(n)[1].lower() or "(none)"
        exts[e] = exts.get(e, 0) + 1
    print(f"\n### {rel}  ver={a.version} files={a.count} dstart={a.dstart} reserved={a.reserved.hex()}")
    print("   extensions:", dict(sorted(exts.items(), key=lambda kv: -kv[1])))
    for n, o, l, u in a.entries[:6]:
        print(f"     {n}  off={o} len={l} unc={u}")
