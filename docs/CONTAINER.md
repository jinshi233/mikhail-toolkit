# Container formats

Everything here was measured from the shipping game, not taken from documentation.

```
obb/pack.bin                     FPD\0 archive, XOR-scrambled, zlib chunks
  └─ <path>.zstd                 Zstandard frame          (or raw, if usize == 0)
       └─ EPK\0                  fuzz-eg "eg runtime core" tag stream
            ├─ <name>.ma.scn     static mesh scene: rig + vertex/index buffers
            ├─ <name>.txb        texture (BC7 / BC5 / BC2 / BC4 / raw RGBA8)
            └─ <name>.editMA     animation take
```

## 1. `obb/pack.bin` — FPD archive

```
0x00  'FPD\0'
0x04  u32BE version          (2 for this game)
0x08  u64BE fileCount        (38,986)
0x10  u64BE dataStart        (1,397,868)
0x18  32 bytes zero
0x38  entry table, fileCount * 32 bytes   XOR-scrambled from keystream byte 0:
        u64BE nameOffset     offset into the (decompressed) name blob
        u64BE dataOffset     offset into the payload region
        u64BE dataLength
        u64BE usize          uncompressed size, 0 when the payload is stored raw
      then: zlib-deflated, NUL-separated UTF-8 name blob
      then: the payload region
```

Each payload is XOR-scrambled independently **from keystream byte 0**, and when
`usize != 0` it is additionally zlib-deflated.  A 64 KiB keystream, applied mod
0x10000, is enough for the whole archive.

Verified: `38,986` entries, every `dataOffset + dataLength` inside the payload
region, every name decompressing to a plausible path.

### The keystream is not in this repository

It is engine data.  Two ways to supply it:

* `python tools/make_key.py` — searches the engine binaries for the window whose
  XOR with the archive's index produces a valid, monotonic entry table.  Works if
  your build keeps it as static data.
* otherwise recover it at runtime: the keystream is resident in the process while
  the archive is open, so a memory scan for a 64 KiB block that decrypts entry 0
  to `nameOffset = 0, dataOffset = 0` identifies it.

Put the result at `decryptKey.bin` (or point `MIKHAIL_KEY` at it).

## 2. `.txb` — textures

```
0x00  header, contains the pixel-format string (e.g. `BC7_SRGB`, `BC5_UNORM`)
0x4C  u32 base-mip byte size  (blocks * blockBytes)
0x50  pixel data, base mip first, mip chain contiguous
```

Formats seen: `BC7_SRGB` (base colour), `BC5_UNORM` (normal), `BC2_UNORM`
(the `_cm` glow/emissive masks), `BC4_UNORM` (AO / metalness / roughness, one
channel), `R8G8B8A8*` / `B8G8R8A8*` (UI).

Two traps that cost a lot of time:

* **The base mip byte count cannot distinguish 2048×2048 from 4096×1024**, and
  both decode "successfully" (they are transposes).  The extent is a 2-way
  ambiguity that has to be broken by something else — the mesh UV aspect ratio,
  or a smoothness score over the block layout.
* **`texture2ddecoder` returns BLUE-first pixels.** Consuming them as RGBA swaps
  red and blue on every texture, which is not a cosmetic detail: grey metal turns
  cold blue, red faction markings turn blue, and the glow masks come out of the
  wrong channels entirely.  `tools/bc1_probe.py` proves the ordering with a
  hand-decoded block whose endpoint 0 is pure red (`0xF800`): the format says
  `[255,0,0]`, the library reports `[0,0,255]`.

`BC4_UNORM` also lands in the **blue** channel of the decoded buffer, not red.

## 3. EPK tag stream

See `SCENE_FORMAT_FINDINGS.md` — the full opcode table, and the key result that
field and class names are stored as the **little-endian CRC32 of the name**, so
`crc32("localTranslation") = 0x339d6090` appears in the stream as `90 60 9d 33`.
That is all you need to re-implement the reader.
