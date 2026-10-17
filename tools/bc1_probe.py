#!/usr/bin/env python3
"""
Is texture2ddecoder returning RGBA or BGRA?

BC7 is complex, but BC1/BC2 share the same 8-byte colour block and are trivial
to decode by hand.  Build a block whose first endpoint is pure red and the second
pure blue, decode it with texture2ddecoder, and see which byte lands where.
"""
import numpy as np
import texture2ddecoder as t2d


def bc1_colour_to_rgb(c):
    return np.array([(c >> 11) & 0x1F, (c >> 5) & 0x3F, c & 0x1F], dtype=np.float64)


def expand(rgb):
    r, g, b = rgb
    return np.array([r * 255 / 31, g * 255 / 63, b * 255 / 31])


def hand_bc1(block):
    """Return the 4x4 RGBA block decoded by hand, as true R,G,B."""
    c0 = int.from_bytes(block[0:2], "little")
    c1 = int.from_bytes(block[2:4], "little")
    idx = int.from_bytes(block[4:8], "little")
    p0, p1 = expand(bc1_colour_to_rgb(c0)), expand(bc1_colour_to_rgb(c1))
    pal = [p0, p1, (p0 + p1) / 2, (2 * p0 + p1) / 3]
    out = np.zeros((4, 4, 4))
    for y in range(4):
        for x in range(4):
            k = (idx >> (2 * (4 * y + x))) & 3
            out[y, x, :3] = pal[k]
            out[y, x, 3] = 255
    return out


# endpoint 0 = pure red (R=31,G=0,B=0), endpoint 1 = pure blue (R=0,G=0,B=31)
RED = (31 << 11)
BLUE = 31
block = bytearray()
block += RED.to_bytes(2, "little")
block += BLUE.to_bytes(2, "little")
block += (0).to_bytes(4, "little")          # every texel uses endpoint 0
raw = bytes(block)
print("BC1 block endpoints: c0=0x%04x (pure RED)  c1=0x%04x (pure BLUE)" % (RED, BLUE))
print("hand decode, texel (0,0) =", hand_bc1(raw)[0, 0].tolist())

px = np.frombuffer(t2d.decode_bc1(raw, 4, 4), dtype=np.uint8).reshape(4, 4, 4)
print("texture2ddecoder decode_bc1, texel (0,0) =", px[0, 0].tolist())
print("=> library returns", "RGBA" if px[0, 0, 0] > 200 and px[0, 0, 2] < 60
      else ("BGRA" if px[0, 0, 2] > 200 and px[0, 0, 0] < 60 else "?"))
