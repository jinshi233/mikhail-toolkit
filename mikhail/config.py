"""Shared configuration.

Nothing in this repository ships game data.  Point it at your own installation:

    MIKHAIL_GAME   game root (default: typical Steam path)
    MIKHAIL_PACK   pack.bin (default: <game>/obb/pack.bin)
    MIKHAIL_KEY    the archive's XOR keystream, 65536 bytes (default: <repo>/decryptKey.bin)
    MIKHAIL_OUT    where exports go (default: ./export)

The keystream is derived from the game and is deliberately NOT distributed, see
docs/CONTAINER.md for how it is obtained.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_GAME = r"C:\Program Files (x86)\Steam\steamapps\common\Project MIKHAIL"

GAME = os.environ.get("MIKHAIL_GAME", DEFAULT_GAME)
PACK = os.environ.get("MIKHAIL_PACK", os.path.join(GAME, "obb", "pack.bin"))
KEY = os.environ.get("MIKHAIL_KEY", os.path.join(REPO, "decryptKey.bin"))
OUT = os.environ.get("MIKHAIL_OUT", os.path.join(REPO, "export"))


def load_key(path=None):
    """Return the keystream bytes, or None when it is absent."""
    p = path or KEY
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return f.read()


def require_key(path=None):
    k = load_key(path)
    if k is None:
        raise SystemExit(
            f"keystream not found at {path or KEY}\n"
            "Set MIKHAIL_KEY to a 65536-byte keystream file, or generate it with\n"
            "  python tools/make_key.py\n"
            "which recovers it from the game's own files.")
    return k
