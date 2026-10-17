# mikhail-toolkit

Reverse-engineering toolkit for **Project MIKHAIL** (fuzz Inc. / aNCHOR, AGES Mk2
/ "eg runtime core").  Reads the game's archives and exports the tactical
fighters (TSF) and BETA enemies as rigged, textured FBX / glTF.

No game data is included: the tooling recovers everything from your own
installation, and the archive keystream has to be supplied by you (see below).

```
archive  →  scene graph + skeleton + bind pose  →  meshes + skinning
         →  textures                           →  Blender  →  FBX / glTF
```

## Why this exists

The format is self-describing but has several traps that each look like "the
data isn't there".  Everything in `docs/` is measured, and the repository keeps
the probes that produced the measurements so the claims can be re-run:

| trap | symptom | reality |
|---|---|---|
| field names are CRC32 hashes | nothing is greppable | `crc32(name)` LE, documented below |
| index element width varies per unit | "this model has no faces" | u32 (`…0x8d`) on most, **u16** (`…0x8b`) on F-14 / MiG-21 / YF-23 |
| vertex record stride varies | geometry silently truncated | 60 B, **64 B**, and 48 B (`kakasi`) |
| skin-block offset moves with stride | "the legs have no binding" | weights at `0x30`/`0x38` for 60 B records, `0x34`/`0x3C` for 64 B |
| indices are global *or* mesh-local | 85 k loose vertices, cone-shaped limbs | two topologies, told apart arithmetically |
| `texture2ddecoder` is BLUE-first | every red marking renders blue | swap to RGBA |

## Requirements

```
Python 3.11+   numpy  Pillow  zstandard  texture2ddecoder
Blender 4.5+   (only for the FBX / glTF step)
```

## Setup

```bash
pip install numpy pillow zstandard texture2ddecoder
export MIKHAIL_GAME="/path/to/Project MIKHAIL"     # holds obb/pack.bin
python tools/make_key.py                            # → decryptKey.bin
```

If `make_key.py` cannot find the keystream as static data in your build, recover
it at runtime (it is resident in the process while the archive is open — scan for
the 64 KiB block that decrypts entry 0 to `nameOffset=0, dataOffset=0`) and drop
it at `decryptKey.bin`.  See `docs/CONTAINER.md` §1.

## Usage

```bash
# list everything in the archive
python -m mikhail.fpd_list --filter /chr/tsf/

# one unit: parse + verify, no export
python -m mikhail.parse_scene root/assets/data/chr/tsf/f22a_00/model/f22a_00.ma.scn.zstd

# export rigged + textured FBX/glTF for a few units
python -m mikhail.export_rigged --kind tsf --name f22a_00 f14_00 type00a_00

# everything (43 TSFs, 13 BETAs — the BETA tree goes to ./export_beta)
python -m mikhail.export_rigged --kind tsf
python -m mikhail.export_rigged --kind beta
```

Output, one folder per unit:

```
export/<unit>/<unit>.fbx          FBX 7.4, textures embedded, LOD0
export/<unit>/<unit>.glb          same as glTF
export/<unit>/textures/*.png      decoded maps
export/<unit>/<unit>.json         rig + meshes + skin weights (reproducible)
export/<unit>/<unit>.indices.npy  triangle list
```

Each unit becomes **one object per body part** (10 for most TSFs, 14 for YF-23
Pav-1/2), and each object carries its own bone set — that is what makes limbs,
hands and mounts independently animatable.

## Verification

These are the checks that caught every bug above, and they run against the
exported FBX rather than the in-memory data:

```bash
blender -b --factory-startup --python verify/acceptance2.py -- export
```

It reports, per unit: object count, vertex/triangle totals, **loose vertices**
(vertices no face uses — the signature of wrong index rebasing), **unbound
vertices**, orphan vertex groups, and a real pose test (rotate the bone with the
most bound vertices, count how many actually move).

Two more oracles used during the investigation:

* **winding agreement** — a triangle's geometric normal must agree with the
  stored vertex normals.  Correct data scores 99.4–100 %; a mis-assigned vertex
  blob scores ~65 %.  This is what distinguishes the right per-mesh blob from the
  wrong one, and it needs no ground truth.
* **bind-pose cross-check** — the file stores each joint's bind matrix.  Rebuilding
  the rig from the tag stream reproduces 906/906 stored matrices to within
  1.0e-06 (f32 precision).  See `docs/SCENE_FORMAT_FINDINGS.md` §6.

## Layout

```
mikhail/
  mikhail_assets.py   FPD archive reader, EPK helpers, txb/image helpers
  parse_scene.py      scene graph, skeleton, bind pose, meshes, vertex decode
  txb_fast.py         texture decoder (BC7/BC5/BC2/BC4/raw) + offset probing
  mesh_blobs.py       vertex-blob location, skin-offset detection, winding oracle
  tri_units.py        per-mesh submesh extraction for mesh-local-index units
  export_rigged.py    unit discovery, assembly, JSON + texture writing, CLI
  blender_rig.py      Blender-side armature + mesh + PBR material builder
verify/               acceptance / deformation / render checks
tools/                standalone probes (see table below)
docs/                 format documentation
```

Probes worth knowing:

| tool | what it establishes |
|---|---|
| `bc1_probe.py` | decoder channel order, with a hand-decoded reference block |
| `skin_offsets.py` | skin-block offset per record stride, from the weights themselves |
| `winding_stats.py` | geometry oracle; separates a correct blob from a wrong one |
| `runtime_bind.py` | per-mesh weighted-bone coverage from the exported JSON |
| `export_audit.py` | sweep over a whole export tree for weight/orphan problems |
| `regress.py` | every TSF against the `vcount == max(index)+1` invariant |

## Known gaps

* Only LOD0 is exported; LOD1–4 are present in the `.json` / `.npy`.
* Animations (`.editMA.take`) are located but not exported.
* Non-square textures (1024×2048, 512×1024) still show banding — the block-layout
  (raster vs column-major) disambiguation is unfinished; square atlases are fine.
* `Material` / `Shader` / `Collision` sections are located but not decoded.
* A handful of unresolved odds and ends are listed at the end of
  `docs/SCENE_FORMAT_FINDINGS.md`.

## Legal

For interoperability, preservation and modding research on software you own.
It ships no game assets and no keys; do not redistribute extracted content.
Project MIKHAIL and its assets belong to their respective rights holders.
