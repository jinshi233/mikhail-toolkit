# Project MIKHAIL `.ma.scn` (EPK) — established byte layouts

All numbers below were measured, not guessed.  Scripts:

| file | purpose |
|---|---|
| `parse_scene.py` | **Deliverable A+B.** Node tree, skeleton, bind pose, mesh/skeleton binding, vertex decode, verification, JSON/OBJ export |
| `analyze_vertex_attrs.py` | raw per-offset statistics for the 60-byte vertex record (the evidence for the vertex layout) |
| `epk_stream.py` | reusable tag-stream value decoder |
| `extract_samples.py` | pulls validation samples out of `obb/pack.bin` |

Validation set: `f22a_00` (F-22A TSF), `type94_jp_00` (Type-94 TSF), `bl74_00` (weapon).

---

## 1. Container

```
0x00  'EPK\0' | u16 ver(0x0203) | u16 flags | u32 totalSize | u32 0
0x10  sorted NUL-terminated name pool (f22a: 237 entries, ends 0xbf5)
...   tag stream (the whole rest of the file)
```

## 2. Tag stream

Every item is `<u8 tag> <payload>`.  `tag >= 0x80` -> `tag & 0x7f` is an opcode;
`tag < 0x80` -> the byte itself is a small unsigned integer.

| opcode | meaning |
|---|---|
| `0x03` | class instance: `03 03 <u32 crc32(className)>`, or `03 02 <NUL-string>` for a new named type |
| `0x06`, `0x26` | NUL-terminated ASCII string |
| `0x07` | **field key: `87 <u32 crc32(fieldName) LE>`** |
| `0x09` | u8        (verified: `nodeIndex` 128..255 == `89 <byte>`) |
| `0x0a`, `0x0b` | u16 / u16 array element |
| `0x0c` | u32 |
| `0x0f` | 1-byte float (`0x00`->0.0, `0x38`->1.0) |
| `0x10` | **IEEE half (f16)** — verified: `3C 00` == 1.0 |
| `0x11` | IEEE float (f32) |
| `0x13` | array `93 <u8 count> <elemTag> <data>` |
| `0x14` | array `94 <u16 count> <elemTag> <data>` |
| `0x96+n` | array of n fixed-size elements, `<elemTag>` follows (n up to ~32) |
| `0xa7` | array `a7 <u8 count>` then `count` self-describing elements |
| `0xc0+n` | array of n self-describing elements (`c0` = empty, `c1` = 1 object, `ca` = 10 strings) |
| `0x84` | no-payload marker, may repeat, ignore |
| `0x85` | end-of-object marker |
| `0x8d` | one-byte marker (the `IndexBuffer` index data starts immediately after it) |
| `0xff` | null (e.g. `parentUid` of the root) |

Field and class names are **never stored at the use site** — they are the
little-endian CRC32 of the name.  That was proven blind: `crc32("localTranslation")=0x339d6090`
appears in the stream as `90 60 9d 33`, and 24 / 500 blind probe combinations matched
exactly, e.g.

```
scalePivot 0x0838523c   jointOrient 0x0dae18bb   rotationOrder 0x20e502e5
localTranslation 0x339d6090   localRotation 0x39971e57   localScaling 0x457cc09d
jointGroup 0x4828ed99   visibility 0x518e4300   uid 0x539b0606   shadowCast 0x5b69f506
name 0x5e237e06   idColor 0x71920bf4   shadowReceive 0x77afa0e3   parentUid 0x7fed4fa6
nodeAlpha 0x8ba0ebe3   staticObject 0x9ea9ab1c   children 0xa197b1ba   nodeKind 0xa5b82866
nodeIndex 0xdb33b588   segmentScaleCompensate 0xe525d656   Node 0x254d477b
VertexBuffer 0x165ba071   IndexBuffer 0xc89b1619   Mesh 0x415d9568   Skeleton 0x24aecb7c
```

Grid/FBX programmers: `crc32(name) & 0xffffffff`, little-endian, is all you need to
re-implement the reader.

## 3. Node records

A node record starts with `83 03 <crc32("Node")> <2 bytes>` and contains, in order

```
scalePivot jointOrient rotationOrder localTranslation localRotation localScaling
[jointGroup] visibility uid shadowCast name
idColor shadowReceive parentUid rotatePivot nodeAlpha staticObject children
nodeKind scaleFactorCompensate scaleOffset rotateOffset nodeIndex segmentScaleCompensate
```

* `uid` = 386 + nodeIndex for every node except the root (`uid` 9); `parentUid` is a
  `uid`, `0xff` for the root.
* `children` is a u16 array of uids.
* `localTranslation` / `localRotation` / `jointOrient` / `localScaling` are
  self-describing vectors (f8 / f16 / f32 element tag chosen per value, count in the tag).
* World transform, verified: **`W = parent(W) @ T(localTranslation) @ R(jointOrient) @ R(localRotation)`**.
  `localScaling` is **not** applied — `Group_JT` carries 0.01 (a DCC cm->m conversion) and
  every other node 1.0; including it shrinks the rig 100x away from the mesh.

## 4. Vertex record — 60 bytes

```
0x00  f32 x3   position             (mesh bbox 1277 x 1900 x 1071, Y-up)
0x0c  f32      == 1.0
0x10  f16 x3   normal               |n| in [0.99,1.01] for 99.897%
0x16  u16      0 (padding)
0x18  f16 x2   UV0                  both components in [0,1] for 100.00%
0x1c  f16 x2   UV1                  (lightmap/detail set; 77.2% inside [0,1], u up to 1.197)
0x20  f16 x2   UV2                  byte-identical to UV0 for whole meshes / absent otherwise
0x24  f16 x2   UV3
0x28  i8  x3   tangent   / 127      |t| in [0.99,1.01] for 99.959%
0x2b  u8       0
0x2c  i8  x3   bitangent / 127      |b| in [0.99,1.01] for 99.879%
0x2f  u8       0
0x30  f16 x4   skin weights         sum == 1.0 for 100.0000% of 230,981 vertices
0x38  u8  x4   skin joint indices   index into THIS mesh's skeleton.nodes
```

Tangent-frame evidence: `t.b == 0` 100.00%, `t.n == 0` 100.00%, `b.n == 0` 100.00%
(mean |dot| 0.0015 / 0.0016).  `/127` beats `/128` (99.96% vs 85.71% unit-length).
The UV duplication is per-mesh and binary: meshes whose material uses 2 UV channels
repeat UV0/UV1 into UV2/UV3 for 100% of their vertices, others for 0%.

## 5. Mesh / skeleton / skin binding

```
Mesh     { vertexCount, boundingRadius, lod, renderTiming, aabb{min,max},
           billboardType, blendShapeCount, skeletonIndex, sortOffset, boundingOffset,
           drawblocks[ { indexOffset, indexCount, materialIndex, iStreamIndex } ],
           instanceID, nodeIndex, vertexFormat, vStreamIndex }
Skeleton { bindPose[4x4 f32 per joint], nodes[u16 joint uids] }
```

* `SUM(vertexCount) == vertex buffer length` **exactly** (230981 / 132927 / 21950) — the
  meshes are consecutive slices of one vertex buffer, in mesh order.
* 5 LOD groups x 10 meshes for the mechs, 50 meshes / 50 skeletons.
* **Skin binding proof**: for 50/50 (f22a, type94) and 7/7 (bl74) meshes,
  `max(jointIndex within the mesh's vertex slice) == nJoints - 1` exactly.
* Weight signature: slot 0 always nonzero, slot 3 never used; when a slot's weight is 0
  its index byte is 0 in 100% of cases.
* `bindPose[j]` = the joint's world matrix, **transposed** (row-vector/Maya convention),
  uniformly scaled by 0.01 (cm -> m), translation in row 3.

## 6. Cross-check (the decisive test)

906 stored `bindPose` matrices (452 + 447 + 7) compared against the world matrices
reconstructed here from the tag stream:

```
f22a_00          max |stored - reconstructed| = 9.97e-07
type94_jp_00     max |stored - reconstructed| = 1.01e-06
bl74_00          max |stored - reconstructed| = 2.24e-10
```

Zero mismatches; the residual is exactly f32 storage precision on ~1000-unit
coordinates.  The hierarchy, the per-node transforms, the joint order and the bind
pose are therefore all confirmed by the file's own data.

## 7. Known incompletely decoded items

* 132 of 324927 indices in `f22a_00` (318 of 324243 in `type94_jp_00`) fall in ranges no
  `drawblocks` entry lists.  Total index count, ranges and the trailing end all agree,
  so this is one un-listed sub-range, not a parse failure.  `bl74_00` has no gap.
* The `f8` type (opcode `0x0f`) is only ever observed as `0x00` and `0x38`; `0x38` must be
  1.0 (`nodeAlpha`, `localScaling`), which E4M3 satisfies exactly, but no second value
  exists in these files to distinguish E4M3 from another 1-byte float with `0x38` = 1.0.
* The bytes after the class marker (`ad 96/97/98` for `Node`, `d1/d3/df/c0` elsewhere) are
  treated as opaque; their meaning is not established.
* `Material` / `Shader` / `Collision` sections in the file tail are located but not decoded.
