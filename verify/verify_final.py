#!/usr/bin/env python3
"""Final verification of a rigged, textured FBX."""
import os
import sys

import bpy
import numpy as np

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
FBX = ARGS[0]
PNG = ARGS[1] if len(ARGS) > 1 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=FBX)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
print(f"FILE        {os.path.basename(FBX)}  ({os.path.getsize(FBX)/1e6:.1f} MB)")
print(f"meshes      {len(meshes)}")
print(f"armatures   {len(arms)}")
if arms:
    b = arms[0].data.bones
    dep = {}
    for x in b:
        d, p = 0, x.parent
        while p:
            d += 1
            p = p.parent
        dep[d] = dep.get(d, 0) + 1
    print(f"bones       {len(b)}   depth histogram {dict(sorted(dep.items()))}")

tv = sum(len(o.data.vertices) for o in meshes)
tf = sum(len(o.data.polygons) for o in meshes)
print(f"vertices    {tv:,}")
print(f"faces       {tf:,}")

uvl = set()
for o in meshes:
    for l in o.data.uv_layers:
        uvl.add(l.name)
print(f"uv layers   {sorted(uvl)}")

mats = {}
for o in meshes:
    for m in o.data.materials:
        if m:
            mats[m.name] = m
print(f"materials   {len(mats)}")
for nm, m in list(mats.items())[:3]:
    imgs = [n.image.name for n in m.node_tree.nodes
            if n.type == "TEX_IMAGE" and n.image]
    print(f"   {nm:26s} images={imgs}")

allimg = sorted({n.image.name for m in mats.values() for n in m.node_tree.nodes
                 if n.type == "TEX_IMAGE" and n.image})
print(f"images      {len(allimg)}")
print(f"   {allimg[:12]}")

# weight integrity
bad = 0
nog = 0
for o in meshes:
    if not o.vertex_groups:
        nog += 1
        continue
    sums = np.zeros(len(o.data.vertices))
    for v in o.data.vertices:
        sums[v.index] = sum(g.weight for g in v.groups)
    bad += int((np.abs(sums - 1.0) > 1e-3).sum())
print(f"weight sum!=1 verts: {bad}   meshes without groups: {nog}")
