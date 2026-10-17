#!/usr/bin/env python3
"""
Deformation audit: rotate ONE bone at a time and measure which vertices move.

This is the decisive test for "does the mesh follow the bones".
"""
import os
import sys

import bpy
import numpy as np

FBX = sys.argv[sys.argv.index("--") + 1]


def eval_positions(objs, depsgraph):
    out = {}
    for o in objs:
        ev = o.evaluated_get(depsgraph)
        me = ev.to_mesh()
        n = len(me.vertices)
        arr = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", arr)
        out[o.name] = arr.reshape(n, 3)
        ev.to_mesh_clear()
    return out


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=FBX)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    arm = arms[0]
    print(f"FBX {os.path.basename(FBX)}")
    print(f"  meshes {len(meshes)}  bones {len(arm.data.bones)}")

    # baseline
    dg = bpy.context.evaluated_depsgraph_get()
    base = eval_positions(meshes, dg)

    tests = ["Head_M", "Knee_L", "Elbow_L", "Chest_M", "Shoulder_R", "Ankle_L"]
    for bone in tests:
        pb = arm.pose.bones.get(bone)
        if pb is None:
            print(f"  {bone:12s} MISSING")
            continue
        pb.rotation_mode = "XYZ"
        old = tuple(pb.rotation_euler)
        pb.rotation_euler = (0.5, 0.0, 0.0)
        dg = bpy.context.evaluated_depsgraph_get()
        now = eval_positions(meshes, dg)
        total = 0
        per = []
        for o in meshes:
            d = np.linalg.norm(now[o.name] - base[o.name], axis=1)
            moved = int((d > 1e-4).sum())
            if moved:
                per.append(f"{o.name.split('_')[-1]}:{moved}")
            total += moved
        print(f"  rotate {bone:12s} -> {total:6d} verts moved   "
              f"[{', '.join(per[:6])}]")
        pb.rotation_euler = old
        bpy.context.view_layer.update()

    # how many vertices follow ANY bone (sanity: all should)
    print("\n  bone influence map (mesh -> dominant bone):")
    for o in meshes[:10]:
        vg = {g.index: g.name for g in o.vertex_groups}
        cnt = {}
        for v in o.data.vertices:
            if not v.groups:
                continue
            g = max(v.groups, key=lambda x: x.weight)
            cnt[vg.get(g.group, "?")] = cnt.get(vg.get(g.group, "?"), 0) + 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:3]
        print(f"    {o.name:26s} {len(o.data.vertices):6d} v  " +
              ", ".join(f"{k}={n}" for k, n in top))


if __name__ == "__main__":
    main()
